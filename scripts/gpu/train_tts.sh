#!/usr/bin/env bash
# Fine-tune MMS-TTS (VITS, Urdu voice) into a Brahui voice with Meta's finetune-hf-vits recipe. Needs no volume:
# data = HF dataset qkhawaja/brahui-tts-cv (scripts/build_tts_set.py), init = qkhawaja/brahui-tts-vits-init
# (generator + discriminator, vocab + ء ڷ), result pushed to $HUB_MODEL. HF_TOKEN (write) must be in the env.
#   HF_TOKEN=... EPOCHS=80 scripts/gpu/train_tts.sh        (≈1 GPU-hour on an A5000/4090; L40S faster)
# finish.sh deletes the pod at the end, success or failure (KEEP_POD=1 to keep it).
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=${RUN:-runs/tts_vits_cv}; HUB_MODEL=${HUB_MODEL:-qkhawaja/brahui-tts-vits}
export HF_UPLOAD_REPO=${HF_UPLOAD_REPO:-qkhawaja/brahui-tts-runs}   # finish.sh uploads logs/ + $RUN here, so a failure keeps its log
DATASET=${DATASET:-qkhawaja/brahui-tts-cv}; INIT=${INIT:-qkhawaja/brahui-tts-vits-init}
EPOCHS=${EPOCHS:-80}; LR=${LR:-2e-5}; BS=${BS:-16}; FP16=${FP16:-true}
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
mkdir -p logs "$RUN"
# On failure, push the newest checkpoint + logs to the Hub with the pod's own venv BEFORE finish.sh deletes the pod.
salvage() { st=$?; if [ $st -ne 0 ] && [ -x /root/vitsenv/bin/python ]; then /root/vitsenv/bin/python scripts/gpu/salvage_upload.py "$HUB_MODEL-salvage" "$RUN" "$PWD/logs" || true; fi; scripts/gpu/finish.sh $st "$RUN"; }
trap salvage EXIT
# Over SSH the pod's env (HF_TOKEN set at create-pod) is not inherited; read it from the container's init process.
if [ -z "${HF_TOKEN:-}" ] && [ -r /proc/1/environ ]; then export HF_TOKEN=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^HF_TOKEN=//p'); fi
[ -n "${HF_TOKEN:-}" ] || { log "HF_TOKEN missing"; exit 2; }

# The recipe's own utils target transformers 4.x -> separate venv, not the project one.
VENV=/root/vitsenv
if [ ! -x $VENV/bin/python ]; then
  log "building venv"
  uv venv -q --python 3.11 $VENV
  [ -d /root/finetune-hf-vits ] || git clone -q https://github.com/ylacombe/finetune-hf-vits.git /root/finetune-hf-vits
  VIRTUAL_ENV=$VENV uv pip install -q "torch==2.4.1" "transformers==4.46.3" "datasets[audio]==3.1.0" "accelerate==1.1.1" \
    "matplotlib<3.10" tensorboard Cython "huggingface_hub<1" "numpy>=2,<2.3" soundfile librosa
  # single-speaker collator bug: speaker_id=None is dropped by accelerate's send_to_device -> KeyError at step 0
  sed -i 's|torch.tensor(\[feature\["speaker_id"\] for feature in features\]) if "speaker_id" in features\[0\] else None|torch.tensor([feature.get("speaker_id", 0) for feature in features])|' /root/finetune-hf-vits/run_vits_finetuning.py
  grep -q 'feature.get("speaker_id", 0)' /root/finetune-hf-vits/run_vits_finetuning.py || { log "collator patch failed"; exit 2; }
  (cd /root/finetune-hf-vits/monotonic_align && mkdir -p monotonic_align && $VENV/bin/python setup.py build_ext --inplace >/dev/null)
fi
$VENV/bin/python -c "import torch,sys; ok=torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" \
  || { log "CUDA preflight failed"; exit 2; }

# The dataset may still be uploading from the Mac; wait for it (up to 40 min).
for i in $(seq 1 80); do
  $VENV/bin/python -c "import sys; from huggingface_hub import HfApi; sys.exit(0 if any(s.rfilename.startswith('data/dev') for s in HfApi().dataset_info('$DATASET').siblings) else 1)" 2>/dev/null && break
  [ $i -eq 1 ] && log "waiting for dataset $DATASET"; sleep 30
done
cat > "$RUN/config.json" <<JSON
{
  "project_name": "brahui_tts_vits", "push_to_hub": true, "hub_model_id": "$HUB_MODEL", "hub_private_repo": true,
  "overwrite_output_dir": true, "output_dir": "$PWD/$RUN/out", "report_to": "tensorboard",
  "dataset_name": "$DATASET", "audio_column_name": "audio", "text_column_name": "text",
  "train_split_name": "train", "eval_split_name": "dev",
  "max_duration_in_seconds": 12.5, "min_duration_in_seconds": 1.0, "max_tokens_length": 500,
  "model_name_or_path": "$INIT", "preprocessing_num_workers": 4,
  "do_train": true, "num_train_epochs": $EPOCHS, "gradient_accumulation_steps": 1, "gradient_checkpointing": false,
  "per_device_train_batch_size": $BS, "learning_rate": $LR, "adam_beta1": 0.8, "adam_beta2": 0.99, "warmup_ratio": 0.01,
  "group_by_length": false, "do_eval": true, "eval_steps": 500, "per_device_eval_batch_size": 16, "max_eval_samples": 25,
  "do_step_schedule_per_epoch": true, "save_steps": 1000, "save_total_limit": 2,
  "weight_disc": 3, "weight_fmaps": 1, "weight_gen": 1, "weight_kl": 1.5, "weight_duration": 1, "weight_mel": 35,
  "fp16": $FP16, "seed": 456
}
JSON
log "VITS fine-tune: $INIT on $DATASET, $EPOCHS epochs, lr $LR, bs $BS, fp16=$FP16 -> $HUB_MODEL"
export HF_HUB_ENABLE_HF_TRANSFER=0
(cd /root/finetune-hf-vits && $VENV/bin/accelerate launch run_vits_finetuning.py "$OLDPWD/$RUN/config.json") 2>&1 | tee -a logs/tts_vits.log | grep --line-buffered -v "it/s\]"
log "TTS done; model pushed to $HUB_MODEL"
