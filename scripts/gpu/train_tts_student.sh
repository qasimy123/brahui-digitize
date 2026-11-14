#!/usr/bin/env bash
# Train the small browser voice (MMS VITS, 36M parameters, 16 kHz) on the teacher's clips: data/tts/student (scripts/build_tts_student_set.py),
# start = MMS Urdu generator + discriminator with a vowelled-Latin vocabulary (tools/vits/latin_init.py).
# Training runs in segments of $SEG_EPOCHS epochs, each continuing from the last; after every segment the voice is scored
# (scripts/tts_student_eval.py: speak 40 dev sentences, read back with the s3 ASR) and the best segment is kept in $RUN/best.
# Known costs (2026-11-14): every segment re-prepares the data set (~3 min of idle GPU and a new 20 GB cache under ~/.cache/huggingface/datasets:
# delete the old one between segments or the disk fills by segment 3); 8 preparation workers crashed the map, 2 work.
# Stops early when $PATIENCE segments in a row bring no improvement. Markers in logs/tts_student.log: "SEG k CER x", "STUDENT DONE".
#   tmux:  SEGS=12 SEG_EPOCHS=15 bash scripts/gpu/train_tts_student.sh        local smoke test: LOCAL=1 SEGS=1 SEG_EPOCHS=1 BS=4 MAX_TRAIN=16 ...
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=${RUN:-runs/tts_student}; DATA=${DATA:-data/tts/student}; SEGS=${SEGS:-12}; SEG_EPOCHS=${SEG_EPOCHS:-15}; LR=${LR:-1e-4}; BS=${BS:-32}; FP16=${FP16:-false}; PATIENCE=${PATIENCE:-3}
VENV=${VENV:-/root/vitsenv}; RECIPE=${RECIPE:-/root/finetune-hf-vits}; case $VENV in /*) ;; *) VENV=$PWD/$VENV;; esac; case $RECIPE in /*) ;; *) RECIPE=$PWD/$RECIPE;; esac; ROOT=$PWD; MT=""; [ -n "${MAX_TRAIN:-}" ] && MT="\"max_train_samples\": $MAX_TRAIN,"; LOG=logs/tts_student.log; mkdir -p logs "$RUN" evals/tts/student
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
if [ -z "${LOCAL:-}" ]; then
  export UV_PROJECT_ENVIRONMENT=/root/venv UV_PYTHON=3.12 PULL_GRACE_MIN=${PULL_GRACE_MIN:-120}; touch /workspace/.hold.student
  trap 'st=$?; rm -f /workspace/.hold.student; bash scripts/gpu/finish.sh $st "$RUN"' EXIT
fi
if [ ! -x $VENV/bin/python ]; then
  log "building venv"; command -v uv >/dev/null || pip install -q --break-system-packages uv
  uv venv -q --python 3.11 $VENV
  uv pip install -q --python $VENV/bin/python "torch==2.4.1" "transformers==4.46.3" "datasets[audio]==3.1.0" "accelerate==1.1.1" "matplotlib<3.10" tensorboard Cython "huggingface_hub<1" "numpy>=2,<2.3" soundfile librosa setuptools
fi
[ -d $RECIPE ] || git clone -q https://github.com/ylacombe/finetune-hf-vits.git $RECIPE
git -C $RECIPE checkout -q 6f3f51f
# single-speaker collator bug: speaker_id=None is dropped by accelerate's send_to_device -> KeyError at step 0
$VENV/bin/python - "$RECIPE/run_vits_finetuning.py" <<'PY' || { log "collator patch failed"; exit 2; }
import sys; p = sys.argv[1]; s = open(p).read(); a = 'torch.tensor([feature["speaker_id"] for feature in features]) if "speaker_id" in features[0] else None'; b = 'torch.tensor([feature.get("speaker_id", 0) for feature in features])'
if b not in s: assert a in s; open(p, "w").write(s.replace(a, b))
PY
ls $RECIPE/monotonic_align/monotonic_align/core.*.so >/dev/null 2>&1 || (cd $RECIPE/monotonic_align && mkdir -p monotonic_align && $VENV/bin/python setup.py build_ext --inplace >/dev/null 2>&1)
ls $RECIPE/monotonic_align/monotonic_align/core.*.so >/dev/null || { log "monotonic_align build failed"; exit 2; }
if [ ! -f "$RUN/init/model.safetensors" ]; then
  [ -f tools/vits/init_urd/model.safetensors ] || (cd $RECIPE && $VENV/bin/python convert_original_discriminator_checkpoint.py --language_code urd-script_arabic --pytorch_dump_folder_path $ROOT/tools/vits/init_urd 2>&1 | tail -1)
  $VENV/bin/python tools/vits/latin_init.py tools/vits/init_urd "$RUN/init" | tee -a "$LOG" || exit 2
fi
[ -f $DATA/audio/train/metadata.jsonl ] && [ -f $DATA/dev.jsonl ] || { log "missing $DATA"; exit 2; }
score() { uv run python scripts/tts_student_eval.py --model "$1" --dev $DATA/dev.jsonl --n ${EVAL_N:-40} --out evals/tts/student/$2 2>>"$LOG" | tail -1; }
if [ -z "${LOCAL:-}" ]; then $VENV/bin/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { log "CUDA preflight failed"; exit 2; }; fi
B0=$(score "$RUN/init" init); log "SEG 0 (start, Urdu sounds on Latin letters): $B0"; echo "$B0" | grep -q cer_mean || { log "scoring preflight failed"; exit 2; }
prev="$RUN/init"; best=9; bad=0; k0=1
for k in $(seq 1 $SEGS); do [ -f "$RUN/seg$k/model.safetensors" ] && { prev="$RUN/seg$k"; k0=$((k+1)); }; done    # resume after a restart
for k in $(seq $k0 $SEGS); do
  out="$RUN/seg$k"; rm -rf "$out"; mkdir -p "$out"
  cat > "$RUN/config_seg.json" <<JSON
{ "project_name": "brahui_tts_student", "push_to_hub": false, "overwrite_output_dir": true, "output_dir": "$PWD/$out", "report_to": "tensorboard",
  "dataset_name": "$PWD/$DATA/audio", "audio_column_name": "audio", "text_column_name": "text", "train_split_name": "train", "eval_split_name": "validation",
  "max_duration_in_seconds": 14.5, "min_duration_in_seconds": 0.8, "max_tokens_length": 600, $MT
  "model_name_or_path": "$PWD/$prev", "preprocessing_num_workers": ${PREP_WORKERS:-2}, "dataloader_num_workers": ${DL_WORKERS:-6},
  "do_train": true, "num_train_epochs": $SEG_EPOCHS, "gradient_accumulation_steps": 1, "gradient_checkpointing": false,
  "per_device_train_batch_size": $BS, "learning_rate": $LR, "adam_beta1": 0.8, "adam_beta2": 0.99, "warmup_ratio": 0.01,
  "group_by_length": false, "do_eval": true, "eval_steps": 100000, "per_device_eval_batch_size": 8, "max_eval_samples": ${EVAL_SAMPLES:-8},
  "do_step_schedule_per_epoch": true, "save_steps": 1000000, "save_total_limit": 1,
  "weight_disc": 3, "weight_fmaps": 1, "weight_gen": 1, "weight_kl": 1.5, "weight_duration": 1, "weight_mel": 35, "fp16": $FP16, "seed": $((456 + k)) }
JSON
  log "segment $k/$SEGS: $SEG_EPOCHS epochs from $prev, lr $LR, bs $BS"
  (cd $RECIPE && HF_HUB_ENABLE_HF_TRANSFER=0 $VENV/bin/accelerate launch run_vits_finetuning.py "$ROOT/$RUN/config_seg.json" 2>&1) | tr '\r' '\n' | awk 'NR % 200 == 0 || !/it\/s\]|s\/it\]/ { print; fflush() }' >> "$LOG.train"
  [ -f "$out/model.safetensors" ] || { log "segment $k produced no model (see $LOG.train)"; tail -5 "$LOG.train" | tee -a "$LOG"; exit 3; }
  rm -rf "$out"/checkpoint-* "$out"/runs
  R=$(score "$out" seg$k); cer=$(echo "$R" | python3 -c "import json,sys; print(json.load(sys.stdin)['cer_mean'])" 2>/dev/null || echo ""); [ -n "$cer" ] || { log "scoring failed after segment $k: $R"; exit 3; }
  if python3 -c "import sys; sys.exit(0 if $cer < $best else 1)"; then best=$cer; bad=0; rm -rf "$RUN/best"; cp -r "$out" "$RUN/best"; echo "$k" > "$RUN/best/segment.txt"; log "SEG $k CER $cer (best so far, kept) $R"
  else bad=$((bad+1)); log "SEG $k CER $cer (best $best, $bad without improvement) $R"; fi
  [ "$prev" != "$RUN/init" ] && [ "$prev" != "$out" ] && rm -rf "$prev"; prev="$out"
  [ $bad -ge $PATIENCE ] && { log "stopping early: $PATIENCE segments without improvement"; break; }
done
log "full dev: $(EVAL_N=100 score "$RUN/best" best_full)"
log "STUDENT DONE best CER $best (segment $(cat "$RUN/best/segment.txt"))"
