#!/usr/bin/env bash
# Straight-shot Brahui ASR training on one GPU pod (RunPod/Lambda). Idempotent: rerun to resume.
#
# Produces runs/ctc1b_brh/final (HF Wav2Vec2ForCTC) and evals/asr/runs/ctc_final__{cv_test,omni_test}.
# Steps: 0) deps  1) zero-shot baselines of the base model on both sealed sets  2) fine-tune  3) eval final  4) tables.
#
# Prereqs on the pod (see docs/GPU_TRAINING.md): repo cloned, data rsynced into data/.
# Set HF_TOKEN and HF_UPLOAD_REPO so results are pushed to a private HF repo before the pod deletes itself.
set -euo pipefail
cd "$(dirname "$0")/../.."

BASE=${BASE:-data/raw/models/omniASR-CTC-1B-v2-hf}
RUN=${RUN:-runs/ctc1b_brh}
STEPS=${STEPS:-5000}
BS=${BS:-8}            # per-device batch; 8 x 30 s fits an 80 GB A100 in bf16 with grad checkpointing. Use 4 on 24 GB.
ACC=${ACC:-4}          # effective batch = BS*ACC clips (~32 clips, ~5 min of audio per step)
LR=${LR:-1e-5}
mkdir -p logs

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
# On exit (success or failure) push results to HF_UPLOAD_REPO and delete this pod. KEEP_POD=1 to keep it.
trap 'scripts/gpu/finish.sh $? "$RUN"' EXIT
# Preflight: fail in seconds, not after 30 min on CPU.
uv run python -c "import torch,sys; ok=torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" \
  || { log "CUDA preflight failed (torch build vs driver mismatch?); see docs/GPU_TRAINING.md"; exit 2; }

log "0/4 deps"
command -v uv >/dev/null || pip install -q uv
uv sync -q
[ -d "$BASE" ] || uv run python -c "from huggingface_hub import snapshot_download as s; s('aadel4/omniASR-CTC-1B-v2', local_dir='$BASE')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

log "1/4 zero-shot baselines (full sealed sets)"
for s in cv_test omni_test; do
  uv run python scripts/asr_eval.py --model "ctc:$BASE" --set $s --device cuda --tag "ctc_$(basename $BASE)__$s" 2>&1 | tail -3
done

log "2/4 chunk Omnilingual segments (if not already done on the Mac)"
[ -f data/manifests/omni_train_chunks.jsonl ] || uv run python scripts/segment_long.py --model "$BASE" --device cuda

log "3/4 fine-tune ($STEPS steps, bs $BS x $ACC, lr $LR)"
RESUME=""; ls -d "$RUN"/checkpoint-* >/dev/null 2>&1 && RESUME="--resume"
uv run python scripts/train_asr.py --base "$BASE" --out "$RUN" --max-steps "$STEPS" --bs "$BS" --grad-accum "$ACC" --lr "$LR" $RESUME 2>&1 | tee -a logs/train_asr.log | grep -v "it/s\]"

log "4/4 eval final model on both sealed sets"
for s in cv_test omni_test; do
  uv run python scripts/asr_eval.py --model "ctc:$RUN/final" --set $s --device cuda --tag "ctc_final__$s" 2>&1 | tail -3
done
uv run python scripts/summarize_results.py > evals/RESULTS_TABLES.md
log "done; finish.sh now uploads $RUN/final, evals/asr/runs, logs and deletes the pod"
