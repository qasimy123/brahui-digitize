#!/usr/bin/env bash
# Stage 3: adapt the stage-2 model on all 80 h with BrahuiSpeech labels normalized to CV/Omnilingual spelling
# (data/manifests/brahuispeech_train_norm.jsonl from scripts/build_bs_normalization.py). Tests whether the
# convention consistency of stage 2 survives putting the 70 h of media audio back in.
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE=${BASE:-runs/ctc1b_brh_s2/final}
RUN=${RUN:-runs/ctc1b_brh_s3}
STEPS=${STEPS:-1000}
LR=${LR:-5e-6}
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
mkdir -p logs
trap 'scripts/gpu/finish.sh $? "$RUN"' EXIT
# Preflight: fail in seconds, not after 30 min on CPU.
uv run python -c "import torch,sys; ok=torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" \
  || { log "CUDA preflight failed (torch build vs driver mismatch?); see docs/GPU_TRAINING.md"; exit 2; }
log "stage 3 from $BASE: $STEPS steps, lr $LR, CV + Omnilingual + normalized BrahuiSpeech"
for c in "$RUN"/checkpoint-*; do [ -d "$c" ] && [ ! -f "$c/trainer_state.json" ] && { log "removing incomplete $c"; rm -rf "$c"; }; done
RESUME=""; ls -d "$RUN"/checkpoint-* >/dev/null 2>&1 && RESUME="--resume"
uv run python scripts/train_asr.py --base "$BASE" --out "$RUN" --max-steps "$STEPS" --lr "$LR" --warmup 100 --eval-every 250 \
  --bs 8 --grad-accum 4 --brahuispeech-manifest brahuispeech_train_norm.jsonl $RESUME 2>&1 | tee -a logs/stage3.log | grep --line-buffered -v "it/s\]"
for s in cv_test omni_test; do
  uv run python scripts/asr_eval.py --model "ctc:$RUN/final" --set $s --device cuda --tag "ctc_s3__$s" 2>&1 | tail -3
done
uv run python scripts/summarize_results.py > evals/RESULTS_TABLES.md
log "stage 3 done; finish.sh now uploads $RUN/final, evals/asr/runs, logs and deletes the pod"
