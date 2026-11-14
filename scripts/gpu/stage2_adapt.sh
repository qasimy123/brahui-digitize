#!/usr/bin/env bash
# Stage 2: short domain/orthography adaptation of the stage-1 model on Common Voice + Omnilingual only
# (no BrahuiSpeech), then eval on both sealed sets so stage 1 vs stage 2 can be compared with CIs.
set -euo pipefail
cd "$(dirname "$0")/../.."
S1=${S1:-runs/ctc1b_brh/final}
RUN=${RUN:-runs/ctc1b_brh_s2}
STEPS=${STEPS:-800}
LR=${LR:-5e-6}
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
mkdir -p logs
# On exit (success or failure) push results to HF_UPLOAD_REPO and delete this pod. KEEP_POD=1 to keep it.
trap 'scripts/gpu/finish.sh $? "$RUN"' EXIT
# Preflight: fail in seconds, not after 30 min on CPU.
uv run python -c "import torch,sys; ok=torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" \
  || { log "CUDA preflight failed (torch build vs driver mismatch?); see docs/GPU_TRAINING.md"; exit 2; }
log "stage 2 from $S1: $STEPS steps, lr $LR, CV + Omnilingual only"
# Resume if a complete checkpoint exists (a checkpoint without trainer_state.json was interrupted mid-save: drop it).
for c in "$RUN"/checkpoint-*; do [ -d "$c" ] && [ ! -f "$c/trainer_state.json" ] && { log "removing incomplete $c"; rm -rf "$c"; }; done
RESUME=""; ls -d "$RUN"/checkpoint-* >/dev/null 2>&1 && RESUME="--resume"
uv run python scripts/train_asr.py --base "$S1" --out "$RUN" --max-steps "$STEPS" --lr "$LR" --warmup 100 --eval-every 200 \
  --bs 8 --grad-accum 2 --no-brahuispeech $RESUME 2>&1 | tee -a logs/stage2.log | grep --line-buffered -v "it/s\]"
for s in cv_test omni_test; do
  uv run python scripts/asr_eval.py --model "ctc:$RUN/final" --set $s --device cuda --tag "ctc_s2__$s" 2>&1 | tail -3
done
uv run python scripts/summarize_results.py > evals/RESULTS_TABLES.md
log "stage 2 done; finish.sh now uploads $RUN/final, evals/asr/runs, logs and deletes the pod"
