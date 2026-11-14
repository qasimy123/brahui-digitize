#!/usr/bin/env bash
# ASR student: fine-tune omniASR-CTC-300M (3x smaller than the 1B teacher s3) on CV + Omnilingual + normalized
# BrahuiSpeech + the stage-3 media pseudo-labels (data/manifests/media_pseudo_s3.jsonl), then score cv_test/omni_test.
# Creates /workspace/.hold while running so another job's finish.sh on the same pod waits instead of deleting it.
# Usage (tmux on the pod): bash scripts/gpu/student_asr.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE=${BASE:-data/raw/models/omniASR-CTC-300M-v2-hf}
RUN=${RUN:-runs/ctc300m_student}
STEPS=${STEPS:-4000}; LR=${LR:-5e-5}; BS=${BS:-8}; ACC=${ACC:-4}; TAG=${TAG:-ctc_300m_student}
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
mkdir -p logs; touch /workspace/.hold.student_asr
export PULL_GRACE_MIN=${PULL_GRACE_MIN:-20}
trap 'rm -f /workspace/.hold.student_asr; scripts/gpu/finish.sh $? "$RUN"' EXIT
uv run python -c "import torch,sys; ok=torch.cuda.is_available(); sys.exit(0 if ok else 1)" || { log "CUDA preflight failed"; exit 2; }
log "ASR student from $BASE: $STEPS steps, lr $LR, bs ${BS}x${ACC}, + media pseudo-labels"
for c in "$RUN"/checkpoint-*; do [ -d "$c" ] && [ ! -f "$c/trainer_state.json" ] && { log "removing incomplete $c"; rm -rf "$c"; }; done
RESUME=""; ls -d "$RUN"/checkpoint-* >/dev/null 2>&1 && RESUME="--resume"
if [ ! -f "$RUN/final/model.safetensors" ]; then
  uv run python scripts/train_asr.py --base "$BASE" --out "$RUN" --max-steps "$STEPS" --lr "$LR" --warmup 300 --eval-every 500 \
    --bs "$BS" --grad-accum "$ACC" --brahuispeech-manifest brahuispeech_train_norm.jsonl --extra-manifest media_pseudo_s3.jsonl \
    $RESUME 2>&1 | tee -a logs/student_asr.log | grep --line-buffered -v "it/s\]"
  rm -rf "$RUN"/checkpoint-*
fi
for s in cv_test omni_test; do
  uv run python scripts/asr_eval.py --model "ctc:$RUN/final" --set $s --device cuda --tag "${TAG}__$s" 2>&1 | tail -2
done
uv run python scripts/summarize_results.py > evals/RESULTS_TABLES.md || true
log "ASR student done; $(grep -h '"cer"' evals/asr/runs/${TAG}__*/summary.json 2>/dev/null | tr -d '\n' | cut -c1-200)"
