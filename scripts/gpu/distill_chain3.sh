#!/usr/bin/env bash
# Chain 3: deep-encoder / shallow-decoder student (24/6, 687M) for the <1 s browser bar with teacher-level quality.
# Browser latency is decoder-bound (teacher encoder 71 ms vs 12-layer 28 ms natively; the 24-layer decoder is the 2 s).
#   1. prune teacher to all 24 encoder layers + 6 decoder layers -> runs/student_c_init
#   2. train student_c on data/distill/distill_pairs_v2.jsonl (teacher pairs + 200k BT), EPOCHS epochs, lr 1e-4
# Self-deletes the pod via finish.sh (per-job hold flag).
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE=runs/nllb_r5_trim/final; RUN=${RUN:-student_c}; EPOCHS=${EPOCHS:-2}; LR=${LR:-1e-4}
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
mkdir -p logs; touch /workspace/.hold.distill_chain3
export PULL_GRACE_MIN=${PULL_GRACE_MIN:-30}
trap 'rm -f /workspace/.hold.distill_chain3; scripts/gpu/finish.sh $? "runs/$RUN"' EXIT
uv run python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { log "CUDA preflight failed"; exit 2; }
[ -f data/distill/distill_pairs_v2.jsonl ] || { log "missing distill_pairs_v2.jsonl"; exit 3; }

if [ ! -f runs/${RUN}_init/config.json ]; then
  log "=== prune teacher -> runs/${RUN}_init (enc 24 / dec 6)"
  uv run python scripts/prune_nllb_layers.py --src "$BASE" --dst "runs/${RUN}_init" --enc 24 --dec 6
fi
if [ ! -f "runs/$RUN/eval_after.json" ]; then
  log "=== train $RUN from runs/${RUN}_init ($EPOCHS epochs, lr $LR)"
  uv run python scripts/train_nllb.py --out "runs/$RUN" --base "$BASE" --init "runs/${RUN}_init" --pairs data/distill/distill_pairs_v2.jsonl --dirs brh2en \
    --epochs "$EPOCHS" --lr "$LR" --bs 32 --grad-accum 1 --warmup 300 --max-len 200 --skip-before --skip-after --seed 0 \
    2>&1 | tee -a "logs/$RUN.log" | grep --line-buffered -v "it/s\]"
  rm -rf "runs/$RUN"/checkpoint-*
fi
log "=== $RUN dev: $(grep -o '"chrfpp": [0-9.]*' runs/$RUN/dev_curve.jsonl | tr '\n' ' ')"
log "CHAIN3 DONE"
