#!/usr/bin/env bash
# Follow-up chain after distill_mt.sh (student_a: 6/3 layers, 3 epochs, dev chrF++ 41.7 vs teacher 51.2, still rising):
#   1. student_a_cont : continue student_a for 6 more epochs at lr 1e-4   (is it under-trained?)
#   2. student_b      : 12 enc / 6 dec layers (~434M), 4 epochs            (is it capacity-limited?)
#   3. student_asr.sh : the 300M CTC student (its own finish.sh deletes the pod at the end)
# Holds the pod (/workspace/.hold) while running. Each step is skipped if its eval_after.json already exists.
set -euo pipefail
cd "$(dirname "$0")/../.."
PAIRS=data/distill/teacher_brh2en.jsonl; BASE=runs/nllb_r5_trim/final
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
mkdir -p logs; touch /workspace/.hold.distill_chain
export PULL_GRACE_MIN=${PULL_GRACE_MIN:-20}
trap 'rm -f /workspace/.hold.distill_chain; scripts/gpu/finish.sh $? runs/student_b' EXIT
uv run python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { log "CUDA preflight failed"; exit 2; }

train() {  # train <run> <init> <epochs> <lr>
  local run=$1 init=$2 ep=$3 lr=$4
  [ -f "runs/$run/eval_after.json" ] && { log "skip $run (done)"; return; }
  log "=== train $run from $init ($ep epochs, lr $lr)"
  uv run python scripts/train_nllb.py --out "runs/$run" --base "$BASE" --init "$init" --pairs "$PAIRS" --dirs brh2en \
    --epochs "$ep" --lr "$lr" --bs 32 --grad-accum 1 --warmup 300 --max-len 200 --skip-before --skip-after --seed 0 \
    2>&1 | tee -a "logs/$run.log" | grep --line-buffered -v "it/s\]"
  rm -rf "runs/$run"/checkpoint-*
  log "=== $run dev: $(grep -o '"chrfpp": [0-9.]*' runs/$run/dev_curve.jsonl | tr '\n' ' ')"
}

train student_a_cont runs/student_a/final 6 1e-4
if [ ! -f runs/student_b_init/model.safetensors ]; then
  uv run python scripts/prune_nllb_layers.py --src "$BASE" --dst runs/student_b_init --enc 12 --dec 6 2>&1 | tee -a logs/student_b.log
fi
train student_b runs/student_b_init 4 2e-4
log "=== MT students done; volume: $(du -sh /workspace | cut -f1)"
bash scripts/gpu/student_asr.sh
