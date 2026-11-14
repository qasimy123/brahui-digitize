#!/usr/bin/env bash
# Chain 2: more distilled data for the brh→en student via back-translation, then continue student_b on the union.
#   1. teacher en→brh on data/distill/en_bt_pool.jsonl  -> data/distill/bt_en2brh.jsonl  (synthetic brh, real en)
#   2. merge with the teacher brh→en pairs -> data/distill/distill_pairs_v2.jsonl (dirs=["brh2en"])
#   3. student_b_v2: continue from runs/student_b/final, 3 epochs, lr 1e-4
# Holds the pod; the last job's finish.sh deletes it. Safe to run concurrently with student_asr.sh (GPU is time-shared).
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE=runs/nllb_r5_trim/final; INIT=${INIT:-runs/student_b/final}; RUN=${RUN:-student_b_v2}
EPOCHS=${EPOCHS:-3}; LR=${LR:-1e-4}; GEN_BS=${GEN_BS:-64}
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
mkdir -p logs; touch /workspace/.hold.distill_chain2
export PULL_GRACE_MIN=${PULL_GRACE_MIN:-20}
trap 'rm -f /workspace/.hold.distill_chain2; scripts/gpu/finish.sh $? "runs/$RUN"' EXIT
uv run python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { log "CUDA preflight failed"; exit 2; }

log "=== back-translation pass (en→brh, bs=$GEN_BS)"
uv run python scripts/distill_generate.py --teacher "$BASE" --dir en2brh --pool data/distill/en_bt_pool.jsonl \
  --out data/distill/bt_en2brh.jsonl --bs "$GEN_BS" --beams 4 --max-new 160 2>&1 | tee -a logs/bt_generate.log | grep --line-buffered -v "it/s\]"

log "=== merge pairs"
uv run python - <<'PY'
import json
out = open("data/distill/distill_pairs_v2.jsonl", "w", encoding="utf-8"); n = 0
for f, dirs in (("data/distill/teacher_brh2en.jsonl", ["brh2en"]), ("data/distill/bt_en2brh.jsonl", ["brh2en"])):
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        if r.get("brh") and r.get("en"):
            out.write(json.dumps({**r, "dirs": dirs}, ensure_ascii=False) + "\n"); n += 1
print("pairs", n)
PY

if [ ! -f "runs/$RUN/eval_after.json" ]; then
  log "=== train $RUN from $INIT ($EPOCHS epochs, lr $LR)"
  uv run python scripts/train_nllb.py --out "runs/$RUN" --base "$BASE" --init "$INIT" --pairs data/distill/distill_pairs_v2.jsonl --dirs brh2en \
    --epochs "$EPOCHS" --lr "$LR" --bs 32 --grad-accum 1 --warmup 300 --max-len 200 --skip-before --skip-after --seed 0 \
    2>&1 | tee -a "logs/$RUN.log" | grep --line-buffered -v "it/s\]"
  rm -rf "runs/$RUN"/checkpoint-*
fi
log "=== $RUN dev: $(grep -o '"chrfpp": [0-9.]*' runs/$RUN/dev_curve.jsonl | tr '\n' ' ')"
