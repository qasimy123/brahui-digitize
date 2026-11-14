#!/usr/bin/env bash
# Sequence-level distillation of the brh→en teacher (trimmed r5) into a shallow student, on one pod.
#   1. trim the teacher vocab if runs/nllb_r5_trim is not on the volume yet
#   2. teacher-translate the whole Brahui pool (resumable; data/distill/teacher_brh2en.jsonl)
#   3. prune the teacher to ENC/DEC layers as the student init
#   4. train the student on the distilled pairs (brh2en only), dev curve per epoch, dev eval at the end
# Results stay on the volume; finish.sh then deletes the pod. Re-running resumes at whichever step is unfinished.
# Usage (in tmux on the pod): bash scripts/gpu/distill_mt.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
STUDENT=${STUDENT:-student_a}; ENC=${ENC:-6}; DEC=${DEC:-3}
GEN_BS=${GEN_BS:-64}; EPOCHS=${EPOCHS:-3}; LR=${LR:-2e-4}; BS=${BS:-32}; ACC=${ACC:-1}
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
mkdir -p logs data/distill
export PULL_GRACE_MIN=${PULL_GRACE_MIN:-20}  # time to read logs / pull results before the pod deletes itself
trap 'scripts/gpu/finish.sh $? "runs/$STUDENT"' EXIT
uv run python -c "import torch,sys; ok=torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" \
  || { log "CUDA preflight failed; see docs/GPU_TRAINING.md"; exit 2; }
log "volume usage: $(du -sh /workspace 2>/dev/null | cut -f1)"; df -h /workspace | tail -1

if [ ! -f runs/nllb_r5_trim/final/model.safetensors ]; then
  log "=== trim teacher vocab"
  uv run python scripts/trim_nllb_vocab.py --src runs/nllb_r5_retrans/final --dst runs/nllb_r5_trim/final 2>&1 | tee -a logs/distill_trim.log
fi

log "=== teacher pass (bs=$GEN_BS)"
uv run python scripts/distill_generate.py --teacher runs/nllb_r5_trim/final --pool data/distill/brh_pool.jsonl \
  --out data/distill/teacher_brh2en.jsonl --bs "$GEN_BS" --beams 4 2>&1 | tee -a logs/distill_generate.log
log "teacher pass rows: $(wc -l < data/distill/teacher_brh2en.jsonl)"

if [ ! -f "runs/${STUDENT}_init/model.safetensors" ]; then
  log "=== prune teacher -> runs/${STUDENT}_init (enc $ENC / dec $DEC)"
  uv run python scripts/prune_nllb_layers.py --src runs/nllb_r5_trim/final --dst "runs/${STUDENT}_init" --enc "$ENC" --dec "$DEC" 2>&1 | tee -a "logs/${STUDENT}.log"
fi

if [ ! -f "runs/$STUDENT/eval_after.json" ]; then
  log "=== train $STUDENT"
  uv run python scripts/train_nllb.py --out "runs/$STUDENT" --base runs/nllb_r5_trim/final --init "runs/${STUDENT}_init" \
    --pairs data/distill/teacher_brh2en.jsonl --dirs brh2en --epochs "$EPOCHS" --lr "$LR" --bs "$BS" --grad-accum "$ACC" \
    --warmup 500 --max-len 200 --skip-before --skip-after --seed 0 2>&1 | tee -a "logs/${STUDENT}.log" | grep --line-buffered -v "it/s\]"
  rm -rf "runs/$STUDENT"/checkpoint-*
fi
log "done; $(cat runs/$STUDENT/eval_after.json 2>/dev/null | tr -d '\n' | cut -c1-300)"
