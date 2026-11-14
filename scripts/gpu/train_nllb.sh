#!/usr/bin/env bash
# NLLB-200 Brahui<->English fine-tune on gold + synthetic (back-translation) pairs. See scripts/train_nllb.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE=${BASE:-facebook/nllb-200-distilled-600M}
RUN=${RUN:-runs/nllb_brh}
EPOCHS=${EPOCHS:-3}; LR=${LR:-1e-4}; BS=${BS:-16}; ACC=${ACC:-2}
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
mkdir -p logs
trap 'scripts/gpu/finish.sh $? "$RUN"' EXIT
uv run python -c "import torch,sys; ok=torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" \
  || { log "CUDA preflight failed; see docs/GPU_TRAINING.md"; exit 2; }
log "NLLB fine-tune from $BASE: $EPOCHS epochs, lr $LR, bs $BS x $ACC"
uv run python scripts/train_nllb.py --base "$BASE" --out "$RUN" --epochs "$EPOCHS" --lr "$LR" --bs "$BS" --grad-accum "$ACC" 2>&1 | tee -a logs/nllb.log | grep --line-buffered -v "it/s\]"
log "NLLB done; finish.sh saves $RUN/final and deletes the pod"
