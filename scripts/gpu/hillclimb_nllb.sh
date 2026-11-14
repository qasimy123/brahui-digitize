#!/usr/bin/env bash
# Run a sequence of NLLB hillclimb configs on one pod, then finish.sh (results stay on the volume; pod self-deletes).
# Each config is one line of $PLAN: "<run name> <train_nllb.py args...>", or "! <shell command>" for a step between runs. Runs already holding final/ are skipped,
# so a restarted pod resumes the plan. Usage: PLAN=plans/r0.txt bash scripts/gpu/hillclimb_nllb.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PLAN=${PLAN:?path to plan file}
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
mkdir -p logs
trap 'scripts/gpu/finish.sh $? "runs/hillclimb"' EXIT
uv run python -c "import torch,sys; ok=torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" \
  || { log "CUDA preflight failed; see docs/GPU_TRAINING.md"; exit 2; }
while read -r name args; do
  [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
  if [ "$name" = "!" ]; then log "=== run: $args"; bash -c "$args" 2>&1 | { grep --line-buffered -v "it/s\]" || true; }; continue; fi
  if [ -f "runs/$name/eval_after.json" ]; then log "skip $name (done)"; continue; fi
  log "=== $name: $args"
  # shellcheck disable=SC2086
  uv run python scripts/train_nllb.py --out "runs/$name" $args 2>&1 | tee -a "logs/$name.log" | grep --line-buffered -v "it/s\]"
  log "=== $name done"
done < "$PLAN"
log "plan done; finish.sh"
