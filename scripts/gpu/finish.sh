#!/usr/bin/env bash
# Called by the job scripts via `trap` on exit, success or failure. Gets results off the pod and then
# deletes the pod, so an unattended run never bills past the end of the work (on 2026-10-11 a finished
# pod sat idle ~7 h until the account's credit ran out, and the stage-2 results died with it).
#
# Usage: finish.sh <run dir> [more paths to save...]
#   HF_UPLOAD_REPO=user/brahui-asr-runs  upload <run>/final, evals/asr/runs, logs/ to this private HF repo
#                                        (needs HF_TOKEN; creates the repo if missing). The pod cannot
#                                        reach the Mac, so this is the only push that works without a
#                                        network volume.
#   KEEP_POD=1                           skip the delete (e.g. for debugging a failure).
#   PULL_GRACE_MIN=N                     wait up to N minutes for /workspace/.pulled (touched by the Mac after it has
#                                        rsynced the results off) before deleting; results also stay on the volume.
#                                        While any /workspace/.hold* exists the wait continues (another job on the pod), up to 8 h. Each job
#                                        uses its own /workspace/.hold.<job> so one job ending cannot release another's hold (2026-10-25 race).
# Inside a RunPod pod, RUNPOD_POD_ID and RUNPOD_API_KEY are set by the platform.
set -uo pipefail
cd "$(dirname "$0")/../.."
STATUS=${1:-0}; shift || true
# Over SSH the RunPod env vars are not inherited; read them from the container's init process instead.
if [ -z "${RUNPOD_POD_ID:-}" ] && [ -r /proc/1/environ ]; then
  eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
fi
RUN=${1:-}; shift || true
log() { echo "[$(date -u +%H:%M:%S)] finish: $*"; }
log "job exit status $STATUS"

if [ -n "${HF_UPLOAD_REPO:-}" ]; then
  if [ -z "${HF_TOKEN:-}" ]; then
    log "HF_UPLOAD_REPO set but HF_TOKEN missing; nothing uploaded"
  else
    uv run python - "$HF_UPLOAD_REPO" "$RUN" "$@" <<'PY'
import os, sys
from huggingface_hub import HfApi
repo, run, *extra = sys.argv[1:]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, private=True, exist_ok=True)
paths = [(f"{run}/final", f"{run}/final"), ("evals/asr/runs", "evals/asr/runs"),
         ("evals/RESULTS_TABLES.md", "evals/RESULTS_TABLES.md"), ("logs", "logs")] + [(p, p) for p in extra]
for src, dst in paths:
    if os.path.isdir(src):
        api.upload_folder(repo_id=repo, folder_path=src, path_in_repo=dst, commit_message=f"{src}")
        print("uploaded dir", src)
    elif os.path.isfile(src):
        api.upload_file(repo_id=repo, path_or_fileobj=src, path_in_repo=dst, commit_message=f"{src}")
        print("uploaded file", src)
    else:
        print("missing, skipped", src)
PY
    log "upload to https://huggingface.co/$HF_UPLOAD_REPO finished with status $?"
  fi
else
  log "HF_UPLOAD_REPO not set: results stay on this pod's /workspace volume only"
fi

if [ "${KEEP_POD:-0}" = "1" ]; then
  log "KEEP_POD=1, leaving pod ${RUNPOD_POD_ID:-?} running (it is still billing)"
elif [ -z "${RUNPOD_POD_ID:-}" ]; then
  log "not on RunPod (RUNPOD_POD_ID unset); terminate the machine yourself"
else
  if [ "${PULL_GRACE_MIN:-0}" -gt 0 ]; then
    rm -f /workspace/.pulled
    log "waiting up to $PULL_GRACE_MIN min for /workspace/.pulled before deleting the pod"
    for i in $(seq 1 480); do
      [ -f /workspace/.pulled ] && { log "pulled after $i min"; break; }
      [ "$i" -ge "$PULL_GRACE_MIN" ] && ! ls /workspace/.hold* >/dev/null 2>&1 && { log "grace over after $i min"; break; }
      sleep 60
    done
  fi
  log "deleting pod $RUNPOD_POD_ID"
  sync
  if command -v runpodctl >/dev/null; then
    runpodctl remove pod "$RUNPOD_POD_ID"
  else
    curl -fsS -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer ${RUNPOD_API_KEY:?}"
  fi
fi
