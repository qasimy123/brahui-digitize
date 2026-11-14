"""Push the newest training checkpoint and the logs to a private HF repo when a pod job fails, so the exit trap can still
delete the pod without losing the run (2026-10-22: a disk-full crash at the final save lost a 65-min VITS run).

  python scripts/gpu/salvage_upload.py <repo> <run dir> <logs dir>     (HF_TOKEN in the env)
"""
import os, sys
from huggingface_hub import HfApi

repo, run, logs = sys.argv[1:4]
api = HfApi(); api.create_repo(repo, private=True, exist_ok=True)
out = os.path.join(run, "out")
cks = sorted((d for d in os.listdir(out) if d.startswith("checkpoint-")), key=lambda d: int(d.split("-")[1])) if os.path.isdir(out) else []
if cks:
    print("salvage: uploading", cks[-1], "->", repo, flush=True)
    api.upload_folder(folder_path=os.path.join(out, cks[-1]), repo_id=repo, path_in_repo=cks[-1])
else:
    print("salvage: no checkpoint found", flush=True)
if os.path.isdir(logs):
    api.upload_folder(folder_path=logs, repo_id=repo, path_in_repo="logs", allow_patterns=["*.log"])
print("salvage: done", flush=True)
