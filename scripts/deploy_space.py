"""Publish the demo to Hugging Face: two private model repos (ASR s3, NLLB r5) + a Gradio Space (demo/hf_space/).

  HF_TOKEN=hf_... uv run python scripts/deploy_space.py [--hardware zero-a10g|cpu-basic] [--public] [--skip-models]

The token must have write access. The Space gets the token as a secret so it can read the private model repos.
Re-running is idempotent (upload_folder only ships changed files).
"""
import argparse, os, sys
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
MODELS = [("runs/ctc1b_brh_s3/final", "qkhawaja/brahui-asr-ctc1b-s3"), ("runs/nllb_r5_retrans/final", "qkhawaja/brahui-nllb-1.3b-r5")]
SPACE = "qkhawaja/brahui-demo"

ap = argparse.ArgumentParser()
ap.add_argument("--hardware", default="zero-a10g"); ap.add_argument("--public", action="store_true"); ap.add_argument("--skip-models", action="store_true")
a = ap.parse_args()
tok = os.environ.get("HF_TOKEN")
api = HfApi(token=tok)
who = api.whoami()
if who["auth"]["accessToken"]["role"] not in ("write", "fineGrained"):
    sys.exit(f"token role is {who['auth']['accessToken']['role']}; need a write token (https://huggingface.co/settings/tokens)")

if not a.skip_models:
    for local, repo in MODELS:
        api.create_repo(repo, private=True, exist_ok=True)
        print("uploading", local, "->", repo, flush=True)
        api.upload_folder(folder_path=str(ROOT / local), repo_id=repo, commit_message="demo checkpoint")
        print("  done", flush=True)

api.create_repo(SPACE, repo_type="space", space_sdk="gradio", private=not a.public, exist_ok=True)
api.add_space_secret(SPACE, "HF_TOKEN", tok)
api.upload_folder(folder_path=str(ROOT / "demo/hf_space"), repo_id=SPACE, repo_type="space", commit_message="demo app")
try:
    api.request_space_hardware(SPACE, a.hardware)
    print("hardware requested:", a.hardware)
except Exception as e:
    print(f"could not set hardware {a.hardware}: {e}\n  -> set it in the Space settings, or rerun with --hardware cpu-basic")
print(f"space: https://huggingface.co/spaces/{SPACE}  (build takes a few minutes; models load on first start)")
