"""Publish the in-browser demo: ONNX model repos on the HF Hub + the static page as an HF static Space.

  HF_TOKEN=hf_... uv run python scripts/deploy_web.py --mt export/student_c --asr export/ctc300m_student [--private] [--skip-models]

Uploads only what transformers.js needs (config/tokenizer/preprocessor files + onnx/*_quantized.onnx[_data]), writes
demo/web/config.js pointing the page at the repos, and uploads demo/web/ (minus models/ and samples' large files) as a
static Space. Model repos must be public (or the viewer logged in) for the browser to fetch them.
"""
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--mt", default="export/student_c"); ap.add_argument("--asr", default="export/ctc300m_student")
ap.add_argument("--user", default=None); ap.add_argument("--space", default="brahui-local")
ap.add_argument("--lm", default="demo/web/models/brh_char6.blm", help="char LM table from scripts/export_char_lm.py (\"\" to skip)");
ap.add_argument("--private", action="store_true"); ap.add_argument("--skip-models", action="store_true")
ap.add_argument("--variants", default="quantized", help="comma list of onnx suffixes to upload (quantized,q4,fp16). The 1B ASR (export/ctc1b_s3) is fp16-only: pass --asr export/ctc1b_s3 --variants fp16 and set config.js asr accordingly.")
a = ap.parse_args()
tok = os.environ.get("HF_TOKEN") or sys.exit("HF_TOKEN (write scope) required")
api = HfApi(token=tok); user = a.user or api.whoami()["name"]
KEEP = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "preprocessor_config.json", "vocab.json"}
repos = {}
for kind, local in (("mt", a.mt), ("asr", a.asr)):
    d = ROOT / local
    if not d.exists():
        print(f"skip {kind}: {d} missing"); continue
    repo = f"{user}/brahui-{kind}-{d.name.replace('_', '-')}-onnx"; repos[kind] = repo
    if a.skip_models:
        continue
    api.create_repo(repo, private=a.private, exist_ok=True)
    allow = [f for f in KEEP] + [f"onnx/*_{v}.onnx" for v in a.variants.split(",")] + [f"onnx/*_{v}.onnx_data" for v in a.variants.split(",")]
    api.upload_folder(folder_path=str(d), repo_id=repo, allow_patterns=allow, commit_message=f"{local} for transformers.js")
    print("uploaded", repo)
    if kind == "asr" and a.lm:  # char n-gram LM table for the browser beam search (demo/web/ctc_lm.js) rides along in the ASR repo
        api.upload_file(path_or_fileobj=str(ROOT / a.lm), path_in_repo=Path(a.lm).name, repo_id=repo, commit_message="char LM for CTC beam search")
        repos["lm"] = f"https://huggingface.co/{repo}/resolve/main/{Path(a.lm).name}"
(ROOT / "demo/web/config.js").write_text("window.BRAHUI_MODELS = " + __import__("json").dumps({"remote": True, **repos}) + ";\n")
space = f"{user}/{a.space}"
api.create_repo(space, repo_type="space", space_sdk="static", private=a.private, exist_ok=True)
api.upload_folder(folder_path=str(ROOT / "demo/web"), repo_id=space, repo_type="space", ignore_patterns=["models/*", "models", "*.py", "__pycache__/*"],
                  commit_message="static in-browser demo")
print(f"space: https://huggingface.co/spaces/{space}  models: {repos}")
