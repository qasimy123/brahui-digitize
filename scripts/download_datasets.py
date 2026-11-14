"""Download all Brahui datasets listed in docs/research/01-brahui-resource-inventory.md.

Usage: uv run python scripts/download_datasets.py [--skip-large]
Everything lands in data/raw/<name>/ . Re-runnable; already-complete files are skipped.
"""
import argparse, os, sys, urllib.request
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"

HF = [
    # (repo_id, repo_type, allow_patterns, local_dir, is_large)
    ("facebook/omnilingual-asr-corpus", "dataset", ["README.md", "data/brh_Arab/*"], "omnilingual_asr_brh", True),
    ("TBOGamer22/BrahuiSpeech-70H-V2", "dataset", None, "brahuispeech_70h_v2", True),
    ("google/smol", "dataset", ["README.md", "smoldoc-factuality-ratings.json", "smoldoc/*brh*", "smolsent/*brh*", "gatitos/*brh*"], "smol_brh", False),
    ("facebook/bouquet", "dataset", ["README.md", "*brh_Arab*"], "bouquet_brh", False),
    ("HuggingFaceFW/fineweb-2", "dataset", ["README.md", "data/brh_Arab/*"], "fineweb2_brh", False),
    ("HuggingFaceFW/finepdfs", "dataset", ["README.md", "data/brh_Arab/*"], "finepdfs_brh", False),
    ("cis-lmu/GlotCC-V1", "dataset", ["README.md", "v1.0/brh-Arab/*"], "glotcc_brh", False),
    ("TBOGamer22/nemo-brahvi-g2p-conformer", "model", None, "models/nemo_brahvi_g2p", False),
    ("TBOGamer22/Brahvi-Whisper-Small", "model", None, "models/brahvi_whisper_small", False),
]

URLS = {
    "ud_brahui_kholum/brh_kholum-ud-test.conllu": "https://raw.githubusercontent.com/UniversalDependencies/UD_Brahui-Kholum/master/brh_kholum-ud-test.conllu",
    "ud_brahui_kholum/README.md": "https://raw.githubusercontent.com/UniversalDependencies/UD_Brahui-Kholum/master/README.md",
    "cldr/brh.xml": "https://raw.githubusercontent.com/unicode-org/cldr/main/common/main/brh.xml",
    "common_voice/cv-corpus-27.0-2026-09-11.json": "https://raw.githubusercontent.com/common-voice/cv-dataset/main/datasets/scripted-speech/cv-corpus-27.0-2026-09-11.json",
    "omnivoice/languages.md": "https://raw.githubusercontent.com/k2-fsa/OmniVoice/main/docs/languages.md",
    "global_recordings/Brahui_brh_mp3.zip": "https://media.globalrecordings.net/language_mp3_zip/Brahui%20%5Bbrh%5D%20mp3.zip",
    "grammar/barjasteh_delforooz_2020_brahui.pdf": "https://ijals.usb.ac.ir/article_5661_6be1605a6671b6661f41e7ebde97c472.pdf",
}

def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 brahui-digitize"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    return "ok"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-large", action="store_true"); ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    for repo, rtype, pats, local, large in HF:
        if a.skip_large and large: continue
        if a.only and not any(o in repo or o in local for o in a.only): continue
        print(f"== {repo} -> {local}", flush=True)
        try: snapshot_download(repo, repo_type=rtype, allow_patterns=pats, local_dir=ROOT / local, max_workers=8)
        except Exception as e: print(f"FAIL {repo}: {str(e)[:200]}", flush=True)
    if not a.only or any(o == "urls" for o in a.only):
        for rel, url in URLS.items():
            try: print(f"{fetch(url, ROOT / rel):4} {rel}", flush=True)
            except Exception as e: print(f"FAIL {rel}: {e}", flush=True)
    print("done")

if __name__ == "__main__":
    main()
