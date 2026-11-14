"""Translate two ASR transcripts of one video with several offline NLLB students: writes data/media/demo_transcripts/nllb_<tag>.jsonl.
Usage: uv run python scripts/demo_compare.py <tag> <model_dir> <transcript.jsonl> [<transcript2.jsonl> ...]
"""
import json, sys
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from train_nllb import translate, BRH, ENG  # noqa: E402
tag, mdir, *files = sys.argv[1:]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(mdir); mdl = AutoModelForSeq2SeqLM.from_pretrained(mdir, dtype=torch.float32).to(dev).eval()
out = ROOT / "data/media/demo_transcripts" / f"nllb_{tag}.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for p in files:
        asr = Path(p).parent.parent.name  # s2 / s3
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        hyps = translate(mdl, tok, [r["text"] for r in rows], BRH, ENG, bs=4, beams=4, max_new=300)
        for r, h in zip(rows, hyps):
            f.write(json.dumps({"asr": asr, "seg": r["seg"], "start": r["start"], "end": r["end"], "brh": r["text"], "en": h}, ensure_ascii=False) + "\n")
print("wrote", out)
