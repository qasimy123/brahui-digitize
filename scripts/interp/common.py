"""Shared bits for the NLLB interpretability scripts: model paths, the tri-parallel Brahui/Urdu/English set, loading."""
import json
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evals/interp"
BASE = "facebook/nllb-200-distilled-1.3B"
FT = str(ROOT / "runs/nllb_r5_retrans/final")
BRH_TAG, URD_TAG, ENG_TAG = "urd_Arab", "urd_Arab", "eng_Latn"  # Brahui was trained under NLLB's urd_Arab code
TRI = OUT / "bouquet_dev_triparallel.jsonl"


def triparallel():
    """504 BOUQuET dev sentences in Brahui, Urdu and English (dev split, not the sealed test)."""
    if not TRI.exists():
        d = ROOT / "data/raw/bouquet_brh/benchmark_data/sentence_level/dev"
        a = pd.read_parquet(d / "brh_Arab-eng_Latn.parquet")[["uniq_id", "src_text", "tgt_text", "domain"]]
        b = pd.read_parquet(d / "urd_Arab-brh_Arab.parquet")[["uniq_id", "src_text"]]
        m = a.merge(b, on="uniq_id", suffixes=("", "_urd"))
        OUT.mkdir(parents=True, exist_ok=True)
        with TRI.open("w") as f:
            for r in m.itertuples():
                f.write(json.dumps({"id": r.uniq_id, "domain": r.domain, "brh": r.src_text, "urd": r.src_text_urd, "en": r.tgt_text},
                                   ensure_ascii=False) + "\n")
    return [json.loads(l) for l in TRI.open()]


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load(which, dtype=torch.float32):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE if which == "base" else FT, dtype=dtype).to(device()).eval()
    return model, tok
