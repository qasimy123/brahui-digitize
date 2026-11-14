"""Extract Brahui sentences from the FineWeb-2 brh_Arab slice (data/raw/fineweb2_brh) for back-translation.

Per-sentence filter (the slice is mixed Brahui/Urdu): 6–60 words, ≥ 90% Arabic script, Brahui function-word
rate ≥ 10%, Urdu function-word rate ≤ 2%, exact dedupe, sealed test sentences excluded. Deterministically
shuffled so any prefix is a random subset.
Writes data/parallel/fineweb_units.jsonl  {id, source: "fineweb2", brh}
"""
from __future__ import annotations
import glob, json, random, re, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from brahui.text import normalize, arabic_script_ratio  # noqa: E402

BRH = set("ٹی نا آن اسے کنے مریک ارے ئے اسہ تینا ننا نمے پدا ہنو داسا ئس مس کرے ہلک جوان دا او ننے اٹی تون ئی ہم آ تے کہ انت ارا امر وڑ ڈول کنا نن نم اوفتا اونا".split())
URD = set("ہے ہیں تھا تھی تھے کے کی کا میں نے کو سے اور یہ وہ ہوں گا گی گے ہو رہا رہی رہے لیے لئے".split())
SPLIT = re.compile(r"(?<=[۔؟!])\s+|\n+")


def main():
    sealed = set()
    for f, k in [("cv_test.jsonl", "text"), ("omni_test.jsonl", "text"), ("smol_heldout.jsonl", "tgt"), ("bouquet_test.jsonl", "src")]:
        sealed |= {normalize(json.loads(l)[k]) for l in open(ROOT / "data/eval" / f, encoding="utf-8")}
    docs = []
    for f in sorted(glob.glob(str(ROOT / "data/raw/fineweb2_brh/data/brh_Arab/*/*.parquet"))):
        docs += pd.read_parquet(f)["text"].tolist()
    seen, out = set(), []
    for d in docs:
        for s in SPLIT.split(d):
            s = normalize(s); w = s.split()
            if not (6 <= len(w) <= 60) or arabic_script_ratio(s) < 0.9 or s in seen or s in sealed:
                continue
            b = sum(x in BRH for x in w) / len(w); u = sum(x in URD for x in w) / len(w)
            if b >= 0.10 and u <= 0.02:
                seen.add(s); out.append(s)
    random.Random(20261019).shuffle(out)
    p = ROOT / "data/parallel/fineweb_units.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i, s in enumerate(out):
            f.write(json.dumps({"id": f"fw2_{i:05d}", "source": "fineweb2", "brh": s}, ensure_ascii=False) + "\n")
    print(f"{len(docs)} docs -> {len(out)} sentences, {sum(len(s.split()) for s in out)} words -> {p}")


if __name__ == "__main__":
    main()
