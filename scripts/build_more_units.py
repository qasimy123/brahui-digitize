"""Extract Brahui sentences from FinePDFs (data/raw/finepdfs_brh) and GlotCC (data/raw/glotcc_brh) with the same
per-sentence filter as scripts/build_fineweb_units.py, minus sentences already in fineweb_units.jsonl or any
training corpus, minus sealed test sentences. Writes data/parallel/more_units.jsonl {id, source, brh}."""
from __future__ import annotations
import glob, json, random, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from brahui.text import normalize, arabic_script_ratio  # noqa: E402
from build_fineweb_units import BRH, URD, SPLIT  # noqa: E402


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    sealed = set()
    for f, k in [("cv_test.jsonl", "text"), ("omni_test.jsonl", "text"), ("smol_heldout.jsonl", "tgt"), ("bouquet_test.jsonl", "src")]:
        sealed |= {normalize(r[k]) for r in rj(ROOT / "data/eval" / f)}
    used = set()
    for f in glob.glob(str(ROOT / "data/parallel/*.jsonl")):
        if Path(f).name == "more_units.jsonl": continue
        used |= {normalize(r["brh"]) for r in rj(f) if "brh" in r}
    docs = []
    for f in sorted(glob.glob(str(ROOT / "data/raw/finepdfs_brh/data/**/*.parquet"), recursive=True)):
        df = pd.read_parquet(f); df = df[df["full_doc_lid_score"] >= 0.7]  # the slice is noisy: median doc LID 0.59
        docs += [("finepdfs", t) for t in df["text"].tolist()]
    for f in sorted(glob.glob(str(ROOT / "data/raw/glotcc_brh/**/*.parquet"), recursive=True)):
        docs += [("glotcc", t) for t in pd.read_parquet(f)["content"].tolist()]
    seen, out, stats = set(), [], {}
    for src, d in docs:
        for s in SPLIT.split(d or ""):
            s = normalize(s); w = s.split()
            stats[src] = stats.get(src, 0) + 1
            if not (6 <= len(w) <= 60) or arabic_script_ratio(s) < 0.9 or s in seen or s in sealed or s in used:
                continue
            b = sum(x in BRH for x in w) / len(w); u = sum(x in URD for x in w) / len(w)
            if b >= 0.10 and u <= 0.02:
                seen.add(s); out.append((src, s))
    random.Random(20261019).shuffle(out)
    p = ROOT / "data/parallel/more_units.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i, (src, s) in enumerate(out):
            f.write(json.dumps({"id": f"{src[:3]}_{i:05d}", "source": src, "brh": s}, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"{len(docs)} docs, raw sentences {stats} -> {len(out)} kept {Counter(s for s, _ in out)}, {sum(len(s.split()) for _, s in out)} words -> {p}")


if __name__ == "__main__":
    main()
