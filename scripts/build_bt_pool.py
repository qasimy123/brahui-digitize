"""English pool for back-translation (teacher en→brh gives synthetic Brahui inputs; the real English is the target).

Samples Tatoeba English sentences not already in data/parallel/en_everyday.jsonl, plus FineWeb-derived English we
already hold (synthetic_fineweb en side is Brahui-origin, so skipped), writing data/distill/en_bt_pool.jsonl.

  uv run python scripts/build_bt_pool.py --n 200000
"""
import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TAT = ROOT / "data/raw/en_everyday/tatoeba_sentences.csv"
OUT = ROOT / "data/distill/en_bt_pool.jsonl"


def ok(s, lo=4, hi=40):
    w = s.split()
    return lo <= len(w) <= hi and re.search(r"[A-Za-z]", s) and not re.search(r"https?://|@|#|\d{5,}", s)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200000); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    used = {json.loads(l)["en"].strip().lower() for l in open(ROOT / "data/parallel/en_everyday.jsonl", encoding="utf-8")}
    for f in ("synthetic_bt_everyday.jsonl",):
        used |= {json.loads(l)["en"].strip().lower() for l in open(ROOT / "data/parallel" / f, encoding="utf-8")}
    eng, seen = [], set()
    csv.field_size_limit(1 << 24)
    with open(TAT, encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.reader(f, delimiter="\t")):  # format: <lang>\t<sentence>
            if len(row) < 2 or row[0] != "eng":
                continue
            s = row[1].strip()
            k = s.lower()
            if k in used or k in seen or not ok(s):
                continue
            seen.add(k); eng.append((i, s))
    print(f"{len(eng)} candidate English sentences")
    rng = random.Random(a.seed); pick = rng.sample(eng, min(a.n, len(eng)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for tid, s in sorted(pick, key=lambda x: int(x[0])):
            f.write(json.dumps({"id": f"tatbt_{tid}", "source": "tatoeba_bt", "en": s}, ensure_ascii=False) + "\n")
    print(f"{len(pick)} rows -> {OUT}")


if __name__ == "__main__":
    main()
