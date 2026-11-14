"""Human everyday-register English for iterated back-translation: DailyDialog utterances (CC BY-NC-SA 4.0) and a
seeded sample of Tatoeba English sentences (CC BY 2.0). Chosen from the brh->en failure buckets of 2026-10-19:
the student mistranslates everyday concrete vocabulary that the news/religion corpus barely contains.

Usage: uv run python scripts/build_en_everyday.py [--tatoeba 60000]
Writes data/parallel/en_everyday.jsonl {id, source, en}. Run scripts/dedup_against_eval.py afterwards."""
import argparse, csv, json, random, re, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; RAW = ROOT / "data/raw/en_everyday"


def clean(s):
    s = re.sub(r"\s+([,.!?;:'’])", r"\1", s.strip()); s = re.sub(r"\s+", " ", s)
    s = s.replace(" n't", "n't").replace("’ ", "’").replace(" ' ", "'")
    return s


def ok(s, lo=4, hi=40):
    w = s.split()
    return lo <= len(w) <= hi and re.fullmatch(r"[A-Za-z0-9 ,.'’\"!?;:\-()$%&/]+", s) is not None and not re.search(r"https?://|www\.", s)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tatoeba", type=int, default=60000); a = ap.parse_args()
    rows, seen = [], set()
    for split in ["train", "validation"]:
        for i, u in enumerate(pd.read_parquet(RAW / f"dailydialog_{split}.parquet")["utterance"]):
            s = clean(u)
            if ok(s) and s.lower() not in seen:
                seen.add(s.lower()); rows.append({"id": f"dd_{split}_{i}", "source": "dailydialog", "en": s})
    n_dd = len(rows)
    eng = []
    with open(RAW / "tatoeba_sentences.csv", encoding="utf-8") as f:
        for i, line in enumerate(csv.reader(f, delimiter="\t")):
            if len(line) >= 2 and line[0] == "eng":
                s = clean(line[1])
                if ok(s, 4, 30) and s.lower() not in seen:
                    seen.add(s.lower()); eng.append((i, s))
    rng = random.Random(2026); pick = rng.sample(eng, min(a.tatoeba, len(eng)))
    rows += [{"id": f"tat_{i}", "source": "tatoeba", "en": s} for i, s in sorted(pick)]
    out = ROOT / "data/parallel/en_everyday.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"dailydialog {n_dd}, tatoeba {len(rows) - n_dd} of {len(eng)} English sentences -> {out}")


if __name__ == "__main__":
    main()
