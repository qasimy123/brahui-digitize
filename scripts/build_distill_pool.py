"""Build the Brahui-side input pool for sequence-level distillation of the brh→en teacher.

Every unique Brahui sentence we have (gold, curated, FineWeb-2 units, synthetic back-translations) becomes one row;
the teacher later translates each to English and the student trains on (brh, teacher_en). Gold rows keep their human
English as `en_ref` so the trainer can use the human target instead of the teacher's.

  uv run python scripts/build_distill_pool.py   -> data/distill/brh_pool.jsonl
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PAR, OUT = ROOT / "data" / "parallel", ROOT / "data" / "distill" / "brh_pool.jsonl"
# (file, is_real_brahui, has_trusted_en)
SOURCES = [
    ("gold_train_nodev.jsonl", True, True),
    ("retranslated_brh_en.jsonl", True, False),
    ("synthetic_fineweb_brh_en_filtered.jsonl", True, False),
    ("fineweb_units.jsonl", True, False),
    ("more_units.jsonl", True, False),
    ("synthetic_bt_everyday.jsonl", False, False),
]
MIN_CHARS, MAX_CHARS = 6, 600


def main():
    dev_ids = {json.loads(l)["id"] for l in open(ROOT / "data/dev/mt_dev.jsonl", encoding="utf-8")}
    dev_brh = {json.loads(l)["brh"].strip() for l in open(ROOT / "data/dev/mt_dev.jsonl", encoding="utf-8")}
    bad = {json.loads(l)["id"] for l in open(PAR / "eval_overlap.jsonl", encoding="utf-8")} if (PAR / "eval_overlap.jsonl").exists() else set()
    for f in (ROOT / "data/eval").glob("*.jsonl"):
        for l in open(f, encoding="utf-8"):
            d = json.loads(l)
            for k in ("brh", "src", "tgt"):
                if isinstance(d.get(k), str) and any("؀" <= c <= "ۿ" for c in d[k]):
                    dev_brh.add(d[k].strip())
    seen, rows, stats = set(), [], Counter()
    for fname, real, trusted in SOURCES:
        for l in open(PAR / fname, encoding="utf-8"):
            d = json.loads(l)
            b = (d.get("brh") or "").strip()
            if not b or b in seen or d["id"] in dev_ids or d["id"] in bad or b in dev_brh:
                stats[f"{fname}:dropped"] += 1; continue
            if not (MIN_CHARS <= len(b) <= MAX_CHARS):
                stats[f"{fname}:len"] += 1; continue
            seen.add(b)
            row = {"id": d["id"], "source": fname.replace(".jsonl", ""), "brh": b, "real": real}
            if trusted and d.get("en"):
                row["en_ref"] = d["en"].strip()
            rows.append(row); stats[fname] += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=1)); print(f"{len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
