"""Turn the OCR'd Shad dictionary (data/lexicon/shad_brahui_english_ocr.jsonl) into a compact lexicon text for in-context use."""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
rows = [json.loads(l) for l in open(ROOT / "data/lexicon/shad_brahui_english_ocr.jsonl", encoding="utf-8")]
seen = {}; order = []
for r in rows:
    hw = re.sub(r"\s+", " ", r["headword"]).strip()
    if not hw or not r["english"].strip(): continue
    k = (hw, r["english"].strip().lower())
    if k in seen: continue
    seen[k] = r; order.append(k)
with open(ROOT / "data/lexicon/shad_brh_en.txt", "w", encoding="utf-8") as f:
    f.write(f"# Shad, Brahui-English Dictionary (2021), OCR'd with Claude vision; {len(order)} entries. Format: headword (roman) [pos]: English; Urdu synonyms\n")
    for k in order:
        r = seen[k]
        pos = f" [{r['pos']}]" if r.get("pos") else ""
        urdu = f"; {r['urdu'].strip()}" if r.get("urdu", "").strip() else ""
        f.write(f"{r['headword'].strip()} ({r['roman'].strip()}){pos}: {r['english'].strip()}{urdu}\n")
print(len(rows), "rows ->", len(order), "entries; chars", (ROOT / "data/lexicon/shad_brh_en.txt").stat().st_size)
