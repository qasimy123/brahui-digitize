"""Carve a fixed MT dev split out of the gold training pairs (never out of data/eval/).

120 SmolSent sentences + whole BOUQuET-dev paragraphs until >= 180 sentences, seeded. Both sides are human
translations from English, so the split serves en->brh and brh->en. scripts/train_nllb.py excludes these ids from
training and reports dev scores every epoch; hillclimb decisions are made on dev, sealed test only confirms.
Writes data/dev/mt_dev.jsonl. Rerunning is idempotent (seeded)."""
import json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
rows = [json.loads(l) for l in open(ROOT / "data/parallel/gold_train.jsonl", encoding="utf-8")]
rng = random.Random(2026)
smol = sorted([r for r in rows if r["id"].startswith("smolsent_")], key=lambda r: r["id"])
dev = rng.sample(smol, 120)
paras = {}
for r in rows:
    if r["id"].startswith("bouquet_"):
        paras.setdefault(r["id"].split("-")[0], []).append(r)
keys = sorted(paras); rng.shuffle(keys)
n = 0
for k in keys:
    if n >= 180: break
    dev += paras[k]; n += len(paras[k])
out = ROOT / "data/dev/mt_dev.jsonl"; out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    for r in dev:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{len(dev)} dev pairs ({sum(r['id'].startswith('smolsent_') for r in dev)} smolsent, {n} bouquet-dev in {sum(1 for k in keys[:keys.index(k)+1])} paragraphs) -> {out}")
