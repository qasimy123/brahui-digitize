"""Score an NLLB checkpoint on the MT dev set (brh2en by default). Reuses train_nllb.translate/score.

  uv run python scripts/eval_mt_dev.py --model runs/nllb_r5_trim/final --out evals/mt/serve/r5_trim_dev
"""
import argparse, json, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from train_nllb import translate, score, read_jsonl, device, BRH, ENG  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--dev", default="data/dev/mt_dev_clean.jsonl"); ap.add_argument("--dir", default="brh2en")
ap.add_argument("--beams", type=int, default=4); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
from transformers import AutoTokenizer, M2M100ForConditionalGeneration
tok = AutoTokenizer.from_pretrained(ROOT / a.model)
model = M2M100ForConditionalGeneration.from_pretrained(ROOT / a.model, dtype=torch.float32).to(device())
rows = read_jsonl(ROOT / a.dev)[: a.limit or None]
sk, tk, src, tgt = ("brh", "en", BRH, ENG) if a.dir == "brh2en" else ("en", "brh", ENG, BRH)
t0 = time.time()
hyps = translate(model, tok, [r[sk] for r in rows], src, tgt, bs=a.bs, beams=a.beams)
dt = time.time() - t0
res = score(hyps, [r[tk] for r in rows], brh_side=(tk == "brh")) | {"n": len(rows), "sec": round(dt, 1), "model": a.model, "dir": a.dir, "beams": a.beams, "device": device()}
out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
with open(out / f"{a.dir}_dev.jsonl", "w", encoding="utf-8") as f:
    for r, h in zip(rows, hyps):
        f.write(json.dumps({"id": r["id"], "src": r[sk], "ref": r[tk], "hyp": h}, ensure_ascii=False) + "\n")
json.dump(res, open(out / f"{a.dir}_summary.json", "w"), indent=2)
print(json.dumps(res))
