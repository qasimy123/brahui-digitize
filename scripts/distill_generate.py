"""Teacher pass for sequence-level distillation: translate every pool row brh→en with the (trimmed) r5 teacher.

Resumable: appends to --out and skips ids already present. Batches are length-sorted for throughput.

  uv run python scripts/distill_generate.py --teacher runs/nllb_r5_trim/final --pool data/distill/brh_pool.jsonl \
      --out data/distill/teacher_brh2en.jsonl --bs 48 --beams 4
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from train_nllb import BRH, ENG, device, read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs/nllb_r5_trim/final")
    ap.add_argument("--pool", default="data/distill/brh_pool.jsonl")
    ap.add_argument("--out", default="data/distill/teacher_brh2en.jsonl")
    ap.add_argument("--bs", type=int, default=48); ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=200); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--dir", default="brh2en", choices=["brh2en", "en2brh"], help="en2brh = back-translate English rows into synthetic Brahui")
    a = ap.parse_args()
    from transformers import AutoTokenizer, M2M100ForConditionalGeneration
    dev = device()
    src_key, out_key = ("brh", "en") if a.dir == "brh2en" else ("en", "brh")
    src_lang, tgt_lang = (BRH, ENG) if a.dir == "brh2en" else (ENG, BRH)
    tok = AutoTokenizer.from_pretrained(ROOT / a.teacher); tok.src_lang = src_lang
    dtype = torch.float16 if (dev == "cuda" and a.fp16) else torch.float32
    model = M2M100ForConditionalGeneration.from_pretrained(ROOT / a.teacher, dtype=dtype).to(dev).eval()
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    rows = [r for r in read_jsonl(ROOT / a.pool) if r["id"] not in done][: a.limit or None]
    rows.sort(key=lambda r: len(r[src_key]), reverse=True)  # longest first: OOM shows up immediately, not at hour 2
    print(f"device={dev} dtype={dtype} teacher={a.teacher} todo={len(rows)} already={len(done)}", flush=True)
    bos = tok.convert_tokens_to_ids(tgt_lang)
    t0, n = time.time(), 0
    with open(out, "a", encoding="utf-8") as f, torch.no_grad():
        for i in range(0, len(rows), a.bs):
            batch = rows[i:i + a.bs]
            enc = tok([r[src_key] for r in batch], return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)
            gen = model.generate(**enc, forced_bos_token_id=bos, num_beams=a.beams, max_new_tokens=a.max_new)
            hyps = tok.batch_decode(gen, skip_special_tokens=True)
            for r, h in zip(batch, hyps):
                f.write(json.dumps({**r, out_key: h.strip()}, ensure_ascii=False) + "\n")
            f.flush(); n += len(batch)
            if (i // a.bs) % 20 == 0:
                el = time.time() - t0
                print(f"{n}/{len(rows)} {el:.0f}s {n/el:.1f} sent/s eta {(len(rows)-n)/(n/el)/60:.0f} min", flush=True)
    print(f"done; {n} rows in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
