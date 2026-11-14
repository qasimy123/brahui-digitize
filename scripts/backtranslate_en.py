"""Back-translate human English into Brahui with the en->brh student, for brh->en training (iterated back-translation).

Usage: uv run python scripts/backtranslate_en.py --model runs/<run>/final --corpus data/parallel/en_everyday.jsonl
           --out data/parallel/synthetic_bt_everyday.jsonl [--bs 64] [--beams 4] [--limit N]
Rows: {id, source, en (human), brh (synthetic), dirs: ["brh2en"], model}. scripts/train_nllb.py reads `dirs` and uses
these pairs for brh->en only, so the student never trains to imitate its own Brahui. Resumable by id.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from train_nllb import read_jsonl, translate, BRH, ENG, device  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--limit", type=int); a = ap.parse_args()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    dev = device(); tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    rows = read_jsonl(a.corpus); rows = rows[:a.limit] if a.limit else rows
    out = Path(a.out); done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"device={dev} model={a.model} {len(todo)} sentences to translate ({len(done)} done)", flush=True)
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), a.bs * 8):
            chunk = todo[i:i + a.bs * 8]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                hyps = translate(model, tok, [r["en"] for r in chunk], ENG, BRH, bs=a.bs, beams=a.beams, max_new=200)
            for r, h in zip(chunk, hyps):
                f.write(json.dumps({"id": r["id"], "source": r["source"], "en": r["en"], "brh": h, "dirs": ["brh2en"], "model": Path(a.model).parent.name}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {i + len(chunk)}/{len(todo)} {(i + len(chunk)) / (time.time() - t0):.1f} sent/s", flush=True)
    print(f"-> {out} ({len(read_jsonl(out))} rows)")


if __name__ == "__main__":
    main()
