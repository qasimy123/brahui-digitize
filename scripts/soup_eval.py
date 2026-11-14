"""Model soup: average the weights of several fine-tunes of the same base and evaluate (dev first, sealed on request).

Usage: uv run python scripts/soup_eval.py --runs runs/nllb_r0_s0/final runs/nllb_r2_bt/final --out runs/soup_r0s0_r2
           [--weights 1,1] [--sealed] [--device mps] [--eval-limit N]
Writes <out>/final and per-sentence outputs under evals/mt/nllb/<out name>_{dev,after}/ like train_nllb.py.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from train_nllb import evaluate, evaluate_dev  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=None); ap.add_argument("--sealed", action="store_true"); ap.add_argument("--device", default=None)
    ap.add_argument("--eval-limit", type=int); a = ap.parse_args()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from safetensors.torch import load_file
    w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0] * len(a.runs); w = [x / sum(w) for x in w]
    out = Path(a.out); (out / "final").mkdir(parents=True, exist_ok=True)
    if not (out / "final" / "model.safetensors").exists():
        acc = None
        for r, wi in zip(a.runs, w):
            sd = load_file(str(Path(r) / "model.safetensors"))
            if acc is None:
                acc = {k: v.float() * wi for k, v in sd.items()}
            else:
                for k in acc:
                    acc[k] += sd[k].float() * wi
        base = AutoModelForSeq2SeqLM.from_pretrained(a.runs[0], dtype=torch.float32)
        missing, unexpected = base.load_state_dict(acc, strict=False); print("missing", missing, "unexpected", unexpected)
        base.save_pretrained(out / "final"); AutoTokenizer.from_pretrained(a.runs[0]).save_pretrained(out / "final")
        print("soup saved:", [f"{r}×{wi:.2f}" for r, wi in zip(a.runs, w)])
    dev = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(out / "final")
    model = AutoModelForSeq2SeqLM.from_pretrained(out / "final", dtype=torch.float32).to(dev).eval()
    evaluate_dev(model, tok, f"{out.name}_dev", a.eval_limit)
    if a.sealed:
        evaluate(model, tok, f"{out.name}_after", a.eval_limit)


if __name__ == "__main__":
    main()
