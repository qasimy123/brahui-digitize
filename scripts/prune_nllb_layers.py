"""Build a shallow student by keeping a subset of a (trimmed) NLLB teacher's layers, weights copied verbatim.

Evenly spaced layers incl. first and last work best as an init for distillation (Sajjad et al. 2023 style).

  uv run python scripts/prune_nllb_layers.py --src runs/nllb_r5_trim/final --dst runs/student_a_init --enc 6 --dec 3
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def spaced(n_keep, n_total):
    if n_keep >= n_total:
        return list(range(n_total))
    if n_keep == 1:
        return [n_total - 1]
    return sorted({round(i * (n_total - 1) / (n_keep - 1)) for i in range(n_keep)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/nllb_r5_trim/final"); ap.add_argument("--dst", required=True)
    ap.add_argument("--enc", type=int, default=6); ap.add_argument("--dec", type=int, default=3)
    ap.add_argument("--enc-layers", default="", help="explicit comma list, overrides --enc"); ap.add_argument("--dec-layers", default="")
    a = ap.parse_args()
    from transformers import AutoTokenizer, M2M100ForConditionalGeneration
    src, dst = ROOT / a.src, ROOT / a.dst
    model = M2M100ForConditionalGeneration.from_pretrained(src, dtype=torch.float32)
    ne, nd = model.config.encoder_layers, model.config.decoder_layers
    ek = [int(x) for x in a.enc_layers.split(",")] if a.enc_layers else spaced(a.enc, ne)
    dk = [int(x) for x in a.dec_layers.split(",")] if a.dec_layers else spaced(a.dec, nd)
    enc, dec = model.model.encoder, model.model.decoder
    enc.layers = torch.nn.ModuleList([enc.layers[i] for i in ek])
    dec.layers = torch.nn.ModuleList([dec.layers[i] for i in dk])
    model.config.encoder_layers, model.config.decoder_layers = len(ek), len(dk)
    model.config.num_hidden_layers = len(ek)
    n = sum(p.numel() for p in model.parameters())
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst, safe_serialization=True)
    AutoTokenizer.from_pretrained(src).save_pretrained(dst)
    for f in ("generation_config.json",):
        if (src / f).exists() and not (dst / f).exists():
            shutil.copy(src / f, dst / f)
    info = {"src": str(a.src), "enc_layers_kept": ek, "dec_layers_kept": dk, "params_M": round(n / 1e6, 1)}
    (dst / "prune_summary.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
