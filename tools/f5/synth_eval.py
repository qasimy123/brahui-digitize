"""Synthesize an exported dev set with an F5-TTS checkpoint, cloning exported reference clips. Run with the F5 venv:

  tools/f5/.venv/bin/python tools/f5/synth_eval.py --ckpt <model_last.pt|.safetensors> --vocab <vocab.txt>
      --dev data/tts/f5/multi_arabic_dev --refs data/tts/f5/multi_arabic_refs --out evals/tts/f5_multi_arabic [--n 0] [--nfe 32] [--listen 5]
      [--export-ema runs/f5_multi_arabic/final/model.safetensors]

--dev/--refs are scripts/tts_f5_export.py dirs in the same text mode the model was trained on. Every dev sentence is
spoken in the first reference voice -> <out>/wavs/<id>.wav (score with scripts/tts_score_wavs.py); the first --listen
sentences are also spoken in the other reference voices -> <out>/ref<k>/. --export-ema writes the EMA weights alone
(1.3 GB instead of the 5.4 GB training checkpoint).
"""
import argparse, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--vocab", required=True); ap.add_argument("--dev", required=True); ap.add_argument("--refs", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=0); ap.add_argument("--nfe", type=int, default=32); ap.add_argument("--listen", type=int, default=5)
ap.add_argument("--export-ema", default=None); ap.add_argument("--speed", type=float, default=1.0); ap.add_argument("--raw", action="store_true")
a = ap.parse_args()
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") == "": os.environ.setdefault("OMP_NUM_THREADS", "5")   # CPU eval beside training: all cores starved the dataloaders (5.3 -> 1.0 update/s)

if a.export_ema:
    # --raw exports the live training weights instead of the EMA. When fine-tuning from the pretrained checkpoint the trainer
    # restores the EMA's step counter (~1.2M), so its decay starts at 0.9999 and after 10k updates the EMA is still ~90% base model.
    import torch
    from safetensors.torch import save_file
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    sd = {"ema_model." + k: v for k, v in ck["model_state_dict"].items()} if a.raw else ck["ema_model_state_dict"]
    del ck; Path(a.export_ema).parent.mkdir(parents=True, exist_ok=True); save_file(sd, a.export_ema); del sd
    print("raw" if a.raw else "EMA", "weights ->", a.export_ema, flush=True); a.ckpt = a.export_ema

def meta(d):
    d = Path(d); rows = [l.rstrip("\n").split("|", 1) for l in open(d / "metadata.csv", encoding="utf-8")][1:]
    return [(d / w, t) for w, t in rows]

from f5_tts.api import F5TTS  # noqa: E402
tts = F5TTS(model="F5TTS_v1_Base", ckpt_file=a.ckpt, vocab_file=a.vocab)
refs, dev = meta(a.refs), meta(a.dev)[:a.n or None]
out = Path(a.out); t0 = time.time(); quiet = lambda *x, **k: None  # noqa: E731
for i, (w, text) in enumerate(dev):
    for k, (rw, rt) in enumerate(refs if i < a.listen else refs[:1]):
        dst = out / ("wavs" if k == 0 else f"ref{k}") / w.name; dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            tts.infer(ref_file=str(rw), ref_text=rt, gen_text=text, file_wave=str(dst), nfe_step=a.nfe, speed=a.speed, seed=0, show_info=quiet, progress=None)
    if (i + 1) % 20 == 0 or i + 1 == len(dev): print(f"  {i + 1}/{len(dev)}  {time.time() - t0:.0f}s", flush=True)
