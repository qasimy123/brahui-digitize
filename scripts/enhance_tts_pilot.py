"""Pilot: does speech enhancement lift BrahuiSpeech clips over the TTS quality bar? Enhances the 10 s centre crop of
N clips from a DNSMOS band with SpeechBrain models and rescored with DNSMOS (same crop as scripts/score_tts_pool.py).

  uv run --with speechbrain --with speechmos python scripts/enhance_tts_pilot.py [--n 40] [--lo 2.0] [--hi 2.8]
Writes evals/tts/enhance_pilot/{results.jsonl,summary.json} and a few before/after wavs for listening.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np, soundfile as sf, torch
from speechmos import dnsmos

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40); ap.add_argument("--lo", type=float, default=2.0); ap.add_argument("--hi", type=float, default=2.8)
a = ap.parse_args()
out = ROOT / "evals/tts/enhance_pilot"; (out / "wavs").mkdir(parents=True, exist_ok=True)
pool = [r for r in map(json.loads, open(ROOT / "data/tts/bs_mos.jsonl")) if "error" not in r and a.lo <= r["ovrl"] < a.hi]
random.Random(0).shuffle(pool); pool = pool[:a.n]

from speechbrain.inference.enhancement import SpectralMaskEnhancement
from speechbrain.inference.separation import SepformerSeparation
models = {
    "metricgan": SpectralMaskEnhancement.from_hparams("speechbrain/metricgan-plus-voicebank", savedir=str(ROOT / "data/raw/models/metricgan")),
    "sepformer_dns4": SepformerSeparation.from_hparams("speechbrain/sepformer-dns4-16k-enhancement", savedir=str(ROOT / "data/raw/models/sepformer_dns4")),
}
def run(name, y):
    x = torch.from_numpy(y)[None]
    with torch.no_grad():
        if name == "metricgan": return models[name].enhance_batch(x, lengths=torch.tensor([1.0]))[0].cpu().numpy()
        return models[name].separate_batch(x)[0, :, 0].cpu().numpy()
mos = lambda y: {k[:-4]: round(float(v), 3) for k, v in dnsmos.run(y / (np.max(np.abs(y)) + 1e-9) * 0.9, sr=16000).items() if k.endswith("_mos")}
rows = []; secs = {k: 0.0 for k in models}
with open(out / "results.jsonl", "w") as f:
    for i, r in enumerate(pool):
        y, sr = sf.read(ROOT / f"data/audio/brahuispeech/{r['id']}.wav", dtype="float32", always_2d=True); y = y.mean(axis=1)
        if len(y) > 160000: o = (len(y) - 160000) // 2; y = y[o:o + 160000]
        rec = {"id": r["id"], "raw": mos(y)}
        for name in models:
            t = time.time(); e = run(name, y).astype(np.float32); secs[name] += time.time() - t; rec[name] = mos(e)
            if i < 6: sf.write(out / "wavs" / f"{r['id']}_{name}.wav", e / (np.abs(e).max() + 1e-9) * 0.9, 16000)
        if i < 6: sf.write(out / "wavs" / f"{r['id']}_raw.wav", y, 16000)
        rows.append(rec); f.write(json.dumps(rec) + "\n"); f.flush(); print(i + 1, rec, flush=True)
summ = {"n": len(rows), "band": [a.lo, a.hi]}
for k in ["raw", *models]:
    o = np.array([r[k]["ovrl"] for r in rows])
    summ[k] = {"ovrl_mean": round(float(o.mean()), 3), "sig_mean": round(float(np.mean([r[k]["sig"] for r in rows])), 3), "bak_mean": round(float(np.mean([r[k]["bak"] for r in rows])), 3),
               "share_ge_3.0": round(float((o >= 3.0).mean()), 2), "share_ge_2.8": round(float((o >= 2.8).mean()), 2), "s_per_clip": round(secs.get(k, 0) / len(rows), 2)}
json.dump(summ, open(out / "summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))
