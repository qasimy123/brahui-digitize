"""Find the dominant speakers in BrahuiSpeech (no speaker labels) for TTS: ECAPA speaker embeddings on every clip,
agglomerative clustering, and a per-clip noise proxy. Writes data/tts/bs_speakers.jsonl (id, cluster, noise_db, duration)
and data/tts/bs_speakers_summary.json (clusters ranked by hours, with mean noise).

  uv run --with speechbrain python scripts/cluster_speakers.py [--manifest data/manifests/brahuispeech_train_norm.jsonl]
      [--max-s 10] [--threshold 0.45]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="data/manifests/brahuispeech_train_norm.jsonl"); ap.add_argument("--max-s", type=float, default=10)
ap.add_argument("--threshold", type=float, default=0.45); ap.add_argument("--out", default="data/tts/bs_speakers")
a = ap.parse_args()
rows = [json.loads(l) for l in open(ROOT / a.manifest, encoding="utf-8")]
out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
emb_path = out.with_suffix(".npy"); meta_path = out.with_suffix(".meta.jsonl")

from speechbrain.inference.speaker import EncoderClassifier
dev = "mps" if torch.backends.mps.is_available() else "cpu"
enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": dev}, savedir=str(ROOT / "data/raw/models/ecapa"))

def noise_db(y, sr=16000):
    """Noise-floor proxy: energy of the quietest 10% of 20 ms frames vs the loudest 10% (dB). Higher = cleaner."""
    f = 320; n = len(y) // f
    if n < 10: return 0.0
    e = (y[:n * f].reshape(n, f) ** 2).mean(axis=1) + 1e-10
    e = np.sort(e); return float(10 * np.log10(e[-max(1, n // 10):].mean() / e[:max(1, n // 10)].mean()))

embs, meta, t0 = [], [], time.time()
done = set()
if meta_path.exists():
    meta = [json.loads(l) for l in open(meta_path)]; done = {m["id"] for m in meta}; embs = list(np.load(emb_path))
    print(f"resuming: {len(done)} done", flush=True)
mf = open(meta_path, "a", encoding="utf-8")
for i, r in enumerate(rows):
    if r["id"] in done: continue
    y, sr = sf.read(ROOT / r["audio"], dtype="float32", always_2d=True); y = y.mean(axis=1)
    assert sr == 16000, sr
    seg = y[: int(a.max_s * sr)]
    with torch.no_grad():
        e = enc.encode_batch(torch.from_numpy(seg)[None].to(dev)).squeeze().cpu().numpy()
    embs.append(e / (np.linalg.norm(e) + 1e-9))
    m = {"id": r["id"], "duration": r["duration"], "noise_db": round(noise_db(y), 1)}; meta.append(m); mf.write(json.dumps(m) + "\n")
    if (i + 1) % 500 == 0:
        mf.flush(); np.save(emb_path, np.stack(embs)); print(f"  {i + 1}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
mf.close(); E = np.stack(embs); np.save(emb_path, E)

from sklearn.cluster import AgglomerativeClustering
lab = AgglomerativeClustering(n_clusters=None, distance_threshold=a.threshold, metric="cosine", linkage="average").fit_predict(E)
with open(out.with_suffix(".jsonl"), "w", encoding="utf-8") as f:
    for m, c in zip(meta, lab): f.write(json.dumps({**m, "cluster": int(c)}) + "\n")
clusters = {}
for m, c in zip(meta, lab):
    d = clusters.setdefault(int(c), {"clips": 0, "hours": 0.0, "noise": []}); d["clips"] += 1; d["hours"] += m["duration"] / 3600; d["noise"].append(m["noise_db"])
summ = sorted(({"cluster": c, "clips": d["clips"], "hours": round(d["hours"], 2), "noise_db_median": round(float(np.median(d["noise"])), 1)} for c, d in clusters.items()), key=lambda x: -x["hours"])
json.dump({"n_clusters": len(summ), "threshold": a.threshold, "top": summ[:25]}, open(out.with_suffix(".summary.json"), "w"), indent=1)
print(json.dumps(summ[:12], indent=1), flush=True)
