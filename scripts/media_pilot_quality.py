"""Pilot: what do "editing models" buy per media source? For a sample of transcribed chunks from each pilot_* source,
score the audio raw, after music separation (Demucs vocals) and after speech enhancement (DeepFilterNet), with DNSMOS
(TTS-quality bar 3.0) and the s3 ASR's confidence.

  uv run --with speechmos python scripts/media_pilot_quality.py sample [--n 30]
  uv run --with demucs python scripts/media_pilot_quality.py demucs            # raw -> demucs (+ vocal energy share)
  uv run --no-project --python 3.11 --with deepfilternet --with torch==2.2.2 --with torchaudio==2.2.2 --with soundfile --with librosa \
      python scripts/media_pilot_quality.py dfn --src raw|demucs               # -> dfn | demucs_dfn
  uv run --with speechmos python scripts/media_pilot_quality.py score          # DNSMOS + ASR for every variant, report

Everything lands in evals/media_pilot/{<variant>/<source>/<id>_<seg>.wav, sample.jsonl, scores.jsonl, report.json}.
The stored media audio is 16 kHz mono, so Demucs (44.1 kHz) and DeepFilterNet (48 kHz) run on upsampled audio.
"""
from __future__ import annotations
import argparse, collections, glob, json, random, sys
from pathlib import Path
import numpy as np, soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "evals/media_pilot"
ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["sample", "demucs", "dfn", "score"]); ap.add_argument("--n", type=int, default=30); ap.add_argument("--src", default="raw")
a = ap.parse_args()

def sample():
    return [json.loads(l) for l in open(OUT / "sample.jsonl", encoding="utf-8")]

def rs(y, sr0, sr1):
    import librosa
    return y if sr0 == sr1 else librosa.resample(y, orig_sr=sr0, target_sr=sr1)

if a.stage == "sample":
    by = collections.defaultdict(list)
    for f in glob.glob(str(ROOT / "data/media/transcripts/pilot_*/*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if 5 <= r["end"] - r["start"] <= 20: by[r["source"]].append(r)
    OUT.mkdir(parents=True, exist_ok=True); rows = []
    for s, rr in sorted(by.items()):
        random.Random(0).shuffle(rr)
        for r in rr[:a.n]:
            y, sr = sf.read(ROOT / f"data/media/audio/{s}/{r['id']}.wav", start=int(r["start"] * 16000), stop=int(r["end"] * 16000), dtype="float32")
            p = OUT / "raw" / s / f"{r['id']}_{r['seg']}.wav"; p.parent.mkdir(parents=True, exist_ok=True); sf.write(p, y, sr)
            rows.append({"key": f"{s}/{r['id']}_{r['seg']}", "source": s, "dur": round(r["end"] - r["start"], 2), "text": r["text"], "conf": r["conf"]})
    with open(OUT / "sample.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "chunks,", round(sum(r["dur"] for r in rows) / 60, 1), "min")

elif a.stage == "demucs":
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = get_model("htdemucs").to(dev).eval(); vi = m.sources.index("vocals"); share = {}
    for i, r in enumerate(sample()):
        y, sr = sf.read(OUT / "raw" / f"{r['key']}.wav", dtype="float32")
        x = torch.tensor(rs(y, sr, m.samplerate))[None, None].repeat(1, 2, 1)
        with torch.no_grad(): st = apply_model(m, x.to(dev), device=dev, split=True, overlap=0.1)[0].cpu()   # (sources, 2, T)
        e = (st ** 2).sum(dim=(1, 2)); share[r["key"]] = round(float(e[vi] / (e.sum() + 1e-9)), 4)
        v = rs(st[vi].mean(0).numpy(), m.samplerate, 16000)
        p = OUT / "demucs" / f"{r['key']}.wav"; p.parent.mkdir(parents=True, exist_ok=True); sf.write(p, v, 16000)
        if (i + 1) % 50 == 0: print(i + 1, flush=True)
    json.dump(share, open(OUT / "vocal_share.json", "w"))

elif a.stage == "dfn":
    import torch
    from df.enhance import enhance, init_df
    model, state, _ = init_df()
    dst = "dfn" if a.src == "raw" else f"{a.src}_dfn"
    for i, r in enumerate(sample()):
        y, sr = sf.read(OUT / a.src / f"{r['key']}.wav", dtype="float32")
        o = enhance(model, state, torch.tensor(rs(y, sr, state.sr()))[None]).squeeze(0).numpy()
        p = OUT / dst / f"{r['key']}.wav"; p.parent.mkdir(parents=True, exist_ok=True); sf.write(p, rs(o, state.sr(), 16000), 16000)
        if (i + 1) % 50 == 0: print(i + 1, flush=True)

else:
    import torch, jiwer
    from speechmos import dnsmos
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
    from transcribe_media import CTC  # noqa: E402
    asr = CTC(str(ROOT / "runs/ctc1b_brh_s3/final"), torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
    variants = [v for v in ("raw", "demucs", "dfn", "demucs_dfn") if (OUT / v).exists()]
    share = json.load(open(OUT / "vocal_share.json")) if (OUT / "vocal_share.json").exists() else {}
    done = {}
    if (OUT / "scores.jsonl").exists():
        for l in open(OUT / "scores.jsonl", encoding="utf-8"):
            x = json.loads(l); done[(x["key"], x["variant"])] = x
    with open(OUT / "scores.jsonl", "a", encoding="utf-8") as f:
        for i, r in enumerate(sample()):
            raw_text = None
            for v in variants:
                if (r["key"], v) in done:
                    if v == "raw": raw_text = done[(r["key"], v)]["text"]
                    continue
                p = OUT / v / f"{r['key']}.wav"
                if not p.exists(): continue
                y, _ = sf.read(p, dtype="float32")
                c = y[len(y) // 2 - 80000: len(y) // 2 + 80000] if len(y) > 160000 else y
                mos = dnsmos.run(c / (np.max(np.abs(c)) + 1e-9) * 0.9, sr=16000)
                t = asr.transcribe([y])[0]
                if v == "raw": raw_text = t["text"]
                x = {"key": r["key"], "source": r["source"], "variant": v, "dur": r["dur"], "ovrl": round(float(mos["ovrl_mos"]), 3), "sig": round(float(mos["sig_mos"]), 3),
                     "bak": round(float(mos["bak_mos"]), 3), "conf": round(t["conf"], 4), "text": t["text"],
                     "cer_vs_raw": round(jiwer.cer(raw_text or " ", t["text"] or " "), 4) if raw_text is not None else None}
                done[(r["key"], v)] = x; f.write(json.dumps(x, ensure_ascii=False) + "\n"); f.flush()
            if (i + 1) % 100 == 0: print(i + 1, flush=True)
    rep = {}
    print(f"{'source':30s} {'vocal':>5} | " + " | ".join(f"{v:>10s} ovrl  >=3 conf >=.9 usable" for v in variants))
    for s in sorted({k[0].split('/')[0] for k in done}):
        vs = [share[r["key"]] for r in sample() if r["source"] == s and r["key"] in share]; line = f"{s[6:]:30s} {np.median(vs) if vs else float('nan'):5.2f} | "; rep[s] = {"vocal_share_median": float(np.median(vs)) if vs else None}
        for v in variants:
            xs = [x for (k, vv), x in done.items() if vv == v and x["source"] == s]
            if not xs: continue
            d = np.array([x["dur"] for x in xs]); w = lambda m: float((d * m).sum() / d.sum())  # noqa: E731
            o = np.array([x["ovrl"] for x in xs]); c = np.array([x["conf"] for x in xs])
            rep[s][v] = {"n": len(xs), "ovrl_median": float(np.median(o)), "ovrl_ge3": w(o >= 3.0), "conf_mean": float(c.mean()), "conf_ge9": w(c >= 0.9), "usable": w((o >= 3.0) & (c >= 0.9)),
                         "usable_loose": w((o >= 2.8) & (c >= 0.85)), "cer_vs_raw": float(np.mean([x["cer_vs_raw"] for x in xs if x["cer_vs_raw"] is not None] or [0]))}
            q = rep[s][v]; line += f"{'':10s} {q['ovrl_median']:4.2f} {q['ovrl_ge3']:4.2f} {q['conf_mean']:4.2f} {q['conf_ge9']:4.2f} {q['usable']:6.2f} | "
        print(line)
    json.dump(rep, open(OUT / "report.json", "w"), indent=1)
