"""Enhance the confident chunks of the bulk-scraped media from their full-rate originals and score audio quality.

  uv run --no-project --python 3.11 --with deepfilternet --with torch==2.2.2 --with torchaudio==2.2.2 --with soundfile --with librosa \
      --with "numpy<2" --with speechmos --with onnxruntime python scripts/enhance_media_bulk.py [--prefix bulk_] [--min-conf 0.8] [--workers 4] [--limit 0]

For every transcribed chunk (data/media/transcripts/<source>/<id>.jsonl) with ASR confidence >= --min-conf and 2-20 s long:
cut it from the kept original (data/media/audio/<source>/<id>.<webm|m4a|...>, decoded at 48 kHz), run DeepFilterNet,
write data/tts/bulk_enh/<source>/<id>/<seg>.flac (24 kHz) and one row per chunk in data/tts/bulk_enh/<source>/<id>.jsonl with
DNSMOS (10 s centre crop, same as scripts/score_tts_pool.py) before (16 kHz wav) and after. Resumable per file.
Music separation is skipped: the pilot (evals/media_pilot) showed it adds almost nothing on these channels.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, soundfile as sf

ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "data/tts/bulk_enh"
ap = argparse.ArgumentParser(); ap.add_argument("--prefix", default="bulk_"); ap.add_argument("--min-conf", type=float, default=0.8)
ap.add_argument("--workers", type=int, default=4); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--reverse", action="store_true")
a = ap.parse_args()
_m = None

def mos(y16):
    import onnxruntime as ort
    if not getattr(ort, "_one_thread", False):   # by default every session starts one thread per host core: 30 workers on a 256-core host thrash
        orig = ort.InferenceSession
        def one(*x, **k):
            so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
            k.setdefault("sess_options", so); return orig(*x, **k)
        ort.InferenceSession = one; ort._one_thread = True
    from speechmos import dnsmos
    c = y16[len(y16) // 2 - 80000: len(y16) // 2 + 80000] if len(y16) > 160000 else y16
    m = dnsmos.run(c / (np.max(np.abs(c)) + 1e-9) * 0.9, sr=16000)
    return round(float(m["ovrl_mos"]), 3), round(float(m["sig_mos"]), 3), round(float(m["bak_mos"]), 3)

def work(tr: str):
    global _m
    import torch, librosa
    tr = Path(tr); src, vid = tr.parent.name, tr.stem; dst = OUT / src / f"{vid}.jsonl"
    if dst.exists(): return src, 0, 0.0
    segs = [r for r in map(json.loads, open(tr, encoding="utf-8")) if r["conf"] >= a.min_conf and 2 <= r["end"] - r["start"] <= 20 and r["text"].strip()]
    orig = [p for p in (ROOT / "data/media/audio" / src).glob(f"{vid}.*") if p.suffix not in (".wav", ".json")]
    if not orig: return src, 0, 0.0   # original not on this machine: leave the file for the machine that has it
    if not segs:
        dst.parent.mkdir(parents=True, exist_ok=True); dst.write_text(""); return src, 0, 0.0
    if _m is None:
        torch.set_num_threads(a.threads)
        from df.enhance import init_df
        _m = init_df(log_level="ERROR")[:2]
    from df.enhance import enhance
    model, state = _m
    # decode to a temp file and memory-map it: holding whole multi-hour files in RAM in 30 workers got workers OOM-killed and the pool hung (2026-11-11)
    tf = Path(tempfile.gettempdir()) / f"enh_{os.getpid()}.f32"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(orig[0]), "-ac", "1", "-ar", "48000", "-f", "f32le", str(tf)], capture_output=True)
    full = np.memmap(tf, dtype=np.float32, mode="r") if tf.exists() and tf.stat().st_size else np.zeros(0, np.float32); wp = ROOT / "data/media/audio" / src / f"{vid}.wav"; w16 = sf.SoundFile(wp) if wp.exists() else None; rows = []  # on a pod only the originals are present
    for r in segs:
        y = full[int(r["start"] * 48000): int(r["end"] * 48000)]
        if len(y) < 48000: continue
        if w16 is not None: w16.seek(int(r["start"] * 16000)); y16 = w16.read(int((r["end"] - r["start"]) * 16000), dtype="float32")
        else: y16 = librosa.resample(y, orig_sr=48000, target_sr=16000)
        e = enhance(model, state, torch.from_numpy(y.copy())[None]).squeeze(0).numpy()
        p = OUT / src / vid / f"{r['seg']}.flac"; p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, librosa.resample(e, orig_sr=48000, target_sr=24000), 24000)
        o0, s0, b0 = mos(y16); o1, s1, b1 = mos(librosa.resample(e, orig_sr=48000, target_sr=16000))
        rows.append({"id": f"{src}/{vid}/{r['seg']}", "source": src, "video": vid, "seg": r["seg"], "start": r["start"], "end": r["end"], "duration": round(r["end"] - r["start"], 2),
                     "text": r["text"], "conf": round(r["conf"], 4), "audio": str(p.relative_to(ROOT)), "ovrl_raw": o0, "ovrl": o1, "sig": s1, "bak": b1})
    del full; tf.unlink(missing_ok=True)
    tmp = dst.with_suffix(".tmp"); tmp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8"); tmp.rename(dst)
    return src, len(rows), sum(x["duration"] for x in rows)

if __name__ == "__main__":
    files = sorted(str(p) for p in (ROOT / "data/media/transcripts").glob(f"{a.prefix}*/*.jsonl")); files = files[::-1] if a.reverse else files; files = files[:a.limit or None]
    t0 = time.time(); n = 0; h = 0.0
    with ProcessPoolExecutor(a.workers) as pool:   # a killed worker raises BrokenProcessPool here (multiprocessing.Pool hangs forever); rerun to resume
        for i, fut in enumerate(as_completed([pool.submit(work, f) for f in files]), 1):
            src, k, d = fut.result()
            n += k; h += d / 3600
            if i % 10 == 0 or i == len(files): print(f"{i}/{len(files)} files, {n} chunks, {h:.2f} h enhanced, {(time.time() - t0) / 60:.1f} min", flush=True)
