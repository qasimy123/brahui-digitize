"""Score every BrahuiSpeech clip for TTS use: DNSMOS quality, then ASR agreement with its label (stage-3 CTC CER).

  uv run --with speechmos python scripts/score_tts_pool.py mos [--workers 6]        # CPU, ~1.1 s/clip/worker
  uv run --with speechmos python scripts/score_tts_pool.py asr [--min-ovrl 2.5]     # MPS, ~1.5 s/clip, only clips above the bar
  ... --manifest data/tts/pool_extra.jsonl --name extra    # other sources (scripts/build_tts_pool_extra.py); rows may carry start/end

For rows whose text is itself an ASR pseudo-label (pseudo=true) the CER is not evidence; use their `conf` instead.

Both stages are resumable. mos -> data/tts/bs_mos.jsonl {id, duration, ovrl, sig, bak, p808};
asr -> data/tts/bs_scores.jsonl (the mos row + cer, hyp). The ASR was trained on these clips, so CER is optimistic:
it catches wrong/garbled labels, it does not certify clean ones. Select with scripts/build_tts_multispeaker.py.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, librosa

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

def load(r):
    """Audio for a manifest row (optionally a start/end slice of a longer file), mono 16 kHz."""
    info = sf.info(ROOT / r["audio"]); sr = info.samplerate
    st = int(r.get("start", 0) * sr); n = int(r["end"] * sr) - st if "end" in r else -1
    y, sr = sf.read(ROOT / r["audio"], start=st, frames=n, dtype="float32", always_2d=True); y = y.mean(axis=1)
    return librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y

def mos_one(r):
    try:
        from speechmos import dnsmos
        y = load(r); c = y
        if len(c) > 160000:  # DNSMOS slides a 9 s window every 1 s; one centre window is enough to rank clips and ~5x cheaper
            o = (len(c) - 160000) // 2; c = c[o:o + 160000]
        m = dnsmos.run(c / (np.max(np.abs(c)) + 1e-9) * 0.9, sr=16000)
        return {"id": r["id"], "duration": round(len(y) / 16000, 2), "ovrl": round(float(m["ovrl_mos"]), 3), "sig": round(float(m["sig_mos"]), 3),
                "bak": round(float(m["bak_mos"]), 3), "p808": round(float(m["p808_mos"]), 3)}
    except Exception as e:  # a bad file is itself a reason to drop the clip
        return {"id": r["id"], "error": repr(e)[:200]}

def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["mos", "asr"])
    ap.add_argument("--manifest", default="data/manifests/brahuispeech_train_norm.jsonl"); ap.add_argument("--asr", default="runs/ctc1b_brh_s3/final")
    ap.add_argument("--name", default="bs", help="output prefix: data/tts/<name>_mos.jsonl, <name>_scores.jsonl"); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--min-ovrl", type=float, default=2.5); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = read(ROOT / a.manifest); mos_path = ROOT / f"data/tts/{a.name}_mos.jsonl"; out = ROOT / f"data/tts/{a.name}_scores.jsonl"
    t0 = time.time()
    if a.stage == "mos":
        done = {r["id"] for r in read(mos_path)}; todo = [r for r in rows if r["id"] not in done][:a.limit or None]
        print(f"{len(done)} done, {len(todo)} to score", flush=True)
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool, open(mos_path, "a", encoding="utf-8") as f:
            for i, rec in enumerate(pool.imap_unordered(mos_one, todo, chunksize=4)):
                f.write(json.dumps(rec) + "\n"); f.flush()
                if (i + 1) % 500 == 0 or i + 1 == len(todo):
                    el = time.time() - t0; print(f"  {i + 1}/{len(todo)}  {el:.0f}s  eta {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min", flush=True)
    else:
        import jiwer
        from brahui.text import normalize, for_scoring
        from asr_eval import make_backend, device_for
        mos = {r["id"]: r for r in read(mos_path) if "error" not in r and r["ovrl"] >= a.min_ovrl}
        done = {r["id"] for r in read(out)}; todo = [r for r in rows if r["id"] in mos and r["id"] not in done][:a.limit or None]
        print(f"{len(mos)} clips with ovrl >= {a.min_ovrl}; {len(done)} done, {len(todo)} to transcribe", flush=True)
        asr = make_backend(f"ctc:{ROOT / a.asr}", device_for())
        with open(out, "a", encoding="utf-8") as f:
            for i, r in enumerate(todo):
                if r.get("pseudo"):  # label is this ASR's own output: agreement is meaningless, keep its confidence instead
                    f.write(json.dumps({**mos[r["id"]], "cer": None, "conf": r.get("conf")}, ensure_ascii=False) + "\n"); continue
                try: hyp_raw = asr.transcribe(load(r))
                except RuntimeError as e:  # e.g. a clip shorter than the encoder's receptive field: unusable either way
                    f.write(json.dumps({**mos[r["id"]], "cer": 9.0, "hyp": "", "error": repr(e)[:120]}, ensure_ascii=False) + "\n"); continue
                ref = for_scoring(normalize(r.get("text_norm") or r["text"])); hyp = for_scoring(normalize(hyp_raw))
                f.write(json.dumps({**mos[r["id"]], "cer": round(jiwer.cer(ref or " ", hyp or " "), 4), "hyp": hyp_raw}, ensure_ascii=False) + "\n"); f.flush()
                if (i + 1) % 200 == 0 or i + 1 == len(todo):
                    el = time.time() - t0; print(f"  {i + 1}/{len(todo)}  {el:.0f}s  eta {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min", flush=True)
