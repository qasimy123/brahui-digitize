"""Check every teacher clip of the small-model data set: transcribe it with the s3 ASR and compare with the text it should say.
  uv run python scripts/tts_distill_filter.py [--dir data/tts/distill/synth]  ->  <dir>/clips.jsonl  {id, text, vowelled, sec, cer, wps, asr}
Resumable. Selection (which clips to train on) is done later from this file."""
import argparse, json, sys
from pathlib import Path
import soundfile as sf, librosa, jiwer
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from brahui.text import normalize, for_scoring  # noqa: E402
from asr_eval import make_backend, device_for  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="data/tts/distill/synth"); ap.add_argument("--script", default="data/tts/distill/script.jsonl"); ap.add_argument("--asr", default="runs/ctc1b_brh_s3/final"); a = ap.parse_args()
D = ROOT / a.dir; out = D / "clips.jsonl"; have = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if out.exists() else set()
rows = [r for r in map(json.loads, open(ROOT / a.script, encoding="utf-8")) if r["id"] not in have and (D / "wavs" / f"{r['id']}.wav").exists()]
asr = make_backend(f"ctc:{ROOT / a.asr}", device_for()); print(len(rows), "clips to check,", len(have), "done", flush=True)
with open(out, "a", encoding="utf-8") as f:
    for i, r in enumerate(rows):
        y, sr = sf.read(D / "wavs" / f"{r['id']}.wav", dtype="float32", always_2d=True); y = y.mean(axis=1); sec = len(y) / sr
        if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        hyp = for_scoring(normalize(asr.transcribe(y))); ref = for_scoring(normalize(r["text"]))
        f.write(json.dumps({"id": r["id"], "source": r["source"], "text": r["text"], "vowelled": r["vowelled"], "sec": round(sec, 2), "cer": round(jiwer.cer(ref or " ", hyp or " "), 4), "wps": round(r["n_words"] / sec, 2), "asr": hyp}, ensure_ascii=False) + "\n")
        if (i + 1) % 500 == 0: f.flush(); print(i + 1, flush=True)
import numpy as np  # noqa: E402
c = [json.loads(l) for l in open(out, encoding="utf-8")]; cer = np.array([x["cer"] for x in c]); sec = np.array([x["sec"] for x in c])
print(json.dumps({"clips": len(c), "hours": round(float(sec.sum()) / 3600, 2), "cer_mean": round(float(cer.mean()), 4), "cer_median": round(float(np.median(cer)), 4),
                  **{f"share_cer_le_{t}": round(float((cer <= t).mean()), 3) for t in (0.05, 0.1, 0.15, 0.2)}, "hours_cer_le_0.1": round(float(sec[cer <= 0.1].sum()) / 3600, 2)}))
print("FILTER DONE")
