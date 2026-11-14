"""Human-free TTS score for a directory of synthesized wavs: ASR-CER against the text each wav should say.

  uv run python scripts/tts_score_wavs.py --wavs evals/tts/f5_multi_arabic/wavs --dev data/tts/multi/dev.jsonl [--asr runs/ctc1b_brh_s3/final] [--tag f5]

Wavs are <id>.wav for ids in --dev (jsonl with id, text in Arabic script, duration). Point --wavs at the exported real dev
clips (data/tts/f5/multi_arabic_dev/wavs) to get the same ASR's CER on human speech, the number to compare against.
Writes <wavs>/../scores[_<tag>].jsonl and summary[_<tag>].json. Same metric as scripts/tts_eval.py (VITS).
"""
from __future__ import annotations
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") == "": os.environ.setdefault("OMP_NUM_THREADS", "5")   # CPU eval beside training: all cores starved the dataloaders (5.3 -> 1.0 update/s)
import argparse, json, sys
from pathlib import Path
import numpy as np, soundfile as sf, librosa, jiwer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from brahui.text import normalize, for_scoring  # noqa: E402
from asr_eval import make_backend, device_for  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--wavs", required=True); ap.add_argument("--dev", default="data/tts/multi/dev.jsonl"); ap.add_argument("--asr", default="runs/ctc1b_brh_s3/final"); ap.add_argument("--tag", default="")
a = ap.parse_args()
wavs = Path(a.wavs) if Path(a.wavs).is_absolute() else ROOT / a.wavs
asr = make_backend(f"ctc:{ROOT / a.asr}", device_for()); res = []
for line in open(ROOT / a.dev, encoding="utf-8"):
    r = json.loads(line); p = wavs / f"{r['id']}.wav"
    if not p.exists(): continue
    y, sr = sf.read(p, dtype="float32", always_2d=True); y = y.mean(axis=1)
    if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    hyp = for_scoring(normalize(asr.transcribe(y))); ref = for_scoring(normalize(r["text"]))
    res.append({"id": r["id"], "source": r.get("source"), "text": r["text"], "asr": hyp, "cer": round(jiwer.cer(ref or " ", hyp or " "), 4),
                "sec": round(len(y) / 16000, 2), "real_sec": r.get("duration")})
cers = [x["cer"] for x in res]; sfx = f"_{a.tag}" if a.tag else ""
summ = {"wavs": str(a.wavs), "n": len(res), "cer_mean": round(float(np.mean(cers)), 4), "cer_median": round(float(np.median(cers)), 4),
        "dur_ratio_vs_real": round(float(np.mean([x["sec"] / x["real_sec"] for x in res if x["real_sec"]])), 3)}
with open(wavs.parent / f"scores{sfx}.jsonl", "w", encoding="utf-8") as f:
    for x in res: f.write(json.dumps(x, ensure_ascii=False) + "\n")
json.dump(summ, open(wavs.parent / f"summary{sfx}.json", "w"), indent=1); print(json.dumps(summ))
