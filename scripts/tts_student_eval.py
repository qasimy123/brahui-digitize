"""Score a student VITS voice: speak the dev sentences from their vowelled text, read them back with the s3 ASR.
  uv run python scripts/tts_student_eval.py --model runs/tts_student/seg1 [--dev data/tts/student/dev.jsonl] [--n 40] [--out evals/tts/student/seg1]
Prints {"cer_mean": ...}; writes wavs + scores.jsonl under --out. Same metric as scripts/tts_score_wavs.py."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, soundfile as sf, jiwer
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from brahui.text import normalize, for_scoring  # noqa: E402
from asr_eval import make_backend, device_for  # noqa: E402
from transformers import VitsModel, VitsTokenizer  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--dev", default="data/tts/student/dev.jsonl"); ap.add_argument("--n", type=int, default=40); ap.add_argument("--out", default=""); ap.add_argument("--asr", default="runs/ctc1b_brh_s3/final"); a = ap.parse_args()
rows = [json.loads(l) for l in open(ROOT / a.dev, encoding="utf-8")][:a.n]; dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = VitsTokenizer.from_pretrained(ROOT / a.model); m = VitsModel.from_pretrained(ROOT / a.model).to(dev).eval(); asr = make_backend(f"ctc:{ROOT / a.asr}", device_for())
out = ROOT / (a.out or f"evals/tts/student/{Path(a.model).name}"); (out / "wavs").mkdir(parents=True, exist_ok=True); res = []
for r in rows:
    torch.manual_seed(0)
    with torch.no_grad(): y = m(**tok(r["vowelled"], return_tensors="pt").to(dev)).waveform[0].float().cpu().numpy()
    sf.write(out / "wavs" / f"{r['id']}.wav", y, m.config.sampling_rate); hyp = for_scoring(normalize(asr.transcribe(y))); ref = for_scoring(normalize(r["text"]))
    res.append({"id": r["id"], "cer": round(jiwer.cer(ref or " ", hyp or " "), 4), "sec": round(len(y) / m.config.sampling_rate, 2), "teacher_sec": r.get("duration"), "asr": hyp})
with open(out / "scores.jsonl", "w", encoding="utf-8") as f:
    for x in res: f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(json.dumps({"model": a.model, "n": len(res), "cer_mean": round(float(np.mean([x["cer"] for x in res])), 4), "cer_median": round(float(np.median([x["cer"] for x in res])), 4),
                  "dur_ratio_vs_teacher": round(float(np.mean([x["sec"] / x["teacher_sec"] for x in res if x["teacher_sec"]])), 3)}))
