"""Human-free TTS eval: synthesize the dev sentences with a VITS checkpoint, transcribe them with our ASR, report CER
(intelligibility) next to the Urdu-voice baseline. Audio is kept so you can listen.

  uv run python scripts/tts_eval.py --model qkhawaja/brahui-tts-vits [--baseline facebook/mms-tts-urd-script_arabic]
      [--dev data/tts/cv_a37a/dev.jsonl] [--asr runs/ctc1b_brh_s3/final] [--out evals/tts/<model name>]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch, jiwer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from brahui.text import normalize, for_scoring  # noqa: E402
from asr_eval import make_backend, device_for  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--baseline", default="facebook/mms-tts-urd-script_arabic")
ap.add_argument("--dev", default="data/tts/cv_a37a/dev.jsonl"); ap.add_argument("--asr", default="runs/ctc1b_brh_s3/final")
ap.add_argument("--out", default=None); ap.add_argument("--n", type=int, default=0)
a = ap.parse_args()
rows = [json.loads(l) for l in open(ROOT / a.dev, encoding="utf-8")]
rows = rows[:a.n] if a.n else rows
dev = device_for(); asr = make_backend(f"ctc:{ROOT / a.asr}", dev)
from transformers import AutoTokenizer, VitsModel

def run(repo, tag):
    tok = AutoTokenizer.from_pretrained(repo); m = VitsModel.from_pretrained(repo).to(dev).eval()
    out = ROOT / (a.out or f"evals/tts/{repo.split('/')[-1]}"); (out / "wavs").mkdir(parents=True, exist_ok=True)
    has_lateral = "ڷ" in tok.get_vocab()
    res = []; t0 = time.time()
    for r in rows:
        txt = r["text"] if has_lateral else r["text"].replace("ڷ", "ل")
        with torch.no_grad():
            wav = m(**tok(txt, return_tensors="pt").to(dev)).waveform[0].float().cpu().numpy()
        p = out / "wavs" / f"{r['id']}.wav"; sf.write(p, wav, m.config.sampling_rate, subtype="PCM_16")
        hyp = for_scoring(normalize(asr.transcribe(wav.astype(np.float32)))); ref = for_scoring(r["text"])
        res.append({"id": r["id"], "text": r["text"], "asr": hyp, "cer": jiwer.cer(ref or " ", hyp or " "), "sec": round(len(wav) / m.config.sampling_rate, 2),
                    "real_sec": r.get("duration")})
    cers = [x["cer"] for x in res]
    summ = {"model": repo, "n": len(res), "cer_mean": round(float(np.mean(cers)), 4), "cer_median": round(float(np.median(cers)), 4),
            "dur_ratio_vs_real": round(float(np.mean([x["sec"] / x["real_sec"] for x in res if x["real_sec"]])), 3), "seconds": round(time.time() - t0, 1)}
    with open(out / "dev.jsonl", "w", encoding="utf-8") as f:
        for x in res: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    json.dump(summ, open(out / "summary.json", "w"), indent=1); print(tag, json.dumps(summ), flush=True)
    return summ

run(a.model, "model   ")
if a.baseline:
    run(a.baseline, "baseline")
