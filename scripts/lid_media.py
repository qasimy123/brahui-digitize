"""Urdu screen for the enhanced bulk media (data/tts/bulk_enh). The Brahui ASR transcribes Urdu confidently, so confidence
does not catch it. Two signals: Urdu vs Brahui function words in the transcript (free), and spoken-language ID
(facebook/mms-lid-1024) on the raw 16 kHz audio (8 s centre crop). MMS-LID rejects Urdu well but labels real Brahui as
Balochi/Pashto/Sindhi and sometimes Urdu (listening check 2026-11-11: clips 13, 22 are Brahui at p_urdu 0.78/0.94), so a
clip is dropped only when both signals say Urdu.

  uv run --with soundfile --with librosa python scripts/lid_media.py     # -> data/tts/bulk_enh/lid.jsonl (resumable)
LID runs on clips whose transcript leans Urdu (urdu markers >= brahui markers) plus a 400-clip control of clearly Brahui text.
"""
import glob, json, random, time
from pathlib import Path
import numpy as np, soundfile as sf, torch
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor

ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "data/tts/bulk_enh/lid.jsonl"
UR = set("ہے ہیں کا کی کے میں اور نے کو تھا تھی تھے ہو گا گی یہ وہ بھی نہیں پر ہوں کر رہے رہا رہی لیے ایک".split())
BR = set("نا ءِ ءُ ءَ اٹ کن اس ہم ارے اف اے دا ہمو انا تینا کنا نن ای نی او تو مس کرے کرینے ءٹ آن آ ئنا اسٹ اسہ پارے اُس کین تون".split())
def markers(t):
    w = t.split(); return sum(x in UR for x in w), sum(x in BR for x in w), len(w)

if __name__ == "__main__":
    rows = [json.loads(l) for f in sorted(glob.glob(str(ROOT / "data/tts/bulk_enh/*/*.jsonl"))) for l in open(f, encoding="utf-8")]
    cand, ctrl = [], []
    for r in rows:
        u, b, k = markers(r["text"]); r["ur"], r["br"] = u, b
        (cand if u >= b else ctrl if (u == 0 and b >= 3) else []).append(r) if (u >= b or (u == 0 and b >= 3)) else None
    random.Random(0).shuffle(ctrl); todo = cand + ctrl[:400]
    done = {json.loads(l)["id"] for l in open(OUT)} if OUT.exists() else set()
    todo = sorted((r for r in todo if r["id"] not in done), key=lambda r: r["duration"])
    print(len(cand), "candidates", round(sum(r["duration"] for r in cand) / 3600, 1), "h;", len(todo), "to do", flush=True)
    M = "facebook/mms-lid-1024"; dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(M); m = Wav2Vec2ForSequenceClassification.from_pretrained(M, torch_dtype=torch.float32 if dev == "cpu" else torch.float16).to(dev).eval()
    L = m.config.id2label; ix = {v: k for k, v in L.items()}; t0 = time.time(); B = 16
    with open(OUT, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), B):
            bt = todo[i:i + B]; ys = []
            for r in bt:
                c = (r["start"] + r["end"]) / 2; s = max(r["start"], c - 4); e = min(r["end"], c + 4)
                y, _ = sf.read(ROOT / f"data/media/audio/{r['source']}/{r['video']}.wav", start=int(s * 16000), stop=int(e * 16000), dtype="float32"); ys.append(y)
            n = min(len(y) for y in ys) // 8000 * 8000; ys = [y[(len(y) - n) // 2:(len(y) - n) // 2 + n] for y in ys]   # equal length, 0.5 s steps: no padding, few distinct shapes (MPS stalls on a new shape every batch)
            x = fe(ys, sampling_rate=16000, return_tensors="pt").input_values.to(dev, m.dtype)
            with torch.no_grad(): pr = m(x).logits.float().softmax(-1).cpu()
            for r, p in zip(bt, pr):
                f.write(json.dumps({"id": r["id"], "duration": r["duration"], "ur": r["ur"], "br": r["br"], "top": L[int(p.argmax())], "p_top": round(float(p.max()), 3),
                                    "p_urdu": round(float(p[ix["urd"]] + p[ix["hin"]]), 3), "p_brh": round(float(p[ix["brh"]]), 3), "control": r["br"] > r["ur"]}) + "\n")
            f.flush()
            if (i // B) % 20 == 0: print(i + len(bt), "/", len(todo), round((time.time() - t0) / 60, 1), "min", flush=True)
    print("DONE", flush=True)
