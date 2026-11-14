"""Export a TTS jsonl (data/tts/multi/train.jsonl, data/tts/cv_a37a/train.jsonl, ...) as an F5-TTS csv_wavs dataset.

  uv run python scripts/tts_f5_export.py --in data/tts/multi/train.jsonl --out data/tts/f5/multi_arabic [--text arabic|roman] [--limit N]

Writes <out>/wavs/<id>.wav (24 kHz mono, the row's start/end slice, silence-trimmed at 35 dB with 80 ms pads,
RMS -20 dBFS) and <out>/metadata.csv ("audio_file|text"). --text roman feeds brahui.text.romanize output, which reuses
F5's pretrained Latin-letter embeddings; arabic keeps the Perso-Arabic script (new embedding rows).
Next: tools/f5/prep_finetune.py.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, soundfile as sf, librosa

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from brahui.text import normalize, romanize  # noqa: E402

# F5 reads metadata.csv with csv.reader: a double quote opening a field would swallow the following lines into one text
PUNCT = str.maketrans({"۔": ".", "،": ",", "؟": "?", "؛": ",", "٪": "%", '"': None, "٭": None, "“": None, "”": None})
ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--text", default="arabic", choices=["arabic", "roman", "roman_speak"]); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
rows = [json.loads(l) for l in open(ROOT / a.inp, encoding="utf-8")][:a.limit or None]
out = ROOT / a.out; (out / "wavs").mkdir(parents=True, exist_ok=True)
n = 0; secs = 0.0
with open(out / "metadata.csv", "w", encoding="utf-8") as f:
    f.write("audio_file|text\n")
    for r in rows:
        wav = out / "wavs" / f"{r['id']}.wav"
        if not wav.exists():
            info = sf.info(ROOT / r["audio"]); sr = info.samplerate
            st = int(r.get("start", 0) * sr); fr = int(r["end"] * sr) - st if "end" in r else -1
            y, sr = sf.read(ROOT / r["audio"], start=st, frames=fr, dtype="float32", always_2d=True); y = y.mean(axis=1)
            if sr != 24000: y = librosa.resample(y, orig_sr=sr, target_sr=24000)
            y, _ = librosa.effects.trim(y, top_db=35, frame_length=1536, hop_length=384)
            pad = np.zeros(int(0.08 * 24000), dtype=np.float32); y = np.concatenate([pad, y, pad])
            y = y * (10 ** (-20 / 20) / (float(np.sqrt(np.mean(y ** 2))) + 1e-9)); peak = float(np.max(np.abs(y)))
            if peak > 0.99: y = y * (0.99 / peak)
            sf.write(wav, y, 24000, subtype="PCM_16")
        text = normalize(r["text"]).replace("|", " ")
        if a.text == "roman": text = romanize(text)
        elif a.text == "roman_speak": text = romanize(text, speak=True)   # fixes glued punctuation and the s+h / k+h digraph collisions
        text = text.translate(PUNCT)  # punctuation F5 already knows, in either script
        f.write(f"wavs/{wav.name}|{text}\n"); n += 1; secs += sf.info(wav).duration
print(f"{n} clips, {secs / 3600:.2f} h -> {out}")
