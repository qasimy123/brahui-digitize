"""Build a single-speaker Brahui TTS set from Common Voice, filtered by our own ASR (no human listening).

  uv run python scripts/build_tts_set.py [--speaker a37a913d512beb9a] [--out data/tts/cv_a37a] [--max-cer 0.15]
      [--asr runs/ctc1b_brh_s3/final] [--push qkhawaja/brahui-tts-cv]

Steps: pick the speaker's clips from data/manifests/cv_train.jsonl; drop clips with Latin letters/digits or outside
1-12 s; resample to 16 kHz mono, trim leading/trailing silence, RMS-normalize to -20 dBFS; transcribe with the ASR and
drop clips whose CER against the label exceeds --max-cer (misread sentences, noisy takes); hold out --dev sentences.
Writes <out>/wavs/*.wav, <out>/{train,dev}.jsonl and <out>/report.json; with --push, uploads a private HF dataset
(columns audio, text, speaker_id; splits train/dev) for finetune-hf-vits.
"""
from __future__ import annotations
import argparse, json, random, re, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, librosa

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from brahui.text import normalize, for_scoring  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--speaker", default="a37a913d512beb9a"); ap.add_argument("--out", default="data/tts/cv_a37a")
ap.add_argument("--manifest", default="data/manifests/cv_train.jsonl"); ap.add_argument("--asr", default="runs/ctc1b_brh_s3/final")
ap.add_argument("--max-cer", type=float, default=0.15); ap.add_argument("--min-s", type=float, default=1.0); ap.add_argument("--max-s", type=float, default=12.0)
ap.add_argument("--dev", type=int, default=40); ap.add_argument("--push", default=None); ap.add_argument("--target-dbfs", type=float, default=-20.0)
a = ap.parse_args()
out = ROOT / a.out; (out / "wavs").mkdir(parents=True, exist_ok=True)

rows = [json.loads(l) for l in open(ROOT / a.manifest, encoding="utf-8")]
rows = [r for r in rows if r.get("speaker") == a.speaker]
print(f"{len(rows)} clips for speaker {a.speaker}", flush=True)
rep = {"speaker": a.speaker, "input": len(rows), "dropped": {}}
latin = re.compile(r"[A-Za-z0-9]")
keep = []
for r in rows:
    t = normalize(r["text"])
    if latin.search(t):
        rep["dropped"]["latin_or_digits"] = rep["dropped"].get("latin_or_digits", 0) + 1; continue
    r["tts_text"] = t; keep.append(r)
rows = keep

# audio: load, resample, trim, normalize
import jiwer
from asr_eval import make_backend, device_for  # noqa: E402
asr = make_backend(f"ctc:{ROOT / a.asr}", device_for())
t0 = time.time(); kept = []; cers = []
for i, r in enumerate(rows):
    y, sr = sf.read(ROOT / r["audio"], dtype="float32", always_2d=True); y = y.mean(axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    y, _ = librosa.effects.trim(y, top_db=35, frame_length=1024, hop_length=256)
    pad = np.zeros(int(0.08 * 16000), dtype=np.float32); y = np.concatenate([pad, y, pad])
    dur = len(y) / 16000
    if not (a.min_s <= dur <= a.max_s):
        rep["dropped"]["duration"] = rep["dropped"].get("duration", 0) + 1; continue
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-9); y = y * (10 ** (a.target_dbfs / 20) / rms)
    peak = float(np.max(np.abs(y)))
    if peak > 0.99:
        y = y * (0.99 / peak)
    hyp = for_scoring(normalize(asr.transcribe(y))); ref = for_scoring(r["tts_text"])
    cer = jiwer.cer(ref or " ", hyp or " "); cers.append(cer)
    if cer > a.max_cer:
        rep["dropped"]["asr_cer"] = rep["dropped"].get("asr_cer", 0) + 1; continue
    wav = out / "wavs" / f"{r['id']}.wav"; sf.write(wav, y, 16000, subtype="PCM_16")
    kept.append({"id": r["id"], "audio": str(wav.relative_to(ROOT)), "text": r["tts_text"], "duration": round(dur, 2), "asr_cer": round(cer, 3), "speaker_id": 0})
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(rows)} kept {len(kept)}  {time.time() - t0:.0f}s", flush=True)

random.Random(0).shuffle(kept)
dev, train = kept[:a.dev], kept[a.dev:]
for name, part in (("train", train), ("dev", dev)):
    with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in sorted(part, key=lambda x: x["id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
rep.update({"kept": len(kept), "train": len(train), "dev": len(dev), "hours": round(sum(r["duration"] for r in kept) / 3600, 2),
            "cer_median": round(float(np.median(cers)), 3), "cer_p90": round(float(np.percentile(cers, 90)), 3), "max_cer": a.max_cer})
json.dump(rep, open(out / "report.json", "w"), indent=1); print(json.dumps(rep, indent=1), flush=True)

if a.push:
    from datasets import Audio, Dataset, DatasetDict
    def ds(part):
        d = Dataset.from_list([{"audio": str(ROOT / r["audio"]), "text": r["text"], "speaker_id": 0, "id": r["id"]} for r in part])
        return d.cast_column("audio", Audio(sampling_rate=16000))
    DatasetDict({"train": ds(train), "dev": ds(dev)}).push_to_hub(a.push, private=True)
    print("pushed", a.push, flush=True)
