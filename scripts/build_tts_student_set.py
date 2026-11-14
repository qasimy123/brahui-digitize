"""Training set for the small browser voice from the checked teacher clips (scripts/tts_distill_filter.py).
  python3 scripts/build_tts_student_set.py [--src data/tts/distill/synth] [--out data/tts/student] [--max-cer 0.15] [--dev 100]
Keeps clips the ASR could read back (CER <= --max-cer), 1-14 s, with a plausible speaking rate; writes an audio folder
<out>/audio/{train,dev}/{metadata.jsonl (file_name, text = vowelled Latin), audio links} and <out>/dev.jsonl (Arabic text, for scoring)."""
import argparse, csv, json, os, random, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("--src", default="data/tts/distill/synth"); ap.add_argument("--out", default="data/tts/student"); ap.add_argument("--max-cer", type=float, default=0.15); ap.add_argument("--dev", type=int, default=100); a = ap.parse_args()
S = ROOT / a.src; O = ROOT / a.out; sub = "flac" if (S / "flac").is_dir() else "wavs"; ext = "." + sub.rstrip("s")
rows = [json.loads(l) for l in open(S / "clips.jsonl", encoding="utf-8")]; n0 = len(rows); h0 = sum(r["sec"] for r in rows) / 3600
keep = [r for r in rows if r["cer"] <= a.max_cer and 1.0 <= r["sec"] <= 14.0 and 1.2 <= r["wps"] <= 3.8 and (S / sub / (r["id"] + ext)).exists()]
random.Random(5).shuffle(keep); n = min(a.dev, len(keep) // 10); parts = {"dev": keep[:n], "train": keep[n:]}
for name, rs in parts.items():
    d = O / "audio" / name; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    with open(d / "metadata.jsonl", "w", encoding="utf-8") as f:     # jsonl, not csv: datasets 3.1 rejects the csv's inferred column type
        for r in rs:
            src = S / sub / (r["id"] + ext); dst = d / (r["id"] + ext)
            try: os.link(src, dst)
            except OSError: shutil.copy(src, dst)
            f.write(json.dumps({"file_name": dst.name, "text": r["vowelled"]}, ensure_ascii=False) + "\n")
with open(O / "dev.jsonl", "w", encoding="utf-8") as f:
    for r in parts["dev"]: f.write(json.dumps({"id": r["id"], "text": r["text"], "vowelled": r["vowelled"], "duration": r["sec"]}, ensure_ascii=False) + "\n")
rep = {"clips_in": n0, "hours_in": round(h0, 2), "kept": len(keep), "hours_kept": round(sum(r["sec"] for r in keep) / 3600, 2), "train": len(parts["train"]), "dev": n, "max_cer": a.max_cer}
json.dump(rep, open(O / "report.json", "w"), indent=1); print(json.dumps(rep))
