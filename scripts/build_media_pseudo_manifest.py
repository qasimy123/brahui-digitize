"""Turn the stage-3 media transcripts (data/media/transcripts/<source>/<id>.jsonl, one row per segment) into a training
manifest of pseudo-labelled chunks for ASR distillation. Files in data/eval/media_test_files.txt (the media test set) and
the demo clips are excluded; low-confidence and over-long segments are dropped.

  uv run python scripts/build_media_pseudo_manifest.py  -> data/manifests/media_pseudo_s3.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TR, AUD, OUT = ROOT / "data/media/transcripts", ROOT / "data/media/audio", ROOT / "data/manifests/media_pseudo_s3.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-conf", type=float, default=0.80); ap.add_argument("--max-seconds", type=float, default=25.0)
    ap.add_argument("--min-seconds", type=float, default=1.0); ap.add_argument("--min-chars", type=int, default=8)
    a = ap.parse_args()
    held = {l.strip() for l in open(ROOT / "data/eval/media_test_files.txt") if l.strip()}
    rows, st = [], Counter()
    for f in sorted(TR.glob("*/*.jsonl")):
        src, vid = f.parent.name, f.stem
        if f"{src}/{vid}" in held or src == "demo_news":
            st["held_out_files"] += 1; continue
        wav = AUD / src / f"{vid}.wav"
        if not wav.exists():
            st["no_audio"] += 1; continue
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            d = r["end"] - r["start"]
            if r.get("conf", 1) < a.min_conf: st["low_conf"] += 1; continue
            if not (a.min_seconds <= d <= a.max_seconds): st["bad_len"] += 1; continue
            if len(r["text"].strip()) < a.min_chars: st["short_text"] += 1; continue
            rows.append({"id": f"media_{vid}_s{r['seg']:04d}", "source": f"media_{src}", "split": "train",
                         "audio": str(wav.relative_to(ROOT)), "start": round(r["start"], 2), "end": round(r["end"], 2),
                         "duration": round(d, 2), "text": r["text"].strip(), "conf": round(r.get("conf", 1), 3)})
            st["kept"] += 1
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(dict(st), f"{sum(r['duration'] for r in rows)/3600:.2f} h", "->", OUT)


if __name__ == "__main__":
    main()
