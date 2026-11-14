"""TTS candidate pool from the Balochistans Today news archive (one anchor, studio): pseudo-labelled segments straight from
data/media/transcripts/balochistans_today_news/*.jsonl (scripts/transcribe_media.py), minus what data/tts/pool_extra.jsonl has.

  uv run python scripts/build_tts_pool_news.py  -> data/tts/pool_news.jsonl
  then: score_tts_pool.py mos|asr --manifest data/tts/pool_news.jsonl --name news ; build_tts_multispeaker.py

Rows: id, source, audio, start, end, duration, text, speaker (one label per bulletin: field reports are other voices),
pseudo=true, conf. Held-out media test files are excluded. The ASR manifest media_pseudo_s3.jsonl is left untouched.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from brahui.text import normalize  # noqa: E402

SRC = "balochistans_today_news"
held = {l.strip() for l in open(ROOT / "data/eval/media_test_files.txt") if l.strip()}
have = {json.loads(l)["id"] for l in open(ROOT / "data/tts/pool_extra.jsonl", encoding="utf-8")}
rows = []
for f in sorted((ROOT / "data/media/transcripts" / SRC).glob("*.jsonl")):
    wav = ROOT / "data/media/audio" / SRC / f"{f.stem}.wav"
    if f"{SRC}/{f.stem}" in held or not wav.exists(): continue
    for l in open(f, encoding="utf-8"):
        r = json.loads(l); d = r["end"] - r["start"]; rid = f"media_{f.stem}_s{r['seg']:04d}"; text = normalize(r["text"]).strip()
        if rid in have or r.get("conf", 1) < 0.80 or not 2 <= d <= 20 or len(text) < 8: continue
        rows.append({"id": rid, "source": f"media_{SRC}", "audio": str(wav.relative_to(ROOT)), "start": round(r["start"], 2), "end": round(r["end"], 2),
                     "duration": round(d, 2), "text": text, "speaker": "media_" + f.stem, "pseudo": True, "conf": round(r.get("conf", 1), 3)})
with open(ROOT / "data/tts/pool_news.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{len(rows)} segments, {sum(r['duration'] for r in rows) / 3600:.1f} h -> data/tts/pool_news.jsonl")
