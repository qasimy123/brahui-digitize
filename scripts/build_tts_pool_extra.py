"""Pool the non-BrahuiSpeech audio for TTS scoring: Omnilingual chunks (human transcripts), all validated Common Voice
clips, and the media pseudo-labelled segments. Writes data/tts/pool_extra.jsonl (id, source, audio, [start, end],
duration, text, speaker, [pseudo, conf]); score it with scripts/score_tts_pool.py --manifest ... --name extra.

  uv run python scripts/build_tts_pool_extra.py
Omnilingual/Common Voice dev+test clips are ASR evaluation data: fine for TTS training, never for ASR training.
"""
from __future__ import annotations
import csv, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from brahui.text import normalize  # noqa: E402

MAN = ROOT / "data/manifests"; rows = []
for split in ("omni_train", "omni_dev", "omni_test"):
    p = MAN / f"{split}_chunks.jsonl"
    if not p.exists():
        print(f"missing {p.name}: run scripts/segment_long.py --split {split}"); continue
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        row = {"id": r["id"], "source": "omnilingual", "audio": r["audio"], "duration": round(r["duration"], 2), "text": normalize(r["text"]),
               "speaker": "omni_" + re.search(r"spk\d+", r["id"]).group(), "asr_split": split}
        if "start" in r: row["start"], row["end"] = r["start"], r["end"]
        rows.append(row)

CV = ROOT / "data/raw/mdc/common_voice_27_brh/cv-corpus-27.0-2026-09-11/brh"
dur = {r["clip"]: int(r["duration[ms]"]) / 1000 for r in csv.DictReader(open(CV / "clip_durations.tsv", encoding="utf-8"), delimiter="\t")}
for r in csv.DictReader(open(CV / "validated.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
    if int(r["down_votes"] or 0) > 0:
        continue
    rows.append({"id": "cv_" + r["path"][:-4], "source": "common_voice", "audio": str((CV / "clips" / r["path"]).relative_to(ROOT)), "duration": dur.get(r["path"], 0.0),
                 "text": normalize(r["sentence"]), "speaker": "cv_" + r["client_id"][:16]})

for line in open(MAN / "media_pseudo_s3.jsonl", encoding="utf-8"):
    r = json.loads(line)
    rows.append({"id": r["id"], "source": r["source"], "audio": r["audio"], "start": r["start"], "end": r["end"], "duration": round(r["duration"], 2),
                 "text": normalize(r["text"]), "speaker": "media_" + Path(r["audio"]).stem, "pseudo": True, "conf": r["conf"]})

with open(ROOT / "data/tts/pool_extra.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
by = {}
for r in rows:
    by.setdefault(r["source"], [0, 0.0]); by[r["source"]][0] += 1; by[r["source"]][1] += r["duration"] / 3600
for k, (n, h) in sorted(by.items()):
    print(f"{k:36s} {n:5d} clips {h:5.1f} h")
