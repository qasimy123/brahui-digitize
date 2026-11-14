"""Second multi-speaker F5 training set: the first set (data/tts/multi/train.jsonl) with the news bulletins capped, plus the
enhanced, Urdu-screened bulk media (data/tts/bulk_enh/tts_grade_brahui.jsonl). The first set was 62% news (22 of 35 h), the
likely source of the news-anchor accent the user dislikes (2026-11-11).

  uv run python scripts/build_tts_multi2.py [--news-hours 5]   ->  data/tts/multi2/{train.jsonl,report.json}
"""
import argparse, collections, json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("--news-hours", type=float, default=5.0); a = ap.parse_args()
old = [json.loads(l) for l in open(ROOT / "data/tts/multi/train.jsonl", encoding="utf-8")]
news = [r for r in old if r["source"] == "media_balochistans_today_news"]; rest = [r for r in old if r["source"] != "media_balochistans_today_news"]
random.Random(0).shuffle(news); keep = []; h = 0.0
for r in news:
    if h >= a.news_hours * 3600: break
    keep.append(r); h += r["duration"]
new = []
for l in open(ROOT / "data/tts/bulk_enh/tts_grade_brahui.jsonl", encoding="utf-8"):
    r = json.loads(l)
    new.append({"id": f"{r['source']}__{r['video']}__{r['seg']}", "source": r["source"], "audio": r["audio"], "text": r["text"], "duration": r["duration"],
                "speaker": f"{r['source']}/{r['video']}", "conf": r["conf"], "ovrl": r["ovrl"]})
rows = rest + keep + new; out = ROOT / "data/tts/multi2"; out.mkdir(parents=True, exist_ok=True)
with open(out / "train.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
hs = collections.Counter()
for r in rows: hs[r["source"]] += r["duration"] / 3600
rep = {"clips": len(rows), "hours": round(sum(hs.values()), 2), "news_cap_hours": a.news_hours, "hours_by_source": {k: round(v, 2) for k, v in hs.most_common()}}
json.dump(rep, open(out / "report.json", "w"), indent=1); print(rep)
