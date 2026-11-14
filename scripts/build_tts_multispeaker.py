"""Select a clean multi-speaker Brahui TTS set (for F5-TTS style fine-tuning) from the scored pools.

  uv run python scripts/build_tts_multispeaker.py [--max-cer 0.10] [--min-ovrl 3.0] [--min-conf 0.9] [--max-cer-bs 0.20] [--min-s 2] [--max-s 20] [--sweep]

Pools (scripts/score_tts_pool.py): BrahuiSpeech (data/tts/bs_scores.jsonl, speaker = ECAPA cluster) and the extra pool
(data/tts/extra_scores.jsonl over data/tts/pool_extra.jsonl: Omnilingual, Common Voice, media). A clip is kept when
DNSMOS overall >= --min-ovrl, its text has no Latin letters/digits, its duration is usable, and its label is trusted:
ASR CER <= --max-cer for human labels (--max-cer-bs for BrahuiSpeech and Omnilingual: spontaneous speech, where the same ASR's median
CER is 17% and 10% against 4% on read Common Voice, so 0.10 would mostly measure the ASR), confidence >= --min-conf for pseudo-labels.
Writes data/tts/multi/{train,dev}.jsonl (id, source, audio, [start, end], text, duration, speaker, cer, ovrl), refs.jsonl
(one 4-11 s clip from each of the 3 largest human-labelled speakers: voices to clone at eval time) and report.json; --sweep only prints hours kept per source across thresholds. Dev = held-out clips, never whole speakers.
"""
from __future__ import annotations
import argparse, collections, json, random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from brahui.text import normalize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/tts/multi")
ap.add_argument("--max-cer", type=float, default=0.10); ap.add_argument("--min-ovrl", type=float, default=3.0); ap.add_argument("--min-conf", type=float, default=0.9); ap.add_argument("--max-cer-bs", type=float, default=0.20)
ap.add_argument("--min-s", type=float, default=2.0); ap.add_argument("--max-s", type=float, default=20.0)
ap.add_argument("--dev", type=int, default=100); ap.add_argument("--sweep", action="store_true")
a = ap.parse_args()

def read(p):
    p = ROOT / p
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []

latin = re.compile(r"[A-Za-z0-9]")
pool = []
# BrahuiSpeech
man = {r["id"]: r for r in read("data/manifests/brahuispeech_train_norm.jsonl")}
spk = {r["id"]: r["cluster"] for r in read("data/tts/bs_speakers.jsonl")}
for s in read("data/tts/bs_scores.jsonl"):
    m = man[s["id"]]
    pool.append({"id": s["id"], "source": "brahuispeech", "audio": m["audio"], "text": normalize(m.get("text_norm") or m["text"]), "duration": s["duration"],
                 "speaker": f"bs_{spk.get(s['id'], -1)}", "cer": s["cer"], "conf": None, "ovrl": s["ovrl"]})
# extra pool
ex = {r["id"]: r for r in read("data/tts/pool_extra.jsonl") + read("data/tts/pool_news.jsonl")}
for s in read("data/tts/extra_scores.jsonl") + read("data/tts/news_scores.jsonl"):
    m = ex[s["id"]]
    row = {"id": s["id"], "source": m["source"], "audio": m["audio"], "text": m["text"], "duration": s["duration"], "speaker": m["speaker"],
           "cer": s.get("cer"), "conf": s.get("conf"), "ovrl": s["ovrl"]}
    if "start" in m: row["start"], row["end"] = m["start"], m["end"]
    pool.append(row)
n_all = len(pool)
pool = [p for p in pool if not latin.search(p["text"]) and a.min_s <= p["duration"] <= a.max_s and p["text"].strip()]
print(f"{n_all} scored clips (ovrl >= the scorer's bar); {len(pool)} pass text+duration ({sum(p['duration'] for p in pool) / 3600:.1f} h)")

def pick(max_cer, min_ovrl):
    return [p for p in pool if p["ovrl"] >= min_ovrl and (p["cer"] <= (max(max_cer, a.max_cer_bs) if p["source"] in ("brahuispeech", "omnilingual") else max_cer) if p["cer"] is not None else (p["conf"] or 0) >= a.min_conf)]

def hours_by(rows, key):
    h = collections.Counter()
    for p in rows: h[p[key]] += p["duration"] / 3600
    return h

if a.sweep:
    for mo in (2.5, 2.8, 3.0, 3.2):
        for mc in (0.05, 0.10, 0.20):
            k = pick(mc, mo); src = hours_by(k, "source"); sp = hours_by(k, "speaker")
            print(f"ovrl>={mo} cer<={mc:<4}: {sum(src.values()):5.1f} h, {len(sp):4d} speakers ({sum(1 for v in sp.values() if v >= 0.25):3d} with >=15 min) | " +
                  ", ".join(f"{s.replace('media_', 'm:')[:14]} {h:.1f}" for s, h in src.most_common()))
    sys.exit()

kept = pick(a.max_cer, a.min_ovrl)
sph = hours_by([p for p in kept if p["cer"] is not None], "speaker"); refs = []
for s_, _ in sph.most_common(3):
    c = [p for p in kept if p["speaker"] == s_]; c = [p for p in c if 4 <= p["duration"] <= 11] or [min(c, key=lambda p: p["duration"])]
    refs.append(min(c, key=lambda p: (p["cer"], -p["ovrl"])))  # F5 cuts references longer than 12 s, which would desync the reference text
ref_ids = {p["id"] for p in refs}
random.Random(0).shuffle(kept)
rest = [p for p in kept if p["id"] not in ref_ids]
dev, train = rest[:a.dev], rest[a.dev:] + refs
out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
for name, part in (("train", train), ("dev", dev), ("refs", refs)):
    with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in sorted(part, key=lambda x: x["id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
src, sp = hours_by(kept, "source"), hours_by(kept, "speaker")
rep = {"kept": len(kept), "hours": round(sum(src.values()), 2), "train": len(train), "dev": len(dev), "max_cer": a.max_cer, "min_ovrl": a.min_ovrl, "min_conf": a.min_conf, "max_cer_bs": a.max_cer_bs, "refs": [p["id"] for p in refs],
       "hours_by_source": {k: round(v, 2) for k, v in src.most_common()}, "speakers": len(sp), "speakers_ge_15min": sum(1 for v in sp.values() if v >= 0.25),
       "top_speakers_hours": {k: round(v, 2) for k, v in sp.most_common(10)}}
json.dump(rep, open(out / "report.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(rep, indent=1, ensure_ascii=False))
