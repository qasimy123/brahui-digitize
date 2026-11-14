"""The script the teacher voice reads for the small browser TTS model: ~20k Brahui sentences, vowelled by the lookup front end.

Sources (Brahui side only), with quotas:
  transcripts   data/parallel/retranslated_brh_en.jsonl             real speech transcripts: every word was heard, spoken style
  written       data/parallel/synthetic_fineweb_brh_en_filtered.jsonl  well-formed written sentences (Opus translations)
  everyday      data/parallel/synthetic_bt_everyday.jsonl            short conversational MT output: what the demo will speak
  gold          data/parallel/gold_train_nodev.jsonl                 human translations
Kept: 4-20 words, no Latin letters or digits, at most one word the lexicons do not know (the letter model writes it) and
none left as a bare guess. Inside each quota, sentences that bring words seen fewer than 3 times so far go first.
  uv run python scripts/build_tts_distill_text.py [--n 20000]  ->  data/tts/distill/script.jsonl, script_report.json
"""
import argparse, collections, json, random, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from brahui.text import normalize  # noqa: E402
from brahui.vowelize_nn import load, STRIP  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20000); a = ap.parse_args()
PUNCT = str.maketrans({"۔": ".", "،": ",", "؟": "?", "؛": ",", "٪": "%", '"': None, "٭": None, "“": None, "”": None})
SRC = [("transcripts", "data/parallel/retranslated_brh_en.jsonl", 0.28), ("written", "data/parallel/synthetic_fineweb_brh_en_filtered.jsonl", 0.38),
       ("everyday", "data/parallel/synthetic_bt_everyday.jsonl", 0.27), ("gold", "data/parallel/gold_train_nodev.jsonl", 0.07)]
hv = load("heard"); known = [hv.review, hv.loans, hv.heard, hv.dict]
held = set()   # sentences used for listening tests and dev
for p in ("data/tts/listen_v3/set.jsonl", "data/tts/drill/r1.jsonl", "data/tts/multi/dev.jsonl"):
    if (ROOT / p).exists(): held |= {normalize(json.loads(l)["text"]) for l in open(ROOT / p, encoding="utf-8")}
rng = random.Random(11); cover = collections.Counter(); seen = set(); out = []; rep = {}
for name, path, share in SRC:
    cand = []
    for l in open(ROOT / path, encoding="utf-8"):
        r = json.loads(l); t = normalize(r.get("brh") or ""); ws = [w for w in (x.strip(STRIP) for x in t.split()) if w]
        if not 4 <= len(ws) <= 20 or len(t) > 170 or t in seen or t in held or re.search(r"[A-Za-z0-9۰-۹()\[\]:؛\"'“”/]", t): continue
        unk = [w for w in ws if not any(w in k for k in known)]
        if len(unk) > 1 or len(unk) / len(ws) > 0.1: continue
        seen.add(t); cand.append((r["id"], t, ws, unk))
    rng.shuffle(cand); quota = round(a.n * share); pick = []; rest = []
    for c in cand:     # first pass: sentences that bring under-covered words
        if len(pick) >= quota: break
        if any(cover[w] < 3 for w in c[2]): pick.append(c); cover.update(c[2])
        else: rest.append(c)
    pick += rest[:quota - len(pick)]; n0 = len(out)
    for i, t, ws, unk in pick:
        if unk and hv.explain(unk[0])[1] == "guess": continue
        out.append({"id": f"d{len(out):05d}", "source": name, "orig": i, "text": t, "vowelled": hv.text(t).translate(PUNCT), "n_words": len(ws)})
    rep[name] = {"candidates": len(cand), "picked": len(out) - n0}; print(name, rep[name], flush=True)
D = ROOT / "data/tts/distill"; D.mkdir(parents=True, exist_ok=True)
with open(D / "script.jsonl", "w", encoding="utf-8") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
words = sum(r["n_words"] for r in out); src = collections.Counter(hv.explain(w)[1] for w in cover.elements())
rep.update({"sentences": len(out), "words": words, "word_types": len(cover), "types_seen_3plus": sum(v >= 3 for v in cover.values()), "est_hours_at_0.5s_per_word": round(words * 0.5 / 3600, 1),
            "spelling_source_of_words": dict(src), "chars": "".join(sorted({c for r in out for c in r["vowelled"]}))})
json.dump(rep, open(D / "script_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1); print(json.dumps(rep, ensure_ascii=False)[:900])
