"""The browser's vowel front end: one table word -> vowelled spelling, for every Brahui word the lexicons know plus every
word (seen >= 2 times) in the Brahui side of the MT training text, which is what the translation model can output.
Unknown words there are written by the letter model now, so the browser needs no model for them.
  uv run python scripts/export_vowel_lexicon_js.py  ->  demo/web/models/vowel_lex.json   {word: spelling}"""
import collections, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from brahui.text import normalize  # noqa: E402
from brahui.vowelize_nn import load, STRIP  # noqa: E402
hv = load("heard"); words = set(hv.review) | set(hv.loans) | set(hv.heard) | set(hv.dict); c = collections.Counter()
for p in ("data/parallel/synthetic_bt_everyday.jsonl", "data/parallel/synthetic_fineweb_brh_en_filtered.jsonl", "data/parallel/gold_train_nodev.jsonl", "data/parallel/retranslated_brh_en.jsonl"):
    for l in open(ROOT / p, encoding="utf-8"):
        for w in normalize(json.loads(l).get("brh") or "").split():
            w = w.strip(STRIP)
            if w: c[w] += 1
extra = [w for w, n in c.items() if n >= 2 and w not in words and len(w) <= 24 and not any(ch.isascii() and ch.isalnum() for ch in w)]
print(len(words), "lexicon words +", len(extra), "MT-corpus words", flush=True)
out = {}; src = collections.Counter()
for i, w in enumerate(sorted(words) + extra):
    s, how = hv.explain(w); out[w] = s; src[how] += 1
    if (i + 1) % 20000 == 0: print(i + 1, flush=True)
p = ROOT / "demo/web/models/vowel_lex.json"; json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print(p, len(out), "words", round(p.stat().st_size / 1e6, 2), "MB", dict(src))
