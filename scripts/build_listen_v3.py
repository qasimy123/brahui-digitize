"""Listening set v3 for the old-voice vs vowelled-voice blind test: complete sentences with a human English translation
(BOUQuET dev and SMOL test, both translated by people), instead of the fragments of running text used on 2026-11-11.

  uv run python scripts/build_listen_v3.py --vowelizer runs/vowelizer_v1
  -> data/tts/listen_v3/set.jsonl {id, text, en, roman, vowelled}
     data/tts/f5/listen3_roman/metadata.csv   text as the current voice was trained (romanize speak mode)
     data/tts/f5/listen3_vow/metadata.csv     text from the text-only vowelizer, i.e. what the new voice gets at synthesis
"""
import argparse, json, random, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from brahui.text import normalize, romanize  # noqa: E402
from brahui.vowelize_nn import load  # noqa: E402
PUNCT = str.maketrans({"۔": ".", "،": ",", "؟": "?", "؛": ",", "٪": "%", '"': None, "٭": None, "“": None, "”": None})
ap = argparse.ArgumentParser(); ap.add_argument("--vowelizer", required=True); ap.add_argument("--n", type=int, default=24); a = ap.parse_args()
pool = []
for name, brh, en in (("bouquet_sent_dev", "src", "tgt"), ("smol_sent_test", "tgt", "src")):
    for l in open(ROOT / "data/manifests" / f"{name}.jsonl", encoding="utf-8"):
        r = json.loads(l); t = normalize(r[brh]); n = len(t.split())
        if 6 <= n <= 13 and len(r[en]) <= 100 and not re.search(r"[A-Za-z0-9۰-۹()\[\]:؛\"'“”]", t): pool.append((name, t, r[en]))
random.Random(3).shuffle(pool); half = a.n // 2
pick = [p for p in pool if p[0] == "bouquet_sent_dev"][:half] + [p for p in pool if p[0] == "smol_sent_test"][:a.n - half]
nv = load(ROOT / a.vowelizer); out = ROOT / "data/tts/listen_v3"; out.mkdir(parents=True, exist_ok=True); rows = []
for k, (src, t, en) in enumerate(pick):
    rows.append({"id": f"L{k + 1:02d}", "source": src, "text": t, "en": en, "roman": romanize(t, speak=True).translate(PUNCT), "vowelled": nv.text(t).translate(PUNCT)})
with open(out / "set.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
for mode, key in (("roman", "roman"), ("vow", "vowelled")):
    d = ROOT / f"data/tts/f5/listen3_{mode}"; d.mkdir(parents=True, exist_ok=True)
    with open(d / "metadata.csv", "w", encoding="utf-8") as f:
        f.write("audio_file|text\n")
        for r in rows: f.write(f"wavs/{r['id']}.wav|{r[key]}\n")
print(len(pool), "candidate sentences;", len(rows), "picked")
for r in rows[:6]: print(r["id"], "|", r["en"], "\n    old:", r["roman"], "\n    new:", r["vowelled"])
