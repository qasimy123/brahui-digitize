"""F5 metadata with vowelled text, reusing the wavs of an existing export (same ids).

  train text   data/tts/vowel_asr/retrans*.jsonl: each utterance with the vowels the acoustic model heard (scripts/vowel_retranscribe.py)
  dev / refs   the text-only vowelizer (brahui/vowelize_nn.py), i.e. exactly what synthesis from MT output will see
Punctuation comes from the original text, as in scripts/tts_f5_export.py.

  uv run python scripts/tts_f5_export_vowel.py --like data/tts/f5/multi2_roman --out data/tts/f5/multi2_vow --vowelizer runs/vowelizer_v1
  (writes <out>, <out>_dev, <out>_refs: metadata.csv only; on the pod set WAVS_FROM=multi2_roman)
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from brahui.text import normalize  # noqa: E402
from brahui.vowelize_nn import load, STRIP  # noqa: E402
PUNCT = str.maketrans({"۔": ".", "،": ",", "؟": "?", "؛": ",", "٪": "%", '"': None, "٭": None, "“": None, "”": None})
ap = argparse.ArgumentParser(); ap.add_argument("--like", required=True); ap.add_argument("--out", required=True); ap.add_argument("--vowelizer", required=True)
ap.add_argument("--train", default="data/tts/multi2/train.jsonl"); ap.add_argument("--dev", default="data/tts/multi/dev.jsonl"); a = ap.parse_args()
nv = load(ROOT / a.vowelizer); heard = {}
for n in ("retrans.jsonl", "retrans_dev.jsonl"):
    for l in open(ROOT / "data/tts/vowel_asr" / n, encoding="utf-8"): r = json.loads(l); heard[r["id"]] = [w[1] for w in r["words"]]
text_of = {}
for p in (a.train, a.dev):
    for l in open(ROOT / p, encoding="utf-8"): r = json.loads(l); text_of[r["id"]] = r["text"]
def vowelled(i):
    toks = normalize(text_of[i]).replace("|", " ").split(); core = [t.strip(STRIP) for t in toks]; ix = [k for k, c in enumerate(core) if c]
    sp = heard.get(i)
    if sp is None or len(sp) != len(ix): sp = nv.words([core[k] for k in ix]); src = "model"
    else: src = "heard"
    for k, s in zip(ix, sp):
        j = toks[k].find(core[k]); toks[k] = toks[k][:j] + s + toks[k][j + len(core[k]):]
    return " ".join(toks).translate(PUNCT), src
stats = {}
for suf in ("", "_dev", "_refs"):
    src = ROOT / (a.like + suf) / "metadata.csv"; dst = ROOT / (a.out + suf); dst.mkdir(parents=True, exist_ok=True); n = {"heard": 0, "model": 0, "missing": 0}
    with open(dst / "metadata.csv", "w", encoding="utf-8") as f:
        f.write("audio_file|text\n")
        for l in open(src, encoding="utf-8").read().splitlines()[1:]:
            wav, old = l.split("|", 1); i = Path(wav).stem
            if i not in text_of: f.write(l + "\n"); n["missing"] += 1; continue
            t, how = vowelled(i); f.write(f"{wav}|{t}\n"); n[how] += 1
    stats[suf or "train"] = n
print(stats)
for suf in ("", "_dev"): print(open(ROOT / (a.out + suf) / "metadata.csv", encoding="utf-8").read().splitlines()[1][:230])
