"""Data for the vowel-aware acoustic model (docs: evals/tts/vowel_asr/README.md): the TTS training audio as 16 kHz FLAC and,
per utterance, every word with its vowelled Latin spelling and a trust label.

  k0 / k1  spelling trusted (Vowelizer confidence high or medium); the word type is in cross-validation fold 0 or 1.
           Model m is trained with fold-m words hidden, so every trusted spelling is later tested by a model that never saw it.
  u        spelling not trusted (fallback, stem+ending, sources disagree, rare letters): never a training label, scored by both models.

  uv run python scripts/build_vowel_asr.py   ->  data/tts/vowel_asr/{train.jsonl,dev.jsonl,types.json,vocab.json,audio/*.flac}
"""
import collections, hashlib, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from brahui.text import normalize, romanize  # noqa: E402
from brahui.vowelize import Vowelizer  # noqa: E402
OUT = ROOT / "data/tts/vowel_asr"; (OUT / "audio").mkdir(parents=True, exist_ok=True)
STRIP = '۔،؟؛!.,?:"\'()…'
v = Vowelizer(); rows = []; freq = collections.Counter()
for l in open(ROOT / "data/tts/multi2/train.jsonl", encoding="utf-8"):
    r = json.loads(l); r["words"] = [w for w in (w.strip(STRIP) for w in normalize(r["text"]).split()) if w]
    if len(r["words"]) >= 2: rows.append(r); freq.update(r["words"])
types = {w: {**v.judge(w), "n": n} for w, n in freq.items()}
for t in types.values(): t["r"] = t["r"].lower().strip(); t["alt"] = [a.lower() for a in t["alt"]]
for w, t in types.items(): t["sk"] = romanize(w, speak=True).lower().strip()   # consonant skeleton: what the TTS reads today
chars = collections.Counter(c for w, t in types.items() if t["conf"] != "low" for c in t["r"] * 1)
occ = collections.Counter()
for w, t in types.items():
    if t["conf"] != "low":
        for c in set(t["r"]): occ[c] += t["n"]
keep = sorted(c for c, n in occ.items() if n >= 30 and (c.isalpha() or c == "'"))
vocab = {"<blank>": 0, "|": 1, **{c: i + 2 for i, c in enumerate(keep)}}
for w, t in types.items():
    ok = t["conf"] != "low" and t["r"] and all(c in vocab for c in t["r"]) and any(c in "aāiīuūeo" for c in t["r"])
    t["t"] = ("k" + str(int(hashlib.md5(w.encode()).hexdigest(), 16) % 2)) if ok else "u"
tok = sum(freq.values()); by = collections.Counter(); byt = collections.Counter()
for w, t in types.items(): by[t["t"]] += t["n"]; byt[t["t"]] += 1
print("utterances", len(rows), "tokens", tok, "types", len(types), "| share of tokens:", {k: round(n / tok, 3) for k, n in by.items()}, "| types:", dict(byt), "| vocab", len(vocab), "".join(keep))

def conv(r):
    dst = OUT / "audio" / (r["id"].replace("/", "_") + ".flac")
    if dst.exists() and dst.stat().st_size > 1000: return dst
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if "end" in r: cmd += ["-ss", f"{r.get('start', 0.0):.3f}", "-t", f"{r['end'] - r.get('start', 0.0):.3f}"]
    subprocess.run(cmd + ["-i", str(ROOT / r["audio"]), "-ac", "1", "-ar", "16000", str(dst)], check=True); return dst
with ThreadPoolExecutor(8) as ex: dsts = list(ex.map(conv, rows))
out = []
for r, d in zip(rows, dsts):
    out.append({"id": r["id"], "audio": str(d.relative_to(ROOT)), "duration": r["duration"], "source": r["source"], "words": [[w, types[w]["r"], types[w]["t"]] for w in r["words"]]})
out.sort(key=lambda r: hashlib.md5(r["id"].encode()).hexdigest()); dev, train = out[:300], out[300:]
for name, part in (("dev", dev), ("train", train)):
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in part: f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump(types, open(OUT / "types.json", "w", encoding="utf-8"), ensure_ascii=False); json.dump(vocab, open(OUT / "vocab.json", "w", encoding="utf-8"), ensure_ascii=False)
print("train", len(train), "dev", len(dev), "hours", round(sum(r["duration"] for r in out) / 3600, 1))
