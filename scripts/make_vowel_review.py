"""Word-pronunciation review page for a native speaker who does not read the Arabic script: the most frequent words of the
TTS training text, each with real recordings of the word (cut out by forced alignment with the s3 ASR), an English gloss
and the vowelled Latin spelling brahui.vowelize proposes, editable. The reviewer listens and fixes the spelling.
Words are ordered least certain first (confidence = how the Vowelizer found the spelling and how much its sources agree).

  uv run --with soundfile --with librosa python scripts/make_vowel_review.py [--n 250]  ->  evals/tts/vowel_review/vowel_review.html
  uv run python scripts/make_vowel_review.py --reorder    # rebuild the page from the clips already in it (no re-alignment)
  uv run python scripts/make_vowel_review.py --apply ~/Downloads/brahui_vowel_review_<name>.json   # -> data/lexicon/vowel_overrides.json
"""
import argparse, base64, collections, json, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from brahui.text import normalize  # noqa: E402
from brahui.vowelize import Vowelizer, key_arabic  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=250); ap.add_argument("--apply"); ap.add_argument("--examples", type=int, default=2)
ap.add_argument("--reorder", action="store_true", help="rebuild the page from the items already in it (no re-alignment): new confidence labels, order and template")
a = ap.parse_args()
OV = ROOT / "data/lexicon/vowel_overrides.json"
if a.apply:
    d = json.load(open(a.apply)); ov = json.load(open(OV)) if OV.exists() else {}
    for r in d["words"]:
        if r["status"] in ("ok", "fixed") and r["roman"].strip(): ov[r["word"]] = r["roman"].strip()
    json.dump(ov, open(OV, "w"), ensure_ascii=False, indent=0); print(len(ov), "overrides ->", OV); sys.exit()

OUT = ROOT / "evals/tts/vowel_review/vowel_review.html"; v = Vowelizer()
def confidence(w):
    """-> conf (low | medium | high), why, alt: other spellings the sources give for the same written word"""
    rom, how = v.word(w); c = collections.Counter(v.bykey.get(key_arabic(w), {}))
    for f, n in v.direct.get(w, {}).items(): c[f] += n
    share = c[rom] / sum(c.values()) if c.get(rom) else 0.0; alt = [f for f, _ in c.most_common(4) if f != rom][:3]
    if how == "fallback": return "low", "No dictionary source: this spelling is a guess.", alt
    if how == "split": return "low", "Not in the dictionary: built from a known stem and a known ending.", alt
    if how == "manual": return "medium", "Entered by hand (name or loanword).", alt
    if share < 0.6: return "low", f"The sources disagree: this spelling is {round(share * 100)}% of the examples.", alt
    if how == "key": return "medium", "Matched to a dictionary word by its consonants, not found as this exact written word.", alt
    if share < 0.85: return "medium", f"In the dictionary, but another spelling also occurs ({round(share * 100)}% this one).", alt
    return "high", "Found in the dictionary as this exact word.", alt
AC = ROOT / "evals/tts/vowel_review/acoustic_check.json"   # scripts/vowel_acoustic.py: which spelling the recordings fit
def acoustic(it, row, acc):
    """adds the measured evidence to a low-confidence word; switches the proposal only when phone model and formants both back another listed spelling"""
    if "candidates" not in row: return
    n = row["n"]; cands = row["candidates"]; pick = row["phone_pick"]; ftop = next(iter(row["formant_votes"]), None)
    if len(cands) > 1:
        it["ac"] = f"Measured on {n} recordings, the sound fits: " + ", ".join(f"{k} {c}" for k, c in sorted(cands.items(), key=lambda x: -x[1])) + "."
        if ftop in cands and pick[0] not in ftop.split(" / "): it["ac"] += f" A second measurement (formants) leans to {ftop}, so this is not settled."
        elif row["phone_share"] >= 0.8 and ftop in cands and it["r"] not in pick:
            it["ac"] += f" Both measurements agree, so the spelling was changed from {it['r']}."; it["alt"] = [it["r"]] + [x for x in it["alt"] if x != pick[0]]; it["r"] = pick[0]
    elif row.get("open_pick") and row["open_share"] >= 0.7:
        it["ac"] = f"Measured on {n} recordings: {round(row['open_share'] * 100)}% fit the vowel pattern \"{row['open_pick'].lower().replace(' ', '-')}\" better than this spelling."
    if it.get("ac"): it["ac"] += f" (Computer measurement: right about {acc} times in 10 on known words.)"
def finish(items):
    for it in items: it["r"], it["how"] = v.word(it["w"]); it["conf"], it["why"], it["alt"] = confidence(it["w"]); it.pop("ac", None)
    if AC.exists():
        d = json.load(open(AC, encoding="utf-8")); rows = {r["w"]: r for r in d["words"]}; pm = d["calibration"]["phone_model"]
        for it in items:
            if it["w"] in rows: acoustic(it, rows[it["w"]], round(10 * pm["word_majority_correct"] / pm["words"]))
    items.sort(key=lambda it: (["low", "medium", "high"].index(it["conf"]), -it["c"]))   # least certain first, frequent first inside a tier
    html = (Path(__file__).parent / "vowel_review_template.html").read_text(encoding="utf-8").replace("/*ITEMS*/[]", json.dumps(items, ensure_ascii=False))
    OUT.write_text(html, encoding="utf-8"); n = collections.Counter(it["conf"] for it in items)
    print(OUT, len(items), "words;", dict(n), ";", sum(not i["ex"] for i in items), "without audio;", sum(not i["en"] for i in items), "without gloss;", round(len(html) / 1e6, 1), "MB")
if a.reorder:
    line = next(l for l in OUT.read_text(encoding="utf-8").splitlines() if l.startswith("const ITEMS = "))
    finish(json.loads(line[len("const ITEMS = "):].rstrip().rstrip(";"))); sys.exit()

import numpy as np, librosa, torch, torchaudio  # noqa: E402
from transformers import AutoProcessor, Wav2Vec2ForCTC  # noqa: E402
STRIP = '۔،؟؛!.,?:"'
freq = collections.Counter(); rows = []
for l in open(ROOT / "data/tts/multi2/train.jsonl", encoding="utf-8"):
    r = json.loads(l); r["words"] = [w.strip(STRIP) for w in normalize(r["text"]).split()]; r["words"] = [w for w in r["words"] if w]; rows.append(r)
    for w in r["words"]: freq[w] += 1
tot = sum(freq.values()); top = [w for w, _ in freq.most_common(a.n)]
gloss = {}
for line in open(ROOT / "data/lexicon/flywheel_brh_en.txt", encoding="utf-8"):
    p = line.split(" | ")
    if len(p) >= 2 and not line.startswith("#"): gloss[p[0].strip()] = "; ".join(re.sub(r"\s*\[\d+\]", "", g).strip() for g in p[1].split(";")[:2])
for line in open(ROOT / "data/lexicon/shad_brahui_english_ocr.jsonl", encoding="utf-8"):
    r = json.loads(line); gloss.setdefault(normalize(r["headword"]), r.get("english", ""))
# examples: clean read speech first, short utterances, the word not at the very edge when possible
rank = {"common_voice": 0, "omnilingual": 1, "brahuispeech": 2}
rows.sort(key=lambda r: (rank.get(r["source"], 3), -r.get("ovrl", 0) if r.get("ovrl") else 0))
want = {w: [] for w in top}
for r in rows:
    if not 2 <= r["duration"] <= 12 or len(r["words"]) < 3: continue
    for i, w in enumerate(r["words"]):
        if w in want and len(want[w]) < a.examples and 0 < i < len(r["words"]) - 1 and all(x[0] is not r for x in want[w]): want[w].append((r, i))
for r in rows:   # words that only occur at an edge
    if not 2 <= r["duration"] <= 12: continue
    for i, w in enumerate(r["words"]):
        if w in want and not want[w]: want[w].append((r, i))
utts = {}
for w, ex in want.items():
    for r, i in ex: utts.setdefault(r["id"], r)
print(len(utts), "utterances to align", flush=True)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
proc = AutoProcessor.from_pretrained(ROOT / "runs/ctc1b_brh_s3/final"); model = Wav2Vec2ForCTC.from_pretrained(ROOT / "runs/ctc1b_brh_s3/final").to(dev).eval()
tok = proc.tokenizer; blank = model.config.pad_token_id; delim = tok.convert_tokens_to_ids(tok.word_delimiter_token); unk = tok.unk_token_id
spans = {}; audio = {}
for k, (uid, r) in enumerate(utts.items()):
    y, _ = librosa.load(ROOT / r["audio"], sr=16000, mono=True, offset=r.get("start", 0.0), duration=(r["end"] - r.get("start", 0.0)) if "end" in r else None)
    ids = []; wi = []
    for j, w in enumerate(r["words"]):
        t = [x for x in tok.convert_tokens_to_ids(list(w)) if x != unk]
        if not t: ids = None; break
        if ids: ids.append(delim); wi.append(-1)
        ids += t; wi += [j] * len(t)
    if not ids: continue
    with torch.no_grad(): lp = torch.log_softmax(model(proc(y, sampling_rate=16000, return_tensors="pt").input_values.to(dev)).logits.float(), -1).cpu()
    if lp.shape[1] < len(ids): continue
    try: al, _ = torchaudio.functional.forced_align(lp, torch.tensor([ids]), blank=blank)
    except Exception: continue
    al = al[0].tolist(); sp = {}; p = -1; prev = blank
    for f, t in enumerate(al):
        if t != blank and t != prev: p += 1
        prev = t
        if t != blank and wi[p] >= 0: s = sp.setdefault(wi[p], [f, f]); s[1] = f
    spans[uid] = {j: (s[0] * 0.02, s[1] * 0.02 + 0.04) for j, s in sp.items()}; audio[uid] = y
    if (k + 1) % 100 == 0: print(k + 1, flush=True)
def mp3(y):
    y = (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)
    return base64.b64encode(subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", "16000", "-ac", "1", "-i", "-", "-b:a", "48k", "-f", "mp3", "-"], input=y.tobytes(), capture_output=True, check=True).stdout).decode()
items = []
for w in top:
    rom, how = v.word(w); ex = []
    for r, i in want[w]:
        sp = spans.get(r["id"])
        if not sp or i not in sp: continue
        y = audio[r["id"]]; lo = max(0, i - 2); hi = min(len(r["words"]) - 1, i + 2)
        if lo not in sp or hi not in sp: continue
        cut = lambda s, e, pad: y[max(0, int((s - pad) * 16000)): int((e + pad) * 16000)]  # noqa: E731
        ex.append({"phrase": mp3(cut(sp[lo][0], sp[hi][1], 0.12)), "word": mp3(cut(sp[i][0], sp[i][1], 0.06)),
                   "txt": " ".join(("<b>" + v.word(x)[0] + "</b>") if j == i else v.word(x)[0] for j, x in enumerate(r["words"]) if lo <= j <= hi), "src": r["source"]})
    items.append({"w": w, "r": rom, "how": how, "c": freq[w], "en": gloss.get(w, ""), "ex": ex})
finish(items); print(round(sum(i["c"] for i in items) / tot * 100), "% of running words")
