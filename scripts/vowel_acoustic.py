"""Acoustic check of the Vowelizer's spellings: which candidate spelling do real recordings of the word support?

For each word: N utterances from the TTS training set, the word located by forced alignment with the s3 ASR (as in
make_vowel_review.py), then two independent cues inside the word span:
  1. phone model (facebook/wav2vec2-xlsr-53-espeak-cv-ft, a universal IPA recogniser that never saw Brahui): its phone
     posteriors are summed into broad classes (A I U E O + consonants) and every candidate spelling is CTC-force-aligned;
     the best-scoring candidate gets the vote of that recording.
  2. formants (Praat Burg F1/F2 at each vowel), classified by centroids learned from the high-confidence words.
Vowel length (a/ā) is not visible to the class scoring; it is judged by vowel duration, calibrated the same way.
The high-confidence words (spelling known) are run through the same procedure against wrong-vowel variants: that is the
accuracy figure to trust the low-confidence verdicts by.

  uv run --with soundfile --with librosa --with praat-parselmouth python scripts/vowel_acoustic.py  ->  evals/tts/vowel_review/acoustic_check.json
"""
import argparse, collections, json, statistics, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
import numpy as np, librosa, torch, torchaudio, parselmouth  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoProcessor, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor  # noqa: E402
from brahui.text import normalize  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=16); ap.add_argument("--n-high", type=int, default=8); ap.add_argument("--high-words", type=int, default=70); a = ap.parse_args()
OUT = ROOT / "evals/tts/vowel_review/acoustic_check.json"
line = next(l for l in (ROOT / "evals/tts/vowel_review/vowel_review.html").read_text(encoding="utf-8").splitlines() if l.startswith("const ITEMS = "))
items = json.loads(line[len("const ITEMS = "):].rstrip().rstrip(";"))

# --- spelling -> broad phone classes
PM = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
vocab = json.load(open(hf_hub_download(PM, "vocab.json"), encoding="utf-8"))
CLS = {"A": "a ɐ ʌ ə æ ɑ ɜ aː ɑː ä ɒ aɪ aʊ", "I": "i ɪ iː ᵻ ɨ y", "U": "u ʊ uː ɯ ʉ", "E": "e ɛ eː ɛː e̞ eɪ", "O": "o ɔ oː ɔː oʊ o̞",
       "b": "b bʰ β", "p": "p pʰ ph", "t": "t t̪ th tʰ t[", "ṭ": "ʈ ʈʰ t", "d": "d d[ dʰ ð", "ḍ": "ɖ ɖʰ d", "k": "k kʰ kh q c", "g": "ɡ ɡʰ", "q": "q k",
       "kh": "x χ kh kʰ", "gh": "ɣ ʁ", "s": "s s̪", "sh": "ʃ ɕ ʂ", "z": "z", "zh": "ʒ", "ch": "tʃ tɕ tʃʰ", "j": "dʒ dʑ ɟ", "f": "f", "w": "w v ʋ", "y": "j",
       "h": "h ħ", "l": "l ɫ ɭ", "ḷ": "ɬ l", "m": "m", "n": "n ŋ ɳ ɲ", "r": "r ɾ ɹ", "ṛ": "ɽ ɾ r"}
NAMES = list(CLS); CID = {c: i + 1 for i, c in enumerate(NAMES)}; PIDS = [[vocab[p] for p in CLS[c].split() if p in vocab] for c in NAMES]
VOW = {"a": ("A", 0), "ā": ("A", 1), "i": ("I", 0), "ī": ("I", 1), "u": ("U", 0), "ū": ("U", 1), "e": ("E", 1), "o": ("O", 1)}
VC = {CID[c] for c in "AIUEO"}
def parse(rom):
    """-> (class ids, length flag per token: 0 short, 1 long, None consonant); doubled letters collapse to one token"""
    r = rom.lower(); out = []; i = 0
    while i < len(r):
        t = r[i:i + 2] if r[i:i + 2] in ("kh", "gh", "sh", "ch", "zh") else r[i]; i += len(t)
        c, ln = VOW[t] if t in VOW else (t, None)
        if c not in CID: continue
        if out and out[-1][0] == CID[c]: out[-1] = (CID[c], max(out[-1][1] or 0, ln or 0) if ln is not None else None); continue
        out.append((CID[c], ln))
    return tuple(c for c, _ in out), [ln for _, ln in out]
def variants(key):   # one vowel swapped for another quality
    return [key[:i] + (v,) + key[i + 1:] for i, c in enumerate(key) if c in VC for v in sorted(VC) if v != c and (i == 0 or key[i - 1] != v) and (i + 1 == len(key) or key[i + 1] != v)]
def show(key): return " ".join(NAMES[c - 1] for c in key)

# --- recordings of each word
low = [it for it in items if it["conf"] == "low"]
high = [it for it in items if it["conf"] == "high" and any(c in VC for c in parse(it["r"])[0]) and len(parse(it["r"])[0]) >= 2][:a.high_words]
need = {it["w"]: a.n for it in low}; need.update({it["w"]: a.n_high for it in high}); want = {w: [] for w in need}
STRIP = '۔،؟؛!.,?:"'; rows = []
for l in open(ROOT / "data/tts/multi2/train.jsonl", encoding="utf-8"):
    r = json.loads(l); r["words"] = [w for w in (w.strip(STRIP) for w in normalize(r["text"]).split()) if w]; rows.append(r)
rank = {"common_voice": 0, "omnilingual": 1, "brahuispeech": 2}
rows.sort(key=lambda r: (rank.get(r["source"], 3), -(r.get("ovrl") or 0)))
for edge in (False, True):
    for r in rows:
        if not 2 <= r["duration"] <= 12 or len(r["words"]) < 3: continue
        for i, w in enumerate(r["words"]):
            if w in want and len(want[w]) < need[w] and (edge or 0 < i < len(r["words"]) - 1) and all(x[0] is not r for x in want[w]): want[w].append((r, i))
utts = {}
for ex in want.values():
    for r, i in ex: utts.setdefault(r["id"], r)
print(len(low), "low words,", len(high), "high words,", len(utts), "utterances", flush=True)

dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
proc = AutoProcessor.from_pretrained(ROOT / "runs/ctc1b_brh_s3/final"); asr = Wav2Vec2ForCTC.from_pretrained(ROOT / "runs/ctc1b_brh_s3/final").to(dev).eval()
tok = proc.tokenizer; blank = asr.config.pad_token_id; delim = tok.convert_tokens_to_ids(tok.word_delimiter_token); unk = tok.unk_token_id
fe = Wav2Vec2FeatureExtractor.from_pretrained(PM); pm = Wav2Vec2ForCTC.from_pretrained(PM).to(dev).eval(); pblank = vocab["<pad>"]
spans = {}; audio = {}; emis = {}
for k, (uid, r) in enumerate(utts.items()):
    y, _ = librosa.load(ROOT / r["audio"], sr=16000, mono=True, offset=r.get("start", 0.0), duration=(r["end"] - r.get("start", 0.0)) if "end" in r else None)
    ids = []; wi = []
    for j, w in enumerate(r["words"]):
        t = [x for x in tok.convert_tokens_to_ids(list(w)) if x != unk]
        if not t: ids = None; break
        if ids: ids.append(delim); wi.append(-1)
        ids += t; wi += [j] * len(t)
    if not ids: continue
    with torch.no_grad():
        lp = torch.log_softmax(asr(proc(y, sampling_rate=16000, return_tensors="pt").input_values.to(dev)).logits.float(), -1).cpu()
        pl = torch.log_softmax(pm(fe(y, sampling_rate=16000, return_tensors="pt").input_values.to(dev)).logits.float(), -1).cpu()[0]
    if lp.shape[1] < len(ids): continue
    try: al, _ = torchaudio.functional.forced_align(lp, torch.tensor([ids]), blank=blank)
    except Exception: continue
    al = al[0].tolist(); sp = {}; p = -1; prev = blank
    for f, t in enumerate(al):
        if t != blank and t != prev: p += 1
        prev = t
        if t != blank and wi[p] >= 0: s = sp.setdefault(wi[p], [f, f]); s[1] = f
    spans[uid] = sp; audio[uid] = y
    emis[uid] = torch.cat([pl[:, pblank:pblank + 1]] + [torch.logsumexp(pl[:, ix], 1, keepdim=True) for ix in PIDS], 1)   # blank + one column per class
    if (k + 1) % 100 == 0: print(k + 1, flush=True)
del asr, pm

def analyse(r, i, keys):
    """one recording of a word -> per candidate key its score; for the best one, the vowel segments (class, seconds, F1, F2)"""
    sp = spans.get(r["id"])
    if not sp or i not in sp: return None
    E = emis[r["id"]]; f0 = max(0, sp[i][0] - 3); f1 = min(E.shape[0], sp[i][1] + 4); seg = E[f0:f1]; sc = {}; best = None
    for key in keys:
        if seg.shape[0] < len(key): continue
        try: al, s = torchaudio.functional.forced_align(seg[None], torch.tensor([list(key)]), blank=0)
        except Exception: continue
        sc[key] = s[0].sum().item() / seg.shape[0]
        if best is None or sc[key] > best[0]: best = (sc[key], key, al[0].tolist())
    if not best: return None
    _, key, al = best; on = []; prev = 0
    for f, t in enumerate(al):
        if t != 0 and t != prev: on.append(f)
        prev = t
    last = max(f for f, t in enumerate(al) if t != 0) + 1; vs = []; y = audio[r["id"]]
    for p, c in enumerate(key):
        if c not in VC or p >= len(on): continue
        s = (f0 + on[p]) * 0.02; e = (f0 + (on[p + 1] if p + 1 < len(on) else last)) * 0.02 + 0.01; mid = (s + e) / 2; F = (None, None)
        try:
            w = y[max(0, int((mid - 0.06) * 16000)): int((mid + 0.06) * 16000)]
            fm = parselmouth.Sound(w.astype(np.float64), 16000).to_formant_burg(maximum_formant=5200.0)
            F = (fm.get_value_at_time(1, 0.06), fm.get_value_at_time(2, 0.06))
        except Exception: pass
        vs.append({"p": p, "c": c, "dur": e - s, "F": F, "final": p == len(key) - 1})
    return sc, key, vs

# --- calibration on high-confidence words: does the true vowel pattern beat its wrong-vowel variants?
cal = []; inst_ok = inst_n = 0; fpts = collections.defaultdict(list); durs = []
for it in high:
    key, lens = parse(it["r"]); keys = [key] + variants(key); wins = collections.Counter()
    for r, i in want[it["w"]]:
        res = analyse(r, i, keys)
        if not res: continue
        sc, bk, _ = res; wins[bk] += 1; inst_n += 1; inst_ok += bk == key
        res = analyse(r, i, [key])                       # segments under the true spelling, for the formant and duration references
        for v in res[2] if res else []:
            if v["F"][0] and v["F"][1] and not np.isnan(v["F"][0]) and not np.isnan(v["F"][1]): fpts[v["c"]].append((np.log(v["F"][0]), np.log(v["F"][1]), it["w"]))
            if lens[v["p"]] is not None and v["c"] in (CID["A"], CID["I"], CID["U"]): durs.append((lens[v["p"]], v["dur"], v["final"]))
    if wins: cal.append({"w": it["w"], "r": it["r"], "ok": wins.most_common(1)[0][0] == key, "share": wins[key] / sum(wins.values()), "n": sum(wins.values())})
cent = {c: (statistics.median(p[0] for p in pts), statistics.median(p[1] for p in pts)) for c, pts in fpts.items() if len(pts) >= 8}
sd = (np.std([p[0] for pts in fpts.values() for p in pts]) + 1e-9, np.std([p[1] for pts in fpts.values() for p in pts]) + 1e-9)
def fclass(F):
    if not F[0] or not F[1] or np.isnan(F[0]) or np.isnan(F[1]): return None
    x = (np.log(F[0]), np.log(F[1])); return min(cent, key=lambda c: ((x[0] - cent[c][0]) / sd[0]) ** 2 + ((x[1] - cent[c][1]) / sd[1]) ** 2)
f_ok = sum(fclass((np.exp(p[0]), np.exp(p[1]))) == c for c, pts in fpts.items() for p in pts); f_n = sum(len(p) for p in fpts.values())
def auc(pos, neg): return sum((x > y) + 0.5 * (x == y) for x in pos for y in neg) / max(1, len(pos) * len(neg))
dcal = {}
for fin in (False, True):
    lo = [d for ln, d, f in durs if ln == 0 and f == fin]; hi = [d for ln, d, f in durs if ln == 1 and f == fin]
    if len(lo) >= 10 and len(hi) >= 10: dcal[fin] = {"short_med": statistics.median(lo), "long_med": statistics.median(hi), "auc": auc(hi, lo), "n": (len(lo), len(hi))}
calib = {"phone_model": {"words": len(cal), "word_majority_correct": sum(c["ok"] for c in cal), "recordings": inst_n, "recordings_correct": inst_ok},
         "formants": {"vowels": f_n, "correct_resubstitution": f_ok, "centroids_hz": {NAMES[c - 1]: [round(float(np.exp(v[0]))), round(float(np.exp(v[1])))] for c, v in cent.items()}},
         "duration": {("final" if k else "medial"): {**v, "short_med": round(v["short_med"], 3), "long_med": round(v["long_med"], 3), "auc": round(v["auc"], 2)} for k, v in dcal.items()}}
print(json.dumps(calib, ensure_ascii=False, indent=1), flush=True)

# --- low-confidence words
out = []
for it in low:
    spell = [it["r"]] + [x for x in it.get("alt", []) if x != it["r"]]; bykey = collections.OrderedDict()
    for s in spell:
        k, _ = parse(s)
        if k: bykey.setdefault(k, []).append(s)
    k0 = parse(it["r"])[0]
    if not k0 or not any(c in VC for k in bykey for c in k): out.append({"w": it["w"], "r": it["r"], "how": it["how"], "c": it["c"], "verdict": "no vowel to test"}); continue
    listed = list(bykey); extra = [k for k in variants(k0) if k not in bykey]
    wl = collections.Counter(); wa = collections.Counter(); fv = collections.Counter(); ld = collections.defaultdict(list); n = 0
    for r, i in want[it["w"]]:
        res = analyse(r, i, listed)
        if not res: continue
        n += 1; wl[res[1]] += 1
        seg = {v["p"]: v for v in res[2]}
        fk = tuple((fclass(seg[p]["F"]) if p in seg else None) if c in VC else c for p, c in enumerate(res[1]))   # same consonants, vowels as the formants classify them
        if None not in fk: fv[fk] += 1
        for v in res[2]: ld[(res[1], v["p"])].append(v["dur"])
        ra = analyse(r, i, listed + extra)
        if ra: wa[ra[1]] += 1
    if not n: out.append({"w": it["w"], "r": it["r"], "how": it["how"], "c": it["c"], "verdict": "no aligned recording"}); continue
    top, tn = wl.most_common(1)[0]; atop, an = wa.most_common(1)[0]
    row = {"w": it["w"], "r": it["r"], "how": it["how"], "c": it["c"], "en": it.get("en", ""), "n": n, "candidates": {" / ".join(bykey[k]): wl[k] for k in listed},
           "phone_pick": bykey[top], "phone_share": round(tn / n, 2), "open_pick": show(atop) if atop not in bykey else None, "open_share": round(an / n, 2),
           "formant_votes": {(" / ".join(bykey[k]) if k in bykey else show(k)): c for k, c in fv.most_common(3)}, "agrees_with_proposed": it["r"] in bykey[top]}
    # length: compare the proposed vowel lengths of the winning pattern's spellings with the measured durations
    notes = []
    for s in bykey[top]:
        _, lens = parse(s)
        for p, ln in enumerate(lens):
            d = ld.get((top, p)); cal_d = dcal.get(p == len(top) - 1)
            if ln is None or not d or not cal_d or top[p] not in (CID["A"], CID["I"], CID["U"]): continue
            med = statistics.median(d); thr = (cal_d["short_med"] + cal_d["long_med"]) / 2
            notes.append({"spelling": s, "vowel": NAMES[top[p] - 1], "written": "long" if ln else "short", "median_ms": round(med * 1000), "threshold_ms": round(thr * 1000), "heard": "long" if med > thr else "short"})
    row["length"] = notes; out.append(row)
    print(it["w"], it["r"], "| phone:", row["candidates"], "| open:", row["open_pick"], row["open_share"], "| formants:", row["formant_votes"], "|", [(x["spelling"], x["vowel"], x["written"], x["median_ms"]) for x in notes], flush=True)
json.dump({"calibration": calib, "calibration_words": cal, "words": out}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("->", OUT)
