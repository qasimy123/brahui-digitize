"""Word-by-word listening drill for the vowelled F5 voice: one sentence at a time, every word tappable (plays that word
alone, cut by forced alignment), the reviewer marks words said wrong and types how they should sound.

  uv run python scripts/make_tts_drill.py --round r1 [--n-new 16]      build round 1: the 24 blind-test sentences + new ones
  uv run python scripts/make_tts_drill.py --apply <results.json> --round r2
        corrections -> data/lexicon/vowel_overrides.json (read first by brahui.vowelize_nn.HeardVowelizer), then round r2:
        every sentence that had a problem, spoken again with the corrected spellings, plus one fresh sentence per corrected word.
Output: evals/tts/drill/<round>/drill_<round>.html (self-contained), data/tts/drill/<round>.jsonl, wavs next to the page.
Synthesis runs locally with tools/f5/.venv (runs/f5_multi2_vow/final, speed 0.75), alignment with runs/ctc1b_brh_s3/final.
"""
import argparse, base64, collections, json, random, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from brahui.text import normalize  # noqa: E402
from brahui.vowelize_nn import load, STRIP  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--round", required=True); ap.add_argument("--apply"); ap.add_argument("--n-new", type=int, default=16)
ap.add_argument("--voice", default="runs/f5_multi2_vow/final"); ap.add_argument("--refs", default="data/tts/f5/multi2_vow_refs"); ap.add_argument("--speed", default="0.75")
a = ap.parse_args()
PUNCT = str.maketrans({"۔": ".", "،": ",", "؟": "?", "؛": ",", "٪": "%", '"': None, "٭": None, "“": None, "”": None})
OV = ROOT / "data/lexicon/vowel_overrides.json"; SETS = ROOT / "data/tts/drill"; SETS.mkdir(parents=True, exist_ok=True)
def pool():
    out = []
    for name, brh, en in (("bouquet_sent_dev", "src", "tgt"), ("smol_sent_test", "tgt", "src")):
        for l in open(ROOT / "data/manifests" / f"{name}.jsonl", encoding="utf-8"):
            r = json.loads(l); t = normalize(r[brh]); n = len(t.split())
            if 5 <= n <= 13 and len(r[en]) <= 100 and not re.search(r"[A-Za-z0-9۰-۹()\[\]:؛\"'“”]", t): out.append({"text": t, "en": r[en]})
    random.Random(7).shuffle(out); return out
def core(t): return [w for w in (x.strip(STRIP) for x in normalize(t).split()) if w]
def typed(s):   # what the reviewer typed -> our spelling: doubled vowels are long
    s = s.strip().lower()
    for x, y in (("aa", "ā"), ("ee", "ī"), ("ii", "ī"), ("oo", "ū"), ("uu", "ū")): s = s.replace(x, y)
    return re.sub(r"[^a-zāīūṭḍṛḷṉ' ]", "", s).strip()
rows = []
if a.apply:
    res = json.load(open(a.apply, encoding="utf-8")); prev = {json.loads(l)["id"]: json.loads(l) for l in open(SETS / f"{res['round']}.jsonl", encoding="utf-8")}
    ov = json.load(open(OV, encoding="utf-8")) if OV.exists() else {}; fixed = {}; nofix = []; tags = collections.Counter(); again = []
    for s in res["sentences"]:
        bad = [w for w in s["words"] if w["status"] == "wrong"]; tags.update(t for t in s["tags"] if t != "fine")
        for w in bad:
            f = typed(w["fix"])
            if f and f != w["said"]: ov[w["word"]] = f; fixed[w["word"]] = (w["said"], f, w["tags"])
            else: nofix.append((s["n"], w["word"], w["said"], w["tags"]))
        if bad or s["note"].strip() or any(t != "fine" for t in s["tags"]): again.append((s, bad))
    json.dump(ov, open(OV, "w", encoding="utf-8"), ensure_ascii=False, indent=0, sort_keys=True)
    print(f"{len(fixed)} word corrections -> {OV} ({len(ov)} in total)")
    for w, (o, f, t) in fixed.items(): print(f"  {w}  {o} -> {f}  {','.join(t)}")
    if nofix: print("marked wrong without a usable correction:"); [print("  ", *x) for x in nofix]
    print("sentence-level problems:", dict(tags))
    for s, bad in again:
        p = prev[s["n"]]; why = "again: " + (", ".join(typed(w["fix"]) or w["said"] for w in bad) if bad else "/".join(t for t in s["tags"] if t != "fine") or "note")
        rows.append({"id": s["n"], "text": p["text"], "en": p["en"], "why": why})
    used = {r["text"] for r in prev.values()}; k = 0
    for w in fixed:     # one fresh sentence per corrected word, when the pool has one
        for p in pool():
            if p["text"] not in used and w in core(p["text"]): used.add(p["text"]); k += 1; rows.append({"id": f"{a.round.upper()}N{k:02d}", "text": p["text"], "en": p["en"], "why": "new sentence with " + fixed[w][1]}); break
    if not rows: print("nothing to drill again"); sys.exit()
else:
    base = [json.loads(l) for l in open(ROOT / "data/tts/listen_v3/set.jsonl", encoding="utf-8")]; used = {r["text"] for r in base}
    rows = [{"id": r["id"], "text": r["text"], "en": r["en"], "why": ""} for r in base]
    hv0 = load("heard"); cand = []
    for p in pool():
        if p["text"] in used: continue
        src = [hv0.explain(w)[1] for w in core(p["text"])]; g = sum(s in ("model", "guess", "loan") for s in src)
        if 1 <= g <= 3: cand.append(p)
    rows += [{"id": f"N{k + 1:02d}", "text": p["text"], "en": p["en"], "why": ""} for k, p in enumerate(cand[:a.n_new])]
hv = load("heard"); out = ROOT / "evals/tts/drill" / a.round; (out / "in").mkdir(parents=True, exist_ok=True)
for r in rows:
    r["words"] = [{"w": w, "r": hv.explain(w)[0], "src": hv.explain(w)[1]} for w in core(r["text"])]; r["vowelled"] = hv.text(r["text"]).translate(PUNCT)
with open(SETS / f"{a.round}.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(out / "in/metadata.csv", "w", encoding="utf-8") as f:
    f.write("audio_file|text\n"); [f.write(f"wavs/{r['id']}.wav|{r['vowelled']}\n") for r in rows]
print(len(rows), "sentences; sources of the spellings:", dict(collections.Counter(w["src"] for r in rows for w in r["words"])), flush=True)
for p in (out / "wavs").glob("*.wav"): p.unlink()     # spellings may have changed since an earlier build
subprocess.run([str(ROOT / "tools/f5/.venv/bin/python"), str(ROOT / "tools/f5/synth_eval.py"), "--ckpt", str(ROOT / a.voice / "model.safetensors"), "--vocab", str(ROOT / a.voice / "vocab.txt"),
                "--dev", str(out / "in"), "--refs", str(ROOT / a.refs), "--out", str(out), "--listen", "0", "--speed", a.speed], check=True)
# ---- word times: forced alignment of the Arabic-script words on the synthetic audio
import numpy as np, librosa, torch, torchaudio  # noqa: E402
from transformers import AutoProcessor, Wav2Vec2ForCTC  # noqa: E402
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
proc = AutoProcessor.from_pretrained(ROOT / "runs/ctc1b_brh_s3/final"); model = Wav2Vec2ForCTC.from_pretrained(ROOT / "runs/ctc1b_brh_s3/final").to(dev).eval()
tok = proc.tokenizer; blank = model.config.pad_token_id; delim = tok.convert_tokens_to_ids(tok.word_delimiter_token); unk = tok.unk_token_id; miss = 0
for r in rows:
    y, _ = librosa.load(out / "wavs" / f"{r['id']}.wav", sr=16000, mono=True); ids = []; wi = []
    for j, w in enumerate(r["words"]):
        t = [x for x in tok.convert_tokens_to_ids(list(w["w"])) if x != unk] or [delim]
        if ids: ids.append(delim); wi.append(-1)
        ids += t; wi += [j] * len(t)
    with torch.no_grad(): lp = torch.log_softmax(model(proc(y, sampling_rate=16000, return_tensors="pt").input_values.to(dev)).logits.float(), -1).cpu()
    sp = {}
    try:
        al = torchaudio.functional.forced_align(lp, torch.tensor([ids]), blank=blank)[0][0].tolist(); p = -1; prevt = blank
        for f, t in enumerate(al):
            if t != blank and t != prevt: p += 1
            prevt = t
            if t != blank and wi[p] >= 0: s = sp.setdefault(wi[p], [f, f]); s[1] = f
    except Exception as e: print("alignment failed", r["id"], e)
    for j, w in enumerate(r["words"]):
        w["t0"], w["t1"] = (round(sp[j][0] * 0.02, 2), round(sp[j][1] * 0.02 + 0.04, 2)) if j in sp else (None, None); miss += j not in sp
    r["audio"] = base64.b64encode(subprocess.run(["ffmpeg", "-v", "error", "-i", str(out / "wavs" / f"{r['id']}.wav"), "-ac", "1", "-b:a", "64k", "-f", "mp3", "-"], capture_output=True, check=True).stdout).decode()
items = [{"n": r["id"], "en": r["en"], "brh": r["text"], "why": r.get("why", ""), "audio": r["audio"], "words": r["words"]} for r in rows]
lede = f"{len(items)} sentences in the new voice, one at a time. Tap a word to hear it alone and say how it should sound."
html = (Path(__file__).parent / "tts_drill_template.html").read_text(encoding="utf-8").replace("/*ITEMS*/[]", json.dumps(items, ensure_ascii=False)).replace("__ROUND__", a.round).replace("__LEDE__", lede)
(out / f"drill_{a.round}.html").write_text(html, encoding="utf-8"); print(out / f"drill_{a.round}.html", round(len(html) / 1e6, 1), "MB;", miss, "words without a time")
