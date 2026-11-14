"""Frame span of every word in the vowel-model data, by forced alignment with the s3 ASR (Arabic script, 20 ms frames).
scripts/vowel_asr.py uses the spans to blank out the words whose spelling is hidden, and to cut scoring windows.

  python scripts/vowel_align.py   ->  data/tts/vowel_asr/spans.json  {utterance id: [[first frame, last frame] or null per word]}
"""
import json, sys, time
from pathlib import Path
import soundfile as sf, torch, torchaudio
torch.set_num_threads(4)
from transformers import Wav2Vec2ForCTC
ROOT = Path(__file__).resolve().parent.parent; D = ROOT / "data/tts/vowel_asr"; M = ROOT / "runs/ctc1b_brh_s3/final"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
vocab = json.load(open(M / "vocab.json", encoding="utf-8")); cfg = json.load(open(M / "tokenizer_config.json", encoding="utf-8"))
delim = vocab[cfg.get("word_delimiter_token", "|")]; model = Wav2Vec2ForCTC.from_pretrained(M).to(dev).eval(); blank = model.config.pad_token_id
rows = [json.loads(l) for n in ("train", "dev") for l in open(D / f"{n}.jsonl", encoding="utf-8")]
if limit: rows = rows[:limit]
order = sorted(range(len(rows)), key=lambda i: rows[i]["duration"]); out = {}; t0 = time.time(); bad = 0
def audio(r):
    x, sr = sf.read(ROOT / r["audio"], dtype="float32"); x = torch.from_numpy(x); return (x - x.mean()) / (x.std() + 1e-7)
b = 0
while b < len(order):
    ix = [order[b]]
    while b + len(ix) < len(order) and rows[order[b + len(ix)]]["duration"] * (len(ix) + 1) <= 240: ix.append(order[b + len(ix)])
    b += len(ix); xs = [audio(rows[i]) for i in ix]; n = torch.tensor([len(x) for x in xs]); x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        lg = model(input_values=x.to(dev), attention_mask=(torch.arange(x.shape[1])[None] < n[:, None]).long().to(dev)).logits
    lp = torch.log_softmax(lg.float(), -1).cpu(); il = model._get_feat_extract_output_lengths(n)
    for k, i in enumerate(ix):
        r = rows[i]; ids = []; wi = []; sp = [None] * len(r["words"])
        for j, (w, _, _) in enumerate(r["words"]):
            t = [vocab[c] for c in w if c in vocab]
            if not t: continue                       # a word with no known letter gets no span
            if ids: ids.append(delim); wi.append(-1)
            ids += t; wi += [j] * len(t)
        e = lp[k, :il[k]]
        try:
            assert ids and e.shape[0] >= len(ids)
            cols = [blank] + sorted(set(ids) - {blank}); cm = {c: n for n, c in enumerate(cols)}   # only the columns this utterance uses: 10k classes made the aligner crawl
            al, _ = torchaudio.functional.forced_align(e[:, cols][None], torch.tensor([[cm[t] for t in ids]]), blank=0); al = [cols[t] for t in al[0].tolist()]; p = -1; prev = blank
            for f, t in enumerate(al):
                if t != blank and t != prev: p += 1
                prev = t
                if t != blank and wi[p] >= 0:
                    if sp[wi[p]] is None: sp[wi[p]] = [f, f]
                    sp[wi[p]][1] = f
        except Exception: bad += 1; sp = None
        out[r["id"]] = {"T": int(il[k]), "spans": sp}
    if len(out) % 500 < len(ix): print(f"{len(out)}/{len(rows)} [{time.time() - t0:.0f} s, {bad} failed]", flush=True)
json.dump(out, open(D / "spans.json", "w"))
print("done;", len(out), "utterances,", bad, "failed,", sum(s is None for v in out.values() if v["spans"] for s in v["spans"]), "words without a span")
