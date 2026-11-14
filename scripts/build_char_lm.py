"""Character n-gram LM (interpolated Kneser-Ney) over Brahui text, for CTC beam-search decoding (scripts/ctc_lm_decode.py,
demo/web). Corpus = real rows of data/distill/brh_pool.jsonl + ASR training transcripts (+ --synthetic for the NLLB
back-translations); every sentence equal to a cv_dev / cv_test / omni_test transcript is dropped (for_scoring form).
Writes <out>.json: {"order": N, "D": discount, "chars": [...], "ngrams": {"<ctx><char>": count...}} (counts, not probs; the decoder
turns them into KN probabilities so the same file feeds Python and JS).
  uv run python scripts/build_char_lm.py --order 6 --out data/lm/brh_char6"""
import argparse, collections, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from brahui.text import for_scoring  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--order", type=int, default=6); ap.add_argument("--out", required=True)
ap.add_argument("--synthetic", action="store_true", help="also use synthetic (NLLB en->brh) rows of brh_pool")
ap.add_argument("--min-count", type=int, default=1, help="drop n-grams (order>=3) seen fewer times (size control for the browser)")
a = ap.parse_args()

def rows(p, key):
    for l in open(ROOT / p, encoding="utf-8"):
        r = json.loads(l); t = r.get(key)
        if t: yield r, t

held = set()
for s in ("cv_dev", "cv_test", "omni_test"):
    for l in open(ROOT / "data" / "eval" / f"{s}.jsonl", encoding="utf-8") if (ROOT / "data" / "eval" / f"{s}.jsonl").exists() else open(ROOT / "data" / "manifests" / f"{s}.jsonl", encoding="utf-8"):
        held.add(for_scoring(json.loads(l)["text"]))
sents, src_n, dropped = [], collections.Counter(), 0
for p, key, name, keep in [("data/distill/brh_pool.jsonl", "brh", "pool", lambda r: r.get("real") or a.synthetic),
                           ("data/manifests/brahuispeech_train_norm.jsonl", "text", "brahuispeech", lambda r: True),
                           ("data/manifests/cv_train.jsonl", "text", "cv_train", lambda r: True),
                           ("data/manifests/omni_train.jsonl", "text", "omni_train", lambda r: True),
                           ("data/manifests/media_pseudo_s3.jsonl", "text", "media_pseudo", lambda r: True)]:
    for r, t in rows(p, key):
        if not keep(r): continue
        t = for_scoring(t)
        if not t: continue
        if t in held: dropped += 1; continue
        sents.append(t); src_n[name] += 1
print(f"{len(sents)} sentences, {sum(map(len, sents))/1e6:.1f} M chars, dropped {dropped} held-out matches; {dict(src_n)}")

N = a.order; cnt = collections.Counter(); BOS = "\x02"; EOS = "\x03"
for s in sents:
    x = BOS * (N - 1) + s + EOS
    for i in range(N - 1, len(x)):
        for k in range(1, N + 1):  # all orders 1..N ending at i
            cnt[x[i - k + 1:i + 1]] += 1
if a.min_count > 1:
    cnt = collections.Counter({g: c for g, c in cnt.items() if len(g) < 3 or c >= a.min_count})
chars = sorted({c for s in sents for c in s} | {EOS})
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
json.dump({"order": N, "D": 0.75, "bos": BOS, "eos": EOS, "chars": chars, "n_sent": len(sents), "ngrams": dict(cnt)}, open(str(out) + ".json", "w", encoding="utf-8"), ensure_ascii=False)
print(f"{len(cnt)} n-grams (order {N}), {len(chars)} chars -> {out}.json ({Path(str(out)+'.json').stat().st_size/1e6:.1f} MB)")
