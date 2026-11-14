"""Near-duplicate check of every training corpus against the sealed sets in data/eval/ (exact normalized string and
chrF > 90 fuzzy match via a char-4-gram candidate index). Writes data/parallel/eval_overlap.jsonl; scripts/train_nllb.py
excludes the listed ids. Run from the repo root after adding any corpus: uv run python scripts/dedup_against_eval.py"""
import json, sys, collections, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); import os; os.chdir(ROOT)
from brahui.text import normalize
from sacrebleu.metrics import CHRF
chrf = CHRF(word_order=2)
def rj(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def key(s): return re.sub(r"[\s\W_]+", "", normalize(s).lower())
def ngrams(s, n=4):
    s = key(s); return {s[i:i+n] for i in range(max(1, len(s)-n+1))}
evals = {}
for f, bk, ek in [("smol_heldout", "tgt", "src"), ("bouquet_test", "src", "tgt")]:
    for r in rj(f"data/eval/{f}.jsonl"):
        evals[(f, r["id"], "brh")] = r[bk]; evals[(f, r["id"], "en")] = r[ek]
for f in ["cv_test", "omni_test"]:
    for r in rj(f"data/eval/{f}.jsonl"):
        t = r.get("text") or r.get("sentence") or r.get("transcript") or ""
        if t: evals[(f, r.get("id", r.get("path")), "brh")] = t
idx = collections.defaultdict(set); exact = {}
for k, s in evals.items():
    exact.setdefault(key(s), []).append(k)
    for g in ngrams(s): idx[g].add(k)
print("eval strings", len(evals))
out = open("data/parallel/eval_overlap.jsonl", "w")
tot = collections.Counter()
import glob
corpora = sorted({"gold_train"} | {Path(p).stem for p in glob.glob("data/parallel/synthetic_*.jsonl")} | {Path(p).stem for p in glob.glob("data/parallel/en_*.jsonl")})
for f in corpora:
    rows = rj(f"data/parallel/{f}.jsonl"); n_exact = n_fuzzy = 0
    for r in rows:
        for side in ["brh", "en"]:
            if side not in r: continue
            s = r[side]; k = key(s)
            if len(k) < 15: continue
            hits = exact.get(k)
            if hits:
                n_exact += 1; out.write(json.dumps({"corpus": f, "id": r["id"], "side": side, "kind": "exact", "eval": hits[0][:2]}, ensure_ascii=False)+"\n"); continue
            g = ngrams(s); cand = collections.Counter()
            for x in g:
                for k2 in idx.get(x, ()): cand[k2] += 1
            for k2, c in cand.most_common(3):
                if c < 0.5*len(g): break
                sc = chrf.sentence_score(s, [evals[k2]]).score
                if sc > 90:
                    n_fuzzy += 1; out.write(json.dumps({"corpus": f, "id": r["id"], "side": side, "kind": "fuzzy", "chrf": round(sc,1), "eval": k2[:2]}, ensure_ascii=False)+"\n"); break
    print(f, len(rows), "exact", n_exact, "fuzzy>90", n_fuzzy, flush=True)
