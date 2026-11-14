"""Paired bootstrap comparison of two MT variants on the same cases.

Usage: uv run python evals/mt/compare.py baseline v1 [--split test] [--n 1000]
Prints per-direction corpus chrF++ for A and B, the delta, and a 95% bootstrap CI on the delta
(resampling cases with replacement, paired). A CI excluding 0 means the change is outside noise.
"""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path

import numpy as np
from sacrebleu.metrics import CHRF

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brahui.text import normalize  # noqa: E402

HC = ROOT / "evals" / "mt" / "hillclimb"
chrf = CHRF(word_order=2)


def load(variant):
    rows = [json.loads(l) for l in open(HC / variant / "results.jsonl", encoding="utf-8")]
    return {(r["prompt_id"], r["rep"]): r for r in rows}


def prep(direction, r):
    h, f = r["meta"]["hyp"], r["meta"]["ref"]
    return (normalize(h), normalize(f)) if direction == "en2brh" else (h, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--split", default="test"); ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    print(f"{'direction':8} {'n':>5} {a.a:>10} {a.b:>10} {'delta':>7}  95% CI (paired bootstrap)")
    for d in ["en2brh", "brh2en"]:
        keys = [k for k, r in A.items() if r["tags"][0] == d and r["tags"][1] == a.split and k in B]
        if not keys:
            continue
        ha, fa = zip(*(prep(d, A[k]) for k in keys)); hb, _ = zip(*(prep(d, B[k]) for k in keys))
        ha, hb, fa = list(ha), list(hb), list(fa)
        sa, sb = chrf.corpus_score(ha, [fa]).score, chrf.corpus_score(hb, [fa]).score
        rng = random.Random(0); n = len(keys); deltas = []
        for _ in range(a.n):
            idx = [rng.randrange(n) for _ in range(n)]
            deltas.append(chrf.corpus_score([hb[i] for i in idx], [[fa[i] for i in idx]]).score
                          - chrf.corpus_score([ha[i] for i in idx], [[fa[i] for i in idx]]).score)
        lo, hi = np.percentile(deltas, 2.5), np.percentile(deltas, 97.5)
        flag = "" if lo <= 0 <= hi else "  <-- outside noise"
        print(f"{d:8} {n:5d} {sa:10.2f} {sb:10.2f} {sb - sa:+7.2f}  [{lo:+.2f}, {hi:+.2f}]{flag}")


if __name__ == "__main__":
    main()
