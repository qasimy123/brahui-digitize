"""Bootstrap CIs and paired comparisons for NLLB runs from their per-sentence outputs in evals/mt/nllb/<tag>/.

Usage:
  uv run python evals/mt/compare_nllb.py <tagA> [<tagB>] [--n 1000] [--metrics en2brh_smol,...]
With one tag: corpus chrF++ and a 95% bootstrap CI per metric. With two tags: the paired bootstrap of B minus A
(resampling sentences with replacement, same sample for both systems). A CI that excludes 0 is outside noise.
Tags look like nllb_brh_v3_after, nllb_r0a_dev, nllb_r0a_dev_ep3.
"""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path

from sacrebleu.metrics import CHRF

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brahui.text import normalize  # noqa: E402

NL = ROOT / "evals" / "mt" / "nllb"
chrf = CHRF(word_order=2)
LEGACY = {"en2brh": "en2brh_smol", "brh2en": "brh2en_bouquet"}  # pre-2026-10-19 file names


def load(tag, metric):
    p = NL / tag / f"{metric}.jsonl"
    if not p.exists():
        legacy = {v: k for k, v in LEGACY.items()}.get(metric)
        if legacy and (NL / tag / f"{legacy}.jsonl").exists():
            p = NL / tag / f"{legacy}.jsonl"
        else:
            return None
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    brh = metric.startswith("en2brh")
    f = normalize if brh else (lambda s: s)
    refkey = "src" if metric.startswith("roundtrip") else "ref"
    return {r["id"]: (f(r["hyp"]), f(r[refkey])) for r in rows}


def corpus(pairs, idx):
    return chrf.corpus_score([pairs[i][0] for i in idx], [[pairs[i][1] for i in idx]]).score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b", nargs="?"); ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--metrics", default="en2brh_smol,en2brh_bouquet,brh2en_bouquet,roundtrip_smol,roundtrip_bouquet,en2brh_dev,brh2en_dev,roundtrip_dev")
    ap.add_argument("--bmetrics", default=None, help="file names under tag B, positionally matching --metrics (e.g. MBR outputs en2brh_dev_mix)")
    a = ap.parse_args()
    rng = random.Random(0)
    print(f"{'metric':18} {'n':>5} {a.a[:18]:>18} {(a.b or '')[:18]:>18} {'delta':>7}  95% CI")
    bms = a.bmetrics.split(",") if a.bmetrics else a.metrics.split(",")
    for m, bm in zip(a.metrics.split(","), bms):
        A = load(a.a, m)
        if A is None:
            continue
        B = load(a.b, bm) if a.b else None
        if a.b and B is None:
            continue
        ids = sorted(A) if B is None else sorted(set(A) & set(B))
        pa = [A[i] for i in ids]; pb = [B[i] for i in ids] if B else None
        N = len(ids); full = list(range(N))
        sa = corpus(pa, full); sb = corpus(pb, full) if pb else None
        samples = []
        for _ in range(a.n):
            idx = [rng.randrange(N) for _ in range(N)]
            samples.append(corpus(pb, idx) - corpus(pa, idx) if pb else corpus(pa, idx))
        samples.sort(); lo, hi = samples[int(0.025 * a.n)], samples[int(0.975 * a.n)]
        if pb:
            flag = "" if lo <= 0 <= hi else ("  BETTER" if lo > 0 else "  WORSE")
            print(f"{m:18} {N:5d} {sa:18.2f} {sb:18.2f} {sb - sa:+7.2f}  [{lo:+.2f}, {hi:+.2f}]{flag}")
        else:
            print(f"{m:18} {N:5d} {sa:18.2f} {'':>18} {'':>7}  [{lo:.2f}, {hi:.2f}]")


if __name__ == "__main__":
    main()
