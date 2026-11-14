"""Controls and per-source breakdown for ASR runs on the sealed media set (data/eval/media_test.jsonl).

Usage: uv run python evals/asr/media_controls.py <tag> [<tag> ...]      (tags under evals/asr/runs/)
Per tag: CER overall with a bootstrap CI, CER by source, CER on chunks whose label the LLM changed vs kept,
and the off-by-one control (each hypothesis scored against the previous sampled chunk's label from the same file;
a label set that carries chunk-specific information scores far lower here). With two or more tags, the paired
bootstrap of each later tag minus the first.
"""
from __future__ import annotations

import json, random, sys
from collections import defaultdict
from pathlib import Path

import jiwer

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "evals" / "asr" / "runs"


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def cer(refs, hyps):
    return 100 * jiwer.cer([r or " " for r in refs], [h or " " for h in hyps])


def boot(fn, n, k=500, seed=0):
    rng = random.Random(seed); xs = []
    for _ in range(k):
        xs.append(fn([rng.randrange(n) for _ in range(n)]))
    xs.sort(); return xs[int(0.025 * k)], xs[int(0.975 * k)]


def main():
    tags = sys.argv[1:]
    meta = {r["id"]: r for r in rj(ROOT / "data/eval/media_test.jsonl")}
    order = [r["id"] for r in rj(ROOT / "data/eval/media_test.jsonl")]
    base = None
    for tag in tags:
        res = {r["id"]: r for r in rj(RUNS / tag / "results.jsonl")}
        ids = [i for i in order if i in res]
        refs = [res[i]["ref_norm"] for i in ids]; hyps = [res[i]["hyp_norm"] for i in ids]; n = len(ids)
        lo, hi = boot(lambda idx: cer([refs[j] for j in idx], [hyps[j] for j in idx]), n)
        print(f"\n== {tag}: n={n} CER {cer(refs, hyps):.2f} [{lo:.2f}, {hi:.2f}]")
        # off-by-one: previous chunk of the same file
        prev = {}
        for a, b in zip(ids, ids[1:]):
            if meta[a]["file" if "file" in meta[a] else "source"] == meta[b]["file" if "file" in meta[b] else "source"]:
                prev[b] = a
        pairs = [(res[prev[i]]["ref_norm"], res[i]["hyp_norm"]) for i in ids if i in prev]
        print(f"   off-by-one control: CER {cer([p[0] for p in pairs], [p[1] for p in pairs]):.2f} over {len(pairs)} pairs")
        by = defaultdict(list)
        for i in ids:
            by[meta[i]["source"]].append(i); by["label=" + meta[i].get("label", "?")].append(i)
        for k in sorted(by):
            g = by[k]; print(f"   {k:32} n={len(g):3d} CER {cer([res[i]['ref_norm'] for i in g], [res[i]['hyp_norm'] for i in g]):6.2f}")
        if base is None:
            base = (tag, res)
        else:
            btag, bres = base; common = [i for i in ids if i in bres]
            r = [res[i]["ref_norm"] for i in common]; ha = [bres[i]["hyp_norm"] for i in common]; hb = [res[i]["hyp_norm"] for i in common]
            lo, hi = boot(lambda idx: cer([r[j] for j in idx], [hb[j] for j in idx]) - cer([r[j] for j in idx], [ha[j] for j in idx]), len(common))
            print(f"   paired {tag} − {btag}: {cer(r, hb) - cer(r, ha):+.2f} CER, 95% CI [{lo:+.2f}, {hi:+.2f}]")


if __name__ == "__main__":
    main()
