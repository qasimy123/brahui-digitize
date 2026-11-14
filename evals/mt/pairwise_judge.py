"""Pairwise outside-judge comparison of two systems' outputs on the same sentences (random A/B order per item).

Usage: uv run python evals/mt/pairwise_judge.py <tagA> <tagB> --metric en2brh_dev [--model opus] [--batch 10]
Reads evals/mt/nllb/<tag>/<metric>.jsonl. For each sentence the judge sees source, reference and both hypotheses in a
random order and returns the winner on meaning preservation first, naturalness second, or "tie", with evidence.
Prints wins/ties and a two-sided sign-test p-value; writes evals/mt/nllb/<tagB>/pairwise_vs_<tagA>_<metric>.jsonl."""
from __future__ import annotations
import argparse, json, random, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
NL = ROOT / "evals/mt/nllb"
SCHEMA = json.dumps({"type": "object", "properties": {"verdicts": {"type": "array", "items": {"type": "object", "properties": {
    "i": {"type": "integer"}, "winner": {"type": "string", "enum": ["1", "2", "tie"]}, "evidence": {"type": "string"}},
    "required": ["i", "winner", "evidence"], "additionalProperties": False}}}, "required": ["verdicts"], "additionalProperties": False})


def rj(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--metric", default="en2brh_dev")
    ap.add_argument("--model", default="opus"); ap.add_argument("--batch", type=int, default=10); ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()
    A, B = rj(NL / a.a / f"{a.metric}.jsonl"), rj(NL / a.b / f"{a.metric}.jsonl")
    ids = sorted(set(A) & set(B)); rng = random.Random(7)
    items = []
    for i in ids:
        flip = rng.random() < 0.5
        items.append({"id": i, "src": A[i]["src"], "ref": A[i]["ref"], "h1": B[i]["hyp"] if flip else A[i]["hyp"], "h2": A[i]["hyp"] if flip else B[i]["hyp"], "flip": flip})
    tgt = "Brahui" if a.metric.startswith("en2brh") else "English"
    system = (f"You are a bilingual reviewer comparing two machine translations into {tgt} of the same source sentence. A human reference "
              f"translation is given to help you judge meaning; the reference's exact wording is NOT the standard, meaning is. For each item say "
              f"which translation better preserves the source meaning (missing, added or wrong content decides first); if both preserve meaning "
              f"equally, prefer the more natural, idiomatic {tgt}" + (" with everyday Brahui vocabulary over Urdu loans where Brahui has its own word" if tgt == "Brahui" else "") +
              f"; if you cannot separate them, answer tie. Give one short evidence span. Return JSON {{\"verdicts\": [...]}} with winner \"1\", \"2\" or \"tie\" per item.")
    batches = [items[i:i + a.batch] for i in range(0, len(items), a.batch)]

    def run(batch):
        lines = []
        for k, it in enumerate(batch):
            lines += [f"### Item {k}", f"Source: {it['src']}", f"Reference: {it['ref']}", f"Translation 1: {it['h1']}", f"Translation 2: {it['h2']}", ""]
        cmd = ["claude", "-p", "--model", a.model, "--system-prompt", system, "--tools", "", "--restricted", "--no-session-persistence",
               "--output-format", "json", "--json-schema", SCHEMA]
        for _ in range(3):
            try:
                p = subprocess.run(cmd, input="\n".join(lines), capture_output=True, text=True, timeout=900)
                if p.returncode != 0:
                    raise RuntimeError(p.stderr[-200:] + p.stdout[-200:])
                d = json.loads(p.stdout); out = d.get("structured_output") or json.loads(d.get("result", ""))
                vs = {v["i"]: v for v in out["verdicts"]}; res = []
                for k, it in enumerate(batch):
                    if k not in vs:
                        continue
                    w = vs[k]["winner"]
                    win = "tie" if w == "tie" else ("B" if (w == "1") == it["flip"] else "A")
                    res.append({"id": it["id"], "winner": win, "evidence": vs[k]["evidence"]})
                return res
            except Exception as e:
                print("batch failed:", str(e)[:200], flush=True); time.sleep(5)
        return []

    with ThreadPoolExecutor(a.concurrency) as ex:
        res = [v for part in ex.map(run, batches) for v in part]
    out = NL / a.b / f"pairwise_vs_{a.a}_{a.metric}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for v in res:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    wa, wb, t = sum(v["winner"] == "A" for v in res), sum(v["winner"] == "B" for v in res), sum(v["winner"] == "tie" for v in res)
    n = wa + wb; p = sum(comb(n, k) for k in range(0, min(wa, wb) + 1)) / 2 ** n * 2 if n else 1.0
    print(f"{a.metric}: {a.a} wins {wa}, {a.b} wins {wb}, ties {t} (n={len(res)}); sign test p = {min(1, p):.3f}")


if __name__ == "__main__":
    main()
