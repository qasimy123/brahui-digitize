"""MT eval runner: Claude (via `claude -p`) translating en->brh and brh->en, graded by chrF++.

Cases (built from data/manifests + data/eval; test splits are the sealed sets):
  en->brh  train = smol_sent_train (663)   test = smol_heldout (200)        ref = Brahui, normalized
  brh->en  train = bouquet_sent_dev (504)  test = bouquet_test (854)        ref = English

The artifact under iteration is the system prompt file (--prompt). Few-shot examples are
retrieved from the *train* pools only, so sealed test references are never in the model's context.

Usage:
  uv run python evals/mt/run_eval.py --variant baseline --prompt evals/mt/prompts/system_v0.md \
      [--model sonnet] [--batch 20] [--concurrency 4] [--split train,test] [--limit-train 300] [--reps 1]
Writes evals/mt/hillclimb/<variant>/{results.jsonl,summary.json,traces/}. Resumable.
"""
from __future__ import annotations

import argparse, json, random, re, subprocess, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brahui.text import normalize  # noqa: E402

MAN, EVAL, HC = ROOT / "data" / "manifests", ROOT / "data" / "eval", ROOT / "evals" / "mt" / "hillclimb"
SEED = 20261006
LANG_NAME = {"eng_Latn": "English", "brh_Arab": "Brahui"}
SCHEMA = json.dumps({"type": "object", "properties": {"translations": {"type": "array", "items": {"type": "string"}}},
                     "required": ["translations"], "additionalProperties": False})
chrf = CHRF(word_order=2)  # chrF++
bleu = BLEU(effective_order=True)


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


# ----------------------------------------------------------------------------- cases + shot pools
def load_cases():
    cases, pools = [], {"en2brh": [], "brh2en": []}
    smol_tr = read_jsonl(MAN / "smol_sent_train.jsonl"); smol_doc = read_jsonl(MAN / "smol_doc_train.jsonl")
    smol_te = read_jsonl(EVAL / "smol_heldout.jsonl")
    bq_dev = read_jsonl(MAN / "bouquet_sent_dev.jsonl"); bq_te = read_jsonl(EVAL / "bouquet_test.jsonl")
    for r in smol_tr:
        cases.append({"id": r["id"], "dir": "en2brh", "split": "train", "src": r["src"], "ref": r["tgt"], "domain": r["domain"]})
    for r in smol_te:
        cases.append({"id": r["id"], "dir": "en2brh", "split": "test", "src": r["src"], "ref": r["tgt"], "domain": r["domain"]})
    for r in bq_dev:
        cases.append({"id": r["id"], "dir": "brh2en", "split": "train", "src": r["src"], "ref": r["tgt"], "domain": r["domain"]})
    for r in bq_te:
        cases.append({"id": r["id"], "dir": "brh2en", "split": "test", "src": r["src"], "ref": r["tgt"], "domain": r["domain"]})
    # shot pools: train data only
    for r in smol_tr + smol_doc:
        pools["en2brh"].append({"id": r["id"], "en": r["src"], "brh": r["tgt"]})
        pools["brh2en"].append({"id": r["id"], "en": r["src"], "brh": r["tgt"]})
    for r in bq_dev:
        pools["brh2en"].append({"id": r["id"], "en": r["tgt"], "brh": r["src"]})
    return cases, pools


_tok = re.compile(r"\w+", re.UNICODE)


def toks(s):
    return set(t.lower() for t in _tok.findall(normalize(s)) if len(t) > 1)


def retrieve(pool, query_key, queries: list[str], exclude: set[str], k_each: int, k_fixed: int, rng):
    """Top-k lexical-overlap shots per query, plus k_fixed random fixed shots (seeded). Dedup, keep order."""
    out, seen = [], set()
    fixed = [p for p in rng.sample(pool, min(len(pool), 200)) if p["id"] not in exclude][:k_fixed]
    for p in fixed:
        if p["id"] not in seen:
            seen.add(p["id"]); out.append(p)
    pool_toks = [(p, toks(p[query_key])) for p in pool if p["id"] not in exclude]
    for q in queries:
        qt = toks(q)
        if not qt:
            continue
        scored = sorted(pool_toks, key=lambda pt: -len(qt & pt[1]) / (len(qt | pt[1]) or 1))[:k_each]
        for p, _ in scored:
            if p["id"] not in seen:
                seen.add(p["id"]); out.append(p)
    return out


# ----------------------------------------------------------------------------- prompt + model call
def build_user_message(direction, batch, shots):
    s, t = ("en", "brh") if direction == "en2brh" else ("brh", "en")
    S, T = LANG_NAME["eng_Latn" if s == "en" else "brh_Arab"], LANG_NAME["eng_Latn" if t == "en" else "brh_Arab"]
    lines = [f"## Reference translations ({S} -> {T})", ""]
    for p in shots:
        lines += [f"{S}: {p[s]}", f"{T}: {p[t]}", ""]
    lines += [f"## Translate each of the following {len(batch)} {S} sentences into {T}.",
              "Return JSON {\"translations\": [...]} with exactly one string per input, in order. No commentary.", ""]
    for i, c in enumerate(batch, 1):
        lines.append(f"{i}. {c['src']}")
    return "\n".join(lines)


def call_claude(system_prompt: str, user_msg: str, model: str, effort: str | None, timeout=900):
    cmd = ["claude", "-p", "--model", model, "--system-prompt", system_prompt, "--tools", "", "--restricted",
           "--no-session-persistence", "--output-format", "json", "--json-schema", SCHEMA]
    if effort:
        cmd += ["--effort", effort]
    t0 = time.time()
    proc = subprocess.run(cmd, input=user_msg, capture_output=True, text=True, timeout=timeout)
    lat = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"claude exit {proc.returncode}: {proc.stderr[-500:]} stdout={proc.stdout[-800:]}")
    d = json.loads(proc.stdout)
    out = d.get("structured_output")
    if not out:
        try:
            out = json.loads(d.get("result", ""))
        except Exception:
            raise RuntimeError(f"no structured output: {str(d.get('result'))[:300]}")
    usage = d.get("usage", {})
    model_used = next(iter(d.get("modelUsage", {}).keys()), model)
    return out["translations"], {"input_tokens": usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
                                 + usage.get("cache_read_input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)}, model_used, lat


# ----------------------------------------------------------------------------- grading
def grade(direction, hyp, ref):
    if direction == "en2brh":
        h, r = normalize(hyp), normalize(ref)
    else:
        h, r = hyp.strip(), ref.strip()
    return {"chrfpp": round(chrf.sentence_score(h, [r]).score, 2), "bleu": round(bleu.sentence_score(h, [r]).score, 2)}


def corpus_scores(rows):
    out = {}
    for d in ["en2brh", "brh2en"]:
        for split in ["train", "test"]:
            rs = [r for r in rows if r["tags"][0] == d and r["tags"][1] == split]
            if not rs:
                continue
            if d == "en2brh":
                hyps = [normalize(r["meta"]["hyp"]) for r in rs]; refs = [normalize(r["meta"]["ref"]) for r in rs]
            else:
                hyps = [r["meta"]["hyp"] for r in rs]; refs = [r["meta"]["ref"] for r in rs]
            out[f"{d}/{split}"] = {"n": len(rs), "chrfpp": round(chrf.corpus_score(hyps, [refs]).score, 2),
                                   "bleu": round(bleu.corpus_score(hyps, [refs]).score, 2),
                                   "mean_sent_chrfpp": round(sum(r["grade"]["chrfpp"] for r in rs) / len(rs), 2)}
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True); ap.add_argument("--prompt", required=True)
    ap.add_argument("--prompt-brh2en", default=None, help="optional separate system prompt for brh->en (default: --prompt)")
    ap.add_argument("--model", default="sonnet"); ap.add_argument("--effort", default=None)
    ap.add_argument("--batch", type=int, default=20); ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--split", default="train,test"); ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-test", type=int, default=None)
    ap.add_argument("--shots-each", type=int, default=3); ap.add_argument("--shots-fixed", type=int, default=12)
    ap.add_argument("--reps", type=int, default=1); ap.add_argument("--description", default="")
    a = ap.parse_args()

    system_prompt = Path(a.prompt).read_text(encoding="utf-8")
    system_prompts = {"en2brh": system_prompt, "brh2en": Path(a.prompt_brh2en).read_text(encoding="utf-8") if a.prompt_brh2en else system_prompt}
    cases, pools = load_cases()
    rng = random.Random(SEED)
    splits = a.split.split(",")
    sel = []
    for d in ["en2brh", "brh2en"]:
        for split in splits:
            cs = [c for c in cases if c["dir"] == d and c["split"] == split]
            lim = a.limit_train if split == "train" else a.limit_test
            if lim:
                r2 = random.Random(SEED + 1); r2.shuffle(cs); cs = sorted(cs[:lim], key=lambda c: c["id"])
            sel += cs
    out = HC / a.variant; (out / "traces").mkdir(parents=True, exist_ok=True)
    res_path = out / "results.jsonl"
    done = {(r["prompt_id"], r["rep"]) for r in read_jsonl(res_path)} if res_path.exists() else set()
    (out / "summary.json").write_text(json.dumps({"description": a.description or f"{a.variant}: {a.prompt} model={a.model}",
                                                   "target": "system_prompt", "prompt_file": a.prompt, "prompt_file_brh2en": a.prompt_brh2en, "model": a.model,
                                                   "effort": a.effort, "shots_each": a.shots_each, "shots_fixed": a.shots_fixed}, indent=2))
    # batches: group by direction, keep train/test cases in separate batches
    jobs = []
    for rep in range(a.reps):
        for d in ["en2brh", "brh2en"]:
            for split in splits:
                cs = [c for c in sel if c["dir"] == d and c["split"] == split and (c["id"], rep) not in done]
                for i in range(0, len(cs), a.batch):
                    jobs.append((rep, d, split, cs[i:i + a.batch]))
    print(f"variant={a.variant} model={a.model} cases={len(sel)} pending_batches={len(jobs)} done_rows={len(done)}", flush=True)

    def run_job(job):
        rep, d, split, batch = job
        jr = random.Random(f"{SEED}-{d}-{split}-{batch[0]['id']}-{rep}")
        exclude = {c["id"] for c in batch}
        shots = retrieve(pools[d], "en" if d == "en2brh" else "brh", [c["src"] for c in batch], exclude, a.shots_each, a.shots_fixed, jr)
        user_msg = build_user_message(d, batch, shots)
        system_prompt = system_prompts[d]
        err = None
        for attempt in range(3):
            try:
                trans, usage, model_used, lat = call_claude(system_prompt, user_msg, a.model, a.effort)
                if len(trans) != len(batch):
                    raise RuntimeError(f"got {len(trans)} translations for {len(batch)} inputs")
                break
            except Exception as e:
                err = repr(e)[:400]; trans = None; time.sleep(5 * (attempt + 1))
        rows = []
        for i, c in enumerate(batch):
            hyp = (trans[i] if trans else "") or ""
            g = grade(d, hyp, c["ref"])
            row = {"prompt_id": c["id"], "rep": rep, "prompt": user_msg, "tags": [d, split, c["domain"]],
                   "grade": g, "model": model_used if trans else a.model, "latency_s": round(lat / len(batch), 2) if trans else None,
                   "usage": {k: v // len(batch) for k, v in usage.items()} if trans else {},
                   "meta": {"src": c["src"], "ref": c["ref"], "hyp": hyp, "batch_first": batch[0]["id"], "error": err if not trans else None}}
            rows.append(row)
            trace = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg},
                     {"role": "assistant", "content": json.dumps({"translations": trans}, ensure_ascii=False) if trans else f"ERROR: {err}"}]
            (out / "traces" / f"{c['id']}_rep{rep}.json").write_text(json.dumps(trace, ensure_ascii=False))
        return rows

    with open(res_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.concurrency) as ex:
        futs = [ex.submit(run_job, j) for j in jobs]
        n = 0
        for fut in as_completed(futs):
            for row in fut.result():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush(); n += 1
            if n % 5 == 0 or n == len(futs):
                print(f"  batches done {n}/{len(futs)}", flush=True)
    rows = read_jsonl(res_path)
    scores = corpus_scores(rows)
    errs = sum(1 for r in rows if r["meta"].get("error"))
    summ = json.loads((out / "summary.json").read_text()); summ.update({"scores": scores, "rows": len(rows), "error_rows": errs})
    (out / "summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
