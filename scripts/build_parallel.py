"""Build a synthetic Brahui->English parallel corpus by translating all real Brahui text we have with Claude
(via `claude -p`). The Brahui side is genuine (human transcripts, or ASR of media at confidence >= 0.85); the English
side is machine-made. Used as back-translation data for en->brh and distillation data for brh->en (scripts/train_nllb.py).

Units: ASR training segments as they are (BrahuiSpeech clips, CV sentences, Omnilingual 20 s chunks, media chunks);
long texts (> --max-words) are split at sentence punctuation. Sealed test sentences are excluded. Few-shot examples
come from the gold train pools exactly as in evals/mt/run_eval.py.

Usage:
  uv run python scripts/build_parallel.py [--model opus] [--batch 20] [--concurrency 4] [--limit N] [--only SOURCE]
Outputs (resumable by id):
  data/parallel/synthetic_brh_en.jsonl   {id, source, brh, en, model}
  data/parallel/gold_train.jsonl         human pairs from smol_sent_train, smol_doc_train, bouquet_sent_dev
"""
from __future__ import annotations

import argparse, glob, json, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evals" / "mt"))
from brahui.text import normalize  # noqa: E402
from run_eval import load_cases, retrieve, build_user_message, call_claude, SEED  # noqa: E402

MAN, EVAL, OUT = ROOT / "data" / "manifests", ROOT / "data" / "eval", ROOT / "data" / "parallel"
SYSTEM = (ROOT / "evals/mt/prompts/system_v0.md").read_text(encoding="utf-8") + (
    "\n\nThe Brahui input may be a transcript of speech (news, talk shows, read sentences): expect spoken style, "
    "spelling variation and occasional recognition errors. Translate the meaning faithfully and completely; do not "
    "summarize, do not add commentary, and if a passage is partly unintelligible translate the intelligible part.")
SENT_SPLIT = re.compile(r"(?<=[۔؟!\.\?])\s+")


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def split_long(text, max_words):
    words = text.split()
    if len(words) <= max_words:
        return [text]
    parts, cur = [], []
    for s in SENT_SPLIT.split(text):
        if cur and len(" ".join(cur).split()) + len(s.split()) > max_words:
            parts.append(" ".join(cur)); cur = []
        cur.append(s)
    if cur:
        parts.append(" ".join(cur))
    # fall back to hard splits for sentences without punctuation
    out = []
    for p in parts:
        w = p.split()
        out += [" ".join(w[i:i + max_words]) for i in range(0, len(w), max_words)] if len(w) > max_words * 1.5 else [p]
    return out


def collect_units(max_words):
    sealed = set()
    for f, k in [("cv_test.jsonl", "text"), ("omni_test.jsonl", "text"), ("smol_heldout.jsonl", "tgt"), ("bouquet_test.jsonl", "src")]:
        sealed |= {normalize(r[k]) for r in read_jsonl(EVAL / f)}
    units, seen = [], set()
    def add(uid, source, text):
        t = normalize(text)
        if not t or t in seen or t in sealed:
            return
        seen.add(t)
        for j, part in enumerate(split_long(t, max_words)):
            units.append({"id": uid if j == 0 and len(part) == len(t) else f"{uid}#{j}", "source": source, "brh": part})
    for f, source in [("brahuispeech_train_norm.jsonl", "brahuispeech"), ("cv_train.jsonl", "cv"), ("cv_dev.jsonl", "cv"),
                      ("omni_train_chunks.jsonl", "omni"), ("omni_dev.jsonl", "omni")]:
        for r in read_jsonl(MAN / f):
            add(r["id"], source, r["text"])
    for f in sorted(glob.glob(str(ROOT / "data/media/transcripts/*/*.jsonl"))):
        for r in read_jsonl(f):
            if r["conf"] >= 0.85 and r["text"]:
                add(f"media_{r['source']}_{r['id']}_{r['seg']}", "media", r["text"])
    return units


def write_gold():
    rows = []
    for f, s, t in [("smol_sent_train.jsonl", "src", "tgt"), ("smol_doc_train.jsonl", "src", "tgt")]:
        rows += [{"id": r["id"], "source": "smol", "brh": normalize(r[t]), "en": r[s]} for r in read_jsonl(MAN / f)]
    rows += [{"id": r["id"], "source": "bouquet", "brh": normalize(r["src"]), "en": r["tgt"]} for r in read_jsonl(MAN / "bouquet_sent_dev.jsonl")]
    (OUT / "gold_train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="opus"); ap.add_argument("--fallback-model", default="opus")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--batch", type=int, default=20); ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-words", type=int, default=60); ap.add_argument("--limit", type=int); ap.add_argument("--only")
    ap.add_argument("--shots-each", type=int, default=2); ap.add_argument("--shots-fixed", type=int, default=10)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("gold pairs:", write_gold(), flush=True)
    units = collect_units(a.max_words)
    if a.only:
        units = [u for u in units if u["source"] == a.only]
    res = OUT / "synthetic_brh_en.jsonl"
    done = {r["id"] for r in read_jsonl(res)} if res.exists() else set()
    todo = [u for u in units if u["id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    _, pools = load_cases()
    pool = pools["brh2en"]
    print(f"units={len(units)} done={len(done)} todo={len(todo)} words_todo={sum(len(u['brh'].split()) for u in todo)} model={a.model}", flush=True)
    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]

    def run(batch):
        rng = random.Random(f"{SEED}-{batch[0]['id']}")
        shots = retrieve(pool, "brh", [u["brh"] for u in batch], set(), a.shots_each, a.shots_fixed, rng)
        msg = build_user_message("brh2en", [{"src": u["brh"]} for u in batch], shots)
        err = None; model = a.model
        for attempt in range(4):
            try:
                trans, usage, model_used, lat = call_claude(SYSTEM, msg, model, a.effort)
                if len(trans) != len(batch):
                    raise RuntimeError(f"{len(trans)} translations for {len(batch)}")
                return [{**u, "en": t.strip(), "model": model_used} for u, t in zip(batch, trans)]
            except Exception as e:
                err = repr(e)[:300]
                if "safeguards flagged" in str(e) and model != a.fallback_model:
                    model = a.fallback_model; continue  # Fable's filter trips on some drama dialogue; Opus takes it
                time.sleep(10 * (attempt + 1))
        print("  batch failed:", batch[0]["id"], err, flush=True)
        return []

    t0 = time.time(); n = 0; written = 0
    with open(res, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.concurrency) as ex:
        for fut in as_completed([ex.submit(run, b) for b in batches]):
            rows = fut.result()
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush(); n += 1; written += len(rows)
            if n % 10 == 0 or n == len(batches):
                el = time.time() - t0
                print(f"  batches {n}/{len(batches)} rows={written} {el/60:.1f} min, eta {el/n*(len(batches)-n)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
