"""Build the native-speaker recording list: English sentences a Brahui speaker reads and speaks in Brahui.

Two halves (default 1,000 each):
  A. sentences that already have a Brahui text reference (SMOL sent/doc train, BOUQuET dev). Her spoken
     rendering gives en-text -> brh-speech pairs with a text reference, i.e. speech-translation eval data
     and TTS text/audio pairs with a check on the ASR transcript.
  B. new everyday conversational sentences generated with Claude across ~25 topics, then greedily
     selected (SMOL-style set cover) to maximize distinct-word coverage with short sentences.
Sealed sets (smol_heldout, bouquet_test) are never used.

Usage:
  uv run python scripts/build_recording_set.py [--n-ref 1000] [--n-new 1000] [--model opus] [--out data/recording/sentences.jsonl]
Resumable: generated candidates are cached in data/recording/candidates.jsonl.
"""
from __future__ import annotations

import argparse, json, random, re, subprocess, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAN, EVAL = ROOT / "data" / "manifests", ROOT / "data" / "eval"
OUT_DIR = ROOT / "data" / "recording"
SEED = 20261006

TOPICS = [
    "greetings and small talk", "family and relatives", "health, symptoms and visiting a doctor", "food, cooking and meals",
    "shopping at a market and bargaining", "travel, roads and transport", "weather and seasons", "work and jobs",
    "school and learning", "money and prices", "time, dates and appointments", "directions and places in a town",
    "home, house and chores", "animals, farming and livestock", "emotions and opinions", "phone calls and messages",
    "government offices and documents", "religion and festivals (everyday, non-doctrinal)", "sports and games",
    "clothes and weddings", "water, electricity and repairs", "emergencies and asking for help", "children and parenting",
    "stories about the past and village life", "numbers, counting and quantities", "questions a visitor asks a local",
]
STYLES = ["short statements (4-9 words)", "questions someone would ask in conversation", "requests, instructions and polite commands",
          "longer sentences with a reason or a condition (10-14 words)"]

SCHEMA = json.dumps({"type": "object", "properties": {"sentences": {"type": "array", "items": {"type": "string"}}}, "required": ["sentences"]})
SYSTEM = ("You write natural spoken English sentences for a speech-translation data collection. A bilingual speaker will read each "
          "sentence and say it in her own language (Brahui, spoken in Balochistan, Pakistan). Sentences must be: everyday and "
          "culturally plausible for rural and urban Balochistan, self-contained, unambiguous, 4 to 14 words, no proper names of "
          "people, no brand names, no numerals (write numbers as words), no quotation marks, no slang or idioms that do not translate. "
          "Vary vocabulary deliberately: avoid repeating content words across sentences. Return only JSON.")

STOP = set("""a an the and or but if so of to in on at by for with from as is are was were be been being am do does did have has had
will would can could should may might must i you he she it we they me him her us them my your his its our their this that these those
there here what which who whom whose when where why how not no yes very too also just only then than up down out over into about""".split())


def words(s: str):
    return [w for w in re.findall(r"[a-z']+", s.lower()) if w not in STOP and len(w) > 1]


def gen_batch(topic: str, style: str, n: int, model: str):
    msg = f"Topic: {topic}.\nStyle: {style}.\nWrite {n} different sentences. Return JSON {{\"sentences\": [...]}}."
    cmd = ["claude", "-p", "--model", model, "--system-prompt", SYSTEM, "--tools", "", "--restricted", "--no-session-persistence",
           "--output-format", "json", "--json-schema", SCHEMA]
    p = subprocess.run(cmd, input=msg, capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-300:])
    d = json.loads(p.stdout)
    out = d.get("structured_output") or json.loads(d.get("result", "{}"))
    return [s.strip() for s in out["sentences"] if 4 <= len(s.split()) <= 14]


def ref_pool():
    rows = []
    for f, src_key in [("smol_sent_train.jsonl", "src"), ("smol_doc_train.jsonl", "src")]:
        for l in open(MAN / f, encoding="utf-8"):
            r = json.loads(l); rows.append({"id": r["id"], "en": r[src_key], "brh_ref": r["tgt"], "source": r["source"]})
    for l in open(MAN / "bouquet_sent_dev.jsonl", encoding="utf-8"):
        r = json.loads(l); rows.append({"id": r["id"], "en": r["tgt"], "brh_ref": r["src"], "source": "bouquet_dev"})
    sealed = {json.loads(l)["src"].strip().lower() for f in ["smol_heldout.jsonl"] for l in open(EVAL / f, encoding="utf-8")}
    sealed |= {json.loads(l)["tgt"].strip().lower() for l in open(EVAL / "bouquet_test.jsonl", encoding="utf-8")}
    rows = [r for r in rows if r["en"].strip().lower() not in sealed and 3 <= len(r["en"].split()) <= 16 and not re.search(r"\d", r["en"])]
    return rows


def greedy_cover(cands, n, seen=None):
    """Pick n sentences maximizing newly covered word types per word spoken (shorter sentences win ties)."""
    seen = set(seen or ()); chosen = []
    pool = [(c, set(words(c["en"]))) for c in cands]
    while pool and len(chosen) < n:
        best = max(pool, key=lambda cw: (len(cw[1] - seen) / max(4, len(cw[0]["en"].split())), -len(cw[0]["en"])))
        if not (best[1] - seen) and len(chosen) >= n // 2:
            break
        chosen.append(best[0]); seen |= best[1]; pool.remove(best)
    return chosen, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ref", type=int, default=1000); ap.add_argument("--n-new", type=int, default=1000)
    ap.add_argument("--per-batch", type=int, default=40); ap.add_argument("--model", default="opus")
    ap.add_argument("--concurrency", type=int, default=4); ap.add_argument("--out", default=str(OUT_DIR / "sentences.jsonl"))
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # A. referenced sentences
    refs = ref_pool()
    rng.shuffle(refs)
    ref_chosen, seen = greedy_cover(refs, a.n_ref)
    print(f"ref pool {len(refs)} -> chosen {len(ref_chosen)}; word types covered {len(seen)}", flush=True)

    # B. generated candidates (cached)
    cache = OUT_DIR / "candidates.jsonl"
    cands = [json.loads(l) for l in open(cache, encoding="utf-8")] if cache.exists() else []
    done = {(c["topic"], c["style"]) for c in cands}
    jobs = [(t, s) for t in TOPICS for s in STYLES if (t, s) not in done]
    print(f"cached candidates {len(cands)}; generating {len(jobs)} batches x {a.per_batch}", flush=True)
    with ThreadPoolExecutor(a.concurrency) as ex, open(cache, "a", encoding="utf-8") as f:
        futs = {ex.submit(gen_batch, t, s, a.per_batch, a.model): (t, s) for t, s in jobs}
        for fu in as_completed(futs):
            t, s = futs[fu]
            try:
                for sent in fu.result():
                    row = {"topic": t, "style": s, "en": sent}; cands.append(row); f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush(); print(f"  ok {t} / {s}", flush=True)
            except Exception as e:
                print(f"  FAIL {t} / {s}: {e}", flush=True)
    # dedupe
    uniq, keys = [], set()
    for c in cands:
        k = re.sub(r"[^a-z ]", "", c["en"].lower()).strip()
        if k and k not in keys:
            keys.add(k); uniq.append({"id": None, "en": c["en"], "brh_ref": None, "source": "generated", "topic": c["topic"]})
    new_chosen, seen2 = greedy_cover(uniq, a.n_new, seen)
    print(f"generated unique {len(uniq)} -> chosen {len(new_chosen)}; total word types covered {len(seen2)}", flush=True)

    # Interleave topics / sources so a recording session is varied, assign stable ids.
    final = ref_chosen + new_chosen
    rng.shuffle(final)
    with open(a.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(final):
            r = dict(r); r["rec_id"] = f"rec_{i:04d}"; r["id"] = r["id"] or r["rec_id"]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_words = sum(len(r["en"].split()) for r in final)
    print(f"wrote {len(final)} sentences ({n_words} words, ~{n_words/2.3/60:.0f} min of speech at 2.3 w/s) -> {a.out}")


if __name__ == "__main__":
    main()
