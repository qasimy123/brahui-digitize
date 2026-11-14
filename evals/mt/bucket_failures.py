"""Bucket MT dev failures by cause with an outside judge (Opus 5.5 via `claude -p`), as checkable claims.

Usage: uv run python evals/mt/bucket_failures.py <tag> [--worst 150] [--batch 15] [--model opus]
Reads evals/mt/nllb/<tag>/{en2brh_dev,brh2en_dev}.jsonl, scores sentences by chrF++, sends the lowest-scoring
--worst per direction to the judge with a fixed taxonomy, and writes <tag>/buckets_<dir>.jsonl plus a count table.
Each verdict carries a quoted evidence span (reference word -> hypothesis word) so it can be checked by hand.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from sacrebleu.metrics import CHRF
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from brahui.text import normalize  # noqa: E402
chrf = CHRF(word_order=2)
TAX = {
 "en2brh": """wrong_word: a content word translated with a word of different meaning
synonym_ok: the meaning is right; the hypothesis uses a different but valid Brahui word or phrasing than the reference (reference divergence, not an error)
omission: a content word or clause of the source is missing
addition: content not in the source was added
entity: a name, number, or foreign term is wrong or transliterated differently
grammar: right words, wrong morphology (case suffix, agreement, verb form, word order) that changes or garbles meaning
orthography: the same word spelled differently (ی/ے, ہ/ھ, spacing, diacritics)
urdu_register: meaning right but Urdu vocabulary or construction where everyday Brahui has its own word
garbled: output is largely unintelligible or repetitive""",
 "brh2en": """wrong_word: a Brahui content word was mistranslated
misread_source: the Brahui source is noisy/misspelled and the system misread it
omission: a content word or clause is missing from the English
addition: content not in the source was added
polarity_tense: negation, tense, modality, or person wrong
entity: a name, number, or foreign term wrong
synonym_ok: meaning preserved; wording differs from the reference (reference divergence, not an error)
garbled: output is largely unintelligible or repetitive""",
}
SCHEMA = json.dumps({"type": "object", "properties": {"verdicts": {"type": "array", "items": {"type": "object", "properties": {
    "i": {"type": "integer"}, "bucket": {"type": "string"}, "evidence": {"type": "string"}, "meaning_preserved": {"type": "boolean"}},
    "required": ["i", "bucket", "evidence", "meaning_preserved"], "additionalProperties": False}}}, "required": ["verdicts"], "additionalProperties": False})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--worst", type=int, default=150)
    ap.add_argument("--batch", type=int, default=15); ap.add_argument("--model", default="opus"); ap.add_argument("--concurrency", type=int, default=5)
    a = ap.parse_args()
    d = ROOT / "evals/mt/nllb" / a.tag
    for direction in ["en2brh", "brh2en"]:
        rows = [json.loads(l) for l in open(d / f"{direction}_dev.jsonl", encoding="utf-8")]
        for r in rows:
            h, f = (normalize(r["hyp"]), normalize(r["ref"])) if direction == "en2brh" else (r["hyp"], r["ref"])
            r["chrf"] = round(chrf.sentence_score(h, [f]).score, 1)
        rows.sort(key=lambda r: r["chrf"]); worst = rows[:a.worst]
        src_l, tgt_l = ("English", "Brahui") if direction == "en2brh" else ("Brahui", "English")
        system = (f"You are a bilingual {src_l}-{tgt_l} reviewer auditing a machine translation system. For each item you get the source, "
                  f"a human reference translation and the system hypothesis. Classify the PRIMARY cause of the hypothesis's divergence from "
                  f"the reference into exactly one bucket:\n{TAX[direction]}\n\nAlso say whether the hypothesis preserves the source meaning "
                  f"(true/false). Give evidence as a short quoted span 'reference word -> hypothesis word' (or 'missing: ...'). Judge meaning, "
                  f"not surface overlap: if the hypothesis is a correct translation with different wording, use synonym_ok with meaning_preserved=true. "
                  f"Return JSON {{\"verdicts\": [...]}} with one verdict per item, same i.")
        batches = [worst[i:i + a.batch] for i in range(0, len(worst), a.batch)]

        def run(batch):
            lines = []
            for i, r in enumerate(batch):
                lines += [f"### Item {i}", f"Source: {r['src']}", f"Reference: {r['ref']}", f"Hypothesis: {r['hyp']}", ""]
            cmd = ["claude", "-p", "--model", a.model, "--system-prompt", system, "--tools", "", "--restricted",
                   "--no-session-persistence", "--output-format", "json", "--json-schema", SCHEMA]
            for _ in range(3):
                try:
                    p = subprocess.run(cmd, input="\n".join(lines), capture_output=True, text=True, timeout=900)
                    if p.returncode != 0:
                        raise RuntimeError(p.stderr[-200:] + p.stdout[-200:])
                    dd = json.loads(p.stdout); out = dd.get("structured_output") or json.loads(dd.get("result", ""))
                    vs = {v["i"]: v for v in out["verdicts"]}
                    return [{"id": r["id"], "chrf": r["chrf"], **{k: vs[i][k] for k in ["bucket", "evidence", "meaning_preserved"]}} for i, r in enumerate(batch) if i in vs]
                except Exception as e:
                    print("batch failed:", str(e)[:200], flush=True); time.sleep(5)
            return []

        with ThreadPoolExecutor(a.concurrency) as ex:
            res = [v for part in ex.map(run, batches) for v in part]
        with open(d / f"buckets_{direction}.jsonl", "w", encoding="utf-8") as f:
            for v in res:
                f.write(json.dumps(v, ensure_ascii=False) + "\n")
        c = Counter(v["bucket"] for v in res); mp = sum(v["meaning_preserved"] for v in res)
        print(f"\n== {a.tag} {direction}: worst {len(res)} of {len(rows)} dev sentences (chrF <= {worst[-1]['chrf']}); meaning preserved in {mp}")
        for b, n in c.most_common():
            print(f"   {b:16} {n:4d}  {100 * n / len(res):5.1f}%")


if __name__ == "__main__":
    main()
