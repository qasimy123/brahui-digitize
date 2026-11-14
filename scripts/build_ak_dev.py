"""Build the second dev split from Ali & Kobayashi 2024 (English written FROM the Brahui, so no translator loss).

Usage: uv run python scripts/build_ak_dev.py [--n 500] [--model opus]
- samples --n sentences (half short stories ch.2, half folktales ch.3; seed 13) from data/eval/ak2024_sentences.jsonl (quality ok, len>=4 words)
- transliterates the romanized Brahui into Perso-Arabic script with Claude (standard Brahui orthography as in the gold corpus; short vowels unwritten)
- writes data/dev/mt_dev_ak.jsonl {id, source:"ak2024", brh (Perso-Arabic), brh_roman, gloss, en, text, chapter}
- writes data/parallel/ak2024_train.jsonl with every other sentence (romanized; for in-context use), never overlapping the dev ids
"""
from __future__ import annotations

import argparse, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evals" / "mt"))
from run_eval import load_cases, SEED  # noqa: E402

MODELS = {"opus": "claude-opus-5-5", "fable": "claude-fable-5-1"}
SCHEMA = {"type": "object", "properties": {"arabic": {"type": "array", "items": {"type": "string"}}}, "required": ["arabic"], "additionalProperties": False}


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=500); ap.add_argument("--model", default="opus", choices=MODELS)
    ap.add_argument("--concurrency", type=int, default=6); a = ap.parse_args()
    rows = [r for r in read_jsonl(ROOT / "data/eval/ak2024_sentences.jsonl") if r.get("quality") == "ok" and len(r["brh"].split()) >= 4 and r["en"].strip()]
    rng = random.Random(13)
    ch2 = [r for r in rows if r["chapter"] == 2]; ch3 = [r for r in rows if r["chapter"] == 3]
    dev = rng.sample(ch2, a.n // 2) + rng.sample(ch3, a.n - a.n // 2)
    dev_ids = {r["id"] for r in dev}
    train = [r for r in rows if r["id"] not in dev_ids]
    write_jsonl(ROOT / "data/parallel/ak2024_train.jsonl", [{"id": r["id"], "source": "ak2024", "brh_roman": r["brh"], "gloss": r["gloss"], "en": r["en"]} for r in train])
    print(f"dev {len(dev)} (ch2 {a.n//2}, ch3 {a.n-a.n//2}), train {len(train)}", flush=True)

    # transliteration: examples of real Brahui orthography from the gold corpus + the book's transcription key
    _, pools = load_cases(); pool = sorted(pools["brh2en"], key=lambda p: p["id"])
    shots = random.Random(SEED).sample(pool, 60)
    key = (ROOT / "data/lexicon/ak_grammar_outline.txt").read_text(encoding="utf-8")[:6000]
    system = ("You convert romanized Brahui (the phonemic transcription of Ali & Kobayashi 2024: macrons = long vowels, ṭ ḍ ṛ ɬ retroflex, š = sh, x = kh, ɣ = gh, "
              "c = ch, ʰ = aspiration; '-' marks suffix boundaries and '=' clitics, which are NOT written) into the standard Brahui Perso-Arabic orthography used in "
              "Brahui newspapers and books. Rules: write words as a Brahui writer would (short vowels unwritten except where a Brahui writer marks them; long ā = ا/آ, "
              "ī = ی, ū = و, e = ے/ی, o = و; ṭ=ٹ, ḍ=ڈ, ṛ=ڑ, ɬ=ڷ or لؕ as the corpus does, š=ش, x=خ, ɣ=غ, c=چ, j=ج, v/w=و, y=ی, h=ہ, ng=نگ, aspiration = ھ); "
              "Arabic/Persian/Urdu loans keep their etymological spelling (e.g. ɣam = غم, vaxt = وخت, sāhib = صاحب, zindagī = زندگی); case suffixes attach or detach exactly as the example corpus does "
              "(e.g. -e → ئے or ے separate, -nā → نا separate, -ṭī → ٹی, -ān → آن, -aṭ → اٹ, -ā → آ, =ta/=te clitics as the corpus writes them). Keep punctuation and quotation marks (use ۔ for the full stop). "
              "Produce exactly one Perso-Arabic string per input, same order. Return JSON {\"arabic\": [...]}.\n\n## Transcription key\n" + key +
              "\n\n## Examples of real Brahui orthography (Perso-Arabic with English)\n" + "\n".join(f"{p['brh']}\n   = {p['en']}" for p in shots))
    load_env(); cl = anthropic.Anthropic(); model = MODELS[a.model]
    batches = [dev[i:i + 10] for i in range(0, len(dev), 10)]

    def run(batch):
        user = "Convert these romanized Brahui sentences to Perso-Arabic script:\n" + "\n".join(f"{i+1}. {r['brh']}" for i, r in enumerate(batch))
        for t in range(3):
            try:
                with cl.messages.stream(model=model, max_tokens=16000, system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                                        messages=[{"role": "user", "content": user}],
                                        output_config={"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "high"}) as st:
                    r = st.get_final_message()
                out = json.loads("".join(b.text for b in r.content if b.type == "text"))["arabic"]
                if len(out) == len(batch): return out
                print("  length mismatch, retry", flush=True)
            except Exception as e:
                print("  fail:", str(e)[:120], flush=True); time.sleep(5)
        return [""] * len(batch)

    t0 = time.time()
    with ThreadPoolExecutor(a.concurrency) as ex:
        outs = [x for part in ex.map(run, batches) for x in part]
    final = [{"id": r["id"], "source": "ak2024", "brh": ar, "brh_roman": r["brh"], "gloss": r["gloss"], "en": r["en"], "text": r["text"], "chapter": r["chapter"]}
             for r, ar in zip(dev, outs)]
    write_jsonl(ROOT / "data/dev/mt_dev_ak.jsonl", final)
    print(f"wrote data/dev/mt_dev_ak.jsonl ({sum(1 for f in final if not f['brh'])} empty) in {time.time()-t0:.0f}s")
    for f in final[:3]: print(f["brh_roman"], "\n ->", f["brh"], "\n  =", f["en"])


if __name__ == "__main__":
    main()
