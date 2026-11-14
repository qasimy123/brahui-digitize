"""Round 0 of PLAN_95: measure the meaning-preserved (MP) rate of teachers and student on the dev split, and bucket failures.

Usage (needs ANTHROPIC_API_KEY in .env):
  uv run python evals/mt/round0_mp.py translate --model opus|fable [--shots 40] [--per-request 20]
      -> evals/mt/round0/<model>/brh2en_dev.jsonl  (id, src, ref, hyp)
  uv run python evals/mt/round0_mp.py judge <tag> [--judge opus|fable]
      <tag> is a dir under evals/mt/round0/ or evals/mt/nllb/ holding brh2en_dev.jsonl
      -> <dir>/mp_<judge>.jsonl + prints MP rate with binomial 95% CI. Judge sees English only (ref vs hyp).
  uv run python evals/mt/round0_mp.py bucket <tag> [--judge opus]
      gloss pass on every MP=false sentence: Brahui source, word-by-word gloss, cause bucket -> <dir>/buckets.jsonl + table
Rules: few-shot pool excludes dev ids; judge model is never the model under test unless --judge says so (audit).
"""
from __future__ import annotations

import argparse, json, math, os, random, subprocess, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evals" / "mt"))
from run_eval import load_cases, SEED  # noqa: E402

MODELS = {"opus": "claude-opus-5-5", "fable": "claude-fable-5-1", "sonnet": "claude-sonnet-5-5"}
DEV = ROOT / "data/dev/mt_dev.jsonl"
OUT = ROOT / "evals/mt/round0"
SYSTEM = (ROOT / "evals/mt/prompts/system_v0.md").read_text(encoding="utf-8")


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"'))


def client():
    if BACKEND == "cli": return None
    load_env(); return anthropic.Anthropic()


BACKEND = os.environ.get("BRH_BACKEND", "cli")  # "cli" = claude -p on the subscription (default since 2026-10-19, API credits exhausted); "api" = Messages API
ALIAS = {v: k for k, v in MODELS.items()}


def call_cli(model, system, user, schema, tries=3, effort=None):
    import tempfile
    text = system if isinstance(system, str) else "\n\n".join(b["text"] for b in system)
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(text); sp = f.name
    cmd = ["claude", "-p", "--model", ALIAS.get(model, model), "--system-prompt-file", sp, "--tools", "", "--restricted",
           "--no-session-persistence", "--output-format", "json", "--json-schema", json.dumps(schema)]
    if effort: cmd += ["--effort", effort]
    try:
        for t in range(tries):
            try:
                p = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=1800)
                if p.returncode != 0: raise RuntimeError((p.stderr or p.stdout)[-300:])
                dd = json.loads(p.stdout); out = dd.get("structured_output") or json.loads(dd.get("result", ""))
                return out, None
            except Exception as e:
                print(f"  cli call failed ({t+1}/{tries}): {str(e)[:160]}", flush=True); time.sleep(5 * (t + 1))
        return None, None
    finally:
        os.unlink(sp)


def call(cl, model, system, user, schema, max_tokens=8000, tries=4, effort=None):
    if BACKEND == "cli": return call_cli(model, system, user, schema, effort=effort)
    oc = {"format": {"type": "json_schema", "schema": schema}}
    if effort: oc["effort"] = effort
    for t in range(tries):
        try:
            with cl.messages.stream(model=model, max_tokens=max_tokens, system=system,
                                    messages=[{"role": "user", "content": user}], output_config=oc) as st:
                r = st.get_final_message()
            if r.stop_reason == "refusal":
                print("  refusal:", getattr(r.stop_details, "explanation", ""), flush=True); return None, r.usage
            txt = "".join(b.text for b in r.content if b.type == "text")
            return json.loads(txt), r.usage
        except Exception as e:
            print(f"  call failed ({t+1}/{tries}): {str(e)[:160]}", flush=True); time.sleep(5 * (t + 1))
    return None, None


def binom_ci(k, n):
    if n == 0: return (0, 0)
    p = k / n; z = 1.96
    d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def find_dir(tag):
    for base in [OUT, ROOT / "evals/mt/nllb"]:
        if (base / tag / "brh2en_dev.jsonl").exists(): return base / tag
    raise SystemExit(f"no brh2en_dev.jsonl under round0/{tag} or nllb/{tag}")


def corpus_block(dev_ids, ak_train=False):
    _, pools = load_cases()
    pool = sorted([p for p in pools["brh2en"] if p["id"] not in dev_ids], key=lambda p: p["id"])
    block = ["## Reference translations (Brahui -> English)", ""]
    for p in pool: block += [f"Brahui: {p['brh']}", f"English: {p['en']}", ""]
    if ak_train:
        ak = [r for r in read_jsonl(ROOT / "data/parallel/ak2024_train.jsonl") if r["id"] not in dev_ids]
        block += ["", "## Reference translations from glossed texts (romanized Brahui, morpheme-segmented -> English)", ""]
        for r in ak: block += [f"Brahui: {r['brh_roman']}", f"Gloss: {r['gloss']}", f"English: {r['en']}", ""]
        pool = pool + ak
    return {"type": "text", "text": "\n".join(block), "cache_control": {"type": "ephemeral", "ttl": "1h"}}, len(pool)


def pick(rows, limit, seed=SEED):
    """Seeded pilot slice so every pilot run sees the same ids."""
    if not limit or limit >= len(rows): return rows
    return random.Random(seed).sample(rows, limit)


# ---------------------------------------------------------------- translate
T_SCHEMA = {"type": "object", "properties": {"translations": {"type": "array", "items": {"type": "string"}}},
            "required": ["translations"], "additionalProperties": False}
G_SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {
    "gloss": {"type": "string"}, "translation": {"type": "string"}}, "required": ["gloss", "translation"], "additionalProperties": False}}},
    "required": ["items"], "additionalProperties": False}


def cmd_translate(a):
    dev = read_jsonl(DEV); dev_ids = {r["id"] for r in dev}; dev = pick(dev, a.limit)
    _, pools = load_cases()
    pool = sorted([p for p in pools["brh2en"] if p["id"] not in dev_ids], key=lambda p: p["id"])
    full = a.mode == "full"
    shots = pool if full else random.Random(SEED).sample(pool, a.shots)
    print(f"shot pool {len(pool)} (dev excluded), using {len(shots)} shots, mode={a.mode}, effort={a.effort}, ak_train={a.ak_train}", flush=True)
    block = ["## Reference translations (Brahui -> English)", ""]
    for p in shots: block += [f"Brahui: {p['brh']}", f"English: {p['en']}", ""]
    if a.ak_train:
        cb, _ = corpus_block(dev_ids, ak_train=True); block = [cb["text"]]
    instr = ("\n\nTranslate the meaning faithfully and completely, sentence by sentence; do not summarize, add commentary, or drop any detail.")
    if full:
        instr += ("\n\nYour prior knowledge of Brahui is thin and the written corpus is small, so do not guess from surface "
                  "resemblance. Work deductively: segment every word into stem and suffixes; for each stem, search the reference "
                  "translations above for the same or a related form and read its meaning from the aligned English; treat Balochi, "
                  "Persian and Urdu cognates as evidence only when the corpus does not contradict them; resolve case suffixes "
                  "(-e/-ئے accusative-dative, -na/نا genitive, -ṭī/ٹی locative, -ān/آن ablative, -aṭ/اٹ instrumental), tense, "
                  "negation and person explicitly; then compose the sentence from the parse. Where a word is still unknown, "
                  "choose the reading most consistent with the whole sentence and the corpus, never a transliteration. Every fact "
                  "in the source must appear in the English, and nothing more.")
    system = [{"type": "text", "text": SYSTEM + instr}]
    if a.lexicon:
        lex = (ROOT / "data/lexicon/ak_grammar_outline.txt").read_text(encoding="utf-8") + "\n\n## Glossary (Ali & Kobayashi 2024; romanized, see the transcription table above to map Perso-Arabic spellings)\n" + (ROOT / "data/lexicon/ak_glossary.txt").read_text(encoding="utf-8")
        lex += "\n\n## Word list (Perso-Arabic script, roman, English; modern everyday vocabulary)\n" + (ROOT / "data/lexicon/shaigle_brh_en.txt").read_text(encoding="utf-8")
        shad = ROOT / "data/lexicon/shad_brh_en.txt"
        if a.lexicon == "all":
            lex += "\n\n## English -> Brahui dictionary (Dixey/Sarparra 2022; romanized Brahui; use it in reverse to confirm a Brahui word's meaning)\n" + (ROOT / "data/lexicon/sarparra_en_brh.txt").read_text(encoding="utf-8")
        if shad.exists() and a.lexicon in ("shad", "all"):
            lex += "\n\n## Dictionary (Shad 2021, Brahui -> English, Perso-Arabic headwords; the main lexical authority)\n" + shad.read_text(encoding="utf-8")
        system.append({"type": "text", "text": lex})
        instr_lex = ("\n\nA grammar outline and a glossary are provided (romanized). Map each Perso-Arabic word to its romanized form "
                     "(unwritten short vowels: try a/i/u; ئے = -e D/A, نا = -nā GEN, ٹی = -ṭī LOC, آن = -ān ABL, اٹ = -aṭ, آ = -ā ALL; ک/گ/غ/خ as k/g/ɣ/x; ڑ=ṛ, ٹ=ṭ, ڈ=ḍ, ش=š, لؑ/ڷ=ɬ) "
                     "and look the stem up in the glossary before guessing. The glossary is authoritative over your prior when they conflict.")
        system[0]["text"] += instr_lex
    system.append({"type": "text", "text": "\n".join(block), "cache_control": {"type": "ephemeral", "ttl": "1h"}})
    model = MODELS[a.model]; cl = client(); tag = a.tag or (a.model + ("_full" if full else "") + (f"_lex_{a.lexicon}" if a.lexicon else ""))
    ctx = {}
    if a.context:
        allrows = read_jsonl(ROOT / "data/eval/ak2024_sentences.jsonl"); by_text = {}
        for r in allrows: by_text.setdefault(r["text"], []).append(r)
        for t, rs in by_text.items():
            for k, r in enumerate(rs): ctx[r["id"]] = [x["brh"] for x in rs[max(0, k - a.context):k]]
        system[0]["text"] += ("\n\nSome items come with the preceding sentences of the same story as context (romanized). Use the context to resolve "
                              "pronouns, referents, gender (Brahui marks none; English must), tense and who is speaking; translate only the target sentence.")
    per = a.per_request if not full else min(a.per_request, 5)
    batches = [dev[i:i + per] for i in range(0, len(dev), per)]

    def run(batch):
        user = [f"## Translate each of the following {len(batch)} Brahui sentences into English.",
                "Return JSON {\"translations\": [...]} with exactly one string per input, in order.", ""]
        for i, r in enumerate(batch, 1):
            if ctx.get(r["id"]): user.append(f"[context for {i}: " + " ".join(ctx[r["id"]]) + "]")
            user.append(f"{i}. {r['brh_roman'] if a.input == 'roman' else r['brh']}")
        if a.gloss_first:
            user[1] = ("Return JSON {\"items\": [{\"gloss\": ..., \"translation\": ...}, ...]}, one per input in order. First write a word-by-word morpheme gloss "
                       "(Leipzig style, as in the glossed references: stem meanings in lowercase, grammatical categories in CAPS, '-' between morphemes), then the translation composed from that gloss.")
            out, usage = call(cl, model, system, "\n".join(user), G_SCHEMA, max_tokens=32000, effort=a.effort)
            if out and "items" in out: out = {"translations": [x["translation"] for x in out["items"]], "glosses": [x["gloss"] for x in out["items"]]}
        else:
            out, usage = call(cl, model, system, "\n".join(user), T_SCHEMA, max_tokens=32000, effort=a.effort)
        if usage: print(f"  usage in={usage.input_tokens} cache_read={usage.cache_read_input_tokens} out={usage.output_tokens}", flush=True)
        if not out or len(out["translations"]) != len(batch):
            print(f"  batch size mismatch/failed; retrying singly", flush=True)
            res = []
            for r in batch:
                o, _ = call(cl, model, system, f"## Translate this Brahui sentence into English.\nReturn JSON {{\"translations\": [\"...\"]}}.\n\n1. {r['brh']}", T_SCHEMA, max_tokens=32000, effort=a.effort)
                res.append(o["translations"][0] if o and o.get("translations") else "")
            return [{"id": r["id"], "src": r["brh"], "ref": r["en"], "hyp": h} for r, h in zip(batch, res)]
        gl = out.get("glosses", [""] * len(batch))
        return [{"id": r["id"], "src": r["brh_roman"] if a.input == "roman" else r["brh"], "ref": r["en"], "hyp": h, **({"gloss_hyp": g} if g else {})} for r, h, g in zip(batch, out["translations"], gl)]

    t0 = time.time()
    with ThreadPoolExecutor(a.concurrency) as ex:
        rows = [r for part in ex.map(run, batches) for r in part]
    empty = sum(1 for r in rows if not r["hyp"])
    write_jsonl(OUT / tag / "brh2en_dev.jsonl", rows)
    print(f"wrote {len(rows)} rows ({empty} empty) to round0/{tag}/brh2en_dev.jsonl in {time.time()-t0:.0f}s")


# ---------------------------------------------------------------- judge
J_SCHEMA = {"type": "object", "properties": {"verdicts": {"type": "array", "items": {"type": "object", "properties": {
    "i": {"type": "integer"},
    "discrepancies": {"type": "array", "items": {"type": "object", "properties": {
        "kind": {"type": "string", "enum": ["added", "dropped", "changed"]},
        "reference": {"type": "string"}, "hypothesis": {"type": "string"}, "note": {"type": "string"}},
        "required": ["kind", "reference", "hypothesis", "note"], "additionalProperties": False}},
    "meaning_preserved": {"type": "boolean"}},
    "required": ["i", "discrepancies", "meaning_preserved"], "additionalProperties": False}}},
    "required": ["verdicts"], "additionalProperties": False}

J_SYSTEM = """You are auditing machine translations into English. For each item you get a REFERENCE English translation (by a human) and a HYPOTHESIS English translation (by a system) of the same original sentence. You do not see the original.

List every factual discrepancy between hypothesis and reference:
- added: the hypothesis states something the reference does not
- dropped: the hypothesis omits something the reference states
- changed: a fact differs (different entity, number, time, polarity, agent/patient, degree, modality, causal relation)
Do NOT list: different wording with the same meaning, synonyms, register, word order, punctuation, spelling, or paraphrase. Minor stylistic differences are not discrepancies.

meaning_preserved = true only if a careful reader would take the same facts from both texts (empty discrepancy list, or only differences a reader would never notice).
Return JSON {"verdicts": [...]} with one verdict per item, same i."""

J_CALIBRATED = J_SYSTEM + """

Calibration (the reference is itself a translation and may be loose; judge facts, not phrasing):
- NOT a discrepancy: restructured clauses with the same propositional content ("was able to help X draw" vs "succeeded in helping X who drew"); modal/aspect wording of the same force ("one could" vs "some can", "informed" vs "were notified", "would tear" vs "tears" in a narrative); generic vs specific subject when both fit ("we" vs "people"); near-synonyms of the same referent ("camp" vs "tent", "portrait" vs "sketch", "cold" vs "iced"); a heading-style fragment rendered as a full sentence.
- A discrepancy: a different referent (snake vs mouse, monthly vs toy), a different action (tears vs sees), a dropped or invented clause, a reversed polarity, a different number/date/name, a different causal or agent/patient relation, a degree word that changes the claim (sometimes vs often only when the reader would act differently on it).
Ask: would a reader who acted on the hypothesis do anything differently from one who acted on the reference? If no, meaning_preserved = true."""


import hashlib
JCACHE = OUT / "judge_cache.jsonl"


def jcache_load():
    return {r["k"]: r["v"] for r in read_jsonl(JCACHE)} if JCACHE.exists() else {}


def jcache_key(judge, rubric, ref, hyp):
    return hashlib.sha1(f"{judge}|{rubric}|{ref}|{hyp}".encode()).hexdigest()


def cmd_judge(a):
    d = find_dir(a.tag); rows = read_jsonl(d / "brh2en_dev.jsonl")
    cache = jcache_load(); hits = [0]
    model = MODELS[a.judge]; cl = client(); system = J_CALIBRATED if a.rubric == "calibrated" else J_SYSTEM
    suffix = f"{a.judge}" + ("_cal" if a.rubric == "calibrated" else "")
    batches = [rows[i:i + a.batch] for i in range(0, len(rows), a.batch)]

    def run(batch):
        done = {}
        todo = []
        for r in batch:
            k = jcache_key(a.judge, a.rubric, r["ref"], r["hyp"])
            if k in cache: done[r["id"]] = dict(cache[k], id=r["id"]); hits[0] += 1
            else: todo.append(r)
        if todo:
            lines = []
            for i, r in enumerate(todo):
                lines += [f"### Item {i}", f"REFERENCE: {r['ref']}", f"HYPOTHESIS: {r['hyp'] or '(empty)'}", ""]
            out, _ = call(cl, model, system, "\n".join(lines), J_SCHEMA)
            vs = {v["i"]: v for v in out["verdicts"]} if out else {}
            for i, r in enumerate(todo):
                if i in vs:
                    v = {"meaning_preserved": vs[i]["meaning_preserved"], "discrepancies": vs[i]["discrepancies"]}
                    done[r["id"]] = dict(v, id=r["id"])
                    with open(JCACHE, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"k": jcache_key(a.judge, a.rubric, r["ref"], r["hyp"]), "v": v}, ensure_ascii=False) + "\n")
        return [done[r["id"]] for r in batch if r["id"] in done]

    with ThreadPoolExecutor(a.concurrency) as ex:
        res = [v for part in ex.map(run, batches) for v in part]
    if a.refs == "clean":
        alt = {r["id"]: r.get("en_alt", []) for r in read_jsonl(DEV.with_name(DEV.stem + "_clean.jsonl"))}
        byid = {r["id"]: r for r in rows}
        redo = [dict(byid[v["id"]], ref=alt[v["id"]][0]) for v in res if not v["meaning_preserved"] and alt.get(v["id"])]
        with ThreadPoolExecutor(a.concurrency) as ex:
            res2 = {v["id"]: v for part in ex.map(run, [redo[i:i + a.batch] for i in range(0, len(redo), a.batch)]) for v in part}
        flipped = 0
        for v in res:
            if v["id"] in res2 and res2[v["id"]]["meaning_preserved"]:
                v.update(res2[v["id"]]); v["via_alt"] = True; flipped += 1
        print(f"  clean refs: {len(redo)} failures re-judged against corrected reference, {flipped} flipped to MP")
        suffix += "_clean"
    write_jsonl(d / f"mp_{suffix}.jsonl", res)
    print(f"  judge cache hits: {hits[0]}")
    n = len(res); k = sum(r["meaning_preserved"] for r in res); lo, hi = binom_ci(k, n)
    kinds = Counter(x["kind"] for r in res for x in r["discrepancies"])
    devrows = {r["id"]: r for r in read_jsonl(DEV)}
    src = {r["id"]: (str(devrows.get(r["id"], {}).get("chapter")) if "chapter" in devrows.get(r["id"], {}) else r["id"].split("_")[0]) for r in rows}
    by_src = Counter(); tot = Counter()
    for r in res: tot[src[r["id"]]] += 1; by_src[src[r["id"]]] += r["meaning_preserved"]
    print(f"{a.tag} judged by {suffix}: MP {k}/{n} = {100*k/n:.1f}%  [95% CI {100*lo:.1f}, {100*hi:.1f}]")
    print("  by source:", {s: f"{100*by_src[s]/tot[s]:.1f}% (n={tot[s]})" for s in tot})
    print("  discrepancy kinds:", dict(kinds))


# ---------------------------------------------------------------- adjudicate (reference audit)
A_SCHEMA = {"type": "object", "properties": {"verdicts": {"type": "array", "items": {"type": "object", "properties": {
    "i": {"type": "integer"}, "faithful": {"type": "boolean"},
    "issues": {"type": "array", "items": {"type": "object", "properties": {
        "kind": {"type": "string", "enum": ["reference_adds", "reference_omits", "reference_changes"]},
        "brahui": {"type": "string"}, "gloss": {"type": "string"}, "reference": {"type": "string"}},
        "required": ["kind", "brahui", "gloss", "reference"], "additionalProperties": False}},
    "en_faithful": {"type": "string"}},
    "required": ["i", "faithful", "issues", "en_faithful"], "additionalProperties": False}}},
    "required": ["verdicts"], "additionalProperties": False}

A_SYSTEM = SYSTEM + """

You are auditing an evaluation set, not translating for a reader. For each item you get a Brahui sentence and an English REFERENCE. The reference was often the original text and the Brahui a human translation of it, so the reference may state things the Brahui does not, omit things the Brahui states, or change a fact. Your job: decide whether the reference is a faithful rendering of the Brahui as written.

Work deductively from the Brahui: segment each word into stem and suffixes, find the stem in the reference translations above or in Balochi/Persian/Urdu cognates, resolve case, tense, negation and person, and compare fact by fact with the reference. Report an issue only for a factual difference a reader would notice: a referent, action, modifier, number, polarity, agent/patient or clause that is present on one side and absent or different on the other. Paraphrase, register and word order are not issues. Be conservative: if the Brahui is ambiguous and the reference is one valid reading, it is faithful.

Then write en_faithful: the reference text edited minimally so that it says exactly what the Brahui says (keep the reference wording wherever it is faithful; if faithful = true, copy the reference unchanged).
Return JSON {"verdicts": [...]} with one verdict per item, same i."""


def cmd_adjudicate(a):
    dev = read_jsonl(DEV); dev_ids = {r["id"] for r in dev}; rows = pick(dev, a.limit)
    block, n_pool = corpus_block(dev_ids)
    model = MODELS[a.model]; cl = client()
    print(f"adjudicating {len(rows)} refs with {a.model} effort={a.effort}, corpus {n_pool} pairs", flush=True)
    batches = [rows[i:i + a.per_request] for i in range(0, len(rows), a.per_request)]

    def run(batch):
        lines = []
        for i, r in enumerate(batch):
            lines += [f"### Item {i}", f"Brahui: {r.get('brh_roman', r['brh']) if a.input == 'roman' else r['brh']}"]
            if a.with_gloss and r.get("gloss"): lines.append(f"Morpheme gloss (authoritative, from the linguists who transcribed it): {r['gloss']}")
            lines += [f"REFERENCE: {r['en']}", ""]
        sysx = A_SYSTEM + ("\n\nWhen a morpheme gloss is given, treat it as ground truth for what each word means; judge the reference against the gloss." if a.with_gloss else "")
        out, usage = call(cl, model, [{"type": "text", "text": sysx}, block], "\n".join(lines), A_SCHEMA, max_tokens=32000, effort=a.effort)
        if usage: print(f"  usage in={usage.input_tokens} cache_read={usage.cache_read_input_tokens} out={usage.output_tokens}", flush=True)
        if not out: return []
        vs = {v["i"]: v for v in out["verdicts"]}
        return [{"id": r["id"], **{k: vs[i][k] for k in ["faithful", "issues", "en_faithful"]}} for i, r in enumerate(batch) if i in vs]

    with ThreadPoolExecutor(a.concurrency) as ex:
        res = [v for part in ex.map(run, batches) for v in part]
    out = OUT / "ref_audit" / f"{DEV.stem}_{a.model}{'_'+str(a.limit) if a.limit else ''}.jsonl" if DEV.stem != "mt_dev" else OUT / "ref_audit" / f"{a.model}{'_'+str(a.limit) if a.limit else ''}.jsonl"
    write_jsonl(out, res)
    k = sum(not r["faithful"] for r in res); kinds = Counter(x["kind"] for r in res for x in r["issues"])
    print(f"{a.model}: {k}/{len(res)} references judged unfaithful; issue kinds {dict(kinds)} -> {out.relative_to(ROOT)}")


def cmd_build_clean(a):
    """mt_dev_clean.jsonl: en_alt = corrections agreed by both adjudicators (and judged equivalent by the English judge)."""
    dev = read_jsonl(DEV)
    pre = "" if DEV.stem == "mt_dev" else DEV.stem + "_"
    suf = f"_{a.limit}" if a.limit else ""
    f = {r["id"]: r for r in read_jsonl(OUT / f"ref_audit/{pre}fable{suf}.jsonl")}
    o = {r["id"]: r for r in read_jsonl(OUT / f"ref_audit/{pre}opus{suf}.jsonl")}
    cands = [r for r in dev if r["id"] in f and r["id"] in o and not f[r["id"]]["faithful"] and not o[r["id"]]["faithful"]]
    cl = client(); model = MODELS[a.judge]
    pairs = [{"id": r["id"], "ref": f[r["id"]]["en_faithful"], "hyp": o[r["id"]]["en_faithful"]} for r in cands]

    def run(batch):
        lines = []
        for i, r in enumerate(batch): lines += [f"### Item {i}", f"REFERENCE: {r['ref']}", f"HYPOTHESIS: {r['hyp']}", ""]
        out, _ = call(cl, model, J_CALIBRATED, "\n".join(lines), J_SCHEMA)
        vs = {v["i"]: v["meaning_preserved"] for v in out["verdicts"]} if out else {}
        return [(r["id"], vs.get(i, False)) for i, r in enumerate(batch)]

    with ThreadPoolExecutor(4) as ex:
        agree = dict(x for part in ex.map(run, [pairs[i:i + 10] for i in range(0, len(pairs), 10)]) for x in part)
    n_alt = n_disp = 0
    for r in dev:
        i = r["id"]; fi, oi = f.get(i), o.get(i)
        if fi and oi and not fi["faithful"] and not oi["faithful"] and agree.get(i):
            r["en_alt"] = [fi["en_faithful"], oi["en_faithful"]]; r["ref_status"] = "corrected"; n_alt += 1
            r["ref_issues"] = [x["kind"] + ": " + x["gloss"] for x in fi["issues"]]
        elif fi and oi and (not fi["faithful"] or not oi["faithful"]):
            r["ref_status"] = "disputed"; n_disp += 1
        else:
            r["ref_status"] = "ok"
    write_jsonl(DEV.with_name(DEV.stem + "_clean.jsonl"), dev)
    print(f"clean dev: {n_alt} corrected (both adjudicators, corrections equivalent), {len(cands)-n_alt} both-flagged but corrections differ, {n_disp} disputed, {len(dev)-n_alt-n_disp} ok")


# ---------------------------------------------------------------- consensus
C_SCHEMA = {"type": "object", "properties": {"verdicts": {"type": "array", "items": {"type": "object", "properties": {
    "i": {"type": "integer"}, "final": {"type": "string"},
    "agreement": {"type": "string", "enum": ["all_agree", "majority", "split"]},
    "disagreements": {"type": "array", "items": {"type": "object", "properties": {
        "brahui": {"type": "string"}, "readings": {"type": "string"}, "chosen": {"type": "string"}, "why": {"type": "string"}},
        "required": ["brahui", "readings", "chosen", "why"], "additionalProperties": False}}},
    "required": ["i", "final", "agreement", "disagreements"], "additionalProperties": False}}},
    "required": ["verdicts"], "additionalProperties": False}

C_SYSTEM = SYSTEM + """

You are the final arbiter in a translation committee. For each item you get a Brahui sentence and several candidate English translations produced independently by strong systems. Where the candidates agree, that reading is almost certainly right. Where they differ, the difference marks a word or construction that is hard; resolve it from the Brahui itself: segment the word, find the stem in the reference translations above (or a cognate), resolve the suffixes, and choose the reading the evidence supports, not the majority by default. Then write the final translation: complete, faithful, every fact of the source and nothing more; reuse the candidates' wording where it is right. List each disagreement you resolved with the Brahui span, the competing readings, the one you chose, and the evidence.
Return JSON {"verdicts": [...]} with one verdict per item, same i."""


def cmd_consensus(a):
    dev_ids = {r["id"] for r in read_jsonl(DEV)}
    cands = {t: {r["id"]: r for r in read_jsonl(find_dir(t) / "brh2en_dev.jsonl")} for t in a.tags}
    ids = [i for i in cands[a.tags[0]] if all(i in c for c in cands.values())]
    rows = [cands[a.tags[0]][i] for i in ids]
    block, _ = corpus_block(dev_ids); model = MODELS[a.model]; cl = client()
    print(f"consensus over {a.tags} on {len(rows)} sentences, arbiter {a.model} {a.effort}", flush=True)
    batches = [rows[i:i + a.per_request] for i in range(0, len(rows), a.per_request)]

    def run(batch):
        lines = []
        for i, r in enumerate(batch):
            lines += [f"### Item {i}", f"Brahui: {r['src']}"]
            lines += [f"Candidate {chr(65+j)}: {cands[t][r['id']]['hyp']}" for j, t in enumerate(a.tags)]
            lines.append("")
        out, usage = call(cl, model, [{"type": "text", "text": C_SYSTEM}, block], "\n".join(lines), C_SCHEMA, max_tokens=32000, effort=a.effort)
        if not out: return []
        vs = {v["i"]: v for v in out["verdicts"]}
        return [{"id": r["id"], "src": r["src"], "ref": r["ref"], "hyp": vs[i]["final"], "agreement": vs[i]["agreement"],
                 "disagreements": vs[i]["disagreements"]} for i, r in enumerate(batch) if i in vs]

    with ThreadPoolExecutor(a.concurrency) as ex:
        res = [v for part in ex.map(run, batches) for v in part]
    write_jsonl(OUT / a.name / "brh2en_dev.jsonl", res)
    print(f"wrote {len(res)} rows to round0/{a.name}; agreement {dict(Counter(r['agreement'] for r in res))}")


def cmd_mbr(a):
    """MBR with the MP judge as utility: pick, per sentence, the candidate that the other candidates 'preserve the meaning of' most. No rewriting."""
    cands = {t: {r["id"]: r for r in read_jsonl(find_dir(t) / "brh2en_dev.jsonl")} for t in a.tags}
    ids = [i for i in cands[a.tags[0]] if all(i in c for c in cands.values())]
    model = MODELS[a.judge]; cl = client(); cache = jcache_load()
    pairs = [(i, x, y) for i in ids for x in a.tags for y in a.tags if x != y]  # (sentence, ref-candidate, hyp-candidate)

    def run(batch):
        res = {}; todo = []
        for (i, x, y) in batch:
            k = jcache_key(a.judge, "calibrated", cands[x][i]["hyp"], cands[y][i]["hyp"])
            if k in cache: res[(i, x, y)] = cache[k]["meaning_preserved"]
            else: todo.append((i, x, y))
        if todo:
            lines = []
            for j, (i, x, y) in enumerate(todo):
                lines += [f"### Item {j}", f"REFERENCE: {cands[x][i]['hyp']}", f"HYPOTHESIS: {cands[y][i]['hyp'] or '(empty)'}", ""]
            out, _ = call(cl, model, J_CALIBRATED, "\n".join(lines), J_SCHEMA)
            vs = {v["i"]: v for v in out["verdicts"]} if out else {}
            for j, (i, x, y) in enumerate(todo):
                if j in vs:
                    v = {"meaning_preserved": vs[j]["meaning_preserved"], "discrepancies": vs[j]["discrepancies"]}
                    res[(i, x, y)] = v["meaning_preserved"]
                    with open(JCACHE, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"k": jcache_key(a.judge, "calibrated", cands[x][i]["hyp"], cands[y][i]["hyp"]), "v": v}, ensure_ascii=False) + "\n")
        return res

    with ThreadPoolExecutor(a.concurrency) as ex:
        agree = {k: v for part in ex.map(run, [pairs[i:i + 12] for i in range(0, len(pairs), 12)]) for k, v in part.items()}
    rows = []; chosen = Counter()
    for i in ids:
        # score(candidate c) = number of other candidates judged meaning-equivalent to c (both directions)
        score = {c: sum(agree.get((i, c, o), False) + agree.get((i, o, c), False) for o in a.tags if o != c) for c in a.tags}
        best = max(a.tags, key=lambda c: (score[c], -a.tags.index(c)))  # ties -> earlier tag (priority order)
        chosen[best] += 1
        rows.append(dict(cands[best][i], mbr_scores=score, mbr_pick=best))
    write_jsonl(OUT / a.name / "brh2en_dev.jsonl", rows)
    print(f"mbr {a.name}: {len(rows)} sentences, picks {dict(chosen)}")


# ---------------------------------------------------------------- bucket
B_SCHEMA = {"type": "object", "properties": {"verdicts": {"type": "array", "items": {"type": "object", "properties": {
    "i": {"type": "integer"}, "gloss": {"type": "string"},
    "bucket": {"type": "string", "enum": ["unknown_word", "loanword", "vowel_ambiguity", "morphology", "entity",
                                          "segmentation", "composition", "reference_error", "source_noise", "judge_error"]},
    "evidence": {"type": "string"}, "correct_english": {"type": "string"}},
    "required": ["i", "gloss", "bucket", "evidence", "correct_english"], "additionalProperties": False}}},
    "required": ["verdicts"], "additionalProperties": False}

B_SYSTEM = """You are a Brahui linguist (Dravidian language of Balochistan, Perso-Arabic script, heavy Balochi/Urdu/Persian loan vocabulary) auditing a machine translation into English. For each item you get the Brahui source, a human English reference, a system hypothesis, and the discrepancies a judge found between the two English texts.

Work word by word. First write a gloss of the span(s) of the source that the discrepancy concerns (Brahui word -> morphemes -> English). Then decide the PRIMARY cause of the error, exactly one bucket:
- unknown_word: a native Brahui content word was not known or was misread
- loanword: an Urdu/Balochi/Persian/English loan was misunderstood or its register mistaken
- vowel_ambiguity: unwritten short vowels made the word ambiguous and the wrong reading was chosen
- morphology: words known, but a case suffix, tense/aspect, negation, person, evidential or agreement marker was misread
- entity: a proper name, number, date or foreign term was wrong
- segmentation: wrong word or clause boundary (spacing, attached clitics, run-on sentence)
- composition: all words and morphemes known, but the sentence was put together with the wrong relations (agent/patient, scope, causality, attachment)
- reference_error: the hypothesis is actually faithful to the Brahui; the human reference is wrong or loose
- source_noise: the Brahui source itself is garbled/truncated so no faithful translation exists
- judge_error: hypothesis and reference mean the same thing; the discrepancy is not real

Give evidence as 'source word -> what it means -> what the system wrote'. Give the correct English for the span.
Return JSON {"verdicts": [...]} with one verdict per item, same i."""


def cmd_bucket(a):
    d = find_dir(a.tag); rows = {r["id"]: r for r in read_jsonl(d / "brh2en_dev.jsonl")}
    mp = read_jsonl(d / f"mp_{a.mp_judge}.jsonl")
    fails = [dict(rows[v["id"]], discrepancies=v["discrepancies"]) for v in mp if not v["meaning_preserved"]]
    print(f"{len(fails)} failures to bucket", flush=True)
    model = MODELS[a.judge]; cl = client()
    batches = [fails[i:i + a.batch] for i in range(0, len(fails), a.batch)]

    def run(batch):
        lines = []
        for i, r in enumerate(batch):
            disc = "; ".join(f"{x['kind']}: ref '{x['reference']}' vs hyp '{x['hypothesis']}' ({x['note']})" for x in r["discrepancies"]) or "(none listed)"
            lines += [f"### Item {i}", f"Brahui: {r['src']}", f"Reference: {r['ref']}", f"Hypothesis: {r['hyp'] or '(empty)'}",
                      f"Discrepancies: {disc}", ""]
        out, _ = call(cl, model, B_SYSTEM, "\n".join(lines), B_SCHEMA, max_tokens=12000)
        if not out: return []
        vs = {v["i"]: v for v in out["verdicts"]}
        return [{"id": r["id"], **{k: vs[i][k] for k in ["bucket", "gloss", "evidence", "correct_english"]}} for i, r in enumerate(batch) if i in vs]

    with ThreadPoolExecutor(a.concurrency) as ex:
        res = [v for part in ex.map(run, batches) for v in part]
    write_jsonl(d / "buckets.jsonl", res)
    c = Counter(r["bucket"] for r in res); n = len(res)
    print(f"{a.tag}: {n} failures bucketed")
    for b, k in c.most_common(): print(f"  {b:16s} {k:4d}  {100*k/n:5.1f}%")


def main():
    global DEV
    ap = argparse.ArgumentParser(); ap.add_argument("--dev", default=str(DEV), help="dev split (mt_dev.jsonl or mt_dev_ak.jsonl)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("translate"); t.add_argument("--model", default="opus", choices=MODELS); t.add_argument("--shots", type=int, default=40)
    t.add_argument("--per-request", type=int, default=20); t.add_argument("--concurrency", type=int, default=6)
    t.add_argument("--mode", default="fewshot", choices=["fewshot", "full"]); t.add_argument("--effort", default=None, choices=[None, "low", "medium", "high", "xhigh", "max"])
    t.add_argument("--tag", default=None); t.add_argument("--limit", type=int, default=0); t.add_argument("--lexicon", nargs="?", const="basic", default=None, choices=["basic", "shad", "all"]); t.add_argument("--ak-train", action="store_true"); t.add_argument("--input", default="arabic", choices=["arabic", "roman"]); t.add_argument("--context", type=int, default=0); t.add_argument("--gloss-first", action="store_true")
    j = sub.add_parser("judge"); j.add_argument("tag"); j.add_argument("--judge", default="opus", choices=MODELS)
    j.add_argument("--rubric", default="calibrated", choices=["strict", "calibrated"]); j.add_argument("--refs", default="orig", choices=["orig", "clean"]); j.add_argument("--batch", type=int, default=10); j.add_argument("--concurrency", type=int, default=6)
    ad = sub.add_parser("adjudicate"); ad.add_argument("--model", default="fable", choices=MODELS); ad.add_argument("--effort", default="xhigh")
    ad.add_argument("--limit", type=int, default=0); ad.add_argument("--per-request", type=int, default=5); ad.add_argument("--concurrency", type=int, default=4)
    ad.add_argument("--with-gloss", action="store_true"); ad.add_argument("--input", default="arabic", choices=["arabic", "roman"])
    bc = sub.add_parser("build-clean"); bc.add_argument("--judge", default="opus"); bc.add_argument("--limit", type=int, default=0)
    c = sub.add_parser("consensus"); c.add_argument("name"); c.add_argument("--tags", nargs="+", required=True); c.add_argument("--model", default="opus", choices=MODELS)
    c.add_argument("--effort", default="xhigh"); c.add_argument("--per-request", type=int, default=5); c.add_argument("--concurrency", type=int, default=4)
    m = sub.add_parser("mbr"); m.add_argument("name"); m.add_argument("--tags", nargs="+", required=True); m.add_argument("--judge", default="opus"); m.add_argument("--concurrency", type=int, default=6)
    b = sub.add_parser("bucket"); b.add_argument("tag"); b.add_argument("--judge", default="opus", choices=MODELS)
    b.add_argument("--mp-judge", default="opus"); b.add_argument("--batch", type=int, default=8); b.add_argument("--concurrency", type=int, default=6)
    a = ap.parse_args(); DEV = Path(a.dev) if Path(a.dev).is_absolute() else ROOT / a.dev
    {"translate": cmd_translate, "judge": cmd_judge, "bucket": cmd_bucket, "adjudicate": cmd_adjudicate, "build-clean": cmd_build_clean, "consensus": cmd_consensus, "mbr": cmd_mbr}[a.cmd](a)


if __name__ == "__main__":
    main()
