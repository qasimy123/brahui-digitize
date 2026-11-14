"""Write every utterance of the vowel-model data with vowels as heard: the decision rule on top of the blind acoustic scores.

Inputs (scripts/vowel_asr.py): evals/tts/vowel_asr/scores_f{0,1}.json (per word type: candidate spellings ranked on up to
48 recordings) and dump_f{0,1}.jsonl (per occurrence: log-likelihood of each candidate). A model only ever judges words it
never saw spelled: trusted words by the model of their own fold, untrusted words by both.

Rule, per word type first:
  trusted (dictionary) spelling   kept, unless the blind model prefers another spelling decisively (>= 8 recordings,
                                  >= MARGIN nats per recording over the runner-up, winner of >= 60% of recordings) AND the
                                  change keeps the consonants and does not just drop a vowel (speakers reduce unstressed
                                  vowels: "zabrdast"; the careful form stays).
  untrusted spelling (a guess)    replaced by the models' winner when both agree; otherwise by the leader of their pooled
                                  per-recording scores if it leads by >= 1 nat; otherwise the guess stays.
  no type-level score (rare)      the best candidate pooled over the word's occurrences if it leads by >= 2 nats, else as is.
then per occurrence: another spelling replaces the type's choice when this recording prefers it by >= TAU nats, it is an
established second form (wins >= 15% of the type's recordings) and it does not drop a vowel: homographs and real variation.
Never: a spelling without its initial vowel for a word written with initial alif / hamza / ain.

  uv run python scripts/vowel_retranscribe.py  ->  data/tts/vowel_asr/retrans.jsonl, data/lexicon/vowel_heard.json, evals/tts/vowel_asr/retrans_report.json
"""
import collections, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; D = ROOT / "data/tts/vowel_asr"; E = ROOT / "evals/tts/vowel_asr"
VOW = "aāiīuūeo"; MARGIN = 3.0; TAU = 4.0
types = json.load(open(D / "types.json", encoding="utf-8")); sc = [json.load(open(E / f"scores_f{f}.json", encoding="utf-8"))["results"] for f in (0, 1)]
def norm(s): return "".join(c for i, c in enumerate(s) if c != "'" and not (i and c == s[i - 1] and c not in VOW))
def cons(s): return "".join(c for c in norm(s) if c not in VOW)
def nvow(s): return sum(c in VOW for c in s)
def margin(v): t = v["top"]; return (-t[1][1] if len(t) > 1 else 9.0), t[0][2]
occ = collections.defaultdict(dict); pooled = collections.defaultdict(lambda: collections.defaultdict(list))
for f in (0, 1):
    for l in open(E / f"dump_f{f}.jsonl", encoding="utf-8"):
        d = json.loads(l); o = occ[(d["id"], d["i"])]
        for c, x in zip(d["c"], d["ll"]): o.setdefault(c, []).append(x); pooled[d["w"]][c].append(x)
# fixed forms the vowel search cannot reach (doubled consonants, a final h): the Arabic blessing formula
KNOWN = {"صلی": "salla", "علیہ": "alaihi", "وسلم": "wasallam", "خوش": "khush", "خوشی": "khushī", "صلاح": "salāh"}   # the last three: Persian/Arabic loans the model got wrong (khash, silāh)
def initial_ok(w, s):   # a word written with an initial alif / hamza / ain starts with a vowel: the model sometimes swallows it ("anā" -> "nā")
    return not (w[0] in "اآأإئعٱ") or (s[:1] in VOW or s[:1] == "'")
def typed(w):
    if w in KNOWN: return KNOWN[w], "fixed form", set()
    c, y, est = typed0(w)
    if not initial_ok(w, c):
        r = types[w]["r"]; return (r if initial_ok(w, r) else c), "guess kept (initial vowel)", est
    return c, y, est
def typed0(w):
    """-> (spelling for this word type, why, spellings established as real second forms)"""
    t = types[w]; r = t["r"]; res = [s[w + "\tr"] for s in sc if w + "\tr" in s]; est = set()
    for v in res: est |= {x[0] for x in v["top"] if x[2] >= 0.15}
    if t["t"] != "u":
        v = res[0] if res else None
        if v and v["final"] != r and norm(v["final"]) != norm(r):
            m, share = margin(v)
            if v["n"] >= 8 and m >= MARGIN and share >= 0.6 and cons(v["final"]) == cons(r) and nvow(v["final"]) >= nvow(r): return v["final"], "heard (overrides dictionary)", est
        return r, "dictionary", est
    if len(res) == 2 and norm(res[0]["final"]) == norm(res[1]["final"]): return res[0]["final"], "heard (both models)", est
    if res:
        # the models disagree (they agree on 54% of these words), or only one scored it: pool both models' log-likelihoods over
        # the spellings both scored on every occurrence, and take a clear leader
        pl = pooled.get(w, {}); full = max((len(x) for x in pl.values()), default=0); p = {c: sum(x) / len(x) for c, x in pl.items() if len(x) == full}
        if len(p) >= 2:
            o = sorted(p, key=p.get, reverse=True)
            if p[o[0]] - p[o[1]] >= 1.0: return o[0], "heard (models combined)", est
        v = max(res, key=lambda v: margin(v)[0])
        if len(res) == 1 and margin(v)[0] >= 1.0: return v["final"], "heard (one model)", est
        return v["start"], "guess kept", est
    p = {c: sum(x) / len(x) for c, x in pooled.get(w, {}).items()}
    if p:
        o = sorted(p, key=p.get, reverse=True); r0 = o[0] if r not in p else r
        if o[0] != r0 and p[o[0]] - p[r0] >= 2.0: return o[0], "heard (rare word)", est
        return r0, "guess kept", est
    return r, "guess kept (not scored)", est
cache = {}; why = collections.Counter(); changed = collections.Counter(); heard = collections.defaultdict(collections.Counter); tok = 0; n_occ = 0
for name in ("train", "dev"):
    rows = [json.loads(l) for l in open(D / f"{name}.jsonl", encoding="utf-8")]
    with open(D / ("retrans.jsonl" if name == "train" else "retrans_dev.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            out = []
            for i, (w, rom, t) in enumerate(r["words"]):
                if w not in cache: cache[w] = typed(w)
                c, y, est = cache[w]; o = occ.get((r["id"], i))
                if o and c in o:
                    s = {k: sum(v) / len(v) for k, v in o.items()}; b = max(s, key=s.get)
                    if b != c and s[b] - s[c] >= TAU and b in est and cons(b) == cons(c) and nvow(b) >= nvow(c) and initial_ok(w, b) and w not in KNOWN: c = b; y = "heard (this recording)"; n_occ += 1
                out.append([w, c, y]); why[y] += 1; tok += 1; heard[w][c] += 1
                if norm(c) != norm(rom): changed[(w, rom, c)] += 1
            f.write(json.dumps({"id": r["id"], "audio": r["audio"], "duration": r["duration"], "source": r["source"], "words": out}, ensure_ascii=False) + "\n")
nch = sum(changed.values())
print(f"{tok} word occurrences; spelling differs from the rule-based Vowelizer in {nch} ({nch / tok:.1%})")
for y, n in why.most_common(): print(f"  {y:32s} {n:7d}  {n / tok:.1%}")
print("most frequent changes:"); top = changed.most_common(40)
for (w, a, b), n in top: print(f"  {w:>12s}  {a:14s} -> {b:14s} {n}")
json.dump({w: dict(c.most_common()) for w, c in heard.items()}, open(ROOT / "data/lexicon/vowel_heard.json", "w", encoding="utf-8"), ensure_ascii=False)
json.dump({"tokens": tok, "changed": nch, "by_reason": dict(why), "top_changes": [[w, a, b, n] for (w, a, b), n in changed.most_common(300)]}, open(E / "retrans_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
