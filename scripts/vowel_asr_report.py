"""What limits the vowel model's blind accuracy? Reads evals/tts/vowel_asr/scores_f*.json (scripts/vowel_asr.py score).

  uv run python scripts/vowel_asr_report.py   ->  prints the breakdown, writes evals/tts/vowel_asr/report.json
"""
import collections, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; E = ROOT / "evals/tts/vowel_asr"; VOW = "aāiīuūeo"; LONG = {"ā": "a", "ī": "i", "ū": "u"}
types = json.load(open(ROOT / "data/tts/vowel_asr/types.json", encoding="utf-8")); res = {}
for f in sorted(E.glob("scores_f*.json")):
    d = json.load(open(f, encoding="utf-8"))
    for k, v in d["results"].items(): res.setdefault(k, []).append(v)
def norm(s): return "".join(c for i, c in enumerate(s) if c != "'" and not (i and c == s[i - 1] and c not in VOW))
def short(s): return "".join(LONG.get(c, c) for c in s)
def cons(s): return "".join(c for c in s if c not in VOW)
def kind(a, b):
    """how the model's spelling b differs from the dictionary's a"""
    a, b = norm(a), norm(b)
    if a == b: return "same"
    if short(a) == short(b): return "length only (a/ā, i/ī, u/ū)"
    if short(a).rstrip(VOW) == short(b).rstrip(VOW): return "word-final vowel"
    ea = short(a).replace("e", "i").replace("o", "u"); eb = short(b).replace("e", "i").replace("o", "u")
    if ea == eb: return "e/i or o/u"
    if cons(a) != cons(b): return "w/y/h reading (consonant letters differ)"
    va = [c for c in short(a) if c in VOW]; vb = [c for c in short(b) if c in VOW]
    if len(va) == len(vb): return "one short vowel quality" if sum(x != y for x, y in zip(va, vb)) == 1 else "several vowel qualities"
    return "vowel inserted or dropped"
def margin(v):   # gap between the winner and the runner-up, per recording (nats); and the share of recordings the winner takes
    t = v["top"]; return (-t[1][1] if len(t) > 1 else 9.0), t[0][2]
K = [(k, v[0]) for k, v in res.items() if k.endswith("\tr") and types[v[0]["w"]]["t"] != "u"]
tot = sum(types[v["w"]]["n"] for _, v in K); out = {}
print(f"hidden trusted words scored: {len(K)} types, {tot} occurrences in the corpus")
ok = [norm(v["final"]) == norm(types[v["w"]]["r"]) for _, v in K]
print(f"dictionary spelling survives: {sum(ok) / len(K):.1%} of types, {sum(types[v['w']]['n'] for (_, v), o in zip(K, ok) if o) / tot:.1%} of occurrences")
print("\n-- by number of recordings scored")
for lo, hi in ((1, 1), (2, 3), (4, 7), (8, 15), (16, 47), (48, 999)):
    g = [(v, o) for (_, v), o in zip(K, ok) if lo <= v["n"] <= hi]
    if g: print(f"  {lo:>3}-{hi:<3} recordings: {len(g):5d} types, survive {sum(o for _, o in g) / len(g):.1%}")
print("\n-- by Vowelizer source (types with >= 8 recordings)")
for key in ("lex/high", "lex/medium", "key/medium", "manual/medium"):
    g = [o for (_, v), o in zip(K, ok) if v["n"] >= 8 and f"{types[v['w']]['how']}/{types[v['w']]['conf']}" == key]
    if g: print(f"  {key:14s} {len(g):5d} types, survive {sum(g) / len(g):.1%}")
print("\n-- what the disagreements are (types with >= 8 recordings)")
kinds = collections.Counter(); kn = collections.Counter(); ex = collections.defaultdict(list); big = [(v, o) for (_, v), o in zip(K, ok) if v["n"] >= 8]
for v, o in big:
    if o: continue
    kd = kind(types[v["w"]]["r"], v["final"]); kinds[kd] += 1; kn[kd] += types[v["w"]]["n"]
    ex[kd].append((types[v["w"]]["n"], v["w"], types[v["w"]]["r"], v["final"], round(margin(v)[0], 2), margin(v)[1]))
nb = sum(types[v["w"]]["n"] for v, _ in big)
for kd, c in kinds.most_common():
    print(f"  {kd:42s} {c:4d} types ({c / len(big):.1%}), {kn[kd] / nb:.1%} of occurrences   e.g. " + "; ".join(f"{w} {a}→{b}" for _, w, a, b, _, _ in sorted(ex[kd], reverse=True)[:5]))
out["kinds"] = {k: {"types": c, "occ_share": round(kn[k] / nb, 4)} for k, c in kinds.items()}
soft = {"length only (a/ā, i/ī, u/ū)", "word-final vowel"}
print(f"\n  counting length-only and word-final-vowel differences as agreement: {sum(o or kind(types[v['w']]['r'], v['final']) in soft for v, o in big) / len(big):.1%} of types, "
      f"{sum(types[v['w']]['n'] for v, o in big if o or kind(types[v['w']]['r'], v['final']) in soft) / nb:.1%} of occurrences")
print("\n-- confidence: accuracy among the verdicts the model is surest of (types with >= 8 recordings; margin = winner minus runner-up, nats per recording)")
rank = sorted(big, key=lambda x: -margin(x[0])[0])
for cov in (0.2, 0.4, 0.6, 0.8, 1.0):
    g = rank[:int(len(rank) * cov)]; print(f"  top {cov:.0%} by margin: {sum(o for _, o in g) / len(g):.1%} survive (margin >= {margin(g[-1][0])[0]:.2f})")
print("\n-- when the model overturns the dictionary, how sure is it? (types with >= 16 recordings, largest margins)")
flips = sorted(((margin(v)[0], margin(v)[1], types[v["w"]]["n"], v["w"], types[v["w"]]["r"], v["final"], types[v["w"]]["how"]) for v, o in big if not o and v["n"] >= 16), reverse=True)
for m, s, n, w, a, b, how in flips[:30]: print(f"  {w:>12s}  dict {a:14s} model {b:14s} margin {m:5.2f}  wins {s:.0%} of recordings  n={n} ({how})")
SK = [(k, v[0]) for k, v in res.items() if k.endswith("\tsk") and types[v[0]["w"]]["t"] != "u" and v[0]["n"] >= 8]
if SK:
    oks = [norm(v["final"]) == norm(types[v["w"]]["r"]) for _, v in SK]
    print(f"\n-- from the consonant skeleton (>= 8 recordings): {sum(oks) / len(SK):.1%} of {len(SK)} types recovered exactly")
    kk = collections.Counter(kind(types[v["w"]]["r"], v["final"]) for (_, v), o in zip(SK, oks) if not o)
    for kd, c in kk.most_common(): print(f"  {kd:42s} {c:4d} ({c / len(SK):.1%})")
both = [(k, v) for k, v in res.items() if len(v) == 2 and k.endswith("\tr")]
if both:
    ag = [norm(v[0]["final"]) == norm(v[1]["final"]) for _, v in both]
    print(f"\n-- untrusted words scored by both models: {len(both)} types, the two models agree on {sum(ag) / len(both):.1%}")
json.dump(out, open(E / "report.json", "w"), ensure_ascii=False, indent=1)
