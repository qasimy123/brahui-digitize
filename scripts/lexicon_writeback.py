"""Lexicon write-back: harvest the teacher's per-sentence `hard_words` lookups into a growing dictionary.

  uv run python scripts/lexicon_writeback.py [--src data/parallel/retranslated_brh_en.jsonl ...] [--min-count 2]

Every teacher row carries `hard_words` = "form=gloss (source), form=gloss, ...". This aggregates them by surface form:
  data/lexicon/flywheel_brh_en.txt     resolved forms (>= --min-count confident sentences), top glosses with counts,
                                       tagged `novel` when the teacher said the dictionaries lacked it
  data/lexicon/flywheel_unresolved.txt forms the teacher marked uncertain in most of their sentences, ranked by
                                       frequency, with example ids: the targets for the next, higher-effort pass
  data/lexicon/flywheel_report.json    counts
Teacher agents should grep flywheel_brh_en.txt next to Shad; each round's output feeds the next.
"""
from __future__ import annotations
import argparse, collections, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--src", nargs="+", default=["data/parallel/retranslated_brh_en.jsonl", "data/parallel/retranslated_fineweb_partial.jsonl"])
ap.add_argument("--min-count", type=int, default=2); ap.add_argument("--exclude-id-prefix", default=None, help="skip rows whose id starts with this (held-out evaluation)"); ap.add_argument("--out", default="data/lexicon/flywheel")
a = ap.parse_args()

SPLIT = re.compile(r"[,;]\s+(?=[^=,;()]{1,40}=)")
UNSURE = re.compile(r"\?|unclear|unresolved|uncertain|garbl|guess|possibly|perhaps|probably|maybe|unknown|not sure|likely", re.I)
MISSING = re.compile(r"not in (shad|ak|the|any|dict)|no (dictionary|entry|hit)|shad has only|not found|unattested", re.I)
SOURCED = re.compile(r"shad|\bak\b|ak2024|ak_|gold|sarparra|shaigle|grammar|corpus|glossary", re.I)
ARABIC = re.compile(r"[؀-ۿ]")
PAREN = re.compile(r"\([^()]*\)")

def norm_form(s: str) -> str:
    s = s.strip().strip("'\"`“”‘’ ").replace("ك", "ک").replace("ي", "ی").replace("‌", " ")
    return re.sub(r"\s+", " ", s)

def norm_gloss(g: str) -> str:
    g = PAREN.sub("", g)
    g = re.sub(r"\s+", " ", g).strip(" .;,'\"`").lower()
    return g

forms: dict[str, dict] = {}
n_rows = n_items = 0
for src in a.src:
    p = ROOT / src
    if not p.exists():
        continue
    for line in open(p, encoding="utf-8"):
        r = json.loads(line); hw = r.get("hard_words")
        if a.exclude_id_prefix and r["id"].startswith(a.exclude_id_prefix):
            continue
        if not hw or not isinstance(hw, str):
            continue
        n_rows += 1
        for item in SPLIT.split(hw):
            if "=" not in item:
                continue
            form, gloss = item.split("=", 1); form = norm_form(form); gloss = gloss.strip()
            if not form or not gloss or len(form) > 40:
                continue
            n_items += 1
            e = forms.setdefault(form, {"n": 0, "unsure": 0, "missing": 0, "sourced": 0, "gloss": collections.Counter(), "raw": {}, "ids": []})
            e["n"] += 1
            if len(e["ids"]) < 3:
                e["ids"].append(r["id"])
            if UNSURE.search(gloss):
                e["unsure"] += 1; continue
            if MISSING.search(gloss): e["missing"] += 1
            elif SOURCED.search(gloss): e["sourced"] += 1
            g = norm_gloss(gloss)
            if g:
                e["gloss"][g] += 1; e["raw"].setdefault(g, gloss)

shad = set()
for line in open(ROOT / "data/lexicon/shad_brh_en.txt", encoding="utf-8"):
    if not line.startswith("#") and " (" in line:
        shad.add(norm_form(line.split(" (", 1)[0]))

resolved, unresolved = [], []
for form, e in forms.items():
    sure = e["n"] - e["unsure"]
    if sure >= a.min_count and e["unsure"] <= sure:
        resolved.append((form, e))
    elif e["unsure"] > sure:
        unresolved.append((form, e))

def tag(form, e):
    if form in shad: return "shad-headword"
    if e["missing"]: return "novel"  # the teacher said the dictionaries lacked it and resolved it from context
    return "form"

out = ROOT / a.out
resolved.sort(key=lambda x: (not ARABIC.search(x[0]), x[0]))
tags = collections.Counter()
with open(f"{out}_brh_en.txt", "w", encoding="utf-8") as f:
    f.write(f"# Flywheel lexicon: surface forms resolved by the retrieval teacher (scripts/lexicon_writeback.py), {len(resolved)} forms.\n"
            "# Format: form | gloss [sentences]; gloss [sentences] | n=<sentences> <shad-headword|form|novel>. Inflected forms kept as seen.\n")
    for form, e in resolved:
        t = tag(form, e); tags[t] += 1
        gl = "; ".join(f"{e['raw'][g]} [{c}]" for g, c in e["gloss"].most_common(3))
        f.write(f"{form} | {gl} | n={e['n'] - e['unsure']} {t}\n")
unresolved.sort(key=lambda x: -x[1]["n"])
with open(f"{out}_unresolved.txt", "w", encoding="utf-8") as f:
    f.write(f"# Forms the teacher marked uncertain in most sentences ({len(unresolved)}), by frequency. form | n | best guesses | example ids\n")
    for form, e in unresolved:
        gl = "; ".join(f"{e['raw'][g]} [{c}]" for g, c in e["gloss"].most_common(2))
        f.write(f"{form} | {e['n']} | {gl} | {' '.join(e['ids'])}\n")
rep = {"rows_with_hard_words": n_rows, "items": n_items, "distinct_forms": len(forms), "resolved": len(resolved), "resolved_by_tag": dict(tags),
       "unresolved": len(unresolved), "unresolved_n>=3": sum(1 for _, e in unresolved if e["n"] >= 3), "singletons_skipped": sum(1 for e in forms.values() if e["n"] == 1)}
json.dump(rep, open(f"{out}_report.json", "w"), indent=1); print(json.dumps(rep, indent=1))
