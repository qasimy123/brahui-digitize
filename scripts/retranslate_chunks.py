"""Chunk the curated real-Brahui corpus for the workflow teacher, and collect the workflow's results.

  uv run python scripts/retranslate_chunks.py build [--per-chunk 30] [--context 3]
      data/parallel/synthetic_brh_en.jsonl -> data/tmp/retrans_chunks/chunk_NNNN.json (id, brahui, old_en, context[], source)
      prints the JSON list of chunk paths (pass as the Workflow args)
  uv run python scripts/retranslate_chunks.py collect <journal.jsonl> [<journal2.jsonl> ...]
      -> data/parallel/retranslated_brh_en.jsonl  {id, source, brh, en (new), en_old, agree, hard_words, model}
"""
from __future__ import annotations

import glob, json, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("cmd"); _ap.add_argument("rest", nargs="*")
_ap.add_argument("--src", default="data/parallel/synthetic_brh_en.jsonl"); _ap.add_argument("--chunks", default="data/tmp/retrans_chunks")
_ap.add_argument("--out", default="data/parallel/retranslated_brh_en.jsonl"); _ap.add_argument("--per-chunk", type=int, default=30); _ap.add_argument("--context", type=int, default=3)
_A = _ap.parse_args()
SRC = ROOT / _A.src
CH = ROOT / _A.chunks
OUT = ROOT / _A.out


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def build(per_chunk=30, context=3):
    rows = read_jsonl(SRC)
    rows.sort(key=lambda r: r["id"])  # clip ids are ordered within a recording
    CH.mkdir(parents=True, exist_ok=True)
    for f in glob.glob(str(CH / "*.json")): os.remove(f)
    prefix = lambda i: re.sub(r"\d+$", "", i)
    items = []
    for k, r in enumerate(rows):
        ctx = [x["brh"] for x in rows[max(0, k - context):k] if prefix(x["id"]) == prefix(r["id"])]
        items.append({"id": r["id"], "source": r["source"], "brahui": r["brh"], "old_en": r["en"], "context": ctx})
    paths = []
    for c in range(0, len(items), per_chunk):
        p = CH / f"chunk_{c // per_chunk:04d}.json"
        json.dump(items[c:c + per_chunk], open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
        paths.append(str(p.relative_to(ROOT)))
    print(json.dumps(paths))
    print(f"{len(items)} sentences -> {len(paths)} chunks", file=sys.stderr)


def collect(journals):
    src = {r["id"]: r for r in read_jsonl(SRC)}
    new, agree, hard = {}, {}, {}
    for j in journals:
        for l in open(j, encoding="utf-8"):
            r = json.loads(l)
            if r.get("type") != "result" or not isinstance(r.get("result"), dict): continue
            res = r["result"]
            for t in res.get("translations", []): new[t["id"]] = t["en"]; hard[t["id"]] = t.get("hard_words", "")
            for v in res.get("verdicts", []): agree[v["id"]] = v["agree"]
    n = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for i, en in new.items():
            if i not in src or not en.strip(): continue
            f.write(json.dumps({"id": i, "source": src[i]["source"], "brh": src[i]["brh"], "en": en, "en_old": src[i]["en"],
                                "agree": agree.get(i), "hard_words": hard.get(i, ""), "model": "workflow-opus-5-5-retrieval"}, ensure_ascii=False) + "\n"); n += 1
    print(f"{n} retranslated rows -> {OUT.relative_to(ROOT)}; agree={sum(1 for i in new if agree.get(i))}, disagree={sum(1 for i in new if agree.get(i) is False)}, unverified={sum(1 for i in new if agree.get(i) is None)}")


if __name__ == "__main__":
    if _A.cmd == "build":
        build(_A.per_chunk, _A.context)
    else:
        collect(_A.rest)
