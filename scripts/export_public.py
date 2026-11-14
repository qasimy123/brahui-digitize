"""Build the public copy of this repository from an explicit allowlist.

    uv run python scripts/export_public.py                # writes ../brahui-digitize-public
    uv run python scripts/export_public.py --out /tmp/x   # dry run somewhere else

Only git-tracked files that match ALLOW and do not match DENY are copied. Files under release/public/ are
laid over the root (README, DATA.md, LICENSE, ...). Parallel-data files are rewritten row by row (media rows and
dictionary-quoting fields dropped). The output is then scanned; any hit fails the export, and nothing should be
pushed until the scan is clean. Anything not listed here stays private by default.
"""
import argparse, fnmatch, json, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ALLOW = [
    "pyproject.toml", "uv.lock", ".python-version",
    "brahui/*", "scripts/*", "tools/*", "docs/*", "tests/*",
    "demo/web/*", "demo/hf_space/*",
    "data/eval/*",
    "evals/RESULTS_TABLES.md", "evals/asr/*", "evals/mt/*", "evals/st/*",
    "evals/interp/*", "evals/cycle/compare.json",
    "evals/media_pilot/report.json", "evals/media_pilot/vocal_share.json",
    "evals/tts/*/summary.json", "evals/tts/student/*.json",
]
DENY = [
    "docs/NEXT_SESSION_PROMPT.md", "docs/RECORDING.md",       # internal / private-speaker context
    "data/eval/ak2024*",                                       # Ali & Kobayashi sentences: rights with the authors
    "evals/mt/round0/*",                                       # judge caches quote A&K and dictionary text
    "evals/mt/hillclimb/*/traces/*", "evals/mt/hillclimb/*/results.jsonl",
    "evals/tts/voices/*",                                      # reference recordings
    "*.wav", "*.flac", "*.mp3", "*.m4a", "*.pt", "*.safetensors", "*.onnx",
]
# Parallel data: file -> fields to drop. Rows whose source is in DROP_SOURCES are removed.
PARALLEL = {
    "data/parallel/gold_train.jsonl": [],
    "data/parallel/gold_train_nodev.jsonl": [],
    "data/parallel/synthetic_brh_en.jsonl": [],
    "data/parallel/retranslated_brh_en.jsonl": ["hard_words"],     # quotes Shad / A&K dictionary glosses
    "data/parallel/synthetic_retranslated_brh_en.jsonl": ["hard_words"],
    "data/parallel/synthetic_fineweb_brh_en_filtered.jsonl": [],
}
DROP_SOURCES = {"media"}  # transcripts of broadcast audio are published separately, never mixed in here

# Text rewrites applied to exported code and docs (not to data files).
REDACT = [
    (re.compile(r"[Pp]od `[a-z0-9]{12,16}`,?\s*"), ""),          # RunPod pod ids
    (re.compile(r"\b([Pp]od) (?=[a-z0-9]*\d)[a-z0-9]{13,15}\b"), r"\1"),
    (re.compile(r"\b43vit6a9b8\b"), "<volume-id>"),                # RunPod network volume
    (re.compile(r"\bwf_[0-9a-f]{8}-[0-9a-f]{3}\b"), "<run-id>"),      # agent workflow run ids
    (re.compile(r"/Users/[A-Za-z0-9_]+/Programming/brahui-digitize/?"), ""),
]
# Scan: (name, regex, paths it applies to). Any match fails the export.
OK_EMAILS = {"info@brahuiacademy.pk", "noreply@anthropic.com", "git@github.com"}
SCAN = [
    ("token", re.compile(r"hf_[A-Za-z0-9]{20,}|sk-ant-[A-Za-z0-9_-]{20,}|rpa_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}")),
    ("home path", re.compile(r"/Users/[A-Za-z0-9_]+|/home/(?!user\b)[a-z][a-z0-9_]+/")),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}")),
]
_terms = ROOT / "release" / "private_terms.txt"   # private names; this file is never exported
PRIVATE = [t.strip() for t in _terms.read_text().splitlines() if t.strip() and not t.startswith("#")] if _terms.exists() else []
if PRIVATE:
    SCAN.append(("private term", re.compile("|".join(re.escape(t) for t in PRIVATE), re.I)))
TEXT_EXT = {".py", ".md", ".txt", ".json", ".jsonl", ".js", ".mjs", ".html", ".sh", ".toml", ".yml", ".yaml", ".csv", ".tsv", ".cfg", ""}
MAX_BYTES = 50_000_000


def match(path, patterns):
    return any(fnmatch.fnmatch(path, p) or fnmatch.fnmatch(path, p.rstrip("*") + "*") and p.endswith("/*") for p in patterns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "brahui-digitize-public"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    tracked = [t for t in tracked if t]
    plan = {}  # dest relative path -> source Path
    for t in tracked:
        if t.startswith("release/public/"):
            plan[t[len("release/public/"):]] = ROOT / t
        elif t.startswith("release/cards/"):
            plan["hf_cards/" + t[len("release/cards/"):]] = ROOT / t
        elif t in PARALLEL or (match(t, ALLOW) and not match(t, DENY)):
            plan[t] = ROOT / t
    missing = [p for p in PARALLEL if p not in plan]
    if missing:
        sys.exit(f"parallel files not tracked: {missing}")

    # remove everything in out except .git, then write the plan
    for child in out.iterdir():
        if child.name == ".git":
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()

    stats = {"files": 0, "bytes": 0, "rows_dropped": 0, "redactions": 0}
    for rel, src in sorted(plan.items()):
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel in PARALLEL:
            drop = PARALLEL[rel]
            with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
                for line in f:
                    r = json.loads(line)
                    if r.get("source") in DROP_SOURCES:
                        stats["rows_dropped"] += 1
                        continue
                    for k in drop:
                        r.pop(k, None)
                    g.write(json.dumps(r, ensure_ascii=False) + "\n")
        elif src.suffix in TEXT_EXT and not rel.startswith("data/"):
            try:
                text = src.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                shutil.copy2(src, dst)
            else:
                for rx, rep in REDACT:
                    text, n = rx.subn(rep, text)
                    stats["redactions"] += n
                dst.write_text(text, encoding="utf-8")
        else:
            shutil.copy2(src, dst)
        stats["files"] += 1
        stats["bytes"] += dst.stat().st_size

    # scan what was written
    hits = []
    for rel in sorted(plan):
        p = out / rel
        if p.stat().st_size > MAX_BYTES:
            hits.append((rel, "size", f"{p.stat().st_size/1e6:.0f} MB"))
        if PRIVATE and any(t.lower() in rel.lower() for t in PRIVATE):
            hits.append((rel, "private term", "in file name"))
        if p.suffix not in TEXT_EXT:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        is_corpus = rel.startswith(("data/parallel/", "data/eval/", "evals/mt/nllb/", "evals/mt/serve/", "evals/interp/bouquet"))
        for name, rx in SCAN:
            if name == "private term" and is_corpus:
                continue  # ordinary word inside translated sentences
            for m in rx.finditer(text):
                s = m.group(0)
                if name == "email" and (s.lower() in OK_EMAILS or is_corpus):
                    continue
                line = text.count("\n", 0, m.start()) + 1
                hits.append((rel, name, f"line {line}: {s if name != 'token' else '<redacted>'}"))
                break
    print(f"exported {stats['files']} files, {stats['bytes']/1e6:.1f} MB to {out}")
    print(f"media rows dropped: {stats['rows_dropped']}, redactions applied: {stats['redactions']}")
    if hits:
        print(f"\nSCAN FAILED: {len(hits)} hit(s). Do not push.")
        for h in hits:
            print("  %s  [%s]  %s" % h)
        sys.exit(1)
    print("scan clean")


if __name__ == "__main__":
    main()
