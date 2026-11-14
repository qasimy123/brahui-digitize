#!/usr/bin/env python3
"""Build data/eval/ak2024_sentences.jsonl from Ali & Kobayashi 2024, *Brahui Texts*.

Inputs
  data/raw/ali_kobayashi_2024/text_brahui.txt   clean Brahui tier (ASCII transliteration),
                                                 one sentence per line, keyed by timestamp/number
  data/raw/ali_kobayashi_2024/BrahuiTexts.pdf    the book (romanized Brahui / gloss / English)

The sentence keys in text_brahui.txt are used as anchors to segment the pdftotext dump of the
book into sentence blocks; each block is then split into Brahui / gloss / English lines.

Usage:  uv run python scripts/parse_ak2024.py [--pdf-txt PATH] [--seed N]
Idempotent: regenerates the pdftotext dump only if missing, rewrites the output file.
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ali_kobayashi_2024"
TXT = RAW / "text_brahui.txt"
PDF = RAW / "BrahuiTexts.pdf"
OUT = ROOT / "data" / "eval" / "ak2024_sentences.jsonl"

CHAPTER_NO = {"Transcripts of Folktales": 3, "Novels": 2}

# ASCII transliteration used in text_brahui.txt (inside <...> in headers) -> phonemic.
TRANSLIT = [
    ("SH", "š"), ("GH", "ɣ"), ("LH", "ɬ"), ("aa", "ā"), ("ii", "ī"), ("uu", "ū"),
    ("ee", "ē"), ("oo", "ō"), ("T", "ṭ"), ("D", "ḍ"), ("R", "ṛ"), ("N", "ṇ"),
]


def translit(s: str) -> str:
    for a, b in TRANSLIT:
        s = s.replace(a, b)
    return s


def clean_title(raw: str) -> str:
    """'\\section{Haebat Kh<aa>n, \\textit{Traveler} (\\textit{mus<aa>fir})}' -> 'Haebat Khān, Traveler (musāfir)'."""
    m = re.search(r"\\(?:section|textbf)\{(.*)\}", raw)
    s = m.group(1) if m else raw
    s = re.sub(r"\\label\{[^}]*\}?", "", s)
    s = re.sub(r"\\medskip", "", s)
    s = re.sub(r"\\textit\{", "", s)
    s = re.sub(r"<([^>]*)>", lambda m: translit(m.group(1)), s)
    s = s.replace("\\\\", "").replace("\\ ", " ").replace("\\'", "'")
    s = s.replace("}", "").replace("{", "")
    s = re.sub(r"\s+", " ", s).strip().rstrip(".")
    # balance a stray paren from "(\textit{m<aa>r-b<aa>va)}}"
    if s.count("(") > s.count(")"):
        s += ")"
    return s


def ascii_fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower())


# ----------------------------------------------------------------------------- keys
TS_RE = re.compile(r"^(\d+)[’'.,](\d\d)$")


def norm_key(k: str) -> tuple[str, str | None]:
    """Return (canonical key, timestamp or None). Canonical: '12' or "0'11"."""
    k = k.strip()
    m = TS_RE.match(k)
    if m:
        ts = f"{int(m.group(1))}'{m.group(2)}"
        return ts, ts
    m = re.match(r"^(\d+)\.?$", k)
    if m:
        return str(int(m.group(1))), None
    return k, None


PDF_KEY_RE = re.compile(r"^\s*(\d+[’.]\d\d|\d+\.)(?=\s|$)")


# ----------------------------------------------------------------------------- text tier
def read_txt(path: Path):
    """Yield sections: dict(chapter, title, meta, subsections=[(title, [(key, ascii_sentence)])])."""
    sections = []
    chapter = None
    cur = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            body = line[1:]
            if body.startswith("\\chapter{"):
                chapter = clean_title(body.replace("\\chapter", "\\section"))
            elif body.startswith("\\section{"):
                cur = {"chapter": chapter, "title": clean_title(body), "meta": "", "subsections": [(None, [])]}
                sections.append(cur)
            elif body.startswith("\\textbf{"):
                cur["subsections"].append((clean_title(body), []))
            else:
                cur["meta"] += body
            continue
        parts = line.rstrip("\n").split("\t")
        key, ts = norm_key(parts[0])
        words = [w for w in parts[1:] if w.strip()]
        cur["subsections"][-1][1].append((key, ts, " ".join(words)))
    for s in sections:
        if len(s["subsections"]) > 1 and not s["subsections"][0][1]:
            s["subsections"].pop(0)
    return sections


# ----------------------------------------------------------------------------- pdf dump
def pdf_lines(pdf_txt: Path) -> list[str]:
    if not pdf_txt.exists():
        pdf_txt.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pdftotext", "-layout", str(PDF), str(pdf_txt)], check=True)
    text = pdf_txt.read_text(encoding="utf-8")
    pages = text.split("\f")
    out: list[str] = []
    RUN_HDR = re.compile(r"^\s*(\d+\s+Chapter \d\.|[23]\.\d+\.\s.*\s\d+\s*$)")
    for page in pages:
        lines = page.split("\n")
        kept = []
        for i, ln in enumerate(lines):
            if RUN_HDR.match(ln):
                continue
            if re.match(r"^\s*\d{1,3}\s*$", ln):
                rest = "".join(lines[i + 1:]).strip()
                if rest:  # footnote marker: drop the footnote and everything after it on this page
                    break
                continue  # bare page number
            kept.append(ln)
        out.extend(kept)
        out.append("")  # page boundary
    return out


HEADING_RE = re.compile(r"^\s*([23]\.\d+)\s+(\S.*?)\s*$")
TAG_RE = re.compile(r"(?<![A-Za-z])(?:[1-3](?:SG|PL)|[A-Z]{2,}(?:\.[A-Z0-9]+)*|D/A)(?![a-z])")
SUBHEAD_RE = re.compile(r"^\s*\d+\.\s+[A-Z][a-z]")
DIACRITICS = re.compile(r"[āīūēōšɣṭḍṛɬṇ]")


def is_gloss(line: str) -> bool:
    return bool(TAG_RE.search(line))


def fold_pdf(s: str) -> set[str]:
    s = s.replace("ɣ", "g").replace("ɬ", "l").replace("ʰ", "h").replace("š", "s")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return set(re.findall(r"[a-z]+", s))


def fold_ascii(s: str) -> set[str]:
    for a, b in [("SH", "s"), ("GH", "g"), ("LH", "l")]:
        s = s.replace(a, b)
    s = s.lower()
    for a, b in [("aa", "a"), ("ii", "i"), ("uu", "u"), ("ee", "e"), ("oo", "o")]:
        s = s.replace(a, b)
    return set(re.findall(r"[a-z]+", s))


def overlap(line: str, ref: set[str]) -> float:
    """Fraction of the line's folded tokens that occur in the ASCII tier's folded tokens."""
    toks = fold_pdf(line)
    if not toks:
        return 0.0
    return len(toks & ref) / len(toks)


def content_score(line: str, brh_ascii: str, ref: set[str]) -> float:
    """Token overlap, or character similarity of the folded strings (robust to spelling variants)."""
    a = "".join(fold_pdf_seq(line))
    b = "".join(fold_ascii_seq(brh_ascii))
    ratio = difflib.SequenceMatcher(None, a, b[: len(a) + 6]).ratio() if a else 0.0
    return max(overlap(line, ref), ratio)


def fold_pdf_seq(s: str) -> list[str]:
    s = s.replace("ɣ", "g").replace("ɬ", "l").replace("ʰ", "h").replace("š", "s")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.findall(r"[a-z]+", s)


def fold_ascii_seq(s: str) -> list[str]:
    for a, b in [("SH", "s"), ("GH", "g"), ("LH", "l")]:
        s = s.replace(a, b)
    s = s.lower()
    for a, b in [("aa", "a"), ("ii", "i"), ("uu", "u"), ("ee", "e"), ("oo", "o")]:
        s = s.replace(a, b)
    return re.findall(r"[a-z]+", s)


def find_headings(lines: list[str]):
    heads = []
    toc = {}  # title prefix -> section number, from the table of contents
    last = None
    for ln in lines[:400]:
        m = re.match(r"^\s*(?:([23]\.\d+)\s+)?(\S.*?)\s+\. \. \.", ln)
        if m and (m.group(1) or last):
            if m.group(1):
                last = m.group(1)
            else:  # unnumbered entry: next number in the chapter
                ch, n = last.split(".")
                last = f"{ch}.{int(n) + 1}"
            toc[m.group(2)[:20]] = last
    for i, ln in enumerate(lines):
        if ln.strip()[:20] in toc and i > 400 and not re.search(r"\d\s*$", ln):
            heads.append((i, toc[ln.strip()[:20]], ln.strip()))
            continue
        if re.match(r"^\s*[23]\.\d+\s*$", ln) and i + 1 < len(lines) and lines[i + 1].strip():
            heads.append((i, ln.strip(), lines[i + 1].strip()))
            continue
        m = HEADING_RE.match(ln)
        if not m or re.search(r"\d\s*$", ln) or ". . ." in ln:
            continue
        heads.append((i, m.group(1), m.group(2)))
    return heads


def split_block(block: list[str], ref: set[str]):
    """Classify lines of one sentence block into brahui / gloss / english.

    Layout is strictly Brahui line -> gloss line (-> Brahui -> gloss ...) -> English prose.
    A gloss line always follows a Brahui line; a Brahui line is recognised by token overlap
    with the ASCII tier (ref), with uppercase category tags as a fallback for gloss lines.
    """
    brh, gloss, en = [], [], []
    kinds = []
    for i, ln in enumerate(block):
        if i == 0:
            kinds.append("b")
        elif kinds[-1] == "b":
            kinds.append("e" if ln.lstrip().lstrip("“\"").startswith("(") and not is_gloss(ln) else "g")  # "(Same as 9'18 to 9'32)"
        elif kinds.count("e"):
            kinds.append("e")
        elif overlap(ln, ref) >= 0.5:
            kinds.append("b")
        elif not is_gloss(ln) and i + 1 < len(block) and is_gloss(block[i + 1]):
            kinds.append("b")  # Brahui line with few known tokens; next line is its gloss
        else:
            kinds.append("e")
    for k, ln in zip(kinds, block):
        s = re.sub(r"\s+", " ", ln).strip()
        {"b": brh, "g": gloss, "e": en}[k].append(s)
    en_s = ""
    for s in en:
        if en_s.endswith("-") and s[:1].islower():
            en_s = en_s[:-1] + s
        else:
            en_s = (en_s + " " + s) if en_s else s
    en_s = re.sub(r"(?<=[A-Za-z.,!?”)\"'])\d{1,2}(?=\s|$)", "", en_s)  # footnote markers
    return " ".join(brh), " ".join(gloss), en_s.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-txt", type=Path, default=ROOT / "data" / "raw" / "ali_kobayashi_2024" / "BrahuiTexts.layout.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    sections = read_txt(TXT)
    lines = pdf_lines(args.pdf_txt)
    heads = find_headings(lines)
    heads_by_ch = defaultdict(list)
    for h in heads:
        heads_by_ch[int(h[1].split(".")[0])].append(h)

    # --- match txt sections to PDF headings (in order, allowing a PDF section with no txt tier)
    ptr = {2: 0, 3: 0}
    skipped = []
    for s in sections:
        ch = CHAPTER_NO[s["chapter"]]
        cands = heads_by_ch[ch][ptr[ch]:ptr[ch] + 2]
        if not cands:
            s["pdf"] = None
            continue
        ratios = [difflib.SequenceMatcher(None, ascii_fold(s["title"]), ascii_fold(c[2])).ratio() for c in cands]
        pick = max(range(len(cands)), key=lambda i: ratios[i])
        if pick > 0:
            skipped.extend(cands[:pick])
        s["pdf"] = cands[pick]
        s["chapter_no"] = ch
        ptr[ch] += pick + 1
    skipped.extend(heads_by_ch[2][ptr[2]:] + heads_by_ch[3][ptr[3]:])

    # section end = next PDF heading (by line index)
    head_idx = sorted(h[0] for h in heads)

    rows = []
    report = []
    for s in sections:
        if s["pdf"] is None:
            report.append((s["title"], "NO PDF HEADING", 0, sum(len(x[1]) for x in s["subsections"])))
            continue
        start = s["pdf"][0]
        nxt = [i for i in head_idx if i > start]
        end = nxt[0] if nxt else len(lines)
        sec_no = s["pdf"][1]
        pos = start + 1
        n_ok = n_total = 0
        has_subs = len(s["subsections"]) > 1
        for si, (sub_title, sents) in enumerate(s["subsections"], 1):
            text_no = f"{sec_no}.{si}" if has_subs else sec_no
            seen = Counter()
            pending = []  # (row, key_line_idx)
            for key, ts, brh_ascii in sents:
                n_total += 1
                seen[key] += 1
                sid = f"{text_no}:{key}" + (f"#{seen[key]}" if seen[key] > 1 else "")
                row = {"id": sid, "text": s["title"], "chapter": s["chapter_no"], "brh": "", "gloss": "",
                       "en": "", "timestamp": ts, "brh_ascii": brh_ascii, "quality": "ok"}
                if has_subs:
                    row["subsection"] = sub_title
                # search for the key line
                found = None
                ref = fold_ascii(brh_ascii)
                exact = None
                for j in range(pos, end):
                    m = PDF_KEY_RE.match(lines[j])
                    if m and not SUBHEAD_RE.match(lines[j]) and norm_key(m.group(1))[0] == key:
                        exact = j
                        break
                # candidates: the exact key line plus the next key lines after pos; pick by content
                cands = [j for j in range(pos, min(pos + 40, end))
                         if PDF_KEY_RE.match(lines[j]) and not SUBHEAD_RE.match(lines[j])]
                if exact is not None and exact not in cands:
                    cands.append(exact)
                scored = sorted(((content_score(PDF_KEY_RE.sub("", lines[j], count=1), brh_ascii, ref), j == exact, -j)
                                 for j in cands), reverse=True)
                if scored and scored[0][0] >= 0.6:
                    found = -scored[0][2]
                    m = PDF_KEY_RE.match(lines[found])
                    if norm_key(m.group(1))[0] != key:
                        row["pdf_key"] = m.group(1).replace("’", "'")
                elif exact is not None and cands and exact == cands[0] and \
                        content_score(PDF_KEY_RE.sub("", lines[exact], count=1), brh_ascii, ref) >= 0.3:
                    found = exact  # same key, content disagrees: keep PDF text, flag it
                    row["quality"] = "suspicious:brh_mismatch"
                if found is None:
                    row["quality"] = "unaligned"
                    rows.append(row)
                    continue
                pending.append((row, found, ref))
                pos = found + 1
            # cut blocks between consecutive key lines
            for i, (row, j, ref) in enumerate(pending):
                j_end = pending[i + 1][1] if i + 1 < len(pending) else end
                block = []
                for ln in lines[j:j_end]:
                    if not ln.strip() or SUBHEAD_RE.match(ln) and ln is not lines[j]:
                        continue
                    if PDF_KEY_RE.match(ln) and ln is not lines[j]:
                        break  # another key we did not expect: stop block here
                    block.append(ln)
                block[0] = PDF_KEY_RE.sub("", block[0], count=1)
                row["brh"], row["gloss"], row["en"] = split_block(block, ref)
                n_ok += 1
            # keep txt order: rows for pending were appended lazily; append them now in order
            for row, _, _ in pending:
                rows.append(row)
        report.append((s["title"], sec_no, n_ok, n_total))

    # restore txt order (unaligned rows were appended before their aligned neighbours)
    by_id = {r["id"]: r for r in rows}
    ordered = []
    for s in sections:
        if s["pdf"] is None:
            continue
        sec_no = s["pdf"][1]
        has_subs = len(s["subsections"]) > 1
        for si, (_, sents) in enumerate(s["subsections"], 1):
            text_no = f"{sec_no}.{si}" if has_subs else sec_no
            seen = Counter()
            for key, _, _ in sents:
                seen[key] += 1
                ordered.append(by_id[f"{text_no}:{key}" + (f"#{seen[key]}" if seen[key] > 1 else "")])
    rows = ordered

    # --- quality flags
    for r in rows:
        if r["quality"] == "unaligned":
            continue
        flags = []
        if not r["en"]:
            flags.append("no_en")
        if not r["gloss"]:
            flags.append("no_gloss")
        if not r["brh"]:
            flags.append("no_brh")
        if r["en"] and len(DIACRITICS.findall(r["en"])) > 3 and "-" in r["en"]:
            flags.append("en_looks_brahui")
        if r["brh"] and not DIACRITICS.search(r["brh"]) and "-" not in r["brh"]:
            flags.append("brh_looks_english")
        if r["en"] and re.search(r"[-.=](?:[1-3](?:SG|PL)|[A-Z]{2,})\b", r["en"]):
            flags.append("en_has_gloss")
        if "brh_mismatch" in r["quality"]:
            flags.append("brh_mismatch")
        elif r["brh"] and content_score(r["brh"], r["brh_ascii"], fold_ascii(r["brh_ascii"])) < 0.5:
            flags.append("brh_mismatch")
        if flags:
            r["quality"] = "suspicious:" + ",".join(flags)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- report
    print(f"wrote {len(rows)} rows -> {args.out}")
    print("\nper-section alignment (title, pdf section, aligned/expected):")
    for t, sec, ok, tot in report:
        mark = "" if ok == tot else "   <-- " + str(tot - ok) + " unaligned"
        print(f"  {sec:>6}  {ok:4d}/{tot:<4d} {t[:60]}{mark}")
    if skipped:
        print("\nPDF sections with no tier in text_brahui.txt (skipped):")
        for h in skipped:
            print(f"  {h[1]} {h[2]}")
    q = Counter(r["quality"].split(":")[0] for r in rows)
    print("\nquality:", dict(q))
    fl = Counter(fl for r in rows if r["quality"].startswith("suspicious") for fl in r["quality"].split(":")[1].split(","))
    print("suspicious flags:", dict(fl))
    print("\n5 random rows:")
    rng = random.Random(args.seed)
    for r in rng.sample(rows, 5):
        print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
