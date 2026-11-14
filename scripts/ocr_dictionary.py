"""OCR a scanned Brahui dictionary with Claude vision into structured entries (no humans).

Usage: uv run python scripts/ocr_dictionary.py <pdf> <out.jsonl> [--first N --last M] [--dpi 130] [--concurrency 6] [--model opus]
       [--layout "Brahui headword (Perso-Arabic) with English gloss and Urdu synonyms"]
Each page -> PNG (pdftoppm) -> one vision request -> JSON entries {headword, roman, pos, english, urdu, note}.
Writes one row per entry with page number; pages that fail are listed at the end and can be re-run with --first/--last.
"""
from __future__ import annotations

import argparse, base64, json, os, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"opus": "claude-opus-5-5", "fable": "claude-fable-5-1"}
SCHEMA = {"type": "object", "properties": {"entries": {"type": "array", "items": {"type": "object", "properties": {
    "headword": {"type": "string"}, "roman": {"type": "string"}, "pos": {"type": "string"},
    "english": {"type": "string"}, "urdu": {"type": "string"}, "note": {"type": "string"}},
    "required": ["headword", "roman", "pos", "english", "urdu", "note"], "additionalProperties": False}},
    "page_header": {"type": "string"}}, "required": ["entries", "page_header"], "additionalProperties": False}


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pdf"); ap.add_argument("out")
    ap.add_argument("--first", type=int, default=1); ap.add_argument("--last", type=int, default=0); ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--concurrency", type=int, default=6); ap.add_argument("--model", default="opus", choices=MODELS)
    ap.add_argument("--layout", default="two columns per page; each entry is a Brahui headword in Perso-Arabic (Nastaliq) on the right of its line with the English gloss on the left of the same line, and a line of Urdu synonyms underneath; read the left column top to bottom, then the right column top to bottom")
    a = ap.parse_args(); load_env(); cl = anthropic.Anthropic(); model = MODELS[a.model]
    info = subprocess.run(["pdfinfo", a.pdf], capture_output=True, text=True).stdout
    n_pages = int([l for l in info.splitlines() if l.startswith("Pages:")][0].split()[1])
    last = a.last or n_pages
    tmp = Path(tempfile.mkdtemp(prefix="ocr_"))
    system = (f"You are digitizing a printed Brahui dictionary page. Layout: {a.layout}. "
              "Transcribe EVERY entry on the page exactly as printed, in reading order: the Brahui headword in Perso-Arabic script "
              "(copy the spelling precisely, including diacritics and hamza where printed), a romanization you infer (lowercase, "
              "macrons for long vowels, ṭ ḍ ṛ for retroflex, š for sh, x for kh, ɣ for gh), part of speech if printed or obvious "
              "(n., v., adj., adv., phr.) else empty, the English gloss(es) separated by '; ', the Urdu synonyms line if present "
              "(copy the script), and a note for anything else printed with the entry. Skip page numbers and running headers but "
              "put the running header in page_header. Return JSON {\"entries\": [...], \"page_header\": \"...\"}.")

    def run(p):
        png = tmp / f"p{p:04d}"
        subprocess.run(["pdftoppm", "-r", str(a.dpi), "-f", str(p), "-l", str(p), "-png", "-singlefile", a.pdf, str(png)], check=True)
        data = base64.standard_b64encode((png.with_suffix(".png")).read_bytes()).decode()
        for t in range(3):
            try:
                with cl.messages.stream(model=model, max_tokens=16000, system=system,
                                        messages=[{"role": "user", "content": [
                                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                                            {"type": "text", "text": f"Page {p}. Transcribe all entries."}]}],
                                        output_config={"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "medium"}) as st:
                    r = st.get_final_message()
                if r.stop_reason == "refusal": return p, None, r.usage
                out = json.loads("".join(b.text for b in r.content if b.type == "text"))
                return p, out, r.usage
            except Exception as e:
                print(f"  page {p} failed ({t+1}/3): {str(e)[:120]}", flush=True); time.sleep(5 * (t + 1))
        return p, None, None

    pages = list(range(a.first, last + 1)); t0 = time.time(); failed = []; n = 0; cost = 0.0
    with open(a.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.concurrency) as ex:
        for p, out, usage in ex.map(run, pages):
            if usage: cost += (usage.input_tokens * 4 + usage.output_tokens * 20) / 1e6
            if not out: failed.append(p); continue
            for e in out["entries"]:
                f.write(json.dumps({"page": p, "header": out["page_header"], **e}, ensure_ascii=False) + "\n"); n += 1
            f.flush()
            if p % 10 == 0: print(f"  page {p}: {n} entries so far, ${cost:.2f}, {time.time()-t0:.0f}s", flush=True)
    print(f"done: {n} entries from {len(pages)-len(failed)} pages, failed pages {failed}, cost ≈ ${cost:.2f}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
