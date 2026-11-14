"""Speaker-free end-to-end benchmark: Brahui ASR -> Claude MT -> English, scored against the JESUS film's
human English subtitles (data/raw/jesusfilm_brh/subtitles/en.vtt).

Windows: ASR chunks (data/media/transcripts*/jesusfilm_brh/jesus.jsonl) are grouped in time order into windows of
>= --window seconds whose boundaries are chunk boundaries; every English cue is assigned to the window containing
its midpoint. All ASR systems share the same VAD chunking so windows are identical across systems.

Translator (--model): opus|sonnet|claude-fable-5-1 via `claude -p`, or nllb:<model dir> for the offline student.
Conditions (--system):
  asr:<path to jesus.jsonl>  the ASR hypothesis text per window (noisy Brahui) -> Claude -> English
  ur                         the human Urdu subtitles per window -> Claude -> English (upper bound: perfect
                             transcript of the dub's meaning, same translator)
Controls reported per run: chrF++ of each window's hypothesis against the NEXT window's reference (off-by-one);
if the transcript carries information, aligned >> off-by-one.

Usage:
  uv run python evals/st/jesus_bench.py --name s3_fable --system asr:data/media/transcripts/jesusfilm_brh/jesus.jsonl \
      [--model claude-fable-5-1] [--window 40] [--batch 5] [--concurrency 4] [--blind]
Writes evals/st/runs/<name>/{windows.jsonl,summary.json}. Resumable.
"""
from __future__ import annotations

import argparse, json, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evals" / "mt"))
from run_eval import call_claude  # noqa: E402

SUBS = ROOT / "data" / "raw" / "jesusfilm_brh" / "subtitles"
OUT = ROOT / "evals" / "st" / "runs"
chrf = CHRF(word_order=2); bleu = BLEU(effective_order=True)

SYSTEM_PROMPT = """You are a careful translator. The user gives numbered passages of text in Perso-Arabic script and asks for
English. Translate what the text actually says, passage by passage, as faithfully and completely as you can.
The text may be imperfect automatic speech recognition output of spoken {lang}: expect misspellings, missing
word spaces, wrong word boundaries and a few garbled words. Use the surrounding passage to resolve them, but
do NOT invent content that is not supported by the text, and do NOT replace a passage with text you may recall
from elsewhere; if a passage is unintelligible, translate the parts you can and leave the rest out.
Return JSON {{"translations": [...]}} with exactly one English string per passage, in order. No commentary."""

CONTEXT_LINE = {"blind": "", "film": "\nContext: these passages are consecutive excerpts of the {lang} dubbing of a feature film about the life of Jesus (dialogue and narration)."}


def parse_vtt(p: Path):
    cues, block = [], []
    def flush():
        if len(block) >= 2 and "-->" in block[0]:
            a, b = block[0].split("-->")
            text = re.sub(r"<[^>]+>", "", " ".join(l.strip() for l in block[1:])).replace("\u200f", "")
            cues.append({"start": ts(a), "end": ts(b.split()[0]), "text": re.sub(r"\s+", " ", text).strip()})
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            flush(); block = []
        else:
            block.append(line)
    flush()
    return cues


def ts(s):
    s = s.strip(); parts = s.split(":")
    h, m, sec = (["0"] + parts)[-3:]
    return int(h) * 3600 + int(m) * 60 + float(sec)


def build_windows(chunks, en, ur, window_s):
    chunks = sorted(chunks, key=lambda c: c["start"])
    wins, cur = [], []
    for c in chunks:
        cur.append(c)
        if cur[-1]["end"] - cur[0]["start"] >= window_s:
            wins.append(cur); cur = []
    if cur:
        wins.append(cur)
    out = []
    for i, w in enumerate(wins):
        start = w[0]["start"] if i else 0.0
        end = wins[i + 1][0]["start"] if i + 1 < len(wins) else 1e9
        mid = lambda c: (c["start"] + c["end"]) / 2
        out.append({"win": i, "start": round(w[0]["start"], 2), "end": round(w[-1]["end"], 2),
                    "asr": " ".join(c["text"] for c in w if c["text"]).strip(),
                    "conf": round(sum(c["conf"] for c in w) / len(w), 3),
                    "en": " ".join(c["text"] for c in en if start <= mid(c) < end).strip(),
                    "ur": " ".join(c["text"] for c in ur if start <= mid(c) < end).strip()})
    return [w for w in out if w["en"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--system", required=True)
    ap.add_argument("--model", default="opus"); ap.add_argument("--effort", default=None)
    ap.add_argument("--window", type=float, default=40.0); ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=4); ap.add_argument("--blind", action="store_true")
    ap.add_argument("--chunks", default=str(ROOT / "data/media/transcripts/jesusfilm_brh/jesus.jsonl"),
                    help="jsonl that defines the window boundaries (any system; all share the VAD)")
    a = ap.parse_args()

    en, ur = parse_vtt(SUBS / "en.vtt"), parse_vtt(SUBS / "ur.vtt")
    chunks = [json.loads(l) for l in open(a.chunks, encoding="utf-8")]
    wins = build_windows(chunks, en, ur, a.window)
    if a.system.startswith("asr:"):
        src_chunks = [json.loads(l) for l in open(a.system[4:], encoding="utf-8")]
        src_wins = build_windows(src_chunks, en, ur, a.window)
        assert [(w["start"], w["end"]) for w in src_wins] == [(w["start"], w["end"]) for w in wins], "VAD windows differ"
        for w, s in zip(wins, src_wins):
            w["src"], w["conf"] = s["asr"], s["conf"]
        lang = "Brahui"
    elif a.system == "ur":
        for w in wins:
            w["src"] = w["ur"]
        lang = "Urdu"
    elif a.system.startswith("en2brh:"):
        # reverse direction: English subtitles -> Claude -> Brahui, scored against the ASR text of the dub
        from brahui.text import normalize
        src_chunks = [json.loads(l) for l in open(a.system[7:], encoding="utf-8")]
        for w, s in zip(wins, build_windows(src_chunks, en, ur, a.window)):
            w["src"], w["en"], w["ref_asr_conf"] = w["en"], normalize(s["asr"]), s["conf"]
        lang = "English"
    else:
        sys.exit("--system must be asr:<jsonl>, ur, or en2brh:<jsonl>")
    if lang == "English":
        global SYSTEM_PROMPT
        SYSTEM_PROMPT = """You are a careful translator into Brahui (Perso-Arabic script, as written in Balochistan, Pakistan).
The user gives numbered English passages. Translate each into natural spoken Brahui, faithfully and completely.
Return JSON {{"translations": [...]}} with exactly one Brahui string per passage, in order. No commentary."""
    out = OUT / a.name; out.mkdir(parents=True, exist_ok=True)
    res = out / "windows.jsonl"
    done = {json.loads(l)["win"]: json.loads(l) for l in open(res, encoding="utf-8")} if res.exists() else {}
    todo = [w for w in wins if w["win"] not in done]
    print(f"{a.name}: {len(wins)} windows, {len(todo)} pending, model={a.model} lang={lang} blind={a.blind}", flush=True)
    sys_prompt = SYSTEM_PROMPT.format(lang=lang) + CONTEXT_LINE["blind" if a.blind else "film"].format(lang=lang)
    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]

    nllb = None
    if a.model.startswith("nllb:"):  # offline cascade: the fine-tuned NLLB student translates instead of Claude
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        sys.path.insert(0, str(ROOT / "scripts")); from train_nllb import translate, BRH, ENG  # noqa: E402
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(a.model[5:]); mdl = AutoModelForSeq2SeqLM.from_pretrained(a.model[5:], dtype=torch.float32).to(dev).eval()
        nllb = (mdl, tok, (BRH, ENG) if lang != "English" else (ENG, BRH))
        a.concurrency = 1

    def run(batch):
        if nllb:
            mdl, tok, (sl, tl) = nllb
            trans = translate(mdl, tok, [w["src"] or "" for w in batch], sl, tl, bs=4, beams=4, max_new=300)
            return [{**w, "hyp": trans[i], "error": None, "chrfpp": round(chrf.sentence_score(trans[i], [w["en"]]).score, 2)} for i, w in enumerate(batch)]
        msg = "\n\n".join(f"{i}. {w['src'] or '(empty)'}" for i, w in enumerate(batch, 1))
        err = None
        for attempt in range(3):
            try:
                trans, usage, _, lat = call_claude(sys_prompt, msg, a.model, a.effort)
                if len(trans) != len(batch):
                    raise RuntimeError(f"{len(trans)} translations for {len(batch)}")
                break
            except Exception as e:
                import traceback
                err = repr(e)[:300]; trans = None
                (out / "errors.log").open("a").write(traceback.format_exc() + "\n"); time.sleep(5 * (attempt + 1))
        rows = []
        for i, w in enumerate(batch):
            hyp = (trans[i] if trans else "") or ""
            if lang == "English":
                from brahui.text import normalize
                hyp = normalize(hyp)
            rows.append({**w, "hyp": hyp, "error": None if trans else err,
                         "chrfpp": round(chrf.sentence_score(hyp, [w["en"]]).score, 2)})
        return rows

    with open(res, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.concurrency) as ex:
        n = 0
        for fut in as_completed([ex.submit(run, b) for b in batches]):
            for r in fut.result():
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); done[r["win"]] = r
            f.flush(); n += 1
            if n % 5 == 0 or n == len(batches):
                print(f"  batches {n}/{len(batches)}", flush=True)
    rows = [done[w["win"]] for w in wins if w["win"] in done]
    hyps, refs = [r["hyp"] for r in rows], [r["en"] for r in rows]
    shifted = refs[1:] + refs[:1]
    summ = {"name": a.name, "system": a.system, "model": a.model, "blind": a.blind, "window_s": a.window,
            "windows": len(rows), "errors": sum(1 for r in rows if r["error"]),
            "chrfpp": round(chrf.corpus_score(hyps, [refs]).score, 2), "bleu": round(bleu.corpus_score(hyps, [refs]).score, 2),
            "mean_win_chrfpp": round(sum(r["chrfpp"] for r in rows) / len(rows), 2),
            "control_offbyone_chrfpp": round(chrf.corpus_score(hyps, [shifted]).score, 2),
            "empty_hyps": sum(1 for h in hyps if not h.strip())}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
