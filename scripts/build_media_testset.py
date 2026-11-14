"""Build a sealed media-domain ASR test set without a transcriber (speaker-free labels, gated).

Whole media files are held out (data/eval/media_test_files.txt; never train on them). From each, chunks of the
stage-3 transcription (data/media/transcripts/) are sampled. Labels are produced in three steps and every
intermediate is kept so the label quality can be audited later:
  1. independent hypotheses: stage 3 (the pseudo-label), stage 2, Whisper-small Brahui (`asr` step)
  2. an LLM (Opus 5.5 via `claude -p`) proposes the most likely true transcript from the hypotheses + context
     (`correct` step). It only sees text, so it can fix what the hypotheses or context make clear, not hear.
  3. a forced-alignment gate (`gate` step): the proposal is accepted only if the CTC models say the audio is
     compatible with it (log-likelihood under stage 2 and stage 3, relative to the stage-3 greedy text, per
     changed character). Otherwise the stage-3 text stays. The gate blocks paraphrase and hallucinated words.
Controls built into the set: the off-by-one score (hypothesis vs the neighbouring chunk's label) must be far
below the aligned score, and system ranking must agree with the sealed CV/Omni sets. The labels are correlated
with the stage-2/3 models (they produced the hypotheses), so a stage-4 model trained on media pseudo-labels is
also judged on the JESUS e2e benchmark (human English), which is independent.

Usage (from the repo root):
  uv run python scripts/build_media_testset.py select
  uv run python scripts/build_media_testset.py asr --model ctc:runs/ctc1b_brh_s2/final --tag s2
  uv run python scripts/build_media_testset.py asr --model whisper:data/raw/models/brahvi_whisper_small --tag whisper
  uv run python scripts/build_media_testset.py correct [--model opus] [--batch 8] [--concurrency 4]
  uv run python scripts/build_media_testset.py gate
  uv run python scripts/build_media_testset.py finalize
Writes data/media/testset/*.jsonl and finally data/eval/media_test.jsonl.
"""
from __future__ import annotations

import argparse, json, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "evals" / "mt"))
from brahui.text import normalize, for_scoring  # noqa: E402

MEDIA, TS, EVAL = ROOT / "data" / "media", ROOT / "data" / "media" / "testset", ROOT / "data" / "eval"
SR = 16000
# Held-out files (whole recordings) and how many chunks to sample from each source.
HELD_OUT = {
    "balochistans_today_news": (["xImib-rZds4", "WLsIh8DgvXo", "Nt37L1pQnCo"], 60),
    "ptv_bolan_current_affairs": (["rXDgo7nuDUM"], 60),
    "ptv_bolan_programs": (["0ZXqYVTSbh4"], 50),
    "brahui_tv_interviews": (["vENDTOMZBqU"], 30),
    "nazir_shakir_interviews": (["58h2UGYyE4g"], 25),
    "rabita_forum_talkshow": (["IqZIBy79tbQ"], 25),
    "jesusfilm_brh": (["jesusfilm_brh"], 40),
}


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if Path(p).exists() else []


def wj(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cid(r):
    return f"{r['id']}_{r['seg']:04d}"


# ----------------------------------------------------------------------------- select
def select(a):
    meta = {r["id"]: r for r in rj(MEDIA / "meta.jsonl")}
    jf = MEDIA / "transcripts" / "jesusfilm_brh"
    rng = random.Random(20261019); out = []; files = []
    for source, (ids, n) in HELD_OUT.items():
        pool = []
        for fid in ids:
            p = MEDIA / "transcripts" / source / f"{fid}.jsonl"
            if not p.exists():  # the film is stored under its own id
                cands = list((MEDIA / "transcripts" / source).glob("*.jsonl")); p = cands[0]; fid = p.stem
            rows = rj(p); files.append(f"{source}/{fid}")
            audio = meta[fid]["path"] if fid in meta else str(p.relative_to(ROOT)).replace("transcripts", "audio").replace(".jsonl", ".wav")
            for i, r in enumerate(rows):
                d = r["end"] - r["start"]
                if 3.0 <= d <= 20.0 and len(r["text"].split()) >= 4:
                    pool.append({"id": cid(r), "source": source, "file": fid, "seg": r["seg"], "audio": audio,
                                 "start": round(r["start"], 2), "end": round(r["end"], 2), "duration": round(d, 2),
                                 "hyp_s3": r["text"], "conf_s3": round(r["conf"], 3),
                                 "ctx_prev": rows[i - 1]["text"] if i else "", "ctx_next": rows[i + 1]["text"] if i + 1 < len(rows) else ""})
        pick = rng.sample(pool, min(n, len(pool))); pick.sort(key=lambda r: (r["file"], r["seg"]))
        out += pick
        print(f"{source}: {len(pool)} candidate chunks in {len(ids)} held-out file(s), sampled {len(pick)}")
    wj(TS / "candidates.jsonl", out)
    (EVAL / "media_test_files.txt").write_text("\n".join(files) + "\n")
    print(f"{len(out)} chunks, {sum(r['duration'] for r in out) / 60:.1f} min -> {TS / 'candidates.jsonl'}; held-out files -> data/eval/media_test_files.txt")


# ----------------------------------------------------------------------------- asr
_audio_cache = {}


def window(r):
    from asr_eval import load_audio
    if r["audio"] not in _audio_cache:
        _audio_cache.clear(); _audio_cache[r["audio"]] = load_audio(ROOT / r["audio"])
    a = _audio_cache[r["audio"]]
    return a[int(r["start"] * SR):int(r["end"] * SR)]


def asr(a):
    from asr_eval import make_backend, device_for
    from tqdm import tqdm
    rows = rj(TS / "candidates.jsonl")
    out = TS / f"hyp_{a.tag}.jsonl"; done = {r["id"]: r for r in rj(out)}
    be = make_backend(a.model, device_for(a.device))
    with open(out, "a", encoding="utf-8") as f:
        for r in tqdm(rows):
            if r["id"] in done:
                continue
            hyp = be.transcribe(window(r))
            f.write(json.dumps({"id": r["id"], "hyp": hyp, "model": be.name}, ensure_ascii=False) + "\n"); f.flush()
    print(f"{a.tag}: {len(rows)} hypotheses -> {out}")


# ----------------------------------------------------------------------------- correct
SYSTEM = """You are a careful transcription editor for Brahui (Perso-Arabic script, Pakistani Balochistan). You will see automatic speech recognition (ASR) output for short chunks of Brahui broadcast audio: the main hypothesis A from the best system, and alternatives B and C from weaker independent systems, plus the ASR text of the chunks just before and after for context. You cannot hear the audio.

Produce the most likely true transcript of what was said, for each chunk:
- Keep A's wording and word order; change only what is clearly a recognition error (a word that is not Brahui/Urdu/Balochi/English as used in Brahui speech, a split or joined word, a wrong letter) and where the alternatives or the context make the intended word clear. Prefer A's spelling conventions (Common Voice / Omnilingual Brahui orthography).
- Never paraphrase, reorder, translate, summarise, add or drop words, or normalise register. Names, numbers, loanwords and disfluencies stay as spoken.
- If you are not confident, return A unchanged. Expect most chunks to need zero or one small edits.
Return JSON {"transcripts": [...]} with exactly one string per chunk, in order."""
SCHEMA = json.dumps({"type": "object", "properties": {"transcripts": {"type": "array", "items": {"type": "string"}}},
                     "required": ["transcripts"], "additionalProperties": False})


def correct(a):
    import subprocess
    rows = rj(TS / "candidates.jsonl")
    hyps = {t: {r["id"]: r["hyp"] for r in rj(TS / f"hyp_{t}.jsonl")} for t in ["s2", "whisper"]}
    out = TS / "llm_corrected.jsonl"; done = {r["id"] for r in rj(out)}
    todo = [r for r in rows if r["id"] not in done]
    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    print(f"{len(todo)} chunks to correct in {len(batches)} batches with {a.model}")

    def msg(batch):
        lines = [f"## {len(batch)} chunks. Return JSON {{\"transcripts\": [...]}} with exactly {len(batch)} strings, in order.", ""]
        for i, r in enumerate(batch, 1):
            lines += [f"### Chunk {i} (source: {r['source']})", f"Context before: {r['ctx_prev']}", f"A: {r['hyp_s3']}",
                      f"B: {hyps['s2'].get(r['id'], '')}", f"C: {hyps['whisper'].get(r['id'], '')}", f"Context after: {r['ctx_next']}", ""]
        return "\n".join(lines)

    def run(batch):
        cmd = ["claude", "-p", "--model", a.model, "--system-prompt", SYSTEM, "--tools", "", "--restricted",
               "--no-session-persistence", "--output-format", "json", "--json-schema", SCHEMA]
        for attempt in range(3):
            try:
                proc = subprocess.run(cmd, input=msg(batch), capture_output=True, text=True, timeout=900)
                if proc.returncode != 0:
                    raise RuntimeError(f"claude exit {proc.returncode}: {proc.stderr[-300:]} {proc.stdout[-300:]}")
                d = json.loads(proc.stdout); outp = d.get("structured_output") or json.loads(d.get("result", ""))
                tr = outp["transcripts"]
                if len(tr) != len(batch):
                    raise RuntimeError(f"count {len(tr)} != {len(batch)}")
                return [{"id": r["id"], "llm": t, "model": next(iter(d.get("modelUsage", {}).keys()), a.model)} for r, t in zip(batch, tr)]
            except Exception as e:
                print("batch failed:", e, flush=True); time.sleep(5)
        return []

    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.concurrency) as ex:
        for res in ex.map(run, batches):
            for r in res:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
    print(f"-> {out} ({len(rj(out))} rows)")


# ----------------------------------------------------------------------------- gate
def ctc_logp(model, proc, audio, text, device):
    """Forced-alignment log-likelihood of `text` given `audio` (sum over frames), or None if unalignable."""
    import torch, torchaudio.functional as F
    tok = proc.tokenizer
    ids = tok(text).input_ids
    if not ids:
        return None
    with torch.no_grad():
        inp = proc.feature_extractor(audio, sampling_rate=SR, return_tensors="pt").input_values.to(device)
        logp = torch.log_softmax(model(inp).logits.float(), dim=-1).cpu()
    if len(ids) >= logp.shape[1]:
        return None
    try:
        ali, scores = F.forced_align(logp, torch.tensor([ids], dtype=torch.int32), blank=tok.pad_token_id)
    except Exception:
        return None
    return float(scores.sum())


def gate(a):
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    from asr_eval import device_for
    from tqdm import tqdm
    import jiwer
    dev = device_for(a.device)
    rows = rj(TS / "candidates.jsonl"); llm = {r["id"]: r["llm"] for r in rj(TS / "llm_corrected.jsonl")}
    models = {}
    for tag, path in [("s2", "runs/ctc1b_brh_s2/final"), ("s3", "runs/ctc1b_brh_s3/final")]:
        proc = AutoProcessor.from_pretrained(ROOT / path); m = Wav2Vec2ForCTC.from_pretrained(ROOT / path).to(dev).eval()
        models[tag] = (m, proc)
    out = []
    for r in tqdm(rows):
        prop = llm.get(r["id"]); base = r["hyp_s3"]
        rec = {**r, "llm": prop}
        if prop is None or normalize(prop) == normalize(base):
            rec.update({"decision": "unchanged", "label": base}); out.append(rec); continue
        audio = window(r)
        n_changed = max(1, round(jiwer.cer(for_scoring(base) or " ", for_scoring(prop) or " ") * len(for_scoring(base))))
        d = {}
        for tag, (m, proc) in models.items():
            lb, lp = ctc_logp(m, proc, audio, base, dev), ctc_logp(m, proc, audio, prop, dev)
            d[tag] = None if lb is None or lp is None else (lp - lb) / n_changed
        rec.update({"n_changed_chars": n_changed, "delta_s2_per_char": d["s2"], "delta_s3_per_char": d["s3"]})
        # Accept if the audio is at least as compatible with the proposal under the independent-ish stage-2 model, or
        # if neither model objects by more than `a.tau` nats per changed character (small, audio-compatible edits).
        ok = d["s2"] is not None and d["s3"] is not None and (d["s2"] >= 0 or (d["s2"] >= -a.tau and d["s3"] >= -a.tau))
        rec.update({"decision": "accepted" if ok else "rejected", "label": prop if ok else base})
        out.append(rec)
    wj(TS / "gated.jsonl", out)
    from collections import Counter
    c = Counter(r["decision"] for r in out)
    print(dict(c), f"tau={a.tau}")
    acc = [r for r in out if r["decision"] == "accepted"]; rej = [r for r in out if r["decision"] == "rejected"]
    for name, grp in [("accepted", acc), ("rejected", rej)]:
        if grp:
            print(f"{name}: median changed chars {np.median([r['n_changed_chars'] for r in grp]):.0f}, "
                  f"median delta s2 {np.median([r['delta_s2_per_char'] or 0 for r in grp]):+.2f}, s3 {np.median([r['delta_s3_per_char'] or 0 for r in grp]):+.2f}")


# ----------------------------------------------------------------------------- finalize
def finalize(a):
    rows = rj(TS / "gated.jsonl")
    out = [{"id": r["id"], "source": r["source"], "split": "test", "audio": r["audio"], "start": r["start"], "end": r["end"],
            "duration": r["duration"], "text": r["label"], "label": r["decision"], "text_s3_raw": r["hyp_s3"]} for r in rows]
    wj(EVAL / "media_test.jsonl", out)
    print(f"{len(out)} rows, {sum(r['duration'] for r in out) / 60:.1f} min -> {EVAL / 'media_test.jsonl'}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["select", "asr", "correct", "gate", "finalize"])
    ap.add_argument("--model", default="opus"); ap.add_argument("--tag"); ap.add_argument("--device")
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--tau", type=float, default=1.0)
    a = ap.parse_args()
    {"select": select, "asr": asr, "correct": correct, "gate": gate, "finalize": finalize}[a.cmd](a)


if __name__ == "__main__":
    main()
