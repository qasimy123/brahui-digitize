"""Transcribe long untranscribed recordings with a fine-tuned omniASR CTC model (pseudo-labels).

Usage:
  uv run python scripts/transcribe_media.py [--model runs/ctc1b_brh_s3/final] [--meta data/media/meta.jsonl]
                                            [--only SOURCE] [--limit N] [--batch 4] [--device mps|cuda|cpu]

For each file in meta.jsonl: energy-based silence detection -> speech chunks of <= 20 s cut at the quietest
point (the CTC model degrades sharply past ~20 s) -> batched greedy CTC decoding -> per-chunk rows with
start/end, text, a confidence score (mean max-softmax over emitting frames) and the blank fraction.
Writes data/media/transcripts/<source>/<id>.{jsonl,txt,srt}; a file is skipped when its .jsonl exists.
Finally prints a per-source summary (hours, speech ratio, chunks, mean confidence).
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from brahui.text import normalize  # noqa: E402

MEDIA = ROOT / "data" / "media"
SR = 16000


def load_audio(path: Path) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != SR:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    return data


# ----------------------------------------------------------------------------- chunking
def speech_regions(audio: np.ndarray, frame_s=0.025, hop_s=0.010, margin_db=10.0, floor_db=-50.0,
                   min_speech_s=0.25, max_gap_s=0.35, pad_s=0.15):
    """Frame-energy VAD: speech = frames louder than (noise floor + margin). Returns [(start_s, end_s)]."""
    frame, hop = int(frame_s * SR), int(hop_s * SR)
    n = max(1, 1 + (len(audio) - frame) // hop)
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt((audio[idx] ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    noise = np.percentile(db, 10)
    thr = max(noise + margin_db, floor_db)
    on = db > thr
    regions, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            regions.append([start * hop_s, i * hop_s]); start = None
    if start is not None:
        regions.append([start * hop_s, n * hop_s])
    merged = []
    for s, e in regions:
        if merged and s - merged[-1][1] <= max_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    total = len(audio) / SR
    out = [(max(0.0, s - pad_s), min(total, e + pad_s)) for s, e in merged if e - s >= min_speech_s]
    return out, db, hop_s


def split_long(s: float, e: float, db: np.ndarray, hop_s: float, max_s: float, min_s=4.0):
    """Recursively split (s, e) at the quietest frame in its middle part until every piece is <= max_s."""
    if e - s <= max_s:
        return [(s, e)]
    lo, hi = int((s + min_s) / hop_s), int(min(e - min_s, s + max_s) / hop_s)
    if hi <= lo:
        cut = s + max_s
    else:
        cut = (lo + int(np.argmin(db[lo:hi]))) * hop_s
    return split_long(s, cut, db, hop_s, max_s, min_s) + split_long(cut, e, db, hop_s, max_s, min_s)


def make_chunks(audio: np.ndarray, max_s=20.0, min_s=0.4):
    regions, db, hop_s = speech_regions(audio)
    chunks = []
    for s, e in regions:
        chunks += split_long(s, e, db, hop_s, max_s)
    # pack neighbouring short pieces together when they fit (fewer, longer chunks; still cut at silences)
    packed = []
    for s, e in chunks:
        if packed and e - packed[-1][0] <= max_s and s - packed[-1][1] <= 1.0:
            packed[-1][1] = e
        else:
            packed.append([s, e])
    return [(s, e) for s, e in packed if e - s >= min_s]


# ----------------------------------------------------------------------------- model
class CTC:
    def __init__(self, model_dir: str, device: torch.device):
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        self.proc = AutoProcessor.from_pretrained(model_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(device).eval()
        self.device = device
        self.blank = self.model.config.pad_token_id

    @torch.no_grad()
    def transcribe(self, pieces: list[np.ndarray]) -> list[dict]:
        lens = torch.tensor([len(p) for p in pieces])
        inp = self.proc(pieces, sampling_rate=SR, return_tensors="pt", padding=True).input_values.to(self.device)
        logits = self.model(inp).logits.float()                      # (B, T, V)
        out_lens = self.model._get_feat_extract_output_lengths(lens)
        probs = torch.softmax(logits, dim=-1)
        pmax, ids = probs.max(dim=-1)
        rows = []
        for b in range(len(pieces)):
            T = int(out_lens[b])
            i, p = ids[b, :T].cpu(), pmax[b, :T].cpu()
            emit = i != self.blank
            text = normalize(self.proc.batch_decode(i[None])[0].strip())
            rows.append({"text": text,
                         "conf": float(p[emit].mean()) if emit.any() else 0.0,
                         "blank_frac": float((~emit).float().mean())})
        return rows


def srt_time(t: float) -> str:
    h, r = divmod(t, 3600); m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s - int(s)) * 1000):03d}"


def transcribe_file(ctc: CTC, row: dict, out_dir: Path, batch: int, max_s: float) -> dict:
    audio = load_audio((ROOT / row["path"] if not row["path"].startswith("/") else Path(row["path"])))
    chunks = make_chunks(audio, max_s=max_s)
    order = sorted(range(len(chunks)), key=lambda k: chunks[k][1] - chunks[k][0])
    results = [None] * len(chunks)
    for k in tqdm(range(0, len(order), batch), desc=row["id"], leave=False):
        sel = order[k:k + batch]
        pieces = [audio[int(chunks[j][0] * SR):int(chunks[j][1] * SR)] for j in sel]
        for j, r in zip(sel, ctc.transcribe(pieces)):
            s, e = chunks[j]
            results[j] = {"id": row["id"], "source": row["source"], "seg": j, "start": round(s, 2), "end": round(e, 2), **r}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{row['id']}.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / f"{row['id']}.txt").write_text("\n".join(r["text"] for r in results if r["text"]) + "\n")
    with open(out_dir / f"{row['id']}.srt", "w") as f:
        for n, r in enumerate(results, 1):
            f.write(f"{n}\n{srt_time(r['start'])} --> {srt_time(r['end'])}\n{r['text']}\n\n")
    speech = sum(e - s for s, e in chunks)
    return {"chunks": len(chunks), "speech_s": speech, "total_s": len(audio) / SR,
            "conf": float(np.mean([r["conf"] for r in results])) if results else 0.0}


def summarize(stats: dict[str, list[dict]]) -> None:
    print(f"\n{'source':32s} {'files':>5s} {'audio h':>8s} {'speech h':>9s} {'chunks':>7s} {'conf':>5s}")
    for src, rows in sorted(stats.items()):
        tot = sum(r["total_s"] for r in rows) / 3600; sp = sum(r["speech_s"] for r in rows) / 3600
        print(f"{src:32s} {len(rows):5d} {tot:8.2f} {sp:9.2f} {sum(r['chunks'] for r in rows):7d} {np.mean([r['conf'] for r in rows]):5.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "runs" / "ctc1b_brh_s3" / "final"))
    ap.add_argument("--meta", default=str(MEDIA / "meta.jsonl"))
    ap.add_argument("--out", default=str(MEDIA / "transcripts"))
    ap.add_argument("--only"); ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=4); ap.add_argument("--max-seconds", type=float, default=20.0)
    ap.add_argument("--device"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    rows = [json.loads(l) for l in open(a.meta)]
    if a.only:
        rows = [r for r in rows if r["source"] == a.only]
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(rows)} files, {sum(r['duration'] or 0 for r in rows) / 3600:.2f} h, model {a.model}, device {device}", flush=True)
    ctc = CTC(a.model, device)
    stats: dict[str, list[dict]] = {}
    t0 = time.time(); done_s = 0.0
    for r in rows:
        out_dir = Path(a.out) / r["source"]
        if (out_dir / f"{r['id']}.jsonl").exists() and not a.force:
            segs = [json.loads(l) for l in open(out_dir / f"{r['id']}.jsonl")]
            stats.setdefault(r["source"], []).append({"chunks": len(segs), "speech_s": sum(s["end"] - s["start"] for s in segs),
                                                      "total_s": r["duration"] or 0, "conf": float(np.mean([s["conf"] for s in segs])) if segs else 0.0})
            continue
        t = time.time()
        st = transcribe_file(ctc, r, out_dir, a.batch, a.max_seconds)
        stats.setdefault(r["source"], []).append(st)
        done_s += st["total_s"]
        print(f"{r['source']}/{r['id']}: {st['total_s'] / 60:.1f} min audio, {st['speech_s'] / 60:.1f} min speech, {st['chunks']} chunks, "
              f"conf {st['conf']:.3f}, {time.time() - t:.0f}s ({st['total_s'] / (time.time() - t):.1f}x RT)", flush=True)
    summarize(stats)
    if done_s:
        print(f"transcribed {done_s / 3600:.2f} h in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
