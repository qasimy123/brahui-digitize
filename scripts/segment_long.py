"""Cut long transcribed segments (Omnilingual ASR corpus, ~88 s each) into <= N-second training chunks.

CTC forced alignment with an omniASR CTC model gives word-level timestamps; chunks are cut between words
so every chunk keeps an exact transcript. Writes data/manifests/<split>_chunks.jsonl with rows carrying
`start`/`end` seconds into the original wav (scripts/train_asr.py reads those). No audio is rewritten.

Usage:
  uv run python scripts/segment_long.py --split omni_train [--split omni_dev] \
      [--model data/raw/models/omniASR-CTC-1B-v2-hf] [--max-seconds 20] [--device cuda]
On the Mac the 300M model is fine for alignment (boundaries only need to be roughly right): ~2 s per 88 s segment on CPU.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from brahui.text import for_scoring  # noqa: E402

MAN = ROOT / "data" / "manifests"
FRAME_S = 0.02  # wav2vec2 stride: 320 samples at 16 kHz


def load_audio(path: Path) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != 16000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    return data


@torch.no_grad()
def word_spans(model, proc, audio: np.ndarray, text: str, device):
    """Return [(word, start_s, end_s)] via CTC forced alignment, or None if alignment is impossible."""
    tok = proc.tokenizer
    words = text.split()
    ids = tok(text).input_ids
    inp = proc.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    logp = torch.log_softmax(model(inp).logits.float(), dim=-1).cpu()  # (1, T, V)
    T = logp.shape[1]
    if len(ids) >= T:
        return None
    tgt = torch.tensor([ids], dtype=torch.int32)
    try:
        ali, _ = F.forced_align(logp, tgt, blank=tok.pad_token_id)
    except Exception:
        return None
    ali = ali[0].tolist()
    # Frame index of each target token's first emission (collapse repeats, skip blanks).
    tok_frames, prev, k = [], None, 0
    for t, a in enumerate(ali):
        if a == tok.pad_token_id:
            prev = None; continue
        if a == prev:
            continue
        tok_frames.append(t); prev = a; k += 1
    if len(tok_frames) != len(ids):
        return None
    # Group token frames into words using the space token positions in `ids`.
    spans, wi, cur = [], 0, []
    for i, tid in enumerate(ids):
        if tid == tok.word_delimiter_token_id or tok.convert_ids_to_tokens(tid) == " ":
            if cur:
                spans.append((words[wi], cur[0] * FRAME_S, (cur[-1] + 1) * FRAME_S)); wi += 1; cur = []
        else:
            cur.append(tok_frames[i])
    if cur:
        spans.append((words[wi], cur[0] * FRAME_S, (cur[-1] + 1) * FRAME_S)); wi += 1
    return spans if wi == len(words) else None


def cut(spans, total_s: float, max_s: float, min_s: float = 1.0):
    """Greedy: extend the current chunk word by word; cut midway in the gap before a word that would overflow."""
    chunks, cur_words, cur_start = [], [], 0.0
    for i, (w, s, e) in enumerate(spans):
        if cur_words and e - cur_start > max_s:
            prev_end = spans[i - 1][2]
            cut_at = (prev_end + s) / 2
            chunks.append((cur_start, cut_at, " ".join(cur_words)))
            cur_words, cur_start = [], cut_at
        cur_words.append(w)
    if cur_words:
        chunks.append((cur_start, total_s, " ".join(cur_words)))
    return [c for c in chunks if c[1] - c[0] >= min_s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="append", default=None, help="manifest name(s) under data/manifests (default omni_train, omni_dev)")
    ap.add_argument("--model", default="data/raw/models/omniASR-CTC-300M-v2-hf")
    ap.add_argument("--max-seconds", type=float, default=20.0); ap.add_argument("--device")
    ap.add_argument("--only-long", type=float, default=30.0, help="segments at or below this length are copied through unchanged")
    a = ap.parse_args()
    splits = a.split or ["omni_train", "omni_dev"]
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    from transformers import AutoProcessor, Wav2Vec2ForCTC
    mp = str(ROOT / a.model) if (ROOT / a.model).exists() else a.model
    proc = AutoProcessor.from_pretrained(mp)
    model = Wav2Vec2ForCTC.from_pretrained(mp).to(device).eval()

    for split in splits:
        rows = [json.loads(l) for l in open(MAN / f"{split}.jsonl", encoding="utf-8")]
        out_rows, failed, n_chunks = [], 0, 0
        for r in tqdm(rows, desc=split):
            text = for_scoring(r["text"])
            if not text or (r["duration"] or 0) <= a.only_long:
                if text:
                    out_rows.append(r)
                continue
            audio = load_audio(ROOT / r["audio"])
            spans = word_spans(model, proc, audio, text, device)
            if spans is None:
                failed += 1; continue
            for j, (s, e, t) in enumerate(cut(spans, len(audio) / 16000, a.max_seconds)):
                out_rows.append({"id": f"{r['id']}_c{j:02d}", "source": r["source"], "split": r["split"], "audio": r["audio"],
                                 "start": round(s, 3), "end": round(e, 3), "duration": round(e - s, 3), "text": t,
                                 "text_norm": t, "speaker": r.get("speaker"), "parent": r["id"]})
                n_chunks += 1
        out = MAN / f"{split}_chunks.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        hours = sum(r["duration"] for r in out_rows) / 3600
        print(f"{split}: {len(rows)} segments -> {len(out_rows)} rows ({n_chunks} chunks, {hours:.2f} h), "
              f"{failed} segments failed alignment -> {out}")


if __name__ == "__main__":
    main()
