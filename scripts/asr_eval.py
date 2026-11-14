"""Run an ASR model over a sealed eval set and score CER / WER.

Usage:
  uv run python scripts/asr_eval.py --model whisper:TBOGamer22/Brahvi-Whisper-Small --set cv_test [--limit N]
  uv run python scripts/asr_eval.py --model mms:facebook/mms-1b-all --set omni_test
  uv run python scripts/asr_eval.py --model ctc:data/raw/models/omniASR-CTC-1B-v2-hf --set cv_test   (HF-format omniASR, or runs/<name>/final)
  uv run python scripts/asr_eval.py --model omni:facebook/omniASR-CTC-300M --set cv_test   (fairseq2 original; needs .venv-omni)

Outputs evals/asr/runs/<tag>/results.jsonl (one row per clip, appended as it goes, resumable)
and summary.json with corpus CER/WER + bootstrap 95% CI, computed from the per-row results.
"""
from __future__ import annotations

import argparse, json, math, random, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from brahui.text import for_scoring, arabic_script_ratio  # noqa: E402

RUNS = ROOT / "evals" / "asr" / "runs"


def load_audio(path: Path) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != 16000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    return data


def device_for(prefer: str | None = None):
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():   # 100 dev clips took > 10 min on a pod's CPU while the GPU idled (2026-11-11)
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------------- backends
class WhisperBackend:
    def __init__(self, model_id, device, language="ur", chunk_s=30):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.proc = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=torch.float32).to(device).eval()
        self.device, self.language, self.chunk = device, language, chunk_s * 16000
        self.name = model_id

    @torch.no_grad()
    def transcribe(self, audio: np.ndarray) -> str:
        outs = []
        for start in range(0, len(audio), self.chunk):
            piece = audio[start:start + self.chunk]
            feats = self.proc(piece, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
            ids = self.model.generate(feats, language=self.language, task="transcribe", max_new_tokens=440, num_beams=1)
            outs.append(self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
        return " ".join(o for o in outs if o)


class MMSBackend:
    def __init__(self, model_id, device, lang="brh"):
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        self.proc = AutoProcessor.from_pretrained(model_id, target_lang=lang)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id, target_lang=lang, ignore_mismatched_sizes=True).to(device).eval()
        self.device, self.name = device, f"{model_id}[{lang}]"

    @torch.no_grad()
    def transcribe(self, audio: np.ndarray) -> str:
        outs = []
        for start in range(0, len(audio), 20 * 16000):  # omniASR CTC degrades sharply past ~20 s of input (CER 19 -> 68 at 40 s)
            piece = audio[start:start + 20 * 16000]
            inp = self.proc(piece, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
            logits = self.model(inp).logits
            outs.append(self.proc.batch_decode(torch.argmax(logits, dim=-1))[0].strip())
        return " ".join(o for o in outs if o)


class CTCBackend(MMSBackend):
    """Any Wav2Vec2ForCTC checkpoint with its own processor: the HF-converted omniASR CTC models
    (aadel4/omniASR-CTC-{300M,1B}-v2) or a directory saved by scripts/train_asr.py."""
    def __init__(self, model_id, device):
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device).eval()
        self.device, self.name = device, f"ctc:{Path(model_id).name}"


class OmniBackend:
    """facebook/omniASR-* via the omnilingual_asr package (fairseq2). Optional - may not install on macOS."""
    def __init__(self, model_id, device, lang="brh_Arab"):
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline  # type: ignore
        card = model_id.split("/")[-1].replace("-", "_")
        self.pipe = ASRInferencePipeline(model_card=card)  # runs on CPU; fairseq2 has no MPS backend
        self.lang, self.name, self.wants_path = lang, model_id, True

    def transcribe(self, path) -> str:
        """fairseq2's audio loader fails on long inputs (~>40 s); chunk anything over 30 s into 30 s temp wavs."""
        import tempfile
        info = sf.info(path)
        if info.frames / info.samplerate <= 35:
            return self.pipe.transcribe([str(path)], lang=[self.lang], batch_size=1)[0]
        audio = load_audio(Path(path)); step = 30 * 16000; outs = []
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i, start in enumerate(range(0, len(audio), step)):
                p = Path(td) / f"chunk{i}.wav"; sf.write(p, audio[start:start + step], 16000, subtype="PCM_16"); paths.append(str(p))
            for p in paths:
                outs.append(self.pipe.transcribe([p], lang=[self.lang], batch_size=1)[0].strip())
        return " ".join(o for o in outs if o)


class ONNXCTCBackend(MMSBackend):
    """A directory produced by tools/onnx_export/export_quant.py: processor files at the root, onnx/<file>.
    Spec: onnx:<dir>[@<file>.onnx]  (default file model_quantized.onnx). Runs on CPU via onnxruntime."""
    def __init__(self, model_id, device):
        import onnxruntime as ort
        from transformers import AutoProcessor
        d, _, fname = model_id.partition("@")
        self.proc = AutoProcessor.from_pretrained(d)
        self.sess = ort.InferenceSession(str(Path(d) / "onnx" / (fname or "model_quantized.onnx")), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.device, self.name = "cpu", f"onnx:{Path(d).name}@{fname or 'model_quantized.onnx'}"

    def transcribe(self, audio: np.ndarray) -> str:
        outs = []
        for start in range(0, len(audio), 20 * 16000):
            piece = audio[start:start + 20 * 16000]
            inp = self.proc(piece, sampling_rate=16000, return_tensors="np").input_values.astype(np.float32)
            logits = self.sess.run(None, {self.inp: inp})[0]
            outs.append(self.proc.batch_decode(torch.from_numpy(np.argmax(logits, axis=-1)))[0].strip())
        return " ".join(o for o in outs if o)


def make_backend(spec: str, device):
    kind, _, mid = spec.partition(":")
    if kind == "onnx":
        return ONNXCTCBackend(mid, device)
    if kind == "whisper":
        return WhisperBackend(mid, device)
    if kind == "mms":
        return MMSBackend(mid, device)
    if kind in ("ctc", "mms-local"):
        return CTCBackend(mid, device)
    if kind == "omni":
        return OmniBackend(mid, device)
    raise SystemExit(f"unknown backend {kind}")


# ----------------------------------------------------------------------------- scoring
def score_rows(rows):
    import jiwer
    refs = [r["ref_norm"] for r in rows]; hyps = [r["hyp_norm"] for r in rows]
    refs_c = [r if r else " " for r in refs]; hyps_c = [h if h else " " for h in hyps]
    cer = jiwer.cer(refs_c, hyps_c); wer = jiwer.wer(refs_c, hyps_c)
    # bootstrap CI over clips
    rng = random.Random(0); n = len(rows); cers = []; wers = []
    for _ in range(300):
        idx = [rng.randrange(n) for _ in range(n)]
        cers.append(jiwer.cer([refs_c[i] for i in idx], [hyps_c[i] for i in idx]))
        wers.append(jiwer.wer([refs_c[i] for i in idx], [hyps_c[i] for i in idx]))
    ci = lambda xs: (float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5)))
    return {"n": n, "cer": cer, "wer": wer, "cer_ci95": ci(cers), "wer_ci95": ci(wers),
            "ref_chars": sum(len(r) for r in refs), "hyp_arabic_ratio": float(np.mean([r["hyp_arabic_ratio"] for r in rows])),
            "empty_hyps": sum(1 for h in hyps if not h), "audio_hours": sum(r["duration"] or 0 for r in rows) / 3600}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--set", required=True)
    ap.add_argument("--limit", type=int); ap.add_argument("--sample", type=int, help="seeded random subset of N clips")
    ap.add_argument("--device"); ap.add_argument("--tag")
    ap.add_argument("--trim", type=float, default=None, metavar="TOP_DB", help="trim leading/trailing silence below TOP_DB (librosa.effects.trim, 25 ms frames) and keep 0.1 s of pad; mirrors the browser demo")
    a = ap.parse_args()
    rows_in = [json.loads(l) for l in open(ROOT / "data" / "eval" / f"{a.set}.jsonl", encoding="utf-8")]
    if a.limit:
        rows_in = rows_in[: a.limit]
    if a.sample and a.sample < len(rows_in):
        rows_in = random.Random(20261006).sample(rows_in, a.sample); rows_in.sort(key=lambda r: r["id"])
    tag = a.tag or f"{a.model.split(':')[0]}_{a.model.split('/')[-1]}__{a.set}" + (f"_s{a.sample}" if a.sample else "")
    out = RUNS / tag; out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.jsonl"
    done = {}
    if res_path.exists():
        for l in open(res_path, encoding="utf-8"):
            r = json.loads(l); done[r["id"]] = r
    device = device_for(a.device)
    backend = make_backend(a.model, device)
    print(f"model={backend.name} device={device} set={a.set} n={len(rows_in)} already={len(done)}", flush=True)
    with open(res_path, "a", encoding="utf-8") as f:
        for r in tqdm(rows_in):
            if r["id"] in done:
                continue
            src = ROOT / r["audio"] if getattr(backend, "wants_path", False) else load_audio(ROOT / r["audio"])
            if "start" in r and not getattr(backend, "wants_path", False):  # media_test rows are windows of long files
                src = src[int(r["start"] * 16000):int(r["end"] * 16000)]
            if a.trim is not None and not getattr(backend, "wants_path", False):
                import librosa
                _, (i0, i1) = librosa.effects.trim(src, top_db=a.trim, frame_length=400, hop_length=160)
                src = src[max(0, i0 - 1600):i1 + 1600]
            t0 = time.time()
            try:
                hyp = backend.transcribe(src)
            except Exception as e:  # keep going; record the failure
                hyp = ""; err = repr(e)[:300]
            else:
                err = None
            row = {"id": r["id"], "set": a.set, "model": backend.name, "duration": r["duration"],
                   "ref": r["text"], "hyp": hyp, "ref_norm": for_scoring(r["text"]), "hyp_norm": for_scoring(hyp),
                   "hyp_arabic_ratio": arabic_script_ratio(hyp), "latency_s": round(time.time() - t0, 3), "error": err}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            done[r["id"]] = row
    rows = [done[r["id"]] for r in rows_in if r["id"] in done]
    summ = score_rows(rows); summ.update({"model": backend.name, "set": a.set, "device": str(device),
                                          "rtf": sum(r["latency_s"] for r in rows) / max(1e-9, sum(r["duration"] or 0 for r in rows))})
    (out / "summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
