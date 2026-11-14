"""Fine-tune omniASR CTC (HF-converted Wav2Vec2ForCTC) on all Brahui speech.

Full fine-tune of encoder + the pretrained multilingual CTC head (vocab kept, so the model's zero-shot
Brahui knowledge is the starting point). The feature encoder (conv front end) is frozen, as in Meta's
recipe. Audio is loaded lazily in the collator, so the 90 h pool never has to fit in RAM or on disk twice.

Train pool (data/manifests): cv_train + cv_dev (minus the monitor subset) + omni_train chunks
(scripts/segment_long.py output, or the few short originals) + brahuispeech_train.
Monitor set: a seeded subset of cv_dev, never used for training. Sealed tests stay untouched.

Usage (GPU):
  uv run python scripts/train_asr.py --out runs/ctc1b_brh --base data/raw/models/omniASR-CTC-1B-v2-hf
CPU smoke test:
  uv run python scripts/train_asr.py --out /tmp/asr_smoke --base data/raw/models/omniASR-CTC-300M-v2-hf --smoke
Then: uv run python scripts/asr_eval.py --model ctc:runs/ctc1b_brh/final --set cv_test
"""
from __future__ import annotations

import argparse, json, random, sys, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from brahui.text import for_scoring  # noqa: E402

MAN = ROOT / "data" / "manifests"
SEED = 20261006


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if Path(p).exists() else []


def load_audio(path: Path, start: float | None = None, end: float | None = None) -> np.ndarray:
    info = sf.info(path)
    s = int((start or 0) * info.samplerate); e = int(end * info.samplerate) if end else -1
    data, sr = sf.read(path, dtype="float32", always_2d=True, start=s, stop=None if e < 0 else e)
    data = data.mean(axis=1)
    if sr != 16000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    return data


def build_pools(a):
    """Return (train_rows, monitor_rows). Rows carry audio path, optional start/end (chunks), text."""
    cv_dev = read_jsonl(MAN / "cv_dev.jsonl")
    rng = random.Random(SEED)
    monitor = rng.sample(cv_dev, min(a.monitor_n, len(cv_dev)))
    mon_ids = {r["id"] for r in monitor}

    train = read_jsonl(MAN / "cv_train.jsonl") + [r for r in cv_dev if r["id"] not in mon_ids]
    chunks = read_jsonl(MAN / "omni_train_chunks.jsonl")
    if chunks:
        train += chunks
    else:
        print("note: data/manifests/omni_train_chunks.jsonl missing (run scripts/segment_long.py); "
              "only Omnilingual segments <= --max-seconds are used", flush=True)
        train += read_jsonl(MAN / "omni_train.jsonl")
    if not a.no_brahuispeech:
        train += read_jsonl(MAN / a.brahuispeech_manifest)
    for extra in [m for m in (a.extra_manifest or "").split(",") if m]:
        rows = read_jsonl(MAN / extra); train += rows
        print(f"extra manifest {extra}: {len(rows)} clips ({sum(r.get('duration', 0) for r in rows)/3600:.2f} h)", flush=True)

    def ok(r):
        d = r.get("duration") or 0
        return 0.3 <= d <= a.max_seconds and for_scoring(r["text"]).strip() != ""
    before = len(train)
    train = [r for r in train if ok(r)]
    print(f"train: {len(train)} clips ({sum(r['duration'] for r in train)/3600:.2f} h), dropped {before-len(train)} "
          f"(empty text or outside 0.3..{a.max_seconds} s); monitor: {len(monitor)} cv_dev clips", flush=True)
    if a.max_train:
        rng.shuffle(train); train = train[: a.max_train]
    return train, monitor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/raw/models/omniASR-CTC-1B-v2-hf", help="HF Wav2Vec2ForCTC dir or hub id (aadel4/omniASR-CTC-1B-v2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=5000); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=500); ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--max-seconds", type=float, default=30.0); ap.add_argument("--monitor-n", type=int, default=200)
    ap.add_argument("--max-train", type=int); ap.add_argument("--no-brahuispeech", action="store_true")
    ap.add_argument("--brahuispeech-manifest", default="brahuispeech_train.jsonl",
                    help="file under data/manifests; brahuispeech_train_norm.jsonl for CV/Omni spelling (scripts/build_bs_normalization.py)")
    ap.add_argument("--extra-manifest", default="", help="comma-separated extra manifests under data/manifests (e.g. media_pseudo_s3.jsonl)")
    ap.add_argument("--layerdrop", type=float, default=0.05); ap.add_argument("--mask-time-prob", type=float, default=0.05)
    ap.add_argument("--no-grad-ckpt", action="store_true"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="resume from the last checkpoint in --out (needs --save-optimizer on the run that wrote it)")
    ap.add_argument("--save-optimizer", action="store_true", help="save optimizer/scheduler state in checkpoints (12 GB each) so --resume works")
    ap.add_argument("--smoke", action="store_true", help="2 steps on 8 clips, CPU-sized")
    ap.add_argument("--mps", action="store_true", help="allow training on Apple MPS (fp32, no bf16; ~2026-10-25 M1: used for the 300M student continuation when GPU credit ran out)")
    a = ap.parse_args()
    if a.mps and not torch.backends.mps.is_available():
        sys.exit("ERROR: --mps given but torch.backends.mps is not available")
    if not a.smoke and not a.mps and not torch.cuda.is_available():
        # A cu130 torch on a CUDA 12.8 host silently falls back to CPU at ~50x the cost (2026-10-18: 40 GPU-minutes lost).
        sys.exit("ERROR: CUDA not available; refusing to train on CPU. Check torch.version.cuda against nvidia-smi "
                 "(docs/GPU_TRAINING.md), or pass --smoke for the CPU smoke test.")
    if a.smoke:
        a.max_train, a.monitor_n, a.bs, a.grad_accum, a.max_steps, a.eval_every, a.warmup, a.workers = 8, 4, 2, 1, 2, 2, 0, 0

    from datasets import Dataset
    from transformers import AutoProcessor, Trainer, TrainingArguments, Wav2Vec2ForCTC

    base = str(ROOT / a.base) if (ROOT / a.base).exists() else a.base
    proc = AutoProcessor.from_pretrained(base)
    tok = proc.tokenizer
    train_rows, monitor_rows = build_pools(a)

    # Character coverage of the pretrained vocab (should be ~100% for Brahui).
    unk = sum(1 for r in train_rows[:5000] for c in for_scoring(r["text"]) if c != " " and c not in tok.get_vocab())
    print(f"chars not in vocab (first 5000 clips): {unk}", flush=True)

    def to_ds(rows):
        return Dataset.from_list([{"audio": r["audio"], "start": r.get("start"), "end": r.get("end"),
                                   "text": for_scoring(r["text"]), "duration": float(r["duration"])} for r in rows])
    ds_train, ds_mon = to_ds(train_rows), to_ds(monitor_rows)

    @dataclass
    class Collator:
        """Loads audio here (in dataloader workers), so datasets stay metadata-only."""
        def __call__(self, rows):
            waves = [load_audio(ROOT / r["audio"], r["start"], r["end"]) for r in rows]
            inputs = proc.feature_extractor(waves, sampling_rate=16000, padding=True, return_tensors="pt", return_attention_mask=True)
            labels = tok([r["text"] for r in rows], padding=True, return_tensors="pt")
            inputs["labels"] = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
            return inputs

    model = Wav2Vec2ForCTC.from_pretrained(
        base, ctc_loss_reduction="mean", ctc_zero_infinity=True, layerdrop=a.layerdrop,
        mask_time_prob=a.mask_time_prob, pad_token_id=tok.pad_token_id)
    model.freeze_feature_encoder()
    if not a.no_grad_ckpt and not a.smoke:
        model.gradient_checkpointing_enable()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_tr/1e6:.0f}M of {sum(p.numel() for p in model.parameters())/1e6:.0f}M", flush=True)

    import jiwer

    def keep_argmax(logits, labels):  # avoid materializing (N, T, 10288) float logits on eval
        return torch.argmax(logits, dim=-1)

    def compute_metrics(pred):
        ids = pred.predictions.copy()
        ids[ids < 0] = tok.pad_token_id  # Trainer pads argmax ids across batches with -100; decode those as blank, not garbage
        lab = pred.label_ids.copy(); lab[lab == -100] = tok.pad_token_id
        hyp = [h if h.strip() else " " for h in tok.batch_decode(ids)]
        ref = [r if r.strip() else " " for r in tok.batch_decode(lab, group_tokens=False)]
        return {"cer": jiwer.cer(ref, hyp), "wer": jiwer.wer(ref, hyp)}

    cuda = torch.cuda.is_available()
    if a.mps: torch.mps.set_per_process_memory_fraction(0.0)  # no hard cap: let MPS spill to unified memory rather than OOM
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(out), per_device_train_batch_size=a.bs, per_device_eval_batch_size=max(1, a.bs // 2),
        gradient_accumulation_steps=a.grad_accum, max_steps=a.max_steps, learning_rate=a.lr, warmup_steps=a.warmup,
        lr_scheduler_type="linear", weight_decay=0.0, max_grad_norm=1.0,
        eval_strategy="steps", eval_steps=a.eval_every, save_steps=a.eval_every, save_total_limit=1,
        logging_steps=1 if a.smoke else 25, bf16=cuda, dataloader_num_workers=a.workers,
        train_sampling_strategy="group_by_length", length_column_name="duration",
        load_best_model_at_end=True, metric_for_best_model="cer", greater_is_better=False,
        save_only_model=not a.save_optimizer,  # weights-only checkpoints (4 GB, not 12): best + newest + in-flight must fit the volume
        report_to=[], remove_unused_columns=False, use_cpu=not (cuda or a.mps), seed=SEED)
    trainer = Trainer(model=model, args=args, data_collator=Collator(), train_dataset=ds_train, eval_dataset=ds_mon,
                      compute_metrics=compute_metrics, preprocess_logits_for_metrics=keep_argmax,
                      processing_class=proc.feature_extractor)
    t0 = time.time()
    zero = trainer.evaluate(); print("monitor before training:", json.dumps(zero), flush=True)
    trainer.train(resume_from_checkpoint=a.resume or None)
    final = out / "final"
    model.save_pretrained(final); proc.save_pretrained(final)
    res = trainer.evaluate()
    res.update({"base": a.base, "train_clips": len(train_rows), "train_hours": sum(r["duration"] for r in train_rows) / 3600,
                "monitor_before": zero, "wall_hours": (time.time() - t0) / 3600, "args": vars(a)})
    (final / "train_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
