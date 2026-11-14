"""Dump per-clip CTC log-probs of a Wav2Vec2ForCTC checkpoint for a manifest/eval set, for offline decoder experiments
(scripts/ctc_lm_decode.py). Output: <out>.npz with id -> float16 [T, V] log-softmax, plus <out>.refs.jsonl.
  uv run python scripts/ctc_dump_logits.py --model runs/ctc300m_student/final --set cv_dev --out data/eval/logits/ctc300m_cv_dev"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from asr_eval import load_audio, device_for  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--set", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--device"); ap.add_argument("--limit", type=int); a = ap.parse_args()
from transformers import AutoProcessor, Wav2Vec2ForCTC
dev = device_for(a.device); proc = AutoProcessor.from_pretrained(a.model); model = Wav2Vec2ForCTC.from_pretrained(a.model).to(dev).eval()
mp = next(p for p in (ROOT / "data" / "eval" / f"{a.set}.jsonl", ROOT / "data" / "manifests" / f"{a.set}.jsonl") if p.exists())
rows = [json.loads(l) for l in open(mp, encoding="utf-8")][: a.limit or None]
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); arrs = {}; t0 = time.time()
with torch.no_grad(), open(str(out) + ".refs.jsonl", "w", encoding="utf-8") as f:
    for i, r in enumerate(rows):
        wav = load_audio(ROOT / r["audio"])[: 20 * 16000]
        inp = proc(wav, sampling_rate=16000, return_tensors="pt").input_values.to(dev)
        lp = torch.log_softmax(model(inp).logits[0].float(), dim=-1).cpu().numpy().astype(np.float16)
        arrs[r["id"]] = lp; f.write(json.dumps({"id": r["id"], "text": r["text"], "duration": r["duration"]}, ensure_ascii=False) + "\n")
        if i % 100 == 0: print(f"{i}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
np.savez_compressed(str(out) + ".npz", **arrs); print("wrote", out, len(arrs), "clips", f"{time.time()-t0:.0f}s")
