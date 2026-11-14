"""Speak dev sentences with an exported student VITS .onnx, tokenizing exactly as the browser will (vocab.json, a pad between letters).
  cd tools/onnx_export && uv run python tts_onnx_synth.py <model.onnx> <vocab.json> <dev.jsonl> <out_dir> [n]"""
import json, sys, time
from pathlib import Path
import numpy as np, onnxruntime as ort, soundfile as sf
onnx, vocab, dev, out = sys.argv[1:5]; n = int(sys.argv[5]) if len(sys.argv) > 5 else 40
V = json.load(open(vocab, encoding="utf-8")); rows = [json.loads(l) for l in open(dev, encoding="utf-8")][:n]
s = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"]); print("inputs", [i.name for i in s.get_inputs()], "outputs", [o.name for o in s.get_outputs()])
Path(out, "wavs").mkdir(parents=True, exist_ok=True); t0 = time.time(); sec = 0
for r in rows:
    ids = [0]
    for c in r["vowelled"].lower():
        if c in V: ids += [V[c], 0]
    x = np.array([ids], dtype=np.int64); y = s.run(None, {"input_ids": x, "attention_mask": np.ones_like(x)})[0][0].astype(np.float32)
    sf.write(str(Path(out, "wavs", r["id"] + ".wav")), y, 16000); sec += len(y) / 16000
print(f"{len(rows)} sentences, {sec:.0f} s audio in {time.time() - t0:.0f} s (x{sec / (time.time() - t0):.1f} real time on CPU), {Path(onnx).stat().st_size / 1e6:.0f} MB")
