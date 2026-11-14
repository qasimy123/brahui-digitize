"""Export a HF checkpoint to ONNX (optimum) and produce int8 dynamic-quantized copies in the layout transformers.js expects.

Run from tools/onnx_export (own venv, transformers 4.51 + optimum 1.25):
  uv run python export_quant.py --model ../../runs/nllb_r5_trim/final --out ../../export/nllb_r5_trim --kind mt
  uv run python export_quant.py --model ../../runs/ctc1b_brh_s3/final --out ../../export/ctc1b_s3 --kind asr

Output layout (what @huggingface/transformers loads from a model dir):
  <out>/config.json, tokenizer*.json, preprocessor_config.json
  <out>/onnx/encoder_model.onnx, decoder_model_merged.onnx  (+ *_quantized.onnx)   [mt]
  <out>/onnx/model.onnx (+ model_quantized.onnx)                                     [asr]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

TASK = {"mt": "text2text-generation-with-past", "asr": "automatic-speech-recognition"}


def sz(p: Path) -> str:
    n = p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return f"{n/1e6:.0f} MB"


def quantize(src: Path, dst: Path, per_channel: bool):
    t0 = time.time()
    m = onnx.load(str(src), load_external_data=False)
    ext = any(t.data_location == onnx.TensorProto.EXTERNAL for t in m.graph.initializer)
    quantize_dynamic(
        str(src), str(dst), weight_type=QuantType.QInt8, per_channel=per_channel, reduce_range=False,
        op_types_to_quantize=["MatMul", "Gemm", "Gather"],  # weights only; Gather = the embedding table (130 MB fp32 for a 32k x 1024 vocab)
        use_external_data_format=ext,
    )
    print(f"  quantized {src.name} -> {dst.name} ({sz(dst)}) in {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", choices=TASK, required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--per-channel", action="store_true")
    ap.add_argument("--skip-export", action="store_true", help="reuse an existing fp32 export in <out>/onnx")
    a = ap.parse_args()
    model, out = Path(a.model).resolve(), Path(a.out).resolve()
    onnx_dir = out / "onnx"
    out.mkdir(parents=True, exist_ok=True)

    if not a.skip_export:
        if onnx_dir.exists():
            shutil.rmtree(onnx_dir)
        cmd = [sys.executable, "-m", "optimum.exporters.onnx", "--model", str(model), "--task", TASK[a.kind],
               "--opset", str(a.opset), "--framework", "pt", str(onnx_dir)]
        if a.kind == "mt":
            # merging fp32 decoders of a >1B model overflows protobuf's 2 GB limit; we merge after quantizing instead
            cmd += ["--library-name", "transformers", "--no-post-process"]
        print("exporting:", " ".join(cmd))
        t0 = time.time()
        subprocess.run(cmd, check=True, env={**os.environ, "TOKENIZERS_PARALLELISM": "false"})
        print(f"export done in {time.time()-t0:.0f}s")
        # optimum writes config/tokenizer files next to the onnx files; transformers.js wants them one level up
        for f in list(onnx_dir.iterdir()):
            if f.suffix in (".json", ".model", ".txt") and not f.name.endswith(".onnx_data"):
                shutil.move(str(f), out / f.name)
        for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "preprocessor_config.json", "vocab.json"):
            if (model / f).exists() and not (out / f).exists():
                shutil.copy(model / f, out / f)

    print("fp32 export:", sz(onnx_dir))
    for f in sorted(onnx_dir.glob("*.onnx")):
        if f.stem.endswith("_quantized"):
            continue
        print(f"  {f.name}: {sz(f)}" + (f" (+{sz(f.with_suffix('.onnx_data'))} external)" if f.with_suffix(".onnx_data").exists() else ""))
    for f in sorted(onnx_dir.glob("*.onnx")):
        if f.stem.endswith("_quantized"):
            continue
        quantize(f, onnx_dir / f"{f.stem}_quantized.onnx", a.per_channel)

    dq, dpq = onnx_dir / "decoder_model_quantized.onnx", onnx_dir / "decoder_with_past_model_quantized.onnx"
    if a.kind == "mt" and dq.exists() and dpq.exists() and not (onnx_dir / "decoder_model_merged_quantized.onnx").exists():
        from optimum.onnx.graph_transformations import merge_decoders
        t0 = time.time()
        merge_decoders(dq, dpq, save_path=onnx_dir / "decoder_model_merged_quantized.onnx", strict=False)
        print(f"  merged int8 decoders -> decoder_model_merged_quantized.onnx ({sz(onnx_dir / 'decoder_model_merged_quantized.onnx')}) in {time.time()-t0:.0f}s")
    subprocess.run([sys.executable, str(Path(__file__).parent / "fix_external_data.py"), str(onnx_dir)], check=True)
    summary = {n: sz(onnx_dir / n) for n in sorted(os.listdir(onnx_dir))}
    json.dump(summary, open(out / "export_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
