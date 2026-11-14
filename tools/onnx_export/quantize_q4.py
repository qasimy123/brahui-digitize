"""4-bit weight-only quantization (MatMulNBits, block 32) of an exported fp32 ONNX dir, written as *_q4.onnx next to the
int8 files so transformers.js can load it with dtype: 'q4'. Halves the download versus int8; WebGPU runs MatMulNBits
natively, WASM decodes it to fp32 on the fly.

  uv run python quantize_q4.py ../../export/student_a            # all fp32 .onnx in <dir>/onnx
"""
import sys
import time
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer



def sz(p: Path) -> str:
    return f"{p.stat().st_size/1e6:.0f} MB" if p.exists() else "-"


def main():
    d = Path(sys.argv[1]).resolve() / "onnx"
    block = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    for f in sorted(d.glob("*.onnx")):
        if f.stem.endswith(("_quantized", "_q4", "_fp16")):
            continue
        out = d / f"{f.stem}_q4.onnx"
        if out.exists():
            print("skip", out.name); continue
        t0 = time.time()
        m = onnx.load(str(f))
        q = MatMulNBitsQuantizer(m, block_size=block, is_symmetric=True, accuracy_level=4)
        q.process()
        q.model.save_model_to_file(str(out), use_external_data_format=False)
        print(f"{f.name} -> {out.name} ({sz(out)}) in {time.time()-t0:.0f}s")
    # merged decoder for transformers.js: merge the two q4 decoders if present
    dq, dpq, merged = d / "decoder_model_q4.onnx", d / "decoder_with_past_model_q4.onnx", d / "decoder_model_merged_q4.onnx"
    if dq.exists() and dpq.exists() and not merged.exists():
        from optimum.onnx.graph_transformations import merge_decoders
        merge_decoders(dq, dpq, save_path=merged, strict=False)
        print(f"merged -> {merged.name} ({sz(merged)})")


if __name__ == "__main__":
    main()
