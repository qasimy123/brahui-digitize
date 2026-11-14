"""fp16 copies of an export's fp32 ONNX files (*_fp16.onnx) for transformers.js dtype 'fp16' on WebGPU.
  uv run python convert_fp16.py ../../export/student_a_g8
  uv run python convert_fp16.py ../../export/ctc1b_s3 --fp32-node /wav2vec2/encoder/layers.0/attention/MatMul --fp32-node /wav2vec2/encoder/layers.0/attention/Softmax

Overflow: WebGPU really computes in fp16 (ORT's CPU EP upcasts internally, so a native fp16 check can pass while the
browser returns all-NaN logits). Find offending tensors with --scan (runs the fp32 graph on a short clip and lists
every intermediate whose |max| exceeds 2e4), then pin those nodes to fp32 with --fp32-node. The MMS 1B (s3) needs
its layer-0 attention scores (peak 1.1e5) kept in fp32; the 300M student needs nothing.
Pinned nodes require shape inference (path-based, works past the 2 GB protobuf limit), and the converter's
fp32->fp16->fp32 cast round-trip between two pinned nodes is collapsed, otherwise the round-trip clips to inf anyway."""
import argparse, time
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto
from onnxruntime.transformers.float16 import convert_float_to_float16
from optimum.onnx.graph_transformations import merge_decoders

ap = argparse.ArgumentParser()
ap.add_argument("export_dir"); ap.add_argument("--fp32-node", action="append", default=[], help="node name to keep in fp32 (repeatable)")
ap.add_argument("--scan", metavar="CLIP_NPY", help="report fp32 intermediates > 2e4 on this 16 kHz float32 clip (CTC models) and exit")
a = ap.parse_args()
d = Path(a.export_dir).resolve() / "onnx"


def cast_to(n): return next(x.i for x in n.attribute if x.name == "to")


def collapse_cast_roundtrips(g):
    prod = {o: n for n in g.node for o in n.output}; rename, dead = {}, set()
    for n in g.node:
        if n.op_type == "Cast" and cast_to(n) == TensorProto.FLOAT:
            p = prod.get(n.input[0])
            if p is not None and p.op_type == "Cast" and cast_to(p) == TensorProto.FLOAT16:
                rename[n.output[0]] = p.input[0]; dead.add(n.name)
    for n in g.node:
        for i, x in enumerate(n.input):
            if x in rename: n.input[i] = rename[x]
    keep = [n for n in g.node if n.name not in dead]; del g.node[:]; g.node.extend(keep)
    return len(dead)


if a.scan:
    import onnxruntime as ort
    si = d / "model_si.onnx"; onnx.shape_inference.infer_shapes_path(str(d / "model.onnx"), str(si))
    m = onnx.load(str(si), load_external_data=False); si.unlink()
    prod = {o: n for n in m.graph.node for o in n.output}; have = {o.name for o in m.graph.output}
    for vi in m.graph.value_info:
        if vi.name not in have and vi.type.tensor_type.elem_type == TensorProto.FLOAT: m.graph.output.append(vi)
    dbg = d / "model_dbg.onnx"; onnx.save(m, str(dbg))
    s = ort.InferenceSession(str(dbg), providers=["CPUExecutionProvider"]); dbg.unlink()
    wav = np.load(a.scan); x = ((wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)).astype(np.float32)[None]
    outs = s.run(None, {s.get_inputs()[0].name: x})
    big = sorted(((float(np.abs(v).max()), o.name, prod[o.name].op_type, prod[o.name].name) for o, v in zip(s.get_outputs(), outs) if v.size and np.abs(v).max() > 2e4), reverse=True)
    print(f"{len(big)} tensors > 2e4 (fp16 max 65504)")
    for mx, nm, op, node in big: print(f"{mx:12.0f} {op:12s} node={node}")
    raise SystemExit

dq, dpq, merged32 = d / "decoder_model.onnx", d / "decoder_with_past_model.onnx", d / "decoder_model_merged.onnx"
if dq.exists() and dpq.exists() and not merged32.exists():  # merge in fp32 first; converting merged graphs is what works
    merge_decoders(dq, dpq, save_path=merged32, strict=False); print(f"merged fp32 -> {merged32.name}")
for f in sorted(d.glob("*.onnx")):
    if f.stem.endswith(("_quantized", "_q4", "_fp16", "_si", "_dbg")) or f.stem in ("decoder_model", "decoder_with_past_model"):
        continue
    out = d / f"{f.stem}_fp16.onnx"
    if out.exists():
        continue
    t0 = time.time()
    if a.fp32_node:
        si = d / f"{f.stem}_si.onnx"; onnx.shape_inference.infer_shapes_path(str(f), str(si)); m = onnx.load(str(si)); si.unlink()
    else:
        m = onnx.load(str(f))
    m16 = convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=True, node_block_list=a.fp32_node or None)
    if a.fp32_node: print(f"  pinned {len(a.fp32_node)} nodes to fp32; collapsed {collapse_cast_roundtrips(m16.graph)} cast round-trips")
    onnx.save(m16, str(out), save_as_external_data=False)
    print(f"{f.name} -> {out.name} ({out.stat().st_size/1e6:.0f} MB) in {time.time()-t0:.0f}s")
