"""Rename external data files from onnxruntime's '<f>.onnx.data' to optimum/transformers.js's '<f>.onnx_data' and
rewrite the location field inside the .onnx. Usage: uv run python fix_external_data.py <onnx_dir>"""
import sys
from pathlib import Path
import onnx
from onnx.external_data_helper import convert_model_to_external_data
d = Path(sys.argv[1])
for f in sorted(d.glob("*.onnx")):
    old = f.with_suffix(".onnx.data")
    if not old.exists():
        continue
    m = onnx.load(str(f))  # loads external data into memory
    new_name = f.name + "_data"
    convert_model_to_external_data(m, all_tensors_to_one_file=True, location=new_name, size_threshold=1024)
    onnx.save_model(m, str(f))
    old.unlink()
    print(f"{f.name}: -> {new_name} ({(d / new_name).stat().st_size/1e6:.0f} MB)")
