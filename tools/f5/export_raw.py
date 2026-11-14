"""Export the live (non-EMA) weights of a training checkpoint as inference weights, on CPU, without loading F5:
  python tools/f5/export_raw.py <model_last.pt> <out.safetensors>     (for scoring a checkpoint on another machine)"""
import sys, torch
from safetensors.torch import save_file
torch.set_num_threads(2); ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
save_file({"ema_model." + k: v for k, v in ck["model_state_dict"].items()}, sys.argv[2]); print("update", ck.get("update"), "->", sys.argv[2])
