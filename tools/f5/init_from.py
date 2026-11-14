"""Start an F5 fine-tune from an earlier fine-tune instead of the base model: copy its weights and re-map the text-embedding
rows from its vocab to the new dataset's vocab (tokens new to this dataset get the mean row). Overwrites the file
prep_finetune.py wrote, which is what finetune_cli loads.

  python tools/f5/init_from.py --weights runs/f5_multi_roman/final/model_final.safetensors --old-vocab runs/f5_multi_roman/final/vocab.txt \
      --new-vocab tools/f5/F5-TTS/data/<name>_custom/vocab.txt --out tools/f5/F5-TTS/ckpts/<name>/pretrained_model_1250000_ext.safetensors
"""
import argparse, torch
from safetensors.torch import load_file, save_file
ap = argparse.ArgumentParser()
for k in ("weights", "old-vocab", "new-vocab", "out"): ap.add_argument("--" + k, required=True)
a = ap.parse_args()
def vocab(p):
    v = open(p, encoding="utf-8").read().split("\n")
    if v[-1] == "": v.pop()
    return v
old, new = vocab(a.old_vocab), vocab(a.new_vocab); sd = load_file(a.weights, device="cpu"); key = "ema_model.transformer.text_embed.text_embed.weight"
e = sd[key]; assert e.shape[0] == len(old) + 1, (e.shape, len(old)); pos = {t: i for i, t in enumerate(old)}
rows = [e[0]]; mean = e[1:].mean(0); miss = [t for t in new if t not in pos]
for t in new: rows.append(e[pos[t] + 1] if t in pos else mean.clone())
sd[key] = torch.stack(rows).to(e.dtype)
# raw (non-EMA) exports lack the EMA module's own buffers, which the trainer's load_state_dict requires; step 0 restarts the EMA warm-up
sd.setdefault("initted", torch.tensor(True)); sd.setdefault("step", torch.tensor(0))
save_file(sd, a.out)
print(f"init from {a.weights}: vocab {len(old)} -> {len(new)}, {len(miss)} new tokens {miss}", flush=True)
