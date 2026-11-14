"""Experiment 1: where did fine-tuning change NLLB-1.3B? Streams both checkpoints tensor by tensor (no forward passes).

Per weight matrix: relative change ||dW||/||W|| and how low-rank dW is (share of its energy in the top 8 singular values).
Per vocabulary row of the shared embedding: how far each token moved, so we can see which subwords fine-tuning rewrote.
Writes evals/interp/weight_diff.json and weight_diff.png.

  uv run --with matplotlib python scripts/interp/weight_diff.py
"""
import json
import re
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
from common import BASE, FT, OUT  # noqa: E402


def open_base():
    p = Path(snapshot_download(BASE, allow_patterns=["*.json", "*.bin", "*.safetensors"]))
    st = sorted(p.glob("*.safetensors"))
    if st:
        f = safe_open(str(st[0]), "pt")
        return {k: (lambda k=k: f.get_tensor(k)) for k in f.keys()}
    sd = torch.load(sorted(p.glob("*.bin"))[0], map_location="cpu", mmap=True, weights_only=True)
    return {k: (lambda k=k: sd[k]) for k in sd}


def norm_key(k):
    return k if k.startswith(("model.", "lm_head")) else "model." + k


def bucket(k):
    m = re.match(r"model\.(encoder|decoder)\.layers\.(\d+)\.(\w+)\.?(\w*)", k)
    if not m:
        return None
    stack, layer, mod, sub = m.groups()
    kind = {"self_attn": "self-attn", "encoder_attn": "cross-attn", "fc1": "ffn", "fc2": "ffn"}.get(mod)
    if mod.endswith("layer_norm"):
        kind = "layernorm"
    return stack, int(layer), kind, sub


def main():
    base = {norm_key(k): v for k, v in open_base().items()}
    ft = safe_open(FT + "/model.safetensors", "pt")
    rows, emb = [], None
    for k in ft.keys():
        kk = norm_key(k)
        if kk not in base or k == "lm_head.weight":
            continue
        w0, w1 = base[kk]().float(), ft.get_tensor(k).float()
        if w0.shape != w1.shape:
            print("shape mismatch", k, w0.shape, w1.shape); continue
        d = w1 - w0
        if kk == "model.shared.weight":
            emb = (w0, d)
            continue
        r = {"name": kk, "rel": (d.norm() / w0.norm()).item(), "n": w0.numel()}
        if d.ndim == 2:
            s = torch.linalg.svdvals(d)
            e = s.pow(2)
            r["top8_energy"] = (e[:8].sum() / e.sum()).item()
        b = bucket(kk)
        if b:
            r.update(stack=b[0], layer=b[1], kind=b[2], sub=b[3])
        rows.append(r)
    # Embedding rows: which tokens moved
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    w0, d = emb
    row_rel = d.norm(dim=1) / w0.norm(dim=1).clamp_min(1e-6)
    top = row_rel.topk(60).indices.tolist()
    lang = {t: row_rel[tok.convert_tokens_to_ids(t)].item() for t in ["urd_Arab", "eng_Latn", "pes_Arab", "pbt_Arab", "snd_Arab", "hin_Deva"]}
    # How many tokens moved at all: the fine-tune only touches embeddings of tokens it saw
    moved = (row_rel > 1e-4).sum().item()
    res = {"matrices": rows, "embedding": {
        "rel_total": (d.norm() / w0.norm()).item(), "rows_moved": moved, "rows": len(row_rel),
        "median_row_rel_moved": row_rel[row_rel > 1e-4].median().item(),
        "lang_tokens": lang,
        "top_tokens": [(tok.convert_ids_to_tokens(i), round(row_rel[i].item(), 4)) for i in top]}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "weight_diff.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    plot(rows, res["embedding"])
    print(json.dumps(res["embedding"], ensure_ascii=False)[:1500])


def plot(rows, emb):
    import matplotlib.pyplot as plt
    import numpy as np
    kinds = ["self-attn", "cross-attn", "ffn"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, stack in zip(axes, ["encoder", "decoder"]):
        for kind in kinds:
            if stack == "encoder" and kind == "cross-attn":
                continue
            ys = []
            for L in range(24):
                rs = [r for r in rows if r.get("stack") == stack and r.get("layer") == L and r.get("kind") == kind and r["name"].endswith("weight")]
                ys.append(np.sqrt(sum(r["rel"] ** 2 * r["n"] for r in rs) / max(1, sum(r["n"] for r in rs))) if rs else np.nan)
            ax.plot(range(24), ys, marker="o", ms=3, label=kind)
        ax.set_title(f"{stack}: relative weight change ||ΔW||/||W||")
        ax.set_xlabel("layer"); ax.grid(alpha=.3); ax.legend()
    fig.suptitle(f"NLLB-1.3B base → r5 fine-tune. Embedding: {emb['rows_moved']:,}/{emb['rows']:,} rows moved; "
                 f"urd_Arab token Δ={emb['lang_tokens']['urd_Arab']:.3f}, eng_Latn Δ={emb['lang_tokens']['eng_Latn']:.3f}", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "weight_diff.png", dpi=130)


if __name__ == "__main__":
    main()
