"""Score an exported ONNX NLLB dir on the MT dev set using onnxruntime (CPU) via optimum.

  uv run python eval_onnx_mt.py --export ../../export/nllb_r5_trim --variant quantized --out ../../evals/mt/serve/r5_trim_onnx_q8
Uses the raw `tokenizers` file (transformers 4.x cannot parse the 5.x NLLB tokenizer_config).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from sacrebleu.metrics import BLEU, CHRF
from tokenizers import Tokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--export", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--variant", default="quantized", help="'' for fp32, 'quantized' for int8")
ap.add_argument("--dev", default="../../data/dev/mt_dev_clean.jsonl"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--beams", type=int, default=4); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--threads", type=int, default=0)
a = ap.parse_args()
exp = Path(a.export).resolve(); suf = f"_{a.variant}" if a.variant else ""
import onnxruntime as ort
so = ort.SessionOptions()
if a.threads: so.intra_op_num_threads = a.threads
model = ORTModelForSeq2SeqLM.from_pretrained(exp, subfolder="onnx", encoder_file_name=f"encoder_model{suf}.onnx",
                                             decoder_file_name=f"decoder_model_merged{suf}.onnx", use_cache=True, session_options=so)
tok = Tokenizer.from_file(str(exp / "tokenizer.json"))
SRC, TGT, EOS, PAD = tok.token_to_id("urd_Arab"), tok.token_to_id("eng_Latn"), tok.token_to_id("</s>"), tok.token_to_id("<pad>")
rows = [json.loads(l) for l in open(a.dev, encoding="utf-8") if l.strip()][: a.limit or None]
hyps, t0 = [], time.time()
for i in range(0, len(rows), a.bs):
    batch = [r["brh"] for r in rows[i:i + a.bs]]
    ids = [[SRC] + tok.encode(t, add_special_tokens=False).ids[:254] + [EOS] for t in batch]
    L = max(map(len, ids))
    input_ids = torch.tensor([x + [PAD] * (L - len(x)) for x in ids]); attn = (input_ids != PAD).long()
    gen = model.generate(input_ids=input_ids, attention_mask=attn, forced_bos_token_id=TGT, num_beams=a.beams, max_new_tokens=256)
    hyps += [tok.decode([t for t in g.tolist() if t not in (PAD, EOS, SRC, TGT, 0)]) for g in gen]
    print(f"{i+len(batch)}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
dt = time.time() - t0
refs = [r["en"] for r in rows]
res = {"chrfpp": round(CHRF(word_order=2).corpus_score(hyps, [refs]).score, 2), "bleu": round(BLEU(effective_order=True).corpus_score(hyps, [refs]).score, 2),
       "n": len(rows), "sec": round(dt, 1), "sec_per_sent": round(dt / len(rows), 2), "export": str(exp), "variant": a.variant or "fp32", "beams": a.beams, "threads": a.threads}
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
with open(out / "brh2en_dev.jsonl", "w", encoding="utf-8") as f:
    for r, h in zip(rows, hyps): f.write(json.dumps({"id": r["id"], "src": r["brh"], "ref": r["en"], "hyp": h}, ensure_ascii=False) + "\n")
json.dump(res, open(out / "brh2en_summary.json", "w"), indent=2); print(json.dumps(res))
