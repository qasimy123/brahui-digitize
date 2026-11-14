"""Trim an NLLB (M2M100) checkpoint's 256k vocabulary down to the tokens actually used by brh/en/urd text.

The 256k x 1024 embedding (tied to lm_head) is ~262M of NLLB-1.3B's parameters and ~1 GB of fp32 weights.
Keeping only tokens seen in our corpora (plus specials, lang codes, and short fallback pieces) removes
most of it with no retraining: the kept rows are copied verbatim, so outputs are identical as long as
the tokenizer segments the input the same way (verified by `--check`).

Usage:
  uv run python scripts/trim_nllb_vocab.py --src runs/nllb_r5_retrans/final --dst runs/nllb_r5_trim/final
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PAR = ROOT / "data" / "parallel"
CORPORA = [
    "gold_train_nodev.jsonl", "ak2024_train.jsonl", "retranslated_brh_en.jsonl",
    "synthetic_fineweb_brh_en_filtered.jsonl", "fineweb_units.jsonl", "en_everyday.jsonl",
    "synthetic_brh_en.jsonl", "more_units.jsonl", "synthetic_finepdfs_brh_en.jsonl", "synthetic_bt_everyday.jsonl",
]
TEXT_KEYS = ("brh", "en", "gloss")
KEEP_SPECIAL = ["<s>", "<pad>", "</s>", "<unk>", "urd_Arab", "eng_Latn", "<mask>"]


def iter_texts(files):
    for f in files:
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            for k in TEXT_KEYS:
                if isinstance(d.get(k), str) and d[k]:
                    yield d[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/nllb_r5_retrans/final")
    ap.add_argument("--dst", default="runs/nllb_r5_trim/final")
    ap.add_argument("--extra", nargs="*", default=[], help="extra jsonl files whose text should be covered")
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--fallback-len", type=int, default=2, help="keep every piece this short (after stripping ▁) made of corpus chars")
    ap.add_argument("--check", action="store_true", help="verify new tokenizer segments the dev set identically")
    ap.add_argument("--keep-arabic", action="store_true", help="keep every Arabic-script piece in the vocab (~14k) for robustness on unseen Brahui words")
    a = ap.parse_args()
    src, dst = ROOT / a.src, ROOT / a.dst
    dst.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer, M2M100ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(src)
    tj = json.load(open(src / "tokenizer.json", encoding="utf-8"))
    vocab: dict[str, int] = tj["model"]["vocab"]
    id2piece = {i: p for p, i in vocab.items()}

    files = [PAR / c for c in CORPORA] + [ROOT / e for e in a.extra]
    missing = [f for f in files if not f.exists()]
    if missing:
        print("WARNING missing corpora (skipped):", [m.name for m in missing])
    files = [f for f in files if f.exists()]
    counts, chars = Counter(), set()
    n = 0
    batch = []

    def flush():
        for ids in tok(batch, add_special_tokens=False).input_ids:
            counts.update(ids)

    for t in iter_texts(files):
        chars.update(t)
        batch.append(t)
        n += 1
        if len(batch) == 2048:
            flush(); batch = []
    if batch:
        flush()
    print(f"tokenized {n} texts, {len(counts)} distinct ids used, {len(chars)} distinct chars")

    keep = {vocab[s] for s in KEEP_SPECIAL if s in vocab}
    keep |= {i for i, c in counts.items() if c >= a.min_count}
    fb = 0
    for p, i in vocab.items():
        core = p.lstrip("▁")
        if 0 < len(core) <= a.fallback_len and set(core) <= chars and i not in keep:
            keep.add(i); fb += 1
    if a.keep_arabic:
        import re
        arab = re.compile(r"^[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\u200c\u200d]+$")
        before = len(keep)
        keep |= {i for p, i in vocab.items() if p.lstrip("▁") and arab.match(p.lstrip("▁"))}
        print(f"keep-arabic added {len(keep)-before} pieces")
    keep_ids = sorted(keep)
    old2new = {o: nnew for nnew, o in enumerate(keep_ids)}
    print(f"keeping {len(keep_ids)} / {len(vocab)} pieces ({fb} short fallback pieces)")

    # --- tokenizer.json surgery ---
    new_vocab = {id2piece[o]: nnew for o, nnew in old2new.items()}
    kept_pieces = set(new_vocab)
    merges = tj["model"]["merges"]
    new_merges = []
    for m in merges:
        x, y = (m if isinstance(m, list) else m.split(" ", 1))
        if x in kept_pieces and y in kept_pieces and (x + y) in kept_pieces:
            new_merges.append(m)
    tj["model"]["vocab"], tj["model"]["merges"] = new_vocab, new_merges
    tj["added_tokens"] = [dict(t, id=old2new[t["id"]]) for t in tj["added_tokens"] if t["id"] in old2new]
    pp = tj.get("post_processor") or {}
    for st in (pp.get("special_tokens") or {}).values():
        st["ids"] = [old2new[i] for i in st["ids"]]
    json.dump(tj, open(dst / "tokenizer.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"merges {len(merges)} -> {len(new_merges)}")

    tc = json.load(open(src / "tokenizer_config.json", encoding="utf-8"))
    if "added_tokens_decoder" in tc:
        tc["added_tokens_decoder"] = {str(t["id"]): {k: v for k, v in t.items() if k not in ("id", "content")} | {"content": t["content"]}
                                      for t in tj["added_tokens"]}
    json.dump(tc, open(dst / "tokenizer_config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    for extra in ("special_tokens_map.json", "sentencepiece.bpe.model"):
        if (src / extra).exists():
            shutil.copy(src / extra, dst / extra)

    # --- model surgery ---
    model = M2M100ForConditionalGeneration.from_pretrained(src, dtype=torch.float32)
    idx = torch.tensor(keep_ids)
    w = model.model.shared.weight.data[idx].clone()
    for emb in (model.model.shared, model.model.encoder.embed_tokens, model.model.decoder.embed_tokens):
        emb.weight = torch.nn.Parameter(w)
        emb.num_embeddings = len(keep_ids)
    model.lm_head.weight = torch.nn.Parameter(w)
    model.lm_head.out_features = len(keep_ids)
    model.config.vocab_size = len(keep_ids)
    model.config.tie_word_embeddings = True
    model.tie_weights()
    model.save_pretrained(dst, safe_serialization=True)
    json.dump({"src": str(a.src), "kept": len(keep_ids), "orig": len(vocab), "corpora": [str(f) for f in files]},
              open(dst / "trim_summary.json", "w"), indent=2)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M, saved to {dst}")

    if a.check:
        new_tok = AutoTokenizer.from_pretrained(dst)
        dev = [json.loads(l) for l in open(ROOT / "data/dev/mt_dev_clean.jsonl", encoding="utf-8")]
        texts = [d["brh"] for d in dev] + [d["en"] for d in dev]
        bad = 0
        for t in texts:
            o = tok(t, add_special_tokens=False).input_ids
            nn_ = new_tok(t, add_special_tokens=False).input_ids
            if [old2new.get(i, -1) for i in o] != nn_:
                bad += 1
                if bad <= 3:
                    print("MISMATCH:", t[:60], tok.convert_ids_to_tokens(o)[:12], new_tok.convert_ids_to_tokens(nn_)[:12])
        print(f"segmentation check: {len(texts)-bad}/{len(texts)} identical")
        m2 = M2M100ForConditionalGeneration.from_pretrained(dst)
        assert m2.lm_head.weight.data_ptr() == m2.model.shared.weight.data_ptr(), "lm_head not tied after reload"
        print("reload ok, lang ids:", new_tok.convert_tokens_to_ids("urd_Arab"), new_tok.convert_tokens_to_ids("eng_Latn"))


if __name__ == "__main__":
    main()
