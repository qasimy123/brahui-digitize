"""Build a tokenizer.json for a Wav2Vec2 CTC vocab so transformers.js can load the ASR export (HF has no fast tokenizer
for CTC, so optimum writes only vocab.json). Mirrors transformers.js's scripts/extra/wav2vec2.py recipe.
  uv run python build_ctc_tokenizer_json.py <export_dir>"""
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
vocab = json.load(open(d / "vocab.json", encoding="utf-8"))
cfg = json.load(open(d / "tokenizer_config.json", encoding="utf-8"))
delim = cfg.get("word_delimiter_token", "|")
by_id = sorted(vocab.items(), key=lambda kv: kv[1])
assert [i for _, i in by_id] == list(range(len(by_id))), "vocab ids must be contiguous"
specials = [k for k in ("<s>", "<pad>", "</s>", "<unk>") if k in vocab]
tj = {
    "version": "1.0", "truncation": None, "padding": None,
    "added_tokens": [{"id": vocab[k], "content": k, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True} for k in specials],
    "normalizer": None if delim == " " else {"type": "Replace", "pattern": {"String": " "}, "content": delim},
    "pre_tokenizer": {"type": "Split", "pattern": {"Regex": ""}, "behavior": "Isolated", "invert": False},
    "post_processor": None,
    # the CTC blank is whatever the tokenizer_config calls pad_token (omniASR/MMS: "<s>", id 0), not necessarily "<pad>"
    "decoder": {"type": "CTC", "pad_token": cfg.get("pad_token", "<pad>"), "word_delimiter_token": delim, "cleanup": True},
    "model": {"type": "Unigram", "unk_id": vocab.get("<unk>", 3), "vocab": [[k, 0.0] for k, _ in by_id], "byte_fallback": False},
}
json.dump(tj, open(d / "tokenizer.json", "w", encoding="utf-8"), ensure_ascii=False)
cfg.setdefault("tokenizer_class", "Wav2Vec2CTCTokenizer")
json.dump(cfg, open(d / "tokenizer_config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("wrote", d / "tokenizer.json", len(by_id), "tokens, delimiter", repr(delim))
