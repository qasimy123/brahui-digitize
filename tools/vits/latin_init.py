"""Turn the converted MMS Urdu VITS checkpoint (generator + discriminator, Arabic-script vocab) into the starting point of
the Brahui student: same weights, new vocabulary of vowelled Latin letters. Each new letter starts from the embedding of
the Urdu letter with the nearest sound, so training starts from Urdu-like speech instead of noise.
  tools/vits/.venv/bin/python tools/vits/latin_init.py tools/vits/init_urd runs/tts_student/init"""
import json, shutil, sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
src, dst = Path(sys.argv[1]), Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
CHARS = " ',.?!abcdefghijklmnopqrstuvwyzāīūḍḷṉṛṭ"; NEAR = {" ": " ", "'": "ئ", "a": "ا", "ā": "ا", "i": "ی", "ī": "ی", "e": "ے", "u": "و", "ū": "و", "o": "و", "b": "ب", "p": "پ", "t": "ت", "ṭ": "ٹ",
    "s": "س", "j": "ج", "c": "چ", "h": "ہ", "k": "ک", "d": "د", "ḍ": "ڈ", "z": "ز", "r": "ر", "ṛ": "ڑ", "g": "گ", "f": "ف", "q": "ق", "l": "ل", "ḷ": "ل", "m": "م", "n": "ن", "ṉ": "ں", "w": "و", "y": "ی"}
old = json.load(open(src / "vocab.json", encoding="utf-8")); sd = load_file(src / "model.safetensors"); E = sd["text_encoder.embed_tokens.weight"]
vocab = {"_": 0, **{c: i + 1 for i, c in enumerate(CHARS)}}; new = E.mean(0, keepdim=True).repeat(len(vocab), 1).clone(); new[0] = E[0]   # id 0 = pad, which VITS also puts between letters
for c, i in vocab.items():
    if c in NEAR and NEAR[c] in old and old[NEAR[c]] != 0: new[i] = E[old[NEAR[c]]]
    elif c == "d": new[i] = E[old["ڈ"]] * 0.5 + E.mean(0) * 0.5      # the Urdu د shares id 0 with the pad
sd["text_encoder.embed_tokens.weight"] = new; save_file(sd, dst / "model.safetensors", metadata={"format": "pt"})
cfg = json.load(open(src / "config.json")); cfg["vocab_size"] = len(vocab); json.dump(cfg, open(dst / "config.json", "w"), indent=2)
json.dump(vocab, open(dst / "vocab.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
tc = json.load(open(src / "tokenizer_config.json")); tc["pad_token"] = "_"; tc["language"] = "brh-latin"; tc.pop("added_tokens_decoder", None)
json.dump(tc, open(dst / "tokenizer_config.json", "w"), indent=2); json.dump({"<unk>": len(vocab)}, open(dst / "added_tokens.json", "w")); json.dump({"pad_token": "_", "unk_token": "<unk>"}, open(dst / "special_tokens_map.json", "w"))
shutil.copy(src / "preprocessor_config.json", dst / "preprocessor_config.json"); print("vocab", len(vocab), "->", dst)
