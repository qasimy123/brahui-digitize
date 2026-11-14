"""Behavioural check that goes with the representation plots: what does the urd_Arab tag mean now?

On 120 tri-parallel BOUQuET dev sentences, base vs fine-tuned:
  urd→en and brh→en chrF++ (did Urdu understanding survive? did Brahui understanding appear?)
  en→urd_Arab output scored against the Urdu reference and the Brahui reference (which language does the tag produce?)

  uv run python scripts/interp/tag_behavior.py
"""
import json
import sys
from pathlib import Path

import sacrebleu
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import ENG_TAG, OUT, URD_TAG, load, triparallel  # noqa: E402

N = 120


@torch.no_grad()
def translate(model, tok, texts, src, tgt, bs=12):
    tok.src_lang = src
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
        gen = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tgt), num_beams=4, max_new_tokens=200)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out


def chrf(h, r):
    return round(sacrebleu.corpus_chrf(h, [r], word_order=2).score, 1)


def main():
    rows = triparallel()[:N]
    brh, urd, en = ([r[k] for r in rows] for k in ["brh", "urd", "en"])
    res, samples = {}, {}
    for which in ["base", "ft"]:
        model, tok = load(which)
        h_urd_en = translate(model, tok, urd, URD_TAG, ENG_TAG)
        h_brh_en = translate(model, tok, brh, URD_TAG, ENG_TAG)
        h_en_x = translate(model, tok, en, ENG_TAG, URD_TAG)
        res[which] = {"urd→en": chrf(h_urd_en, en), "brh→en": chrf(h_brh_en, en),
                      "en→urd_Arab vs Urdu ref": chrf(h_en_x, urd), "en→urd_Arab vs Brahui ref": chrf(h_en_x, brh)}
        samples[which] = [{"en": e, "en→urd_Arab": x, "urd→en": ue, "brh→en": be} for e, x, ue, be in list(zip(en, h_en_x, h_urd_en, h_brh_en))[:8]]
        print(which, res[which], flush=True)
        del model
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    (OUT / "tag_behavior.json").write_text(json.dumps({"n": N, "chrfpp": res, "samples": samples}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
