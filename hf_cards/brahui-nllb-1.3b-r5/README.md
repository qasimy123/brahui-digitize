---
language:
- brh
- en
license: cc-by-nc-4.0
library_name: transformers
pipeline_tag: translation
base_model: facebook/nllb-200-distilled-1.3B
tags:
- brahui
- nllb
- low-resource
- synthetic-data
metrics:
- chrf
---

# Brahui ↔ English translation: NLLB-200 1.3B fine-tuned (round 5)

Translates between Brahui (`brh`, Perso-Arabic script) and English, in both directions. NLLB-200 has no Brahui, so
this is `facebook/nllb-200-distilled-1.3B` fine-tuned on a parallel corpus that is almost entirely machine-made:
real Brahui text translated into English by a large language model, plus back-translations of English.

This is a research snapshot from a one-person project, not a supported product.

## Results

chrF++ on sealed test sets that were never used for training or model selection.

| direction | test set | chrF++ |
|---|---|---|
| English → Brahui | SMOL held-out (200 sentences) | 40.0 |
| English → Brahui | BOUQuET test (854 sentences) | 38.2 |
| Brahui → English | BOUQuET test (854 sentences) | 50.2 |
| English → Brahui → English round trip | BOUQuET test | 66.2 |

An LLM judge rated whether each Brahui → English output preserved the meaning of the source, on a development set
with cleaned references: 41% did. That is the honest headline. The model gets the gist of news-style text and often
mistranslates content words elsewhere.

This model ties the previous round (r4, same architecture, older training targets) on every measure above.
Retranslating the training corpus with a much better teacher (80% meaning preserved against about 35%) did not
make the student measurably better.

## Use

Brahui uses NLLB's Urdu language code, `urd_Arab`. There is no separate Brahui code.

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

name = "qkhawaja/brahui-nllb-1.3b-r5"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSeq2SeqLM.from_pretrained(name).eval()

def translate(text, src, tgt):
    tok.src_lang = src
    enc = tok(text, return_tensors="pt")
    out = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
                         num_beams=4, max_new_tokens=200)
    return tok.batch_decode(out, skip_special_tokens=True)[0]

translate("Where is the nearest hospital?", "eng_Latn", "urd_Arab")   # English → Brahui
translate(brahui_text, "urd_Arab", "eng_Latn")                        # Brahui → English
```

Because the Urdu code is reused, the model no longer translates Urdu.

## Training data

About 270,000 sentence pairs, trained from the base model for 3 epochs.

| part | pairs | how it was made |
|---|---|---|
| Curated real Brahui, retranslated | 22,776 | Brahui sentences from open corpora, translated into English by Claude Opus agents that looked words up in dictionaries and a glossed corpus |
| Everyday English, back-translated | about 130,000 | DailyDialog and Tatoeba English, translated into Brahui by an earlier version of this model; used for Brahui → English only |
| Web Brahui, translated | about 45,000 | FineWeb-2 Brahui text translated into English by Claude |
| Human-translated pairs | the remainder | SMOL and BOUQuET training portions, and other small open sets |

The synthetic pairs with open licences are published as a dataset; see the project repository.

## Limitations

- Most of the training targets were written by a machine. Errors in the teacher's understanding of Brahui are
  baked in.
- Strongest on news, religious and political text, which dominate written Brahui online. Weakest on everyday
  conversation and on rare words.
- Evaluation is automatic (chrF++ and an LLM judge). One native speaker has spot-checked the output; no systematic
  human evaluation has been done.
- Do not rely on it for medical, legal or safety-critical translation.

## Licence

CC-BY-NC-4.0, inherited from NLLB-200. Non-commercial use only. The back-translated portion of the training data
derives from DailyDialog, which is CC BY-NC-SA 4.0.

Code, evaluation scripts and the full write-up: https://github.com/qasimy123/brahui-digitize

If you have a concern about this model, open a discussion in the Community tab of this repository.
