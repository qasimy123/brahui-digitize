---
language:
- brh
- en
license: cc-by-nc-4.0
library_name: transformers.js
pipeline_tag: translation
base_model: qkhawaja/brahui-nllb-1.3b-r5
tags:
- brahui
- nllb
- onnx
- low-resource
---

# Brahui ↔ English translation for the browser (NLLB 1.3B r5, trimmed vocabulary, int8 ONNX)

[qkhawaja/brahui-nllb-1.3b-r5](https://huggingface.co/qkhawaja/brahui-nllb-1.3b-r5) with its vocabulary trimmed to
the tokens that occur in Brahui and English text, exported to ONNX and quantized to int8. Same model, same
behaviour, 1.4 GB instead of 5.5 GB. It translates in both directions; the in-browser demo uses it for
English → Brahui. For Brahui → English alone,
[qkhawaja/brahui-mt-student-c-onnx](https://huggingface.co/qkhawaja/brahui-mt-student-c-onnx) is faster.

On the project's development set the int8 build scores chrF++ 51.2 for Brahui → English, at about 0.7 seconds per
sentence on a laptop CPU with beam 4. See the parent model's card for full results, training data and limitations;
they all apply here.

Brahui uses NLLB's Urdu language code, `urd_Arab`. English is `eng_Latn`.

This is a research snapshot from a one-person project, not a supported product.

## Licence

CC-BY-NC-4.0, inherited from NLLB-200. Non-commercial use only.

Code and the full write-up: https://github.com/qasimy123/brahui-digitize
