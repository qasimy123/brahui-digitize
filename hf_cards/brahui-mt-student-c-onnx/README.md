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
- distillation
- low-resource
metrics:
- chrf
---

# Brahui → English translation, small enough for the browser (student C, ONNX)

A distilled version of [qkhawaja/brahui-nllb-1.3b-r5](https://huggingface.co/qkhawaja/brahui-nllb-1.3b-r5) that
translates Brahui (`brh`, Perso-Arabic script) into English. It matches its teacher's score and translates a
20-word sentence in under a second in Chrome on a laptop, with no server.

**One direction only.** Given English input it echoes the input. Use the teacher for English → Brahui.

This is a research snapshot from a one-person project, not a supported product.

## Results

| model | parameters | download (int8) | dev chrF++, Brahui → English | browser time, 20-word sentence |
|---|---|---|---|---|
| teacher (NLLB 1.3B r5, vocabulary trimmed) | 1.14B | 1.42 GB | 51.2 | about 2 s |
| **this model, int8** | 687M | 1.1 GB | **51.2** | **0.88 s** |

Browser time is Chrome on an Apple M1, WebAssembly with 8 threads, beam 4. chrF++ is on the project's 302-sentence
development set. The teacher's limits carry over: an LLM judge found about 41% of the teacher's outputs fully
preserved the source meaning.

## How it was made

1. The teacher's vocabulary was trimmed to the tokens that occur in Brahui and English text.
2. The student keeps all 24 encoder layers and 6 of the 24 decoder layers (layers 0, 5, 9, 14, 18, 23). Translation
   time in the browser is dominated by the decoder, so a deep encoder with a shallow decoder keeps quality while
   cutting latency.
3. It was trained for 2 epochs on the teacher's own translations of about 220,000 Brahui sentences, half of them real text and half back-translated from English (sequence-level
   distillation), then exported to ONNX and quantized to int8.

Training cost about $1.40 of GPU time.

## Files

- `onnx/encoder_model_quantized.onnx` (+ `.onnx_data`) and `onnx/decoder_model_merged_quantized.onnx`: the int8
  build the demo uses.
- `onnx/encoder_model.onnx`, `onnx/decoder_model.onnx`, `onnx/decoder_with_past_model.onnx`: full precision.

Brahui uses NLLB's Urdu language code: source `urd_Arab`, target `eng_Latn`.

The working reference for loading and running it with transformers.js is the in-browser demo in the project
repository (`demo/web/`).

## Licence

CC-BY-NC-4.0, inherited from NLLB-200. Non-commercial use only.

Code, evaluation scripts and the full write-up: https://github.com/qasimy123/brahui-digitize

If you have a concern about this model, open a discussion in the Community tab of this repository.
