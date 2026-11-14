---
language:
- brh
license: apache-2.0
library_name: transformers.js
pipeline_tag: automatic-speech-recognition
base_model: aadel4/omniASR-CTC-300M-v2
tags:
- brahui
- wav2vec2
- ctc
- onnx
- low-resource
metrics:
- cer
---

# Brahui ASR for the browser: omniASR CTC-300M student + character language model (ONNX)

A 315M-parameter speech recognizer for Brahui (`brh`, Perso-Arabic script) that runs locally in a web browser,
with a small character language model that makes it slightly more accurate than the 1B model it learned from.

This is a research snapshot from a one-person project, not a supported product.

## Results

Character error rate on the Common Voice Brahui test split (882 clips).

| model and decoding | CER |
|---|---|
| 1B teacher ([qkhawaja/brahui-asr-ctc1b-s3](https://huggingface.co/qkhawaja/brahui-asr-ctc1b-s3)), greedy | 8.6 |
| this model, greedy | 9.9 |
| **this model, beam search + character 6-gram language model** | **8.0** |

On spontaneous speech (Omnilingual test) the model scores 12.6 greedy, against 12.0 for the teacher.

Speed in Chrome on an Apple M1: with WebGPU and the fp16 build, 0.21 s for a 3-second sentence and 0.65 s for a
9-second clip. With WebAssembly and the int8 build, about 0.3 times real time.

## Files

| file | size | use |
|---|---|---|
| `onnx/model_quantized.onnx` | 366 MB | int8, for WebAssembly / CPU |
| `onnx/model_fp16.onnx` | 652 MB | half precision, for WebGPU |
| `onnx/model.onnx` | 1.3 GB | full precision |
| `brh_char6.blm` | 20 MB | pruned character 6-gram language model for beam search |

Input is 16 kHz mono audio; output is CTC logits over the tokenizer in this repository. Keep clips to 20 seconds
or less. The beam search decoder that reads `brh_char6.blm` is `demo/web/ctc_lm.js` in the project repository
(beam 16, alpha 0.5, beta 0.5), and the demo there is the working reference for running the model with
transformers.js.

## Training data

The acoustic model is omniASR CTC-300M v2 fine-tuned for 4,000 steps on:

- BrahuiSpeech-70H-V2 (MIT), transcripts normalized to Common Voice spelling.
- 10.4 hours of Brahui broadcast audio from public YouTube channels, transcribed automatically by the 1B teacher
  (pseudo-labels). That audio is not redistributed, here or anywhere else in this project.

The language model is a Kneser-Ney character 6-gram over 111,000 Brahui sentences: transcripts of the open speech
datasets, Brahui web text, and the automatic broadcast transcripts. Sentences that appear in the test sets were
removed first.

## Limitations

- The language model pulls output toward common spellings. That helps on read speech and can hurt on names and
  rare words.
- Same caveats as the teacher: little conversational speech in training, unstandardized spelling, and no
  systematic review by native speakers.

## Licence

Apache-2.0, the licence of the base model.

Code, evaluation scripts and the full write-up: https://github.com/qasimy123/brahui-digitize

If you have a concern about this model, including if you are a speaker in the broadcast audio it learned from,
open a discussion in the Community tab of this repository.
