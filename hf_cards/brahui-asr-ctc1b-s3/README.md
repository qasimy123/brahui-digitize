---
language:
- brh
license: apache-2.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
base_model: aadel4/omniASR-CTC-1B-v2
tags:
- brahui
- wav2vec2
- ctc
- low-resource
datasets:
- mozilla-foundation/common_voice
- facebook/omnilingual-asr-corpus
- TBOGamer22/BrahuiSpeech-70H-V2
metrics:
- cer
- wer
---

# Brahui ASR: omniASR CTC-1B fine-tuned (stage 3)

Speech recognition for Brahui (`brh`), a Dravidian language of Balochistan, written in Perso-Arabic script.
This is Meta's omniASR CTC-1B v2 (in Hugging Face `Wav2Vec2ForCTC` format) fully fine-tuned on about 80 hours of
Brahui speech.

This is a research snapshot from a one-person project, not a supported product.

## Results

Character and word error rate on two held-out sets. Brackets are 95% confidence intervals from 1,000 bootstrap
resamples over clips.

| model | Common Voice test CER (read speech, 882 clips) | Omnilingual test CER (spontaneous, 106 segments) |
|---|---|---|
| Whisper-Small Brahui (community model) | 18.9 [18.3–19.8] | 19.1 [18.3–20.0] |
| omniASR CTC-1B v2, zero-shot | 19.6 [18.9–20.4] | 16.4 [15.9–17.1] |
| **this model** | **8.6 [8.0–9.2]** | **12.0 [11.5–12.5]** |

Word error rate for this model: 27.7 on Common Voice test, 41.2 on Omnilingual test.

An earlier checkpoint ("stage 2", trained last on Common Voice and Omnilingual only) scores the same within noise.
This one had its last gradient steps on all three datasets, with BrahuiSpeech transcripts normalized to Common
Voice spelling, and is the one I used to transcribe broadcast media.

## Use

```python
import torch, librosa
from transformers import Wav2Vec2ForCTC, AutoProcessor

name = "qkhawaja/brahui-asr-ctc1b-s3"
processor = AutoProcessor.from_pretrained(name)
model = Wav2Vec2ForCTC.from_pretrained(name).eval()

audio, _ = librosa.load("clip.wav", sr=16000)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    ids = model(**inputs).logits.argmax(-1)
print(processor.batch_decode(ids)[0])
```

**Split audio into chunks of 20 seconds or less.** Accuracy collapses on longer inputs: on the same audio, CER was
19 at 20-second chunks, 27 at 30 seconds and 68 at 40 seconds.

## Training data

| dataset | hours | licence |
|---|---|---|
| Common Voice Brahui, train + dev | about 10 | CC0 |
| Omnilingual ASR corpus, `brh_Arab` train | about 11 | CC-BY-4.0 |
| BrahuiSpeech-70H-V2 | about 70 | MIT |

Total: 18,888 clips, 79.6 hours. No scraped or privately recorded audio was used to train this model.

Training ran in three stages from the base model: 5,000 steps on everything (learning rate 1e-5), 800 steps on
Common Voice and Omnilingual only, then 1,000 steps on everything with normalized BrahuiSpeech labels (learning
rate 5e-6). The feature encoder was frozen throughout.

## Limitations

- Trained mostly on read speech and broadcast-style speech. Expect higher error on conversational speech, noisy
  recordings, and dialects other than those in the training sets.
- Brahui spelling is not standardized. The model writes in the conventions of Common Voice, so a correct
  transcription can still differ from how a given writer would spell it.
- Short vowels are usually unwritten, as in normal Brahui text.
- No native speaker has systematically reviewed the output. Evaluation is automatic, against existing transcripts.

## Licence and attribution

Released under Apache-2.0, the licence of the base model. The Omnilingual ASR corpus requires attribution
(CC-BY-4.0): Omnilingual ASR, Meta AI. Thanks to the Common Voice contributors and the BrahuiSpeech authors.

Code, evaluation scripts and the full write-up: https://github.com/qasimy123/brahui-digitize

If you have a concern about this model, open a discussion in the Community tab of this repository.
