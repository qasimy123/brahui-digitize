---
language:
- brh
license: cc-by-nc-4.0
library_name: transformers.js
pipeline_tag: text-to-speech
base_model: facebook/mms-tts-urd-script_arabic
tags:
- brahui
- vits
- onnx
- distillation
- low-resource
---

# Brahui text-to-speech for the browser (distilled VITS, ONNX)

A single-voice speech synthesizer for Brahui (`brh`), a Dravidian language of Balochistan. It is 114 MB and
speaks about three times faster than real time in Chrome on a laptop, with no server.

This is a research snapshot from a one-person project, not a supported product.

## About the voice

The voice is modelled on a real person: a native Brahui speaker in the author's family, who agreed to this
release. She is not named, and her recordings are not published.

**Do not use this model to impersonate anyone, to deceive listeners about who is speaking, or to put words in a
real person's mouth.** Say that speech is synthetic wherever a listener could take it for a recording.

## Results

There is no standard test for Brahui speech synthesis. The score here is a read-back test: synthesize a sentence,
transcribe it with the project's Brahui speech recognizer, and count character errors against the input. It
measures intelligibility to that recognizer, not naturalness.

| voice | read-back character error rate |
|---|---|
| real recordings | 4.4 |
| Meta's Urdu MMS voice reading Brahui | 20.7 |
| the larger F5-TTS teacher this model learned from | 9.8 |
| **this model** | **9.7** |

The rows use different test sentences, so read them as rough levels. One native speaker preferred the teacher
voice to an earlier version in a blind test (16 to 4 over 20 decided pairs). Nobody has yet compared this smaller
model with its teacher by ear.

## Input

The model reads **vowelled Latin-script Brahui**, not the Perso-Arabic script Brahui is normally written in. The
Arabic script leaves short vowels unwritten, and a voice that has to guess them mispronounces words, so the vowels
are restored before synthesis. The vocabulary is the lowercase Latin letters, a few letters with diacritics for
long vowels and retroflex sounds, and basic punctuation (`vocab.json`).

Converting ordinary Brahui text to that form is done by the project's front end (`brahui/vowelize.py`, and the
word table the browser demo loads). The demo at `demo/web/tts.html` in the project repository is the working
reference.

## Files

- `onnx/model.onnx`: full precision, 114 MB. Use this one.
- `onnx/model_quantized.onnx`: int8, 38 MB. Smaller, but slower in the browser and less accurate (11.2 against
  10.0 on the same 40 sentences).

- `vowel_lex.json`: 65,834 Brahui words in Arabic script mapped to the vowelled Latin spelling the model expects.
  About 40,000 spellings are the ones most often heard in recordings by a vowel-aware speech recognizer; the
  rest come from a small spelling model trained on those and a hand list of loanwords. No entry is copied from a
  published dictionary. Words not in the table fall back to an unvowelled spelling.

Output is 16 kHz mono audio.

## How it was made

1. An F5-TTS model was fine-tuned on about 46 hours of Brahui speech from open datasets and public broadcasts,
   with the text written in vowelled Latin script as heard in the audio.
2. That model read 17,257 Brahui sentences aloud in the reference speaker's voice. Clips the speech recognizer
   could not read back accurately were dropped, leaving 22.9 hours.
3. A VITS model, started from Meta's Urdu MMS voice, was trained on those 22.9 hours of synthetic speech. This is
   the model here. It cost about $6 of GPU time.

The F5-TTS teacher is not released. It can copy any voice from a short clip and was trained on broadcast audio
that this project does not redistribute. This model only ever heard the teacher speaking in one voice.

## Limitations

- Unknown words and names fall back to a consonant skeleton and are often mispronounced.
- Vowel qualities follow the project's sources, which lean toward some dialects more than others.
- Speech is about 11% faster than the teacher's, and prosody is flat on long sentences.
- Evaluated by one native speaker and an automatic read-back test.

## Licence

CC-BY-NC-4.0, inherited from the MMS base model. Non-commercial use only, and subject to the no-impersonation
terms above.

Code, evaluation scripts and the full write-up: https://github.com/qasimy123/brahui-digitize

If you have a concern about this model, including if you are a speaker in audio it learned from, open a
discussion in the Community tab of this repository.
