# Data in this repository

## Sealed evaluation sets (`data/eval/`)

| file | source | licence |
|---|---|---|
| `cv_test.jsonl` | Common Voice 27.0 Brahui, official test split (text only; audio from Mozilla Data Collective) | CC0 |
| `omni_test.jsonl` | Omnilingual ASR corpus `brh_Arab` test split (text only) | CC-BY-4.0 |
| `smol_heldout.jsonl` | 200 Google SmolSent en→brh pairs, seeded hold-out | CC-BY-4.0 |
| `bouquet_test.jsonl` | BOUQuET brh_Arab ↔ eng_Latn sentence-level test | CC-BY-4.0 |

Verify with `shasum -a 256 -c SHA256SUMS`.

## Training pairs (`data/parallel/`)

| file | Brahui side | English side | licence |
|---|---|---|---|
| `gold_train.jsonl`, `gold_train_nodev.jsonl` | SMOL + BOUQuET human pairs (train portion) | human | CC-BY-4.0 |
| `synthetic_brh_en.jsonl` | transcripts of BrahuiSpeech-70H (MIT), Common Voice (CC0), Omnilingual (CC-BY-4.0) | Claude single-pass translation (Opus 5.5 for most rows, Fable 5.1 for the rest; see each row's `model` field) | CC-BY-4.0 |
| `retranslated_brh_en.jsonl`, `synthetic_retranslated_brh_en.jsonl` | same rows | Claude Opus 5.5 agents with dictionary and glossed-corpus lookup (80.2% meaning preserved on a 500-sentence folktale check against 35.2% for a single prompt; a tie, 72% against 70%, on 302 written sentences) | CC-BY-4.0 |
| `synthetic_fineweb_brh_en_filtered.jsonl` | FineWeb-2 `brh_Arab` sentences | Claude Opus translation | ODC-By 1.0 (FineWeb-2 terms) |

The English side of the synthetic files is machine output. Treat it as training data, not as reference
translations.

## Deliberately not included

- **Broadcast audio and its transcripts.** No scraped audio is redistributed anywhere. The 3,359 rows translated
  from broadcast transcripts were removed from the pair files above. The machine transcripts themselves are
  published separately, as video IDs, timestamps and text with no audio:
  [`qkhawaja/brahui-media-pseudolabels`](https://huggingface.co/datasets/qkhawaja/brahui-media-pseudolabels).
  `scripts/download_media.py` and `transcribe_media.py` rebuild them from the original videos.
- **DailyDialog-derived back-translations** (CC BY-NC-SA): used for the en→brh pool. `scripts/build_en_everyday.py`
  rebuilds it; Tatoeba (CC BY 2.0) is the redistributable half.
- **Dictionary OCR** (Shad Brahui–English, Sarparra, Shaigle glossaries) and the Ali & Kobayashi (2024) sentence
  list: used as lookup targets by the retranslation teacher and as a dev set. Rights belong to the authors.
  `scripts/ocr_dictionary.py`, `build_shad_lexicon.py`, `parse_ak2024.py` and `build_ak_dev.py` rebuild them from
  the sources. The `hard_words` notes the teacher wrote while translating quote those dictionaries and were
  removed from `retranslated_brh_en.jsonl`.
- **JESUS film (Brahui dub) audio and subtitles**: used for the end-to-end benchmark in `evals/st/jesus_bench.py`.
  Only the script and the scores are here.
- **F5-TTS checkpoints**: a voice-cloning model trained partly on broadcast audio. Not released.
- **Recorded native-speaker audio**: personal recordings used for listening checks only.

## Datasets on the Hugging Face Hub

The pair files above, plus a noisier FinePDFs set, are also published as
[`qkhawaja/brahui-synthetic-parallel`](https://huggingface.co/datasets/qkhawaja/brahui-synthetic-parallel).

## Models

Published on the Hugging Face Hub under `qkhawaja/`; each model card states its licence and training data.
omniASR is Apache-2.0, so the speech recognizers are Apache-2.0. NLLB-200, MMS-TTS and F5-TTS are CC-BY-NC, so the
translation models and the voice are non-commercial.
