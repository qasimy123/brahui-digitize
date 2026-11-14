---
language:
- brh
- en
license: cc-by-4.0
task_categories:
- translation
tags:
- brahui
- synthetic
- machine-generated
- low-resource
size_categories:
- 10K<n<100K
configs:
- config_name: retranslated
  data_files: retranslated/train.jsonl
  default: true
- config_name: single_pass
  data_files: single_pass/train.jsonl
- config_name: fineweb
  data_files: fineweb/train.jsonl
- config_name: finepdfs
  data_files: finepdfs/train.jsonl
---

# Brahui–English synthetic parallel data

Real Brahui sentences paired with **machine-written English translations**. Brahui (`brh`) is a Dravidian language
of Balochistan with almost no human-translated parallel text, so this project made its own: a large language model
(Claude) translated existing Brahui text into English.

**The English side is machine output. Use it as training data. Do not use it as reference translations or as a
source of truth about Brahui.**

## Configurations

| config | rows | Brahui side | English side |
|---|---|---|---|
| `retranslated` (default) | 19,417 | transcripts from BrahuiSpeech-70H (15,624), Common Voice (2,207), Omnilingual ASR (1,586) | Claude Opus agents that looked up words in dictionaries and a glossed corpus before translating |
| `single_pass` | 19,417 | the same sentences | one Claude Opus prompt per batch, no lookup |
| `fineweb` | 45,141 | FineWeb-2 `brh_Arab` web text | one-pass Claude Opus translation, filtered |
| `finepdfs` | 7,642 | FinePDFs and GlotCC Brahui text; OCR, visibly noisy | one-pass Claude translation; unsure spans are marked `(?)` |

Fields: `id`, `source`, `brh`, `en`, `model`. The `retranslated` config adds `en_single_pass` (the earlier
translation of the same sentence) and `meaning_agrees` (whether the retranslation judged the earlier one to carry
the same meaning; it did not for 9,063 of the 19,417 rows).

## How good is it

Measured by an LLM judge against human references, as "does the English preserve the meaning of the Brahui":

| check set | single prompt | agents with dictionary and corpus lookup |
|---|---|---|
| 500 sentences of transcribed folktales (spoken register) | 35% | 80% |
| 302 sentences of written text (news-like and everyday) | 70% | 72% (a tie) |

The lookup method helps most on spoken, narrative Brahui, which is what most of the `retranslated` rows are. Expect
roughly one in five of those rows to contain a meaning error, and more in the one-pass configs. The retranslation
deliberately keeps the hesitations and repetitions of speech.

One result worth knowing before you train on this: a 1.3B NLLB model trained on `retranslated` scored the same as
one trained on `single_pass`, on every measure. The better teacher did not produce a measurably better student.

No native speaker has reviewed these translations.

## What is not here

Rows translated from broadcast transcripts, back-translations of DailyDialog (non-commercial licence), and the
dictionary notes the agents wrote while translating (they quote copyrighted dictionaries) are left out.

## Licence and sources

CC-BY-4.0 for the translations. The Brahui text keeps its original terms and needs attribution where they require
it: BrahuiSpeech-70H-V2 (MIT), Common Voice (CC0), Omnilingual ASR corpus (CC-BY-4.0, Meta AI), FineWeb-2 and
FinePDFs (ODC-By 1.0, subject to the Common Crawl terms of use), GlotCC (CC0 packaging, Common Crawl terms).

Method, code and results: https://github.com/qasimy123/brahui-digitize

If you have a concern about this dataset, open a discussion in the Community tab.
