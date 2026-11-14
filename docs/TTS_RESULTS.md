# Brahui TTS: results

Dates: 2026-10-22 to 2026-11-14. This document covers the text-to-speech work in order: a VITS fine-tune from Meta's
Urdu MMS voice, an F5-TTS fine-tune on romanized Brahui, cleaning broadcast audio for training, writing the short
vowels the script leaves out, and distilling the result into a small VITS model that runs in a browser.

## Read this first

- **The metric is the project's own ASR.** "CER" below means: synthesize the sentence, transcribe it with the stage-3
  Brahui CTC model (`docs/ASR_RESULTS.md`), and compute character error rate against the input text. It measures
  intelligibility to that model, not naturalness or accent. The ASR writes Arabic script without short vowels, so it
  cannot see vowel errors at all. Real recordings of the same dev sentences score 4.4% on this metric.
- **Human evaluation is one person.** A native speaker in the author's family did two blind A/B listening tests of 24
  sentence pairs each. The second one, which decided the final voice, had 20 pairs with a preference (16 to 4). No
  other listener has rated anything here.
- Twice the CER and the listener disagreed (sections 2 and 4). Differences under about 1 CER point should be read as
  ties.
- **What is released.** The released voice is the distilled VITS model (section 5). The F5-TTS checkpoints are not
  released: they were trained on broadcast audio that is not redistributed, and F5 clones the voice of any reference
  clip, so the checkpoints could reproduce those speakers' voices. The reference recordings used for the final voice
  are private and not released. The VITS models start from MMS checkpoints (CC BY-NC 4.0), so they are non-commercial.

## Headline

| voice | dev set | CER | note |
|---|---|---|---|
| real recordings | 100 sentences | 4.4 | floor for this metric |
| Urdu MMS voice reading Brahui | 40 sentences (VITS dev) | 20.7 | baseline |
| VITS fine-tuned from MMS Urdu, best | 40 sentences (VITS dev) | 21.4 | tie with the baseline |
| F5-TTS, first run (`multi`) | 100 sentences | 9.1 | speed 0.75 |
| F5-TTS, news capped + media (`multi2`) | 100 sentences | 8.5 | lost the first blind test 3 to 7 |
| F5-TTS, vowels as heard (`multi2_vow`) | 100 sentences | 9.8 | won the second blind test 16 to 4; teacher |
| distilled VITS student (released) | 100 student dev sentences | 9.7 | 114 MB ONNX, runs in the browser |

The VITS, F5 and student dev sets are different sentences, so compare within a block, not across.

## 1. VITS fine-tune from MMS Urdu (2026-10-22)

Full write-up: [`docs/PLAN_TTS.md`](PLAN_TTS.md). One Common Voice speaker, 987 clips / 1.64 h after an ASR-agreement
filter, fine-tuned from `facebook/mms-tts-urd-script_arabic` with the `finetune-hf-vits` recipe. CER on its 40 dev
sentences: real recordings 3.4, Urdu voice reading Brahui 20.7, lr 2e-5 for 80 or 200 epochs 28.1 and 28.3,
lr 1e-4 for 100 epochs 23.2, lr 1e-4 for 300 epochs 21.4. More epochs did nothing at the low learning rate; the
higher one helped, but the best run only tied the untuned Urdu voice. 1.6 h of one speaker was not enough for this
recipe, which led to F5-TTS and a larger multi-speaker set. Artefacts: `evals/tts/urd_baseline/`,
`evals/tts/brahui-tts-vits*/`.

## 2. F5-TTS on romanized Brahui (2026-11-10 to 11-12)

### Training set

F5 needs clean audio from many speakers. Every candidate clip was scored with DNSMOS (`scripts/score_tts_pool.py mos`)
and with ASR agreement against its label (`asr`). BrahuiSpeech, the largest corpus, is mostly too noisy: 15,626 clips /
69.5 h, median DNSMOS overall 2.24, only 3.3 h at 3.0 or above. A 40-clip enhancement pilot on the 2.0 to 2.8 band
(`evals/tts/enhance_pilot/`) moved the mean from 2.48 to 2.71 with SepFormer-DNS4 (10% of clips reach 3.0) and to 2.53
with MetricGAN+ (which lowers the signal score), so BrahuiSpeech stayed a small part of the set.

The first set (`data/tts/multi/report.json`, built by `scripts/build_tts_multispeaker.py`): DNSMOS >= 3.0, ASR
confidence >= 0.9 for pseudo-labelled media, label CER <= 0.10 (0.20 for spontaneous speech). 8,831 clips / 35.9 h,
687 speaker clusters, split 8,731 train / 100 dev. By source: news bulletins from one anchor 22.5 h, Omnilingual 7.2,
BrahuiSpeech 2.5, Common Voice 2.2, other broadcast 1.6.

### First run (`multi`)

F5-TTS v1 Base, vocabulary extended for Brahui (`tools/f5/prep_finetune.py`), text romanized with
`brahui.text.romanize`. lr 5e-5, 6,400 frames per batch, 33 epochs (about 70k updates), one RTX 4090, about $6
including false starts. Curve on 20 dev sentences (16 sampling steps):

| update | 15k | 20k | 25k | 30k | 35k | 40k | 45k | 50k | 55k | 60k | 65k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CER | 13.7 | 13.7 | 20.9 | 12.3 | 10.7 | 10.0 | 11.1 | 10.2 | 9.3 | 7.7 | 8.2 |

Full dev (100 sentences, Common Voice reference clip): 10.0 at default speed, **9.1 at speed 0.75**. Default speed
comes out at 0.71 of the real duration, 0.75 gives 0.94, so 0.75 is the setting used from here on. The 20-sentence
curve reads about 2 points lower than the full dev for the same weights; it is a trend indicator only.

Two earlier attempts at lr 1e-5 were stopped at 10k updates with curve readings of 46.2 (romanized) and 79.1
(Arabic script) on the same 20 sentences; the untrained base model reads 49.8 on romanized text. The curve was
scoring EMA weights at the time, which was a false signal (see pitfalls): the Arabic-script checkpoint's live
weights scored 32.6. Arabic script was not retried at the higher learning rate, so it is untested, not ruled out.

### Second run (`multi2`): news capped, media added

By ear the first voice carried the news anchor's accent; news was 63% of the hours. The second
set (`scripts/build_tts_multi2.py`, `data/tts/multi2/report.json`) caps news at 5.0 h and adds 27.9 h of enhanced
broadcast audio from section 3: 11,388 clips / 46.2 h. Training continued from the first run's final weights, 17 epochs
(about 47k updates), RTX 4090, about $3.40. The romanizer's "speak" mode was used (punctuation separated from words,
س + ہ no longer collapsing into "sh"); on the old weights that mode scored 10.6 against 11.5 on 21 sentences
(`evals/tts/f5_romfix_*`).

On the 100 dev sentences `multi2` scored 9.6 at default speed and **8.5 at speed 0.75**, against 10.0 and 9.1 for
`multi`.

**Blind test 1** (24 pairs, `multi` against `multi2`, summary under `evals/tts/listen/set30/`): old better 7, new
better 3, same 8, both bad 6. The 0.6-point CER gain did not carry to a native ear, so `multi2` was not adopted. The
listener's notes pointed at the text front end: wrong word beginnings and wrong short vowels, because romanized text
without vowels is a consonant skeleton ("makhafng", "hmo"). The listening sentences were also fragments cut from
running text, several of them not understandable in isolation, so the set was rebuilt from complete sentences for
the next test. Artefacts: `evals/tts/f5_multi_roman*/`, `evals/tts/f5_multi2_roman*/`.

## 3. Broadcast audio: pilot, bulk scrape, enhancement (2026-11-11)

### Pilot: enhancement against music separation

20 YouTube sources (dramas, lessons, stories, talk, sermons, poetry), 30 chunks each, 600 chunks / 2.4 h. Each chunk
was scored raw, after Demucs vocal separation, after DeepFilterNet speech enhancement, and after both
(`scripts/media_pilot_quality.py`, `evals/media_pilot/`).

| variant | DNSMOS >= 3.0 | also ASR confidence >= 0.9 | transcript change against raw (CER) |
|---|---|---|---|
| raw | 18% | 15% | |
| Demucs | 26% | 21% | 6% |
| DeepFilterNet | 46% | 37% | 19% |
| Demucs + DeepFilterNet | 47% | 37% | 13% |

Speech enhancement does almost all of the work. Music separation only matters where there is a music bed: one story
channel went from 31% to 96% with Demucs alone, and to 100% with DeepFilterNet alone. Usable share after
DeepFilterNet by source ranged from 100% and 91% (two story channels) through 51 to 78% (lessons, talk, mixed
programming) down to 11 to 24% for dramas and under 10% for sermons, poetry and three other channels. Caveat:
enhancement changes the ASR transcript by 6 to 19% of characters on average, and the enhanced audio was only
spot-checked by ear for artefacts.

### Bulk scrape and enhancement

The eight best sources were downloaded in full: 951 files, about 108 h of speech. Transcription with the stage-3 ASR
ran on one rented GPU (about $1), DeepFilterNet on a 32-core CPU machine (about $2) with `scripts/enhance_media_bulk.py`.
Chunks with confidence >= 0.8 and length 2 to 20 s: 23,857 chunks / 88.8 h.

TTS-grade means DNSMOS >= 3.0 and ASR confidence >= 0.9. Hours by source:

| source | enhanced set | TTS-grade raw | TTS-grade enhanced | after language screen |
|---|---|---|---|---|
| Brahvi Tawar Media | 27.7 | 3.4 | 12.2 | 8.7 |
| Kirthar Range | 16.7 | 2.2 | 7.8 | 5.8 |
| Mehr TV | 12.9 | 0.7 | 4.4 | 3.8 |
| Al Quran in Brahui | 11.6 | 2.5 | 3.4 | 2.6 |
| Balochistan Talent | 9.0 | 2.0 | 4.2 | 2.3 |
| Brahui Garrok | 4.3 | 0.9 | 1.7 | 1.5 |
| Brahvi channel | 3.7 | 2.0 | 2.3 | 0.6 |
| Torch of Hope | 3.0 | 0.7 | 2.7 | 2.6 |
| **total** | **88.8** | **14.4** | **38.7** | **27.9** |

Enhancement raised the TTS-grade pool from 14.4 h to 38.7 h. A language screen then removed Urdu, Arabic recitation
and repeated lesson drills, which the Brahui ASR transcribes with high confidence: a clip is dropped when its
transcript has more Urdu function words than Brahui ones, or when a Brahui character language model scores it in
the bottom 5% (`scripts/lid_media.py`). The final manifest is 6,808 clips / **27.9 h**. The audio is not
redistributed.

## 4. Vowelization (2026-11-11 to 11-13)

Brahui in Perso-Arabic script leaves short vowels unwritten. The steps below try to recover them without a human
transcriber.

**Rule and lookup vowelizer** (`brahui/vowelize.py`). Arabic-script words and vowelled Latin words are reduced to a
shared key (consonants plus long-vowel positions) and looked up in vowelled sources: the Shad dictionary, a word
list, and the Ali & Kobayashi 2024 texts and glossary. Unknown inflected forms are split into known stem plus known
suffix; anything else falls back to the skeleton. Feeding vowelled text to the existing F5 model made CER worse than
its 6.4 on the same 24 sentences, because that model had learned skeleton spelling. Vowelled input can only be
judged after retraining on it.

**Acoustic check** (`scripts/vowel_acoustic.py`, `evals/tts/vowel_review/acoustic_check.json`). A zero-shot phone
recognizer (xlsr-53 espeak) scored candidate spellings inside force-aligned word spans: right for 57 of 70
high-confidence words by majority (81%), 411 of 559 recordings (74%). Formants and vowel duration were weaker (66%
resubstitution; duration AUC 0.77 word-medial, 0.46 word-final). A hint for a review page, not an automatic judge.

**Vowel-aware ASR** (`scripts/build_vowel_asr.py`, `scripts/vowel_asr.py`). The stage-3 CTC model was fine-tuned with
a 34-symbol vowelled Latin output on 11,379 utterances / 46.2 h, in two folds. In each fold the trusted spellings of
half the word types are hidden during training (their frames are masked out of the loss), so the model's choice for
those words is blind. Candidates for each word are then scored in context on up to 48 recordings. Two L40S
fine-tunes, about $4.60.

| blind check against the dictionary spelling | types | occurrences |
|---|---|---|
| both folds (7,660 hidden types; single folds 56.5% and 59.4% of types) | 58.0% | 74.9% |
| types with >= 8 recordings, ignoring vowel length and word-final vowel differences | 80.1% | 88.4% |

Agreement rises with evidence (78.4% of types with 48 or more recordings) and with source quality (82.4% for exact
dictionary entries, 66.1% for key matches). Among the 20% of verdicts with the largest margin, 96.3% agree with the
dictionary; among the top 40%, 91.9%. Many confident disagreements are the model correcting a wrong lookup, so the
dictionary is not clean gold and these figures understate the model. It also makes real errors and tends to drop
reduced vowels. Artefacts: `evals/tts/vowel_asr/`.

**Retranscription** (`scripts/vowel_retranscribe.py`, `evals/tts/vowel_asr/retrans_report.json`). The training text
was rewritten with vowels as heard, under guards (no consonant changes and no dropped vowels on dictionary words).
Of 396,975 tokens, 21.6% changed from the rule vowelizer's spelling. By origin: dictionary kept 66.5%, heard 27.5%,
guess kept 5.7%, fixed forms 0.3%.

**Text vowelizer for new text.** A character-level encoder-decoder with a lexicon hint (`brahui/vowelize_nn.py`)
reached 90.6% exact words on held-out utterances; plain lookup of each word's most-heard spelling reached 91.3%.
Context did not help, so lookup is the front end in use (`HeardVowelizer`).

**F5 retrained on vowels as heard** (`multi2_vow`, `scripts/gpu/f5_vow.sh`). Same audio as `multi2`, continued from
the first run's final weights, 17 epochs, RTX 4090, about $2.70.

CER at speed 0.75: 9.8 on the 100 dev sentences (`multi`: 9.1) and 12.3 on a new listening set of 24 complete
sentences (`multi`: 11.3). Flat to slightly worse, which says nothing about vowels because the scoring ASR does not
write them.

**Blind test 2** (`evals/tts/listen_v3/review_2026-11-13.json`, 24 pairs, complete sentences with human English):
new better 16, old better 4, same 2, both bad 2. The new voice won whether it was played first (8 to 3) or second
(8 to 1); a two-sided sign test on the 20 decided pairs gives p of about 0.012. One listener, so a clear preference
and not a population estimate. `multi2_vow` became the teacher for section 5. Remaining complaints: loanwords
spelled wrongly by lookup, a few words left without vowels, and prosody (rising pitch at word ends, pauses). A
loanword list and a fallback for unvowelled words were added afterwards and have not been tested by a listener.

## 5. Distillation to a browser VITS (2026-11-14)

F5 is too large and slow for a browser, so the vowelled F5 voice was distilled into a VITS model.

- **Voice.** The teacher spoke in the voice of a short reference clip from the reference speaker (private, not
  released), cleaned with DeepFilterNet. A 300-clip test of two reference lengths gave 10.3% read-back CER for the
  longer clip and 9.4% for a shorter clip with trailing silence; the shorter one was used. The longer one sometimes
  leaked its last words into the output.
- **Script.** `scripts/build_tts_distill_text.py`: 17,257 sentences (3,878 from transcripts, 7,588 written, 5,392
  everyday dialogue, 399 from the gold test sets), vowelled by the front end above.
- **Teacher audio.** 17,257 clips / 27.0 h synthesized on one RTX 4090, mean read-back CER 9.6%
  (`scripts/tts_distill_filter.py`). Kept at CER <= 0.15: 14,426 clips / 22.9 h, split 14,326 train / 100 dev
  (`scripts/build_tts_student_set.py`).
- **Student.** MMS Urdu VITS with its vocabulary replaced by vowelled Latin letters (`tools/vits/latin_init.py`),
  lr 1e-4, batch 32, fp32, 16 kHz, trained in segments of 15 epochs with a CER check after each
  (`scripts/gpu/train_tts_student.sh`). Total cost about $6.

| segment (40 dev sentences) | init | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| CER | 41.6 | 10.9 | 12.2 | 10.7 | 10.9 | 10.8 | 11.6 |

Training stopped early after segment 6; segment 3 is the kept model. On the full 100 dev sentences it scores **9.7**.
Its speech is about 11% shorter than the teacher's for the same sentences. The curve is flat after the first
segment, so most of what this recipe can learn arrived in 15 epochs.

ONNX exports on 40 dev sentences: fp32 114 MB, CER 10.0; int8 on all operators 38 MB, 11.2; int8 on MatMul only
112 MB, 11.8.

The fp32 export is the one shipped: in Chrome (WASM) it ran at roughly three times real time on a Mac laptop, while
the 38 MB int8 model ran at about real time, so quantization bought size and lost both quality and speed.
`demo/web/tts.html` takes Brahui text, vowelizes it with a 65,850-word lexicon exported by
`scripts/export_vowel_lexicon_js.py` (unknown words fall back to the skeleton), and speaks it locally.

Not done: the blind test of student against teacher. The page exists (`evals/tts/student/blind_student_vs_teacher.html`)
but no verdict has been recorded, so "the student matches the teacher" rests on CER alone (9.7 against the teacher's
9.8 on a different 100 sentences), which sections 2 and 4 show is not enough. Artefacts: `evals/tts/student/`.

## 6. What did not work, and pitfalls

- **ASR-CER does not predict a native listener's preference.** A 0.6-point gain lost the blind test 3 to 7; a
  0.7-point loss won 16 to 4. Use CER to catch broken models, not to rank close ones.
- **More and cleaner audio** (`multi2`, +28 h) did not improve the voice by ear. The limit was the text front end.
- **Cycle consistency as a label filter** (`scripts/cycle_consistency.py`, `evals/cycle/`): re-speaking the ASR
  transcript with F5 and comparing it to the original audio separated 30 wrong from 27 right transcripts with AUC
  0.68; ASR confidence alone gave 0.95. The true label's re-speak was closer in only 27% of the wrong clips.
- **Language ID.** MMS-LID mislabelled real Brahui and also missed Urdu; transcript function words plus a character
  LM worked better. Before the screen existed, Urdu clips reached a listening set.
- **EMA weights are useless early in an F5 fine-tune.** The trainer restores the pretrained EMA step count, so the
  EMA barely moves: 45.2 at 10k updates (base model 49.8) while live weights reached 13.7 by 15k. Evaluate and
  export the live weights (`tools/f5/synth_eval.py --raw`). lr 1e-5 was also too low; 5e-5 worked.
- **F5 reference clips longer than 12 s** are cut without cutting their text. Keep references at 4 to 11 s.
- **A CTC "wildcard" target for hidden words** in the vowel-aware ASR is wrong: the loss goes negative and the model
  learns to emit blanks. Mask the frames instead.
- **Curve evaluation beside a full training batch** has no GPU memory left, and on CPU it starved the data loaders.
  Export live weights and score them on another machine, or leave headroom.
- **Parallel CPU workers and formats.** Thirty workers fetching one model file corrupted it; one onnxruntime thread
  pool per worker stalled a 32-core machine. `datasets` audiofolder rejected `metadata.csv` (use `metadata.jsonl`);
  a double quote in text merges rows in F5's CSV reader.

## 7. Reproduce

Each script has a usage header. Local ones run with `uv run python <script>`; `scripts/gpu/*.sh` expect their data
synced to a GPU machine first.

| step | scripts |
|---|---|
| VITS from MMS Urdu | `scripts/build_tts_set.py` (speaker set), `scripts/gpu/train_tts.sh` (fine-tune), `scripts/tts_eval.py` (dev CER) |
| score candidate audio | `scripts/build_tts_pool_extra.py`, `scripts/build_tts_pool_news.py` (pools), `scripts/score_tts_pool.py` (DNSMOS and ASR agreement), `scripts/enhance_tts_pilot.py` (enhancement pilot) |
| F5 training sets | `scripts/build_tts_multispeaker.py` (`multi`), `scripts/build_tts_multi2.py` (`multi2`), `scripts/tts_f5_export.py` and `scripts/tts_f5_export_vowel.py` (F5 format, skeleton or vowelled text) |
| F5 training and scoring | `scripts/gpu/train_f5.sh` (fine-tune with CER curve; `INIT=` continues from earlier weights), `scripts/gpu/f5_vow.sh` (vowelled retrain), `tools/f5/*.py` (vocabulary extension, warm start, synthesis, live-weight export), `scripts/tts_score_wavs.py` (CER of a wav directory) |
| broadcast audio | `scripts/media_pilot_quality.py` (pilot comparison), `scripts/download_media.py`, `scripts/transcribe_media.py`, `scripts/enhance_media_bulk.py` (DeepFilterNet + DNSMOS), `scripts/lid_media.py` (language screen) |
| vowelizer | `brahui/vowelize.py` (rule/lookup), `brahui/vowelize_nn.py` (heard-spelling lookup, letter model), `scripts/train_vowelizer.py`, `scripts/vowel_acoustic.py` (acoustic check) |
| vowel-aware ASR | `scripts/build_vowel_asr.py` (data), `scripts/vowel_align.py` (word spans), `scripts/vowel_asr.py` (train / score / dump), `scripts/gpu/vowel_asr.sh` (both folds), `scripts/vowel_asr_report.py` (breakdown), `scripts/vowel_retranscribe.py` (vowels as heard) |
| native-speaker pages | `scripts/make_vowel_review.py`, `scripts/make_tts_drill.py`, `scripts/build_listen_v3.py`, `scripts/make_blind_review_v3.py`, `scripts/make_blind_review_student.py` |
| distillation | `scripts/build_tts_distill_text.py` (sentences), `tools/f5/synth_batch.py` (teacher audio), `scripts/tts_distill_filter.py` (read-back CER), `scripts/build_tts_student_set.py`, `tools/vits/latin_init.py` (student init), `scripts/gpu/train_tts_student.sh`, `scripts/tts_student_eval.py`, `scripts/export_vowel_lexicon_js.py` (browser lexicon) |
| negative result | `scripts/cycle_consistency.py` |

Browser demo: `python3 demo/web/serve.py 8097`, then open `tts.html`.
