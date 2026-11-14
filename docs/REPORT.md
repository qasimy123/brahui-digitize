# Speech and translation for Brahui with almost no human-labelled data: what worked and what didn't

Qasim Khawaja, November 2026. Code: https://github.com/qasimy123/brahui-digitize

## Summary

Brahui is a Dravidian language of Balochistan with about 2.5 million speakers, written in Perso-Arabic script. Over
about six weeks (6 October to 14 November 2026) I built speech recognition, Brahui↔English translation and speech
synthesis for it, working alone, with no paid annotators. The supervision came from three places: the small open
datasets that already existed, a large language model (LLM) used as a translation teacher, and the models' own
output fed back as training data. The smallest version of each model runs locally in a web browser.

What was measured:

| component | result | measured on |
|---|---|---|
| Speech recognition, 1B parameters | character error rate (CER) 8.6% [8.0, 9.2] read speech; 12.0% [11.5, 12.5] spontaneous speech | Common Voice test, 882 clips; Omnilingual test, 106 segments |
| Speech recognition, 300M, in browser | CER 8.0% with a character language model (9.9% without); 0.21 s for a 3-second sentence | Common Voice test; Chrome on an Apple M1, WebGPU |
| LLM teacher, Brahui→English | 80% meaning preserved (single prompt: 35%); 72% on written text (single prompt: 70%, a tie) | 500 folktale sentences; 302 written sentences |
| Trained translator, NLLB 1.3B | chrF++ 50.2 Brahui→English, 38.2 English→Brahui; 41% meaning preserved Brahui→English | BOUQuET test, 854 sentences; 302-sentence development set |
| Browser translator, Brahui→English | chrF++ 51.2, equal to its teacher; 0.88 s per 20-word sentence | development set; Chrome on an M1 |
| Speech synthesis, browser voice | read-back CER 9.7% (real recordings: 4.4%); 114 MB | 100 sentences |

Terms: CER is the percentage of characters that must be inserted, deleted or substituted to turn the output into the
reference. chrF++ is a 0 to 100 overlap score between a translation and a human reference, computed on character
and word n-grams. "Meaning preserved" (MP) is the share of sentences for which an LLM judge, comparing the output
with the human English reference, finds no added, dropped or changed fact. Brackets are 95% bootstrap confidence
intervals.

For scale on the 41%: on the same 302 sentences and the same strict judge, a few-shot LLM scores 59% and the
lookup-agent teacher 72%. My own target for the project was 95% with no human annotators, and it was not met.

Three limitations apply to everything below. Human validation is one native speaker, a member of my family, who did
two blind listening tests and one informal review; nothing else was checked by a person who speaks Brahui. All other
evaluation is automatic metrics and LLM judges, and the judge is the same model family as the translation
teacher. This is a snapshot of finished work, with no promise of support.

## Why Brahui, and what existed before

Brahui had no machine translation system I could find and was not in NLLB-200, FLORES-200 or Google Translate. What
did exist in October 2026 made the project possible, and all of it is other people's work:

- **Speech with transcripts, about 90 hours.** Mozilla Common Voice (9.9 h, CC0), Meta's Omnilingual ASR corpus
  (about 11 h of spontaneous speech, CC-BY-4.0), and BrahuiSpeech-70H-V2 (69.5 h of broadcast-style speech, MIT),
  whose author also published a Whisper-Small fine-tune that served as my baseline.
- **Human translations, about 2,000 sentence pairs.** Google's SMOL and Meta's BOUQuET.
- **Text without translations.** FineWeb-2 and FinePDFs Brahui slices, GlotCC.
- **Scholarship.** Ali and Kobayashi's *Brahui Texts* (2024), with morpheme-by-morpheme glosses and English for
  several thousand sentences of stories and folktales; Shad's Brahui–English dictionary (2021); Sarparra's
  English–Brahui dictionary (2022).
- **Base models.** Meta's omniASR (speech recognition, already trained on a little Brahui), NLLB-200 (translation),
  MMS Urdu VITS and F5-TTS (speech synthesis).

Several other Brahui projects appeared in the same months. I have not compared against them.

## The method

**An LLM as teacher.** In the first week I found that Claude Opus 5.5, given 15 example pairs, translated Brahui
into English at chrF++ 57.2 on the BOUQuET test set. It was much weaker in the other direction (34.9). Brahui shares
its script and a large stock of loanwords with Urdu, Persian, Arabic and Balochi, and the model appears to read it
through those. A model that can read Brahui turns Brahui text that has no translation into parallel data: I had it
translate about 23,000 transcript sentences and 45,000 web sentences into English. The Brahui side of every such
pair is real; only the English is machine-written.

**Students and back-translation.** A smaller translation model (the student) trained on those pairs learns to write
Brahui from real Brahui targets, so the teacher's weakness at writing Brahui does not reach it. The student then
translated 130,000 everyday English sentences into Brahui. Those pairs have real English on the target side and
were used to train the Brahui→English direction only. This is back-translation, a standard technique.

**Lookup agents.** A single prompt cannot hold a dictionary and a glossed corpus. The stronger teacher was a set of
Opus agents that, for each batch of 20 to 30 sentences with a few sentences of preceding context, searched the
dictionaries and the glossed sentences for every uncertain word stem before translating. The Shad dictionary had to
be digitized first, with the same model reading page images.

**Pseudo-labels.** A pseudo-label is a model's own output used as a training label. The speech recognizer
transcribed public broadcast audio, and those transcripts trained the smaller recognizer, the character language
model and the speech synthesizer.

**The decision instrument.** Four test sets were sealed at the start and never trained on. Model choices were made
on a separate development set and only confirmed on the sealed sets. Every comparison used a paired bootstrap (the
two systems are resampled on the same sentences, which gives a confidence interval for their difference), and I
measured the noise first: two training runs differing only in random seed differ by about ±0.3 chrF++ on 854
sentences, and two runs of the same LLM prompt flip the judge's verdict on 32 of 302 sentences. A change was kept
only if it cleared that noise on both the development and the sealed set. Most of the negative results below exist
because of this rule.

**How this was built.** The code and experiments were written and run with Claude Code, an AI coding agent, under my
direction. Claude models were also the translation teacher and the MP judge. An LLM judging models taught by the
same LLM is circular, and I did not remove that risk. What limits it: the references are human-written (BOUQuET,
SMOL, the linguists' English in *Brahui Texts*, film subtitles); the judge sees only two English texts and never the
Brahui; a second judge model agreed with the first on 282 of 302 verdicts; chrF++ needs no judge; and decisions
rested on paired differences. One step adds risk. Many human references were found to be unfaithful to the Brahui,
and the "cleaned" references were corrected by two Claude models. They worked independently, never saw a system
output, and a correction was accepted only when both flagged the sentence and agreed. Scores on cleaned references
are 3 to 9 points higher than on the originals.

## Results

### Speech recognition

The recognizer is Meta's omniASR CTC-1B, fully fine-tuned on 79.6 hours (18,888 clips) of the three open datasets.
CTC (connectionist temporal classification) models emit one character prediction per audio frame in a single pass.

| model | Common Voice test CER | Omnilingual test CER |
|---|---|---|
| Whisper-Small Brahui (community model) | 18.9 [18.3, 19.8] | 19.1 [18.3, 20.0] |
| omniASR CTC-1B, not fine-tuned | 19.6 [18.9, 20.4] | 16.4 [15.9, 17.1] |
| fine-tuned, stage 1 (all 80 h) | 9.3 [8.8, 10.0] | 12.5 [12.1, 13.0] |
| stage 2 (800 more steps on Common Voice and Omnilingual only) | 8.6 [8.0, 9.2] | 11.8 [11.3, 12.3] |
| stage 3 (released; see negative results) | 8.6 [8.0, 9.2] | 12.0 [11.5, 12.5] |

These are the lowest error rates among the systems I ran on these two test sets. Meta reports CER 8.2 for its much
larger 7B model on its own Brahui evaluation; I did not run that model. Word error rate is still 27.7% on read
speech, because the remaining errors are mostly spelling variants of the same sound and word-boundary spaces, in a
language whose spelling is not standardized. Accuracy collapses on inputs longer than 20 seconds (CER 19, 27 and 68
at 20, 30 and 40 second chunks of the same audio), so audio must be chunked.

One test did not depend on Brahui transcripts. A feature film dubbed into Brahui has independent human English
subtitles. Recognizing the dub and translating it with an LLM scored chrF++ 43.9 against those subtitles over 156
windows of about 40 seconds; translating the film's human Urdu subtitles with the same LLM scored 47.4. Recognition
errors therefore cost about 3.5 points. Scoring each output against the neighbouring window's reference gives about
21, so the scores reflect translation of the audio and not recall of the film.

### Translation

**Teacher quality.**

| teacher | folktales (500 sentences) | written text (302 sentences) |
|---|---|---|
| Opus, 40 examples in the prompt | not measured | 58.9% |
| Opus, all 1,690 human pairs in the prompt, highest reasoning effort | 35.2% | 70.5% (rerun: 68.5%) |
| Opus agents with dictionary and glossed-corpus lookup | 80.2% | 71.9% |

MP on cleaned references. On folktales the agents beat the prompt on 236 sentences and lost on 11. On written text
the two are tied. The folktale gain needs a caveat: the dictionaries and glossed sentences the agents searched come
partly from the same book as the test sentences (test sentences excluded), so 80% describes a register where good
lookup material exists. Analysis of the remaining failures blamed the human reference, not the model, for a
large share; I report that as the analysis's finding, not as a correction to the scores.

**Student quality.** All students are NLLB-200 fine-tunes, with Brahui using NLLB's Urdu language code.

| model | English→Brahui chrF++ (BOUQuET test) | Brahui→English chrF++ (BOUQuET test) | Brahui→English MP (302 dev, cleaned) |
|---|---|---|---|
| NLLB 600M, no fine-tuning | 20.7 (SMOL held-out, 200) | 21.0 | not measured |
| round 0: 600M on about 70,000 pairs, nearly all teacher-made | 36.7 | 46.8 | not measured |
| round 2: + 130,000 back-translated pairs | 37.1 | 48.2 | not measured |
| round 4: same data, NLLB 1.3B | 38.2 | 49.8 | 43.4% |
| round 5: 1.3B, retranslated targets (released) | 38.2 | 50.2 | 41.4% |
| Opus, 15 examples | 34.9 (SMOL held-out) | 57.2 | 58.9% (40 examples) |

Round 2 and round 4 each beat the previous round with paired intervals that exclude zero on development and sealed
sets (Brahui→English +1.4 [+0.8, +2.1] and +1.6 [+0.9, +2.3] on the sealed set). Round 5 is a tie with round 4.

For English→Brahui the student passed its teacher on chrF++: 38.3 [36.5, 39.8] on the 200-sentence SMOL held-out
set against 34.9 for Opus. An LLM judge rated 72 to 78% of the 600M students' English→Brahui outputs as meaning
preserving, but that judge has to read Brahui, so it is a weaker instrument than the English-only judge.

**The gap.** The teacher reaches 70 to 80% MP and the best student 41 to 43%. In an analysis of the 600M student's
150 worst development sentences, 80 had a mistranslated content word, and the words were everyday ones (cucumber,
homework, spider) that are rare in a corpus dominated by news, religion and politics. On the film benchmark the
offline cascade scores chrF++ 41.2 with round 4, against 43.9 with an LLM translating.

**Inside the model.** Comparing the base and fine-tuned 1.3B models on 504 sentences in Brahui, Urdu and English:
fine-tuning moved Brahui sentence representations toward Urdu's (the distance between their centroids fell from
about 0.7 to 0.15 of the within-language spread from encoder layer 4 up). Urdu→English improved (chrF++ 58.4 to
63.6, 120 sentences), while asking for Urdu output now produces Brahui. The model cannot be used to write Urdu.

### Speech synthesis

All synthesis scores are read-back CER: synthesize a sentence, transcribe it with the stage-3 recognizer, and count
character errors against the input. This measures intelligibility to that recognizer only. Real recordings score
4.4%.

**Voice.** Fine-tuning Meta's Urdu VITS voice on 1.6 hours of one Common Voice speaker failed (negative results).
Fine-tuning F5-TTS, a voice-cloning model, on 35.9 hours from 687 speaker clusters, with text converted to Latin
letters, reached 9.1% on 100 sentences.

**Broadcast audio.** To get more speakers I transcribed public broadcasts and cleaned them. On a 600-chunk pilot,
18% of raw chunks met a quality threshold (DNSMOS, an automatic speech-quality score, of 3.0 or more), against 46%
after speech enhancement with DeepFilterNet. In bulk, 88.8 hours of confident chunks yielded 38.7 hours at
synthesis grade after enhancement (14.4 before), and 27.9 hours after a screen that removed Urdu, Arabic recitation
and repeated lesson drills.

**Vowels.** Brahui script omits short vowels, and the listener's main complaint about the first voices was wrong
vowels. Without a human transcriber, I fine-tuned the recognizer to write vowelled Latin text, hiding the dictionary
spelling of half the word types during training so that its choice for those words was blind. For hidden word types
with at least 8 recordings, its choice agreed with the dictionary for 80.1% of types, ignoring vowel length and
word-final vowels. The training text was rewritten with vowels as heard (21.6% of 396,975 tokens changed) and the
voice retrained. Read-back CER did not improve (9.8% against 9.1%). In a blind test of 24 sentence pairs, the native
speaker preferred the new voice in 16, the old in 4, heard no difference in 2 and rejected both in 2 (two-sided sign
test on the 20 decided pairs, p about 0.012). This is one listener's clear preference, not a population estimate.

**Distillation.** Distillation trains a small model on a large model's output. The F5 voice read 17,257 sentences
aloud in one consenting speaker's voice; 22.9 hours that the recognizer could read back were kept; a VITS model
trained on them scores 9.7% on 100 sentences. No listener has compared it with its teacher, so "matches the teacher"
rests on CER alone, which this project showed to be unreliable for close comparisons.

### Running in a browser

Reducing weights to 8-bit integers (int8) cost nothing measurable for either translation or recognition, but the 1B
models were still over 1 GB each.

| model | download | quality | time in Chrome on an Apple M1 |
|---|---|---|---|
| translator, pruned to 6 encoder / 3 decoder layers | 312 MB | dev chrF++ 41.7 (teacher 51.2) | 0.41 s per 20-word sentence |
| translator, 12 / 6 layers | 514 MB | 47.4 | 0.69 s |
| translator, 24 / 6 layers (student C, released) | 1.1 GB | 51.2 | 0.88 s |
| recognizer, 300M + character language model (released) | 652 MB + 20 MB | Common Voice test CER 8.0 | 0.21 s for 2.8 s of audio, 0.65 s for 8.9 s |
| recognizer, 1B in half precision | 1.95 GB | CER 8.6 | about 0.14 s per second of audio |
| voice, VITS (released) | 114 MB | read-back CER 10.0 (40 sentences) | about 3 times faster than real time |

Browser translation time is dominated by the decoder, which runs once per output token; the encoder runs once per
sentence. Keeping the whole encoder and one decoder layer in four recovered the teacher's
score at under a second, for about $1.40 of training. Student C translates Brahui→English only. In a two-way spoken
demo page, Brahui speech to English text took 0.60 s and English speech to Brahui text 2.1 s.

## What did not work

These cost the most to learn.

1. **Normalizing spelling and retraining on all the speech (ASR stage 3) tied stage 2.** Stage 2 gained 0.7 CER from
   800 steps on the two smaller datasets. Mapping BrahuiSpeech's spellings onto that convention and retraining on
   all 80 hours gave 8.61 against 8.55 on Common Voice (paired difference [−0.09, +0.22], a tie) and was 0.16 worse
   on spontaneous speech ([+0.07, +0.27]). The film benchmark tied too.
2. **A far better teacher did not make a better student (MT round 5).** Retranslating the 22,776 curated sentences
   with the 80% teacher changed the meaning of roughly half the targets. The 1.3B student trained on them tied the
   previous student on every sealed chrF++ measure, on MP (41.4% against 43.4%; 15 sentences gained, 21 lost) and on
   judged discrepancies per film window (4.54 against 4.52). Its film chrF++ was 1.0 lower ([−1.59, −0.43]), with
   outputs 6% longer. The retranslated pairs were 8% of the training examples; whether dilution explains the tie
   was not tested.
3. **Cheap translation tricks.** More epochs (flat from epoch 3), averaging two checkpoints, and reranking by
   round-trip translation all lost. Fine-tuning on human pairs only raised chrF++ by 1 to 2 points while the judge
   saw no difference and round-trip consistency fell. Three rounds of prompt changes moved the LLM translator by
   less than its run-to-run noise of 0.5 chrF++. Consensus over several teacher runs tied the best single run.
4. **Round-trip and cycle checks did not find errors.** Re-speaking a transcript with the synthesizer and comparing
   it with the original audio separated 30 wrong transcripts from 27 right ones with an AUC of 0.68. The
   recognizer's own confidence scored 0.95.
5. **Music separation lost to speech enhancement.** Demucs raised the usable share of broadcast chunks from 18% to
   26%, DeepFilterNet to 46%, both together to 47%.
6. **The neural vowelizer lost to a lookup table.** A character-level model with sentence context got 90.6% of
   held-out words exactly right; looking up each word's most frequently heard spelling got 91.3%.
7. **Read-back CER disagreed with the listener twice.** A voice 0.6 points better on CER lost a blind test 3 to 7;
   the vowelled voice, 0.7 points worse, won 16 to 4. Adding 28 hours of cleaner audio did not help by ear.
8. **VITS on 1.6 hours tied the untuned Urdu voice.** Best run 21.4%; the Urdu voice reading Brahui, 20.7%.
9. **The 300M recognizer missed its target until a language model fixed it without training.** The student scored
   9.9% against a target of 9.6%. Beam search with a character 6-gram language model built from 111,000 Brahui
   sentences (test sentences removed) brought it to 8.0%, below the 1B teacher's 8.6% without one. Try decoding
   with a language model before paying for more training.
10. **Browser details.** int8 models on WebGPU were slower than on WebAssembly, and the translator's output was
    garbled. The int8 voice was smaller, less accurate and about three times slower than full precision.

## What it cost

All figures are approximate and come from run logs.

| item | cost |
|---|---|
| Rented GPUs, recognizer stages 2 and 3 (stage 1, about 3 hours on an A100, was not logged) | about $9 |
| Rented GPUs, all translation training | about $31 |
| Rented GPUs, browser distillation | about $15 |
| Rented GPUs and CPUs, speech synthesis, vowel models, broadcast processing | about $26 |
| Metered LLM API: web-text translation ($36.58 for 45,706 pairs, $0.74 per 1,000 sentences), PDF text ($6.64), dictionary digitization ($18.80), one evaluation run (about $6) | about $68 |

About $16 of the GPU total was waste: a job that silently fell back to CPU, checkpoints lost to full disks, and
a machine that billed without becoming reachable. The dollar figures understate the LLM cost. The first 23,000
translations, the agent retranslation (16.6 hours, 108 million tokens, 26,202 lookups), much of the judging and the
coding agent itself ran on a flat-rate subscription and have no metered price.

## What is released, and what is not

Released:

| what | where | licence |
|---|---|---|
| Code, evaluation scripts, per-round records | https://github.com/qasimy123/brahui-digitize | MIT |
| Speech recognizer, 1B | `qkhawaja/brahui-asr-ctc1b-s3` | Apache-2.0 |
| Speech recognizer, 300M ONNX + character language model | `qkhawaja/brahui-asr-ctc300m-student-onnx` | Apache-2.0 |
| Translator, NLLB 1.3B, both directions | `qkhawaja/brahui-nllb-1.3b-r5` | CC-BY-NC-4.0 |
| Translator for the browser, Brahui→English | `qkhawaja/brahui-mt-student-c-onnx` | CC-BY-NC-4.0 |
| Browser voice (VITS), one voice | `qkhawaja/brahui-tts-vits-browser` | CC-BY-NC-4.0, no impersonation |
| Synthetic parallel data | `qkhawaja/brahui-synthetic-parallel` (dataset) | CC-BY-4.0 |
| Machine transcripts of 160 hours of broadcast audio | `qkhawaja/brahui-media-pseudolabels` (dataset) | research use |

The non-commercial licences are inherited from NLLB-200 and MMS. The voice is modelled on a native speaker in my
family who agreed to the release; she is not named and her recordings are not published. The transcript dataset
contains video IDs, timestamps and uncorrected machine text, and no audio.

Withheld:

- **All scraped audio.** It belongs to its broadcasters.
- **The F5-TTS checkpoints.** F5 copies any voice from a short clip and was trained on that audio.
- **Dictionary text**, including the digitized Shad dictionary and the agents' lookup notes, which quote
  copyrighted dictionaries.
- **The Ali and Kobayashi sentences.** A script rebuilds the 500-sentence set from the authors' publication.
- **DailyDialog back-translations.** DailyDialog's licence is non-commercial share-alike.
- **Film benchmark material.** Only the script and the numbers are published.

## Limitations, and what to do next

- **One listener.** Every human judgement here comes from one native speaker in my family. Her overall view is
  that the recognition, the translations and the voice are good enough to be legible and usable, though not perfect. In an informal review of
  one news bulletin she preferred the stage-3 recognizer and the round-5 translator, both of which my test sets
  called ties, and her complaints were about recognition (clipped segment edges, a misheard place name), not
  translation. That is a single impression.
- **No native speaker has reviewed the translations or the synthetic dataset systematically.** A 41% MP rate means
  most Brahui→English sentences from the trained model have at least one factual discrepancy. Do not use it for
  medical, legal or safety-critical text.
- **Judge circularity**, as described above. I also did not test whether the public benchmark sentences appear in
  the teacher's pretraining data; the film controls address recall only for the film.
- **Coverage.** Training speech is mostly read or broadcast; text is mostly news, religion and politics. Dialect
  balance is unknown, and the vowel sources lean toward some dialects.
- **Small test sets.** The 302-sentence development set cannot detect differences under about 8 MP points.

For someone continuing this work, in order:

1. **Get native speakers to review outputs.** An hour of blind pairwise listening settled questions that weeks of
   automatic metrics could not. Start with translations on the cleaned development sets and the student voice
   against its teacher.
2. **Test the dilution hypothesis** for the round-5 tie: train on the retranslated pairs alone or upweighted.
3. **Try a student that can look things up.** The teacher's gain came from dictionary lookup; the students have
   none.
4. **Fix recognition before translation for broadcast speech:** segmentation at pauses and better handling of names.
5. **Give Brahui its own language tag** if Urdu must keep working.

If you speak Brahui and see something wrong in a transcript, a translation or the voice, please open an issue on
GitHub or a discussion on the relevant Hugging Face page. Corrections from speakers are what this work most lacks.

## Acknowledgements

This work stands on Mozilla Common Voice and its Brahui contributors; Meta's Omnilingual ASR, NLLB-200, MMS and
BOUQuET; the BrahuiSpeech-70H-V2 dataset and its Whisper baseline; Google's SMOL; FineWeb-2, FinePDFs and GlotCC;
F5-TTS, DeepFilterNet and Demucs; DailyDialog and Tatoeba; Ali and Kobayashi's *Brahui Texts* (2024); and the
dictionaries of Shad and Sarparra. Detailed records: `docs/ASR_RESULTS.md`, `docs/TTS_RESULTS.md`,
`docs/PLAN_SERVE.md`, `evals/mt/hillclimb/`.
