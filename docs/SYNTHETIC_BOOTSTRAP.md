# The synthetic bootstrap: why a non-expert teacher is the unlock

> **Read with `docs/REPORT.md`.** This is the argument as written on 2026-10-19, before it was tested, and it is kept
> unchanged below. Two of its claims did not hold up. (1) "The student inherits teacher quality": a teacher that
> improved from 35% to 80% meaning preserved produced no measurable gain in the 1.3B student (round 5). (2) "Humans
> are not the bottleneck": the one native listener settled questions that the automatic metrics got wrong twice, and
> the lack of native review is the project's main limitation. What did hold: an LLM can turn untranslated Brahui into
> training data that takes a small model from chrF++ 21 to 50.

*Written 2026-10-19 after the first two rounds of synthetic parallel data. Numbers: `docs/ASR_RESULTS.md`.*

## The observation

A frontier LLM translates Brahui into English at chrF++ 57 on a sealed test set (Opus 5.5 and Fable 5.1, within
a point of each other). Neither model was trained to be a Brahui expert. Brahui is a Dravidian language, but it is
written in Perso-Arabic script and carries a thick layer of Urdu, Persian, Arabic and Balochi vocabulary; its
grammar is regular; and most sentences in the wild are about things the model already knows (news, religion,
politics, everyday life). The model reads Brahui the way a well-read Urdu speaker with a linguist's instincts
would: it recognises the loanwords, infers the native function words from distribution and context, and
reconstructs the meaning. Two controls on the JESUS-film benchmark show this is real translation and not recall:
the aligned score is double the off-by-one score, and withholding the film context costs only one point.

That is the unlock. A language with 2.5 million speakers, no MT engine and no NLP paper has, today, a translator
that is good enough to be a teacher. Everything below follows from that.

## What a teacher buys

A teacher that reads Brahui converts **monolingual Brahui** into **parallel data**. Brahui text is not rare: the
ASR transcripts (680k words), FineWeb-2 (1.1M words after a sentence filter), FinePDFs, GlotCC, dictionaries,
and the 15.7 h of broadcast audio our ASR already turns into text. Each of these becomes English on the other
side for under $1 per thousand sentences through the Batch API.

The student then learns from pairs whose **Brahui side is authentic**. For English→Brahui this matters more
than teacher quality: the target the model learns to produce is real Brahui, and teacher errors sit on the
source side as noise, which sequence-to-sequence training is robust to. This is why the 600M-parameter student
passed its own teacher on English→Brahui after two rounds (38.3 vs 34.9 chrF++), although the teacher never
wrote a single Brahui sentence it trained on. The teacher's weakness in writing Brahui is irrelevant; only its
reading matters, and its reading is good.

The same mechanism runs the other way. A student that writes Brahui converts **human English** into pairs whose
English side is authentic: translate conversational English corpora into Brahui, and the Brahui→English student
trains on real English targets. Alternating the two directions is iterated back-translation, the standard engine
of low-resource MT, and we now have both halves of it.

## The loop

```
real Brahui text ──teacher (LLM)──► Brahui/English pairs ──► student en→brh  (target side real)
real English text ──student en→brh──► Brahui/English pairs ──► student brh→en (target side real)
better students ──► better pseudo-labels for ASR ──► more Brahui text ──► round N+1
teacher quality ──► improves with paragraph context, consensus of samples, student-informed glossaries
```

Each arrow is cheap and measurable. Two rounds so far cost under $45 and moved en→brh from 20.7 (zero-shot)
to 38.3. Nothing in the recipe is exhausted.

## Levers, in order, with the size we expect

| lever | what it does | expected effect |
|---|---|---|
| more epochs, same data | loss was still falling at 2.53 | +1 to +2 both directions |
| bigger student (NLLB-1.3B, 3.3B, or a small LLM) | capacity for 100k+ pairs | +1 to +3 |
| teacher with paragraph context and 3-sample consensus | raises teacher quality 2 to 4 points; student inherits | +2 to +4 brh→en |
| iterated back-translation from human English (DailyDialog, OpenSubtitles, news) | real English targets in the everyday register | +3 to +6 brh→en, first conversational data |
| remaining Brahui text (FinePDFs, rejected FineWeb share re-judged by the LLM, dictionaries, OCR of the religious PDFs) | ~0.5M more words, rare vocabulary | +1 to +3 en→brh |
| ASR label correction: LLM proposes, forced-alignment score gates | cleaner pseudo-labels, stage-4 ASR | ASR CER down on spontaneous speech; more usable text |
| TTS from the same student's output and real text | ASR⇄TTS cycle; multiplies speech data | second-order; unlocks the spoken product |

Stacked, that is a plausible path to the mid-40s en→brh and to or beyond the single-pass teacher on brh→en on
the sealed sets, and the loop keeps going after that, because every round creates the data for the next.

*Measured 2026-10-19 (`evals/mt/hillclimb/nllb/ROUNDS.md`):* more epochs: 0 (dev flat from epoch 3). Iterated
back-translation from 130k everyday English sentences: brh→en +1.4 sealed / +3.5 dev, round-trip +4.6, en→brh 0.
NLLB-1.3B: +1.0 to +1.9 en→brh, +1.6 brh→en, round-trip +2.0. Decoding tricks (MBR, reverse reranking) and model
soups: negative. A gold-only adaptation raises chrF++ by 1–2 but an outside judge sees no gain and round-trip falls,
which is the single-reference ceiling showing. Current best: en→brh 38.2 / 39.5, brh→en 49.8, judged meaning
preserved 72–78% en→brh and ~45% brh→en.

## How we know, without a speaker

- **Sealed sets** with bootstrap intervals (`data/eval/`): written prose, two directions.
- **Human-reference end-to-end**: the JESUS film's English subtitles against ASR→MT output, with the off-by-one
  control that separates translation from recall.
- **Round-trip consistency** (en→brh→en) as a drift alarm; it falls if the Brahui side slides toward Urdu.
- **LLM-judged adequacy** on held-out conversational Brahui: the teacher that reads Brahui at 57 can grade
  meaning preservation, and grading is an easier task than translating.
- **Agreement across independent systems**: two ASR models agree at chrF 86 on the same speech; a translation
  that two unrelated students and the teacher agree on is right far more often than not.

These instruments have blind spots, chiefly the register of the output (natural Brahui versus Urdu wearing
Brahui grammar) and anything all our models are wrong about in the same way. That is what human listening is
for, as verification at the end of a round, in twenty-minute sessions, not as a prerequisite for the next one.
`docs/RECORDING.md` describes that step for when it is wanted.

## What "excellent" means here

The target is an offline system that handles everyday Brahui in both directions well enough to use: news,
conversation, instructions, the content of the broadcast corpus. Single-reference chrF++ saturates in the 60s
even for professional translators, so the sealed numbers will stop moving before quality does; the LLM-judged
and round-trip instruments carry the measurement after that. The programme above reaches that target with
synthetic supervision alone. Humans add a voice, a check, and eventually a community of users who correct it;
they are not the bottleneck.
