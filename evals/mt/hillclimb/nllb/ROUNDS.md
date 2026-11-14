# NLLB hillclimb rounds (started 2026-10-19)

Instruments: `scripts/train_nllb.py` writes per-sentence outputs to `evals/mt/nllb/<run>_{after,dev,dev_epN}/`;
`evals/mt/compare_nllb.py A B` gives the paired bootstrap of B minus A per metric. Decisions are made on
`data/dev/mt_dev.jsonl` (302 gold pairs carved out of the gold training set, 120 SmolSent + 44 BOUQuET-dev
paragraphs); the sealed sets only confirm. Dedup of all corpora against `data/eval/`: 0 exact, 2 fuzzy (two CV
train clips whose sentence re-occurs in CV test), excluded via `data/parallel/eval_overlap.jsonl`.

## Noise floor (measured before round 1)

| metric | n | run 3 score | 95% bootstrap CI (eval sampling) | paired delta CI, run 3 − run 2 |
|---|---|---|---|---|
| en→brh chrF++, smol_heldout | 200 | 38.25 | [36.5, 39.8] (width 3.3) | [+2.0, +5.1] |
| brh→en chrF++, bouquet_test | 854 | 46.90 | [45.7, 47.9] (width 2.2) | [+1.6, +3.0] |
| en→brh chrF++, bouquet_test (new, reverse use of the sealed set) | 854 | – | expected width ≈ 1.5 | – |
| round-trip en→brh→en chrF++ | 200 / 854 | – (first measured in round 0) | | |

The 200-sentence smol set is too noisy to read a 1-point change; the paired delta is the decision statistic,
and the 854-sentence BOUQuET set scored en→brh is added as the primary en→brh instrument. Seed noise
(same recipe, seeds 0 and 1) is measured in round 0 below. Inference is deterministic (beam 4), so rerunning an
identical checkpoint has zero noise; the noise that matters is training seed + eval sampling.

ASR noise floor (per-clip paired bootstrap, `evals/asr/runs/ctc_s{2,3}__*`): CV test CER 8.55 vs 8.61, paired
s3−s2 CI [−0.09, +0.22]; Omni test 11.82 vs 11.98, paired CI [+0.07, +0.27]: stage 3 is a small real regression
on spontaneous speech that the unpaired CIs had hidden. A change must move CV by > 0.2 and Omni by > 0.3 CER,
paired, to count. JESUS e2e: paired s2 vs s3 CI [−0.55, +0.66] chrF++, plus ~0.5 run-to-run LLM noise; a change
must move it by > 1 point.

## Round 0: seed noise + baseline on the new instruments
Pod (H100 SXM, $3.99/h, US-MO-1), plan `plans/r0.txt`: run-3 recipe (NLLB-600M, 5 epochs, lr 1e-4,
bs 32, label smoothing 0.1) with seeds 0 and 1, gold minus the dev split.

Seed 0 (`nllb_r0_s0`, 41 min on the H100): dev curve en→brh 36.2 / 37.2 / 37.4 / 37.1 / 37.6 and brh→en 45.1 / 46.1 /
47.0 / 46.8 / 46.5 over epochs 1–5; round-trip dev 58.4. Sealed: en→brh smol 37.78, bouquet 36.67, brh→en 46.79,
round-trip smol 54.75 / bouquet 59.41. Paired vs run 3 (which trained on the 302 dev pairs): en→brh −0.47
[−1.18, +0.25], brh→en −0.10 [−0.48, +0.27]: withholding dev costs nothing measurable. The dev curve is flat from
epoch 3, which predicts the 10-epoch candidate (round 1) will not move.

### Failure buckets on dev (seed 0), Opus 5.5 judge over the 150 lowest-chrF sentences per direction
`evals/mt/bucket_failures.py nllb_r0_s0_dev`; each verdict has a quoted evidence span in
`evals/mt/nllb/nllb_r0_s0_dev/buckets_*.jsonl`.

| en→brh | n | brh→en | n |
|---|---|---|---|
| synonym_ok (meaning right, wording differs from reference) | 75 | wrong_word (content word mistranslated) | 80 |
| wrong_word | 27 | synonym_ok | 42 |
| urdu_register | 22 | entity | 16 |
| entity | 12 | polarity/tense | 4 |
| omission | 7 | misread source | 4 |
| grammar / orthography | 7 | omission / addition | 4 |
| meaning preserved | 114 / 150 | meaning preserved | 43 / 150 |

Readings. (1) en→brh is close to its single-reference ceiling: three quarters of its "failures" preserve the meaning.
The real residue is lexical choice (wrong word + Urdu register, 33%). (2) brh→en has genuine comprehension failures:
53% wrong content words with meaning lost in 71% of the worst half. (3) The wrong words are not out-of-vocabulary on
the Brahui side (0% OOV; rare-word rate 11.9% vs 11.3% in the best half); they are everyday concrete vocabulary:
cucumber, carrot, clown, homework, salty, zoo, spider, fence appear 2–11 times each on the English side of the
70k-pair corpus (news, religion, politics, speech), and wrong-word sentences carry 50% more rare English content
words than the best half (14.6% vs 9.6%). Largest bucket, cause: register/vocabulary mismatch between the corpus
and everyday language.

**Round 2 change chosen from this bucket:** back-translate human everyday English (DailyDialog 70,781 utterances +
60,000 Tatoeba sentences, `scripts/build_en_everyday.py`, deduped against the sealed English sides: 1 exact hit
excluded) into Brahui with the seed-0 student (`scripts/backtranslate_en.py`, beam 4) and train brh→en on those
pairs only (`dirs: ["brh2en"]`, so the model never imitates its own Brahui). Prediction: brh→en dev up by several
points, en→brh unchanged, round-trip up.

### Seed noise (seed 1 minus seed 0, paired bootstrap, 500 resamples)
| metric | seed 0 | seed 1 | delta | 95% CI |
|---|---|---|---|---|
| en→brh dev | 37.61 | 37.65 | +0.04 | [−0.66, +0.70] |
| brh→en dev | 46.49 | 46.77 | +0.29 | [−0.44, +1.00] |
| round-trip dev | 58.44 | 58.28 | −0.16 | [−1.24, +0.83] |
| en→brh smol (sealed) | 37.78 | 38.01 | +0.23 | [−0.74, +1.07] |
| en→brh bouquet (sealed) | 36.67 | 36.91 | +0.24 | [−0.15, +0.63] |
| brh→en bouquet (sealed) | 46.79 | 46.77 | −0.03 | [−0.37, +0.34] |
| round-trip bouquet (sealed) | 59.41 | 59.91 | +0.50 | [−0.33, +1.27] |

Seed noise is about ±0.3 chrF++ on the 854-sentence sets and ±0.7 on the 302-sentence dev set. **Decision rule from
here:** a change is kept only if the paired CI against the seed-0 baseline excludes 0 on dev AND on the matching
sealed set, with the point estimate above the seed band (> 0.7 dev, > 0.4 bouquet). Baseline for all later rounds:
`nllb_r0_s0`.

## Round 1 (cancelled): 10 epochs
Hypothesis: training loss was still falling, so more epochs help. Both seeds' dev curves are flat from epoch 3
(seed 0: 37.4 → 37.6 en→brh, 47.0 → 46.5 brh→en over epochs 3–5), so the falling loss is fit to the synthetic corpus,
not generalization. The queued 10-epoch run was cancelled before starting (user agreed, 2026-10-19 07:00 UTC),
saving ~80 min / ~$5. Knowledge: epochs are not a lever at this data size.

## Round 2: iterated back-translation from everyday English (`nllb_r2_bt`)
Change: + 130,106 pairs (DailyDialog + Tatoeba English, Brahui by the seed-0 student, greedy), brh→en direction only;
everything else as round 0. Training examples 140,208 → 270,314. Started 07:10 UTC on the same pod.

Result (73 min training on the H100, ~$5.50 of pod time; corpus generation ~20 min shared GPU; API $0):

| metric | r0 seed 0 | r2_bt | paired delta | 95% CI | verdict |
|---|---|---|---|---|---|
| brh→en dev | 46.49 | 49.99 | +3.50 | [+2.54, +4.48] | better |
| brh→en bouquet (sealed) | 46.79 | 48.20 | +1.41 | [+0.77, +2.08] | better |
| round-trip dev | 58.44 | 62.88 | +4.44 | [+2.88, +6.07] | better |
| round-trip bouquet (sealed) | 59.41 | 64.02 | +4.61 | [+3.45, +5.62] | better |
| round-trip smol (sealed) | 54.75 | 58.67 | +3.92 | [+2.15, +5.55] | better |
| en→brh dev | 37.61 | 37.87 | +0.25 | [−0.47, +0.90] | noise |
| en→brh bouquet (sealed) | 36.67 | 37.14 | +0.48 | [+0.02, +0.84] | borderline, below the seed band |
| en→brh smol (sealed) | 37.78 | 37.62 | −0.16 | [−1.34, +0.81] | noise |

**Kept.** Dev and sealed agree on brh→en and round-trip; en→brh is unchanged, as predicted (the BT pairs train only
brh→en). The sealed gain (+1.4) is smaller than dev (+3.5): dev is half SmolSent, the everyday register the new
data targets, while BOUQuET test is mixed-domain prose. Dev curve brh→en 48.8 / 49.1 / 49.8 / 50.2 / 50.0: most of
the gain is in epoch 1, i.e. the register exposure, not the volume. Knowledge: the brh→en gap to the teacher was
partly register; 130k everyday pairs closed 1.4 of the 10.4 sealed points and 3.5 of the dev points. New best:
`runs/nllb_r2_bt/final` (brh→en 48.2; en→brh 37.1 bouquet / 37.6 smol; round-trip 64.0).

Pod deleted 08:27 UTC. Session GPU spend: 3.07 h × $3.99 ≈ $12.20 (round 0 incl. the cancelled run ≈ $6.70,
round 2 ≈ $5.50). Remaining RunPod credit ≈ $14; next GPU rounds go on an A100 ($1.39/h, ≈ $2 per 600M round).

## Round 3a (free, Mac): model soup of r0 seed 0 + r2 (equal weights)
Dev: en→brh 37.33, brh→en 49.17, round-trip 61.45, all below r2 alone (37.87 / 49.99 / 62.88). **Reverted.** A soup
of runs trained on different data averages away the data gain; a soup of same-data seeds (s0 + s1) would be the
fairer test but s1's weights were left on the volume to save transfer; expected gain ≤ 0.5, not worth a pod.

## Round 3b (free, Mac): MBR decoding + reverse-model reranking for en→brh (`scripts/mbr_decode.py`, r2 model)
Dev, 16 epsilon-sampled candidates + beam: beam 37.98, chrF-MBR 37.56, reverse-model (round-trip) pick 35.38,
mix (λ=0.3) 37.14. **Reverted.** The samples are worse than the beam output and the reverse model prefers Brahui
that back-translates easily, not Brahui that is right. Knowledge: decoding is not the en→brh lever; the model's
distribution is.

## Round 3c data: FinePDFs + GlotCC teacher translation
20,762 sentences (doc LID ≥ 0.7) submitted to Opus 5.5 via the Batch API: 306 of 520 requests succeeded, 12,202
pairs, $6.64. The 214 failures are refusals (religious verse) and off-by-one counts. The Brahui side is OCR text
and visibly noisier than FineWeb (the teacher marks "(?)" on unsure spans); filtered before training (below).

## Adequacy instrument (outside judge, Opus 5.5; `evals/mt/bucket_failures.py --worst 302`, `evals/mt/pairwise_judge.py`)
Full dev set, both directions. "Meaning preserved" is a checkable yes/no per sentence with a quoted evidence span.

| model | en→brh meaning preserved | brh→en meaning preserved |
|---|---|---|
| r0 seed 0 | 235 / 302 (78%) | 125 / 302 (41%) |
| r2_bt | 217 / 302 (72%) | 135 / 302 (45%) |

Pairwise, random order, same judge (wins A / wins B / ties; two-sided sign test):
- en→brh, r0 vs r2: 116 / 101 / 85, p = 0.34 — no difference (the absolute drop above is single-judge noise, ~±2.5 pp).
- brh→en, r0 vs r2: 73 / 147 / 82, p < 0.001 — r2 better, agreeing with the sealed chrF++ gain.

Reading for the "parity" question: by adequacy, en→brh (72–78%) is already ahead of brh→en (41–45%). The chrF++
asymmetry (37 vs 48) is the metric across scripts and references, not the translations. Parity therefore means
raising brh→en adequacy toward en→brh's, which round 2 began, and lifting en→brh's register (wrong word + Urdu
register still 30% of its buckets), which needs more real Brahui or more capacity, not decoding. The pairwise test is
the decision instrument for en→brh from here, alongside the paired chrF++ CI.

## Offline end-to-end (JESUS film, human English subtitles, 156 windows; `evals/st/jesus_bench.py --model nllb:<dir>`)
| cascade | chrF++ | BLEU | off-by-one |
|---|---|---|---|
| stage-3 ASR → Fable 5.1 (LLM translator, film context) | 43.9 | 19.6 | 21.3 |
| stage-3 ASR → NLLB r4 1.3B (offline) | 41.19 | 17.5 | 20.8 |
| stage-3 ASR → NLLB r2 (offline) | 39.25 | 14.6 | 20.8 |
| stage-3 ASR → NLLB r0 seed 0 (offline) | 38.64 | 14.1 | 20.7 |
Paired: r2 − r0 [+0.11, +1.13] (round 2 helps speech too, less than on prose); Fable − r2 [+3.7, +5.7] is the
remaining offline gap on spoken, noisy input. First fully offline number for the mission's cascade.

## Round 4: capacity, NLLB-200-distilled-1.3B on the round-2 data, 3 epochs (`nllb_r4_1p3b`)
L40S 48 GB at $1.09/h (no A100 stock in US-MO-1); bs 8 × 4, 1.76 it/s, ~4 h. First attempt
crashed at the epoch-1 checkpoint: the 50 GB volume was full (five old NLLB runs at 4.7 GB each). Freed 24 GB
(runs already on the Mac) and relaunched 10:46 UTC; ~$1.65 lost. Epoch-1 dev from the lost attempt: en→brh 37.91,
brh→en 50.56, round-trip 62.51, i.e. at round 2's final level after one epoch. Queued after it: gold-only
adaptation of r2 and r4 (the ASR stage-2 analogue; motivated by the register evidence: the model writes
transliterations and Urdu loans, فٹنس / ٹریک / متنازعہ, where the references use native Brahui, جانجوڑی / پٹ ولوٹ /
جنجالی).

Result (relaunch: 4.2 h training on the L40S; pod 09:16–15:35 UTC ≈ $6.90 incl. the crash and the gold runs):

| metric | r2_bt | r4_1p3b | paired delta | 95% CI | verdict |
|---|---|---|---|---|---|
| en→brh dev | 37.87 | 39.51 | +1.64 | [+0.68, +2.50] | better |
| en→brh smol (sealed) | 37.62 | 39.49 | +1.87 | [+0.58, +3.07] | better |
| en→brh bouquet (sealed) | 37.14 | 38.18 | +1.04 | [+0.45, +1.67] | better |
| brh→en dev | 49.99 | 51.40 | +1.41 | [+0.36, +2.65] | better |
| brh→en bouquet (sealed) | 48.20 | 49.83 | +1.63 | [+0.89, +2.26] | better |
| round-trip dev | 62.88 | 65.59 | +2.72 | [+1.04, +4.32] | better |
| round-trip bouquet (sealed) | 64.02 | 65.97 | +1.95 | [+0.95, +3.03] | better |
| round-trip smol (sealed) | 58.67 | 60.64 | +1.97 | [−0.03, +3.83] | borderline |

Pairwise judge, en→brh dev: r4 wins 150, r2 wins 113, ties 39, p = 0.026. **Kept.** Dev curve en→brh 37.9 / 39.1 /
39.5, brh→en 50.7 / 51.3 / 51.4. Knowledge: with ~270k examples the 600M student was capacity-bound in both
directions; the 1.3B is the first change that moved en→brh outside noise on every instrument including the judge.
**New best: `runs/nllb_r4_1p3b/final`** (en→brh 39.5 smol / 38.2 bouquet, brh→en 49.8, round-trip 66.0).

## Round 4-gold: gold-only adaptation (`nllb_r2_gold` 3 epochs; `nllb_r4_gold` 1 epoch, lr 2e-5)
| comparison | en→brh dev | en→brh smol | en→brh bouquet | brh→en bouquet | round-trip dev | round-trip bouquet |
|---|---|---|---|---|---|---|
| r2_gold − r2 | +1.24 [+0.59, +1.95] | 40.02 vs 37.62 | 38.44 vs 37.14 | 48.01 vs 48.20 | −0.21 [−1.54, +1.15] | 62.86 vs 64.02 |
| r4_gold − r4 | +0.67 [+0.16, +1.24] | 41.81 vs 39.49 | 39.09 vs 38.18 | 49.70 vs 49.83 | −1.58 [−2.93, −0.35] | 64.83 vs 65.97 |

Pairwise judge, en→brh dev, r4_gold vs r4: 93 vs 100 wins, 109 ties, p = 0.67: no judged difference. chrF++ rises
(the output now matches the references' purist wording more often) while round-trip falls outside noise and the judge
sees no gain in meaning or naturalness. **Reverted as the main model** under the round-trip rule; kept on disk
(`runs/nllb_r4_gold/final`) as the "reference-style" variant. Knowledge: a short gold pass buys 1–2 chrF++ of
reference matching but not judged quality; the metric is saturating on wording, as the bucket analysis predicted.
The r4_gold second epoch crashed the volume quota again (1.3B checkpoints); the 1-epoch rerun is what is reported.
Sealed paired CIs for the gold variants: r2_gold − r2: en→brh bouquet +1.29 [+0.82, +1.78], brh→en −0.19 [−0.66, +0.28],
round-trip bouquet −1.17 [−2.10, −0.25], smol −1.83 [−3.41, −0.37]. r4_gold − r4: en→brh smol +2.32 [+1.53, +3.34],
bouquet +0.91 [+0.50, +1.31], brh→en −0.13 [−0.54, +0.34], round-trip bouquet −1.14 [−1.89, −0.33], smol −1.57
[−3.06, +0.08]. Same shape on dev, sealed and both base models: the revert stands.

## Round 5: data swap — the 22,776 curated real-Brahui pairs retranslated by the workflow teacher (`nllb_r5_retrans`)
L40S $1.09/h, 2026-10-20 23:25 UTC. Same recipe as r4 (1.3B from base, 3 epochs, bs 8×4, 1.72 it/s, 4.1 h) with
`synthetic_brh_en.jsonl` replaced by `synthetic_retranslated_brh_en.jsonl` (teacher: Opus agents with dictionary/glossed-corpus retrieval,
80.2 MP on clean dev2 vs ~35 for the old single-pass; 52% of the targets changed meaning, 49% on the spontaneous-speech subset).
Dev curve en→brh 38.2 / 39.0 / 39.3, brh→en 49.4 / 51.3 / 51.1 (r4: 37.9 / 39.1 / 39.5, 50.7 / 51.3 / 51.4).

| metric | r4_1p3b | r5_retrans | paired delta | 95% CI | verdict |
|---|---|---|---|---|---|
| en→brh smol (sealed) | 39.49 | 39.97 | +0.48 | [−0.63, +1.58] | tie |
| en→brh bouquet (sealed) | 38.18 | 38.23 | +0.05 | [−0.47, +0.58] | tie |
| brh→en bouquet (sealed) | 49.83 | 50.21 | +0.38 | [−0.16, +0.99] | tie |
| round-trip bouquet (sealed) | 65.97 | 66.19 | +0.22 | [−0.57, +1.08] | tie |
| brh→en dev MP (clean refs) | 43.4% | 41.4% | +15/−21 flips | — | tie |
| JESUS e2e chrF++ (ASR s3 → model, 156 windows) | 41.19 | 40.17 | −1.02 | [−1.59, −0.43] | **worse** |

JESUS outputs are 6% longer with r5 (68.5 vs 64.3 words; reference 64.9): the retranslation kept hedges and repetitions of the speech
sources ("do not polish away"), and the student reproduces that register against polished subtitles. MP judge on the JESUS windows: see below.
MP judge on the 156 JESUS windows (English-only, calibrated; a window passes only if every sentence does, so window MP is tiny and the
discrepancy count is the usable signal): r4 4.52 vs r5 4.54 judge discrepancies per window, paired diff +0.02 [−0.17, +0.21]: **identical
meaning preservation**. Verdict for run 1: the teacher's +45 MP on its own register did not transfer to the 1.3B student at all; the only
visible effect is style (longer, more literal English). Hypotheses, in order: (1) the 22.8k changed targets are 8% of the 270k examples
and are diluted by the 130k back-translated everyday pairs and 45k FineWeb pairs; (2) NLLB at 1.3B learns surface regularities, not the
compositional corrections the teacher made; (3) chrF/BLEU on polished subtitles penalise faithful-but-wordy outputs. (1) is testable
cheaply (upweight or isolate the retranslated set); (2) is the Student-B question (decoder LLM with retrieval). Run 2 (continued from r4,
2 epochs on the new data) reports next.

### Round 5 run 2: continued from r4 on the retranslated data (`nllb_r5_retrans_from_r4`, 2 epochs, lr 5e-5)
Dev curve: epoch 1 en→brh 39.21 / brh→en 51.51 / round-trip 65.13; epoch 2 39.64 / 51.22 / 65.07 (r4 final 39.51 / 51.40 / 65.59). **Tie
on dev**, same as run 1. The final save failed with "Disk quota exceeded" (volume at 43 GB of 50 before the save: run 1's final 5.2 GB +
run 2's epoch-1 checkpoint 5.2 GB + the epoch-2 checkpoint being written). The epoch-2 checkpoint is a 96 KB stub and I removed the
epoch-1 checkpoint before verifying that, so run 2 has no weights and no sealed eval (≈ $3 of GPU). Not re-run: its dev curve already
shows a tie, and the sealed sets have never contradicted a dev tie in rounds 0–5.
**Round 5 verdict:** retranslating the curated corpus with the 80-MP teacher changes nothing the 1.3B student can show, on written text
or on spoken input, at meaning level. Next: the dilution test (retranslated set alone or upweighted, on top of r4, one GPU hour, with
≥ 25 GB free on the volume first), then the decoder student if that ties too.
