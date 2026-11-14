# Round 0 (2026-10-19): meaning-preserved (MP) rate on dev, brh→en

Tool: `evals/mt/round0_mp.py` (translate / judge / bucket). Dev = `data/dev/mt_dev.jsonl` (302: 182 BOUQuET, 120 SMOL). Judge sees English only (ref vs hyp), lists discrepancies, returns MP bool. Binomial 95% CI ≈ ±5.5.

## MP rate (calibrated rubric, Opus 5.5 judge)
| system | prompt | MP | CI | SMOL | BOUQuET |
|---|---|---|---|---|---|
| NLLB-1.3B r4 (student) | — | 37.4% | 32–43 | 25.0 | 45.6 |
| Fable 5.1 | 40 shots, default effort, 20/req | 51.7% | 46–57 | 35.8 | 62.1 |
| Opus 5.5 | 40 shots, effort medium (default), 20/req | 55.0% | 49–61 | 35.8 | 67.6 |
| **Opus 5.5 "full"** | all 1,690 gold pairs cached in system, effort xhigh, 5/req, deductive-parse instruction | **63.2%** | 58–69 | 47.5 | 73.6 |

Paired opus → opus_full: +43 / −18 sentences, sign test p = 0.002. Strict (uncalibrated) rubric gave 45.7 / 41.7 / 33.1 for opus / fable / student; the calibration (paraphrase, modal wording, near-synonyms are not discrepancies) is in the script. Judge agreement Opus vs Fable on the same output: 282/302.

## Failure buckets (gloss pass with Brahui source)
| bucket | opus 40-shot (164 fails) | opus_full, Opus bucketer (111) | opus_full, Fable bucketer (111) |
|---|---|---|---|
| reference_error | 51 | 65 | 55 |
| unknown_word | 57 | 19 | 22 |
| loanword | 14 | 8 | 11 |
| composition | 16 | 6 | 9 |
| morphology | 8 | 4 | 4 |
| judge_error | 5 | 4 | 6 |
| other (entity, segmentation, source_noise, vowel) | 13 | 9 | 4 |

Of opus_full's 111 failures, both bucketers independently agree on 47 reference errors and 35 model errors (29 disagree). Samples of the agreed reference errors hold up by eye: the English reference is the *original* (SMOL/BOUQuET were translated en→brh by humans) and the Brahui translator dropped or changed content ("dentist"→ڈاکٹر "doctor", "sneaky cunning fox… legendary treasure"→"clever fox… treasure", "long-term goal"→بھلو مسخت "main goal"). A faithful brh→en translation cannot recover these.

## Conclusions
1. The user's hypothesis (LLM teacher ≈95%) is not yet reached but the direction is right: corpus-in-context + xhigh effort + deductive instruction moved MP +8 and cut unknown-word failures by two thirds. Effort and evidence both matter; neither alone.
2. Estimated MP against *corrected* references ≈ (302 − 35 − ~15)/302 ≈ 83% for opus_full. This is an estimate, not a measurement; it needs a cleaned dev set.
3. Next levers, in order: (a) clean dev: for every double-flagged reference error, store the Brahui-faithful English alongside the original and judge against either (never from one model's output alone — require two bucketers + gloss evidence); grow dev from Ali & Kobayashi, whose English was written *from* the Brahui; (b) lexicon + morphology table in context (the remaining 20% unknown_word); (c) consensus of ≥3 full-mode passes for the hard bucket; (d) SMOL sentences are context-free and 20 pts harder than BOUQuET — document context for BOUQuET paragraphs should lift it further.
4. Cost: opus_full ≈ $6 for 302 sentences (cache read 1,690 pairs ≈ 150k tokens per request). Run pilots on 60-sentence slices before the full dev.

# Round 1 (2026-10-19, same day): clean dev, lexicon, consensus, noise floor

## Clean dev (`data/dev/mt_dev_clean.jsonl`)
Two independent adjudicators (Fable, Opus; full corpus in context, xhigh; they see Brahui + reference, never a hypothesis) each judged every dev reference for faithfulness to the Brahui. Pilot 60: 9 both-unfaithful. Full 302: Fable 51, Opus 57 unfaithful; **36 corrected** (both flag it and their corrections are judged equivalent), 5 both-flag-but-differ, 31 disputed (one flags), 230 ok. Judge `--refs clean` re-judges a failure against the corrected reference and accepts either. This is deliberately conservative; the hypothesis-aware bucketer still calls ~45% of remaining failures reference errors, but that instrument sees the model's answer and is biased toward it.

## MP on clean dev (Opus calibrated judge)
| system | MP | SMOL | BOUQuET |
|---|---|---|---|
| NLLB-1.3B r4 | 43.4 | — | — |
| Fable 40-shot | 53.3 | | |
| Opus 40-shot | 58.9 | | |
| Opus full (corpus + xhigh) | **70.5** | 55.0 | 80.8 |
| Opus full, re-run (test–retest) | 68.5 | 51.7 | 79.7 |
| Opus full + A&K glossary + shaigle list in context | 70.9 | 50.0 | **84.6** |

Pilot slice (60 ids): base 39, +A&K glossary 38, +both lexicons 43, Fable full 40, rewriting consensus 38, MBR-select 42; oracle over candidates 48/60.

## Noise floor (the main finding of the round)
Two runs of the identical config flip **32/302 sentences (+13/−19)**; only 122/302 hypotheses are string-identical. A +6/−2 pilot gain became +18/−17 on the full set. **The 302-sentence dev cannot see a lever smaller than ~8 points.** Levers must be judged on ≥1,000 sentences or on k repeated runs. This is why dev grows from Ali & Kobayashi next.

## What remains (Fable bucketer on opus_full_lex2's 88 failures)
BOUQuET is nearly solved: 28 failures, 15 of them reference-side. SMOL/ok holds the real model mass: 16 unknown_word + 9 loanword out of 42. The unknown words are ordinary native vocabulary (ھلی horse, پام understanding, بیرہ just, سوبری stability, توئی monthly, سلہ dust, گنج enough, کنجا here) that neither the A&K folktale glossary nor the 1,100-word blog list contains. → Shad's modern Brahui–English dictionary (284 pp, archive.org OCR unusable for Arabic script) is being digitized with Opus vision (`scripts/ocr_dictionary.py`, ≈$0.07/page, 30/30 entries correct on the pilot page).

## Decisions
- Lexicon stays the lever, but with the right lexicon (Shad), and judged on a 1,000+ dev.
- Consensus/MBR shelved: candidates fail on the same sentences; selection gains ≤ noise.
- Opus 5.5 effort must be set explicitly (default medium). Fable full ≈ Opus full on the pilot; Opus is 4× cheaper.

# Round 2 (2026-10-19, night): dev2 from Ali & Kobayashi; Shad dictionary is the lever

API credits ran out mid-round; `round0_mp.py` now runs on `claude -p` (BRH_BACKEND=cli).

## Dev2: `data/dev/mt_dev_ak.jsonl`
500 sentences (250 short stories, 250 oral folktales) from Ali & Kobayashi 2024 (`data/eval/ak2024_sentences.jsonl`, 4,606 glossed sentences parsed by `scripts/parse_ak2024.py`). English was written *from* the Brahui, so there is no translator loss. Romanized source plus an Opus transliteration to Perso-Arabic (`brh`). Hard register: Jhalawan/Sarawan dialect, oral narrative, no grammatical gender (pronoun gender must come from context).

| config (Opus 5.5, full corpus in context, xhigh) | MP on 200-slice | paired vs base |
|---|---|---|
| base, Arabic input (500) | 32.0 | — |
| base, roman input | 34.5 (69/200) | — |
| + story context (5 preceding sentences) | 42.0 | +30/−15 |
| + Shad dictionary in context (7,816 entries) | 52.0 | +45/−10 |
| **+ Shad + context 5** | **62.0** | **+65/−10** |

Buckets of the base's 340 failures (Fable): unknown_word 35%, reference 21%, loanword 14%, morphology 8%, judge 7%, composition 5%, entity 5%.

## Resources digitized today (no humans)
- Shad 2021 Brahui→English dictionary: `scripts/ocr_dictionary.py` (Opus vision, 284 pages, 7,816 entries, $18.80; 30/30 on the checked page) → `data/lexicon/shad_brh_en.txt`.
- Dixey/Sarparra English→Brahui (2022): text layer parsed → `data/lexicon/sarparra_en_brh.txt` (4,403 entries, roman Brahui; the Arabic column's font is unreadable).
- A&K glossary (3,418) + grammar outline; shaigle word list (1,100). 18 scanned books in `data/raw/dictionaries/` (README there).

## Reading
The 302 dev could not see the lexicon (noise ±8); dev2 sees it at +17.5 with p ≪ 0.001. The lever ladder is now: lexicon → context → (next) more context, Sarparra, in-register gold, then consensus on the hard bucket.

# Round 3 (2026-10-20, 00:xx): clean dev2 with gloss-aware adjudication

Two adjudicators (Opus, Fable; full corpus; the A&K morpheme gloss given as ground truth) audited the 200-slice references of dev2: 67 and 65 of 200 unfaithful; **48 corrected** (both flag, corrections equivalent), 9 differ, 27 disputed. `data/dev/mt_dev_ak_clean.jsonl`.

| config (Opus 5.5 full corpus xhigh, roman input, 200 slice) | MP orig refs | MP clean refs |
|---|---|---|
| base | 34.5 | 37.5 |
| + context 5 | 42.0 | 48.5 |
| + Shad | 52.0 | 61.5 |
| + Shad + context 5 | 62.0 | **71.0** |
| same, re-run (test–retest) | 59.5 | 68.0 |
| + Shad + Sarparra + context 5 | 64.0 | **74.5** |
| + Shad + context 10 | 57.5 | — |

Noise floor at this level: +13/−18 flips between identical runs (≈ ±3 pts on 200). Sarparra's +3.5 is inside it; context 10 < context 5 is at the edge.

Net: the lever stack moved dev2 from 37.5 to ~72 on clean references in one evening, all without humans: a dictionary digitized by vision OCR, story context, and a reference audit anchored on linguists' glosses.

## Round 3 addenda
- MBR-MP selection over the three best runs: 75.5 (oracle 82.0). Selection ≈ best single run; shelved again.
- Gloss-first scaffold (model writes the morpheme gloss, then translates): 69.5 vs 74.5 (+11/−21). No gain; shelved.
- In-register glossed gold in context (3,931 A&K pairs with gloss lines, ~1.4 MB on top of the lexicons): every `claude -p` call failed (prompt too large for the CLI path). Needs the API or a retrieval step (nearest 50 glossed sentences per item) instead of the whole set.
- Remaining failures at 74.5 (Opus bucketer): reference 21/51, morphology 7, composition 7, unknown_word 6, loanword 5, entity 5. Model-side ≈ 30/200 → true MP at this config ≈ 85%.

## Next (in order)
1. Retrieval instead of whole-corpus: per item, nearest 50 gold pairs + nearest 50 glossed A&K pairs (by token overlap) + dictionary entries for the item's tokens only. Smaller prompt, so the CLI path works and Fable becomes affordable; also the shape the offline student will use.
2. Re-adjudicate the 27 disputed + 9 differing dev2 references with a third pass; grow the clean dev2 slice from 200 to 500.
3. BOUQuET paragraph context on the 302 dev (its ids carry P###-S# paragraph structure); SMOL has none.
4. Then distil: the teacher config is "Opus xhigh + Shad + Sarparra + context 5 + retrieved gold"; translate the held Brahui corpus with it for the student.

# Round 4 (2026-10-20): workflow teacher with retrieval-by-grep — 81.5

Claude Code Workflow (`scripts/workflows_brahui_teacher.js`): ten Opus 5.5 agents (effort xhigh), 20 sentences each with 5 sentences of story context; each agent greps the Shad dictionary, Sarparra, the A&K glossary and the 3,931 glossed A&K sentences for every uncertain stem, then composes. 324 tool calls, 1.38M subagent tokens, 10.5 min wall-clock for 200 sentences.

| teacher (dev2 200-slice, clean refs) | MP | paired vs workflow |
|---|---|---|
| prompt-based best (Shad + Sarparra + ctx5, whole corpus in context) | 74.5 | +24/−10 |
| MBR over three prompt-based runs | 75.5 | +21/−9 |
| **workflow, retrieval by grep, Opus 5.5** | **81.5** (ch2 82, ch3 81) | — |

Above the noise floor (±15 flips). The gain comes from what the prompt could not hold: the glossed corpus consulted per stem, and tool-driven lookups with multiple spellings. Each row carries `hard_words` (stem=meaning (source)), which is a free lexicon of attested lookups for the student.

Next: a second independent pass + agreement filter in the same workflow, then run it over the 22,776 curated real-Brahui sentences (`data/parallel/synthetic_brh_en.jsonl`) with document context; ~19 h at 10 concurrent agents if the usage window allows.

# Scale run (2026-10-20): curated corpus retranslated by the workflow teacher
`scripts/workflows_brahui_retranslate.js`, run <run-id>: 760 chunks × 30 sentences, Opus 5.5 xhigh agents grepping Shad (Arabic headwords), the gold corpus and the glossed corpus; 16.6 h wall-clock at 16 concurrent agents, 108M subagent tokens, 26,202 tool calls. **22,759 / 22,776 sentences retranslated** → `data/parallel/retranslated_brh_en.jsonl` (new en, old en, agree flag, hard_words lookups). The workflow's 1,000-agent lifetime cap cut the verify stage to 240 chunks (agreement with the old single-pass translation there: 3,360 / 7,199 = 47%, consistent with the old teacher's ~55% MP); the remaining 520 chunks are verified in run <run-id>, and the 17 missing sentences translated there. Like-for-like score of this teacher on the 302 Arabic dev: run <run-id>.
Lesson: one Workflow run counts every agent() call against a 1,000 cap, so two-stage pipelines over >500 items must be split into runs.

## Status at the weekly-limit stop (2026-10-20)
- `data/parallel/retranslated_brh_en.jsonl`: all 22,776 sentences retranslated; agreement verdicts for 607/760 chunks (18,209 sentences: 9,380 agree, 8,829 disagree ≈ 52% agree). Chunks 607–759 unverified (resume run <run-id> after the reset; cached agents replay).
- Workflow teacher on the Arabic 302 dev (dev-free gold): 242/302 translated into `evals/mt/round0/mtdev_wf_opus/`, **not yet judged** (judge needs claude -p). Resume run <run-id> for the last 3 chunks, then `judge mtdev_wf_opus --refs clean`.
- Subscription weekly limit resets 2026-10-24 15:00 America/Los_Angeles; API credit is 0. Nothing model-side can run before one of those changes.

## Workflow teacher on the Arabic 302 dev (2026-10-20, dev-free gold): 71.9
`evals/mt/round0/mtdev_wf_opus/`: 217/302 on clean refs (BOUQuET 81.9, SMOL 56.7). Paired vs prompt-based Opus full 70.5: +26/−22, i.e. a tie. Retrieval by grep is worth +7 on the folktale dev2 (74.5 → 81.5) and 0 on news/web. Reading: the 302 dev's residual failures are not lexical (reference faults ≈ 45% of them by the hypothesis-aware bucketer; SMOL sentences have no context). Next lever for this register is a second, stricter reference audit of the 31 disputed + 5 differing refs and BOUQuET paragraph context; the teacher itself is already past where this dev can measure a change below ±8.

# Headline (2026-10-20): workflow teacher on the full clean dev2 (500) — 80.2
Clean dev2 now covers all 500 sentences (gloss-aware adjudication by Opus and Fable: 94 corrected, 92 disputed, 314 ok).
| teacher | MP (500, clean refs) | ch.2 stories | ch.3 folktales |
|---|---|---|---|
| Opus 5.5 full corpus in context, xhigh (Arabic input) | 35.2 | 44.0 | 26.4 |
| **workflow: Opus 5.5 agents, retrieval by grep (Shad, Sarparra, A&K glossary, glossed corpus), 5-sentence context** | **80.2** | 82.0 | 78.4 |
Paired +236/−11 (n=500). The 200-slice estimate (81.5) held on the 300 unseen sentences.
On the 302 Arabic news/web dev the same teacher is 71.9 (= prompt-based). The remaining gap to 95 is now mostly reference-side on both devs plus composition/morphology; vocabulary is largely solved by the dictionary.
