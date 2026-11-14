# Plan: 95%+ meaning preserved, brh→en, no humans in the loop

Written 2026-10-19. Target: an offline Brahui→English model whose output preserves the meaning of ≥95% of sentences on the sealed BOUQuET test, judged by an outside LLM, and translations a bilingual reader would call excellent. Today: NLLB-1.3B r4 ≈45% (dev); Opus single-pass teacher unmeasured on this metric (round 0 measures it).

**No artificial ceiling.** Nothing in the data or the methods caps this below 95. The single-pass teacher is not the ceiling: a consensus of several teachers with a lexicon, glossed gold, and document context is a better teacher than any one pass, and iterated back-translation from human English lets the student pass its teacher (already shown for en→brh). Human verification is a later audit, never a plan step and never a prerequisite. If a round stalls, the stall rule applies: find the broken grader, bad reference, or missing resource; do not lower the target.

## The metric
- **MP rate** = fraction of sentences where the judge finds no added, dropped, or changed fact between hypothesis and reference English. Judge = Opus 5.5 (never the model under test); rubric returns the list of discrepancies, not a score. Both sides are English, so the judge needs no Brahui, which is why MP is trustworthy where chrF++ is not.
- Noise: binomial. 302 dev → ±5.5 pts; 854 test → ±3.3. Decisions need > noise on dev AND test. Grow dev toward 1,000 sentences from the new gold (never from test).
- At 95 the judge itself must be audited: sample judge verdicts with a second judge model; when they disagree, the rubric is wrong, not the model.
- chrF++ and round-trip remain regression guards only.

## Gold and lexical resources found 2026-10-19 (no humans in our loop)
1. **Ali & Kobayashi 2024, *Brahui Texts* (ILCAA, open access).** 2,335 sentences with morpheme segmentation, interlinear gloss, English translation; folktale transcripts time-stamped with audio. Romanized; Arabic-script original of one source (Khwaja Ahmed's stories) on the same site for building and validating the transliterator. `https://bkm.aa-ken.jp/brahui/`
2. **Modern dictionaries on archive.org:** Sarparra, English→Brahui (Brahui Academy 2022); Shad, Brahui→English (2021). Arabic script.
3. **Bray 1909/1934** grammar + etymological vocabulary, and **Bray 1913 *Life-history of a Brahui*** (bilingual). Public domain, archive.org.
4. **Brahui New Testament (1998–2023)** exists (PLAN.md §Phase 2 already lists the NT and a Quran translation as PDF scans); text location still to be found. Verse-aligned to English → thousands of gold pairs.
5. UD Brahui treebank (`data/raw/ud_brahui_kholum`), CLDR, GATITOS already on disk.

## Round 0: self-assessment (API only, ~$5)
1. Translate dev with Fable and Opus (plain prompt). MP rate for each.
2. Bucket every teacher failure with a gloss pass (word-by-word): unknown word / loanword / unwritten-vowel ambiguity / morphology (case, tense, negation, evidentials) / named entity / segmentation / reference error. The gloss separates "did not know the word" from "knew the words, composed them wrong".
3. Same buckets for the student (r4). The histograms order the levers.

## Phase 1: build a teacher that beats any single pass (API, ~$40)
- **L1 Lexicon + morphology in context.** Lexicon from the two modern dictionaries + Bray + the Ali & Kobayashi glosses (which give morpheme → function). Retrieve entries for every token and put them in the prompt. Where MTOB-style methods work, it is here.
- **L2 Document context.** Whole BOUQuET paragraphs; media in 30 s windows.
- **L3 Semantic consensus.** ≥3 independent translations (Opus plain, Opus glossed, Fable glossed); judge picks the majority meaning and lists disagreements. The disagreement set is the hard bucket fed back to L1.
- **L4 Retrieval few-shot from gold.** Nearest gold sentences (Ali & Kobayashi, SMOL, BOUQuET dev, NT when found) in the prompt.
- Exit: teacher MP ≥ 90 on dev. If it stalls, re-bucket.

## Phase 2: distil without copying errors (GPU, ~$60–100 top-up)
- Retranslate all held Brahui with the phase-1 teacher. **Agreement filter** (consensus agreed) and **round-trip filter** (en→brh→en judged equal). Disagreements dropped or low-weighted.
- Student A: NLLB-1.3B/3.3B (known pipeline). Student B: decoder LLM (Qwen3/Gemma 4–9B, LoRA) with lexicon and retrieved gold in the prompt, local index, fully offline. B is the bet: composition is where NLLB loses meaning.
- Multi-task: translation + glossing (from Ali & Kobayashi) so the student learns morphology explicitly.

## Phase 3: self-improvement to 95
- **Hard-example mining.** Student/teacher disagreements → retranslate with full apparatus → add. Repeat until the disagreement rate flattens.
- **Iterated back-translation** from human English in the conversational register with the phase-2 en→brh; the student's brh→en trains on real English targets and passes its teacher.
- **New media** (dramas, cartoon, sermons from the 2026-10-19 list) → ASR → conversational Brahui source → teacher pairs. LID per segment.
- **Comparable-corpus mining**: Brahui/Urdu versions of the same news pages, matched by our own translations.
- **Judge-as-utility MBR at training time**; final distillation of the best cascade into one offline model.
- **Audit the eval at the top**: at ≥90 MP, residual "failures" are as likely reference errors as model errors; re-judge references against the Brahui with the full apparatus and record which side was wrong.

## Rules carried over
One change per round, dev decides, test confirms, never read test, dedup against eval, outside judge only, record in `evals/mt/hillclimb/`. Report MP rate first in every round summary.
