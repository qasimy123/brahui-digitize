# Hillclimb narrative (final)

**Setup.** Claude translating en->brh (SMOL: 663 train / 200 sealed test) and brh->en (BOUQuET: 504 dev as train / 854 sealed test) with 12 fixed + 3 retrieved few-shot pairs per sentence, drawn from train pools only. Metric: corpus chrF++ per direction; headline = mean of the two test values. Paired-bootstrap 95% CI on test ≈ ±1.2 (en->brh, n=200) and ±0.5 (brh->en, n=854).

| variant | model | en->brh test | brh->en test | headline | vs v0-Opus (test, paired CI) |
|---|---|---|---|---|---|
| baseline | Sonnet 5.5 | 30.8 | 50.4 | 40.6 | – |
| baseline_opus | Opus 5.5 | 34.9 | 57.2 | **46.1** | model lever: +4.2 [+2.8,+5.5] / +6.8 [+5.9,+7.7] |
| v1 (glossary + grammar + brh->en hints) | Opus | 35.3 | 56.7 | 46.0 | +0.4 [-0.9,+1.7] / -0.6 [-1.1,-0.1] |
| v2 (glossary, no brh->en section) | Opus | 35.4 | 56.2 | 45.8 | +0.4 [-0.8,+1.6] / -1.0 [-1.6,-0.5] |
| v3 (en->brh: v2; brh->en: v0 again) | Opus | 35.3 | 56.7 | 46.0 | +0.4 [-0.6,+1.6] / -0.5 [-1.1, 0.0] |

**What moved the number.** Switching Sonnet -> Opus 5.5: +5.5 headline, far outside noise. Nothing else did.

**What didn't.** The round-1 analyzer correctly diagnosed the dominant failure (Opus/Sonnet reaching for Urdu vocabulary where references use native Brahui words) and wrote a verified glossary. On Opus the glossary is worth at most +0.4 en->brh (inside the CI) and costs 0.5–1.0 brh->en. Round 3 reran the *identical* v0 brh->en configuration and still measured -0.54, so Opus's own sampling variance is ≈0.5 chrF++ on 854 sentences; the brh->en "regressions" in v1/v2 are mostly that. Conclusion: with retrieved few-shots already in context, prompt text is a saturated lever for this task.

**Decision.** Ship `prompts/system_v0.md` + Opus 5.5 as the prototype translator. Further MT gains need data, not prompts: fine-tune NLLB / an open LLM on SMOL + mined/OCR'd parallel text (Phase 2), and build a Brahui-English parallel set with native speakers (plan §Phase 2.3).
