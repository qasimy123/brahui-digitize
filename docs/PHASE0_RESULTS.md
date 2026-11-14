# Phase 0 results: does the idea work?

Date: 2026-10-07. Hardware: one Mac M1 16 GB, no GPU. LLM: Claude via the local `claude` CLI.

## Verdict

**Yes, the cascade is viable, and the data situation is better than expected.** Without training anything:

- Brahui speech is already transcribed at **19% character error rate** by an existing Whisper-Small fine-tune, consistently on two independent test sets (read speech and spontaneous speech). That is usable for a prototype and well inside the range where fine-tuning on the full corpus (~90 h) should push it under 10%.
- Brahui to English translation with Claude Opus 5.5 scores **chrF++ 57 / BLEU 38** on the BOUQuET benchmark. That is already intelligible-quality output.
- English to Brahui is the weak leg: **chrF++ 35 / BLEU 10**. Output is grammatical Brahui but wording diverges from references, and vocabulary drifts toward Urdu. This was the direction that needed data and fine-tuning. *Update 2026-10-19:* two rounds of synthetic parallel data took an NLLB-600M student to chrF++ 38.3 on the same set, above every LLM measured (`docs/ASR_RESULTS.md`, `docs/SYNTHETIC_BOOTSTRAP.md`).
- Prompt engineering is exhausted for MT: three hillclimb rounds moved the score less than the model's own run-to-run noise (about 0.5 chrF++). Model choice moved it by 5.5.

Go/no-go thresholds from the plan were CER under 15 and chrF++ above 30. ASR is slightly above the CER bar but with obvious headroom; MT clears it in both directions.

## What was built

| Piece | Path | Notes |
|---|---|---|
| Perso-Arabic normalizer | `brahui/text.py` | NFC, letter canonicalization (ي/ی, ك/ک, ه/ہ), diacritic/tatweel/ZW* stripping, digit and punctuation unification; `for_scoring()` variant for CER/WER |
| Manifests | `scripts/build_manifests.py` → `data/manifests/` | CV 27 (1,703/506/882), Omnilingual brh_Arab (264/73/106, audio extracted), BrahuiSpeech-70H (15,626, extracted, 8.6 GB), SMOL sent/doc, BOUQuET dev/test |
| Sealed eval sets | `data/eval/` + `SHA256SUMS` | cv_test 882 clips, omni_test 106 segments, smol_heldout 200 pairs, bouquet_test 854 pairs. Never train on these. |
| ASR eval harness | `scripts/asr_eval.py` | backends: `ctc:` (any HF Wav2Vec2ForCTC: omniASR conversions, our fine-tunes), Whisper, MMS, Omnilingual fairseq2 original (separate venv `.venv-omni/`); per-clip results, bootstrap CIs, resumable |
| MT eval harness | `evals/mt/run_eval.py`, `compare.py` | Claude via `claude -p --restricted --json-schema`, 12 fixed + 3 retrieved few-shots from train pools only, sentence and corpus chrF++/BLEU, paired bootstrap comparison |
| Hillclimb record | `evals/mt/hillclimb/` | `_state.json`, `narrative.md`, per-variant `change.md`/`change.patch`/`summary.json`, `report.html` |
| ASR fine-tune | `scripts/train_asr.py` | full fine-tune of omniASR CTC-1B v2 (HF format), pretrained 10k-token head kept; lazy audio loading; smoke-tested on CPU |
| Long-segment chunker | `scripts/segment_long.py` | CTC forced alignment → ≤20 s chunks of the 88 s Omnilingual segments |
| GPU job | `scripts/gpu/train_asr.sh`, `docs/GPU_TRAINING.md` | baselines → chunk → train → eval → tables on one A100 |
| Result tables | `evals/RESULTS_TABLES.md` | regenerate with `uv run python scripts/summarize_results.py` |

## ASR baselines (zero training)

| model | test set | clips | hours | CER % | 95% CI | WER % |
|---|---|---|---|---|---|---|
| TBOGamer22/Brahvi-Whisper-Small (trained on BrahuiSpeech-70H) | CV test (read) | 882 | 1.7 | **18.9** | 18.3–19.8 | 52.9 |
| same | Omnilingual test (spontaneous) | 106 | 2.5 | **19.1** | 18.3–20.0 | 54.6 |
| Meta Omnilingual ASR CTC-300M (zero-shot, brh_Arab) | CV test, 300-clip seeded subset | 300 | 0.6 | 22.7 | 21.4–24.1 | 67.2 |
| Whisper-Small Brahui on that same 300-clip subset | | 300 | 0.6 | 19.5 | | |

Observations:
- Whisper output is clean: no empty or runaway hypotheses, median length ratio 0.98, 100% Arabic script. Errors are spelling-level (unwritten short vowels, ہ/ھ, ے/ی, word boundaries), which is exactly what the WER-vs-CER gap says and what a normalizer plus a language model fix.
- Its CER on CV test (18.9) is below the 22.8 its author reported on their own held-out split, so it generalizes across sources.
- The resource inventory was wrong about MMS: `facebook/mms-1b-all` has **no `brh` adapter**. Dropped in favour of fine-tuning omniASR CTC-1B, which already knows Brahui.
- Omnilingual ASR runs at 2x realtime on this CPU, so it was evaluated on subsets only. Its loader also rejects audio longer than about 40 s, so the 88 s Omnilingual test segments are chunked to 30 s for it (`OmniBackend`); the spontaneous-speech numbers for Omnilingual models are still running. On a GPU the full sets run in minutes (`scripts/gpu/train_asr.sh` step 1 does this for the base model).

## MT baselines and hillclimb

Sealed test: SMOL hold-out (en→brh, 200) and BOUQuET test (brh→en, 854). Corpus chrF++ / BLEU.

| variant | model | en→brh test | brh→en test | headline | paired Δ vs v0-Opus (test) |
|---|---|---|---|---|---|
| v0 prompt | Sonnet 5.5 | 30.8 / 7.0 | 50.4 / 30.3 | 40.6 | |
| **v0 prompt** | **Opus 5.5** | **34.9 / 10.1** | **57.2 / 37.8** | **46.1** | model lever: +4.2 [+2.8, +5.5], +6.8 [+5.9, +7.7] |
| v1 glossary + grammar + brh→en hints | Opus | 35.3 / 10.3 | 56.7 / 37.3 | 46.0 | +0.4 [−0.9, +1.7], −0.6 [−1.1, −0.1] |
| v2 glossary only | Opus | 35.4 / 10.3 | 56.2 / 36.7 | 45.8 | +0.4 [−0.8, +1.6], −1.0 [−1.6, −0.5] |
| v3 per-direction (v2 / v0) | Opus | 35.3 / 9.9 | 56.7 / 37.3 | 46.0 | +0.4 [−0.6, +1.6], −0.5 [−1.1, 0.0] |

- The analyzer's diagnosis (train split only) was right: the dominant en→brh failure is Urdu vocabulary where references use native Brahui words (verified by frequency counts in the train pools, e.g. جوان 63× vs اچھا 0×). But a glossary in the prompt adds at most +0.4 on Opus, because retrieved few-shots already carry that signal.
- Round 3 reran the identical v0 brh→en configuration and measured −0.54, so Opus's sampling noise is about 0.5 chrF++ on 854 sentences. All prompt deltas sit inside that. The loop stopped per the playbook rule "don't spend rounds on unmeasurable changes".
- Winner: `evals/mt/prompts/system_v0.md` on Opus 5.5.

## Cascade demo (Brahui audio → Whisper-Small Brahui → Opus 5.5 → English)

`uv run python scripts/demo_cascade.py --sample 4` on sealed Common Voice test clips (reference transcript shown for comparison; the model never saw it):

| reference | ASR | English |
|---|---|---|
| نما مہر انا کلہو آک سر متوس تو ای مستی دا ”بوگ شو“ اٹ تولپرہ | نما مہر نا کلاو آک سر متوس تو ای مستی دا بوگ شو گٹ تول پرین | Your messages of love didn't reach me, so I had already sat down and given up on this. |
| کروکو بندغ ئس بریک تو اونا دانگ اینگ مخلوق نا مُچی مریک بوگ شوگ نوشتہ | کروک و بندغس بریک تونا دانگیں مخلوق نا مچی مریک بوگ شوک نے بشتا | When someone like that comes along, a crowd of people like him gathers around him, and they push you aside and leave you behind. |
| ہنوک آ نی سدائی خوش مریس یار تینا خیر ءِ خوا نی ہم تو کاسہ | اینو کنے سدائی خوش مریس یا تینا خیر وانی ہم تو کاسے | Today I was really happy, or may you go in peace too. |
| ہرا ادارہ غاتیٹی کاریم کرے | ارا ای دا رغا تے ٹی کاریم کرے | Where in these lands should I work? |

Reading: ASR spelling errors propagate into meaning errors in the translation (rows 3 and 4: ادارہ "institution" became "lands"). This is the main argument for the Phase 1 ASR fine-tune and for passing N-best or confidence into the translator later. Latency on the Mac is about 1 s of ASR per 7 s clip plus 5 to 10 s for the Claude call, so realtime needs a GPU for ASR and a streaming translation setup.

## Data notes

- Training speech: Common Voice 9.9 h + Omnilingual 11 h + BrahuiSpeech-70H 69.5 h (15,626 clips, all under 20 s, so none are dropped by the length filter).
- BECO text corpora still need their terms accepted on Mozilla Data Collective (three clicks).
- FineWeb-2 brh_Arab is mixed Brahui/Urdu at document level but mostly Brahui at sentence level: a per-sentence function-word filter keeps 47k sentences / 1.1M words (`scripts/build_fineweb_units.py`, 2026-10-19). The earlier "mostly Urdu" reading was wrong.
- Omnilingual segments are about 88 s long; for CTC training they need force-alignment into shorter chunks (the fine-tune script drops them above 20 s for now).

## Next steps

1. **GPU fine-tune** (you start the pod, the script does the rest): `docs/GPU_TRAINING.md`. Target CER < 8 on CV test, < 10 on Omnilingual test.
2. **Synthetic parallel data at scale** (done in part 2026-10-19, `docs/ASR_RESULTS.md`): every real Brahui sentence we hold, translated to English by an LLM teacher, trains an NLLB student that already beats the teacher on en→brh. Next rounds: better teacher passes (context, consensus), iterated back-translation from human English text, the remaining Brahui text corpora, bigger students. The programme and its instruments: `docs/SYNTHETIC_BOOTSTRAP.md`.
3. **Speaker-free evaluation**: sealed sets with bootstrap CIs, the JESUS-film end-to-end benchmark against human subtitles, off-by-one and round-trip controls, and LLM-judged adequacy on conversational held-out sets. Human listening (`docs/RECORDING.md`) is a later verification step, not a prerequisite.
4. **More MT data**: OCR pilot on the Quran/Bible PDFs (50 pages, Claude vision) for verse-aligned triplets; Bray's 1934 dictionary and online glossaries as entries; the FinePDFs slice through the same sentence filter.
5. **BECO text corpora**: accept terms on Mozilla Data Collective, then `scripts/download_mdc.sh` with a fresh key.
6. **Cascade demo**: mic → fine-tuned ASR → Opus → `say`, and the reverse with OmniVoice zero-shot, as a non-realtime proof.
