## ASR baselines (sealed test sets)

| model | set | clips | hours | CER | CER 95% CI | WER | empty | RTF |
|---|---|---|---|---|---|---|---|---|
| ctc:ctc1b_brh (stage 1) | cv_test | 882 | 1.71 | 9.3 | 8.8–10.0 | 30.2 | 5 | 0.01 |
| ctc:ctc1b_brh (stage 1) | omni_test | 106 | 2.48 | 12.5 | 12.1–13.0 | 42.8 | 0 | 0.01 |
| ctc:omniASR-CTC-1B-v2-hf | cv_test | 882 | 1.71 | 19.6 | 18.9–20.4 | 61.3 | 6 | 0.01 |
| ctc:omniASR-CTC-1B-v2-hf | omni_test | 106 | 2.48 | 16.4 | 15.9–17.1 | 55.6 | 0 | 0.02 |
| ctc:omniASR-CTC-300M-v2-hf | cv_test | 300 | 0.57 | 25.0 | 23.5–26.4 | 71.5 | 3 | 0.11 |
| ctc:ctc1b_brh_s2 (stage 2) | cv_test | 882 | 1.71 | 8.6 | 8.0–9.2 | 27.6 | 6 | 0.01 |
| ctc:ctc1b_brh_s2 (stage 2) | omni_test | 106 | 2.48 | 11.8 | 11.3–12.3 | 41.0 | 0 | 0.01 |
| ctc:ctc1b_brh_s3 (stage 3, normalized BS) | cv_test | 882 | 1.71 | 8.6 | 8.0–9.2 | 27.7 | 4 | 0.01 |
| ctc:ctc1b_brh_s3 (stage 3, normalized BS) | omni_test | 106 | 2.48 | 12.0 | 11.5–12.5 | 41.2 | 0 | 0.01 |
| facebook/omniASR-CTC-1B | cv_test | 300 | 0.57 | 18.3 | 17.0–19.7 | 58.1 | 3 | 2.74 |
| facebook/omniASR-CTC-1B | omni_test | 40 | 0.90 | 20.0 | 18.3–21.9 | 62.1 | 0 | 3.76 |
| facebook/omniASR-CTC-300M | cv_test | 300 | 0.57 | 22.7 | 21.4–24.1 | 67.2 | 3 | 1.99 |
| facebook/omniASR-CTC-300M | omni_test | 40 | 0.90 | 21.3 | 20.0–22.6 | 66.5 | 0 | 2.20 |
| data/raw/models/brahvi_whisper_small | cv_test | 882 | 1.71 | 18.9 | 18.3–19.8 | 52.9 | 0 | 0.13 |
| data/raw/models/brahvi_whisper_small | omni_test | 106 | 2.48 | 19.1 | 18.3–20.0 | 54.6 | 0 | 0.15 |

Paired bootstrap (per clip) stage 3 − stage 2: CV test [−0.09, +0.22] CER, Omni test [+0.07, +0.27] (`evals/asr/hillclimb/ROUNDS.md`).

## Media set v1 (`data/eval/media_test.jsonl`, 284 chunks / 68.8 min, labels = stage-3 output + gated LLM edits; regression guard only, see evals/asr/hillclimb/ROUNDS.md)

| model | CER | 95% CI | off-by-one control |
|---|---|---|---|
| stage 3 (labeler) | 1.11 | 0.95–1.32 | 84.8 |
| stage 2 | 4.61 | 4.26–4.97 | 85.2 |
| Whisper-Small Brahui | 17.58 | 16.49–18.76 | 84.9 |

## MT variants (chrF++ corpus / BLEU corpus)

| variant | model | en→brh train | en→brh test | brh→en train | brh→en test | headline (mean test chrF++) | errors |
|---|---|---|---|---|---|---|---|
| baseline | sonnet | 32.6 / 9.0 | 30.8 / 7.0 | 48.6 / 28.1 | 50.4 / 30.3 | 40.57 | 0 |
| baseline_opus | opus | 34.1 / 9.7 | 34.9 / 10.1 | 54.8 / 35.1 | 57.2 / 37.8 | 46.07 | 0 |
| v1 | sonnet | 32.7 / 8.9 | 32.4 / 7.9 | 48.5 / 28.0 | 50.1 / 29.8 | 41.27 | 0 |
| v1_opus | opus | 34.3 / 10.2 | 35.3 / 10.3 | 54.3 / 34.7 | 56.7 / 37.3 | 45.98 | 0 |
| v2_opus | opus | 34.7 / 10.2 | 35.4 / 10.3 | 54.1 / 34.0 | 56.2 / 36.7 | 45.79 | 0 |
| v3_opus | opus | 34.4 / 10.2 | 35.3 / 9.9 | 54.7 / 34.8 | 56.7 / 37.3 | 45.98 | 0 |
| baseline_fable | claude-fable-5-1 | – | 35.5 / 9.1 | – | 56.3 / 36.9 | 45.87 | 0 |


## Speech translation, JESUS film (chrF++ / BLEU over 156 windows, ref = human English subtitles)

| input to Fable 5.1 | chrF++ | BLEU | off-by-one control |
|---|---|---|---|
| Urdu subtitles (ceiling) | 47.4 | 22.3 | 21.8 |
| stage-3 ASR | 43.9 | 19.6 | 21.3 |
| stage-2 ASR | 43.9 | 19.3 | 21.4 |
| stage-3 ASR, blind | 42.7 | 17.6 | 21.2 |
| **offline:** stage-3 ASR → NLLB r4 1.3B (`runs/nllb_r4_1p3b/final`) | **41.2** | 17.5 | 20.8 |
| offline: stage-3 ASR → NLLB r2 | 39.3 | 14.6 | 20.8 |
| offline: stage-3 ASR → NLLB r0 seed 0 | 38.6 | 14.1 | 20.7 |

Paired bootstrap over windows: r2 − r0 [+0.11, +1.13]; Fable − r2 [+3.7, +5.7]. Runner: `jesus_bench.py --model nllb:<dir>`.

Details in docs/ASR_RESULTS.md; runner `evals/st/jesus_bench.py`.

## NLLB fine-tune on back-translated corpus (sealed sets)

| model | en→brh chrF++ / BLEU | brh→en chrF++ / BLEU |
|---|---|---|
| NLLB-600M zero-shot | 20.7 / 2.0 | 21.0 / 2.9 |
| NLLB-600M run 2 (`runs/nllb_brh_v2/final`) | 34.7 / 9.6 | 44.6 / 24.3 |
| NLLB-600M run 3 + FineWeb-2, 5 ep (`runs/nllb_brh_v3/final`) | 38.3 / 11.9 | 46.9 / 26.4 |
| Claude Opus 5.5 few-shot | 34.9 / 10.1 | 57.2 / 37.8 |

## NLLB hillclimb (2026-10-19, `evals/mt/hillclimb/nllb/ROUNDS.md`; gold minus the 302-pair dev split; chrF++)

| run | en→brh smol (200) | en→brh bouquet (854) | brh→en bouquet (854) | round-trip bouquet | dev en→brh / brh→en |
|---|---|---|---|---|---|
| r0 seed 0 (baseline) | 37.78 | 36.67 | 46.79 | 59.41 | 37.61 / 46.49 |
| r0 seed 1 (noise) | 38.01 | 36.91 | 46.77 | 59.91 | 37.65 / 46.77 |
| r2 + 130k everyday BT pairs (brh→en only) `runs/nllb_r2_bt/final` | 37.62 | 37.14 | 48.20 | 64.02 | 37.87 / 49.99 |
| **r4: NLLB-1.3B, r2 data, 3 epochs** `runs/nllb_r4_1p3b/final` | **39.49** | **38.18** | **49.83** | **65.97** | **39.51 / 51.40** |
| r4 + 1-epoch gold-only adaptation (reverted: round-trip falls, judge tie) `runs/nllb_r4_gold/final` | 41.81 | 39.09 | 49.70 | 64.83 | 40.18 / 51.89 |
| r2 + 3-epoch gold-only adaptation (same verdict) | 40.02 | 38.44 | 48.01 | 62.86 | 39.10 / 49.70 |

Judged adequacy on dev (Opus 5.5, meaning preserved): r0 en→brh 78% / brh→en 41%; r2 72% / 45%. Pairwise r2 vs r0: en→brh tie (p=0.34), brh→en r2 wins 147–73 (p<0.001). Pairwise r4 vs r2 en→brh: r4 wins 150–113 (p=0.026); r4_gold vs r4: tie (p=0.67). Model soup (r0+r2) and MBR/reverse reranking both reverted (worse on dev).
