# ASR hillclimb (started 2026-10-19)

## Noise floor
Per-clip paired bootstrap of stage 3 minus stage 2: CV test [−0.09, +0.22] CER, Omni test [+0.07, +0.27] (stage 3 is
a small real regression on spontaneous speech that the unpaired CIs reported as a tie). A change counts when the
paired CI excludes 0 on both sets; practically > 0.2 CER on CV and > 0.3 on Omni. JESUS e2e noise: ±0.6 chrF++.

## Sealed media set v1 (`data/eval/media_test.jsonl`, 284 chunks, 68.8 min, 9 held-out recordings)
Built by `scripts/build_media_testset.py`: stage-3 pseudo-labels, Opus 5.5 proposes corrections from the stage-2 /
stage-3 / Whisper hypotheses plus context (edited 108 of 284 chunks, median edit 1 character), forced-alignment gate
under stage 2 and stage 3 accepts 69 and rejects 39. Scores (`evals/asr/media_controls.py`):

| system | CER | 95% CI | off-by-one control |
|---|---|---|---|
| stage 3 (the labeler) | 1.11 | [0.95, 1.32] | 84.8 |
| stage 2 | 4.61 | [4.26, 4.97] | 85.2 |

**Verdict: broken as a quality measure, kept as a regression guard.** 97% of the label characters are stage-3
output, so the labeler scores near zero by construction and a better model cannot score higher (the "stronger
model does not win → fix the eval" rule). The off-by-one control shows the labels are chunk-specific, and the
held-out file list (`data/eval/media_test_files.txt`) is the useful part: no future training touches those
recordings. A media *quality* number needs labels independent of the CTC family; until then media-domain quality is
measured end to end on the JESUS film (human English subtitles) and on the Omnilingual spontaneous set. The
"< 10 CER on a sealed media set" target is therefore not claimable from v1 and is recorded as unmeasured.
