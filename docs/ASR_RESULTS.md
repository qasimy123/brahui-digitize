# Brahui ASR: first fine-tune results

Date: 2026-10-11. One A100 80 GB (RunPod, US-WA-1), about 3 h of training. Base model: Meta omniASR CTC-1B v2
in Hugging Face format (`aadel4/omniASR-CTC-1B-v2`), full fine-tune with the pretrained 10k-token head kept.
Training pool: Common Voice train+dev (minus 200 monitor clips) + Omnilingual train chunked to ≤20 s +
BrahuiSpeech-70H: **18,888 clips, 79.6 h**. 5,000 steps, effective batch 32 clips, lr 1e-5 linear decay,
feature encoder frozen, bf16, gradient checkpointing. Best checkpoint by monitor CER (step 4500).

## Headline

| model | CV test CER (read, 882 clips) | Omnilingual test CER (spontaneous, 106 segments) |
|---|---|---|
| Whisper-Small Brahui (community, trained on the same 70 h) | 18.9 [18.3–19.8] | 19.1 [18.3–20.0] |
| omniASR CTC-1B v2, zero-shot | 19.6 [18.9–20.4] | 16.4 [15.9–17.1] |
| **omniASR CTC-1B v2, fine-tuned (stage 1)** | **9.3 [8.8–10.0]** | **12.5 [12.1–13.0]** |
| **stage 2: + 800 steps on CV + Omnilingual only** | **8.6 [8.0–9.2]** | **11.8 [11.3–12.3]** |
| stage 3: stage 2 + 1,000 steps on all 80 h, BrahuiSpeech labels normalized to CV spelling | 8.6 [8.0–9.2] | 12.0 [11.5–12.5] |

WER: CV test 61.3 → 30.2, Omnilingual test 55.6 → 42.8. 95% CIs from 1,000 bootstrap resamples over clips.
Brackets never overlap between zero-shot and fine-tuned, so both gains are unambiguous. Fine-tuning halves the
error of Meta's released 1B model on both sets, and these are the lowest error rates of any system I could run on
them. For context, Meta's results file lists CER 8.2 for Brahui with its 7B LLM-ASR system; it does not say on what
test data, so that figure is not comparable with the ones here.

Monitor set (200 Common Voice dev clips, never trained on): CER 5.9, WER 22.1 for the stage-1 model, measured
with the corrected metric. The sealed CV test is harder than dev (different speakers), hence 9.3 there.

## What the errors are now

Character-level alignment on the monitor set at step 1500 (CER 8.3 at that point):

- 46% substitutions, 28% deletions, 26% insertions.
- Top substitutions are spelling variants of the same sound: ا↔آ, ا↔ہ, و↔ء, ا↔ئ, ے↔ء, ے↔ی, ڈ↔ڑ, س↔ص, ہ↔ح.
- 22% of all errors are word-boundary spaces (90 deleted, 101 inserted), i.e. the model joins or splits words
  where the reference does the opposite.
- Five CV test clips (0.6%) produce an empty hypothesis; the base model produced six. Two of them are single
  words, so these are likely near-silent or clipped recordings.

Very few errors change the consonant skeleton of a word. For the translation step that follows, these are
much less damaging than the base model's errors: in the cascade demo on the same three sealed clips as in
Phase 0, the English output is now semantically right on all three (Phase 0 had one of three wrong).

## Things found on the way (fixed)

- **omniASR CTC collapses past ~20 s of input**: CER 19 at 20 s chunks, 27 at 30 s, 68 at 40 s on the same
  segments. The eval backend now chunks at 20 s. The first Omnilingual baseline (47.9) was an artefact of
  40 s chunks. Fixed chunks still cut words at boundaries; silence-aware chunking would recover a little on
  long spontaneous audio.
- **Trainer metric bug**: padded prediction ids (-100) decoded as a literal `<unk>`, inflating the monitor
  CER printed during stage 1 by about 10 points. The true values were measured independently; the final
  numbers above come from `scripts/asr_eval.py`, which was never affected.
- The HF v2 conversion prefers Pashto letter variants (ټ ګ ړ ډ) zero-shot; fine-tuning removed them within
  the first 500 steps.
- Omnilingual segments (~88 s) were chunked by CTC forced alignment: 264 segments → 1,247 chunks, 6.13 h,
  0 failures.

## Training curve (monitor WER, corrected metric unavailable mid-run)

| step | 0 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| WER | 59.1 | 43.3 | 37.7 | 35.4 | 34.2 | 30.0 | 30.7 | 30.2 | 29.2 | 28.4 | 28.8 |

Most of the gain arrives by step 2500 (about 4 epochs); the rest is the learning-rate decay.

## Stage 2

Short adaptation of the stage-1 model on Common Voice + Omnilingual only (10.0 h, 800 steps, lr 5e-6),
testing whether BrahuiSpeech's transcription conventions were holding the Common Voice score back.

Run 2026-10-18 on one A100 80 GB (pod up 1 h 45 min including two restarts, about $3.10). Training itself took
7 min for 800 steps (3,262 clips, 10.0 h, effective batch 16). Results, same sealed sets and bootstrap CIs:

| model | CV test CER | CV test WER | Omni test CER | Omni test WER | monitor CER |
|---|---|---|---|---|---|
| stage 1 | 9.3 [8.8–10.0] | 30.2 | 12.5 [12.1–13.0] | 42.8 | 5.9 |
| stage 2 | 8.6 [8.0–9.2] | 27.6 | 11.8 [11.3–12.3] | 41.0 | 5.1 |

Stage 2 is better on both sets by about 0.7 CER points and 2 to 3 WER points. The CIs overlap slightly on CV and
not on Omnilingual, so the Omnilingual gain is unambiguous and the CV gain is likely. Most of the improvement arrived
by step 400 (monitor CER 5.9 → 5.2); steps 400 to 800 added 0.1. Empty hypotheses on CV test went from 5 to 6.
The hypothesis holds: BrahuiSpeech's transcription conventions were costing points on the Common Voice and
Omnilingual orthography, and a short pass on in-domain text recovers them without forgetting the 70 h.
**`runs/ctc1b_brh_s2/final` is now the model to use.**

Operational lessons from the run, both fixed in the scripts: a 40 GB volume is too small (each checkpoint is 10 GB
with optimizer state and the trainer deletes the old one only after writing the new one; the run was killed twice at
the quota with no error message), and over SSH the RunPod env vars are not inherited, so `finish.sh` could not find
the pod id to delete; it now reads them from `/proc/1/environ`. The HF upload also failed because the local token is
read-only; create a write token before relying on that path.

## Stage 3: normalized BrahuiSpeech, all 80 h

Hypothesis from stage 2: BrahuiSpeech's spelling conventions, not its audio, were costing points. If so, mapping its
labels onto the Common Voice / Omnilingual spelling and retraining on all 80 h should keep the stage-2 gain *and* the
media-domain exposure. `scripts/build_bs_normalization.py` learns a word-level map from the training manifests
(613 entries, plurality spelling in CV+Omni wins when it holds >= 50% of a lenient-key's mass and >= 3 occurrences;
6.5% of BrahuiSpeech tokens change, e.g. ای/اے→ئے, نے→نی, اس→ئس, ہے→ہی; 568 ambiguous forms left alone). Training:
from the stage-2 model, 1,000 steps, lr 5e-6, effective batch 32, 18,888 clips / 79.6 h, run 2026-10-18 on an A100
PCIe (37 min training + 10 min evals).

| model | CV test CER | CV WER | Omni test CER | Omni WER | monitor CER | empty CV hyps |
|---|---|---|---|---|---|---|
| stage 2 | 8.55 [8.0–9.2] | 27.6 | 11.82 [11.3–12.3] | 41.0 | 5.13 | 6 |
| stage 3 | 8.61 [8.0–9.2] | 27.7 | 11.98 [11.5–12.5] | 41.2 | 5.34 | 4 |

A statistical tie: the CIs coincide on both sets. The normalized retrain holds the stage-2 level but does not beat
it, so the convention gap was the whole stage-2 story and the 70 h add nothing measurable on these two test sets.
Monitor CER dipped to 5.38 at step 250 and recovered only to 5.34, versus 5.13 before.

**Which model to use.** Both are kept. `runs/ctc1b_brh_s2/final` is numerically ahead on everything measured (all
within noise) and is the default for the read-speech demo. `runs/ctc1b_brh_s3/final` had its final gradient steps on
70 h of media audio in the target orthography, which is the domain the transcription run (PTV Bolan, talk shows)
will hit and which neither sealed set measures; it is the better bet for pseudo-labeling that audio. The map has not
been human-checked; the synthetic QA pass (LLM proposes a correction, the CTC model's forced-alignment score accepts
or rejects it, `docs/SYNTHETIC_BOOTSTRAP.md`) is how wrong entries get found. What settles stage 2 vs stage 3 is a
media-domain benchmark: an LLM-judged end-to-end set over PTV Bolan built the same way as the JESUS film benchmark
below, which needs no transcriber.

Cost of the stage-3 test: about $5.60 of GPU, of which $3.40 was waste (a cu130 torch falling back to CPU on a
CUDA 12.8 host for 40 min; a 50 GB network volume filling at the third 11.7 GB checkpoint, after which RunPod
terminated the pod and the surviving checkpoint was then lost to a careless cleanup command). Guards added:
`train_asr.py` refuses to run without CUDA, job scripts preflight the GPU, checkpoints are weights-only (3.7 GB),
`scripts/gpu/watchdog.sh` alerts on a stalled step counter, and `finish.sh` reads the pod id from `/proc/1/environ`
so the self-delete fires (it did, on the successful run). Data and models now live on network volume `<volume-id>`.

## Reproduce

`docs/GPU_TRAINING.md`. Artefacts: `runs/ctc1b_brh/final` (stage 1), `runs/ctc1b_brh_s2/final` (stage 2, demo default) and `runs/ctc1b_brh_s3/final` (stage 3, for media-domain transcription; HF Wav2Vec2ForCTC + processor, 3.9 GB each),
`evals/asr/runs/{ctc_final,ctc_s2,ctc_s3}__{cv_test,omni_test}/` (per-clip results), `logs/gpu/`.
Use it: `uv run python scripts/asr_eval.py --model ctc:runs/ctc1b_brh_s2/final --set cv_test` or
`uv run python scripts/demo_cascade.py --sample 3`.

## Next for ASR

1. Pseudo-label new audio (PTV Bolan, YouTube talk shows) with this model (done 2026-10-18, below); correct the
   labels synthetically: an LLM proposes fixes from context, the CTC model's forced-alignment score gates them, and
   the result trains stage 4. Spontaneous speech is where the model is weakest and where this loop pays most.
2. Silence-aware chunking in the eval/inference backend for long recordings.
3. A small Brahui character LM for shallow fusion, which targets exactly the spelling-variant and spacing
   errors that remain.
4. Synthetic speech as training data: once a Brahui TTS exists, the 1.1M words of FineWeb-2 Brahui become read
   speech in many voices (ASR⇄TTS cycle, `docs/PLAN.md` §3.3). Human recordings add a voice and a new speaker when
   available, but the data loop does not wait for them.

## Media transcription run (2026-10-18)

Stage 3 run over untranscribed Brahui media on the M1 (MPS, 5–11x realtime): `scripts/download_media.py`
(sources in `data/media/sources.json`, yt-dlp, original audio track pinned via `format_note=original` because
YouTube auto-dubs some channels into English) then `scripts/transcribe_media.py` (energy VAD, ≤20 s chunks cut at
the quietest point, batched greedy CTC, per-chunk confidence = mean max-softmax over emitting frames).
Output: `data/media/transcripts/<source>/<id>.{jsonl,txt,srt}`; audio is gitignored (research use only, never redistribute).

| source | files | audio h | chunks | mean conf |
|---|---|---|---|---|
| Balochistans Today news bulletins | 30 | 3.09 | 764 | 0.929 |
| PTV Bolan Current Affairs Time (Brahvi) | 5 | 3.77 | 845 | 0.857 |
| PTV Bolan programs (Platform 3, Brahui Time, Ulfat) | 5 | 3.52 | 899 | 0.830 |
| Brahui TV interviews | 6 | 1.60 | 358 | 0.888 |
| Nazir Shakir interviews | 3 | 0.81 | 183 | 0.893 |
| Rabita Forum talk show | 2 | 0.75 | 183 | 0.904 |
| JESUS film, Brahui dub (full) | 1 | 2.13 | 681 | 0.800 |
| **total** | **52** | **15.7** | **3,913** | |

Read news scores highest, multi-speaker panel shows lowest, as expected. The JESUS film also has time-aligned
English and Urdu subtitles (`data/raw/jesusfilm_brh/subtitles/{en,ur}.vtt`, 1,262 / 1,322 cues, same 2:07:42 cut),
so its ASR output windowed by cue gives rough Brahui/English/Urdu triplets (spot check at 10:00: census, Bethlehem,
Mary, Joseph all present in the Brahui). The labels are unchecked pseudo-labels; the planned QA is the synthetic
correction pass above, and the SRT files double as a listening aid for any later human spot check.

## Speaker-free end-to-end benchmark: JESUS film (2026-10-18)

`evals/st/jesus_bench.py`. The film's human English subtitles are the reference; the pipeline is ASR (stage 2 or 3)
-> Claude Fable 5.1 (`claude -p`, v0-style plain prompt) -> English, scored by chrF++ over 156 windows of ~40 s cut
at VAD chunk boundaries (both ASR systems share the chunking). The human Urdu subtitles translated by the same model
give the ceiling: English and Urdu subtitles are independent free translations of the dub, so a perfect transcript
cannot score 100 here.

| input to Fable | chrF++ | BLEU | off-by-one control |
|---|---|---|---|
| Urdu subtitles (ceiling) | 47.4 | 22.3 | 21.8 |
| stage-3 ASR, film context in prompt | 43.9 | 19.6 | 21.3 |
| stage-2 ASR, film context in prompt | 43.9 | 19.3 | 21.4 |
| stage-3 ASR, blind (no film context) | 42.7 | 17.6 | 21.2 |

Readings. (1) The transcript carries the meaning: aligned scores are double the off-by-one control (hypothesis
scored against the neighbouring window's reference). (2) ASR errors cost ~3.5 of a possible 47 points; the LLM
recovers most of the meaning from noisy Brahui. (3) Stage 2 vs stage 3 is a tie again: paired bootstrap of the
difference −0.55 to +0.66. The film does not decide between them; a PTV-domain benchmark of the same speaker-free
kind (LLM-judged adequacy over talk-show windows) would.
(4) Fable's knowledge of the Gospel adds only ~1.2 points (context vs blind), and spot checks show it translating
the noisy text rather than reciting (ref "Blessed are you poor" -> ASR lost "poor" -> hyp "you who seek faith, you
are fortunate"). (5) Windows with ASR confidence ≥ 0.85 score 43.7, below 0.85 score 40.5: confidence is a usable
quality filter. Artefacts: `evals/st/runs/{s2_fable,s3_fable,s3_fable_blind,ur_fable}/`, stage-2 transcript in
`data/media/transcripts_s2/`.

Fable 5.1 on the sealed MT sets (`evals/mt/hillclimb/baseline_fable`, v0 prompt, test split): brh->en chrF++ 56.3
(Opus 5.5: 57.2), en->brh 35.5 (34.9). Same model-lever result as Opus; no gain from the newer model.

Reverse direction (`--system en2brh:<jsonl>`, Opus 5.5): English subtitles -> Brahui, scored against the stage-3
ASR text of the dub (normalized). chrF++ 27.0; controls: off-by-one 17.0, Urdu-subtitles-vs-ASR 16.7, stage-2-vs-
stage-3 ASR of the same speech 85.6. The generated Brahui is only ~10 points above chance against the dub, but the
dub is itself a free paraphrase (ref "Blessed are you poor" was dubbed as "you who seek faith are fortunate"), so
this undercounts a faithful translation; the sealed set (en->brh chrF++ 35) remains the better measure. A Fable
en2brh run hit an unexplained IndexError on 76/156 windows and was abandoned (`evals/st/runs/en2brh_fable`).

## NLLB back-translation MT (2026-10-18/19)

*Superseded 2026-10-19 by the hillclimb rounds in `evals/mt/hillclimb/nllb/ROUNDS.md`: current best is NLLB-1.3B round 4 (en→brh 38.2 BOUQuET / 39.5 smol, brh→en 49.8). The section below is the run-2/run-3 history.*

`scripts/build_parallel.py` translated every real Brahui segment we hold (BrahuiSpeech, CV, Omnilingual transcripts;
media chunks at ASR confidence ≥ 0.85; 22,776 segments, 680k words, sealed test sentences excluded) into English with
Claude Opus 5.5 via `claude -p` (first 1,980 with Fable 5.1; 68 min at concurrency 6). Together with 1,992 gold pairs
(SmolSent/SmolDoc train, BOUQuET dev) that is the training set for `scripts/train_nllb.py`: NLLB-200-distilled-600M,
both directions in one model, Brahui tagged as `urd_Arab`, 3 epochs, lr 1e-4, bs 32, label smoothing 0.1, fp32
weights + bf16 autocast, 15 min on one A100 (`scripts/gpu/train_nllb.sh`; ~$2.50 of GPU over two runs).

| model | en→brh chrF++ / BLEU (smol_heldout) | brh→en chrF++ / BLEU (bouquet_test) |
|---|---|---|
| NLLB-600M zero-shot (Brahui as Urdu) | 20.7 / 2.0 | 21.0 / 2.9 |
| run 1 (bf16 weights, bug: loss stuck at 4.55) | 19.4 / 1.8 | 31.5 / 11.1 |
| **run 2 (fp32 weights)** `runs/nllb_brh_v2/final` | **34.7 / 9.6** | **44.6 / 24.3** |
| **run 3** (+ FineWeb-2 corpus, 5 epochs) `runs/nllb_brh_v3/final` | **38.3 / 11.9** | **46.9 / 26.4** |
| Claude Opus 5.5, 15-shot prompt | 34.9 / 10.1 | 57.2 / 37.8 |
| Claude Fable 5.1, same prompt | 35.5 / 9.1 | 56.3 / 36.9 |

Readings. (1) Back-translation works for the direction the LLM is worst at: the 600M offline model matches the
LLM teacher on en→brh (34.7 vs 34.9) although it never saw an LLM-written Brahui sentence; its target side is 100%
real Brahui, so teacher errors are source-side noise, which seq2seq training tolerates. (2) brh→en is a first-pass
distillation at 23k pairs and sits 12.6 under the single-pass teacher; that gap is closed from both ends, by more
pairs and by a better teacher (paragraph context, consensus of several samples), and iterated back-translation from
human English text lets the student pass a single-pass teacher (`docs/SYNTHETIC_BOOTSTRAP.md`). (3) Training loss
was still falling at the end (3.35 at step 4,550; 3 epochs), so more epochs, the 1.3B model and the 1.7M-word
FineWeb-2 Brahui slice were the next levers (FineWeb and epochs are now done, below). (4) Run 1 is a pitfall worth remembering: loading the model in bf16
and training it directly with AdamW makes updates round to zero; the loss sat at 4.55 from epoch 1 and the en→brh
output stayed Urdu. Per-sentence outputs: `evals/mt/nllb/nllb_brh_v2_{before,after}/`.

**FineWeb-2 step (2026-10-19).** `scripts/build_fineweb_units.py` extracts 47,106 Brahui sentences (1.1M words) from
the FineWeb-2 `brh_Arab` slice with a per-sentence function-word filter (the slice is mixed Brahui/Urdu; the old
"mostly Urdu" note was wrong at sentence level). `scripts/translate_api.py` translated them with Opus 5.5 through
the Message Batches API: fixed 40-shot block cached (1 h TTL), 40 sentences per request, strict JSON; 1,173
requests finished in 15 min, 35 failed (refusals, off-by-one counts), 45,706 pairs for $36.58 all-in ($0.74 per
1,000 sentences). Run 3 = run 2 recipe on 70,474 pairs for 5 epochs (64 min A100, ~$2): en->brh **38.3** chrF++,
3.4 above the Opus few-shot teacher-direction score and 17.5 above zero-shot; brh->en 46.9. Loss was still falling
(2.53 at the end), so epochs remain a lever; the news-domain data moved the written-prose test set most. Two
corpus-building rounds, under $40 of API credit and $5 of GPU, took en→brh from 20.7 to 38.3 and past every LLM
measured. Nothing in the recipe is used up: the levers and the expected size of each are in
`docs/SYNTHETIC_BOOTSTRAP.md`.
