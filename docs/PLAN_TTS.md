# Brahui TTS (started 2026-10-21)

Goal: a Brahui voice for the demo (`demo/hf_space/`), built with the same recipe as the ASR: take Meta's MMS-TTS
checkpoint for the closest language (Urdu, same script), fine-tune on single-speaker Brahui read speech, evaluate
without humans. MMS checkpoints are CC BY-NC 4.0, so the voice is non-commercial.

## Data (`scripts/build_tts_set.py` -> `data/tts/cv_a37a/`, HF dataset `qkhawaja/brahui-tts-cv`, private)
Common Voice speaker `a37a913d512beb9a` (the largest: 1,062 clips, 2.08 h). BrahuiSpeech (69.5 h) has no speaker
labels, so it is left for a later diarization step. Filters: no Latin letters/digits (6), 1-12 s after silence trim
(33), and **ASR agreement**: stage-3 CTC transcript vs label, CER <= 0.15 (36 dropped; median 0.034, p90 0.103).
Kept 987 clips = 1.64 h; 947 train / 40 dev. Audio: 16 kHz mono, trimmed at 35 dB, 80 ms pad, RMS -20 dBFS.

## Init checkpoint (`qkhawaja/brahui-tts-vits-init`, private)
`facebook/mms-tts-urd-script_arabic` generator + the original MMS discriminator (converted with
finetune-hf-vits' `convert_original_discriminator_checkpoint.py --language_code urd-script_arabic`, run under
transformers 4.46 - the repo's utils break on 5.x). Vocab extended: ء (id 58, embedding copied from ئ) and ڷ (59,
from ل); `<unk>` moved to 60, embedding padded to 62 rows. Without this the tokenizer silently drops both letters
(ء appears 512 times in the speaker's text).

## Training (`scripts/gpu/train_tts.sh`, pod without a volume, data and init from the Hub)
Meta's `finetune-hf-vits` recipe: lr 2e-5, bs 16, 80 epochs = 4,800 steps, fp16, loss weights from the English
example (mel 35, kl 1.5, disc 3). ~2.4 it/s on an A40 ($0.59/h) -> 35 min, ~$0.35. Pitfalls fixed on the way
(all in the script): pyarrow needs numpy 2; matplotlib >= 3.10 breaks the repo's plot helper; the collator sets
`speaker_id=None` for a single speaker and accelerate's `send_to_device` drops it -> KeyError at step 0 (patched with
sed after the clone). `finish.sh` deletes the pod on exit; `HF_UPLOAD_REPO=qkhawaja/brahui-tts-runs` keeps the logs.
Two pods died to the first two pitfalls before a local CPU smoke run (32 clips, 8 steps) caught the rest: smoke-test
the recipe locally before paying for a pod.

## Eval (`scripts/tts_eval.py` -> `evals/tts/<model>/`)
Synthesize the 40 dev sentences, transcribe with the stage-3 ASR, CER against the text = intelligibility. Baseline
Urdu voice reading Brahui: **CER 20.7 % mean / 18.6 % median**; real recordings of the speaker score ~3.4 %.
Audio kept under `evals/tts/<model>/wavs/` for listening.

## Results so far (ASR CER on the 40 dev sentences, lower = more intelligible)
| voice | mean | median |
|---|---|---|
| real recordings of the speaker | 0.034 | |
| Urdu MMS voice reading Brahui (baseline) | 0.207 | 0.186 |
| lr 2e-5, 80 ep, bs 16, fp16 (`qkhawaja/brahui-tts-vits`) | 0.281 | 0.274 |
| lr 2e-5, 200 ep, bs 32, fp16 (`-200ep`) | 0.283 | 0.264 |
| lr 1e-4, 100 ep, bs 32, fp32 (`-lr1e4`) | 0.232 | 0.212 |
| lr 1e-4, 300 ep, bs 32, fp32 (`-lr1e4-300ep`) **deployed in the Space** | 0.214 | 0.199 |

Trend at lr 1e-4: 100 ep 23.2 % -> 300 ep 21.4 % (diminishing). The first 300-ep attempt was lost to a full
30 GB container disk (18 x 1.7 GB checkpoints); the job now keeps 2 checkpoints and salvages on failure.
Lessons: epochs alone did nothing (80 = 200); the learning rate is the lever (mean param change 0.0016 at 2e-5).
Noise is not the CV ceiling: raw CV clips have a 69 dB loud/quiet ratio vs 32 dB for the cleanest BrahuiSpeech
speaker. BrahuiSpeech speaker clustering (`scripts/cluster_speakers.py`, ECAPA + agglomerative, cosine 0.65):
top single speaker = cluster 64, 1,192 clips / 5.3 h, a preacher with background noise (ids in
`data/tts/bs_cluster64_ids.json`); usable only after denoising. `evals/tts/listen/` holds side-by-side wavs.

## Deploy
Push to `qkhawaja/brahui-tts-vits`; the Space reads `TTS_REPO` (default now that repo) and drops the ڷ->ل
substitution when the vocab has ڷ. Next levers: more data from BrahuiSpeech after speaker clustering; F5-TTS
for naturalness if VITS sounds flat.
