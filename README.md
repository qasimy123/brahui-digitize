# brahui-digitize

Speech recognition, translation and speech synthesis for **Brahui** (براہوئی, `brh`), a Dravidian language with
about 2.5 million speakers in Balochistan and no commercial language technology. All three run in a browser tab
with no server.

- **Speech recognition** at less than half the error of Meta's released 1B model on Brahui (19.6% → 8.6%), and a
  300M version that reaches 8.0% in the browser.
- **Translation** at more than twice the score of untuned NLLB in both directions (21 → 50 and 21 → 40 chrF++),
  trained almost entirely on machine-made data, and ahead of a few-shot frontier LLM at writing Brahui.
- **A Brahui voice** that a native listener chose four times out of five in a blind test, once the unwritten
  short vowels were recovered from audio.

A native speaker who has used all three judges them good enough to be legible and usable, though not perfect.

Built by one person in six weeks (October to November 2026) for about $150 of rented GPU time and metered API
calls, with no paid annotators. **[`docs/REPORT.md`](docs/REPORT.md)** is the full account: method, results,
failures, costs.

## Results

| | result | for comparison |
|---|---|---|
| Speech recognition, in the browser | 8.0% character error rate on Common Voice test, 315M parameters | Meta's released 1B model 19.6%; community Whisper model 18.9% |
| Translation, Brahui → English | chrF++ 50.2 on the sealed BOUQuET test | untuned NLLB 21.0; few-shot Claude Opus 57.2 |
| Translation, English → Brahui | chrF++ 40.0 on the sealed SMOL hold-out | untuned NLLB 20.7; few-shot Claude Opus 34.9 |
| Meaning fully preserved, Brahui → English | 41% of sentences, strict LLM judge | few-shot Opus 59%; lookup-agent teacher 72%, same set and judge |
| Browser translator | matches its 1.3B teacher at 0.9 s per sentence | distilled for $1.40 |
| Speech synthesis | 9.7% read-back error | real recordings 4.4%; Urdu voice reading Brahui 20.7% |

What I think is worth your time here:

- **How decisions were made.** Four test sets were sealed on day one. Seed noise and LLM run-to-run noise were
  measured before the first comparison, and a change was kept only if its paired confidence interval cleared zero
  on both the development set and a sealed set. Predictions were written down before runs.
  [`evals/mt/hillclimb/nllb/ROUNDS.md`](evals/mt/hillclimb/nllb/ROUNDS.md) is the round-by-round record.
- **A better teacher did not make a better student.** Lookup agents raised the teacher from 35% to 80% meaning
  preserved on spoken-register text. A 1.3B student trained on the retranslated data tied the previous one on every
  measure (round 5). The cheapest test of why, upweighting the retranslated pairs, has not been run.
- **What fine-tuning did inside NLLB.** Brahui was merged into Urdu's representation, not written over it, and
  Urdu → English improved. The embedding rows of the 230,000 tokens never seen in training shrank to 65% of their
  length, and the weight change is fairly low-rank. [`evals/interp/REPORT.md`](evals/interp/REPORT.md).
- **The metric disagreed with the listener, twice.** A voice that scored worse on automatic read-back won a blind
  test 16 to 4. [`docs/TTS_RESULTS.md`](docs/TTS_RESULTS.md).
- **Getting it into a browser.** A 20 MB character language model made a 300M recognizer beat the 1B model it
  learned from, with no training. A deep-encoder, shallow-decoder translator kept its teacher's score at under a
  second. Also: an fp16 overflow fix for the 1B recognizer, and int8 on WebGPU against WebAssembly.
  [`docs/PLAN_SERVE.md`](docs/PLAN_SERVE.md).

## Limits

- **Almost no human validation.** One native speaker checked outputs by ear and finds them usable. Everything else is automatic metrics
  and LLM judges. No linguist or panel of speakers has reviewed the models or the data.
- **Translation is gist-level**, as the 41% above says. My own target was 95%.
- **The training translations are machine-made.** Claude wrote the English side of nearly all the parallel data,
  and Claude is also the judge.
- Speech data is mostly read and broadcast speech. Expect worse results on conversation, phone audio and dialects
  far from Kalat and Quetta.

This is a snapshot of finished work and is not maintained. Corrections from Brahui speakers are welcome as issues.

## What is here

| piece | model | measured | in the browser |
|---|---|---|---|
| Brahui speech recognition, fast | [`brahui-asr-ctc300m-student-onnx`](https://huggingface.co/qkhawaja/brahui-asr-ctc300m-student-onnx): omniASR CTC-300M fine-tuned, plus a character language model | Common Voice test CER **8.0** | 366 MB; 0.2 s for a 3 s clip (WebGPU) |
| Brahui speech recognition, accurate | [`brahui-asr-ctc1b-s3`](https://huggingface.co/qkhawaja/brahui-asr-ctc1b-s3): omniASR CTC-1B fine-tuned on about 80 h | Common Voice test CER **8.6**, Omnilingual test CER **12.0** (untuned: 19.6 / 16.4) | optional, about 2 GB fp16 |
| Brahui ↔ English translation | [`brahui-nllb-1.3b-r5`](https://huggingface.co/qkhawaja/brahui-nllb-1.3b-r5): NLLB-200 1.3B fine-tuned on synthetic pairs | BOUQuET test chrF++ **50.2** to English, **38.2** to Brahui (untuned NLLB: about 21) | 1.4 GB int8 |
| Brahui → English, small | [`brahui-mt-student-c-onnx`](https://huggingface.co/qkhawaja/brahui-mt-student-c-onnx): distilled, 24 encoder / 6 decoder layers | dev chrF++ 51.2, same as its teacher | 1.1 GB; 0.9 s per sentence |
| Brahui speech synthesis | [`brahui-tts-vits-browser`](https://huggingface.co/qkhawaja/brahui-tts-vits-browser): one voice, distilled from an F5-TTS teacher | read-back CER **9.7** (real speech 4.4, Urdu voice reading Brahui 20.7) | 114 MB; 3x real time |

CER is character error rate (lower is better). chrF++ is a character-overlap translation score (higher is better).
The test sets in `data/eval/` are sealed with checksums: nothing was trained, tuned or selected on them.

Datasets:

- [`brahui-synthetic-parallel`](https://huggingface.co/datasets/qkhawaja/brahui-synthetic-parallel): 72,200 real
  Brahui sentences with machine-written English: 19,417 speech transcripts, 45,141 web sentences and 7,642 from
  PDFs. (The report's 23,000 transcript sentences include 3,359 from broadcasts, which are not redistributed.)
- [`brahui-media-pseudolabels`](https://huggingface.co/datasets/qkhawaja/brahui-media-pseudolabels): machine
  transcripts of 160 hours of Brahui broadcasts, as video IDs and timestamps. No audio.

`DATA.md` lists the licence of every file and what was deliberately left out. Models built on NLLB or MMS are
non-commercial (CC-BY-NC-4.0). The voice is modelled on a real, consenting speaker and must not be used to
impersonate anyone.

## The method in one paragraph

A large language model already reads Brahui well enough to act as a teacher. It turns Brahui text that has no
translation (speech transcripts, web text) into parallel data, and does this much better when it can look words up
in dictionaries and a glossed corpus first. A small translation model is trained on that data, back-translation
fills in the other direction, and the result is distilled until it fits in a browser. Speech recognition is a
fine-tune of Meta's omniASR on about 80 hours of open data. Speech synthesis needed the short vowels that Brahui
script leaves out, recovered from audio, before it sounded right to a native listener. The argument is in
`docs/SYNTHETIC_BOOTSTRAP.md` as first written; the report says which parts of it held.

## Try it

```bash
python3 demo/web/serve.py           # http://localhost:8080
```

`index.html` is speech recognition and translation with sample clips. `talk.html` is the spoken two-way demo.
`tts.html` speaks Brahui text. Models are fetched from the Hugging Face Hub and cached by the browser; see
`DEPLOY.md`.

## Results and write-ups

| file | what |
|---|---|
| `docs/REPORT.md` | the whole project in one document |
| `docs/ASR_RESULTS.md` | speech recognition, with confidence intervals |
| `evals/mt/hillclimb/nllb/ROUNDS.md` | translation, round by round, with paired tests |
| `docs/PLAN_SERVE.md` | shrinking the models to run in a browser |
| `docs/TTS_RESULTS.md` | speech synthesis, broadcast audio cleaning, vowels |
| `evals/interp/REPORT.md` | what fine-tuning changed inside NLLB |
| `evals/RESULTS_TABLES.md` | run-by-run numbers (first two tables regenerated by `scripts/summarize_results.py`) |

The `docs/PLAN*.md` files and the rounds logs are working records, written during the work by the coding agent
and kept as they were. In them, "I" is the agent and "the user" is me.

## Layout

| path | what |
|---|---|
| `brahui/text.py` | Perso-Arabic normalizer and romanizer used for every CTC target and every score |
| `data/eval/` | sealed test sets + `SHA256SUMS` (Common Voice, Omnilingual, SMOL, BOUQuET). Never train on these. |
| `data/parallel/` | the gold pairs and the LLM-generated synthetic pairs the MT models were trained on (`DATA.md`) |
| `scripts/train_asr.py`, `scripts/gpu/train_asr.sh`, `student_asr.sh` | CTC fine-tune (1B) and the 300M student |
| `scripts/build_char_lm.py`, `export_char_lm.py`, `ctc_lm_decode.py` | Kneser-Ney character LM + CTC prefix beam search (Python); `demo/web/ctc_lm.js` is the browser port |
| `scripts/train_nllb.py`, `scripts/gpu/train_nllb.sh` | NLLB fine-tune, both directions, sealed-set eval before/after |
| `scripts/build_parallel.py`, `translate_api.py`, `retranslate_chunks.py`, `workflows_*.js` | the synthetic-data teacher (Claude via API or CLI; the `workflows` variants grep dictionaries and a glossed corpus while translating) |
| `scripts/distill_generate.py`, `prune_nllb_layers.py`, `trim_nllb_vocab.py`, `scripts/gpu/distill_*.sh` | sequence-level distillation of the MT students |
| `tools/onnx_export/` | ONNX export, int8/q4/fp16 quantization (fp16 overflow fix for the 1B CTC in `convert_fp16.py`) |
| `scripts/asr_eval.py`, `evals/mt/*.py`, `evals/st/jesus_bench.py` | ASR CER/WER, MT chrF++/BLEU with paired bootstrap, LLM meaning-preservation judge, end-to-end speech-translation benchmark |
| `scripts/build_tts_*.py`, `score_tts_pool.py`, `scripts/gpu/train_f5.sh`, `train_tts_student.sh`, `tools/f5/`, `tools/vits/` | speech synthesis: data selection, F5-TTS fine-tune, distillation to VITS (`docs/TTS_RESULTS.md` section 7 lists every script) |
| `brahui/vowelize.py`, `scripts/vowel_*.py` | restoring short vowels: lookup front end and the vowel-aware recognizer |
| `scripts/download_media.py`, `transcribe_media.py`, `enhance_media_bulk.py` | collecting, transcribing and cleaning broadcast audio (the audio itself is not redistributed) |
| `scripts/export_public.py` | how this public copy is generated from the private working repository |
| `scripts/deploy_web.py` | publish ONNX models to the Hub and the static page as a free HF Space |
| `docs/` | results (`ASR_RESULTS.md`, `PHASE0_RESULTS.md`), plans, GPU notes, and source-linked research on Brahui resources |

## Setup

`uv run --with pytest python -m pytest tests` checks the text normalizer that every score depends on.

Two levels of reproducibility. Loading each released model and re-running its headline evaluation is meant to work
as documented. The training and GPU scripts are published as they were run, with their pitfalls written down, and
may need changes for your machine.

```bash
uv sync
uv run python scripts/download_datasets.py        # Common Voice (needs MDC_API_KEY), Omnilingual, SMOL, BOUQuET, BrahuiSpeech-70H
uv run python scripts/build_manifests.py
cd data/eval && shasum -a 256 -c SHA256SUMS && cd -
uv run python scripts/asr_eval.py --model ctc:facebook/omniASR-CTC-300M --set cv_test   # a baseline
```

GPU jobs are plain shell scripts under `scripts/gpu/` written for a RunPod pod with the repo at `/workspace`;
`docs/GPU_TRAINING.md` has the pitfalls (volume quota, silent CPU fallback, self-terminating pods).

## Built with an AI coding agent

The code and experiments were written with Claude Code, directed by me. Claude models were also the translation
teacher and the judge of meaning preservation. The report discusses what that circularity means for the numbers.

## Credits

Data: Mozilla Common Voice contributors, Meta Omnilingual ASR corpus, Google SMOL and BOUQuET, BrahuiSpeech-70H-V2
(TBOGamer22), FineWeb-2. Base models: Meta omniASR and MMS, NLLB-200, F5-TTS, OpenAI Whisper. Linguistic
references: Bray (1934), Ali & Kobayashi (2024), and the Brahui dictionaries listed in `docs/research/`.
Listening checks by a native Brahui speaker.

Code is MIT. See `DATA.md` for the licence of each data file and each model card for the licence of each model.

Qasim Khawaja, November 2026.
