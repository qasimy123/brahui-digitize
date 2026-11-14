# Plan: cheap serving, ideally fully in-browser

Goal (2026-10-24): serve brh↔en MT (NLLB-1.3B r5) and Brahui ASR (MMS-1B CTC s3) at near-zero cost while keeping
quality, ideally as a static page that runs the models locally (transformers.js on ONNX Runtime Web, WebGPU/WASM).
TTS is parked.

## Step 1 — free compression on the Mac (done 2026-10-24)

| model | change | params | int8 size | quality | CPU speed (M1) |
|---|---|---|---|---|---|
| NLLB r5 | baseline fp32 | 1.37B | 5.5 GB fp32 | dev brh→en chrF++ 51.14 | — |
| NLLB r5 | vocab trim 256k→31.7k (`scripts/trim_nllb_vocab.py`) | 1.14B | 4.6 GB fp32 | 51.11 | 3.5 min / 302 sents (MPS) |
| NLLB r5 trim | + int8 ONNX, merged decoder (`tools/onnx_export/`) | 1.14B | **1.42 GB** (enc 640 + dec 779 MB) | **51.2** | 0.72 s/sent, beam 4, batch 8 |
| CTC s3 | baseline fp32 | 965M | 3.9 GB fp32 | cv_test CER 8.6 | — |
| CTC s3 | int8 ONNX | 965M | **1.03 GB** | CER 8.38 vs 8.2 fp32 on the same 200 clips | 1358 s audio in 204 s (6.7× realtime, CPU) |

Browser check (Chrome, M1, WASM backend, transformers.js 3.5.2): the int8 trimmed r5 loads from a static server in
5.6 s (local network) and translates a dev sentence in 4.9 s with beam 4, output identical in meaning to the server
run. So the whole in-browser path works today; what is missing is only the size.

Takeaways: vocab trim and int8 are free (no measurable loss). But 1.4 GB + 1.0 GB is a desktop-only download;
the browser target needs ≤100–300 MB per model, i.e. 5–10× fewer parameters. That is a distillation problem.

Artifacts: `runs/nllb_r5_trim/final`, `export/nllb_r5_trim/`, `export/ctc1b_s3/` (transformers.js layout),
`demo/web/index.html` (static page; symlink the export dirs under `demo/web/models/`).

## Step 2 — distillation (needs GPU, ~$15 RunPod left → one L40S run each, ≈4–6 h total)

### MT student
Data: sequence-level KD. Teacher = r5 (trimmed, fp16 on GPU, ~100 sent/s batched).
1. Brahui-side inputs: gold_train_nodev + retranslated (23k) + fineweb_units/more_units (~50k monolingual brh)
   + synthetic brh produced by r5 en→brh from English monolingual (en_everyday, FineWeb units) — this is how to get
   to ~500k diverse brh inputs; Brahui monolingual alone is the bottleneck.
2. Teacher-translate all of it brh→en with beam 4 → distilled pairs. Keep human/gold pairs as-is (they are the
   references the teacher was trained toward).
Student options, in order of preference:
- **A. Layer-pruned r5 (init from teacher)**: keep 6 encoder / 3 decoder layers of the trimmed r5 (every 4th /
  every 8th layer), fine-tune on distilled data. ~233M params → ~230 MB int8, ~120 MB q4. Fastest to converge
  because weights already encode Brahui. Try first.
- **B. Transformer-base from scratch** (d=512, ffn=2048, 6/6, shared 31.7k vocab, SentencePiece reused):
  ~60M params → 60 MB int8 / 30 MB q4. Needs more distilled data (≥500k) and longer training; the real
  mobile-friendly end state. Marian/OPUS-style, trains fine in HF `MarianMTModel` or fairseq.
Gates: dev brh→en chrF++ within 1.0 of 51.2 (paired bootstrap, `evals/mt/pairwise_judge.py` for MP spot-check).

### ASR student
Teacher = s3. Data = gold ASR train + 15.7 h media pseudo-labels (already transcribed by s3).
- **A. Layer-pruned s3**: keep the bottom 24 of 48 transformer layers (CTC heads tolerate this well), re-fine-tune
  CTC ~1 epoch. ~500M → 500 MB int8. Cheap first check.
- **B. wav2vec2-XLS-R-300M / MMS-300M** CTC fine-tune on the same data: ~315M → 300 MB int8. Likely the sweet spot.
- **C. wav2vec2-base (95M)**: ~95 MB int8, only if B's CER is still close to 8.6.
Gates: cv_test CER within +1.0 abs of s3 (same 200-clip seeded subset), omni_test CER within +1.5.

### Run log
- **2026-10-25, pod (A40 secure, $0.59/h)**. Teacher pass: 221,158 Brahui sentences → English with the
  trimmed r5 (fp16, beam 4) in 65 min (`data/distill/teacher_brh2en.jsonl`). Student A = teacher pruned to encoder layers
  [0,5,9,14,18,23] / decoder [0,12,23] (234M), 3 epochs, lr 2e-4, bs 32: dev brh→en chrF++ 38.7 → 41.0 → 41.7
  (teacher 51.2). Curve still rising at the end of the schedule, so queued on the same pod (`scripts/gpu/distill_chain.sh`):
  student_a_cont (6 more epochs, lr 1e-4), student_b (12/6 layers, ~434M, 4 epochs), then the ASR student.

### Students so far (dev brh→en chrF++, teacher 51.2) and browser latency (Chrome on M1, WASM, beam 4, 20-word sentence)

| student | layers enc/dec | params | int8 download | dev chrF++ | int8 ONNX chrF++ | CPU s/sent | browser |
|---|---|---|---|---|---|---|---|
| A | 6/3 | 234M | 508 MB (312 MB with int8 embeddings, same chrF) | 41.7 (43.1 after 9 ep) | 41.6 | 0.13 | 0.94 s single-thread, **0.41 s** with 8 threads |
| B | 12/6 | 434M | 514 MB (int8 embeddings) | 46.9 | 46.9 | 0.24 | **0.73 s** with 8 threads |
| B v2 | 12/6 | 434M | 514 MB | 47.2 | **47.4** | 0.23 | **0.69 s** with 8 threads |

Findings: (1) int8 is free at every size; (2) greedy decoding saves nothing in the browser (per-step overhead dominates)
and costs 0.3 chrF, keep beam 4; (3) WebGPU int8 is *slower* than WASM on the M1 (1.55 s warm); (4) multi-threaded
WASM needs cross-origin isolation — `demo/web/serve.py` sends COOP/COEP locally and `coi-serviceworker.min.js` emulates
them on static hosts; (5) the 6/3 student saturates at ~43 after 9 epochs, so the gap is capacity/data, not schedule; (6) B v2 = B continued
3 epochs (lr 1e-4) on the teacher pairs + 200k back-translated Tatoeba sentences (`scripts/gpu/distill_chain2.sh`,
`data/distill/bt_en2brh.jsonl`): dev 47.03/47.31/47.24 by epoch, +0.5 over B, flat after epoch 2, so more BT data
would need a bigger student; (7) WebGPU int8 is not just slow, its MT output is garbled (`during liked mail ahead saw
dis Sidwide ends ...` for a normal sentence) on the M1, so the demo now defaults to WASM.

### ASR student result (2026-10-25)
omniASR-CTC-300M fine-tuned 4000 steps (lr 5e-5, bs 8x4) on brahuispeech_train_norm + 10.4 h of s3 media pseudo-labels
(`scripts/gpu/student_asr.sh`, `runs/ctc300m_student/final`, 1.2 GB fp32 → **366 MB int8**, `export/ctc300m_student/`).

| model | params | int8 download | cv_test CER (full) | omni_test CER | int8 CER (200-clip subset) | CPU RTF (ORT) | browser, 8.9 s clip |
|---|---|---|---|---|---|---|---|
| s3 (teacher) | 1B | 1.03 GB | 8.6 | 12.0 | 8.4 | 0.15 | 7.4 s |
| 300M student | 315M | 366 MB | 9.9 | 12.6 | 9.5 | 0.066 | **2.8 s** (0.31x realtime) |

Gate: cv_test +1.3 abs (gate was +1.0, slightly missed), omni_test +0.6 (passes). Browser transcript of the sample clip differs from the
Common Voice reference in 3 characters out of ~55 (ئ/ا, a space in بددعا, final ہ/ے). A 3-second spoken sentence therefore transcribes in under a second on 8-thread WASM.

### ASR in the browser (2026-10-25)
The 1B int8 CTC export (1.03 GB) loads via transformers.js once a `tokenizer.json` is generated for the CTC vocab
(`tools/onnx_export/build_ctc_tokenizer_json.py`; the CTC blank must be the config's pad token, `<s>` id 0 for
omniASR/MMS, not `<pad>`). Common Voice clip, 8.9 s: correct transcript in 7.4 s on 8-thread WASM (0.83x realtime).
The 300M student should be ~3x faster; WebGPU is the other lever since CTC is one batched pass, not a token loop.

Open: WebGPU. int8 on WebGPU is slower than WASM (MT 1.55 s, ASR 7.7 s); an fp16 export (`convert_fp16.py`, merge the
fp32 decoders first) fails to load in transformers.js 3.5.2 ("load failed: undefined"). Not needed on 8-core machines;
revisit for phones/tablets with q4 (`quantize_q4.py`, MatMulNBits) which WebGPU runs natively.

## Step 3 — ship
- Static page: `demo/web/` → GitHub Pages or an HF *static* Space (free; no PRO needed). Models hosted on the
  HF Hub (public repos; the page fetches them with `env.remoteHost`), cached by the browser after first load.
- WebGPU first, WASM fallback; int8 (`dtype: 'q8'`) for WASM, fp16/q4 for WebGPU once students exist.
- Fallback for weak devices: the int8 ONNX models above on a free HF CPU Space or a $5 VPS (CTranslate2 would be
  ~2× faster still for NLLB on CPU).

## Pitfalls hit
- optimum 1.25 needs transformers <4.52 → separate venv `tools/onnx_export/`.
- torch ≥2.6 → dynamo ONNX exporter → fails on external-data files; pin torch 2.5.1.
- Merged fp32 decoder of a 1B model exceeds protobuf's 2 GB → export with `--no-post-process`, quantize the two
  decoders, then `merge_decoders` the int8 pair (deduplication still works; merged = 779 MB vs 773+722).
- onnxruntime's quantizer writes external data as `<f>.onnx.data`; transformers.js only finds `<f>.onnx_data` →
  `fix_external_data.py` rewrites the location (now part of export_quant.py). The page must pass
  `use_external_data_format` per file (true for files that have a `_data`, false for the merged decoder).
- transformers.js caches model files in the Cache API; after re-exporting, stale files are served from cache —
  the page disables the cache on localhost.
- transformers 4.x cannot parse the 5.19 NLLB tokenizer_config → eval scripts use the raw `tokenizers` file.

### Final pick (2026-10-25) against the goal "runs in the browser, locally, under a second per sentence"
Shipped in `demo/web/` (defaults: MT `student_b_v2`, ASR `ctc300m_student`, WASM 8 threads):

| | download | quality | browser latency (Chrome, M1, 8 threads) |
|---|---|---|---|
| MT brh→en **student_c** int8 (24/6) | 1.1 GB | dev chrF++ **51.17** (teacher 51.2) | **0.88 s** / 20-word sentence |
| MT brh→en student_b_v2 int8 (12/6, small option) | 514 MB | dev chrF++ 47.4 | 0.69 s / 20-word sentence |
| ASR ctc300m_student int8 | 366 MB | cv_test CER 9.5 (1B 8.4) | 2.8 s / 8.9 s clip; **0.84 s** for a 2.8 s sentence clip (`?clip=cv_short.mp3`, output = reference minus one final diacritic) |

All gates met with student_c (quality gate: within 1.0 chrF of the teacher, actually -0.03). The 12/6 student stays as
the small-download option.

Pod pitfall (2026-10-25): two chains on one pod each used the same `/workspace/.hold`; when chain2 exited its trap
removed the flag and the *other* chain's finish.sh (already past its 20-min grace) deleted the pod within a minute,
before student_b_v2 could be pulled. Recovered via a 10-min pull pod on the volume ($0.10). Fixed: per-job
`/workspace/.hold.<job>` flags, finish.sh waits while any exists.

### Student C: deep encoder / shallow decoder (2026-10-25) — the final MT model
Browser MT latency is decoder-bound: int8 encoders on the 39-token test sentence take 28 ms (12 layers), 63 ms (24 layers),
71 ms (teacher) natively, i.e. the 24-layer *decoder* is what makes the teacher ~2 s. Student C keeps all 24 encoder
layers and 6 decoder layers [0,5,9,14,18,23] (687M, ~750 MB int8), trained 2 epochs (lr 1e-4) on distill_pairs_v2
(`scripts/gpu/distill_chain3.sh`, pod L40S $1.09/h, ~6 it/s, 26k steps). Result: dev chrF++ 49.67 → **51.02** by epoch (teacher 51.2), int8 ONNX **51.17**, CPU 0.25 s/sent,
browser **0.88 s** for the 20-word sentence on 8-thread WASM, output at teacher quality. 1.1 GB int8 download. Run cost ~$1.40.
This is the shipped default (`demo/web` mt=student_c). Download size is the remaining lever (q4 / fp16 embeddings).
Pitfalls hit: the pod venv lives on container disk (rebuild with `UV_PROJECT_ENVIRONMENT=/root/venv uv sync`, 22 s);
the volume quota bit again at 44 GB (freed 9 GB of byte-verified, already-pulled student runs; keep ≤35 GB).

### ASR speed solved by WebGPU fp16 (2026-10-25)
Profile of the int8 CTC model (7 s audio, 8 threads, native ORT): DynamicQuantizeMatMul 29%, Conv 26%, MatMulIntegerToFloat
11%, LayerNorm 9%. Quantizing Conv too: 4% faster, CER 9.45 → 10.6, discarded (`model_q8conv.onnx`). The win is WebGPU with
an fp16 export (`onnxruntime.transformers.float16.convert_float_to_float16`, keep_io_types, single file so no merge issue):
8.9 s clip **0.52 s** (WASM int8 2.7 s), 2.8 s clip **0.20 s**, transcripts identical to int8; native CER on the 200-clip subset 9.32 (int8 9.45, so fp16 is lossless). Unlike the MT decoder,
CTC is one batched pass, so WebGPU's per-kernel overhead is amortised and fp16 is numerically fine. The page now picks
webgpu/fp16 for ASR when `navigator.gpu` exists (`?asrdev=wasm&asrdtype=q8` to override) and keeps WASM int8 for MT.

### ASR quality gate closed without training: the full 1B (s3) in fp16 on WebGPU (2026-10-25)

With no GPU credit for the 300M continuation, the other way to meet the gate is to serve s3 itself. `export/ctc1b_s3/onnx/model_fp16.onnx` (1.95 GB, single file) runs on WebGPU in Chrome on the M1:

| clip | 1B s3 fp16 | 300M student fp16 |
|---|---|---|
| 2.8 s (cv_short) | 0.46-0.48 s | 0.20 s |
| 8.9 s (cv_sample) | 1.59-1.67 s | 0.52 s |

So the 1B is ~0.18 s per second of audio: a sentence up to ~5.5 s transcribes in under a second, a 7 s cv_test-median clip in ~1.25 s. Quality is s3's by construction (CER 8.6 full cv_test); 200-clip cv_test subset CER: 1B fp16 8.19 (1B int8 8.38, 300M fp16 9.32, 300M int8 9.45); the two empty hypotheses are the same two clips the int8 build also leaves empty, so not an fp16 artefact. Results in `evals/asr/runs/onnx_ctc1b_s3_fp16__cv_test_s200` (gitignored).

Pitfall (cost ~1 h): the straight fp16 conversion loads and runs on WebGPU with no error but every logit is NaN, while the same file gives correct text under ORT's CPU EP (which upcasts internally). Scanning the fp32 graph's intermediates on a clip found one tensor over fp16 range: layer-0 attention scores peak at 1.1e5 (fp16 max 65504). Fix = pin that MatMul and its Softmax to fp32 (`convert_fp16.py --fp32-node ...`), which needs path-based shape inference (model > 2 GB) and collapsing the converter's fp32->fp16->fp32 cast round-trip between the two pinned nodes (it clips to inf otherwise). Keeping all 105 LayerNorms and 48 Softmaxes in fp32 instead was no slower, but no faster either.

Idle re-timing (nothing else on the GPU): 8.9 s clip 1.22-1.30 s, i.e. ~0.14 s per audio-second, so on this M1 the 1B is under a second up to ~7 s of audio: the 200-clip cv_test subset has median 6.9 s / p75 8.9 s, so the median sentence lands at ~0.96 s and the upper quartile at ~1.2 s.

Silence trim: CTC cost is linear in length and CV clips carry ~0.7-0.9 s of silence. `scripts/asr_eval.py --trim DB` and `trimSilence()` in the demo drop frames below peak-DB (25 ms frames, 0.1 s pad). s3 on the 200-clip subset: untrimmed CER 8.20, 50 dB 8.29 (median clip 6.9 -> 6.2 s), 35 dB 8.52 (-> 6.0 s). Demo uses 50 dB (`?trim=0` disables); worth ~0.1 s on a typical clip, not a game changer.

MPS training (`train_asr.py --mps`, benchmarked 2026-10-25): 5.4 s per 4-clip step on the M1, so the 6000-step continuation (32 clips/step) would take ~70 h. Not a substitute for an hour on an L40S.

Demo: `demo/web/index.html` now has an ASR picker; default is `ctc1b_s3` when `navigator.gpu` exists, else `ctc300m_student` on WASM. `?asr=` and `config.js` still override.

### ASR gate closed for the fast model, no training: CTC beam search + character 6-gram LM (2026-10-25)

The 300M student's remaining gap to s3 was a decoding problem, not (only) an acoustic one. A Kneser-Ney character 6-gram LM
over 111k Brahui sentences / 12 M chars (real rows of `data/distill/brh_pool.jsonl`, BrahuiSpeech / CV / omni train transcripts,
s3 media pseudo-labels; 491 sentences equal to cv_dev / cv_test / omni_test transcripts removed) plus CTC prefix beam search
(beam 16, 8 candidates per frame):

| decode | cv_dev (250 clips) | cv_test (882) |
|---|---|---|
| 300M greedy | 4.64 | 9.89 |
| 300M beam + LM (alpha 0.5, beta 0.5) | 3.58 | **8.00** |
| 1B s3 greedy (reference) | - | 8.61 |

So the fast model is now 0.6 CER *better* than the 1B teacher on cv_test and well inside the gate (<= 9.6). Tuned on dev only
(alpha 0.5-0.7, beta 0-0.5 are all within 0.03); the pruned browser table (count >= 2 from order 3, 1.6 M of 2.5 M n-grams,
19.7 MB `demo/web/models/brh_char6.blm`) reproduces the full model within 0.1 CER.

Browser: `demo/web/ctc_lm.js` (backoff table with FNV-hashed keys in typed arrays, prefix beam search in JS) is the default ASR
path. The 300M fp16 on WebGPU + LM decode: 2.8 s clip 0.21 s, 8.9 s clip 0.65 s. The 1B stays in the picker as the slower option.
Scripts: `ctc_dump_logits.py` (log-probs per clip) -> `build_char_lm.py` (counts) -> `export_char_lm.py` (.blm) and
`ctc_lm_decode.py` (Python twin of the JS decoder, `--grid` for alpha/beta, `--lm-bin` to score the exact browser table).
Not yet tried: the same LM on the 1B's logits (expect ~7.x), word-level LM, lexicon constraints.

## Status 2026-10-25 (end of session)

- MT: done (student_c, chrF++ 51.17, 0.88 s in browser).
- ASR speed: done (WebGPU fp16, ~0.5 s per 7 s clip).
- ASR: met with the 300M student + char-LM beam search: cv_test CER 8.00 (gate <= 9.6, 1B teacher greedy 8.61), 0.21 s for a 3 s sentence and 0.65 s for a 9 s clip on WebGPU fp16. The 1B (CER 8.2 subset greedy, 0.14 s per audio-second) remains the slower picker option. The 300M student (CER 9.9, +1.3 vs gate +1.0) stays as the fast/WASM option; its 6000-step continuation (`student_asr.sh` with BASE/RUN/STEPS/LR/TAG env, ~1 h on an A40/L40S) is still the lever to make the fast option also pass the gate.
- RunPod balance is exhausted (402 on create-pod); no longer blocking the goal. The H100 pod never became reachable and billed $7.53 before vanishing. Rule going forward: delete pods with no runtime after 10 min; use A40/L40S.

## Spoken two-way demo (2026-10-25, `demo/web/talk.html`)

User ask: speak English -> Brahui script + romanized; speak Brahui -> English. Built on the same local stack plus Whisper base.en
for English ASR and NLLB r5 (trimmed, int8) for English->Brahui (student_c only translates brh->en: on English input it echoes).
Language auto-detect = Brahui char-LM score of the Brahui CTC's greedy output (Brahui -1.3..-2.1 nats/char vs English -2.7..-4.5
on 8 macOS `say` sentences; threshold -2.6), so no second model pass is needed for the decision.
Measured (M1, Chrome): Brahui 2.3 s clip -> English in 0.60 s total; English 2.2 s clip -> Brahui in 2.1 s (Whisper 0.7 s,
r5 MT 1.2 s). Whisper on WebGPU needs fp32 encoder + q4 decoder (fp16 decoder throws a bare ORT allocation number).
Next: an en->brh student (same recipe as student_c, 24/6, from r5's back-translations) would bring that direction to <1 s;
deploy_web.py does not yet upload whisper / r5 / talk.html (config.js keys `asr_en`, `mt_e2b` are read by the page).
