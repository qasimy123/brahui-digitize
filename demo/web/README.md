# Browser demo (no server-side inference)

1. Export models: see `tools/onnx_export/export_quant.py`. Then link/copy them here:
   `mkdir -p demo/web/models && ln -s ../../../export/student_c demo/web/models/student_c && ln -s ../../../export/ctc300m_student demo/web/models/ctc300m_student && ln -s ../../../export/ctc1b_s3 demo/web/models/ctc1b_s3`
   transformers.js expects `<model>/config.json`, `tokenizer*.json`, `preprocessor_config.json` and `onnx/{encoder_model,decoder_model_merged}[_quantized].onnx` (MT) or `onnx/model[_quantized].onnx` (ASR).
   ASR picker: `ctc300m_student` (650 MB fp16 / 366 MB int8, ~0.07 s per second of audio) decoded with the character-LM beam search in `ctc_lm.js` is the default: cv_test CER 8.0 (greedy 9.9). It needs `models/brh_char6.blm` (20 MB, from `scripts/export_char_lm.py`; `?lm=0` falls back to greedy). `ctc1b_s3` (full MMS-1B s3, greedy CER 8.6, `onnx/model_fp16.onnx` 1.95 GB, WebGPU only, ~0.14 s per second of audio) is the slower option. The 1B fp16 file must be built with `convert_fp16.py --fp32-node` (layer-0 attention overflows fp16, see the script docstring); the plain conversion returns all-NaN logits on WebGPU.
2. Serve statically (SharedArrayBuffer is not required): `python3 -m http.server -d demo/web 8080` and open http://localhost:8080.
   For GitHub Pages / HF static Space, upload the same tree; files >100 MB need Git LFS or the HF Hub.

## Spoken demo: `talk.html`

Hold the mic (or the space bar), speak English or Brahui, release. Auto-detect runs the Brahui CTC first and scores its greedy
output with the Brahui character LM: Brahui speech scores about -1.3 to -2.1 nats/char, English speech -2.7 to -4.5, threshold
-2.6 (`?lmthresh=`). Brahui -> English shows the Brahui transcript, its romanization (`romanize.js`, port of `brahui.text.romanize`)
and the English; English -> Brahui shows the English transcript, the Brahui script and its romanization.
Models (all local, loaded once): `ctc300m_student` + `brh_char6.blm`, `whisper-base.en` (onnx-community, WebGPU fp32 encoder +
q4 decoder; the fp16 decoder fails to load), `student_c` for Brahui->English, `nllb_r5_trim` for English->Brahui (student_c is a
one-direction distillation and just copies English input). M1 timings: Brahui sentence 0.6 s end to end; English sentence ~2.1 s
(Whisper 0.7 s + NLLB-1.3B int8 on WASM 1.2 s). `?autoload=1` loads on open. Download whisper with
`huggingface_hub.snapshot_download("onnx-community/whisper-base.en", local_dir="demo/web/models/whisper-base.en")`
(config/tokenizer JSON + `onnx/encoder_model.onnx`, `onnx/decoder_model_merged_q4.onnx`).
