# Deploying the demo for free

The demo is a static page. Inference runs in the visitor's browser (transformers.js on ONNX Runtime Web), so the
only hosted things are files. Both hosts below are free; total cost is $0/month.

## 1. Models → Hugging Face Hub (free, unlimited public storage)

From the private working repo that holds `export/`:

```bash
HF_TOKEN=hf_...   # write-scope token from https://huggingface.co/settings/tokens
uv run python scripts/deploy_web.py --mt export/student_c --asr export/ctc300m_student \
    --lm demo/web/models/brh_char6.blm --space brahui-demo
```

This creates `qkhawaja/brahui-mt-student-c-onnx` and `qkhawaja/brahui-asr-ctc300m-student-onnx` (int8 ONNX +
tokenizer files + the 20 MB character LM), rewrites `demo/web/config.js` to point at them, and uploads `demo/web/`
as a **static** Space at `https://huggingface.co/spaces/qkhawaja/brahui-demo`. Static Spaces need no PRO plan and
never sleep.

To add the 1B fp16 WebGPU option: `--asr export/ctc1b_s3 --variants fp16` in a second run, then set `asr1b` in
`config.js`.

## 2. Page → GitHub Pages (also free)

Commit the rewritten `demo/web/config.js` to this repo. The workflow in `.github/workflows/pages.yml` publishes
`demo/web/` to `https://<user>.github.io/brahui-digitize/` on every push to `main`. Enable it once:
repository Settings → Pages → Source: GitHub Actions.

Because `config.js` points at the Hub, GitHub Pages serves only ~1 MB of HTML/JS and the browser fetches models
from Hugging Face directly. Multithreaded WASM needs cross-origin isolation, which `coi-serviceworker.min.js`
provides on static hosts.

## Alternatives, in case the browser path is not enough

| option | cost | when |
|---|---|---|
| HF static Space + Hub (above) | $0 | default |
| GitHub Pages + Hub (above) | $0 | same, if you prefer the repo and demo to live together |
| HF CPU Space with the int8 ONNX models server-side (`demo/hf_space` in the private repo) | $0 (2 vCPU, sleeps after 48 h idle) | phones and old laptops that cannot run a 366 MB model |
| HF ZeroGPU Space | $9/month PRO | not needed; the browser models match the 1B teacher within 0.6 CER |
| Cloudflare Pages + R2 for models | $0 | if Hub download speed is a problem in Pakistan |
