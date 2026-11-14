# GPU training: Brahui ASR, straight shot

One job, one pod, one script: `scripts/gpu/train_asr.sh`. It fine-tunes Meta's omniASR CTC-1B v2 (the
strongest zero-shot Brahui model, in Hugging Face format) on all 90 h of Brahui speech, evaluates on
the sealed test sets before and after, and writes the result tables.

## What you need

| Item | Spec | Why |
|---|---|---|
| GPU | **1x A100 80 GB** (RunPod secure/community, Lambda) | 1B params full fine-tune in bf16 with 8 x 30 s clips per micro-batch. A 4090/A5000 24 GB works with `BS=4 ACC=8`, about 2x slower. |
| Disk | 60 GB volume (stage 2: 40 GB is **not** enough) | 9 GB audio + 4 GB base model + checkpoints (**10 GB each**: 4 GB weights + 6 GB optimizer, 2 kept) + 6 GB venv. On 2026-10-17 a 40 GB volume filled at the second checkpoint and the whole tmux session was killed without a traceback. Keep the venv on the container disk (`UV_PROJECT_ENVIRONMENT=/root/venv`). |
| Time | ~3 h on A100 for 5,000 steps (~5 epochs over 90 h), plus ~20 min for baselines and evals | |
| Cost | ~$5–8 on A100 at $1.6–2/h. A second run with different hyperparameters doubles that. | |
| Image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` or newer; the script installs `uv` and the project venv (Python 3.12) | |

The base model is public and all data is rsynced from this Mac. A Hugging Face token **is** needed on the pod
for the results upload (below): the pod cannot reach the Mac, so finished models go to a private HF repo.

## Persistent data: the network volume (since 2026-10-18)

All audio, manifests and models live on RunPod network volume **`<volume-id>`** (`brahui-asr-data`, 50 GB STANDARD,
**US-MO-1**, ~$3.50/month). Create pods in US-MO-1 with `mounts.network = [{volumeId: "<volume-id>", path: "/workspace"}]`
and the repo is already at `/workspace/brahui-digitize` with `data/` and `runs/` populated; only the code changes need
an rsync. Network volumes are pinned to one data center, so if US-MO-1 has no A100 stock either wait or fall back to
an H100 there. Keep the venv on the container disk (`UV_PROJECT_ENVIRONMENT=/root/venv`, 6 GB) so it never competes
with checkpoints for volume space. **Pin `gpu.minCudaVersion` to `13.0`** when creating the pod: `uv.lock` resolves
torch to a cu130 build, which fails with "driver too old" on a CUDA 12.8 host (driver 570) and silently falls back
to CPU (2026-10-18: a run sat 30 min at 97 s/batch with 0 MiB on the GPU). If you do land on a 12.8 host, fix it with
`uv pip install --python /root/venv/bin/python --no-cache --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128`
and add `export UV_NO_SYNC=1` to `env.sh` so `uv run` does not re-sync back to the lock; clear `/root/.cache/uv`
first or the 20 GB container disk fills. The volume is only written by job scripts: results are pulled back to the Mac with
rsync after each run (the HF upload path in `finish.sh` needs a write-scope token, which the Mac does not have).

Layout on the volume:

| path | content |
|---|---|
| `brahui-digitize/data/{manifests,eval}` | all manifests incl. `brahuispeech_train_norm.jsonl`, sealed eval sets |
| `brahui-digitize/data/raw/mdc/.../clips`, `data/audio/{omnilingual,brahuispeech}` | 215 MB + 1.1 GB + 7.5 GB of audio |
| `brahui-digitize/runs/ctc1b_brh/final`, `runs/ctc1b_brh_s2/final` | stage-1 and stage-2 models (3.9 GB each) |
| `env.sh` | `UV_PROJECT_ENVIRONMENT` and any run-specific exports; `source` it before launching |

## Steps

1. **Start the pod** with the volume mounted at `/workspace`. Note its SSH host/port.
2. **Push code and data** from the Mac (about 9.3 GB; `tar` + `rsync` keeps it to one transfer):
   ```bash
   POD=root@<host> PORT=<port>
   ssh -p $PORT $POD 'mkdir -p /workspace && cd /workspace && git clone <repo url> brahui-digitize'
   rsync -avzP -e "ssh -p $PORT" --relative \
     data/manifests data/eval \
     data/raw/models/omniASR-CTC-1B-v2-hf \
     data/raw/mdc/common_voice_27_brh/cv-corpus-27.0-2026-09-11/brh/clips \
     data/audio/omnilingual data/audio/brahuispeech \
     $POD:/workspace/brahui-digitize/
   ```
   The manifests reference these exact relative paths, so nothing else needs configuring.
3. **Run** (inside `tmux` so an SSH drop does not kill it). Export `HF_TOKEN` (write scope) and `HF_UPLOAD_REPO`
   first; the script's exit trap (`scripts/gpu/finish.sh`) uploads `runs/*/final`, `evals/asr/runs` and `logs/`
   there and then **deletes the pod**, success or failure, so it never idles until the credit runs out.
   `KEEP_POD=1` keeps the pod alive for debugging.
   ```bash
   ssh -p $PORT $POD
   cd /workspace/brahui-digitize
   export HF_TOKEN=hf_... HF_UPLOAD_REPO=<user>/brahui-asr-runs
   tmux new -s train 'bash scripts/gpu/train_asr.sh 2>&1 | tee logs/run.log'
   ```
   Check the account credit covers the run with margin before launching ($1.79/h; a 5,000-step run is ~$7).
   Progress: `tail -f logs/train_asr.log`. Monitor CER on 200 held-out Common Voice dev clips prints every 500 steps; the best checkpoint by that CER is what gets saved to `final/`.
4. **Pull results back** on the Mac, from the HF repo (the pod is already gone):
   ```bash
   uv run hf download $HF_UPLOAD_REPO --repo-type model --local-dir . --include "runs/*" "evals/*" "logs/*"
   mv logs/*.log logs/gpu/ 2>/dev/null; uv run python scripts/summarize_results.py > evals/RESULTS_TABLES.md
   ```
   If the pod is still up (KEEP_POD=1), the old way works too:
   `rsync -avzP -e "ssh -p $PORT" $POD:/workspace/brahui-digitize/{evals/asr/runs,runs/ctc1b_brh/final,logs} .`

   Stage 2 (`scripts/gpu/stage2_adapt.sh`) needs `runs/ctc1b_brh/final` on the pod (rsync it up, 3.9 GB) and
   follows the same upload-then-delete flow.

## Knobs (environment variables for the script)

| var | default | notes |
|---|---|---|
| `STEPS` | 5000 | Meta's recommended fine-tune length. 90 h at ~5 min audio/step is ~1,000 steps per epoch. |
| `BS` / `ACC` | 8 / 4 | per-device clips x grad accumulation. Lower `BS`, raise `ACC` if OOM. |
| `LR` | 1e-5 | Meta's recommendation for continuing a CTC checkpoint. Try 3e-5 if the monitor CER plateaus early. |
| `BASE` | `data/raw/models/omniASR-CTC-1B-v2-hf` | `aadel4/omniASR-CTC-300M-v2` for a cheap dry run. |

Rerunning the script resumes from the last checkpoint in `runs/ctc1b_brh/`.

## What success looks like

| model | CV test CER | Omnilingual test CER |
|---|---|---|
| Whisper-Small Brahui (community, trained on the same 70 h) | 18.9 | 19.1 |
| omniASR CTC-1B zero-shot (fairseq2, 300-clip subset) | 18.3 | 20.0 |
| **target after fine-tune** | **< 8** | **< 10** |

Meta reports a mean CER of about 3 for languages with 10 to 50 h of fine-tuning data, so single digits on
read speech is expected and sub-5 is plausible. Spontaneous speech (Omnilingual test) will lag by a few points.

## Verified locally before spending GPU money

- The HF-converted 300M v2 model reproduces the fairseq2 original's hypotheses on the 300-clip CV subset (CER 25.0 vs 22.7; the gap is Pashto letter variants ټ ګ ړ that the v2 tokenizer prefers, which fine-tuning removes).
- The pretrained vocabulary covers 100% of the characters in the Brahui training transcripts, so the CTC head is kept and fine-tuned rather than re-initialised.
- `scripts/train_asr.py --smoke` trains, saves and reloads end to end on CPU; the saved model evaluates through `scripts/asr_eval.py --model ctc:<dir>`.
- `scripts/segment_long.py` aligns the 88 s Omnilingual segments word by word (249/249 words on the first segment) and cuts them into 20 s chunks, recovering ~6 h of spontaneous speech that would otherwise be dropped.
