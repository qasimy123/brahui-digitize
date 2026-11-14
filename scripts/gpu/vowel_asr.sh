#!/usr/bin/env bash
# Vowel-aware acoustic model, two cross-validation folds: train, then score every hidden / untrusted spelling (scripts/vowel_asr.py).
# The Mac rsyncs first: scripts/, data/tts/vowel_asr/ (scripts/build_vowel_asr.py), runs/ctc1b_brh_s3/final. Word spans (scripts/vowel_align.py) are made here if missing.
# Usage (tmux on the pod):  bash scripts/gpu/vowel_asr.sh        FOLDS="0 1" EPOCHS=4
# Results: evals/tts/vowel_asr/scores_f<m>.json, runs/vowel_asr_f<m>/final, logs/vowel_asr.log ("ALL DONE" at the end).
# finish.sh deletes the pod on exit, success or failure, after waiting PULL_GRACE_MIN for the Mac to pull (touch /workspace/.pulled).
set -euo pipefail
cd "$(dirname "$0")/../.."
FOLDS=${FOLDS:-"0 1"}; EPOCHS=${EPOCHS:-3}; LOG=logs/vowel_asr.log
mkdir -p logs evals/tts/vowel_asr /workspace; touch /workspace/.hold.vowel
export PULL_GRACE_MIN=${PULL_GRACE_MIN:-90}
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
trap 'st=$?; rm -f /workspace/.hold.vowel; scripts/gpu/finish.sh $st runs/vowel_asr_f0 2>&1 | tee -a "$LOG"' EXIT
python -c "import transformers, soundfile" 2>/dev/null || pip install -q --break-system-packages "transformers==5.19.0" soundfile 2>&1 | tail -2
python - <<'PY' 2>&1 | tee -a "$LOG"
import sys, torch, torchaudio, transformers
ok = torch.cuda.is_available(); print("torch", torch.__version__, "torchaudio", torchaudio.__version__, "transformers", transformers.__version__, "cuda", ok, torch.cuda.get_device_name(0) if ok else "")
if not ok: sys.exit(1)
x = torch.randn(8, 100, 36, device="cuda").log_softmax(-1); t = torch.randint(1, 35, (8, 20), device="cuda")
print("ctc ok", float(torch.nn.functional.ctc_loss(x.transpose(0, 1), t, torch.full((8,), 100), torch.full((8,), 20), reduction="mean")) > 0)
torchaudio.functional.forced_align(x[:1].cpu(), t[:1].cpu(), blank=0); print("forced_align ok")
PY
[ "${PIPESTATUS[0]}" = 0 ] || { log "preflight failed"; exit 2; }
[ -f data/tts/vowel_asr/spans.json ] || { log "word alignment (scripts/vowel_align.py)"; python -u scripts/vowel_align.py 2>&1 | grep -E "^[0-9]+/|done;" | tee -a "$LOG"; }
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1); MAXSEC=${MAXSEC:-$((MEM * 26 / 10000))}
log "GPU memory ${MEM} MiB -> $MAXSEC s of audio per batch; folds: $FOLDS; epochs $EPOCHS"
for f in $FOLDS; do
  timeout 5h python -u scripts/vowel_asr.py train --fold $f --out runs/vowel_asr_f$f --epochs $EPOCHS --max-sec $MAXSEC 2>&1 | tee -a "$LOG" | grep -v "Warning"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "training fold $f failed"; exit 3; }
  timeout 2h python -u scripts/vowel_asr.py score --fold $f --model runs/vowel_asr_f$f/final --out evals/tts/vowel_asr/scores_f$f.json --max-sec $MAXSEC 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "scoring fold $f failed"; exit 4; }
  log "FOLD $f COMPLETE"
done
log "ALL DONE"
