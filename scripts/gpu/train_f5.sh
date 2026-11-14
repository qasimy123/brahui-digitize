#!/usr/bin/env bash
# Fine-tune F5-TTS v1 Base on an exported Brahui multi-speaker set, with a quality curve measured during training.
# The Mac exports and rsyncs the data first (docs/PLAN_TTS.md):
#   data/tts/f5/<NAME>/{wavs,metadata.csv}  data/tts/f5/<NAME>_dev  data/tts/f5/<NAME>_refs  data/tts/multi/dev.jsonl
#   runs/ctc1b_brh_s3/final (the ASR that scores the synthetic speech)
#   (for a second text mode, rsync only its metadata.csv files and set WAVS_FROM=<other NAME> to reuse the wavs)
# Usage (tmux on the pod):  NAME=multi_arabic UPDATES=30000 bash scripts/gpu/train_f5.sh
#
# Safety:
#   disk     only one rolling training checkpoint exists (model_last.pt, 5.4 GB, rewritten every $LAST updates) plus one
#            temporary copy while it is being evaluated; the job refuses to start with < $MIN_FREE_GB free and stops
#            training if free space drops below 8 GB.
#   quality  every time the checkpoint is rewritten, a sidecar synthesizes $CURVE_N dev sentences from its live (non-EMA) weights and
#            (the EMA restored from the pretrained checkpoint moves too slowly: at 15k updates EMA 45% CER vs live 13.7%, 2026-11-11;
#            EVAL_CPU=1 runs the curve evals on CPU when the training batch leaves no GPU memory)
#            scores them with the ASR -> logs/f5_<NAME>_curve.jsonl {update, cer}. The best-scoring EMA weights are kept
#            as runs/f5_<NAME>/final/model.safetensors, so a run that degrades late still returns its best point.
# Results: runs/f5_<NAME>/final/{model.safetensors,vocab.txt}, evals/tts/f5_<NAME>/ (curve samples, full dev wavs + CER), logs/.
# finish.sh deletes the pod at the end, success or failure (KEEP_POD=1 to keep it; PULL_GRACE_MIN waits for the Mac's rsync).
set -euo pipefail
cd "$(dirname "$0")/../.."
NAME=${NAME:-multi_arabic}; UPDATES=${UPDATES:-30000}; LR=${LR:-1e-5}; BATCH=${BATCH:-9600}   # frames/batch: ~9600 on 48 GB, ~4800 on 24 GB
WARMUP=${WARMUP:-1000}; LAST=${LAST:-2500}; CURVE_N=${CURVE_N:-20}; MIN_FREE_GB=${MIN_FREE_GB:-30}
WAVS_FROM=${WAVS_FROM:-}; F5_COMMIT=${F5_COMMIT:-77d3ec6}
RUN=runs/f5_$NAME; F5=tools/f5/F5-TTS; VENV=/root/f5env; PY=$VENV/bin/python; D=data/tts/f5; LOG=logs/f5_$NAME.log; CURVE=logs/f5_${NAME}_curve.jsonl
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-/root/venv}; export UV_PYTHON=${UV_PYTHON:-3.12}
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
free_gb() { df -BG --output=avail . | tail -1 | tr -dc 0-9; }
mkdir -p logs "$RUN/final" /workspace; touch /workspace/.hold.f5_$NAME
export PULL_GRACE_MIN=${PULL_GRACE_MIN:-60}
trap 'st=$?; touch /tmp/f5_train_done; rm -f /workspace/.hold.f5_'"$NAME"'; scripts/gpu/finish.sh $st "$RUN" evals/tts/f5_'"$NAME" EXIT
rm -f /tmp/f5_train_done

[ "$(free_gb)" -ge "$MIN_FREE_GB" ] || { log "only $(free_gb) GB free, need $MIN_FREE_GB"; exit 2; }
command -v uv >/dev/null || pip install -q uv
if [ ! -x $PY ]; then
  log "building F5 venv"
  uv venv -q --python 3.11 $VENV
  [ -d $F5/.git ] || git clone -q https://github.com/SWivid/F5-TTS.git $F5
  git -C $F5 checkout -q "$F5_COMMIT"
  # torchaudio <= 2.8 loads audio without torchcodec (2.9+ needs torchcodec + ffmpeg: the first batch crashed locally on 2026-11-09)
  # --python: with UV_PYTHON=3.12 exported above, uv otherwise installs into the wrong interpreter and torch is silently missing (2026-11-11)
  uv pip install -q --python $PY "torch==2.4.1" "torchaudio==2.4.1" --index-url https://download.pytorch.org/whl/cu124
  uv pip install -q --python $PY -e $F5 "torch==2.4.1" "torchaudio==2.4.1"
fi
$PY -c "import torch,torchaudio,sys; ok=torch.cuda.is_available(); print('torch', torch.__version__, torchaudio.__version__, 'cuda', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" \
  || { log "CUDA preflight failed (F5 venv)"; exit 2; }

for s in "" _dev _refs; do
  [ -f $D/$NAME$s/metadata.csv ] || { log "missing $D/$NAME$s/metadata.csv"; exit 2; }
  if [ -n "$WAVS_FROM" ] && [ ! -e $D/$NAME$s/wavs ]; then ln -s "../$WAVS_FROM$s/wavs" $D/$NAME$s/wavs; fi
  [ -d $D/$NAME$s/wavs/ ] || { log "missing $D/$NAME$s/wavs"; exit 2; }
done
[ -d runs/ctc1b_brh_s3/final ] || { log "missing runs/ctc1b_brh_s3/final (the scoring ASR)"; exit 2; }

# Scoring preflight before any training: the same ASR on the real dev clips. Proves the scoring env works on this GPU
# and gives the number synthetic speech is compared against.
score() { uv run python scripts/tts_score_wavs.py "$@" 2>>"$LOG" | tail -1; }
REAL=$(score --wavs $D/${NAME}_dev/wavs --tag real); log "real dev speech: $REAL"
echo "$REAL" | grep -q cer_mean || { log "scoring preflight failed"; exit 2; }

DS=$PWD/$F5/data/${NAME}_custom; CK=$PWD/$F5/ckpts/$NAME
if [ ! -f $CK/pretrained_model_1250000_ext.safetensors ] || [ ! -f $DS/raw.arrow ]; then $PY tools/f5/prep_finetune.py $D/$NAME --name $NAME 2>&1 | tee -a "$LOG" | tail -4; fi
cp $DS/vocab.txt "$RUN/final/vocab.txt"
# INIT=<weights.safetensors> INIT_VOCAB=<its vocab.txt>: continue from an earlier fine-tune instead of the base model
if [ -n "${INIT:-}" ] && [ ! -f $CK/.init_done ]; then $PY tools/f5/init_from.py --weights "$INIT" --old-vocab "$INIT_VOCAB" --new-vocab $DS/vocab.txt --out $CK/pretrained_model_1250000_ext.safetensors 2>&1 | tee -a "$LOG" | tail -1; touch $CK/.init_done; fi
EPOCHS=$($PY -c "import json,math; d=json.load(open('$DS/duration.json'))['duration']; upe=max(1, sum(d)*24000/256/$BATCH); print(math.ceil($UPDATES/upe))")
log "F5 fine-tune $NAME: ~$UPDATES updates = $EPOCHS epochs, $BATCH frames/batch, lr $LR, checkpoint + curve point every $LAST updates"

eval_ckpt() {  # copy the rolling checkpoint, synthesize a few dev sentences from its EMA weights, score, keep the best
  local up out cer best
  up=$(tr '\r' '\n' < "$LOG" | grep -o 'update=[0-9]*' | tail -1 | cut -d= -f2); up=$(( ${up:-0} / LAST * LAST ))
  cp $CK/model_last.pt /tmp/f5_eval.pt || return 0
  out=evals/tts/f5_$NAME/curve/u$up
  $PY tools/f5/synth_eval.py --ckpt /tmp/f5_eval.pt --vocab $DS/vocab.txt --dev $D/${NAME}_dev --refs $D/${NAME}_refs --out $out \
      --n $CURVE_N --listen 0 --nfe 16 --raw --export-ema /tmp/f5_eval_ema.safetensors >>"$LOG.curve" 2>&1 || { log "curve: synthesis failed at update $up (see $LOG.curve)"; rm -f /tmp/f5_eval.pt; return 0; }
  rm -f /tmp/f5_eval.pt
  cer=$(score --wavs $out/wavs | $PY -c "import json,sys; print(json.load(sys.stdin)['cer_mean'])" 2>/dev/null || echo "")
  [ -n "$cer" ] || { log "curve: scoring failed at update $up"; return 0; }
  best=$( [ -f "$CURVE" ] && $PY -c "import json; print(min(json.loads(l)['cer'] for l in open('$CURVE')))" || echo 9 )
  echo "{\"update\": $up, \"cer\": $cer, \"free_gb\": $(free_gb)}" >> "$CURVE"
  if $PY -c "import sys; sys.exit(0 if $cer < $best else 1)"; then mv /tmp/f5_eval_ema.safetensors "$RUN/final/model.safetensors"; echo "$up" > "$RUN/final/update.txt"; log "curve: update $up CER $cer (best so far, kept)"
  else rm -f /tmp/f5_eval_ema.safetensors; log "curve: update $up CER $cer (best $best)"; fi
}
sidecar() {
  local last=0 m
  while [ ! -f /tmp/f5_train_done ] && [ ! -f /tmp/f5_train_over ]; do
    sleep 30
    if [ "$(free_gb)" -lt 8 ]; then log "ALERT: $(free_gb) GB free, stopping training"; pkill -f finetune_cli.py || true; fi
    [ -f $CK/model_last.pt ] || continue
    [ "${NO_CURVE:-0}" = 1 ] && continue   # 24 GB card: no memory for evals beside training; only the final state is evaluated
    m=$(stat -c %Y $CK/model_last.pt)
    if [ "$m" -gt "$last" ] && [ $(( $(date +%s) - m )) -gt 40 ]; then last=$m; echo "$m" > /tmp/f5_last_eval; if [ "${EVAL_CPU:-0}" = 1 ]; then (export CUDA_VISIBLE_DEVICES=""; eval_ckpt); else eval_ckpt; fi; fi
  done
}
rm -f /tmp/f5_train_over /tmp/f5_last_eval; sidecar & SIDE=$!

(cd $F5 && $VENV/bin/accelerate launch --mixed_precision fp16 src/f5_tts/train/finetune_cli.py --exp_name F5TTS_v1_Base --dataset_name $NAME --finetune \
    --pretrain $CK/pretrained_model_1250000_ext.safetensors --tokenizer custom --tokenizer_path $DS/vocab.txt \
    --learning_rate $LR --batch_size_per_gpu $BATCH --batch_size_type frame --max_samples 64 --epochs $EPOCHS --num_warmup_updates $WARMUP \
    --save_per_updates 10000000 --last_per_updates $LAST --keep_last_n_checkpoints 1) 2>&1 | tee -a "$LOG" | tr '\r' '\n' | awk 'NR % 500 == 0 || !/update\/s|s\/update|it\/s|s\/it/' || log "training exited with an error"

touch /tmp/f5_train_over; wait $SIDE || true
# the trainer writes model_last.pt once more at the end: give that final state a curve point too
if [ -f $CK/model_last.pt ] && [ "$(stat -c %Y $CK/model_last.pt)" -gt "$(cat /tmp/f5_last_eval 2>/dev/null || echo 0)" ]; then eval_ckpt; fi
[ -f "$RUN/final/model.safetensors" ] || { log "no checkpoint was ever evaluated"; exit 3; }
log "curve: $(tr '\n' ' ' < "$CURVE")"
log "synthesizing the full dev set from the best weights (update $(cat "$RUN/final/update.txt"))"
$PY tools/f5/synth_eval.py --ckpt "$RUN/final/model.safetensors" --vocab $DS/vocab.txt --dev $D/${NAME}_dev --refs $D/${NAME}_refs --out evals/tts/f5_$NAME 2>&1 | tee -a "$LOG" | tail -2
log "full dev: $(score --wavs evals/tts/f5_$NAME/wavs)   real speech: $REAL"
if [ -n "${FINAL_SPEED:-}" ]; then   # the natural-duration setting (default speed comes out ~1.4x too fast)
  $PY tools/f5/synth_eval.py --ckpt "$RUN/final/model.safetensors" --vocab $DS/vocab.txt --dev $D/${NAME}_dev --refs $D/${NAME}_refs --out evals/tts/f5_${NAME}_speed --listen 0 --speed "$FINAL_SPEED" 2>&1 | tee -a "$LOG" | tail -1
  log "full dev at speed $FINAL_SPEED: $(score --wavs evals/tts/f5_${NAME}_speed/wavs)"
fi
rm -f $CK/model_last.pt   # 5.4 GB with optimizer state; the EMA export is what we keep
log "F5 $NAME done;"
