#!/usr/bin/env bash
# F5 retrain on vowelled text (data/tts/f5/multi2_vow, scripts/tts_f5_export_vowel.py), then the blind listening set:
# the same 24 sentences (scripts/build_listen_v3.py) spoken by the old voice from its own text and by the new voice from
# vowelled text, same Common Voice reference clip, speed 0.75.
# The Mac uploads: data/tts/f5/multi2_roman/flac, multi2_{roman,vow}{,_dev,_refs}/metadata.csv, multi_arabic_{dev,refs}/wavs,
# listen3_{roman,vow}/metadata.csv, data/tts/listen_v3/set.jsonl, data/tts/multi/dev.jsonl, runs/ctc1b_brh_s3/final, runs/f5_multi_roman/final.
# Usage (tmux on the pod):  bash scripts/gpu/f5_vow.sh   -> logs/f5_vow.log; markers "LISTEN DONE" / "LISTEN FAILED"
set -uo pipefail
cd "$(dirname "$0")/../.."; mkdir -p logs /workspace; L=logs/f5_listen.log; D=data/tts/f5; PY=/root/f5env/bin/python
touch /workspace/.hold.listen
export UV_PROJECT_ENVIRONMENT=/root/venv UV_PYTHON=3.12
command -v uv >/dev/null || pip install -q --break-system-packages uv || pip install -q uv
if [ ! -d $D/multi2_roman/wavs ] || [ "$(ls $D/multi2_roman/wavs | wc -l)" -lt "$(ls $D/multi2_roman/flac | wc -l)" ]; then
  echo "[$(date -u +%H:%M:%S)] flac -> wav"; mkdir -p $D/multi2_roman/wavs
  uv run python - <<'PY' || { echo "wav conversion failed"; rm -f /workspace/.hold.listen; exit 2; }
import soundfile as sf, glob, os
from multiprocessing import Pool
def f(p):
    o = p.replace("/flac/", "/wavs/")[:-5] + ".wav"
    if not os.path.exists(o): y, sr = sf.read(p, dtype="int16"); sf.write(o + ".tmp.wav", y, sr, subtype="PCM_16"); os.rename(o + ".tmp.wav", o)
fs = sorted(glob.glob("data/tts/f5/multi2_roman/flac/*.flac"))
with Pool(8) as p: p.map(f, fs, chunksize=64)
print(len(fs), "wavs")
PY
fi
for s in _dev _refs; do [ -e $D/multi2_roman$s/wavs ] || ln -s ../multi_arabic$s/wavs $D/multi2_roman$s/wavs; done
rm -f /tmp/f5_train_done
(
  while [ ! -f /tmp/f5_train_done ]; do sleep 20; done
  NEW=runs/f5_multi2_vow/final; OLD=runs/f5_multi_roman/final; O=evals/tts/listen_v3; ok=1
  if [ -f $NEW/model.safetensors ]; then
    $PY tools/f5/synth_eval.py --ckpt $NEW/model.safetensors --vocab $NEW/vocab.txt --dev $D/listen3_vow --refs $D/multi2_vow_refs --out $O/new --listen 0 --speed 0.75 >>$L 2>&1 || ok=0
    $PY tools/f5/synth_eval.py --ckpt $OLD/model.safetensors --vocab $OLD/vocab.txt --dev $D/listen3_roman --refs $D/multi2_roman_refs --out $O/old --listen 0 --speed 0.75 >>$L 2>&1 || ok=0
    for v in new old; do echo "listen $v: $(uv run python scripts/tts_score_wavs.py --wavs $O/$v/wavs --dev data/tts/listen_v3/set.jsonl 2>>$L | tail -1)" | tee -a $L; done
  else ok=0; echo "no new model: nothing to synthesize" >>$L; fi
  [ $ok = 1 ] && [ "$(ls $O/new/wavs 2>/dev/null | wc -l)" = "$(ls $O/old/wavs 2>/dev/null | wc -l)" ] && echo "LISTEN DONE" | tee -a $L || echo "LISTEN FAILED" | tee -a $L
  rm -f /workspace/.hold.listen
) &
NAME=multi2_vow WAVS_FROM=multi2_roman INIT=runs/f5_multi_roman/final/model_final.safetensors INIT_VOCAB=runs/f5_multi_roman/final/vocab.txt \
  LR=${LR:-5e-5} BATCH=${BATCH:-6400} UPDATES=${UPDATES:-40000} FINAL_SPEED=0.75 NO_CURVE=${NO_CURVE:-0} PULL_GRACE_MIN=${PULL_GRACE_MIN:-90} bash scripts/gpu/train_f5.sh
