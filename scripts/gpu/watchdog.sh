#!/usr/bin/env bash
# Progress watchdog for a training run on a pod. Prints one line per interval with the step counter and GPU
# utilization, and an ALERT line when the step has not advanced or the GPU has been idle for two consecutive
# checks. Run it locally against the pod: scripts/gpu/watchdog.sh <ssh wrapper> <remote log> [interval s]
# Exits when the log reports completion ("done;") or the finish trap fires, so a Monitor on it ends naturally.
set -u
SSH=$1; LOG=$2; INT=${3:-180}
prev_step=-1; idle=0
while true; do
  out=$($SSH "tr '\r' '\n' < $LOG 2>/dev/null | grep -oE '[0-9]+/[0-9]+ \[' | tail -1 | cut -d/ -f1; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; grep -cE 'done;|finish:' $LOG 2>/dev/null" 2>/dev/null)
  step=$(echo "$out" | sed -n 1p); util=$(echo "$out" | sed -n 2p); fin=$(echo "$out" | sed -n 3p)
  step=${step:--1}; util=${util:-0}
  echo "$(date -u +%H:%M) step=$step gpu=${util}%"
  if [ "${fin:-0}" -gt 0 ]; then echo "job finished"; exit 0; fi
  if [ "$step" = "$prev_step" ] || [ "${util%.*}" -lt 10 ]; then idle=$((idle+1)); else idle=0; fi
  [ $idle -ge 2 ] && echo "ALERT: no progress for $((idle*INT/60)) min (step=$step gpu=${util}%)"
  prev_step=$step; sleep "$INT"
done
