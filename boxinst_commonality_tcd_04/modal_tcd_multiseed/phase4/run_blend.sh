#!/usr/bin/env bash
# Autonomous seed-0 gamma-blend eval. Both maskers + detector already on the volume, so
# this is one A100 eval (~$2). Reports BOTH mask mAP50 (expected flat ~0.579) and
# mAP50-95 (the sigmoid's localization axis; beta=0 ref 0.1948). Heartbeats + deadline.
set -u
cd "$(dirname "$0")" || exit 1
MODAL=../../../.venv/bin/modal
S0=65; TAU=18
ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*"; }

say "=== gamma-blend seed0 eval START (s0=${S0} tau=${TAU}; refs: beta=0 mask50 0.579 / mAP50-95 0.1948) ==="
nohup $MODAL run phase4_modal.py::eval_blend --seed 0 --s0 $S0 --tau $TAU > blend_eval.log 2>&1 &
sleep 5
start=$SECONDS last=0
while true; do
  if grep -qF '"mask_mAP50"' blend_eval.log 2>/dev/null; then say "EVAL: SUCCESS"; break; fi
  if ! pgrep -f "phase4_modal.py::eval_blend" >/dev/null 2>&1; then
    sleep 4
    grep -qF '"mask_mAP50"' blend_eval.log 2>/dev/null && { say "EVAL: SUCCESS (late)"; break; }
    say "EVAL: FAILED — modal run exited without result"
    grep -iE "error|assert|traceback|exception" blend_eval.log | tail -4
    exit 1
  fi
  el=$((SECONDS-start))
  [ "$el" -ge 5400 ] && { say "RUNAWAY >90min; killing"; pkill -f "phase4_modal.py::eval_blend"; exit 2; }
  [ $((el-last)) -ge 600 ] && { last=$el; say "HEARTBEAT ${el}s | $(tail -1 blend_eval.log 2>/dev/null | cut -c1-90)"; }
  sleep 30
done

m50=$(grep -oE '"mask_mAP50": [0-9.]+' blend_eval.log | head -1 | grep -oE '[0-9.]+$')
m5095=$(grep -oE '"mask_mAP50_95": [0-9.]+' blend_eval.log | head -1 | grep -oE '[0-9.]+$')
box=$(grep -oE '"box_mAP50": [0-9.]+' blend_eval.log | head -1 | grep -oE '[0-9.]+$')
say "SUMMARY: gamma-blend seed0 -> mask_mAP50=$m50 (beta0 0.579) | mask_mAP50-95=$m5095 (beta0 0.1948) | box=$box"
say "=== gamma-blend seed0 eval DONE ==="
