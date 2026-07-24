#!/usr/bin/env bash
# Autonomous 4-phase beta=0 eval at mask_thr=0.25 (grow masks to recover under-covered
# small crowns on predicted boxes). det_t8 proxy: +0.043 mAP50 / +0.035 mAP50-95 @0.25.
# Clean cancel = modal app stop (pkill alone leaves the remote container running).
set -u
cd "$(dirname "$0")" || exit 1
MODAL=../../../.venv/bin/modal
ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*"; }

say "=== 4phase beta=0 mask_thr=0.25 eval START (ref: beta=0@0.5 = mask 0.5794 / mAP50-95 0.1948) ==="
nohup $MODAL run phase4_modal.py::eval_selfmask --seed 0 --beta 0 --mask-thr 0.25 > thr025_eval.log 2>&1 &
sleep 5
# capture the ephemeral app id so we can app-stop it if needed
sleep 20; APP=$(grep -oE 'ap-[A-Za-z0-9]+' thr025_eval.log | head -1)
say "app id: ${APP:-<pending>}"
start=$SECONDS last=0
while true; do
  if grep -qF '"mask_mAP50"' thr025_eval.log 2>/dev/null; then say "EVAL: SUCCESS"; break; fi
  if ! pgrep -f "phase4_modal.py::eval_selfmask" >/dev/null 2>&1; then
    sleep 4
    grep -qF '"mask_mAP50"' thr025_eval.log 2>/dev/null && { say "EVAL: SUCCESS (late)"; break; }
    say "EVAL: FAILED — modal run exited without result"
    grep -iE "error|assert|traceback|exception" thr025_eval.log | tail -4
    exit 1
  fi
  el=$((SECONDS-start))
  [ "$el" -ge 5400 ] && { say "RUNAWAY >90min; app stop ${APP}"; [ -n "$APP" ] && $MODAL app stop "$APP" --yes 2>/dev/null; exit 2; }
  [ $((el-last)) -ge 600 ] && { last=$el; say "HEARTBEAT ${el}s | $(tail -1 thr025_eval.log 2>/dev/null | cut -c1-90)"; }
  sleep 30
done

m50=$(grep -oE '"mask_mAP50": [0-9.]+' thr025_eval.log | head -1 | grep -oE '[0-9.]+$')
m5095=$(grep -oE '"mask_mAP50_95": [0-9.]+' thr025_eval.log | head -1 | grep -oE '[0-9.]+$')
box=$(grep -oE '"box_mAP50": [0-9.]+' thr025_eval.log | head -1 | grep -oE '[0-9.]+$')
say "SUMMARY: 4phase beta=0 mask_thr=0.25 -> mask_mAP50=$m50 mask_mAP50-95=$m5095 box=$box (vs @0.5: 0.5794/0.1948)"
say "=== DONE ==="
