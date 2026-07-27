#!/usr/bin/env bash
# Generate + persist predictions (boxes + RLE masks) for the best model
# (4-phase L24 detector + β=0 self-mask @ mask_thr=0.25) to the Modal volume.
set -u
cd "$(dirname "$0")" || exit 1
MODAL=../../../.venv/bin/modal
ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*"; }

say "=== save_preds run START (β=0 @ mask_thr=0.25, boxes+masks) ==="
nohup $MODAL run phase4_modal.py::eval_selfmask --save-preds > savepreds.log 2>&1 &
sleep 25; APP=$(grep -oE 'ap-[A-Za-z0-9]+' savepreds.log | head -1)
say "app id: ${APP:-<pending>}"
start=$SECONDS last=0
while true; do
  if grep -qF "saved predictions" savepreds.log 2>/dev/null; then say "PREDS: SAVED"; break; fi
  if ! pgrep -f "phase4_modal.py::eval_selfmask" >/dev/null 2>&1; then
    sleep 4
    grep -qF "saved predictions" savepreds.log 2>/dev/null && { say "PREDS: SAVED (late)"; break; }
    say "FAILED — modal run exited without saving preds"; grep -iE "error|assert|traceback|exception" savepreds.log | tail -4; exit 1
  fi
  el=$((SECONDS-start))
  [ "$el" -ge 5400 ] && { say "RUNAWAY >90min; app stop ${APP}"; [ -n "$APP" ] && $MODAL app stop "$APP" --yes 2>/dev/null; exit 2; }
  [ $((el-last)) -ge 600 ] && { last=$el; say "HEARTBEAT ${el}s | $(tail -1 savepreds.log 2>/dev/null | cut -c1-90)"; }
  sleep 30
done
say "SUMMARY: $(grep -F 'saved predictions' savepreds.log | tail -1 | sed 's/.*saved/saved/')"
say "=== DONE ==="
