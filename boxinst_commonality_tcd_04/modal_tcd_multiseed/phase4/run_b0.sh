#!/usr/bin/env bash
# Chain the beta=0 payoff: wait for the (already-running) CPU fit, then auto-launch the
# A100 eval_selfmask --beta 0, parse the real-pipeline mask mAP50. Heartbeats + deadlines.
set -u
cd "$(dirname "$0")" || exit 1
MODAL=../../../.venv/bin/modal
ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*"; }

wait_stage() {   # NAME LOG TOKEN DEADLINE_S PROC
  local name=$1 log=$2 token=$3 deadline=$4 proc=$5 start=$SECONDS last=0 el
  while true; do
    if grep -qF "$token" "$log" 2>/dev/null; then say "$name: SUCCESS"; return 0; fi
    if ! pgrep -f "$proc" >/dev/null 2>&1; then
      sleep 4
      grep -qF "$token" "$log" 2>/dev/null && { say "$name: SUCCESS (late)"; return 0; }
      say "$name: FAILED — 'modal run' exited without success marker"
      grep -iE "error|assert|traceback|exception" "$log" | tail -4; return 1
    fi
    el=$((SECONDS-start))
    [ "$el" -ge "$deadline" ] && { say "$name: RUNAWAY >$((deadline/60))min; killing"; pkill -f "$proc"; return 2; }
    [ $((el-last)) -ge 600 ] && { last=$el; say "HEARTBEAT $name ${el}s | $(tail -1 "$log" 2>/dev/null | cut -c1-100)"; }
    sleep 30
  done
}

say "=== beta=0 payoff START (goal: box-0.605 reaches mask mAP; refs: b0.5 self 0.449, fixed 0.504) ==="
say "FIT: waiting for fit_masker_4p --beta 0 (deadline 45min)"
wait_stage FIT fit_b0.log "done in" 2700 "fit_masker_4p" || { say "ABORT: fit failed"; exit 1; }

say "EVAL: launching eval_selfmask --beta 0 (A100)"
nohup $MODAL run phase4_modal.py::eval_selfmask --beta 0 > eval_b0.log 2>&1 &
sleep 5
wait_stage EVAL eval_b0.log '"mask_mAP50"' 7200 "eval_selfmask" || { say "ABORT: eval failed"; exit 1; }

mask=$(grep -oE '"mask_mAP50": [0-9.]+' eval_b0.log | head -1 | grep -oE '[0-9.]+$')
box=$(grep -oE '"box_mAP50": [0-9.]+' eval_b0.log | head -1 | grep -oE '[0-9.]+$')
gap=$(grep -oE '"box_minus_mask_mAP50": [0-9.-]+' eval_b0.log | head -1 | grep -oE '[0-9.-]+$')
say "SUMMARY: 4phase-L24 seed0 beta=0 self-mask -> mask=$mask box=$box box->mask=$gap | vs b0.5 self 0.449, fixed 0.504"
say "=== beta=0 payoff DONE ==="
