#!/usr/bin/env bash
# Autonomous overnight orchestrator for the 4-phase L24 seed-0 go/no-go.
# Stage 1: wait for the already-running extract_4p to finish (poll extract.log).
# Stage 2: auto-launch train_eval_4p (seed 0), wait, parse mask/box mAP50.
# Emits timestamped HEARTBEAT lines every 15 min (elapsed + spend est + last log line)
# so a hung/runaway job is caught, not silent. Per-stage deadlines kill runaways.
# Everything isolated under this folder; Modal's own 6h function timeouts are a backstop.
set -u
cd "$(dirname "$0")" || exit 1
MODAL=../../../.venv/bin/modal
TARGET=0.502                       # interp-L24 mask mAP50 to beat
RATE=0.035                         # A100 $/min, for spend estimate

ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*"; }

# wait_stage NAME LOGFILE SUCCESS_TOKEN DEADLINE_S PROC_PATTERN
wait_stage() {
  local name=$1 log=$2 token=$3 deadline=$4 proc=$5
  local start=$SECONDS last_hb=0 el
  while true; do
    if grep -qF "$token" "$log" 2>/dev/null; then say "$name: SUCCESS"; return 0; fi
    if ! pgrep -f "$proc" >/dev/null 2>&1; then
      sleep 4
      if grep -qF "$token" "$log" 2>/dev/null; then say "$name: SUCCESS (late)"; return 0; fi
      say "$name: FAILED — 'modal run' exited without the success marker"
      grep -iE "error|assert|traceback|exception" "$log" 2>/dev/null | tail -4
      return 1
    fi
    el=$((SECONDS - start))
    if [ "$el" -ge "$deadline" ]; then
      say "$name: RUNAWAY — exceeded $((deadline/60))min; killing '$proc'"
      pkill -f "$proc"; return 2
    fi
    if [ $((el - last_hb)) -ge 900 ]; then
      last_hb=$el
      say "HEARTBEAT $name elapsed=$((el/60))min spend≈\$$(awk "BEGIN{printf \"%.1f\", $el/60*$RATE}") | $(tail -1 "$log" 2>/dev/null | cut -c1-110)"
    fi
    sleep 60
  done
}

say "=== overnight orchestrator START (goal: beat mask mAP50 $TARGET) ==="

# --- Stage 1: extraction (already running; poll its log) ---
say "Stage1 EXTRACT: waiting for extract_4p (deadline 4h)"
wait_stage EXTRACT extract.log "done train=900 test=439 native=439" 14400 \
  "phase4_modal.py::extract_4p"
rc=$?
if [ $rc -ne 0 ]; then say "ABORT: extraction did not complete (rc=$rc). Not launching train_eval."; exit 1; fi
say "extraction complete: $(grep 'extract_4p] done' extract.log | tail -1)"

# --- Stage 2: train + eval seed 0 (auto-launch) ---
say "Stage2 TRAIN_EVAL: launching train_eval_4p seed 0"
nohup $MODAL run phase4_modal.py::train_eval_4p --seed 0 > train_eval.log 2>&1 &
sleep 5
wait_stage TRAIN_EVAL train_eval.log '"mask_mAP50"' 14400 \
  "phase4_modal.py::train_eval_4p"
rc=$?
if [ $rc -ne 0 ]; then say "ABORT: train_eval did not complete (rc=$rc)."; exit 1; fi

# --- parse + verdict ---
mask=$(grep -oE '"mask_mAP50": [0-9.]+' train_eval.log | head -1 | grep -oE '[0-9.]+$')
box=$(grep -oE '"box_mAP50": [0-9.]+' train_eval.log | head -1 | grep -oE '[0-9.]+$')
verdict=$(awk "BEGIN{print ($mask>$TARGET)?\"BEATS\":\"below\"}")
say "SUMMARY: 4phase-L24 seed0 mask_mAP50=$mask box_mAP50=$box | target(interp-L24)=$TARGET -> $verdict"
say "=== overnight orchestrator DONE ==="
