#!/usr/bin/env bash
set -u
cd /Users/tompitts/dphil/CanopyAI2 || exit 1
PY=.venv/bin/python
MOD=boxinst_commonality_tcd_04.mps_multiseed.eval_pr
LOG=boxinst_commonality_tcd_04/mps_multiseed/logs
for s in 1 2 3 4; do
  echo "=== $(date '+%F %T') PR seed $s START ==="
  $PY -u -m "$MOD" --seed "$s" > "$LOG/pr_s$s.log" 2>&1
  echo "=== $(date '+%F %T') PR seed $s DONE rc=$? ==="
done
echo "=== $(date '+%F %T') PR ALL DONE ==="
