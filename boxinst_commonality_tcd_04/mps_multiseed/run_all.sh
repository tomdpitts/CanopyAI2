#!/usr/bin/env bash
# Sequential 5-seed MPS variance run. ONE MPS job at a time (no parallelism —
# OOM risk, no speedup). Idempotent: existing checkpoints / eval jsons are skipped,
# so a killed run resumes where it stopped. Everything is written under
# mps_multiseed/ only.
set -u
cd /Users/tompitts/dphil/CanopyAI2 || exit 1
PY=.venv/bin/python
MOD=boxinst_commonality_tcd_04.mps_multiseed.run_seed
LOG=boxinst_commonality_tcd_04/mps_multiseed/logs
mkdir -p "$LOG"

echo "=== $(date '+%F %T') 5-seed MPS variance run START ==="
for s in 0 1 2 3 4; do
  echo "=== $(date '+%F %T') seed $s START (train+eval) ==="
  PYTORCH_ENABLE_MPS_FALLBACK=1 $PY -u -m "$MOD" --seed "$s" --stage both \
      >"$LOG/seed_$s.log" 2>&1
  rc=$?
  echo "=== $(date '+%F %T') seed $s DONE rc=$rc ==="
  if [ $rc -ne 0 ]; then
    echo "!!! seed $s FAILED (rc=$rc) — see $LOG/seed_$s.log; continuing ==="
  fi
done
echo "=== $(date '+%F %T') ALL SEEDS COMPLETE ==="
