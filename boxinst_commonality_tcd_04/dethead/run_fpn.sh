#!/bin/bash
# pre-registration 2 driver (2026-11-02): FPN train -> val selection (FPN, flat FCOS) -> one test run each at val-selected NMS
cd /Users/tompitts/dphil/CanopyAI2; export PYTHONPATH=.
LOG=boxinst_commonality_tcd_04/native_raster/logs; R=boxinst_commonality_tcd_04/dethead/results
.venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.train --head fpn --tag fpn_s0 >> $LOG/dethead_train_fpn_s0.log 2>&1 || { echo "FPN TRAIN FAILED"; exit 1; }
echo "$(date '+%F %T') FPN trained"
for x in "fpn_s0 fpn" "fcos_s0 fcos"; do set -- $x
  .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.select_val --tag $1 --head $2 >> $LOG/dethead_select_$1.log 2>&1 || { echo "SELECT $1 FAILED"; exit 1; }
  NMS=$(python3 -c "import json;print(json.load(open('$R/select_val_$1.json'))['nms'])")
  echo "$(date '+%F %T') $1 val-selected nms $NMS"
  .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.test run --tag $1 --head $2 --nms $NMS >> $LOG/dethead_test_$1_selected.log 2>&1 || { echo "TEST $1 FAILED"; exit 1; }
  echo "$(date '+%F %T') $1 tested"
done
echo "$(date '+%F %T') ALL DONE"
