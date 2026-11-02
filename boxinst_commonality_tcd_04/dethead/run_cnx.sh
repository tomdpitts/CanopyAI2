#!/bin/bash
# pre-registration 3 driver (2026-11-02): ConvNeXt-L features (val, train) -> ATSS/GFL train -> val selection -> single test
cd /Users/tompitts/dphil/CanopyAI2; export PYTHONPATH=.
LOG=boxinst_commonality_tcd_04/native_raster/logs
while pgrep -f "dethead.cnx extract --split val" >/dev/null || pgrep -f run_fpn.sh >/dev/null; do sleep 60; done
echo "$(date '+%F %T') GPU free; extracting train (disk guard 15 GB)"
.venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.cnx extract --split val >> $LOG/cnxl_extract_val_2026-11-02.log 2>&1 || { echo "VAL EXTRACT FAILED"; exit 1; }
.venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.cnx extract --split train >> $LOG/cnxl_extract_train_2026-11-02.log 2>&1 || { echo "TRAIN EXTRACT STOPPED (see log: disk guard?)"; exit 1; }
echo "$(date '+%F %T') features done"
.venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.train_cnx --tag cnxl_s0 >> $LOG/dethead_train_cnxl_s0.log 2>&1 || { echo "CNX TRAIN FAILED"; exit 1; }
echo "$(date '+%F %T') CNX trained"
.venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.cnx_eval select --tag cnxl_s0 >> $LOG/dethead_select_cnxl_s0.log 2>&1 || { echo "CNX SELECT FAILED"; exit 1; }
echo "$(date '+%F %T') CNX selected $(python3 -c "import json;print(json.load(open('boxinst_commonality_tcd_04/dethead/results/select_val_cnxl_s0.json'))['selected'])")"
.venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.cnx_eval test --tag cnxl_s0 >> $LOG/dethead_test_cnxl_s0.log 2>&1 || { echo "CNX TEST FAILED"; exit 1; }
echo "$(date '+%F %T') CNX tested; ALL DONE"
