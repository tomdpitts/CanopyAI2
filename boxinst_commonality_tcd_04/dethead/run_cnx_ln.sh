#!/bin/bash
cd /Users/tompitts/dphil/CanopyAI2; export PYTHONPATH=.
LOG=boxinst_commonality_tcd_04/native_raster/logs
.venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.train_cnx --tag cnxl_ln_s0 >> $LOG/dethead_train_cnxl_ln_s0.log 2>&1 || { echo "CNX TRAIN FAILED"; exit 1; }
echo "$(date '+%F %T') CNX trained"
.venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.cnx_eval select --tag cnxl_ln_s0 >> $LOG/dethead_select_cnxl_ln_s0.log 2>&1 || { echo "CNX SELECT FAILED"; exit 1; }
echo "$(date '+%F %T') CNX selected $(python3 -c "import json;print(json.load(open('boxinst_commonality_tcd_04/dethead/results/select_val_cnxl_ln_s0.json'))['selected'])")"
.venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.cnx_eval test --tag cnxl_ln_s0 >> $LOG/dethead_test_cnxl_ln_s0.log 2>&1 || { echo "CNX TEST FAILED"; exit 1; }
echo "$(date '+%F %T') CNX tested; ALL DONE"
