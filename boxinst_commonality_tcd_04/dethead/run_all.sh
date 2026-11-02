#!/bin/bash
# dethead end-to-end driver (2026-11-02): wait for train features -> FCOS train/test -> CenterNet local control train/test
cd /Users/tompitts/dphil/CanopyAI2; export PYTHONPATH=.
LOG=boxinst_commonality_tcd_04/native_raster/logs
D=/Users/tompitts/dphil/feat_cache/feat_4p_train
until [ "$(ls $D | grep -c 'npy$')" -ge 792 ] && ! pgrep -f _pull.sh >/dev/null; do sleep 60; done
echo "$(date '+%F %T') features ready: $(ls $D | grep -c 'npy$')"
.venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.train --head fcos --tag fcos_s0 >> $LOG/dethead_train_fcos_s0.log 2>&1 || { echo "FCOS TRAIN FAILED"; exit 1; }
echo "$(date '+%F %T') FCOS trained"
.venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.test run --tag fcos_s0 --head fcos >> $LOG/dethead_test_fcos_s0.log 2>&1 || { echo "FCOS TEST FAILED"; exit 1; }
echo "$(date '+%F %T') FCOS tested"
.venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.train --head centernet --tag cn_local_s0 >> $LOG/dethead_train_cn_local_s0.log 2>&1 || { echo "CN TRAIN FAILED"; exit 1; }
echo "$(date '+%F %T') CN trained"
.venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.test run --tag cn_local_s0 --head centernet >> $LOG/dethead_test_cn_local_s0.log 2>&1 || { echo "CN TEST FAILED"; exit 1; }
echo "$(date '+%F %T') ALL DONE"
