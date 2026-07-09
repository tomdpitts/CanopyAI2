#!/bin/zsh
# Ceiling sweep: isolate each anti-overfitting lever, box-only throughout.
# Fixed 50/50 val/test; train size varied via --n_train on the 340-crop pool.
cd /Users/tompitts/dphil/CanopyAI2
PY=.venv/bin/python
MID=boxinst_tcd/cache
G() { grep -v "RuntimeWarning\|matmul"; }

echo "== S0 deep tower2 n100 (baseline reproduce) =="
$PY -m boxinst_tcd.train --tag s0_deep_t2_n100 --det_tower 2 --n_train 100 \
    --epochs 300 --eval_every 60 2>&1 | G | tail -1

echo "== S1 deep tower2 n340 (data volume) =="
$PY -m boxinst_tcd.train --tag s1_deep_t2_n340 --det_tower 2 --n_train 340 \
    --epochs 300 --eval_every 60 2>&1 | G | tail -1

echo "== S2 mid tower2 n340 (data + mid layers) =="
$PY -m boxinst_tcd.train --tag s2_mid_t2_n340 --det_tower 2 --n_train 340 \
    --feat_dir $MID/feat_L3-6-9-12 --pair_dir $MID/pair_L3-6-9-12 \
    --comm_dir $MID/comm_L3-6-9-12 --epochs 300 --eval_every 60 2>&1 | G | tail -1

echo "== S3 mid tower0 n340 (+ lean probe head) =="
$PY -m boxinst_tcd.train --tag s3_mid_t0_n340 --det_tower 0 --n_train 340 \
    --feat_dir $MID/feat_L3-6-9-12 --pair_dir $MID/pair_L3-6-9-12 \
    --comm_dir $MID/comm_L3-6-9-12 --epochs 300 --eval_every 60 2>&1 | G | tail -1

echo "== S4 mid tower2 n100 (mid layers at low data) =="
$PY -m boxinst_tcd.train --tag s4_mid_t2_n100 --det_tower 2 --n_train 100 \
    --feat_dir $MID/feat_L3-6-9-12 --pair_dir $MID/pair_L3-6-9-12 \
    --comm_dir $MID/comm_L3-6-9-12 --epochs 300 --eval_every 60 2>&1 | G | tail -1

echo "== EVAL ALL =="
$PY -m boxinst_tcd.eval_masks s0_deep_t2_n100 s1_deep_t2_n340 s2_mid_t2_n340 \
    s3_mid_t0_n340 s4_mid_t2_n100 2>&1 | G | tail -8
echo "== SWEEP DONE =="
