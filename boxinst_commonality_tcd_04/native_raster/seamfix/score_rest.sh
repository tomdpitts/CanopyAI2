#!/bin/zsh
# Score the remaining seam-fix rows (2026-10-14). Never overwrites (score_coco has no guard -> test -e).
set -u
cd /Users/tompitts/dphil/CanopyAI2
export VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2
SF=boxinst_commonality_tcd_04/native_raster/seamfix
S() { [ -e $SF/$4 ] && { echo "exists $4"; return; }; .venv/bin/python -m boxinst_commonality_tcd_04.score_coco --preds $SF/$1 --gt $2 --res 2048 --score_floor 0.05 --max_dets $3 --out $SF/$4; }
T=boxinst_commonality_tcd_04/test_gt.json
S preds_selvabox_seamfix_off_r2048.json $T 600  scored_selvabox_seamfix_off_r2048_md600.json
S preds_selvabox_seamfix_on_r2048.json  $T 600  scored_selvabox_seamfix_on_r2048_md600.json
S preds_selvabox_seamfix_off_r2048.json $T 1600 scored_selvabox_seamfix_off_r2048_md1600.json
S preds_selvabox_seamfix_on_r2048.json  $T 1600 scored_selvabox_seamfix_on_r2048_md1600.json
S preds_dt2_sparse_s0v2_r2048_seamfix.json data/tcd_sparse/sparse_gt.json 600 scored_dt2_sparse_s0v2_r2048_seamfix.json
echo SCORE_DONE
