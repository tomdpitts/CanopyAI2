#!/bin/zsh
# Tier A launcher (approved 2026-10-14): waits for A4 (gtbox_sam_persub.json), then
#   A3 ellipse (CPU)  ->  A1 seeds 0,1,2 (MPS, sequential)  ||  A2 (CPU pool, after its gate and the 439 feature files)
set -u
cd /Users/tompitts/dphil/CanopyAI2
export PYTHONPATH=. VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
PY=.venv/bin/python; NR=boxinst_commonality_tcd_04/native_raster; L=$NR/logs
FEAT=/Users/tompitts/dphil/feat_cache/feat_4p_test_dl/feat_4p_test
until [ -f $NR/audit_2026-10-14/selvabox_evidence/gtbox_sam_persub.json ]; do sleep 30; done
echo "[tierA] A4 done $(date)"
$PY -m boxinst_commonality_tcd_04.native_raster.tierA.a3_ellipse > $L/tierA_a3.log 2>&1 &
( for s in 0 1 2; do VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 $PY -m boxinst_commonality_tcd_04.native_raster.tierA.a1_lace_sam3_persub --seed $s > $L/tierA_a1_s$s.log 2>&1 || exit 1; done ) &
# A2: all 439 feature files present at full size (128 MiB each) and the download process finished
until [ $(find $FEAT -name "*.npy" -size +130000k 2>/dev/null | wc -l) -ge 439 ] && ! pgrep -f "volume get tcd04-phase4-vol feat_4p_test" >/dev/null; do sleep 30; done
echo "[tierA] features complete $(date)"
$PY -m boxinst_commonality_tcd_04.native_raster.tierA.a2_selvabox_lace_masker --feat_dir $FEAT --workers 8 > $L/tierA_a2.log 2>&1
echo "[tierA] A2 exit $? $(date)"
wait
echo "[tierA] ALL_DONE $(date)"
