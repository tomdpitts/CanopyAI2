#!/bin/zsh
# A6 launcher: wait for the 236 sparse feature files, run A6 (gate first), then A5 strata on the sparse output.
set -u
cd /Users/tompitts/dphil/CanopyAI2
export PYTHONPATH=. VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2
F=/Users/tompitts/dphil/feat_cache/feat_4p_sparse_dl/feat_4p_sparse; T=boxinst_commonality_tcd_04/native_raster/tierA
until [ $(find $F -name "*.npy" -size +130000k 2>/dev/null | wc -l) -ge 236 ] && ! pgrep -f "volume get tcd-sparse-vol" >/dev/null; do sleep 30; done
echo "[a6] features complete $(date)"
.venv/bin/python -W ignore::RuntimeWarning -m boxinst_commonality_tcd_04.native_raster.tierA.a6_sparse_gtbox --feat_dir $F || { echo "[a6] FAILED"; exit 1; }
.venv/bin/python -m boxinst_commonality_tcd_04.native_raster.tierA.a5_strata --gt data/tcd_sparse/sparse_gt.json \
   --sam_dir $T/a6_tiles --lace $T/a6_lace_sparse.jsonl --img_dir data/tcd_sparse/test --out $T/a6_strata.json
echo "[a6] ALL_DONE $(date)"
