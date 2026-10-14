#!/bin/zsh
# Seam fix, 2026-10-14. Gate (filter OFF, byte-compare vs published) then fixed (edge_px 10) per row.
# Gate outputs are byte-duplicates of published files -> scratchpad; only *_GATE.json is kept in the repo.
set -u
cd /Users/tompitts/dphil/CanopyAI2
export VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv/bin/python; M=boxinst_commonality_tcd_04.native_raster.run_stitch_seam
NR=boxinst_commonality_tcd_04/native_raster; SF=$NR/seamfix; GD=/private/tmp/claude-501/-Users-tompitts-dphil-CanopyAI2/b085cec2-62cb-4a30-a5e2-6f6b537b156d/scratchpad/seam_gate
mkdir -p $GD
run() {  # name raw gt ref model
  local name=$1 raw=$2 gt=$3 ref=$4 model=$5
  if [ ! -f $SF/preds_${name}_r2048_GATE.json ]; then
    $PY -m $M --pred_dir "$raw" --gt "$gt" --res 2048 --min_score 0.05 --workers 12       --model "GATE filter-off $model" --out $GD/preds_${name}_r2048.json --gate_ref "$ref"       && cp $GD/preds_${name}_r2048_GATE.json $SF/ || return 1
  fi
  [ -f $SF/preds_${name}_r2048_seamfix.json ] || $PY -m $M --pred_dir "$raw" --gt "$gt" --res 2048       --min_score 0.05 --edge_px 10 --workers 12 --model "$model; seam-fixed (interior-edge filter 10 px before dedup)"       --out $SF/preds_${name}_r2048_seamfix.json
}
R=/Users/tompitts/dphil/feat_cache
run box2mask_r50_s0 $R/b2m_raw/r50/pred_raw_box2mask_s0_test boxinst_commonality_tcd_04/test_gt.json $NR/preds_box2mask_r50_s0_r2048.json "Box2Mask-T R-50 (BoxInstSeg, TPAMI 2024), COCO-box-pretrained, box-only, canopy-blacked training, 1024@50% subtiles, clean_crowns dedup; re-stitched at native 2048"
run box2mask_swinl_s0 $R/b2m_raw/swinl/pred_raw_box2mask_swin-l_s0_test boxinst_commonality_tcd_04/test_gt.json $NR/preds_box2mask_swinl_s0_r2048.json "Box2Mask-T Swin-L (BoxInstSeg, TPAMI 2024), COCO-box-pretrained, box-only, canopy-blacked training, 1024@50% subtiles, clean_crowns dedup; re-stitched at native 2048"
run dt2_s0v2 $R/dt2_raw/s0v2/pred_raw_s0v2 boxinst_commonality_tcd_04/test_gt.json $NR/preds_dt2_s0v2_r2048.json "DetecTree2 (Mask R-CNN R101-FPN) fine-tuned on 792/108, 1024@50% subtiles, clean_crowns dedup; re-stitched at native 2048 (s0v2)"
run dt2_sparse_s0v2 $R/dt2_raw/sparse_s0v2/pred_raw_sparse_s0v2 data/tcd_sparse/sparse_gt.json $NR/preds_dt2_sparse_s0v2_r2048.json "DetecTree2 (Mask R-CNN R101-FPN) fine-tuned on 792/108, 1024@50% subtiles, clean_crowns dedup; sparse-236, re-stitched at native 2048 (s0v2)"
echo ALL_DONE
