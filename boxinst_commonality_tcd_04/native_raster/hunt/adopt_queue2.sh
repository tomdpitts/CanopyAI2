#!/bin/bash
# lace4px_simple adoption, Phase A rescoring queue (waits for the detscore rescore of test, pid 51806)
cd /Users/tompitts/dphil/CanopyAI2; export PYTHONPATH=.
while kill -0 51806 2>/dev/null; do sleep 30; done
NPROC=4 .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.hunt.factors4 >> boxinst_commonality_tcd_04/native_raster/logs/hunt_adopt_factors4.log 2>&1
.venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.ablation_r2048.run_strata_guided --script t03 --res 2048 --tag lace4px_simple --out boxinst_commonality_tcd_04/ablation/results_r2048/strata_tile_product_r2048_lace4px_simple.json >> boxinst_commonality_tcd_04/native_raster/logs/hunt_adopt_strata_t03.log 2>&1
.venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.ablation_r2048.run_strata_guided --script t04 --res 2048 --tag lace4px_simple --out boxinst_commonality_tcd_04/ablation/results_r2048/strata_instance_r2048_lace4px_simple.json >> boxinst_commonality_tcd_04/native_raster/logs/hunt_adopt_strata_t04.log 2>&1
.venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.ablation_r2048.box_iou_strata_res_seamfix --rows seamfix --out boxinst_commonality_tcd_04/ablation/results_r2048/box_iou_strata_seamfix_r2048_GATE_rerun_2026-10-26.json >> boxinst_commonality_tcd_04/native_raster/logs/hunt_adopt_boxiou_gate.log 2>&1
.venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.ablation_r2048.box_iou_strata_res_seamfix --rows lace4px_simple --out boxinst_commonality_tcd_04/ablation/results_r2048/box_iou_strata_lace4px_simple_r2048.json >> boxinst_commonality_tcd_04/native_raster/logs/hunt_adopt_boxiou.log 2>&1
echo QUEUE2 DONE >> boxinst_commonality_tcd_04/native_raster/logs/hunt_adopt_queue2.log
