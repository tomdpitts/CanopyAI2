#!/bin/bash
# lace4px_simple adoption, Phase A render queue (waits for the alpha/kappa/bilinear test render chain, pid 51807)
cd /Users/tompitts/dphil/CanopyAI2; export PYTHONPATH=.
while kill -0 51807 2>/dev/null; do sleep 30; done
.venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.hunt.arms b29_simple_guide_controls boxinst_commonality_tcd_04/native_raster/hunt/arms_specs/b29_simple_guide_controls.json >> boxinst_commonality_tcd_04/native_raster/logs/hunt_b29_simple_guide_controls.log 2>&1
.venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.hunt.gtbox4 boxinst_commonality_tcd_04/native_raster/hunt/FROZEN_simple_2026-10-26.json lace4px_simple >> boxinst_commonality_tcd_04/native_raster/logs/hunt_adopt_gtbox4.log 2>&1
echo QUEUE1 DONE >> boxinst_commonality_tcd_04/native_raster/logs/hunt_adopt_queue1.log
