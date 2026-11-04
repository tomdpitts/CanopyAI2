#!/bin/zsh
# SAM 3 on D1 boxes: test -> sparse -> score both. Resumable (re-run this script). Logs in native_raster/logs/.
set -e
cd /Users/tompitts/dphil/CanopyAI2
M=boxinst_commonality_tcd_04.dino_det.sam3_on_d1; L=boxinst_commonality_tcd_04/native_raster/logs
PYTHONPATH=. .venv/bin/python -u -m $M run --split test   >> $L/sam3_d1_test.log 2>&1
PYTHONPATH=. .venv/bin/python -u -m $M run --split sparse >> $L/sam3_d1_sparse.log 2>&1
PYTHONPATH=. .venv/bin/python -u -m $M score --split test   >> $L/sam3_d1_score_test.log 2>&1
PYTHONPATH=. .venv/bin/python -u -m $M score --split sparse >> $L/sam3_d1_score_sparse.log 2>&1
echo "[run_sam3_d1] ALL DONE $(date)" >> $L/sam3_d1_test.log
