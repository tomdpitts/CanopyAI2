#!/bin/sh
# 4-shift and 1-shift LACE timing arms, then a third SAM 3 pass (drift check). Run from masker_cost/.
export PYTHONPATH=/Users/tompitts/dphil/CanopyAI2
PYF=/Users/tompitts/dphil/feat_cache/venv_feat_t457/bin/python
$PYF -u bench.py lace4 1 > logs/lace4_p1.log 2>&1 && $PYF -u bench.py lace1 1 > logs/lace1_p1.log 2>&1 && ../../.venv/bin/python -u bench.py sam 3 > logs/sam_p3.log 2>&1
echo "chain exit $?" > logs/chain2.log
