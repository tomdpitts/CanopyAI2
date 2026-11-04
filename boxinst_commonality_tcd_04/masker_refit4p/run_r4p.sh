#!/bin/bash
# R4P chain (pre-registration README.md): fit n120 -> val n120 -> (wait for 250-tile pull) fit n250 -> val n250 -> select.
# Stops before `final` (test/sparse), which is run separately after the val result is reported. Every stage is resume-safe.
set -e -o pipefail
cd /Users/tompitts/dphil/CanopyAI2
PY=.venv/bin/python; R=boxinst_commonality_tcd_04/masker_refit4p; Z=/Users/tompitts/dphil/feat_cache/z4p_train
fit() { PYTHONPATH=$R $PY -u -c "
import fit4p
fit4p.fit_nopca('$Z', 'boxinst_commonality_tcd_04/train_tiles_gt.json', $1, '/Users/tompitts/dphil/feat_cache/em_model_4p_fix.npz', '$R/results/fit_n$1')" 2>&1 | grep --line-buffered -v -E "RuntimeWarning|^  (zc|a|return|bg_all|newC|neg) "; }
val() { [ -f $R/results/val_$1.json ] || PYTHONPATH=. $PY -u -m boxinst_commonality_tcd_04.masker_refit4p.eval4p val --tag $1 2>&1 | grep --line-buffered -v -E "RuntimeWarning|^  (zc|a|return) "; }
echo "$(date '+%F %T') R4P start"
fit 120; val n120
while [ "$(ls $Z/*.npy | wc -l | tr -d ' ')" -lt 250 ] || pgrep -f z4p_train/_pull.sh >/dev/null; do sleep 30; done
echo "$(date '+%F %T') pull complete: $(ls $Z/*.npy | wc -l) tiles"
fit 250; val n250
[ -f $R/results/select.json ] || PYTHONPATH=. $PY -u -m boxinst_commonality_tcd_04.masker_refit4p.eval4p select
echo "$(date '+%F %T') R4P val stage DONE"
