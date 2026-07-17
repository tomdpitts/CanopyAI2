#!/usr/bin/env bash
# Upload the RGB-only NEON data to the Modal Volume for the H100 run.
# Uploads: train patches (PNG) + GT, the 194 eval RGB tiles + GT, and the parity ref.
# NO LiDAR/CHM/HSI is staged or uploaded.
set -eu
cd "$(dirname "$0")"
PY=../../.venv/bin/python
MODAL=../../.venv/bin/modal
VOL=neon-multiseed-vol

[ -d train_patches ] || { echo "no train_patches/ — run prepare_neon_train.py first"; exit 1; }

echo "== staging 194 eval RGB tiles =="
rm -rf eval_rgb_stage && mkdir -p eval_rgb_stage
$PY - <<'PY'
import json, os, shutil
gt = json.load(open("neon_gt.json"))
src = "NeonTreeEvaluation/evaluation/RGB"
n = 0
for p in gt:
    shutil.copy(os.path.join(src, p + ".tif"), "eval_rgb_stage/"); n += 1
print(f"staged {n} eval tiles")
PY

echo "== uploading to $VOL =="
$MODAL volume put -f "$VOL" train_patches /train_patches
$MODAL volume put -f "$VOL" eval_rgb_stage /eval_rgb
$MODAL volume put -f "$VOL" train_patches_gt.json /train_patches_gt.json
$MODAL volume put -f "$VOL" neon_gt.json /neon_gt.json
$MODAL volume put -f "$VOL" ref_feat.npz /ref_feat.npz
echo "UPLOAD DONE"
