#!/bin/bash
# Local (macOS, MPS) toolchain for the DINO-Swin-L smoke test. Isolated venv OUTSIDE the repo; does not touch .venv / .venv_df.
set -euo pipefail
W=/Users/tompitts/dphil/dino_local
PY=/Users/tompitts/.pyenv/versions/3.10.20/bin/python
cd $W
[ -d venv ] || $PY -m venv venv
V=$W/venv/bin
$V/pip install -q --upgrade pip wheel "setuptools<81"   # detrex imports pkg_resources (removed in setuptools 81)
$V/pip install -q torch torchvision "numpy<2" opencv-python-headless pycocotools pillow timm scipy huggingface_hub
[ -d detrex ] || git clone --recursive https://github.com/IDEA-Research/detrex.git detrex
echo "detrex $(git -C detrex rev-parse --short HEAD); detectron2 $(git -C detrex/detectron2 rev-parse --short HEAD)"
$V/pip install --no-build-isolation -e detrex/detectron2
$V/pip install --no-build-isolation -e detrex
$V/python -c "import torch, detectron2, detrex; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available(), 'd2', detectron2.__version__)"
echo INSTALL_DONE
