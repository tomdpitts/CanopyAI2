"""Train Detector8 on a compressed-feature variant WITHOUT touching the parent.

Thin wrapper over boxinst_commonality_tcd_04.train_detector_tiles: monkeypatches
the module's ART (checkpoints -> feat_ablation/artifacts/) and feature loading
(variants.load_feat), then calls its train() verbatim — identical recipe,
losses, val-selection, and incremental best-checkpointing. in_dim is inferred
from the loaded features (256 / 1024 / 4096).

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.feat_ablation.train_variant \
        --variant pca256 --epochs 40 --bs 3 --eval_every 5 --seed 0
"""
import argparse
import os

import numpy as np
import torch

import boxinst_commonality_tcd_04.train_detector_tiles as tdt
from boxinst_commonality_tcd_04.feat_ablation.variants import (VARIANTS,
                                                               feat_dir,
                                                               load_feat)

HERE = os.path.abspath(os.path.dirname(__file__))


def patch(variant):
    tdt.ART = os.path.join(HERE, "artifacts")     # checkpoints stay in-folder
    tdt.cache_dir = lambda arm: feat_dir(variant, "feat_traintile")

    def _feat(self, t):
        return torch.from_numpy(load_feat(variant, "feat_traintile", t))
    tdt.TileData._feat = _feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--tower", type=int, default=3)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    args.tag = f"fa_{args.variant}"
    args.arm = args.variant                        # recorded in the ckpt cfg
    patch(args.variant)
    tdt.train(args)


if __name__ == "__main__":
    main()
