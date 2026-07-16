"""Cache DOWNSCALED (0.5x) web features per 2048 TEST tile — the big-tree arm.

Feeding the tile at half resolution makes a 256px crown look ~128px to the frozen
patch-16 detector (its 32-128px sweet spot), recovering the big trees the native
scale misses (recall >256px = 0.03, 128-256px = 0.36). One 1024 DINO pass per tile
(2048 downsampled to 1024) -> a 64x64 grid at 32 native-px/patch. INFERENCE-ONLY:
these features feed detection only; masks still come from the native EM.

Product: cache/<arm>/feat_test_down/<tid>.npy   (C,64,64) fp16

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.cache_test_down [--limit N]
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from dapt.backbone import FrozenDinoV3Features
from boxinst.cache_feats import LAYERS
from boxinst_commonality_tcd_04.prepare_test import OUT, TCD_TEST

Image.MAX_IMAGE_PIXELS = None
DOWN = 1024                          # 2048 -> 1024 (0.5x); 64x64 patch grid


def cache_dir(arm):
    return os.path.join(OUT, "cache", arm, "feat_test_down")


def down_feature(net, img):
    """PIL 2048 RGB -> (C,64,64) fp16 from one 1024 (0.5x) DINO pass."""
    small = img.resize((DOWN, DOWN), Image.BILINEAR)
    arr = np.asarray(small, np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1)[None]
    return net.extract(x)[0].to(torch.float16).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="web")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    gt = json.load(open(os.path.join(OUT, "test_gt.json")))
    tiles = sorted(gt)
    if args.limit:
        tiles = tiles[:args.limit]
    cdir = cache_dir(args.arm)
    os.makedirs(cdir, exist_ok=True)
    todo = [t for t in tiles if not os.path.exists(os.path.join(cdir, t + ".npy"))]
    print(f"[cache_test_down:{args.arm}] {len(todo)}/{len(tiles)} tiles", flush=True)
    if not todo:
        return
    net = FrozenDinoV3Features(args.arm, layers=LAYERS, device=args.device)
    for i, tid in enumerate(todo):
        img = Image.open(os.path.join(TCD_TEST, tid + ".tif")).convert("RGB")
        np.save(os.path.join(cdir, tid + ".npy"), down_feature(net, img))
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            print(f"  {i+1}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
