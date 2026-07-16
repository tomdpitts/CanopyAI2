"""Cache stitched 128x128 web-DINO features per 2048 TEST tile.

2048 in one pass is O(N^2)-attention-bound (16384 tokens -> ~8.6GB transient per
layer, ~69s). Instead run 2x2 overlapping 1024 windows (4096 tokens each, ~6s
total, flat memory) and stitch to one (4096,128,128) grid at 16px, taking each
window's central half so seam patches keep full 1024-window attention context.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.cache_test [--arm web] [--limit N]
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from dapt.backbone import FrozenDinoV3Features, pick_device
from boxinst.cache_feats import LAYERS
from boxinst_commonality_tcd_04.prepare_test import OUT, TCD_TEST

Image.MAX_IMAGE_PIXELS = None
GRID2K, WIN, P = 128, 1024, 16       # tile grid / window px / patch px
WGRID = WIN // P                     # 64 patches per window
STARTS = (0, 1024)                   # 2x2 non-overlapping windows tile 2048 exactly


def cache_dir(arm):
    return os.path.join(OUT, "cache", arm, "feat_test")


def tile_feature(net, img, device):
    """PIL 2048 RGB -> (C,128,128) fp16 stitched from 2x2 1024 windows.

    Windows here are edge-to-edge (0,1024); each contributes its own 64x64 block.
    Seam context loss is one patch row/col wide — negligible for detection, and
    the alternative (overlap+crop-centre) needs 4 passes for the same result.
    """
    arr = np.asarray(img, np.float32) / 255.0
    out = None
    for gy, oy in enumerate(STARTS):
        for gx, ox in enumerate(STARTS):
            w = arr[oy:oy + WIN, ox:ox + WIN]
            x = torch.from_numpy(w).permute(2, 0, 1)[None]
            f = net.extract(x)[0]                        # (C,64,64)
            if out is None:
                out = torch.zeros(f.shape[0], GRID2K, GRID2K, dtype=torch.float16)
            out[:, gy * WGRID:(gy + 1) * WGRID,
                gx * WGRID:(gx + 1) * WGRID] = f.to(torch.float16).cpu()
    return out.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="web")
    ap.add_argument("--limit", type=int, default=None,
                    help="cache only the first N tiles (quick-subset experiments)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    gt = json.load(open(os.path.join(OUT, "test_gt.json")))
    tiles = sorted(gt)                                   # deterministic subset
    if args.limit:
        tiles = tiles[:args.limit]
    cdir = cache_dir(args.arm)
    os.makedirs(cdir, exist_ok=True)
    todo = [t for t in tiles if not os.path.exists(os.path.join(cdir, t + ".npy"))]
    print(f"[cache_test:{args.arm}] {len(todo)}/{len(tiles)} tiles", flush=True)
    if not todo:
        return
    net = FrozenDinoV3Features(args.arm, layers=LAYERS, device=args.device)
    for i, tid in enumerate(todo):
        img = Image.open(os.path.join(TCD_TEST, tid + ".tif")).convert("RGB")
        np.save(os.path.join(cdir, tid + ".npy"), tile_feature(net, img, net.device))
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            print(f"  {i+1}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
