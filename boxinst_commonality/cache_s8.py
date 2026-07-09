"""Half-stride (8px) DINO features: 4 shifted passes interleaved to 64x64.

DINO stays at patch-16 and the input stays the 512-padded tile; the canvas is
shifted by +-4px per axis so that every fine 8px cell k (centre 8k+4) gets a
16px-window feature centred on it. NO new pixel information (source imagery is
native 10cm GSD) — this is purely finer feature localisation.

Output: (4096, 64, 64) fp16 per tile in dapt/cache/web_last4_s8/.

Usage:
    .venv/bin/python -m boxinst_commonality.cache_s8
"""
import argparse
import json
import os

import numpy as np
import torch

from dapt.backbone import FrozenDinoV3Features, load_tile, pick_device
from dapt.cache_features import cache_key
from dapt.data.cohort import REPO
from boxinst.cache_feats import LAYERS

OUT_DIR = os.path.join(REPO, "dapt/cache/web_last4_s8")
FINE = 64
SHIFTS = (-4, 4)                 # -4 -> even fine rows/cols, +4 -> odd


def shifted(x, oy, ox):
    """canvas[y, x] = img[y + oy, x + ox], zero-padded out of range."""
    out = torch.zeros_like(x)
    ys, xs = max(0, -oy), max(0, -ox)
    yo, xo = max(0, oy), max(0, ox)
    h, w = x.shape[-2] - abs(oy), x.shape[-1] - abs(ox)
    out[..., ys:ys + h, xs:xs + w] = x[..., yo:yo + h, xo:xo + w]
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    split = json.load(open(os.path.join(REPO, "dapt/data/split.json")))
    tiles = list(split["tiles"])
    os.makedirs(OUT_DIR, exist_ok=True)
    todo = [p for p in tiles if not os.path.exists(
        os.path.join(OUT_DIR, cache_key(p) + ".npy"))]
    print(f"[s8] {len(todo)}/{len(tiles)} tiles to extract "
          f"(4 shifted passes each, layers={LAYERS})", flush=True)
    if not todo:
        return
    net = FrozenDinoV3Features("web", layers=LAYERS, device=args.device)
    for i, p in enumerate(todo):
        x, _ = load_tile(p)
        fine = None
        for iy, oy in enumerate(SHIFTS):
            for ix, ox in enumerate(SHIFTS):
                f = net.extract(shifted(x, oy, ox))[0].half().cpu()  # (C,32,32)
                if fine is None:
                    fine = torch.zeros(f.shape[0], FINE, FINE,
                                       dtype=torch.float16)
                fine[:, iy::2, ix::2] = f
        np.save(os.path.join(OUT_DIR, cache_key(p) + ".npy"), fine.numpy())
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            print(f"  {i+1}/{len(todo)}", flush=True)
    json.dump({"layers": list(LAYERS), "stride_px": 8, "shifts": SHIFTS,
               "note": "4 shifted passes interleaved; window still 16px"},
              open(os.path.join(OUT_DIR, "_meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
