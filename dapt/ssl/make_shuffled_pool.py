"""Patch-shuffle control pool: destroy coherent scene/crown structure, KEEP the arid
pixel-statistics marginals (per-channel color/contrast/texture blocks intact).

Each 512x512 pool tile is cut into BLOCK x BLOCK tiles and the blocks are randomly
permuted (seeded per source tile, deterministic). A DAPT run on this pool sees arid
color/texture but no coherent overhead scene — so if it STILL matches the real DAPT
gain, the gain is low-level arid stats; if it drops to web, the gain needs coherent
arid structure. Either way isolates what adaptation buys.

Usage:
    .venv/bin/python -m dapt.ssl.make_shuffled_pool --block 64
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image

from dapt.data.cohort import REPO

SRC = os.path.join(REPO, "dapt/ssl/pool/tiles")
DST = os.path.join(REPO, "dapt/ssl/pool_shuffled/tiles")


def shuffle_tile(arr, block, rng):
    H, W = arr.shape[:2]
    nby, nbx = H // block, W // block
    out = arr.copy()
    perm = rng.permutation(nby * nbx)
    src_blocks = [arr[by * block:(by + 1) * block, bx * block:(bx + 1) * block]
                  for by in range(nby) for bx in range(nbx)]
    for dst_i, src_i in enumerate(perm):
        dy, dx = divmod(dst_i, nbx)
        out[dy * block:(dy + 1) * block, dx * block:(dx + 1) * block] = src_blocks[src_i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    tiles = sorted(glob.glob(os.path.join(SRC, "*", "*.png")))
    print(f"{len(tiles)} tiles, block={args.block}px -> {os.path.relpath(DST, REPO)}")
    n = 0
    for i, p in enumerate(tiles):
        rel = os.path.relpath(p, SRC)
        dst = os.path.join(DST, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        arr = np.asarray(Image.open(p).convert("RGB"))
        # per-tile deterministic rng keyed on index (order is sorted+stable)
        rng = np.random.default_rng(args.seed * 100003 + i)
        Image.fromarray(shuffle_tile(arr, args.block, rng)).save(dst)
        n += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(tiles)}", flush=True)
    print(f"wrote {n} shuffled tiles")


if __name__ == "__main__":
    main()
