"""Cache (a) last-4-block DINOv3 features and (b) DINO pairwise-affinity maps.

The boxinst prototype uses the LAST four transformer blocks (21-24 of ViT-L's 24),
per the one-shot spec ("concatenate the last ~4 hidden layers"). NOTE this diverges
from the frozen dapt probe protocol (blocks 3/6/9/12) — deliberate, and reported.

Stage 1: frozen backbone pass per tile -> (4096, 32, 32) fp16 in dapt/cache/web_last4/.
Stage 2 (no backbone): bilinearly upsample cached features to the 128x128 mask grid,
re-L2-normalize per pixel, and store neighbour cosine similarity for 4 unique
directions x dilations (2, 4) -> (8, 128, 128) fp16 in dapt/cache/pairwise_last4/.
Direction order: [(0,2),(2,0),(2,2),(2,-2)] at d=2, then the same at d=4 (dy,dx in
mask-grid px; 1 mask px = 4 image px). Also prints the similarity distribution so the
pairwise threshold tau can be chosen without ever looking at masks.

Usage:
    .venv/bin/python -m boxinst.cache_feats            # both stages
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from dapt.backbone import FrozenDinoV3Features, load_tile, pick_device
from dapt.cache_features import cache_key
from dapt.data.cohort import REPO

LAYERS = (21, 22, 23, 24)          # last 4 blocks of ViT-L/16
FEAT_DIR = os.path.join(REPO, "dapt/cache/web_last4")
PAIR_DIR = os.path.join(REPO, "dapt/cache/pairwise_last4")
MASK_RES = 128                     # 512 / 4  (stride-4 mask grid)
OFFSETS = [(0, 2), (2, 0), (2, 2), (2, -2),
           (0, 4), (4, 0), (4, 4), (4, -4)]   # (dy,dx) in mask px


def cache_features(tiles, device=None):
    os.makedirs(FEAT_DIR, exist_ok=True)
    todo = [p for p in tiles if not os.path.exists(
        os.path.join(FEAT_DIR, cache_key(p) + ".npy"))]
    print(f"[feats] {len(todo)}/{len(tiles)} tiles to extract (layers={LAYERS})")
    if not todo:
        return
    net = FrozenDinoV3Features("web", layers=LAYERS, device=device)
    for i, p in enumerate(todo):
        x, _ = load_tile(p)
        feat = net.extract(x)[0].to(torch.float16).cpu().numpy()
        np.save(os.path.join(FEAT_DIR, cache_key(p) + ".npy"), feat)
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f"  {i+1}/{len(todo)}")
    json.dump({"layers": list(LAYERS), "out_dim": net.out_dim},
              open(os.path.join(FEAT_DIR, "_meta.json"), "w"), indent=2)


def pairwise_maps(feat_hw: torch.Tensor) -> torch.Tensor:
    """feat_hw:(C,32,32) fp32 -> (8,MASK_RES,MASK_RES) neighbour cosine sims.

    sim[k, y, x] = cos(f[y, x], f[y+dy, x+dx]); out-of-range -> 0.
    """
    up = F.interpolate(feat_hw[None], size=(MASK_RES, MASK_RES), mode="bilinear",
                       align_corners=False)[0]
    up = F.normalize(up, dim=0)
    sims = torch.zeros(len(OFFSETS), MASK_RES, MASK_RES, device=up.device)
    for k, (dy, dx) in enumerate(OFFSETS):
        shifted = torch.roll(up, shifts=(-dy, -dx), dims=(1, 2))
        s = (up * shifted).sum(0)
        # zero out wrapped rows/cols
        if dy > 0:
            s[MASK_RES - dy:, :] = 0
        if dx > 0:
            s[:, MASK_RES - dx:] = 0
        elif dx < 0:
            s[:, :(-dx)] = 0
        sims[k] = s
    return sims


def cache_pairwise(tiles, device=None):
    os.makedirs(PAIR_DIR, exist_ok=True)
    dev = pick_device(device)
    all_samples = []
    todo = [p for p in tiles if not os.path.exists(
        os.path.join(PAIR_DIR, cache_key(p) + ".npy"))]
    print(f"[pairwise] {len(todo)}/{len(tiles)} tiles to compute (device={dev})")
    for i, p in enumerate(todo):
        f = np.load(os.path.join(FEAT_DIR, cache_key(p) + ".npy"))
        feat = torch.from_numpy(f.astype(np.float32)).to(dev)
        sims = pairwise_maps(feat).cpu()
        np.save(os.path.join(PAIR_DIR, cache_key(p) + ".npy"),
                sims.numpy().astype(np.float16))
        all_samples.append(sims[:, ::8, ::8].flatten())
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f"  {i+1}/{len(todo)}")
    if all_samples:
        s = torch.cat(all_samples).numpy()
        s = s[s != 0]
        q = np.percentile(s, [5, 25, 50, 75, 90, 95])
        print(f"[pairwise] sim percentiles p5={q[0]:.3f} p25={q[1]:.3f} "
              f"p50={q[2]:.3f} p75={q[3]:.3f} p90={q[4]:.3f} p95={q[5]:.3f}")
        json.dump({"offsets": OFFSETS, "mask_res": MASK_RES,
                   "sim_percentiles": {k: float(v) for k, v in
                                       zip(["p5", "p25", "p50", "p75", "p90", "p95"], q)}},
                  open(os.path.join(PAIR_DIR, "_meta.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dapt/data/split.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    split = json.load(open(os.path.join(REPO, args.split)))
    tiles = list(split["tiles"])
    cache_features(tiles, args.device)
    cache_pairwise(tiles, args.device)


if __name__ == "__main__":
    main()
