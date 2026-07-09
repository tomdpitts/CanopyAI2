"""Cache frozen DINOv3 features (last-4 blocks), pairwise DINO-affinity maps, and
the commonality channel for the TCD tiles. Mirrors boxinst.cache_feats /
boxinst.commonality but self-contained under boxinst_tcd/cache/.

The commonality LDA is fit on TRAIN boxes only (in-box vs clear-background patches),
using boxes.json — never a polygon. Stages:
  feat : (4096,32,32) fp16   per tile
  pair : (8,128,128)  fp16   neighbour cosine sims (for the BoxInst pairwise loss)
  comm : lda (1,32,32) + z (64,32,32) fp16   (commonality channel + PCA embed)

Usage:
    .venv/bin/python -m boxinst_tcd.cache
"""
import json
import os

import numpy as np
import torch

from dapt.backbone import FrozenDinoV3Features, load_tile, pick_device
from boxinst.cache_feats import LAYERS, OFFSETS, pairwise_maps
from boxinst_tcd.prepare import OUT

CACHE = os.path.join(OUT, "cache")
FEAT = os.path.join(CACHE, "feat")
PAIR = os.path.join(CACHE, "pair")
COMM = os.path.join(CACHE, "comm")
GRID, STRIDE = 32, 16


def key(p):
    return os.path.splitext(os.path.basename(p))[0]


def stage_feat(tiles, device):
    os.makedirs(FEAT, exist_ok=True)
    todo = [p for p in tiles if not os.path.exists(os.path.join(FEAT, key(p) + ".npy"))]
    print(f"[feat] {len(todo)}/{len(tiles)}")
    if not todo:
        return
    net = FrozenDinoV3Features("web", layers=LAYERS, device=device)
    for i, p in enumerate(todo):
        x, _ = load_tile(p)
        np.save(os.path.join(FEAT, key(p) + ".npy"),
                net.extract(x)[0].to(torch.float16).cpu().numpy())
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f"  {i+1}/{len(todo)}")


def stage_pair(tiles, device):
    os.makedirs(PAIR, exist_ok=True)
    dev = pick_device(device)
    todo = [p for p in tiles if not os.path.exists(os.path.join(PAIR, key(p) + ".npy"))]
    print(f"[pair] {len(todo)}/{len(tiles)}")
    for i, p in enumerate(todo):
        f = torch.from_numpy(np.load(os.path.join(FEAT, key(p) + ".npy")
                                     ).astype(np.float32)).to(dev)
        np.save(os.path.join(PAIR, key(p) + ".npy"),
                pairwise_maps(f).cpu().numpy().astype(np.float16))
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f"  {i+1}/{len(todo)}")


def canopy_cell_mask(canopy_pixels):
    """(G,G) bool: cell is canopy if >50% of its 16px block is canopy.
    canopy_pixels: (512,512) bool mask (from build_canopy.load_canopy_mask) or None."""
    if canopy_pixels is None or not np.any(canopy_pixels):
        return np.zeros((GRID, GRID), bool)
    a = np.asarray(canopy_pixels, bool).reshape(GRID, STRIDE, GRID, STRIDE)
    return a.mean((1, 3)) > 0.5


def cell_labels(boxes, canopy_pixels=None, res=512):
    """(G,G) int8: 1 in a tree box, 0 clear-bg, -1 near-box/canopy/pad.

    Canopy cells are IGNORE (-1), never background: canopy is tree-like, so counting
    it as negative would poison the tree-vs-background commonality direction.
    """
    cy, cx = np.mgrid[0:GRID, 0:GRID] * STRIDE + STRIDE / 2.0
    inbox = np.zeros((GRID, GRID), bool)
    near = np.zeros((GRID, GRID), bool)
    for x0, y0, x1, y1 in boxes:
        inbox |= (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
        near |= (cx >= x0 - STRIDE) & (cx < x1 + STRIDE) & \
                (cy >= y0 - STRIDE) & (cy < y1 + STRIDE)
    lab = -np.ones((GRID, GRID), np.int8)
    lab[inbox] = 1
    canopy = canopy_cell_mask(canopy_pixels)
    lab[(~near) & (~canopy)] = 0
    return lab


def stage_comm(split, boxes, device):
    """Fit LDA + PCA on TRAIN boxes, cache commonality channel for all tiles."""
    from boxinst_tcd.build_canopy import load_canopy_mask
    os.makedirs(COMM, exist_ok=True)
    train = [p for p, t in split["tiles"].items() if t["partition"] == "train"]
    feats, labs = [], []
    for p in train:
        f = np.load(os.path.join(FEAT, key(p) + ".npy")).astype(np.float32)
        feats.append(f.reshape(f.shape[0], -1).T)
        labs.append(cell_labels(boxes[p], load_canopy_mask(p)).ravel())
    X = np.concatenate(feats)
    y = np.concatenate(labs)
    fg, bg = X[y == 1], X[y == 0]
    mu = X[y >= 0].mean(0)
    w_md = fg.mean(0) - bg.mean(0)
    Xc = np.concatenate([fg - fg.mean(0), bg - bg.mean(0)])
    d = X.shape[1]
    cov = (Xc.T @ Xc) / len(Xc)
    cov = 0.9 * cov + 0.1 * (np.trace(cov) / d) * np.eye(d, dtype=np.float32)
    w_lda = np.linalg.solve(cov, w_md)
    U = np.linalg.svd((X[y >= 0][::3] - mu)[::2], full_matrices=False)[2][:64].T
    # AUC on TRAIN (fit quality only; test AUC is reported by eval)
    s = (X[y >= 0] - mu) @ w_lda
    yy = y[y >= 0]
    order = np.argsort(s)
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    n1, n0 = (yy == 1).sum(), (yy == 0).sum()
    auc = (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    print(f"[comm] train fit: fg={len(fg)} bg={len(bg)} train-AUC={auc:.3f}")
    np.savez(os.path.join(COMM, "_basis.npz"), w_lda=w_lda, mu=mu, U=U.astype(np.float32))
    for p in split["tiles"]:
        f = np.load(os.path.join(FEAT, key(p) + ".npy")).astype(np.float32)
        fl = f.reshape(f.shape[0], -1).T
        lda = ((fl - mu) @ w_lda).reshape(1, GRID, GRID)
        z = (fl - mu) @ U
        z = (z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)).T.reshape(64, GRID, GRID)
        np.savez(os.path.join(COMM, key(p) + ".npz"),
                 lda=lda.astype(np.float16), z=z.astype(np.float16))


def main():
    split = json.load(open(os.path.join(OUT, "split.json")))
    boxes = {p: np.array(v, np.float32) for p, v in
             json.load(open(os.path.join(OUT, "boxes.json"))).items()}
    tiles = list(split["tiles"])
    device = pick_device(None)
    stage_feat(tiles, device)
    stage_pair(tiles, device)
    stage_comm(split, boxes, device)
    print("done")


if __name__ == "__main__":
    main()
