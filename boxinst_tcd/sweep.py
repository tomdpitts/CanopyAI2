"""Ceiling sweep for box-supervised instance seg on frozen DINOv3 / TCD.

Diagnosis that motivated this: the baseline head hits train boxAP50=1.00 but
val/test 0.12/0.07 — pure overfitting (100 crops, deep-only features, a 3-conv
head). This sweep varies the anti-overfitting levers, box-only throughout:
  * feature layers   deep (21-24) vs mid (3,6,9,12, the dapt choice) vs wide
  * detection head   det_tower depth (2 = baseline, 0 = lean per-patch probe)
  * train size       100 vs the full available pool

Everything else (losses, thresholds, frozen backbone, eval) is unchanged. Reports
box mAP50, mask mAP50 (mask-IoU), and GT-box-prompted mask mIoU per arm.

Usage:
    .venv/bin/python -m boxinst_tcd.sweep --cache 3,6,9,12 6,12,18,24   # cache layer sets
    .venv/bin/python -m boxinst_tcd.sweep --run                        # train+eval arms
"""
import argparse
import json
import os

import numpy as np
import torch

from dapt.backbone import FrozenDinoV3Features, load_tile, pick_device
from boxinst.cache_feats import pairwise_maps
from boxinst_tcd.cache import COMM, cell_labels, key
from boxinst_tcd.prepare import OUT

CACHE = os.path.join(OUT, "cache")


def layers_tag(layers):
    return "L" + "-".join(str(x) for x in layers)


def cache_layers(layers, device=None):
    """Cache features + pairwise for a layer set into cache/feat_<tag>, pair_<tag>."""
    tag = layers_tag(layers)
    fdir = os.path.join(CACHE, "feat_" + tag)
    pdir = os.path.join(CACHE, "pair_" + tag)
    os.makedirs(fdir, exist_ok=True); os.makedirs(pdir, exist_ok=True)
    split = json.load(open(os.path.join(OUT, "split.json")))
    tiles = list(split["tiles"])
    todo = [p for p in tiles if not os.path.exists(os.path.join(fdir, key(p) + ".npy"))]
    print(f"[{tag}] feat {len(todo)}/{len(tiles)}")
    if todo:
        net = FrozenDinoV3Features("web", layers=layers, device=device)
        for i, p in enumerate(todo):
            x, _ = load_tile(p)
            np.save(os.path.join(fdir, key(p) + ".npy"),
                    net.extract(x)[0].to(torch.float16).cpu().numpy())
            if (i + 1) % 40 == 0 or i + 1 == len(todo):
                print(f"  feat {i+1}/{len(todo)}")
    dev = pick_device(device)
    todo = [p for p in tiles if not os.path.exists(os.path.join(pdir, key(p) + ".npy"))]
    print(f"[{tag}] pair {len(todo)}/{len(tiles)}")
    for i, p in enumerate(todo):
        f = torch.from_numpy(np.load(os.path.join(fdir, key(p) + ".npy")
                                     ).astype(np.float32)).to(dev)
        np.save(os.path.join(pdir, key(p) + ".npy"),
                pairwise_maps(f).cpu().numpy().astype(np.float16))
    # commonality channel for this layer set (train-fit LDA)
    fit_commonality(split, fdir, tag)
    print(f"[{tag}] done -> feat_{tag}, pair_{tag}, comm_{tag}")


def fit_commonality(split, fdir, tag):
    cdir = os.path.join(CACHE, "comm_" + tag)
    os.makedirs(cdir, exist_ok=True)
    boxes = {p: np.array(v, np.float32).reshape(-1, 4) for p, v in
             json.load(open(os.path.join(OUT, "boxes.json"))).items()}
    train = [p for p, t in split["tiles"].items() if t["partition"] == "train"]
    feats, labs = [], []
    for p in train:
        f = np.load(os.path.join(fdir, key(p) + ".npy")).astype(np.float32)
        feats.append(f.reshape(f.shape[0], -1).T)
        labs.append(cell_labels(boxes[p]).ravel())
    X = np.concatenate(feats); y = np.concatenate(labs)
    fg, bg = X[y == 1], X[y == 0]; mu = X[y >= 0].mean(0)
    w_md = fg.mean(0) - bg.mean(0)
    Xc = np.concatenate([fg - fg.mean(0), bg - bg.mean(0)])
    d = X.shape[1]; cov = (Xc.T @ Xc) / len(Xc)
    cov = 0.9 * cov + 0.1 * (np.trace(cov) / d) * np.eye(d, dtype=np.float32)
    w_lda = np.linalg.solve(cov, w_md)
    U = np.linalg.svd((X[y >= 0][::3] - mu)[::2], full_matrices=False)[2][:64].T
    for p in split["tiles"]:
        f = np.load(os.path.join(fdir, key(p) + ".npy")).astype(np.float32)
        fl = f.reshape(f.shape[0], -1).T
        lda = ((fl - mu) @ w_lda).reshape(1, 32, 32)
        z = (fl - mu) @ U
        z = (z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)).T.reshape(64, 32, 32)
        np.savez(os.path.join(cdir, key(p) + ".npz"),
                 lda=lda.astype(np.float16), z=z.astype(np.float16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", nargs="*", default=[],
                    help="layer sets to cache, e.g. 3,6,9,12 6,12,18,24")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    for spec in args.cache:
        cache_layers(tuple(int(x) for x in spec.split(",")), args.device)


if __name__ == "__main__":
    main()
