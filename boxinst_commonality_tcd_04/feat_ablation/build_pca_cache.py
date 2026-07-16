"""Fit a PCA-256 basis on cached 4096-dim train-tile features and project both
the train and test caches into feat_ablation/cache/pca256/. No DINO forwards —
pure linear projection of the parent fp16 caches.

Stage FIT : sample ~200k patch vectors (seeded) across all cached train tiles,
            streaming f64 covariance -> eigh -> top-256 basis -> pca256.npz.
Stage PROJ: per tile Y = W @ X - (W @ mean) -> (256,128,128) fp16. Resumable
            (skips existing outputs).

The block-1024 variant needs NO cache: the last 1024 channels (block 24) are a
contiguous slice of the parent .npy, mmap-sliced at load time by the wrappers.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.feat_ablation.build_pca_cache
"""
import argparse
import json
import os
import time

import numpy as np

from boxinst_commonality_tcd_04.cache_train_tiles import cache_dir as train_cache
from boxinst_commonality_tcd_04.cache_test import cache_dir as test_cache

HERE = os.path.abspath(os.path.dirname(__file__))
PCA_NPZ = os.path.join(HERE, "pca256.npz")
K = 256


def out_dir(split):                      # split: feat_traintile | feat_test
    return os.path.join(HERE, "cache", "pca256", split)


def fit(n_samples, seed):
    cdir = train_cache("web")
    tiles = sorted(f for f in os.listdir(cdir) if f.endswith(".npy"))
    per = int(np.ceil(n_samples / len(tiles)))
    rng = np.random.default_rng(seed)
    print(f"[fit] {len(tiles)} train tiles, {per} vectors/tile "
          f"-> ~{per * len(tiles)} samples", flush=True)
    s1 = np.zeros(4096, np.float64)                  # streaming sum
    s2 = np.zeros((4096, 4096), np.float64)          # streaming sum of outer
    n = 0
    t0 = time.time()
    for i, f in enumerate(tiles):
        x = np.load(os.path.join(cdir, f)).reshape(4096, -1)
        cols = rng.choice(x.shape[1], size=per, replace=False)
        v = x[:, cols].astype(np.float64)            # (4096, per)
        s1 += v.sum(1)
        s2 += v @ v.T
        n += per
        if (i + 1) % 100 == 0 or i + 1 == len(tiles):
            print(f"[fit] {i+1}/{len(tiles)} tiles  n={n}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    mean = s1 / n
    cov = s2 / n - np.outer(mean, mean)
    print("[fit] eigh(4096x4096)...", flush=True)
    w, v = np.linalg.eigh(cov)                       # ascending
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    evr = float(w[:K].sum() / w.sum())
    W = v[:, :K].T.astype(np.float32)                # (256, 4096)
    np.savez(PCA_NPZ, W=W, mean=mean.astype(np.float32),
             eigenvalues=w.astype(np.float32),
             n_samples=n, seed=seed, explained_var_256=evr)
    print(f"[fit] top-{K} explained variance = {evr:.4f} -> {PCA_NPZ}", flush=True)


def project():
    z = np.load(PCA_NPZ)
    W, mean = z["W"], z["mean"].astype(np.float32)
    off = (W @ mean)[:, None]                        # (256,1)
    for split, cdir in (("feat_traintile", train_cache("web")),
                        ("feat_test", test_cache("web"))):
        odir = out_dir(split)
        os.makedirs(odir, exist_ok=True)
        tiles = sorted(f for f in os.listdir(cdir) if f.endswith(".npy"))
        todo = [f for f in tiles if not os.path.exists(os.path.join(odir, f))]
        print(f"[proj] {split}: {len(todo)}/{len(tiles)} to project", flush=True)
        t0 = time.time()
        for i, f in enumerate(todo):
            x = np.load(os.path.join(cdir, f)).reshape(4096, -1).astype(np.float32)
            y = (W @ x - off).reshape(K, 128, 128).astype(np.float16)
            tmp = os.path.join(odir, f + ".tmp.npy")
            np.save(tmp, y)
            os.replace(tmp, os.path.join(odir, f))
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                dt = time.time() - t0
                print(f"[proj] {split} {i+1}/{len(todo)}  "
                      f"{dt/(i+1):.2f}s/tile  eta {(len(todo)-i-1)*dt/(i+1):.0f}s",
                      flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_fit", action="store_true")
    args = ap.parse_args()
    if not args.skip_fit and not os.path.exists(PCA_NPZ):
        fit(args.n_samples, args.seed)
    elif os.path.exists(PCA_NPZ):
        print(f"[fit] {PCA_NPZ} exists, skipping fit", flush=True)
    project()


if __name__ == "__main__":
    main()
