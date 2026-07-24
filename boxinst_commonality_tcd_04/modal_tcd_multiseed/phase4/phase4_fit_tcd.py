"""Refit the box->mask EM masker ON the 4-phase L24 cells ("self-mask").

The vaulted masker was fit on native 16px cells (local features); applied to the
4-phase real-8px boxes it under-converts (box->mask gap doubled to 0.10 in the seed-0
run), capping mask mAP50. Here we refit the SAME EM recipe on the 4-phase L24 features
the detector actually uses (1024-dim, 256-grid, stride 8) so the masker is fully
in-distribution to the 4-phase boxes.

Reuses em.fit VERBATIM (identical PCA-whiten -> spherical_kmeans bg/fg -> contrastive
EM) via monkeypatch: only `load_train` (tile-based, reads feat_4p_train +
train_tiles_gt.json) and MODEL_PATH/ART are swapped. The geometry label helpers are
copied here (verbatim from em.py, with a grid-generalised canopy_cell_mask) so the fit
needs no boxinst_tcd.cache — nothing shared is imported for its side effects.

Memory: the EM assumes each tile's features are the FULL grid in raster order (it
locates in-box cells by grid geometry), so cells can't be subsampled within a tile.
At 256-grid that's 65536 cells x 1024-dim = 268 MB/tile, so we fit on a TILE SUBSET
(default 120 tiles ~= 3400 ITC boxes, already more than the original 720-crop fit) to
bound RAM (~40 GB peak). Detector/eval are unchanged; only the masker differs.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None


def _canopy_cell_mask(canopy_px, g):
    """(g,g) bool: cell is canopy if >50% of its block is canopy. Grid-generalised
    (block = canopy_px.shape[0]//g) so it works at any tile resolution."""
    if canopy_px is None or not np.any(canopy_px):
        return np.zeros((g, g), bool)
    a = np.asarray(canopy_px, bool)
    blk = a.shape[0] // g
    a = a[:g * blk, :g * blk].reshape(g, blk, g, blk)
    return a.mean((1, 3)) > 0.5


def _raster_canopy(polys, res):
    if not polys:
        return np.zeros((res, res), bool)
    m = Image.new("L", (res, res), 0)
    d = ImageDraw.Draw(m)
    for poly in polys:
        if poly and len(poly) >= 6:
            d.polygon([tuple(v) for v in np.asarray(poly).reshape(-1, 2)], fill=1)
    return np.asarray(m, bool)


def _cell_labels(boxes, canopy_px, g, s):          # == em.cell_labels_canopy
    cy, cx = np.mgrid[0:g, 0:g] * s + s / 2.0
    inbox = np.zeros((g, g), bool); near = np.zeros((g, g), bool)
    for x0, y0, x1, y1 in boxes:
        inbox |= (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
        near |= ((cx >= x0 - s) & (cx < x1 + s) & (cy >= y0 - s) & (cy < y1 + s))
    canopy = _canopy_cell_mask(canopy_px, g)
    lab = -np.ones((g, g), np.int8); lab[inbox] = 1
    lab[(~near) & (~canopy)] = 0
    return lab.ravel()


def _ring_cells(boxes, canopy_px, g, s):           # == em.ring_cells_canopy
    cy, cx = np.mgrid[0:g, 0:g] * s + s / 2.0
    inbox = np.zeros((g, g), bool); near = np.zeros((g, g), bool)
    for x0, y0, x1, y1 in boxes:
        inbox |= (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
        near |= ((cx >= x0 - s) & (cx < x1 + s) & (cy >= y0 - s) & (cy < y1 + s))
    canopy = _canopy_cell_mask(canopy_px, g)
    return (near & ~inbox & ~canopy).ravel()


def make_load_train_4p(feat_dir, gt_path, n_tiles):
    gt = json.load(open(gt_path))
    tids = sorted(t for t, v in gt.items() if v["partition"] == "train"
                  and os.path.exists(os.path.join(feat_dir, t + ".npy")))
    if n_tiles:
        tids = tids[:n_tiles]

    def load_train(_ignored):
        feats, labs, rings, bxs = [], [], [], []
        g = s = None
        for i, t in enumerate(tids):
            f = np.load(os.path.join(feat_dir, t + ".npy"))     # (1024,256,256)
            g = f.shape[-1]; s = 2048 // g                      # 256 -> 8px
            bx = np.array(gt[t]["boxes"], np.float32).reshape(-1, 4)
            can = _raster_canopy(gt[t].get("canopy", []), res=g * s)
            feats.append(f.reshape(f.shape[0], -1).T.astype(np.float32))
            labs.append(_cell_labels(bx, can, g, s))
            rings.append(_ring_cells(bx, can, g, s))
            bxs.append(bx)
            if (i + 1) % 20 == 0 or i + 1 == len(tids):
                print(f"  loaded {i+1}/{len(tids)}", flush=True)
        return tids, feats, labs, rings, bxs, g, s

    return load_train, len(tids)


def fit_masker_4p(feat_dir, gt_path, out_dir, out_npz, n_tiles=120, seed=0, pca=128,
                  k=16, k_bg=12, bins=8, kappa=10.0, iters=30,
                  contrastive_beta=0.5, no_contrast=None):
    """Fit the masker on 4-phase L24 cells -> out_npz (1024-dim prototypes, s_px=8).

    contrastive_beta / no_contrast control the contrastive term exactly as em.fit +
    masker_lab/sweep.py do: beta=0 (with no_contrast=True) is the ROBUST masker (fill +
    light carve, no prototype collapse); beta=0.5 is the collapsing default that the
    forensics showed over-carves TCD's small crowns. If no_contrast is None it defaults
    to (contrastive_beta == 0) — the sweep's convention."""
    import boxinst_commonality_tcd_04.em as EM
    if no_contrast is None:
        no_contrast = (contrastive_beta == 0)
    load_train, n = make_load_train_4p(feat_dir, gt_path, n_tiles)
    EM.load_train = load_train              # tile-based loader (feat_4p + tile GT)
    EM.MODEL_PATH = out_npz
    EM.ART = out_dir
    os.makedirs(out_dir, exist_ok=True)
    args = argparse.Namespace(seed=seed, feat_dir=feat_dir, pca=pca, k=k, k_bg=k_bg,
                              bins=bins, kappa=kappa, iters=iters, prune_after=5,
                              prune_frac=0.25, contrastive_beta=contrastive_beta,
                              no_contrast=no_contrast)
    print(f"[fit_masker_4p] {n} 4-phase L24 tiles (g=256,s=8) pca={pca} k={k} "
          f"k_bg={k_bg} beta={contrastive_beta} no_contrast={no_contrast} -> {out_npz}",
          flush=True)
    EM.fit(args)                            # identical EM recipe, byte-for-byte
    return out_npz
