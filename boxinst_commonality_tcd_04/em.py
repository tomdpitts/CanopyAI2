"""Training-free commonality EM for TCD individual-tree crowns (boxes only).

Ports the boxinst_commonality method (whiten + within-box contrast + contrastive
prototype repel + size-conditioned spatial priors + auto-K) to TCD, reusing the
pure-math core from boxinst_commonality.em. Differences from the dryland fit:

  - fits on the boxinst_tcd 720 train crops (ITC boxes = category 2), boxes only;
  - CANOPY (category 1) is an IGNORE label — never a background negative and
    never a foreground positive (canopy is tree-like; counting it as background
    would poison the tree-vs-ground commonality, exactly as in boxinst_tcd.cache);
  - the fitted model is grid-agnostic: it stores the stride (px/cell), not a
    tile size, so the same prototypes apply to 512 train crops (32x32) and to
    2048 test tiles (128x128) — see TCDMasker.

No polygon is ever read here. Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.em --seed 0
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

from boxinst_commonality.em import (BIN_CAP, FG_PRIOR_MASS, contrastive_update,
                                    estep, logsumexp, spherical_kmeans)
from boxinst_tcd.build_canopy import load_canopy_mask
from boxinst_tcd.cache import FEAT, canopy_cell_mask, key
from boxinst_tcd.prepare import OUT as TCD_OUT

OUT = os.path.abspath(os.path.dirname(__file__))
ART = os.path.join(OUT, "artifacts")
MODEL_PATH = os.path.join(ART, "em_model.npz")


def cell_labels_canopy(boxes, canopy_px, g, s):
    """(g*g,) int8: 1 in a tree box, 0 clear background (>=1 cell from any box AND
    not canopy), -1 near-box / canopy / pad. Canopy is IGNORE, never background."""
    cy, cx = np.mgrid[0:g, 0:g] * s + s / 2.0
    inbox = np.zeros((g, g), bool)
    near = np.zeros((g, g), bool)
    for x0, y0, x1, y1 in boxes:
        inbox |= (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
        near |= ((cx >= x0 - s) & (cx < x1 + s) & (cy >= y0 - s) & (cy < y1 + s))
    canopy = canopy_cell_mask(canopy_px) if canopy_px is not None \
        else np.zeros((g, g), bool)
    lab = -np.ones((g, g), np.int8)
    lab[inbox] = 1
    lab[(~near) & (~canopy)] = 0
    return lab.ravel()


def ring_cells_canopy(boxes, canopy_px, g, s):
    """Near-box cells that are OUTSIDE every box and NOT canopy (context-matched
    non-tree background for the bg mixture / contrastive negatives)."""
    cy, cx = np.mgrid[0:g, 0:g] * s + s / 2.0
    inbox = np.zeros((g, g), bool)
    near = np.zeros((g, g), bool)
    for x0, y0, x1, y1 in boxes:
        inbox |= (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
        near |= ((cx >= x0 - s) & (cx < x1 + s) & (cy >= y0 - s) & (cy < y1 + s))
    canopy = canopy_cell_mask(canopy_px) if canopy_px is not None \
        else np.zeros((g, g), bool)
    return (near & ~inbox & ~canopy).ravel()


class TCDMasker:
    """Grid-agnostic posterior applier: prototypes fit at stride s apply to any
    tile. box_mask(zn, g, box) returns (cell_idx, P(fg)) over the tile's g*g grid.
    """

    def __init__(self, path=MODEL_PATH):
        d = np.load(path, allow_pickle=False)
        self.mu, self.U, self.scale = d["mu"], d["U"], d["scale"]
        self.C, self.Gbg, self.wbg = d["C"], d["Gbg"], d["wbg"]
        self.pi, self.kappa = d["pi"], float(d["kappa"])
        self.size_edges = d["size_edges"]
        self.s = int(d["s_px"])
        self.NB = self.pi.shape[2]

    def project(self, feat):
        z = (feat.reshape(feat.shape[0], -1).T.astype(np.float32) - self.mu) \
            @ self.U / self.scale
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)

    def bg_ll(self, zn):
        return logsumexp(np.log(self.wbg)[None] + self.kappa * (zn @ self.Gbg.T), 1)

    def size_bin(self, box):
        return int(np.searchsorted(self.size_edges,
                                   np.sqrt(max(box[2] - box[0], 1) *
                                           max(box[3] - box[1], 1))))

    def box_mask(self, zn, g, box, contrast=True):
        """zn:(g*g,D) whitened cells; box xyxy in tile px. -> (idx, P(fg))."""
        s = self.s
        cy, cx = np.mgrid[0:g, 0:g] * s + s / 2.0
        x0, y0, x1, y1 = box
        pad = s / 2.0
        m = ((cx >= x0 - pad) & (cx < x1 + pad) &
             (cy >= y0 - pad) & (cy < y1 + pad))
        idx = np.flatnonzero(m.ravel())
        if len(idx) == 0:
            d2 = (cx - (x0 + x1) / 2) ** 2 + (cy - (y0 + y1) / 2) ** 2
            idx = np.array([int(d2.ravel().argmin())])
        u = np.clip((cx.ravel()[idx] - x0) / max(x1 - x0, 1), 0, 1)
        v = np.clip((cy.ravel()[idx] - y0) / max(y1 - y0, 1), 0, 1)
        bu = np.minimum((u * self.NB).astype(int), self.NB - 1)
        bv = np.minimum((v * self.NB).astype(int), self.NB - 1)
        sb = self.size_bin(box)
        pfg, _ = estep(zn[idx], self.pi[sb][:, bv, bu], self.kappa, self.C,
                       self.bg_ll(zn[idx]), contrast)
        return idx, pfg


def load_train(feat_dir):
    """boxinst_tcd train crops -> (tiles, feats(list C,32,32-flat), labs, rings,
    boxes, g, s). Reuses the existing 720-crop ITC split + web features."""
    split = json.load(open(os.path.join(TCD_OUT, "split.json")))
    boxes = json.load(open(os.path.join(TCD_OUT, "boxes.json")))
    tiles = sorted(p for p, t in split["tiles"].items()
                   if t["partition"] == "train")
    feats, labs, rings, bxs = [], [], [], []
    g = s = None
    for p in tiles:
        f = np.load(os.path.join(feat_dir, key(p) + ".npy"))
        g = f.shape[-1]; s = 512 // g
        canopy = load_canopy_mask(p)
        bx = np.array(boxes[p], np.float32).reshape(-1, 4)
        feats.append(f.reshape(f.shape[0], -1).T)
        labs.append(cell_labels_canopy(bx, canopy, g, s))
        rings.append(ring_cells_canopy(bx, canopy, g, s))
        bxs.append(bx)
    return tiles, feats, labs, rings, bxs, g, s


def fit(args):
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    tiles, feats, labs, rings, boxes, g, s = load_train(args.feat_dir)
    print(f"TCD train crops={len(tiles)} grid={g} stride={s}px", flush=True)

    # centred, whitened PCA on tree+clear-bg cells (canopy excluded)
    mu_acc, nv = 0.0, 0
    for f, l in zip(feats, labs):
        v = l >= 0
        mu_acc = mu_acc + f[v].astype(np.float64).sum(0); nv += int(v.sum())
    mu = (mu_acc / nv).astype(np.float32)
    sub = np.concatenate([f[l >= 0][::5].astype(np.float32) - mu
                          for f, l in zip(feats, labs)])
    _, sv, Vt = np.linalg.svd(sub, full_matrices=False)
    U = Vt[:args.pca].T
    scale = sv[:args.pca] / np.sqrt(len(sub)) + 1e-6
    del sub
    Zn = [((f.astype(np.float32) - mu) @ U / scale) for f in feats]
    Zn = [z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8) for z in Zn]
    del feats
    print(f"PCA {args.pca}, whitened", flush=True)

    clearZ = np.concatenate([z[l == 0] for z, l in zip(Zn, labs)])
    ringZ = np.concatenate([z[r] for z, r in zip(Zn, rings)])
    bgZ = np.concatenate([clearZ, ringZ])
    if len(bgZ) > 80000:
        bgZ = bgZ[rng.choice(len(bgZ), 80000, replace=False)]
    negZ = np.concatenate([ringZ, ringZ, clearZ])
    if len(negZ) > 60000:
        negZ = negZ[rng.choice(len(negZ), 60000, replace=False)]
    Gbg, a = spherical_kmeans(bgZ, args.k_bg, rng)
    wbg = np.bincount(a, minlength=args.k_bg) / len(a)
    wbg = np.clip(wbg, 1e-3, None); wbg /= wbg.sum()
    print(f"bg mixture K_bg={args.k_bg} on {len(clearZ)} clear + {len(ringZ)} ring",
          flush=True)

    sizes = np.array([np.sqrt((b[2] - b[0]) * (b[3] - b[1]))
                      for bx in boxes for b in bx])
    size_edges = np.percentile(sizes, [100 / 3, 200 / 3])
    # in-box instance cells (strict centre-in-box)
    inst = []
    for ti, bx in enumerate(boxes):
        cy, cx = np.mgrid[0:g, 0:g] * s + s / 2.0
        for b in bx:
            x0, y0, x1, y1 = b
            m = (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
            idx = np.flatnonzero(m.ravel())
            if not len(idx):
                continue
            u = (cx.ravel()[idx] - x0) / max(x1 - x0, 1)
            v = (cy.ravel()[idx] - y0) / max(y1 - y0, 1)
            sb = int(np.searchsorted(size_edges, np.sqrt((x1 - x0) * (y1 - y0))))
            inst.append((ti, idx, u, v, sb))
    print(f"EM over {len(inst)} boxes / {sum(len(i[1]) for i in inst)} cells; "
          f"size terciles {np.round(size_edges, 1)}px", flush=True)

    kappa = args.kappa
    bg_all = [logsumexp(np.log(wbg)[None] + kappa * (z @ Gbg.T), 1) for z in Zn]

    def bg_ll(ti, idx):
        return bg_all[ti][idx]

    fgZ = np.concatenate([Zn[ti][idx] for ti, idx, *_ in inst])
    fg_bgll = np.concatenate([bg_ll(ti, idx) for ti, idx, *_ in inst])
    fgZ = fgZ[fg_bgll < np.median(fg_bgll)]
    if len(fgZ) > 60000:
        fgZ = fgZ[rng.choice(len(fgZ), 60000, replace=False)]
    C, _ = spherical_kmeans(fgZ, args.k, rng)

    NB, SB = args.bins, 3
    pi = np.full((SB, C.shape[0], NB, NB), FG_PRIOR_MASS / C.shape[0])
    trace = []
    for it in range(args.iters):
        K = C.shape[0]
        acc = np.zeros((SB, K, NB, NB)); cnt = np.zeros((SB, NB, NB))
        newC = np.zeros_like(C); fg_mass, n_tot = 0.0, 0
        for ti, idx, u, v, sb in inst:
            bu = np.minimum((u * NB).astype(int), NB - 1)
            bv = np.minimum((v * NB).astype(int), NB - 1)
            pfg, r = estep(Zn[ti][idx], pi[sb][:, bv, bu], kappa, C, bg_ll(ti, idx),
                           contrast=not args.no_contrast)
            np.add.at(acc[sb], (slice(None), bv, bu), r.T)
            np.add.at(cnt[sb], (bv, bu), 1)
            newC += r.T @ Zn[ti][idx]; fg_mass += pfg.sum(); n_tot += len(idx)
        C = contrastive_update(newC, negZ, kappa, args.contrastive_beta)
        pi = np.clip(acc / np.maximum(cnt, 1)[:, None], 1e-4, None)
        for sb in range(SB):
            pi[sb] *= FG_PRIOR_MASS / max(pi[sb].sum(0).mean(), 1e-6)
            bs = pi[sb].sum(0); over = bs > BIN_CAP
            if over.any():
                pi[sb][:, over] *= BIN_CAP / bs[over]
        if it >= args.prune_after and C.shape[0] > 2:
            share = acc.sum((0, 2, 3)); share = share / max(share.sum(), 1e-9)
            keep = share >= args.prune_frac / C.shape[0]
            if (~keep).any():
                C, pi = C[keep], pi[:, keep]
                print(f"  it{it+1:2d}: prune -> K={C.shape[0]}", flush=True)
        trace.append(round(fg_mass / n_tot, 4))
        if (it + 1) % 5 == 0:
            print(f"  it{it+1:2d}: mean P(fg)={trace[-1]:.3f}", flush=True)

    os.makedirs(ART, exist_ok=True)
    np.savez(MODEL_PATH, mu=mu, U=U, scale=scale, C=C, Gbg=Gbg, wbg=wbg, pi=pi,
             kappa=kappa, size_edges=size_edges, s_px=s)
    json.dump({"seed": args.seed, "feat_dir": args.feat_dir, "stride_px": int(s),
               "pca": args.pca, "k_init": args.k, "k_effective": int(C.shape[0]),
               "k_bg": args.k_bg, "bins": args.bins, "kappa": args.kappa,
               "contrastive_beta": args.contrastive_beta,
               "size_edges_px": [round(float(e), 1) for e in size_edges],
               "n_boxes": len(inst), "fg_mass_trace": trace,
               "note": "boxes-only, canopy-ignore, no polygon ever read"},
              open(os.path.join(ART, "em_fit_report.json"), "w"), indent=2)
    print(f"saved {MODEL_PATH}  (K={C.shape[0]})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feat_dir", default=FEAT)
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--k_bg", type=int, default=12)
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--prune_after", type=int, default=5)
    ap.add_argument("--prune_frac", type=float, default=0.25)
    ap.add_argument("--contrastive_beta", type=float, default=0.5)
    ap.add_argument("--no_contrast", action="store_true")
    fit(ap.parse_args())


if __name__ == "__main__":
    main()
