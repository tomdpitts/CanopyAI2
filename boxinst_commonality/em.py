"""Embedding-commonality EM: hand-computed figure/ground from boxes only.

The "duck principle" made precise: foreground = embeddings that recur across the
annotated boxes AND are rare outside boxes; background = embeddings common inside
and outside alike. The only real label anywhere — "out-of-box is background" —
anchors the decomposition.

Model (centred PCA-D, L2-normed DINO patch space; vMF likelihoods = kappa*cos):
  bg  K_bg-component mixture fit by spherical k-means on clear-background train
      cells, then FROZEN — foreground can never absorb it during EM;
  fg  K crown prototypes C_k with spatial priors pi[s,k,v,u] over box-normalized
      coords (u,v), separately per box-size tercile s, learned by EM over all
      train-box cells. P(bg|u,v,s) = 1 - sum_k pi.
Posterior P(fg | z, u, v, s) inside a (GT or predicted) box IS the mask. No mask
label and no mask-based selection anywhere; hyperparameters are fixed a priori
and recorded, never tuned on masks (same discipline as tau in boxinst).

Upgrades over boxinst/commonality.py cmd_em (which stays untouched): frozen
multi-component background (was one drifting vector), size-conditioned spatial
priors, K=8, prototype patch-montage diagnostics.

Usage:
    .venv/bin/python -m boxinst_commonality.em --seed 0
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dapt.cache_features import cache_key
from dapt.data.cohort import REPO
from boxinst.commonality import cell_labels, load_split

# default: 8px half-stride cache (boxinst_commonality.cache_s8) — measurably
# tighter masks than the plain 16px cache (dapt/cache/web_last4, blocks 21-24)
DEFAULT_FEATS = os.path.join(REPO, "dapt/cache/web_last4_s8")

G, S_PX = 32, 16                       # patch grid / stride (512-padded tiles)
# A tight bbox of a ~convex crown contains ~pi/4 crown (inscribed ellipse): the
# prior's TOTAL fg mass is pinned to this a-priori constant each M-step, so EM
# can only redistribute it spatially — without this, 8 adaptive prototypes
# out-fit the frozen bg mixture on in-box soil and the prior saturates at ~1
# (observed: mean P(fg)=0.97, box-filler masks).
FG_PRIOR_MASS = np.pi / 4
BIN_CAP = 0.95                         # no (u,v) bin may be certain-foreground
ART = os.path.join(REPO, "boxinst_commonality/artifacts")
OUT = os.path.join(REPO, "claude_outputs/boxinst_commonality")
MODEL_PATH = os.path.join(ART, "em_model.npz")


def out_dir(tag=""):
    """Numbered render subfolder per model tag: OUT/NN_<tag> (reused if it
    already exists, else next number)."""
    base = tag.lstrip("_") if tag else "primary"
    os.makedirs(OUT, exist_ok=True)
    dirs = [d for d in sorted(os.listdir(OUT))
            if os.path.isdir(os.path.join(OUT, d))]
    for d in dirs:
        if d.split("_", 1)[-1] == base:
            return os.path.join(OUT, d)
    p = os.path.join(OUT, f"{len(dirs) + 1:02d}_{base}")
    os.makedirs(p, exist_ok=True)
    return p


def draw_grid(ax, H0, W0, s=S_PX):
    """Overlay the patch-cell grid (diagnoses cell quantisation)."""
    for x in range(0, W0 + 1, s):
        ax.axvline(x, color="w", lw=0.3, alpha=0.5)
    for y in range(0, H0 + 1, s):
        ax.axhline(y, color="w", lw=0.3, alpha=0.5)


def draw_active_cells(ax, idx, r, g=G, s=S_PX, thr=0.5):
    """Gently outline (purple, no fill) the cells with posterior >= thr —
    the cells actually driving a mask, pre-upsampling."""
    import matplotlib.pyplot as plt
    for ci in idx[r >= thr]:
        cy, cx = divmod(int(ci), g)
        ax.add_patch(plt.Rectangle((cx * s, cy * s), s, s,
                                   fill=False, ec="#b45cff", lw=0.7,
                                   alpha=0.75))


def cell_labels_g(bx, H0, W0, g=G, s=S_PX):
    """(g,g) labels at stride s: 1 in some box, 0 clear background (valid,
    >=1 cell from any box), -1 boundary/pad. Grid-parameterized version of
    boxinst.commonality.cell_labels."""
    cy, cx = np.mgrid[0:g, 0:g] * s + s / 2.0
    lab = -np.ones((g, g), np.int8)
    valid = (cy < H0) & (cx < W0)
    inbox = np.zeros((g, g), bool)
    near = np.zeros((g, g), bool)
    for x0, y0, x1, y1 in bx:
        inbox |= (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
        near |= ((cx >= x0 - s) & (cx < x1 + s) &
                 (cy >= y0 - s) & (cy < y1 + s))
    lab[valid & inbox] = 1
    lab[valid & ~near] = 0
    return lab


def ring_cells(bx, H0, W0, g=G, s=S_PX):
    """Flat bool mask of cells OUTSIDE every box but within one cell of one.

    These are tree-adjacent background: ViT patch embeddings carry attention
    context, so soil-next-to-a-crown embeds differently from open-ground soil.
    A bg mixture fit on clear background only cannot explain in-box soil (it
    pushes tree-adjacent soil to fg -> box-filler masks, observed); the ring
    supplies context-matched background evidence. Boxes are tight, so the ring
    is overwhelmingly not-crown.
    """
    cy, cx = np.mgrid[0:g, 0:g] * s + s / 2.0
    valid = (cy < H0) & (cx < W0)
    inbox = np.zeros((g, g), bool)
    near = np.zeros((g, g), bool)
    for x0, y0, x1, y1 in bx:
        inbox |= (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
        near |= ((cx >= x0 - s) & (cx < x1 + s) &
                 (cy >= y0 - s) & (cy < y1 + s))
    return (valid & near & ~inbox).ravel()


def collect_feats(split, boxes, part, feat_dir):
    """Like boxinst.commonality.collect but with a selectable feature cache.

    Keeps features fp16 (callers upcast per tile) and infers the grid from the
    cache shape, so stride-16 (32x32) and stride-8 (64x64) caches both work.
    Returns (tiles, feats (n_cells,C) fp16 each, labs, g, s).
    """
    tiles = sorted(p for p, t in split["tiles"].items()
                   if t["partition"] == part)
    feats, labs, g = [], [], None
    for p in tiles:
        f = np.load(os.path.join(feat_dir, cache_key(p) + ".npy"))
        g = f.shape[-1]
        s = 512 // g
        W0, H0 = Image.open(p).size
        labs.append(cell_labels_g(boxes.get(p, np.zeros((0, 4))), H0, W0,
                                  g, s).ravel())
        feats.append(f.reshape(f.shape[0], -1).T)
    return tiles, feats, labs, g, 512 // g


def spherical_kmeans(X, k, rng, iters=25):
    """X:(N,D) L2-normed -> centroids (k,D) L2-normed, assignment (N,)."""
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        a = np.argmax(X @ C.T, 1)
        C = np.stack([X[a == j].mean(0) if (a == j).any() else C[j]
                      for j in range(k)])
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    return C, np.argmax(X @ C.T, 1)


def logsumexp(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def softmax(a, axis):
    e = np.exp(a - a.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def contrastive_update(newC, negZ, kappa, beta):
    """Repel each prototype from the outside-box signature it most resembles.

    newC:(K,D) responsibility-weighted in-box sums (the "recurs across boxes"
    pull). negZ:(M,D) outside-box cells (clear bg + near-box ring). For each
    prototype we softmax-assign the negatives to it, take that prototype's own
    negative centroid, and subtract it: C_k = normalise(pos_k - beta * neg_k).
    A soil-like prototype's nearest negatives ARE soil, so the subtraction tears
    it away from soil — commonality is defined as agreement-across-boxes-MINUS-
    outside-signature, not just "whatever fills the box". beta=0 recovers the
    generative M-step.
    """
    pos = newC / (np.linalg.norm(newC, axis=1, keepdims=True) + 1e-8)
    if beta <= 0:
        return pos
    a = softmax(kappa * (negZ @ pos.T), axis=1)             # (M,K)
    neg = (a.T @ negZ) / (a.sum(0)[:, None] + 1e-8)         # (K,D)
    neg = neg / (np.linalg.norm(neg, axis=1, keepdims=True) + 1e-8)
    C = pos - beta * neg
    return C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)


def estep(z, pis, kappa, C, bgll, contrast=True):
    """Shared E-step for the cells of ONE box.

    z (n,D) cells; pis (K,n) spatial prior at each cell's (u,v) bin; bgll (n,).
    Returns (pfg (n,) total fg posterior, r (n,K) per-prototype responsibility).

    contrast recentres the appearance log-ratio within the box: the ABSOLUTE
    fg/bg ratio is confounded by ViT context (all in-box cells score fg), while
    the within-box ordering carries the local crown-vs-soil signal (verified
    against a supervised brightness ceiling). Wide spread (crown+soil box) ->
    carving; narrow spread (dense canopy) -> box stays filled.
    """
    zc = kappa * (z @ C.T)                              # (n,K)
    psum = np.clip(pis.sum(0), 1e-4, 1 - 1e-4)
    lw = zc + np.log((pis / psum).T + 1e-9)             # component log-weights
    A = logsumexp(lw, 1) - bgll                         # appearance log-ratio
    if contrast and len(A) > 1:
        A = A - A.mean()
    pfg = 1.0 / (1.0 + np.exp(-(A + np.log(psum / (1 - psum)))))
    e = lw - lw.max(1, keepdims=True)
    wk = np.exp(e)
    wk /= wk.sum(1, keepdims=True)
    return pfg, pfg[:, None] * wk


class EMModel:
    """Loadable posterior model. All numpy, no torch."""

    def __init__(self, path=MODEL_PATH):
        d = np.load(path, allow_pickle=False)
        self.feat_dir = str(d["feat_dir"]) if "feat_dir" in d else DEFAULT_FEATS
        self.mu, self.U = d["mu"], d["U"]                    # (C,), (C,D)
        self.scale = d["scale"] if "scale" in d else np.ones(self.U.shape[1])
        self.C, self.Gbg, self.wbg = d["C"], d["Gbg"], d["wbg"]
        self.pi = d["pi"]                                    # (S,K,NB,NB)
        self.kappa = float(d["kappa"])
        self.size_edges = d["size_edges"]                    # (S-1,)
        self.NB = self.pi.shape[2]
        self.grid = int(d["grid"]) if "grid" in d else G     # cells per side
        self.s = int(d["s_px"]) if "s_px" in d else S_PX     # px per cell

    def project(self, feat):
        """feat (C,G,G) -> (G*G, D) L2-normed (whitened if fit with --whiten)."""
        z = (feat.reshape(feat.shape[0], -1).T.astype(np.float32) - self.mu) \
            @ self.U / self.scale
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)

    def project_tile(self, p):
        """Tile path -> (G*G, D), loading from this model's feature cache."""
        return self.project(np.load(
            os.path.join(self.feat_dir, cache_key(p) + ".npy")).astype(np.float32))

    def bg_ll(self, zn):
        return logsumexp(np.log(self.wbg)[None] + self.kappa * (zn @ self.Gbg.T), 1)

    def size_bin(self, box):
        s = np.sqrt(max(box[2] - box[0], 1) * max(box[3] - box[1], 1))
        return int(np.searchsorted(self.size_edges, s))

    def box_posterior(self, zn, box, H0, W0, expand=True, contrast=True):
        """Posterior P(fg) for the cells covering `box`.

        Fitting used strict centre-in-box cells; at inference we expand by half a
        cell (cells overlapping the box) with (u,v) clamped to [0,1], so pixel-
        precision box edges are not zeroed by cell quantisation.

        contrast=True recentres the appearance log-ratio per box before adding
        the spatial-prior logit. Rationale (measured): the ABSOLUTE fg/bg ratio
        is confounded by ViT context (~all in-box cells score fg; supervised
        ceiling shows the within-box ORDERING is where the local signal lives).
        Recentring uses only the ranking: heterogeneous boxes (crown+soil) have
        a wide spread -> carving; homogeneous boxes (dense canopy) a narrow
        spread -> stay filled. No new constants; boxes-only throughout.
        Returns (cell_idx (n,), r (n,)) into the flattened (grid,grid) grid.
        """
        g, s = self.grid, self.s
        cy, cx = np.mgrid[0:g, 0:g] * s + s / 2.0
        valid = (cy < H0) & (cx < W0)
        x0, y0, x1, y1 = box
        pad = s / 2.0 if expand else 0.0
        m = ((cx >= x0 - pad) & (cx < x1 + pad) &
             (cy >= y0 - pad) & (cy < y1 + pad) & valid)
        idx = np.flatnonzero(m.ravel())
        if len(idx) == 0:                       # sub-cell box: nearest valid cell
            d2 = (cx - (x0 + x1) / 2) ** 2 + (cy - (y0 + y1) / 2) ** 2
            d2[~valid] = np.inf
            idx = np.array([int(d2.ravel().argmin())])
        u = np.clip((cx.ravel()[idx] - x0) / max(x1 - x0, 1), 0, 1)
        v = np.clip((cy.ravel()[idx] - y0) / max(y1 - y0, 1), 0, 1)
        bu = np.minimum((u * self.NB).astype(int), self.NB - 1)
        bv = np.minimum((v * self.NB).astype(int), self.NB - 1)
        s = self.size_bin(box)
        z = zn[idx]
        pis = self.pi[s][:, bv, bu]                              # (K,n)
        pfg, _ = estep(z, pis, self.kappa, self.C, self.bg_ll(z), contrast)
        return idx, pfg


def gather_train_instances(tiles, boxes, size_edges, g=G, s=S_PX):
    """-> list of (tile_idx, cell_idx, u, v, size_bin); strict centre-in-box."""
    inst = []
    cy, cx = np.mgrid[0:g, 0:g] * s + s / 2.0
    for ti, p in enumerate(tiles):
        for b in boxes.get(p, []):
            x0, y0, x1, y1 = b
            m = (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
            idx = np.flatnonzero(m.ravel())
            if len(idx) == 0:
                continue
            u = (cx.ravel()[idx] - x0) / max(x1 - x0, 1)
            v = (cy.ravel()[idx] - y0) / max(y1 - y0, 1)
            s = int(np.searchsorted(size_edges,
                                    np.sqrt((x1 - x0) * (y1 - y0))))
            inst.append((ti, idx, u, v, s))
    return inst


def fit(args):
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    split, boxes = load_split()
    tiles, feats, labs, g, s_px = collect_feats(split, boxes, "train",
                                                args.feat_dir)
    print(f"train tiles={len(tiles)} feat_dir={args.feat_dir} "
          f"grid={g} stride={s_px}px", flush=True)

    # centred PCA-D on valid non-boundary cells (labels {0,1}); per-tile fp32
    # upcasts keep peak memory sane for the 4x-denser stride-8 cache
    mu_acc, nv = 0.0, 0
    for f, l in zip(feats, labs):
        v = l >= 0
        mu_acc = mu_acc + f[v].astype(np.float64).sum(0)
        nv += int(v.sum())
    mu = (mu_acc / nv).astype(np.float32)
    sub = np.concatenate([f[l >= 0][::7].astype(np.float32) - mu
                          for f, l in zip(feats, labs)])
    _, sv, Vt = np.linalg.svd(sub, full_matrices=False)
    U = Vt[:args.pca].T                                           # (C,D)
    # whitening: cosine in raw PCA space is dominated by the few high-variance
    # shared "context" directions; unit-variance components let the low-variance
    # local (crown-vs-soil) directions count in the vMF likelihoods
    scale = (sv[:args.pca] / np.sqrt(len(sub)) + 1e-6) if args.whiten \
        else np.ones(args.pca, np.float32)
    n_sub = len(sub)
    del sub
    Zn = []
    for f in feats:
        z = (f.astype(np.float32) - mu) @ U / scale
        Zn.append(z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8))
    del feats
    print(f"PCA done: D={args.pca} whiten={args.whiten} ({n_sub} fit rows)",
          flush=True)

    # frozen background mixture on clear-background cells (label 0) PLUS the
    # near-box ring (context-matched background; see ring_cells)
    clearZ = np.concatenate([z[l == 0] for z, l in zip(Zn, labs)])
    ringZ = np.concatenate([
        z[ring_cells(boxes.get(p, np.zeros((0, 4))),
                     *Image.open(p).size[::-1], g, s_px)]
        for p, z in zip(tiles, Zn)])
    bgZ = np.concatenate([clearZ, ringZ])
    if len(bgZ) > 80000:
        bgZ = bgZ[rng.choice(len(bgZ), 80000, replace=False)]
    # contrastive negatives: emphasise the near-box ring (context-matched soil,
    # the exact stuff a prototype could wrongly absorb) alongside clear bg
    negZ = np.concatenate([ringZ, ringZ, clearZ])
    if len(negZ) > 60000:
        negZ = negZ[rng.choice(len(negZ), 60000, replace=False)]
    Gbg, a = spherical_kmeans(bgZ, args.k_bg, rng)
    wbg = np.bincount(a, minlength=args.k_bg) / len(a)
    wbg = np.clip(wbg, 1e-3, None); wbg /= wbg.sum()
    print(f"bg mixture: K_bg={args.k_bg} on {len(clearZ)} clear + {len(ringZ)} "
          f"ring cells, weights={np.round(wbg, 3)}", flush=True)

    # size terciles over train boxes (px, sqrt-area)
    sizes = np.array([np.sqrt((b[2] - b[0]) * (b[3] - b[1]))
                      for p in tiles for b in boxes.get(p, [])])
    size_edges = np.percentile(sizes, [100 / 3, 200 / 3])
    inst = gather_train_instances(tiles, boxes, size_edges, g, s_px)
    n_cells = sum(len(i[1]) for i in inst)
    print(f"EM over {len(inst)} boxes / {n_cells} cells; size terciles at "
          f"{np.round(size_edges, 1)} px", flush=True)

    kappa = args.kappa
    # background log-lik is constant across EM iters: precompute per tile
    bg_all = [logsumexp(np.log(wbg)[None] + kappa * (z @ Gbg.T), 1) for z in Zn]

    def bg_ll(ti, idx):
        return bg_all[ti][idx]

    # init fg prototypes: k-means on the in-box cells LEAST explained by the
    # bg mixture (below-median bg log-lik) — "in boxes AND rare outside", the
    # commonality criterion applied at init so prototypes don't start as soil
    fgZ = np.concatenate([Zn[ti][idx] for ti, idx, *_ in inst])
    fg_bgll = np.concatenate([bg_ll(ti, idx) for ti, idx, *_ in inst])
    fgZ = fgZ[fg_bgll < np.median(fg_bgll)]
    if len(fgZ) > 60000:
        fgZ = fgZ[rng.choice(len(fgZ), 60000, replace=False)]
    C, _ = spherical_kmeans(fgZ, args.k, rng)

    NB, K, SB = args.bins, args.k, 3
    pi = np.full((SB, K, NB, NB), FG_PRIOR_MASS / K)

    trace = []
    for it in range(args.iters):
        K = C.shape[0]
        acc = np.zeros((SB, K, NB, NB)); cnt = np.zeros((SB, NB, NB))
        newC = np.zeros_like(C)
        fg_mass, n_tot = 0.0, 0
        for ti, idx, u, v, s in inst:
            bu = np.minimum((u * NB).astype(int), NB - 1)
            bv = np.minimum((v * NB).astype(int), NB - 1)
            pfg, r = estep(Zn[ti][idx], pi[s][:, bv, bu], kappa, C,
                           bg_ll(ti, idx), contrast=not args.no_contrast)
            np.add.at(acc[s], (slice(None), bv, bu), r.T)
            np.add.at(cnt[s], (bv, bu), 1)
            newC += r.T @ Zn[ti][idx]                         # bg FROZEN
            fg_mass += pfg.sum(); n_tot += len(idx)
        C = contrastive_update(newC, negZ, kappa, args.contrastive_beta)
        pi = np.clip(acc / np.maximum(cnt, 1)[:, None], 1e-4, None)
        # pin total prior fg mass to the tight-box constant; EM only decides
        # WHERE the mass goes, then cap per-bin certainty
        for s in range(SB):
            pi[s] *= FG_PRIOR_MASS / max(pi[s].sum(0).mean(), 1e-6)
            bs = pi[s].sum(0)
            over = bs > BIN_CAP
            if over.any():
                pi[s][:, over] *= BIN_CAP / bs[over]
        # component pruning: the annotations pick the effective K (truncated
        # DP-style). Start over-provisioned (--k); after burn-in, delete any
        # prototype whose share of fg responsibility mass is below prune_frac
        # of the uniform share 1/K. Deterministic; final K is recorded.
        if it >= args.prune_after and C.shape[0] > 2:
            share = acc.sum((0, 2, 3))
            share = share / max(share.sum(), 1e-9)
            keep = share >= args.prune_frac / C.shape[0]
            if (~keep).any():
                C, pi = C[keep], pi[:, keep]
                print(f"  it{it+1:2d}: pruned {int((~keep).sum())} starved "
                      f"prototype(s) -> K={C.shape[0]}", flush=True)
        trace.append(round(fg_mass / n_tot, 4))
        print(f"  it{it+1:2d}: mean P(fg)={trace[-1]:.3f}", flush=True)

    os.makedirs(ART, exist_ok=True)
    out_path = MODEL_PATH.replace(".npz", f"_{args.tag}.npz") if args.tag \
        else MODEL_PATH
    np.savez(out_path, mu=mu, U=U, scale=scale, C=C, Gbg=Gbg, wbg=wbg, pi=pi,
             kappa=kappa, size_edges=size_edges, feat_dir=args.feat_dir,
             grid=g, s_px=s_px)
    cfg = {"seed": args.seed, "feat_dir": args.feat_dir, "whiten": args.whiten,
           "grid": int(g), "stride_px": int(s_px),
           "contrast": not args.no_contrast,
           "contrastive_beta": args.contrastive_beta,
           "pca": args.pca, "k_init": args.k, "k_effective": int(C.shape[0]),
           "prune_frac": args.prune_frac, "prune_after": args.prune_after,
           "k_bg": args.k_bg,
           "bins": args.bins, "kappa": args.kappa, "iters": args.iters,
           "size_edges_px": [round(float(e), 1) for e in size_edges],
           "n_boxes": len(inst), "n_cells": int(n_cells),
           "fg_mass_trace": trace,
           "fg_prior_mass": round(FG_PRIOR_MASS, 4), "bin_cap": BIN_CAP,
           "fg_init": "duck-filtered k-means (in-box cells with bg_ll < median)",
           "note": "hyperparams fixed a priori, never tuned on masks"}
    json.dump(cfg, open(os.path.join(
        ART, f"fit_report_{args.tag}.json" if args.tag else "fit_report.json"),
        "w"), indent=2)
    print(f"saved {out_path}", flush=True)

    render_priors(pi, tag=args.tag)
    render_montages(inst, Zn, C, pi, kappa, bg_ll, tiles, rng, tag=args.tag,
                    g=g, s=s_px)


def render_priors(pi, tag=""):
    od = out_dir(tag)
    SB, K, NB, _ = pi.shape
    fig, axs = plt.subplots(SB, K + 1, figsize=(2.2 * (K + 1), 2.4 * SB), dpi=120)
    for s in range(SB):
        for k in range(K):
            axs[s, k].imshow(pi[s, k], cmap="viridis", vmin=0, vmax=pi.max())
            axs[s, k].set_title(f"pi[s{s},k{k}]", fontsize=8); axs[s, k].axis("off")
        im = axs[s, K].imshow(pi[s].sum(0), cmap="magma", vmin=0, vmax=1)
        axs[s, K].set_title(f"P(fg|u,v) size-bin {s}", fontsize=8)
        axs[s, K].axis("off")
    fig.colorbar(im, ax=axs[:, K], fraction=0.03)
    fig.suptitle("Learned spatial priors over box-normalized coords "
                 "(rows: small/med/large boxes)", fontsize=10)
    fig.savefig(os.path.join(od, "spatial_priors.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"rendered {od}/spatial_priors.png", flush=True)


def render_montages(inst, Zn, C, pi, kappa, bg_ll, tiles, rng, top=36, crop=32,
                    tag="", g=G, s=S_PX):
    """Top-responsibility 32px crops per prototype: what did each C_k learn?"""
    od = out_dir(tag)
    K = C.shape[0]; NB = pi.shape[2]
    per_k = [[] for _ in range(K)]
    for ti, idx, u, v, s in inst:
        bu = np.minimum((u * NB).astype(int), NB - 1)
        bv = np.minimum((v * NB).astype(int), NB - 1)
        _, r = estep(Zn[ti][idx], pi[s][:, bv, bu], kappa, C, bg_ll(ti, idx))
        for k in range(K):
            j = int(r[:, k].argmax())
            per_k[k].append((float(r[j, k]), ti, int(idx[j])))
    fig, axs = plt.subplots(1, K, figsize=(2.6 * K, 3.0), dpi=130)
    for k in range(K):
        cand = sorted(per_k[k], reverse=True)
        seen, crops = {}, []
        for r_, ti, ci in cand:
            if seen.get(ti, 0) >= 3:
                continue
            seen[ti] = seen.get(ti, 0) + 1
            img = Image.open(tiles[ti]).convert("RGB")
            cx = (ci % g) * s + s // 2
            cy = (ci // g) * s + s // 2
            crops.append(np.asarray(img.crop((cx - crop // 2, cy - crop // 2,
                                              cx + crop // 2, cy + crop // 2))))
            if len(crops) >= top:
                break
        n = int(np.ceil(np.sqrt(len(crops)))) if crops else 1
        canvas = np.zeros((n * crop, n * crop, 3), np.uint8)
        for i, c in enumerate(crops):
            r0, c0 = (i // n) * crop, (i % n) * crop
            canvas[r0:r0 + c.shape[0], c0:c0 + c.shape[1]] = c
        axs[k].imshow(canvas); axs[k].set_title(f"prototype {k}", fontsize=9)
        axs[k].axis("off")
    fig.suptitle("Highest-responsibility 32px patches per crown prototype "
                 "(<=3 per tile)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(od, "prototype_montage.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"rendered {od}/prototype_montage.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--k", type=int, default=16,
                    help="INITIAL prototype count; starved components are "
                         "pruned during EM, the data picks the effective K")
    ap.add_argument("--prune_frac", type=float, default=0.25,
                    help="prune a prototype whose fg-mass share < this fraction "
                         "of the uniform share 1/K")
    ap.add_argument("--prune_after", type=int, default=5,
                    help="EM burn-in iterations before pruning starts")
    ap.add_argument("--k_bg", type=int, default=12)
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--feat_dir", default=DEFAULT_FEATS,
                    help="patch-feature cache; default web_last4_s8 = 8px "
                         "half-stride (cache_s8). Alternatives: web_last4 "
                         "(16px, blocks 21-24), web (blocks 3/6/9/12)")
    ap.add_argument("--tag", default="",
                    help="suffix for model/report/render filenames")
    ap.add_argument("--no_whiten", action="store_true",
                    help="ablation: raw-variance PCA components (whitening is "
                         "default; chosen on the darkness diagnostic, not the "
                         "mask proxies)")
    ap.add_argument("--no_contrast", action="store_true",
                    help="ablation: absolute (non-recentred) E-step posterior")
    ap.add_argument("--contrastive_beta", type=float, default=0.5,
                    help="repel prototypes from the outside-box signature in the "
                         "M-step (0=generative). Fixed a priori, not tuned on masks")
    args = ap.parse_args()
    args.whiten = not args.no_whiten
    fit(args)


if __name__ == "__main__":
    main()
