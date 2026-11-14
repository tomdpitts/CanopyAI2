"""LACE box->mask on arbitrary (non-square) images: per-image test-time fit + deployed lace4px_simple render.

Every numeric primitive (estep, spherical_kmeans, contrastive_update, logsumexp, colour guided filter) is the VERBATIM
repo code (_verbatim.py, source-identity checked). What is written here is the glue, ported line by line from:

  fit_image   <- em.fit (boxinst_commonality_tcd_04/em.py) body, same order of RNG calls. Changes, all geometric:
                 (a) one image instead of 120 tiles; grid (gh, gw) not g x g;
                 (b) the 8 px lattice is the even sub-lattice of our 16-phase 4 px features (= the deployed 4-offset
                     fit lattice), cell centre 8X+8 -> origin 8 (em.fit with cell_origin_frac=1.0);
                 (c) no canopy class (LIVECell has no ignore label) -> canopy mask all False;
                 (d) cells whose centre falls outside the image (zero pad to a multiple of 16) are label -1 and never
                     ring / background (TCD tiles are 2048 = 128 x 16 so this never arose there);
                 (e) if fewer than `pca` rows survive the [::5] PCA subsample, all labelled rows are used (logged).
  Masker4     <- evaluate.TCDMasker.box_mask with the grid generalised to (gh, gw); s = 4, origin = 8 (= btmask m4).
  prob4       <- guided_ablate_modal._prob (EDGE_FILL True) at native resolution (scale 1), non-square canvas.
  render_post4 <- hunt/render.post4 with the lace4px_simple spec {c_peak 0.33, r 12, eps 1e-4, alpha 0.3, kappa 1.6}
                  (lo 0 / hi 1 clip as in post4), window margin 16, mask clipped to the box.
"""
import time

import numpy as np
from PIL import Image

from _verbatim import (BIN_CAP, FG_PRIOR_MASS, color_guided_filter, contrastive_update, estep, logsumexp,
                       spherical_kmeans)

# Frozen deployed settings (sources in README; checked against em_model_4p_fix.npz by check_frozen()).
FIT = dict(seed=0, pca=128, k=16, k_bg=12, bins=8, kappa=10.0, iters=30, prune_after=5, prune_frac=0.25,
           contrastive_beta=0.5)
APPLY = dict(prior_weight=0.3, kappa_scale=1.6, s=4, origin=8.0)
RENDER = dict(r=12, eps=1e-4, c_peak=0.33, margin=16)
DEPLOYED_NPZ = "/Users/tompitts/dphil/CanopyAI2/boxinst_commonality_tcd_04/modal_tcd_multiseed/phase4/em_model_4p_fix.npz"


def check_frozen(npz=DEPLOYED_NPZ):
    d = np.load(npz)
    assert float(d["kappa"]) == FIT["kappa"] and float(d["prior_weight"]) == APPLY["prior_weight"]
    assert float(d["kappa_scale"]) == APPLY["kappa_scale"] and d["C"].shape[0] == FIT["k"]
    assert d["Gbg"].shape[0] == FIT["k_bg"] and d["U"].shape[1] == FIT["pca"] and d["pi"].shape[2] == FIT["bins"]
    assert int(d["s_px"]) == 8 and float(d["cell_origin"]) == 8.0
    return True


# ------------------------------------------------------------------ fit (em.fit, per image)
def _labels(boxes, gh, gw, s, origin, H, W):
    """em.cell_labels_canopy + ring_cells_canopy (canopy none) on a (gh, gw) grid, plus (d) pad cells -> -1."""
    cy, cx = np.mgrid[0:gh, 0:gw] * s + origin
    valid = (cy < H) & (cx < W)
    inbox = np.zeros((gh, gw), bool); near = np.zeros((gh, gw), bool)
    for x0, y0, x1, y1 in boxes:
        inbox |= (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
        near |= ((cx >= x0 - s) & (cx < x1 + s) & (cy >= y0 - s) & (cy < y1 + s))
    lab = -np.ones((gh, gw), np.int8)
    lab[inbox & valid] = 1
    lab[(~near) & valid] = 0
    ring = near & ~inbox & valid
    return lab.ravel(), ring.ravel()


def fit_image(F8, boxes, H, W, log=print):
    """F8: (gh, gw, 1024) float32 raw L2-normed DINOv3 cells on the 8 px lattice (centre 8X+8). boxes (N,4) xyxy px.
    Returns (model dict, report dict). em.fit body; comments mark the deviations (a)-(e)."""
    a = FIT; t0 = time.time()
    rng = np.random.default_rng(a["seed"]); np.random.seed(a["seed"])
    gh, gw, D = F8.shape; s, origin = 8, 8.0
    f = F8.reshape(-1, D)
    l, ring = _labels(boxes, gh, gw, s, origin, H, W)
    rep = {"grid8": [gh, gw], "n_inbox_cells": int((l == 1).sum()), "n_clear_cells": int((l == 0).sum()),
           "n_ring_cells": int(ring.sum()), "deviations": []}
    # centred, whitened PCA on tree+clear-bg cells
    v = l >= 0
    mu = (f[v].astype(np.float64).sum(0) / int(v.sum())).astype(np.float32)
    sub = f[v][::5].astype(np.float32) - mu
    if len(sub) < a["pca"]:                                                             # (e)
        sub = f[v].astype(np.float32) - mu; rep["deviations"].append(f"pca subsample <{a['pca']} rows: used all {len(sub)}")
    _, sv, Vt = np.linalg.svd(sub, full_matrices=False)
    U = Vt[:a["pca"]].T
    scale = sv[:a["pca"]] / np.sqrt(len(sub)) + 1e-6
    del sub
    Zn = (f.astype(np.float32) - mu) @ U / scale
    Zn = Zn / (np.linalg.norm(Zn, axis=1, keepdims=True) + 1e-8)

    clearZ = Zn[l == 0]; ringZ = Zn[ring]
    bgZ = np.concatenate([clearZ, ringZ])
    if len(bgZ) > 80000:
        bgZ = bgZ[rng.choice(len(bgZ), 80000, replace=False)]
    negZ = np.concatenate([ringZ, ringZ, clearZ])
    if len(negZ) > 60000:
        negZ = negZ[rng.choice(len(negZ), 60000, replace=False)]
    if len(bgZ) < a["k_bg"]:
        raise RuntimeError(f"only {len(bgZ)} background cells < K_bg {a['k_bg']}")
    Gbg, asg = spherical_kmeans(bgZ, a["k_bg"], rng)
    wbg = np.bincount(asg, minlength=a["k_bg"]) / len(asg)
    wbg = np.clip(wbg, 1e-3, None); wbg /= wbg.sum()

    sizes = np.array([np.sqrt((b[2] - b[0]) * (b[3] - b[1])) for b in boxes])
    size_edges = np.percentile(sizes, [100 / 3, 200 / 3])
    inst = []
    cy, cx = np.mgrid[0:gh, 0:gw] * s + origin
    for b in boxes:
        x0, y0, x1, y1 = b
        m = (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
        idx = np.flatnonzero(m.ravel())
        if not len(idx):
            continue
        u = (cx.ravel()[idx] - x0) / max(x1 - x0, 1)
        vv = (cy.ravel()[idx] - y0) / max(y1 - y0, 1)
        sb = int(np.searchsorted(size_edges, np.sqrt((x1 - x0) * (y1 - y0))))
        inst.append((idx, u, vv, sb))
    rep.update(n_boxes=len(boxes), n_boxes_in_fit=len(inst), size_edges_px=[round(float(e), 2) for e in size_edges])
    kappa = a["kappa"]
    bg_all = logsumexp(np.log(wbg)[None] + kappa * (Zn @ Gbg.T), 1)
    fgZ = np.concatenate([Zn[idx] for idx, *_ in inst])
    fg_bgll = np.concatenate([bg_all[idx] for idx, *_ in inst])
    fgZ = fgZ[fg_bgll < np.median(fg_bgll)]
    if len(fgZ) > 60000:
        fgZ = fgZ[rng.choice(len(fgZ), 60000, replace=False)]
    if len(fgZ) < a["k"]:
        raise RuntimeError(f"only {len(fgZ)} fg-init cells < K_fg {a['k']}")
    C, _ = spherical_kmeans(fgZ, a["k"], rng)

    NB, SB = a["bins"], 3
    pi = np.full((SB, C.shape[0], NB, NB), FG_PRIOR_MASS / C.shape[0])
    trace = []
    for it in range(a["iters"]):
        K = C.shape[0]
        acc = np.zeros((SB, K, NB, NB)); cnt = np.zeros((SB, NB, NB))
        newC = np.zeros_like(C); fg_mass, n_tot = 0.0, 0
        for idx, u, vv, sb in inst:
            bu = np.minimum((u * NB).astype(int), NB - 1)
            bv = np.minimum((vv * NB).astype(int), NB - 1)
            pfg, r = estep(Zn[idx], pi[sb][:, bv, bu], kappa, C, bg_all[idx], contrast=True)
            np.add.at(acc[sb], (slice(None), bv, bu), r.T)
            np.add.at(cnt[sb], (bv, bu), 1)
            newC += r.T @ Zn[idx]; fg_mass += pfg.sum(); n_tot += len(idx)
        C = contrastive_update(newC, negZ, kappa, a["contrastive_beta"])
        pi = np.clip(acc / np.maximum(cnt, 1)[:, None], 1e-4, None)
        for sb in range(SB):
            pi[sb] *= FG_PRIOR_MASS / max(pi[sb].sum(0).mean(), 1e-6)
            bs = pi[sb].sum(0); over = bs > BIN_CAP
            if over.any():
                pi[sb][:, over] *= BIN_CAP / bs[over]
        if it >= a["prune_after"] and C.shape[0] > 2:
            share = acc.sum((0, 2, 3)); share = share / max(share.sum(), 1e-9)
            keep = share >= a["prune_frac"] / C.shape[0]
            if (~keep).any():
                C, pi = C[keep], pi[:, keep]
        trace.append(round(fg_mass / n_tot, 4))
    model = dict(mu=mu, U=U, scale=scale.astype(np.float32), C=C, Gbg=Gbg, wbg=wbg, pi=pi, kappa=kappa,
                 size_edges=size_edges, s_px=8, cell_origin=8.0, prior_weight=APPLY["prior_weight"],
                 kappa_scale=APPLY["kappa_scale"])
    rep.update(k_effective=int(C.shape[0]), fg_mass_trace=trace, fit_s=round(time.time() - t0, 3))
    return model, rep


# ------------------------------------------------------------------ apply (TCDMasker.box_mask, non-square)
class Masker4:
    def __init__(self, d):
        d = dict(d)
        self.mu, self.U, self.scale = d["mu"], d["U"], d["scale"]
        self.C, self.Gbg, self.wbg = d["C"], d["Gbg"], d["wbg"]
        self.pi, self.kappa = d["pi"], float(d["kappa"])
        self.size_edges = d["size_edges"]
        self.prior_weight = float(d["prior_weight"]); self.kappa_scale = float(d["kappa_scale"])
        self.NB = self.pi.shape[2]
        self.s, self.origin = APPLY["s"], APPLY["origin"]                  # applied on the 4 px lattice (= btmask m4)

    def project(self, F):
        """F (..., 1024) raw cells -> (n, 128) whitened unit vectors (TCDMasker.project on cell-last layout)."""
        z = (F.reshape(-1, F.shape[-1]).astype(np.float32) - self.mu) @ self.U / self.scale
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)

    def bg_ll(self, zn, kappa=None):
        k = self.kappa if kappa is None else kappa
        return logsumexp(np.log(self.wbg)[None] + k * (zn @ self.Gbg.T), 1)

    def size_bin(self, box):
        return int(np.searchsorted(self.size_edges, np.sqrt(max(box[2] - box[0], 1) * max(box[3] - box[1], 1))))

    def box_mask(self, zn, gh, gw, box, contrast=True, prior_weight=None, kappa_scale=None):
        prior_weight = self.prior_weight if prior_weight is None else prior_weight
        kappa_scale = self.kappa_scale if kappa_scale is None else kappa_scale
        s = self.s
        cy, cx = np.mgrid[0:gh, 0:gw] * s + self.origin
        x0, y0, x1, y1 = box
        pad = s / 2.0
        m = ((cx >= x0 - pad) & (cx < x1 + pad) & (cy >= y0 - pad) & (cy < y1 + pad))
        idx = np.flatnonzero(m.ravel())
        if len(idx) == 0:
            d2 = (cx - (x0 + x1) / 2) ** 2 + (cy - (y0 + y1) / 2) ** 2
            idx = np.array([int(d2.ravel().argmin())])
        u = np.clip((cx.ravel()[idx] - x0) / max(x1 - x0, 1), 0, 1)
        v = np.clip((cy.ravel()[idx] - y0) / max(y1 - y0, 1), 0, 1)
        bu = np.minimum((u * self.NB).astype(int), self.NB - 1)
        bv = np.minimum((v * self.NB).astype(int), self.NB - 1)
        sb = self.size_bin(box)
        k = self.kappa * kappa_scale
        pfg, _ = estep(zn[idx], self.pi[sb][:, bv, bu], k, self.C, self.bg_ll(zn[idx], kappa=k), contrast,
                       prior_weight=prior_weight)
        return idx, pfg


def prob4(masker, zn, gh, gw, b, H, W):
    """guided_ablate_modal._prob at scale 1 (EDGE_FILL True) on a (gh*s, gw*s) canvas, cropped to (H, W)."""
    s, origin = masker.s, masker.origin
    delta = int(round(origin - s / 2.0))
    idx, r = masker.box_mask(zn, gh, gw, b, prior_weight=APPLY["prior_weight"], kappa_scale=APPLY["kappa_scale"])
    grid = np.zeros(gh * gw, np.float32); grid[idx] = r
    RH, RW = gh * s, gw * s
    prob = np.array(Image.fromarray(grid.reshape(gh, gw)).resize((RW, RH), Image.BILINEAR))
    if delta:
        sh = np.zeros_like(prob)
        sh[delta:RH, delta:RW] = prob[0:RH - delta, 0:RW - delta]; prob = sh
        if delta > 0:
            prob[:delta, :] = prob[delta, :]; prob[:, :delta] = prob[:, delta:delta + 1]
    prob = prob[:H, :W]
    x0, y0, x1, y1 = np.asarray(b)
    return (max(0, int(y0)), int(np.ceil(y1)), max(0, int(x0)), int(np.ceil(x1))), prob, r


def render_post4(rgbf, clip, prob, H, W):
    """hunt/render.post4 for the lace4px_simple spec -> (H, W) bool mask clipped to the box."""
    by0, by1, bx0, bx1 = clip; M = RENDER["margin"]
    wy0, wy1, wx0, wx1 = max(0, by0 - M), min(H, by1 + M), max(0, bx0 - M), min(W, bx1 + M)
    q = color_guided_filter(rgbf[wy0:wy1, wx0:wx1], prob[wy0:wy1, wx0:wx1], RENDER["r"], RENDER["eps"])
    qb = q[by0 - wy0:by1 - wy0, bx0 - wx0:bx1 - wx0]
    tau = float(np.clip(RENDER["c_peak"] * (float(qb.max()) if qb.size else 0.0), 0.0, 1.0))
    m = np.zeros((H, W), bool); m[wy0:wy1, wx0:wx1] = q >= tau
    c = np.zeros((H, W), bool); c[by0:by1, bx0:bx1] = True
    return m & c


def render_boxes(masker, zn, gh, gw, boxes, rgbf, H, W):
    """-> list of (H, W) bool masks and the per-box 4 px cell posteriors' mean (diagnostic)."""
    out, pbar = [], []
    for b in boxes:
        clip, prob, r = prob4(masker, zn, gh, gw, b, H, W)
        out.append(render_post4(rgbf, clip, prob, H, W)); pbar.append(float(r.mean()))
    return out, pbar


def box_clip(b, H, W):
    """Filled-box control: the same integer clip LACE's mask is intersected with."""
    x0, y0, x1, y1 = b
    m = np.zeros((H, W), bool); m[max(0, int(y0)):int(np.ceil(y1)), max(0, int(x0)):int(np.ceil(x1))] = True
    return m


def ellipse(b, H, W):
    """hunt/render.ellipse (p = 2) on an (H, W) canvas: inscribed ellipse, pixel centres +0.5, clipped to the box."""
    x0, y0, x1, y1 = [float(v) for v in b]
    by0, by1, bx0, bx1 = max(0, int(y0)), min(int(np.ceil(y1)), H), max(0, int(x0)), min(int(np.ceil(x1)), W)
    w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    ys = np.arange(by0, by1)[:, None] + 0.5; xs = np.arange(bx0, bx1)[None] + 0.5
    m = np.zeros((H, W), bool)
    m[by0:by1, bx0:bx1] = (2 * (xs - (x0 + x1) / 2) / w) ** 2 + (2 * (ys - (y0 + y1) / 2) / h) ** 2 <= 1
    return m
