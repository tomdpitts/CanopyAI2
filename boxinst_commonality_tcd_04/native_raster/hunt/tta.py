"""Per-tile test-time EM for the LACE masker (transductive, unsupervised, training-free).

The deployed masker's vMF prototypes C (fg, K=16) and background mixture Gbg (12) are fit once on train tiles. Here,
for ONE tile and its detections, we run `n_iter` EM steps of LACE's own E-step (TCDMasker.box_mask -> per-cell fg
posterior and per-prototype responsibility) over all in-box cells of the tile's detections (optionally only the top
`top` by product score), then a MAP M-step on the unit sphere anchored to the global model:
    C_k   <- normalize(lam * C_k   + sum_i r_ik z_i)          (r_ik = fg responsibility of prototype k)
    Gbg_j <- normalize(lam * Gbg_j + sum_i b_ij z_i)          (b_ij = (1 - pfg_i) * bg responsibility j, in-box cells
                                                              plus ring cells outside every box with weight 1)
lam is in units of cells (prior pseudo-count). Mixture weights, kappa, spatial prior, size bins are NOT changed.
Returns an adapted copy of the masker (same interface), so _prob / render code is unchanged.
"""
import copy
import numpy as np
from scipy.special import logsumexp


def adapt(m, zn, g, boxes, scores=None, lam=200.0, n_iter=2, top=None, ring_px=16, adapt_bg=True):
    if len(boxes) == 0:
        return m
    order = np.argsort(-np.asarray(scores)) if scores is not None else np.arange(len(boxes))
    if top:
        order = order[:top]
    s, o = m.s, m.origin
    cy, cx = np.mgrid[0:g, 0:g] * s + o
    cyr, cxr = cy.ravel(), cx.ravel()
    covered = np.zeros(g * g, bool); near = np.zeros(g * g, bool)
    for b in np.asarray(boxes)[order]:
        x0, y0, x1, y1 = b; pad = s / 2.0
        covered |= ((cxr >= x0 - pad) & (cxr < x1 + pad) & (cyr >= y0 - pad) & (cyr < y1 + pad))
        near |= ((cxr >= x0 - pad - ring_px) & (cxr < x1 + pad + ring_px) & (cyr >= y0 - pad - ring_px) & (cyr < y1 + pad + ring_px))
    ring = np.flatnonzero(near & ~covered)
    mm = copy.copy(m); mm.C = m.C.copy(); mm.Gbg = m.Gbg.copy()
    k = m.kappa * m.kappa_scale
    for _ in range(n_iter):
        SC = np.zeros_like(mm.C); SB = np.zeros_like(mm.Gbg)
        for b in np.asarray(boxes)[order]:
            idx, pfg = mm.box_mask(zn, g, b)
            z = zn[idx]
            # per-prototype fg responsibility (as estep's wk, recomputed with the box's prior)
            x0, y0, x1, y1 = b; bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
            u = np.clip((cxr[idx] - x0) / bw, 0, 1); v = np.clip((cyr[idx] - y0) / bh, 0, 1)
            bu = np.minimum((u * mm.NB).astype(int), mm.NB - 1); bv = np.minimum((v * mm.NB).astype(int), mm.NB - 1)
            pis = mm.pi[mm.size_bin(b)][:, bv, bu]
            psum = np.clip(pis.sum(0), 1e-4, 1 - 1e-4)
            lw = k * (z @ mm.C.T) + np.log((pis / psum).T + 1e-9)
            wk = np.exp(lw - logsumexp(lw, 1, keepdims=True))
            SC += (pfg[:, None] * wk).T @ z
            if adapt_bg:
                lb = np.log(mm.wbg)[None] + k * (z @ mm.Gbg.T)
                SB += ((1 - pfg)[:, None] * np.exp(lb - logsumexp(lb, 1, keepdims=True))).T @ z
        if adapt_bg and len(ring):
            zr = zn[ring]; lb = np.log(mm.wbg)[None] + k * (zr @ mm.Gbg.T)
            SB += np.exp(lb - logsumexp(lb, 1, keepdims=True)).T @ zr
        C = lam * m.C + SC; mm.C = C / np.linalg.norm(C, axis=1, keepdims=True)
        if adapt_bg:
            G = lam * m.Gbg + SB; mm.Gbg = G / np.linalg.norm(G, axis=1, keepdims=True)
    return mm
