"""LACE cell posterior with a different CONTRAST CENTRE (the only change vs TCDMasker.box_mask / estep).

Deployed estep: pfg = sigmoid((A - mean_box(A)) + alpha * logit(psum)), A = appearance log-ratio of each in-box cell
(cells within box + s/2). Centring on the in-box mean forces part of every box to background: right for a loose box
(crown + soil), wrong for a tight box the crown fills. Alternatives, all training-free:
  'box'   mean over in-box cells (deployed; gate)
  'ring'  mean over RING cells: cell centres within `ring_px` outside the box(+pad) -- the local background
  'mid'   midpoint of the box CORE (cells in the central half of the box, mean) and the ring mean
  'mix'   w * box + (1 - w) * mid
Ring cells use the same appearance term with the prior bin clamped to the box edge (as box_mask does for pad cells).
prob_from_cells reproduces GA._prob's grid -> 2048 bilinear resize and half-cell shift exactly.
"""
import numpy as np
from PIL import Image
from scipy.special import logsumexp


def appearance(m, z, pis, k):
    zc = k * (z @ m.C.T)
    psum = np.clip(pis.sum(0), 1e-4, 1 - 1e-4)
    lw = zc + np.log((pis / psum).T + 1e-9)
    bgll = logsumexp(np.log(m.wbg)[None] + k * (z @ m.Gbg.T), 1)
    return logsumexp(lw, 1) - bgll, psum


def cells(m, g, box, lo_pad, hi_pad):
    s, o = m.s, m.origin
    cy, cx = np.mgrid[0:g, 0:g] * s + o
    x0, y0, x1, y1 = box
    sel = (cx >= x0 - lo_pad) & (cx < x1 + hi_pad) & (cy >= y0 - lo_pad) & (cy < y1 + hi_pad)
    return sel, cx, cy


def posterior(m, zn, g, box, centre="box", ring_px=16, w=0.5, alpha=None, kappa_scale=None):
    """-> (idx, pfg) over the deployed in-box cell set (same idx as box_mask)."""
    alpha = m.prior_weight if alpha is None else alpha
    k = m.kappa * (m.kappa_scale if kappa_scale is None else kappa_scale)
    x0, y0, x1, y1 = [float(v) for v in box]; bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    pad = m.s / 2.0
    inb, cx, cy = cells(m, g, box, pad, pad)
    idx = np.flatnonzero(inb.ravel())
    if len(idx) == 0:
        d2 = (cx - (x0 + x1) / 2) ** 2 + (cy - (y0 + y1) / 2) ** 2
        idx = np.array([int(d2.ravel().argmin())])
    sb = m.size_bin(box); NB = m.NB

    def A_of(ix):
        u = np.clip((cx.ravel()[ix] - x0) / bw, 0, 1); v = np.clip((cy.ravel()[ix] - y0) / bh, 0, 1)
        bu = np.minimum((u * NB).astype(int), NB - 1); bv = np.minimum((v * NB).astype(int), NB - 1)
        return appearance(m, zn[ix], m.pi[sb][:, bv, bu], k)

    A, psum = A_of(idx)
    c_box = A.mean() if len(A) > 1 else 0.0
    if centre == "box":
        c = c_box
    else:
        outer, _, _ = cells(m, g, box, pad + ring_px, pad + ring_px)
        ring = np.flatnonzero((outer & ~inb).ravel())
        c_ring = A_of(ring)[0].mean() if len(ring) else c_box
        if centre == "ring":
            c = c_ring
        else:
            core = np.flatnonzero(((cx >= x0 + bw / 4) & (cx < x1 - bw / 4) & (cy >= y0 + bh / 4) & (cy < y1 - bh / 4)).ravel())
            c_core = A_of(core)[0].mean() if len(core) else A.max()
            c_mid = 0.5 * (c_core + c_ring)
            c = c_mid if centre == "mid" else w * c_box + (1 - w) * c_mid
    pfg = 1.0 / (1.0 + np.exp(-((A - c) + alpha * np.log(psum / (1 - psum)))))
    return idx, pfg


def prob_from_cells(m, g, idx, r, box, res=2048):
    """GA._prob body for given cell posteriors -> (clip, prob)."""
    scale = 2048.0 / res
    delta = int(round((m.origin - m.s / 2.0) / scale))
    grid = np.zeros(g * g, np.float32); grid[idx] = r
    prob = np.array(Image.fromarray(grid.reshape(g, g)).resize((res, res), Image.BILINEAR))
    if delta:
        sh = np.zeros_like(prob)
        s_ = slice(max(0, -delta), res - max(0, delta)); d_ = slice(max(0, delta), res - max(0, -delta))
        sh[d_, d_] = prob[s_, s_]; prob = sh
    x0, y0, x1, y1 = np.asarray(box) / scale
    return (max(0, int(y0)), int(np.ceil(y1)), max(0, int(x0)), int(np.ceil(x1))), prob
