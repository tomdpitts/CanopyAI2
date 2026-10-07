"""Soft box: the detector's box becomes a PRIOR on crown extent, not a hard wall.

Deployed masker (em.TCDMasker.box_mask + evaluate.pred_instance_masks) uses the box three ways:
  1. support   only cells whose centre lies inside the box (+ half-cell pad) are scored;
  2. prior     the learned spatial prior pi is indexed in box-relative (u,v) in [0,1];
  3. clip      the rendered mask is ANDed with the box rectangle.
A crown the box cuts off can therefore never be recovered (61% of near-miss detections on the
val tiles have >20% of their crown outside the box; NATIVE_RASTER.md, 2026-10-06).

Soft box changes exactly those three things and nothing else:
  1. support   cells inside the box expanded by `margin` x (w, h) on every side;
  2. prior     inside the box: unchanged. Outside: the nearest edge bin's prior (u,v clipped,
               as box_mask already does) with its FG mass scaled by exp(-d / lam), where d is
               the distance beyond the box edge in units of box size (max over x, y);
  3. clip      to the expanded window, not the box.
The E-step's contrast recentring (A - mean A) is taken over the ORIGINAL in-box cells only, so
in-box posteriors are unchanged by the extra context; out-of-box cells are scored against the
same reference. With margin=0 and clip="box" this is the deployed path, bit for bit (gate).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from boxinst_commonality.em import logsumexp


def _estep_ref(z, pis, kappa, C, bgll, ref, prior_weight):
    """estep() from boxinst_commonality/em.py with contrast=True, except the recentring mean
    is taken over the cells flagged `ref` (the original box) instead of all cells."""
    zc = kappa * (z @ C.T)
    psum = np.clip(pis.sum(0), 1e-4, 1 - 1e-4)
    lw = zc + np.log((pis / psum).T + 1e-9)
    A = logsumexp(lw, 1) - bgll
    if len(A) > 1:
        A = A - A[ref].mean()
    return 1.0 / (1.0 + np.exp(-(A + prior_weight * np.log(psum / (1 - psum)))))


def soft_box_mask(masker, zn, g, box, margin, lam, prior_weight, kappa_scale):
    """-> (idx, pfg) over the expanded window. margin=0 reproduces masker.box_mask exactly."""
    s = masker.s
    cy, cx = np.mgrid[0:g, 0:g] * s + masker.origin
    x0, y0, x1, y1 = box
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    pad = s / 2.0
    inb = ((cx >= x0 - pad) & (cx < x1 + pad) & (cy >= y0 - pad) & (cy < y1 + pad)).ravel()
    wx0, wy0, wx1, wy1 = x0 - margin * w, y0 - margin * h, x1 + margin * w, y1 + margin * h
    win = ((cx >= wx0 - pad) & (cx < wx1 + pad) & (cy >= wy0 - pad) & (cy < wy1 + pad)).ravel()
    idx = np.flatnonzero(win)
    if not inb.any():                                     # deployed fallback: nearest cell
        d2 = (cx - (x0 + x1) / 2) ** 2 + (cy - (y0 + y1) / 2) ** 2
        c = int(d2.ravel().argmin()); inb[c] = True
        if c not in set(idx.tolist()):
            idx = np.sort(np.append(idx, c))
    ref = inb[idx]
    px, py = cx.ravel()[idx], cy.ravel()[idx]
    u = np.clip((px - x0) / w, 0, 1); v = np.clip((py - y0) / h, 0, 1)
    bu = np.minimum((u * masker.NB).astype(int), masker.NB - 1)
    bv = np.minimum((v * masker.NB).astype(int), masker.NB - 1)
    sb = masker.size_bin(box)
    pis = masker.pi[sb][:, bv, bu].copy()                 # (K, n)
    if margin > 0:
        dx = np.maximum(np.maximum(x0 - px, px - x1), 0) / w
        dy = np.maximum(np.maximum(y0 - py, py - y1), 0) / h
        d = np.maximum(dx, dy)
        out = d > 0
        if out.any():
            pis[:, out] *= np.exp(-d[out] / lam)[None]    # decay FG mass outside the box
    k = masker.kappa * kappa_scale
    pfg = _estep_ref(zn[idx], pis, k, masker.C, masker.bg_ll(zn[idx], kappa=k), ref, prior_weight)
    return idx, pfg, (wx0, wy0, wx1, wy1)


def soft_instance_masks(masker, zn, g, boxes, res, scale, mask_thr, prior_weight, kappa_scale,
                        margin, lam, clip="window"):
    """evaluate.pred_instance_masks with the soft box. clip in {"box", "window"}."""
    out = []
    origin = getattr(masker, "origin", getattr(masker, "s", 2 * scale) / 2.0)
    s = getattr(masker, "s", 2 * scale)
    delta = int(round((origin - s / 2.0) / scale))
    for b in boxes:
        idx, r, wb = soft_box_mask(masker, zn, g, b, margin, lam, prior_weight, kappa_scale)
        grid = np.zeros(g * g, np.float32); grid[idx] = r
        prob = np.array(Image.fromarray(grid.reshape(g, g)).resize((res, res), Image.BILINEAR))
        if delta:
            shifted = np.zeros_like(prob)
            src = slice(max(0, -delta), res - max(0, delta)); dst = slice(max(0, delta), res - max(0, -delta))
            shifted[dst, dst] = prob[src, src]; prob = shifted
        m = prob >= mask_thr
        x0, y0, x1, y1 = (np.asarray(b if clip == "box" else wb, float) / scale)
        cm = np.zeros((res, res), bool)
        cm[max(0, int(y0)):int(np.ceil(y1)), max(0, int(x0)):int(np.ceil(x1))] = True
        out.append(m & cm)
    return np.array(out) if out else np.zeros((0, res, res), bool)
