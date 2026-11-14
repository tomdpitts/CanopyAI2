"""VERBATIM copies (generated 2026-11-12 by ast.get_source_segment; do not edit) of:
  boxinst_commonality/em.py: spherical_kmeans, logsumexp, softmax, estep, contrastive_update  (file sha1 2735a5a1d5ec)
  boxinst_commonality_tcd_04/native_raster/guided_ablate_modal.py: color_guided_filter  (file sha1 67112a6921d9)
  boxinst_commonality_tcd_04/box_supervised_baselines/gtbox_lib.py: per_crown_iou_rle, double_assignment_rle  (file sha1 781aadee0806)
verify() re-reads the source files and asserts each function's text is still identical.
"""
import ast
import numpy as np
from pycocotools import mask as maskUtils

FG_PRIOR_MASS = np.pi / 4   # boxinst_commonality/em.py L49
BIN_CAP = 0.95              # boxinst_commonality/em.py L50
SOURCES = {'boxinst_commonality/em.py': ['spherical_kmeans', 'logsumexp', 'softmax', 'estep', 'contrastive_update'], 'boxinst_commonality_tcd_04/native_raster/guided_ablate_modal.py': ['color_guided_filter'], 'boxinst_commonality_tcd_04/box_supervised_baselines/gtbox_lib.py': ['per_crown_iou_rle', 'double_assignment_rle']}


# --- from boxinst_commonality/em.py
def spherical_kmeans(X, k, rng, iters=25):
    """X:(N,D) L2-normed -> centroids (k,D) L2-normed, assignment (N,)."""
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        a = np.argmax(X @ C.T, 1)
        C = np.stack([X[a == j].mean(0) if (a == j).any() else C[j]
                      for j in range(k)])
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    return C, np.argmax(X @ C.T, 1)


# --- from boxinst_commonality/em.py
def logsumexp(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


# --- from boxinst_commonality/em.py
def softmax(a, axis):
    e = np.exp(a - a.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# --- from boxinst_commonality/em.py
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


# --- from boxinst_commonality/em.py
def estep(z, pis, kappa, C, bgll, contrast=True, prior_weight=1.0):
    """Shared E-step for the cells of ONE box.

    z (n,D) cells; pis (K,n) spatial prior at each cell's (u,v) bin; bgll (n,).
    Returns (pfg (n,) total fg posterior, r (n,K) per-prototype responsibility).

    contrast recentres the appearance log-ratio within the box: the ABSOLUTE
    fg/bg ratio is confounded by ViT context (all in-box cells score fg), while
    the within-box ordering carries the local crown-vs-soil signal (verified
    against a supervised brightness ceiling). Wide spread (crown+soil box) ->
    carving; narrow spread (dense canopy) -> box stays filled.

    prior_weight (default 1.0 = original) scales the box-normalized spatial-prior
    logit. <1 relaxes trust in the (possibly imprecise) box so the box-INDEPENDENT
    appearance term A drives — a box-robustness knob for imprecise PREDICTED boxes
    (fit/train paths keep the default). Sharpening the appearance term is done by the
    caller scaling `kappa` (both zc and bgll), not here.
    """
    zc = kappa * (z @ C.T)                              # (n,K)
    psum = np.clip(pis.sum(0), 1e-4, 1 - 1e-4)
    lw = zc + np.log((pis / psum).T + 1e-9)             # component log-weights
    A = logsumexp(lw, 1) - bgll                         # appearance log-ratio
    if contrast and len(A) > 1:
        A = A - A.mean()
    pfg = 1.0 / (1.0 + np.exp(-(A + prior_weight * np.log(psum / (1 - psum)))))
    e = lw - lw.max(1, keepdims=True)
    wk = np.exp(e)
    wk /= wk.sum(1, keepdims=True)
    return pfg, pfg[:, None] * wk


# --- from boxinst_commonality_tcd_04/native_raster/guided_ablate_modal.py
def color_guided_filter(I, p, r, eps):
    """He et al. 2010, colour guide -- identical to lace_guided_modal.py / guided_rgb_ext_val.py."""
    from scipy.ndimage import uniform_filter
    k = 2 * r + 1
    mean = lambda x: uniform_filter(x, size=k, mode="reflect")
    R, G, B = I[..., 0], I[..., 1], I[..., 2]
    mR, mG, mB, mp = mean(R), mean(G), mean(B), mean(p)
    cR, cG, cB = mean(R * p) - mR * mp, mean(G * p) - mG * mp, mean(B * p) - mB * mp
    rr = mean(R * R) - mR * mR + eps; gg = mean(G * G) - mG * mG + eps; bb = mean(B * B) - mB * mB + eps
    rg = mean(R * G) - mR * mG; rb = mean(R * B) - mR * mB; gb = mean(G * B) - mG * mB
    i_rr = gg * bb - gb * gb; i_rg = gb * rb - rg * bb; i_rb = rg * gb - gg * rb
    i_gg = rr * bb - rb * rb; i_gb = rb * rg - rr * gb; i_bb = rr * gg - rg * rg
    det = i_rr * rr + i_rg * rg + i_rb * rb
    aR = (i_rr * cR + i_rg * cG + i_rb * cB) / det
    aG = (i_rg * cR + i_gg * cG + i_gb * cB) / det
    aB = (i_rb * cR + i_gb * cG + i_bb * cB) / det
    b = mp - aR * mR - aG * mG - aB * mB
    return mean(aR) * R + mean(aG) * G + mean(aB) * B + mean(b)


# --- from boxinst_commonality_tcd_04/box_supervised_baselines/gtbox_lib.py
def per_crown_iou_rle(pred_rles, gt_rles):
    """IoU of prediction i against GT crown i, computed on RLE -- pairwise, not a matrix.

    Same quantity as `per_crown_iou`, but the intersection runs in pycocotools' C layer instead
    of over dense 2048^2 boolean arrays, which at this resolution is both far faster and the
    difference between fitting in memory and not."""
    assert len(pred_rles) == len(gt_rles), \
        f"arm returned {len(pred_rles)} masks for {len(gt_rles)} boxes -- must be 1:1"
    out = np.zeros(len(gt_rles), np.float64)
    for i, (p, g) in enumerate(zip(pred_rles, gt_rles)):
        inter = float(maskUtils.area(maskUtils.merge([p, g], intersect=1)))
        union = float(maskUtils.area(p)) + float(maskUtils.area(g)) - inter
        out[i] = inter / union if union > 0 else 0.0
    return out


# --- from boxinst_commonality_tcd_04/box_supervised_baselines/gtbox_lib.py
def double_assignment_rle(pred_rles):
    """(sum of per-instance areas - area of union) / area of union, on RLE."""
    if not pred_rles:
        return None
    union = float(maskUtils.area(maskUtils.merge(list(pred_rles))))
    if union == 0:
        return None
    return float((sum(float(maskUtils.area(r)) for r in pred_rles) - union) / union)


def verify(repo="/Users/tompitts/dphil/CanopyAI2"):
    import inspect, os, sys
    me = sys.modules[__name__]
    for f, fns in SOURCES.items():
        text = open(os.path.join(repo, f)).read()
        orig = {n.name: ast.get_source_segment(text, n) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
        for fn in fns:
            assert inspect.getsource(getattr(me, fn)).rstrip() == orig[fn].rstrip(), f"{fn} drifted from {f}"
    return True
