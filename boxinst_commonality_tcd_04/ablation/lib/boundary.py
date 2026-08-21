"""Boundary-F1 (BF-score) on MATCHED true positives.

Why this file exists rather than reusing phase4_research/boundary_f1.py:
that script's "ours" arm is `crown_mask` -- the RGB-guided-filter research variant that
phase4/README.md records as FAILED ("RGB/vegetation-guided refinement FAILED: TCD
boundaries are crown-to-CROWN, image-invisible"). Its 0.286 therefore says nothing about
the deployed EM masker. It also scored 13 tiles x <=30 size-filtered crowns on GT boxes at
2048 px. Here we measure the DEPLOYED system: all 439 tiles, predicted boxes, the 512
scoring raster, matched TPs only.

The `bf` formula is ported verbatim from that script so the two remain commensurable in
form (see `phase4_research/boundary_f1.py:17-23`).

TOLERANCE: computed on the 512 raster, so a tolerance of t px here corresponds to 4t px in
2048 tile coordinates. Reported tolerances are always labelled with both.
"""
import numpy as np
from scipy import ndimage


def bf(pred, gt, tol=3):
    """Boundary F1 within `tol` px. Ported from phase4_research/boundary_f1.py:17."""
    bp = pred ^ ndimage.binary_erosion(pred)
    bg = gt ^ ndimage.binary_erosion(gt)
    if bp.sum() == 0 or bg.sum() == 0:
        return 0.0
    dg = ndimage.distance_transform_edt(~bg)
    dp = ndimage.distance_transform_edt(~bp)
    prec = (dg[bp] <= tol).mean()
    rec = (dp[bg] <= tol).mean()
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def match_tps(iou, scores, ignore, iou_thr=0.5):
    """Greedy score-ordered matching, identical rule to evaluate._greedy_ap:96.

    Returns list of (pred_idx, gt_idx) for matched TPs. `ignore` is accepted for signature
    parity with the AP core; it only affects which UNMATCHED preds are dropped, so it
    cannot change the matched set.
    """
    if len(scores) == 0 or iou.shape[1] == 0:
        return []
    order = np.argsort(-scores)
    matched = np.zeros(iou.shape[1], bool)
    out = []
    for i in order:
        j = int(np.argmax(iou[i]))
        if iou[i, j] >= iou_thr and not matched[j]:
            matched[j] = True
            out.append((int(i), j))
    return out


def box_fill_mask(box_2048, res=512, scale=4.0):
    """The filled predicted box, on the mask raster -- the trivial-baseline boundary."""
    m = np.zeros((res, res), bool)
    x0, y0, x1, y1 = np.asarray(box_2048, float) / scale
    x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1, y1 = min(res, int(np.ceil(x1))), min(res, int(np.ceil(y1)))
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = True
    return m


def bf_multi(pred, gt, tols=(1, 2, 3, 5)):
    """BF at several tolerances for one mask pair, in one pass.

    Two optimisations over calling `bf` per tolerance, both exact (not approximations):
      1. The distance transforms are tolerance-INDEPENDENT -- compute once, threshold many.
      2. Crop to the union bounding box + margin. Crowns occupy ~5-55 px of the 512 raster,
         so a full-frame EDT wastes ~99% of the work. The margin (> max tol + erosion
         reach) guarantees both masks stay interior to the crop, so erosion and the
         <=tol tests are identical to the uncropped result.
    Returns {tol: f1}.
    """
    ys, xs = np.nonzero(pred | gt)
    if len(ys) == 0:
        return {t: 0.0 for t in tols}
    m = int(max(tols)) + 2
    y0, y1 = max(0, ys.min() - m), min(pred.shape[0], ys.max() + m + 1)
    x0, x1 = max(0, xs.min() - m), min(pred.shape[1], xs.max() + m + 1)
    p, g = pred[y0:y1, x0:x1], gt[y0:y1, x0:x1]
    bp = p ^ ndimage.binary_erosion(p)
    bg = g ^ ndimage.binary_erosion(g)
    if bp.sum() == 0 or bg.sum() == 0:
        return {t: 0.0 for t in tols}
    dg = ndimage.distance_transform_edt(~bg)[bp]
    dp = ndimage.distance_transform_edt(~bp)[bg]
    out = {}
    for t in tols:
        prec, rec = (dg <= t).mean(), (dp <= t).mean()
        out[t] = 0.0 if prec + rec == 0 else float(2 * prec * rec / (prec + rec))
    return out
