"""GT-box bake-off — the shared core: build the prompt boxes, score an arm's masks.

THE EXPERIMENT. Every box→mask module is handed the SAME ground-truth boxes and asked for a
mask. Because the prompt box is derived from GT crown i, the mask it returns is compared
against crown i and nothing else -- there is no matching step, no ranking, no score floor, no
detection confound. Whatever separates the arms is the masker.

This replaces `mps_tcd_multiseed_4phase/masker_lab/sam_bakeoff_results.json` (EM 0.7473 vs SAM
0.7242), which is not citeable: SAM **1** ViT-H against the **16 px** beta=0 EM rather than the
shipped 8 px masker, 60 of 439 tiles, and SAM internally resized 2048->1024 while the EM ran at
native stride -- a handicap in our favour. Metric names here mirror that file so the old and
new numbers can be laid side by side.

FAIRNESS RULES, both enforced here rather than left to each arm:
  * The VALID crown set is computed from GT alone (`valid_mask`) and shared by every arm, so no
    method can improve its mean by declining to segment the hard crowns.
  * Canopy plays no part. We prompt only with individual-crown boxes and score IoU against that
    crown, so a mask spilling into unlabelled canopy is neither rewarded nor punished. This is
    the one place in the project where canopy-ignore is irrelevant rather than merely satisfied.
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as maskUtils

from boxinst_commonality_tcd_04 import evaluate as E

RES = E.RES                     # 512
SCALE = E.SCALE                 # 2048 / 512
RATE_THRS = (0.5, 0.75, 0.9)    # as masker_lab reported them
MIN_GT_PX = 4                   # a crown whose 512-raster mask is smaller than this is a
                                # sub-pixel polygon; IoU against it is noise, not signal


def gt_boxes_2048(polys):
    """Tight xyxy boxes in 2048px tile coords, one per GT crown, in crown order.

    Derived from the polygon rather than from any stored annotation box so the prompt is
    exactly the crown's own extent -- the ideal box, which is the point of the experiment."""
    if not polys:
        return np.zeros((0, 4), np.float32)
    out = []
    for p in polys:
        v = np.asarray(p, np.float32).reshape(-1, 2)
        out.append([*v.min(0), *v.max(0)])
    return np.asarray(out, np.float32).reshape(-1, 4)


def gt_masks_and_valid(polys, res=RES):
    """-> (masks (N,res,res) bool, valid (N,) bool).

    `valid` is a property of the GROUND TRUTH only, and is deliberately evaluated at the
    REFERENCE 512 raster whatever `res` is asked for. Were it recomputed at each resolution the
    512 and 2048 runs would score different crown sets and could not be laid side by side --
    which is the whole point of running both."""
    if not polys:
        return np.zeros((0, res, res), bool), np.zeros(0, bool)
    ref = np.asarray(E.raster(polys, res=RES, scale=2048.0 / RES))
    valid = ref.reshape(len(ref), -1).sum(1) >= MIN_GT_PX
    m = ref if res == RES else np.asarray(E.raster(polys, res=res, scale=2048.0 / res))
    return m, valid


def to_rle(masks):
    """Dense masks -> COCO RLE (bytes `counts`, for pycocotools' C layer)."""
    return [maskUtils.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]


def gt_rles_and_valid(polys, res=RES):
    """GT polygons -> (list of RLE at `res`, valid (N,) bool), WITHOUT ever holding the stack.

    Each polygon is rasterised and encoded one at a time, so peak memory is a single mask
    rather than N. At 2048 the stack for a dense tile would be ~1.8 GB; this is ~4 MB.
    `valid` is still pinned to the 512 reference -- see `gt_masks_and_valid`."""
    if not polys:
        return [], np.zeros(0, bool)
    valid = np.asarray([np.asarray(E.raster([p], res=RES, scale=2048.0 / RES)[0]).sum()
                        >= MIN_GT_PX for p in polys], bool)
    rles = [maskUtils.encode(np.asfortranarray(
        np.asarray(E.raster([p], res=res, scale=2048.0 / res)[0]).astype(np.uint8)))
        for p in polys]
    return rles, valid


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


def double_assignment_rle(pred_rles):
    """(sum of per-instance areas - area of union) / area of union, on RLE."""
    if not pred_rles:
        return None
    union = float(maskUtils.area(maskUtils.merge(list(pred_rles))))
    if union == 0:
        return None
    return float((sum(float(maskUtils.area(r)) for r in pred_rles) - union) / union)


def per_crown_iou(pred_masks, gt_masks):
    """IoU of prediction i against GT crown i -- PAIRWISE, not a matching matrix.

    The two are index-aligned by construction: prompt box i came from crown i."""
    assert len(pred_masks) == len(gt_masks), \
        f"arm returned {len(pred_masks)} masks for {len(gt_masks)} boxes -- must be 1:1"
    if not len(gt_masks):
        return np.zeros(0, np.float64)
    p = pred_masks.reshape(len(pred_masks), -1)
    g = gt_masks.reshape(len(gt_masks), -1)
    inter = np.logical_and(p, g).sum(1).astype(np.float64)
    union = np.logical_or(p, g).sum(1).astype(np.float64)
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def double_assignment(pred_masks):
    """(sum of per-instance masks - union) / union, over one tile.

    Carried over from the masker_lab bake-off because it ran AGAINST us there (EM 0.1058 vs
    SAM 0.0524): a prompt-local module double-claims pixels shared by touching crowns, and a
    per-crown IoU mean cannot see that. Keep measuring it."""
    if not len(pred_masks):
        return None
    stack = pred_masks.reshape(len(pred_masks), -1)
    union = np.logical_or.reduce(stack, axis=0).sum()
    if union == 0:
        return None
    return float((stack.sum() - union) / union)


def summarise(ious, extra=None):
    """Per-crown IoUs -> the same summary shape masker_lab reported."""
    ious = np.asarray(ious, np.float64)
    out = {"n": int(len(ious)),
           "mean_iou": round(float(ious.mean()), 4) if len(ious) else None,
           "rates": {f">={t}": round(float((ious >= t).mean()), 4) for t in RATE_THRS}
                    if len(ious) else None}
    if extra:
        out.update(extra)
    return out
