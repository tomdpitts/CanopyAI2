"""SelvaBox box selection replayed from the persisted per-subtile `raw_dets`, with the released
CanopyRS edge band optionally applied at INTERIOR subtile edges. Pure numpy; imported by the
local planner and shipped into the Modal refill app.

Released rule (geodataset `aggregator/aggregator.py::_filter_edge_polygons`, called from
`Aggregator.__init__` BEFORE score thresholding and NMS; github.com/hugobaudchon/geodataset
@184141a; CanopyRS presets set `edge_band_buffer_percentage: 0.05`): for each tile, safe area =
tile extent `buffer(-0.05 * tile_width)`; a polygon is kept only if `geometry.within(safe)`.
For our 1024 px subtiles: margin 51.2 px; a box is kept iff it lies inside
[o + 51.2, o + 1024 - 51.2] on every axis (boundary contact allowed, as `within`).

We apply it only at edges interior to the 2048 tile; the outer frame is not culled (paper
appendix "SelvaBox edge trim": on the outer frame it caps recall against a GT that contains
frame-cut crowns).

Replay precision. `raw_dets` were persisted ROUNDED (boxes 2 dp, scores 4 dp) while the
original selection ran on unrounded float32. Plain replay (`stored_keys=None`) therefore
breaks score ties differently (overlapping subtiles emit near-duplicate boxes with equal 4-dp
scores) and mis-floors scores that round to 0.05. With `stored_keys` (the kept set of the
original run) ties are broken toward the box the original run kept, and a 0.05-rounded score
passes the floor only if the original kept it. Edge band OFF, the kept SET equals the stored
one on 330/439 tiles; overall 95 replay-only and 90 stored-only boxes of 491,847 (2026-10-14,
seamfix/selvabox_seam_plan.json); the residue is 2-dp box rounding at the IoU-0.7 boundary.
The gate is therefore at AP level, after the SAM refill (selvabox_seamfix_modal.py).
"""
from __future__ import annotations

import numpy as np

TILE, SUB = 2048, 1024
NMS_IOU, SCORE_FLOOR, KEEP_DETS = 0.7, 0.05, 1600
EDGE_FRAC = 0.05


def nms(boxes, scores, thr):
    """Verbatim `selvabox_native_modal._nms`."""
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = rest[inter / (a[i] + a[rest] - inter + 1e-9) <= thr]
    return np.asarray(keep, int)


def interior_safe(b, oy, ox, margin, sub=SUB, tile=TILE):
    """Bool per box (n,4 xyxy, whole-tile coords): inside the inset subtile at interior edges."""
    ok = np.ones(len(b), bool)
    if ox > 0:
        ok &= b[:, 0] >= ox + margin
    if ox + sub < tile:
        ok &= b[:, 2] <= ox + sub - margin
    if oy > 0:
        ok &= b[:, 1] >= oy + margin
    if oy + sub < tile:
        ok &= b[:, 3] <= oy + sub - margin
    return ok


def nms_ordered(boxes, order, thr):
    """`nms` with the visiting order given (for oracle tie-breaking)."""
    a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        order = rest[inter / (a[i] + a[rest] - inter + 1e-9) <= thr]
    return np.asarray(keep, int)


def select(raw_dets, edge_frac=None, stored_keys=None):
    """raw_dets: list of {oy, ox, boxes_2048, scores}. Returns (boxes, scores, n_edge_dropped).
    stored_keys: set of `key`s kept by the original run -> oracle tie-breaking (see module doc)."""
    B, S, n_drop = [], [], 0
    for r in raw_dets:
        b = np.asarray(r["boxes_2048"], np.float64).reshape(-1, 4)
        s = np.asarray(r["scores"], np.float64)
        if edge_frac is not None:
            k = interior_safe(b, r["oy"], r["ox"], edge_frac * SUB)
            n_drop += int((~k).sum())
            b, s = b[k], s[k]
        B.append(b); S.append(s)
    if not B:
        return np.zeros((0, 4)), np.zeros(0), n_drop
    B = np.concatenate(B); S = np.concatenate(S)
    if stored_keys is None:
        k = S >= SCORE_FLOOR; B, S = B[k], S[k]
        k = nms(B, S, NMS_IOU)
    else:
        ink = np.array([key(b, s) in stored_keys for b, s in zip(B, S)], bool)
        k = (S > SCORE_FLOOR + 1e-9) | ((np.abs(S - SCORE_FLOOR) < 1e-9) & ink)
        B, S, ink = B[k], S[k], ink[k]
        k = nms_ordered(B, np.lexsort((~ink, -S)), NMS_IOU)
    B, S = B[k], S[k]
    if len(S) > KEEP_DETS:
        k = np.argsort(-S)[:KEEP_DETS]; k.sort(); B, S = B[k], S[k]
    return B, S, n_drop


def key(box, score):
    """Identity of a prompted box as stored (2 dp box, 4 dp score)."""
    return tuple(round(float(v), 2) for v in box) + (round(float(score), 4),)
