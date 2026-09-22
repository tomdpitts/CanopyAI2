"""RES-parametrised re-implementation of `detectree2_baseline/stitch.py`.

The published stitcher hardcodes `RES = 512` at module scope and derives SCALE/SUB512 from
it, so it cannot be run at another raster without editing a file that produced published
numbers. This module takes `res` as an argument instead. THE ALGORITHM IS OTHERWISE
CHARACTER-FOR-CHARACTER THE SAME -- same decode, same PIL BILINEAR>=128 resample, same
`_bbox`, same greedy `clean_crowns` dedup with the bbox-overlap prefilter, same
serialisation and rounding. It is gated by reproducing the published 512 files EXACTLY
(see `gate_512.py`).

At res=2048: SCALE=1.0, SUB512=1024, so the subtile resize is 1024->1024 (identity) and the
subtile origins {0,512,1024} place directly onto the canvas. Nothing is interpolated and no
information is invented -- Box2Mask emits its masks at 1024 per 1024^2 subtile, i.e. already
at the native 0.1 m/px raster.
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils

SUB = 1024
STRIDE = 512


def subtile_origins(tile=2048, sub=SUB, stride=STRIDE):
    ys = list(range(0, tile - sub + 1, stride))
    return [(oy, ox) for oy in ys for ox in ys]


def _decode_rle(seg):
    counts = seg["counts"]
    return maskUtils.decode({"size": seg["size"],
                             "counts": counts.encode("ascii") if isinstance(counts, str)
                             else counts}).astype(bool)


def _to_canvas(m1024, oy, ox, res):
    """(1024,1024) bool subtile mask -> (res,res) bool on the whole-tile scoring raster."""
    scale = 2048 / res
    subr = SUB // int(scale)
    small = np.asarray(Image.fromarray(m1024.astype(np.uint8) * 255).resize(
        (subr, subr), Image.BILINEAR)) >= 128
    canvas = np.zeros((res, res), bool)
    ry, rx = oy // int(scale), ox // int(scale)
    canvas[ry:ry + subr, rx:rx + subr] = small
    return canvas


def _bbox(m):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)  # x0,y0,x1,y1


def load_tile_crowns(pred_dir, tid, res, min_score=0.1):
    crowns = []
    for (oy, ox) in subtile_origins():
        fp = os.path.join(pred_dir, f"Prediction_{tid}_{oy}_{ox}.json")
        if not os.path.exists(fp):
            continue
        for inst in json.load(open(fp)):
            if float(inst["score"]) < min_score:
                continue
            m = _to_canvas(_decode_rle(inst["segmentation"]), oy, ox, res)
            bb = _bbox(m)
            if bb is not None:
                crowns.append((m, float(inst["score"]), bb))
    return crowns


def _bbox_overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def dedup(crowns, iou_thr=0.7, cont_thr=0.85):
    order = sorted(range(len(crowns)), key=lambda i: -crowns[i][1])
    kept = []          # (mask, score, bbox, area)
    for i in order:
        m, s, bb = crowns[i]
        a = int(m.sum())
        drop = False
        for km, ks, kbb, ka in kept:
            if not _bbox_overlap(bb, kbb):
                continue
            inter = int((m & km).sum())
            if inter == 0:
                continue
            if inter / (a + ka - inter) > iou_thr or inter / a > cont_thr:
                drop = True
                break
        if not drop:
            kept.append((m, s, bb, a))
    return [(m, s, bb) for m, s, bb, _ in kept]


def serialize_tile(crowns, res):
    scale = 2048 / res
    boxes, scores, rles = [], [], []
    for m, s, bb in crowns:
        x0, y0, x1, y1 = bb
        boxes.append([float(x0 * scale), float(y0 * scale),
                      float(x1 * scale), float(y1 * scale)])          # xyxy @2048
        scores.append(round(s, 4))
        r = maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
        rles.append({"size": [res, res], "counts": r["counts"].decode("ascii")})
    return {"boxes_2048": boxes, "scores": scores, "masks_rle": rles}


def stitch_all(pred_dir, tids, res=512, iou_thr=0.7, cont_thr=0.85, min_score=0.1,
               progress=25):
    preds = {}
    for k, tid in enumerate(tids):
        crowns = dedup(load_tile_crowns(pred_dir, tid, res, min_score), iou_thr, cont_thr)
        preds[tid] = serialize_tile(crowns, res)
        if progress and ((k + 1) % progress == 0 or k + 1 == len(tids)):
            print(f"  stitched {k+1}/{len(tids)} tiles", flush=True)
    meta = {"mask_res": res, "scale_box_to_mask": 2048 / res, "n_tiles": len(preds),
            "dedup": {"iou_thr": iou_thr, "cont_thr": cont_thr, "min_score": min_score}}
    return {"meta": meta, "preds": preds}
