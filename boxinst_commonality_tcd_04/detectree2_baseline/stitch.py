"""Stitch DetecTree2 per-subtile predictions back to whole 2048 tiles and serialize to OUR
prediction schema (boxes_2048 + masks_rle@512 + scores), so the same evaluate.py scorer +
canopy-ignore rule grades DetecTree2 and our method identically.

Each 2048 tile was predicted as a 3x3 grid of 1024 subtiles @50% overlap, so a crown can
appear in up to 4 subtiles. We dedup with DetecTree2's own clean_crowns rule — cluster
crowns that overlap by IoU>iou_thr OR containment>cont_thr, keep the highest-confidence one
— reimplemented on the 512 scoring raster (fixed-pixel tiles, no geo-projection needed).

Everything lands on the RES=512 raster: subtile origins oy,ox in {0,512,1024} map to 512-
canvas offsets {0,128,256}, and each 1024 subtile mask downsamples by 4 to a 256 block.
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils

RES = 512
SCALE = 2048 / RES                 # 4.0  (2048 tile px per 512 raster px)
SUB = 1024
STRIDE = 512
SUB512 = SUB // int(SCALE)         # 256: a subtile occupies 256x256 of the 512 canvas


def subtile_origins(tile=2048, sub=SUB, stride=STRIDE):
    ys = list(range(0, tile - sub + 1, stride))
    return [(oy, ox) for oy in ys for ox in ys]


def _decode_rle(seg):
    counts = seg["counts"]
    return maskUtils.decode({"size": seg["size"],
                             "counts": counts.encode("ascii") if isinstance(counts, str)
                             else counts}).astype(bool)


def _to_512_canvas(m1024, oy, ox):
    """(1024,1024) bool subtile mask -> (512,512) bool on the whole-tile scoring raster."""
    small = np.asarray(Image.fromarray(m1024.astype(np.uint8) * 255).resize(
        (SUB512, SUB512), Image.BILINEAR)) >= 128
    canvas = np.zeros((RES, RES), bool)
    ry, rx = oy // int(SCALE), ox // int(SCALE)
    canvas[ry:ry + SUB512, rx:rx + SUB512] = small
    return canvas


def _bbox(m):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)  # x0,y0,x1,y1


def load_tile_crowns(pred_dir, tid, min_score=0.1):
    """Read all Prediction_<tid>_<oy>_<ox>.json subtile files -> [(mask512 bool, score, bbox)].

    min_score drops junk low-confidence instances BEFORE decode (DetecTree2's own clean_crowns
    floors at 0.2, so 0.1 is a conservative, recipe-consistent cut) — this is what makes the
    decode + dedup tractable given dets=500/subtile."""
    crowns = []
    for (oy, ox) in subtile_origins():
        fp = os.path.join(pred_dir, f"Prediction_{tid}_{oy}_{ox}.json")
        if not os.path.exists(fp):
            continue
        for inst in json.load(open(fp)):
            if float(inst["score"]) < min_score:
                continue
            m = _to_512_canvas(_decode_rle(inst["segmentation"]), oy, ox)
            bb = _bbox(m)
            if bb is not None:
                crowns.append((m, float(inst["score"]), bb))
    return crowns


def _bbox_overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def dedup(crowns, iou_thr=0.7, cont_thr=0.85):
    """clean_crowns rule on 512 masks: greedily keep highest-score crowns, drop a lower one if
    IoU>iou_thr OR it is >cont_thr contained in an already-kept crown. A bbox-overlap prefilter
    skips the full 512x512 mask-AND for non-overlapping crowns (the dedup speed-up)."""
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


def serialize_tile(crowns):
    """[(mask512, score, bbox)] -> per-tile pred record (boxes @2048, masks RLE @512, scores)."""
    boxes, scores, rles = [], [], []
    for m, s, bb in crowns:
        x0, y0, x1, y1 = bb
        boxes.append([float(x0 * SCALE), float(y0 * SCALE),
                      float(x1 * SCALE), float(y1 * SCALE)])          # xyxy @2048
        scores.append(round(s, 4))
        r = maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
        rles.append({"size": [RES, RES], "counts": r["counts"].decode("ascii")})
    return {"boxes_2048": boxes, "scores": scores, "masks_rle": rles}


def stitch_all(pred_dir, tids, iou_thr=0.7, cont_thr=0.85, min_score=0.1):
    """-> {meta, preds{tid: {boxes_2048, scores, masks_rle}}} in our schema."""
    preds = {}
    for k, tid in enumerate(tids):
        crowns = dedup(load_tile_crowns(pred_dir, tid, min_score), iou_thr, cont_thr)
        preds[tid] = serialize_tile(crowns)
        if (k + 1) % 25 == 0 or k + 1 == len(tids):
            print(f"  stitched {k+1}/{len(tids)} tiles", flush=True)
    meta = {"model": "DetecTree2 (Mask R-CNN R101-FPN) fine-tuned on 792/108, "
                     "1024@50% subtiles, clean_crowns dedup", "mask_res": RES,
            "scale_box_to_mask": SCALE, "n_tiles": len(preds),
            "dedup": {"iou_thr": iou_thr, "cont_thr": cont_thr, "min_score": min_score}}
    return {"meta": meta, "preds": preds}
