"""Phase 0 — build DetecTree2/Detectron2 COCO datasets for the apples-to-apples baseline.

Same cohort as our result: 792 train + 108 val + 439 test (the FIXED split baked into
train_tiles_gt.json / test_gt.json). Each 2048px OAM-TCD tile is sub-tiled to 1024x1024 @
50% overlap (stride 512 -> a 3x3 grid), DetecTree2's native operating resolution. Crown
supervision is the FULL-RES cat-2 `segmentation` from the tile meta (not the coarsened
scoring GT) so the fully-supervised baseline gets its intended mask supervision. Canopy
(cat-1) is emitted as `iscrowd=1` IGNORE regions (category "tree"), mirroring how our own
pipeline + scorer treat canopy as ignore rather than background.

I/O is decoupled from the polygon math so the SAME builder runs locally (from data/tcd,
for dev + sanity renders) and on Modal (from HF restor/tcd images). `build_tile` takes a
2048 RGB array + the raw coco_annotations list and returns per-subtile COCO image/annotation
records (+ the cropped pixel arrays). A crown is included (clipped) in EVERY subtile where
its in-tile area >= MIN_AREA so training tiles stay comprehensively labelled across overlaps;
duplicates are resolved at prediction-stitch time.
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as maskUtils

TILE = 2048
SUB = 1024
STRIDE = 512
MIN_AREA = 40          # px^2 in a 1024 subtile; drop tiny boundary fragments
CAT_TREE = 1           # single thing class "tree" (crowns); canopy = same id, iscrowd=1


def subtile_origins(tile=TILE, sub=SUB, stride=STRIDE):
    """Top-left (oy, ox) of each subtile. stride=sub/2 -> 50% overlap, full coverage."""
    ys = list(range(0, tile - sub + 1, stride))
    xs = list(range(0, tile - sub + 1, stride))
    if ys[-1] != tile - sub:
        ys.append(tile - sub)
    if xs[-1] != tile - sub:
        xs.append(tile - sub)
    return [(oy, ox) for oy in ys for ox in xs]


def seg_to_mask(seg, h=TILE, w=TILE):
    """cat `segmentation` (polygon list OR COCO RLE, compressed or uncompressed) -> (h,w) bool."""
    if isinstance(seg, dict):                       # RLE
        counts = seg.get("counts")
        if isinstance(counts, list):                # uncompressed RLE
            rle = maskUtils.frPyObjects(seg, h, w)
        else:                                       # compressed RLE (counts str/bytes)
            rle = {"size": seg["size"],
                   "counts": counts.encode("ascii") if isinstance(counts, str) else counts}
        return maskUtils.decode(rle).astype(bool)
    # polygon list -> merge parts
    rles = maskUtils.frPyObjects(seg, h, w)
    rle = maskUtils.merge(rles)
    return maskUtils.decode(rle).astype(bool)


def _mask_to_rle(m):
    """(H,W) bool subtile mask -> compressed COCO RLE (counts as ascii str, JSON-safe).
    Lossless (no contour approximation); Detectron2 consumes it with MASK_FORMAT='bitmask'."""
    r = maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
    return {"size": [int(m.shape[0]), int(m.shape[1])],
            "counts": r["counts"].decode("ascii")}


def build_tile(tid, img, anns, img_id0, ann_id0, sub=SUB, stride=STRIDE,
               min_area=MIN_AREA, want_pixels=True):
    """One 2048 tile -> per-subtile COCO records.

    img: (2048,2048,3) uint8. anns: raw coco_annotations list (each {category_id,
    segmentation,...}); cat 2 = crown (positive), cat 1 = canopy (iscrowd ignore).
    Returns (images, annotations, crops) where crops[k]=(file_name,(SUB,SUB,3) uint8)
    aligned to images[k]; img_id/ann_id continue from img_id0/ann_id0.
    """
    h, w = img.shape[:2]
    # precompute full-tile masks once per ann (cat 1 & 2 only)
    masks = []
    for a in anns:
        c = a.get("category_id")
        if c in (1, 2):
            masks.append((c, seg_to_mask(a["segmentation"], h, w)))
    images, annotations, crops = [], [], []
    iid, aid = img_id0, ann_id0
    for (oy, ox) in subtile_origins(h, sub, stride):
        fn = f"{tid}_{oy}_{ox}.png"
        sub_anns = []
        for c, m in masks:
            sm = m[oy:oy + sub, ox:ox + sub]
            area = int(sm.sum())
            if area < min_area:
                continue
            ys, xs = np.where(sm)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            sub_anns.append({
                "id": aid, "image_id": iid, "category_id": CAT_TREE,
                "segmentation": _mask_to_rle(sm), "area": float(area),
                "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],   # COCO xywh, subtile coords
                "iscrowd": 1 if c == 1 else 0})               # canopy -> ignore
            aid += 1
        # keep a subtile only if it carries at least one crown OR canopy (comprehensive
        # tiles); empty-sky subtiles add nothing and skew the background stats
        if not sub_anns:
            continue
        images.append({"id": iid, "file_name": fn, "width": sub, "height": sub,
                       "tile_id": tid, "oy": oy, "ox": ox})
        annotations.extend(sub_anns)
        if want_pixels:
            crops.append((fn, np.ascontiguousarray(img[oy:oy + sub, ox:ox + sub])))
        iid += 1
    return images, annotations, crops


def coco_dict(images, annotations):
    return {"images": images, "annotations": annotations,
            "categories": [{"id": CAT_TREE, "name": "tree"}]}
