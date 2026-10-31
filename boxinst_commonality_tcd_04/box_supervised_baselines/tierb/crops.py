"""Tier B geometry and schedule arithmetic: pure python, imported by the containers and the local
tests alike, so the rules below exist in exactly one place.

TRAINING CROPS (B1 BoxInst, B2 BoxTeacher). LACE trains on whole 2048 tiles, every pixel once per
epoch. The released BoxInst/BoxTeacher recipes train on ~1 MP images (COCO short side 640-800,
batch 16), and a 1024^2 crop is the same pixel count, so each tile is split into a JITTERED 2x2
PARTITION:
  * per crop, a scale s is drawn from the released multi-scale set taken RELATIVE to the released
    test size (MIN_SIZE_TRAIN / MIN_SIZE_TEST = 640..800 / 800 -> s in {0.80, ..., 1.00}); the test
    scale is native 0.1 m/px (s = 1), exactly as for every other row;
  * the tile is resized to L = int(2048 s + 0.5) and the crop is the quadrant at offset 0 or L-1024
    on each axis; horizontal flip (released p = 0.5) is applied AFTER the crop.
Each of the 4 quadrant crops covers at least its own original quadrant whatever s is drawn (proved
by `coverage_check`), so EVERY TRAINING PIXEL IS SEEN AT LEAST ONCE AND AT MOST TWICE PER EPOCH -- the
property LACE's whole-tile training has, which uniform random crops do not (they starve the tile
frame). One epoch = 792 tiles x 4 = 3,168 crops = 198 iterations at batch 16, and detectron2's
TrainingSampler consumes exactly one permutation of the 3,168 per 198 iterations.

SCHEDULE (amendment 2026-10-15, before any number): convergence-driven (plateau.py) -- LR held
at the released base after warm-up, x0.1 on a val plateau (twice, as in the released two-step
schedules), early stop on the plateau after the second decay; min 4, max 20 epochs. The warm-ups
keep their released FRACTION of the released 1x length (12 epochs = 2,376 iterations here): LR
warm-up 1k/90k, BoxInst pairwise and BoxTeacher pseudo-mask warm-ups 10k/90k.
"""
from __future__ import annotations

TILE = 2048
CROP = 1024
SUB_STRIDE = 512                     # test-time subtile grid (= DT2 / Box2Mask rows)
# test-time geometries of B1/B2 (amendment 2026-10-29, TIERB.md section 000): the 3x3 1024 subtiles
# only = the training canvas, as the DT2/Box2Mask rows; was ("whole", "sub") with a val choice
GEOMETRIES = ("sub",)

# released multi-scale training set, relative to the released test short side (both configs)
RELEASED_MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
RELEASED_MIN_SIZE_TEST = 800
SCALES = tuple(m / RELEASED_MIN_SIZE_TEST for m in RELEASED_MIN_SIZE_TRAIN)

N_TRAIN_TILES = 792
CROPS_PER_TILE = 4
IMS_PER_BATCH = 16
EPOCHS = 12                          # released 1x length: the reference for warm-up fractions only
MAX_EPOCHS = 20                      # plateau.MAX_EPOCHS (budget cap)
RELEASED_3X = {"max_iter": 270000, "steps": (210000, 250000), "warmup": 1000,     # the YAMLs we load
               "pairwise_warmup": 10000, "boxteacher_mask_warmup": 10000}
RELEASED_1X = {"max_iter": 90000, "steps": (60000, 80000), "warmup": 1000,       # the fractions we use
               "pairwise_warmup": 10000, "boxteacher_mask_warmup": 10000}


def scaled_size(s, tile=TILE):
    """detectron2's own rounding (ResizeShortestEdge: int(x + 0.5))."""
    return int(tile * s + 0.5)


def quadrant_offset(L, q, crop=CROP):
    """(oy, ox) of quadrant q in {0,1,2,3} (row-major) on an L x L image."""
    assert L >= crop and q in (0, 1, 2, 3)
    qy, qx = divmod(q, 2)
    return (0 if qy == 0 else L - crop), (0 if qx == 0 else L - crop)


def original_interval(L, off, crop=CROP, tile=TILE):
    """Original-tile pixel interval [a, b) that a crop at `off` on the L-scaled tile covers."""
    s = L / tile
    return off / s, (off + crop) / s


def coverage_check(scales=SCALES, tile=TILE, crop=CROP):
    """Assert: for every scale, quadrant 0/1 along an axis covers [0, tile/2] / [tile/2, tile]
    of the ORIGINAL tile. Hence any per-crop draw of scales covers the whole tile, every epoch."""
    for s in scales:
        L = scaled_size(s, tile)
        a0, b0 = original_interval(L, 0, crop, tile)
        a1, b1 = original_interval(L, L - crop, crop, tile)
        assert a0 == 0 and b0 >= tile / 2, (s, a0, b0)
        assert a1 <= tile / 2 and abs(b1 - tile) < 1e-9, (s, a1, b1)
    return True


def teacher_size(L, crop=CROP, tile=TILE):
    """BoxTeacher's teacher sees the student's image re-sized to the TEST scale (released:
    `resize_images` to MIN/MAX_SIZE_TEST, i.e. x 800/short_side = x 1/s). The crop's content at
    native scale is crop * tile / L pixels; same int(x + 0.5) rounding as `resize_images`."""
    return int(crop * tile / L + 0.5)


def schedule(epochs=EPOCHS, n_tiles=N_TRAIN_TILES, batch=IMS_PER_BATCH, ref=RELEASED_1X):
    crops = n_tiles * CROPS_PER_TILE
    assert crops % batch == 0
    ipe = crops // batch
    f = ipe * epochs / ref["max_iter"]
    out = {"iters_per_epoch": ipe, "max_iter": ipe * MAX_EPOCHS, "max_epochs": MAX_EPOCHS,
           "steps": (), "schedule": "plateau (plateau.py)",
           "warmup_iters": int(round(ref["warmup"] * f)),
           "pairwise_warmup_iters": int(round(ref["pairwise_warmup"] * f)),
           "boxteacher_mask_warmup_iters": int(round(ref["boxteacher_mask_warmup"] * f)),
           "fraction_of_released_1x": f}
    return out


def subtile_origins(tile=TILE, sub=CROP, stride=SUB_STRIDE):
    """= native_raster/stitch_res.subtile_origins (3x3 at stride 512)."""
    ys = list(range(0, tile - sub + 1, stride))
    return [(oy, ox) for oy in ys for ox in ys]


# ------------------------------------------------------------------ MAL (B3)
# MAL drops training boxes with area <= --min_obj_size (released 2048 px^2). On COCO train2017 that
# removes 42.18% of the boxes with w, h > 2 (859,094 -> 496,706; computed from
# instances_train2017.json, recorded in gates/coco_min_obj_size.json). On our 792 training tiles a
# literal 2048 would remove 61.49% (median crown 36 px across). The threshold is therefore
# re-based to the SAME FRACTION: the 42.18th percentile of our training box areas.
MAL_COCO_EXCLUDED_FRACTION = 0.4218
MAL_RELEASED_MIN_OBJ_SIZE = 2048


def mal_min_obj_size(train_boxes_xyxy, frac=MAL_COCO_EXCLUDED_FRACTION):
    """Area threshold t such that MAL's own filter (keep iff area > t and w > 2 and h > 2) removes
    `frac` of our valid training boxes, as 2048 removes `frac` of COCO's."""
    import math
    areas = sorted((b[2] - b[0]) * (b[3] - b[1]) for b in train_boxes_xyxy
                   if (b[2] - b[0]) > 2 and (b[3] - b[1]) > 2)
    k = int(math.floor(frac * len(areas)))
    return float(areas[k - 1]), {"n_valid": len(areas), "n_excluded": k,
                                 "excluded_fraction": k / len(areas)}
