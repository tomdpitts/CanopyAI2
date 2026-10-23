"""Dependency-free copies of `box_supervised_baselines/gtbox_lib.py` (the tab:gtbox core), for the
MAL container, which cannot import `evaluate.py`. Verbatim logic; `raster` is `evaluate.raster`
(via score_lite). `parity()` checks every function against gtbox_lib on real test tiles and must
PASS before any Modal run (`python -m ...tierb.gtbox_pure`).

Used in-container only for MAL checkpoint selection (val mean IoU). Every reported GT-box number
is recomputed on CPU with gtbox_lib itself (tierb_modal.py::score).
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as maskUtils

from tierb.score_lite import raster

RES = 512
MIN_GT_PX = 4


def gt_boxes_2048(polys):
    if not polys:
        return np.zeros((0, 4), np.float32)
    out = []
    for p in polys:
        v = np.asarray(p, np.float32).reshape(-1, 2)
        out.append([*v.min(0), *v.max(0)])
    return np.asarray(out, np.float32).reshape(-1, 4)


def gt_rles_and_valid(polys, res=RES):
    if not polys:
        return [], np.zeros(0, bool)
    valid = np.asarray([np.asarray(raster([p], res=RES, scale=2048.0 / RES)[0]).sum()
                        >= MIN_GT_PX for p in polys], bool)
    rles = [maskUtils.encode(np.asfortranarray(
        np.asarray(raster([p], res=res, scale=2048.0 / res)[0]).astype(np.uint8)))
        for p in polys]
    return rles, valid


def per_crown_iou_rle(pred_rles, gt_rles):
    assert len(pred_rles) == len(gt_rles)
    out = np.zeros(len(gt_rles), np.float64)
    for i, (p, g) in enumerate(zip(pred_rles, gt_rles)):
        inter = float(maskUtils.area(maskUtils.merge([p, g], intersect=1)))
        union = float(maskUtils.area(p)) + float(maskUtils.area(g)) - inter
        out[i] = inter / union if union > 0 else 0.0
    return out


def parity(n_tiles=6):
    """gtbox_pure == gtbox_lib on the first n non-empty test tiles (boxes, RLEs, validity, IoU)."""
    import json
    import os
    from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as G
    pkg = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    gt = json.load(open(os.path.join(pkg, "test_gt.json")))
    tids = [t for t in sorted(gt) if gt[t]["trees"]][:n_tiles]
    n = 0
    for t in tids:
        polys = gt[t]["trees"]
        assert np.array_equal(gt_boxes_2048(polys), G.gt_boxes_2048(polys))
        a, va = gt_rles_and_valid(polys, res=2048)
        b, vb = G.gt_rles_and_valid(polys, res=2048)
        assert np.array_equal(va, vb) and all(x["counts"] == y["counts"] for x, y in zip(a, b))
        rng = np.random.default_rng(0)                     # perturbed predictions vs GT
        pred = [maskUtils.encode(np.asfortranarray(np.roll(maskUtils.decode(r), int(rng.integers(-3, 4)), 0)))
                for r in a]
        assert np.array_equal(per_crown_iou_rle(pred, a), G.per_crown_iou_rle(pred, b))
        n += len(polys)
    return {"tiles": tids, "crowns": n, "pass": True}


if __name__ == "__main__":
    import json
    import os
    import time
    r = parity()
    r["date"] = time.strftime("%Y-%m-%d %H:%M")
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gates", "gtbox_pure_parity.json")
    assert not os.path.exists(p), p
    json.dump(r, open(p, "w"), indent=1)
    print("GTBOX_PURE PARITY PASS", r)
