"""THE frozen scorer for the OAM-TCD 439 table. One protocol, every row.

Replaces the per-row scoring scattered across evaluate.py / score_detectree2.py /
phase4_lib_tcd.py for TABLE 1 ONLY. Nothing here mutates those modules, so every published
number stays reproducible; this is a second, standard gate beside them.

PROTOCOL (frozen -- see PROTOCOL_439.md)
    scorer      pycocotools COCOeval, iouType segm then bbox
    unit        439 whole 2048^2 tiles
    raster      masks at `res` (default 512; 2048 = native imagery resolution)
    iouThrs     np.linspace(0.5, 0.95, 10)   <- COCO convention. evaluate.py uses
                np.arange(0.5,0.96,0.05), whose 0.60/0.75/0.85 sit one ulp high and reject
                IoUs landing exactly there (worth +0.0008 AP50:95; see
                ablation/results/cocoeval_parity.json).
    recThrs     101-point (COCO default)
    maxDets     600  (densest test tile holds 450 GT instances)
    areaRng     'all' = [0, 1e10] only. Per-instance areas ARE persisted, so a
                size-stratified cut can be made later without re-scoring.
    score floor 0.05, applied uniformly to every row before scoring
    GT          all 439 image ids always present -- a method that predicted nothing on a
                tile is charged zero recall there, never a shrunken denominator.

THREE ARMS, always reported together:
    canopy_fp              GT carries no canopy at all -> canopy predictions are plain FPs.
    canopy_neutral_crowd   canopy polygons as iscrowd=1 GT. COCO's crowd rule (intersection
                           over DETECTION area > t) equals our >50%-in-canopy rule EXACTLY at
                           t=0.50, and is strictly stricter above it.
    canopy_neutral_legacy  our own fixed-50%-per-prediction rule, via evaluate.match_tile.
                           Reported so the crowd-vs-legacy difference is MEASURED rather than
                           assumed -- the two agree at AP50 and must diverge at AP50:95.

Everything is RLE end to end; no dense (N, res*res) array is ever built. That is what makes
res=2048 tractable here while phase4_lib_tcd.eval_selfmask (which accumulates dense masks for
all 439 tiles, ~49 GB at 512) cannot go above 512 in a 64 GB container.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.score_coco \
        --preds boxinst_commonality_tcd_04/detectree2_baseline/preds_dt2_s0_fullcov.json \
        --res 512 --out results.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from boxinst_commonality_tcd_04 import evaluate as E

IOU_THRS = np.linspace(0.5, 0.95, 10)      # COCO convention; NOT evaluate.IOU_50_95
REC_THRS = np.linspace(0.0, 1.00, 101)
MAX_DETS = 600
SCORE_FLOOR = 0.05
CAT = [{"id": 1, "name": "tree"}]


# ---------------------------------------------------------------- RLE helpers
def _as_bytes(rle):
    """COCO RLE with `counts` as bytes (pycocotools' C layer requires bytes, JSON gives str)."""
    c = rle["counts"]
    return {"size": list(rle["size"]),
            "counts": c.encode("ascii") if isinstance(c, str) else c}


def _polys_to_rle(polys, res):
    """Polygons (2048-space coords) -> COCO RLEs at `res`. Rasterised DIRECTLY at res via
    evaluate.raster -- never drawn at one resolution and resampled to another."""
    if not polys:
        return []
    masks = E.raster(polys, res=res, scale=2048.0 / res)
    return [maskUtils.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]


def _union(rles, res):
    """Single RLE covering the union (canopy is one region for the >50% test)."""
    if not rles:
        return maskUtils.encode(np.asfortranarray(np.zeros((res, res), np.uint8)))
    return maskUtils.merge(rles)


def _areas(rles):
    """Areas, ONE RLE PER CALL -- do not "optimise" into a single list call.

    pycocotools' `_mask.area` builds `np.array((n,), dtype=np.uint8)` internally, so under
    numpy>=2 (2.2.6 here) it raises OverflowError for any list of 256 or more RLEs. Verified:
    255 OK, 256 raises. Single-RLE calls are unaffected. `maskUtils.iou` does not share the
    bug -- COCOeval calls it with 968-detection tiles here without complaint."""
    return np.asarray([float(maskUtils.area(r)) for r in rles], np.float64)


def _frac_inside(pred_rles, region_rle):
    """For each prediction: fraction of ITS OWN area inside `region_rle`, computed on RLE."""
    if not pred_rles:
        return np.zeros(0, bool)
    areas = _areas(pred_rles)
    inter = np.asarray([float(maskUtils.area(maskUtils.merge([p, region_rle], intersect=1)))
                        for p in pred_rles], np.float64)
    return np.divide(inter, areas, out=np.zeros_like(inter), where=areas > 0)


# ---------------------------------------------------------------- COCO builders
def build_gt(gt, res, with_canopy):
    """-> (coco_dict, tid_to_image_id, per-tile tree RLEs, per-tile canopy-union RLE).

    ALL 439 tiles get an image entry regardless of predictions. Canopy is emitted as
    iscrowd=1 annotations of the SAME category when `with_canopy`, which is what makes
    COCOeval ignore detections that fall inside it."""
    images, anns, tree_rles, canopy_union = [], [], {}, {}
    tids = sorted(gt)
    tid2img = {t: i + 1 for i, t in enumerate(tids)}
    aid = 1
    for t in tids:
        images.append({"id": tid2img[t], "file_name": t, "width": res, "height": res})
        trs = _polys_to_rle(gt[t]["trees"], res)
        tree_rles[t] = trs
        for r in trs:
            anns.append({"id": aid, "image_id": tid2img[t], "category_id": 1,
                         "segmentation": r, "iscrowd": 0,
                         "area": float(maskUtils.area(r)),
                         "bbox": [float(v) for v in maskUtils.toBbox(r)]})
            aid += 1
        can = _polys_to_rle(gt[t]["canopy"], res)
        canopy_union[t] = _union(can, res)
        if with_canopy:
            for r in can:
                anns.append({"id": aid, "image_id": tid2img[t], "category_id": 1,
                             "segmentation": r, "iscrowd": 1,
                             "area": float(maskUtils.area(r)),
                             "bbox": [float(v) for v in maskUtils.toBbox(r)]})
                aid += 1
    return ({"images": images, "annotations": anns, "categories": CAT},
            tid2img, tree_rles, canopy_union)


def build_dt(preds, tid2img, res, score_floor=SCORE_FLOOR):
    """Our preds/*.json schema -> COCO detections + the per-tile arrays the legacy arm needs.

    Fails loudly if a stored mask is not at `res`: comparing a 512-stored mask against a
    2048 GT would be an upsample of information that is not there, which is exactly the
    trap this protocol exists to avoid."""
    dets, per_tile = [], {}
    scale = 2048.0 / res
    for t, rec in preds.items():
        if t not in tid2img:
            continue
        rles = [_as_bytes(r) for r in rec["masks_rle"]]
        for r in rles:
            if tuple(r["size"]) != (res, res):
                raise ValueError(f"{t}: stored mask {r['size']} != scoring res {res}. "
                                 f"Re-generate predictions at {res}; do NOT resample.")
        sc = np.asarray(rec["scores"], np.float32)
        bx = np.asarray(rec["boxes_2048"], np.float32).reshape(-1, 4) / scale
        keep = np.flatnonzero(sc >= score_floor)
        rles = [rles[i] for i in keep]; sc = sc[keep]; bx = bx[keep]
        per_tile[t] = {"rles": rles, "scores": sc, "boxes": bx}
        for i, r in enumerate(rles):
            x0, y0, x1, y1 = bx[i]
            dets.append({"image_id": tid2img[t], "category_id": 1, "segmentation": r,
                         "score": float(sc[i]),
                         "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                         "area": float(maskUtils.area(r))})
    return dets, per_tile


# ---------------------------------------------------------------- evaluation
def _cocoeval(gt_dict, dets, iou_type, max_dets):
    """COCOeval with our frozen params. AP is read from eval['precision'] directly rather
    than via summarize(), so maxDets is a single value and areaRng is 'all' only."""
    cg = COCO(); cg.dataset = gt_dict; cg.createIndex()
    if not dets:
        return {"AP50": 0.0, "AP50_95": 0.0, "AP75": 0.0, "n_det": 0}
    cd = cg.loadRes([dict(d) for d in dets])
    ev = COCOeval(cg, cd, iouType=iou_type)
    ev.params.iouThrs = IOU_THRS
    ev.params.recThrs = REC_THRS
    ev.params.maxDets = [max_dets]
    ev.params.areaRng = [[0.0, 1e10]]
    ev.params.areaRngLbl = ["all"]
    ev.evaluate(); ev.accumulate()
    p = ev.eval["precision"]                       # [T, R, K, A, M]

    def _ap(ti):
        s = p[ti, :, :, 0, 0] if ti is not None else p[:, :, :, 0, 0]
        s = s[s > -1]
        return float(s.mean()) if s.size else float("nan")
    i50 = int(np.argmin(np.abs(IOU_THRS - 0.50)))
    i75 = int(np.argmin(np.abs(IOU_THRS - 0.75)))
    return {"AP50": round(_ap(i50), 4), "AP75": round(_ap(i75), 4),
            "AP50_95": round(_ap(None), 4), "n_det": len(dets)}


def _legacy_arm(per_tile, tree_rles, canopy_union, tids, max_dets):
    """Our own rule: a prediction with >50% of its area in unlabelled canopy that fails to
    match is DROPPED, not counted FP -- fixed at 50% for every IoU threshold. Routed through
    evaluate.match_tile/_greedy_ap (the repo's single matcher) with RLE-computed IoU, so the
    matching rule is identical to every other AP in the project."""
    Ious, Scores, Ign, n_gt = [], [], [], 0
    for t in tids:
        g = tree_rles[t]; n_gt += len(g)
        d = per_tile.get(t)
        if not d or not len(d["scores"]):
            continue
        rles, sc = d["rles"], d["scores"]
        if len(sc) > max_dets:                     # same cap as the COCO arms
            k = np.argsort(-sc, kind="mergesort")[:max_dets]
            rles = [rles[i] for i in sorted(k)]; sc = sc[sorted(k)]
        iou = (np.asarray(maskUtils.iou(rles, g, [0] * len(g)), np.float64).reshape(len(rles), len(g))
               if g else np.zeros((len(rles), 0)))
        Ious.append(iou); Scores.append(sc)
        Ign.append(_frac_inside(rles, canopy_union[t]) > 0.5)
    out = {t: round(E._greedy_ap(Ious, Scores, Ign, n_gt, t), 4)
           for t in (0.5, 0.75)}
    ap5095 = float(np.nanmean([E._greedy_ap(Ious, Scores, Ign, n_gt, t) for t in IOU_THRS]))
    return {"AP50": out[0.5], "AP75": out[0.75], "AP50_95": round(ap5095, 4),
            "n_gt": n_gt}


def score_many(preds_paths, gt_path, res=512, max_dets=MAX_DETS, score_floor=SCORE_FLOOR):
    """Score several rows against ONE GT build. Rasterising the 439 tiles costs ~47 s at 512²
    and ~11 min at 2048², so it is done once here and shared across every row -- which also
    guarantees each row is scored against a byte-identical ground truth."""
    gt = json.load(open(gt_path))
    tids = sorted(gt)
    print(f"[score_coco] rasterising GT at {res}^2 (once, shared) ...", flush=True)
    gt_fp, tid2img, tree_rles, canopy_union = build_gt(gt, res, with_canopy=False)
    gt_cr, _, _, _ = build_gt(gt, res, with_canopy=True)
    ctx = (gt_fp, gt_cr, tid2img, tree_rles, canopy_union, tids)
    return [_score_one(p, ctx, res, max_dets, score_floor) for p in preds_paths]


def score(preds_path, gt_path, res=512, max_dets=MAX_DETS, score_floor=SCORE_FLOOR):
    return score_many([preds_path], gt_path, res, max_dets, score_floor)[0]


def _score_one(preds_path, ctx, res, max_dets, score_floor):
    gt_fp, gt_cr, tid2img, tree_rles, canopy_union, tids = ctx
    P = json.load(open(preds_path))
    preds = P.get("preds", P)
    missing = [t for t in tids if t not in preds]
    if missing:
        print(f"[score_coco] {os.path.basename(preds_path)}: {len(missing)}/{len(tids)} GT "
              f"tiles have NO predictions -> zero-prediction (recall penalised)", flush=True)
    dets, per_tile = build_dt(preds, tid2img, res, score_floor)
    print(f"[score_coco] {os.path.basename(preds_path)}: {len(dets)} dets >= {score_floor}",
          flush=True)

    res_out = {
        "preds_file": os.path.relpath(preds_path),
        "model": P.get("meta", {}).get("model", os.path.basename(preds_path)),
        "protocol": {"scorer": "pycocotools COCOeval", "res": res, "max_dets": max_dets,
                     "score_floor": score_floor, "n_tiles": len(tids),
                     "iou_thrs": "linspace(0.5,0.95,10)", "rec_thrs": 101,
                     "area_rng": "all only"},
        "n_gt_trees": sum(len(v) for v in tree_rles.values()),
        "canopy_fp": {"mask": _cocoeval(gt_fp, dets, "segm", max_dets)},
        "canopy_neutral_crowd": {"mask": _cocoeval(gt_cr, dets, "segm", max_dets),
                                 "box": _cocoeval(gt_cr, dets, "bbox", max_dets)},
        "canopy_neutral_legacy": {"mask": _legacy_arm(per_tile, tree_rles, canopy_union,
                                                      tids, max_dets)},
        "areas": {t: _areas(per_tile[t]["rles"]).tolist()
                  for t in per_tile if per_tile[t]["rles"]},
    }
    return res_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", default="boxinst_commonality_tcd_04/test_gt.json")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--max_dets", type=int, default=MAX_DETS)
    ap.add_argument("--score_floor", type=float, default=SCORE_FLOOR)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = score(a.preds, a.gt, a.res, a.max_dets, a.score_floor)
    slim = {k: v for k, v in r.items() if k != "areas"}
    print(json.dumps(slim, indent=2), flush=True)
    if a.out:
        json.dump(r, open(a.out, "w"), indent=2)
        print(f"\n-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
