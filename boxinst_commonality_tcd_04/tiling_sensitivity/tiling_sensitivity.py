"""Does per-tile AP averaging get MORE generous as the tiles get smaller?

`aggregation_sensitivity/` measured pooled vs per-tile averaging at OUR tile granularity
(whole 2048 px tiles, ~59 GT crowns each) and found box AP50-95 essentially unmoved:
0.2495 -> 0.2492. That result was then used to argue aggregation cannot explain the gap to
SelvaBox's 44.29. **That argument had a hole**: SelvaMask does not average over 2048 px
tiles, it averages over 1024 px subtiles at 0.5 overlap. Tile SIZE and tile AGGREGATION are
not separable -- per-tile averaging is a different estimator at every granularity.

The mechanism it tests: per-tile AP normalises precision WITHIN a tile. A subtile holding
1 GT crown and 20 false positives scores AP = 1.0 as long as the true positive outranks
its own tile's FPs -- those 20 FPs are never weighed against any other tile's true
positives. Pooled AP charges every one of them. So the fewer GT per tile, the more
per-tile AP degenerates from "average precision" toward "did the top-ranked box in this
tile hit", and the more forgiving of false positives it becomes. Our detector emits ~360
boxes per 2048 tile against ~59 GT; that ratio is what pooled AP punishes and per-tile AP
at fine granularity hides.

Sweep: score the identical saved predictions under subtile grids from 2048 (= no
subtiling, reproduces the aggregation study) down to 256 px, at 0.5 overlap like theirs,
plus a no-overlap arm to separate the overlap term from the size term.

Assignment is by BOX CENTRE (a GT crown or a prediction belongs to the subtile its centre
falls in), so each object is scored exactly once per grid and IoU is computed on the
original uncropped geometry. This is deliberately NOT their pipeline -- geodataset clips
annotations to the tile and filters by an area ratio, which would additionally CHANGE the
IoUs near borders. Centre assignment isolates the aggregation-granularity term alone,
which is what we want to measure; it is a lower bound on the full tiling effect.

Empty subtiles (no GT) are dropped, matching per-tile averaging's NaN-skip -- so their
false positives go uncharged, exactly as in the coarse study.

SENSITIVITY ONLY. Published numbers are pooled AP over whole 2048 tiles.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.tiling_sensitivity.tiling_sensitivity
"""
import argparse
import json
import os

import numpy as np

from boxinst_commonality_tcd_04 import evaluate as E
from boxinst_commonality_tcd_04.aggregation_sensitivity import aggregation_sensitivity as A
from dapt.eval import iou_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results_tiling.json")
TILE = 2048
SIZES = (2048, 1024, 768, 512, 256)


def subtile_origins(size, overlap):
    """Top-left corners of a `size` grid over a 2048 tile at the given overlap fraction."""
    if size >= TILE:
        return [(0, 0)]
    step = int(round(size * (1 - overlap)))
    xs = list(range(0, TILE - size + 1, step))
    if xs[-1] != TILE - size:
        xs.append(TILE - size)
    return [(x, y) for y in xs for x in xs]


def _centres(b):
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1) if len(b) \
        else np.zeros((0, 2))


def split(tile, size, overlap):
    """One 2048 tile -> list of per-subtile (pred idx, gt idx) index arrays.

    Centre-in-subtile assignment. With overlap > 0 the grids overlap, but an object's
    centre lands in several subtiles only where the strips cross; assigning it to every
    subtile that contains its centre is what an overlapped tiler does, and it is why the
    overlap arm is reported separately from the no-overlap one.
    """
    pc, gc = _centres(tile["pbox"]), _centres(tile["gbox"])
    out = []
    for (x0, y0) in subtile_origins(size, overlap):
        x1, y1 = x0 + size, y0 + size
        pi = np.where((pc[:, 0] >= x0) & (pc[:, 0] < x1) &
                      (pc[:, 1] >= y0) & (pc[:, 1] < y1))[0] if len(pc) else np.zeros(0, int)
        gi = np.where((gc[:, 0] >= x0) & (gc[:, 0] < x1) &
                      (gc[:, 1] >= y0) & (gc[:, 1] < y1))[0] if len(gc) else np.zeros(0, int)
        out.append((pi, gi))
    return out


def load(preds_path, gt):
    """Per 2048 tile: predicted boxes (2048 coords) + scores + canopy-ignore, GT boxes.

    Box-ignore is the recorded rule (canopy mean over the box crop > 0.5), computed on the
    512 raster exactly as `phase4_lib_tcd.eval_4p_selfmask` does, then carried per box.
    """
    preds = json.load(open(preds_path))["preds"]
    T = []
    for k, tid in enumerate(sorted(gt)):
        p = preds[tid]
        pbox = np.array(p["boxes_2048"], np.float64).reshape(-1, 4)
        sc = np.array(p["scores"], np.float64)
        gbox = (np.array([[*(np.asarray(t).reshape(-1, 2).min(0)),
                           *(np.asarray(t).reshape(-1, 2).max(0))]
                          for t in gt[tid]["trees"]], np.float64).reshape(-1, 4)
                if gt[tid]["trees"] else np.zeros((0, 4), np.float64))
        can = np.array(E.raster(gt[tid]["canopy"]))
        can = can.any(0) if len(can) else np.zeros((E.RES, E.RES), bool)
        ign = np.zeros(len(pbox), bool)
        for i, (x0, y0, x1, y1) in enumerate(pbox / E.SCALE):
            sub = can[slice(max(0, int(y0)), int(np.ceil(y1))),
                      slice(max(0, int(x0)), int(np.ceil(x1)))]
            ign[i] = sub.size > 0 and sub.mean() > 0.5
        T.append({"pbox": pbox, "sc": sc, "ign": ign, "gbox": gbox})
        del can
        if (k + 1) % 100 == 0 or k + 1 == len(gt):
            print(f"    {k + 1}/{len(gt)}", flush=True)
    return T


def _ap_one(iou, sc, ign, n_gt, thr):
    """Greedy match + 101-pt AP for a single tile, same core as evaluate._greedy_ap."""
    if n_gt == 0:
        return float("nan")
    if len(sc) == 0:
        return 0.0
    o = np.argsort(-sc)
    iou, sc, ign = iou[o], sc[o], ign[o]
    matched = np.zeros(iou.shape[1], bool)
    tp = np.zeros(len(sc), bool)
    keep = np.ones(len(sc), bool)
    for i in range(len(sc)):
        j = int(np.argmax(iou[i])) if iou.shape[1] else -1
        if j >= 0 and iou[i, j] >= thr and not matched[j]:
            matched[j] = True
            tp[i] = True
        elif ign[i]:
            keep[i] = False
    return A._ap101(sc[keep], tp[keep], n_gt)


def score(T, size, overlap):
    """Per-subtile AP averaged over subtiles that hold at least one GT crown."""
    per_iou = []
    gt_per_tile = []
    for thr in E.IOU_50_95:
        aps = []
        for t in T:
            for pi, gi in split(t, size, overlap):
                if len(gi) == 0:
                    continue                       # NaN-skip, as per-tile averaging does
                pb, gb = t["pbox"][pi], t["gbox"][gi]
                aps.append(_ap_one(iou_matrix(pb, gb), t["sc"][pi], t["ign"][pi],
                                   len(gi), float(thr)))
                if thr == E.IOU_50_95[0]:
                    gt_per_tile.append(len(gi))
        per_iou.append(float(np.nanmean(aps)))
    return {"AP50": round(per_iou[0], 4),
            "AP50_95": round(float(np.nanmean(per_iou)), 4),
            "per_iou": [round(x, 4) for x in per_iou],
            "n_scored_subtiles": len(gt_per_tile),
            "mean_gt_per_subtile": round(float(np.mean(gt_per_tile)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    gt = json.load(open(A.GT))
    runs = {}
    for label, pp, _ in A.RUNS:
        if label not in ("ours_s0", "detectree2_s0"):
            continue                               # seed band adds nothing to a protocol sweep
        print(f"[{label}]", flush=True)
        T = load(pp, gt)
        cells = {}
        for size in SIZES:
            for ov in (0.5, 0.0):
                if size >= TILE and ov != 0.0:
                    continue                       # one tile, overlap undefined
                r = score(T, size, ov)
                cells[f"{size}|ov{ov}"] = r
                print(f"  {size:>5}px ov{ov}: AP50 {r['AP50']:.4f}  AP50-95 {r['AP50_95']:.4f}"
                      f"  ({r['n_scored_subtiles']} subtiles, "
                      f"{r['mean_gt_per_subtile']} GT each)", flush=True)
        runs[label] = cells
        del T

    json.dump({"label": "per-tile AP vs tiling granularity, OAM-TCD 439",
               "note": ("SENSITIVITY ONLY. Per-tile AP averaging at varying subtile size, "
                        "centre assignment, empty subtiles NaN-skipped. Isolates the "
                        "aggregation-granularity term; does NOT clip annotations to subtiles "
                        "as geodataset does, so it is a lower bound on the full tiling "
                        "effect. Published numbers are pooled AP over whole 2048 tiles."),
               "metric": "box AP (canopy-ignore on, uncapped maxDets)",
               "per_run": runs},
              open(a.out, "w"), indent=2)
    print(f"\n-> {os.path.relpath(a.out, os.path.dirname(PKG))}", flush=True)


if __name__ == "__main__":
    main()
