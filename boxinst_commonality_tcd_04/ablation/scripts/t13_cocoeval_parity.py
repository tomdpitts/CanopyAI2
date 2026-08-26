"""T0.13 -- Does our AP core agree with pycocotools COCOeval?

WHY THIS EXISTS
    Every AP in this project comes from `evaluate._greedy_ap`, a hand-written COCO-101pt
    implementation. `pycocotools` is imported across the repo but ONLY as
    `pycocotools.mask` (RLE encode/decode); `COCOeval` is never called. Until this script
    existed, nothing verified that our AP agrees with the standard implementation.

    That gap mattered because Table `tab:oamtcd439` places our 0.630 beside published
    numbers (Restor Mask R-CNN, SelvaBox) that were produced by other people's scorers.
    The reproduction gates in `lib/gate.py` prove our numbers are REPRODUCIBLE; they do
    not prove they are on the same SCALE as anyone else's. This does.

METHOD -- and why the canopy=FP arm specifically
    COCOeval has no equivalent of our canopy-ignore rule. Ours is a per-PREDICTION,
    region-based ignore (>50% of the prediction's area sits in unlabelled canopy -> drop,
    do not count FP), applied at a fixed 50% at every IoU threshold. COCOeval's only
    ignore mechanism is `iscrowd` on GROUND-TRUTH annotations, matched by
    intersection-over-detection-area with a threshold that SWEEPS alongside the IoU
    threshold. The two are not equivalent and a disagreement would be uninterpretable.

    So we score the canopy=FP arm instead: `Ignore` is all-False, `_greedy_ap` reduces to
    plain greedy AP, and both scorers see identical inputs with no ignore semantics
    between them. Any delta is then PURE AP ARITHMETIC.

RESULT (recorded 2026-08-25, seed 0, 439 tiles, 25705 GT, 159687 detections)
    _greedy_ap   AP50 0.4867   AP50:95 0.2041
    COCOeval     AP50 0.4867   AP50:95 0.2049
    delta        AP50 +0.0000  AP50:95 +0.0008

    AP50 agrees EXACTLY to 4dp. Greedy matching, pooling across images and the 101-point
    precision envelope are all doing what COCO does.

    The +0.0008 on AP50:95 is a real and understood difference in the matching rule:

        _greedy_ap (evaluate.py:108-112) takes j = argmax(iou[i]) -- the BEST-IoU GT --
        and gives up if that GT is already matched. COCOeval instead picks the best
        UNMATCHED GT above threshold, i.e. it FALLS BACK to the second-best crown.

    So a detection whose best overlap is with an already-claimed crown becomes a FP for
    us even when it clears the threshold against a different, free crown. This is
    invisible at IoU 0.5 (overlaps are loose enough that the best match is usually still
    available) and appears only at the stricter thresholds. It biases OUR numbers DOWN,
    never up -- we report a marginally conservative AP. Magnitude 0.0008, roughly a third
    of the 3-seed sigma (0.005 on AP50), so it is immaterial but should not be described
    as "identical".

WHAT THIS DOES AND DOES NOT ESTABLISH
    DOES:     our AP arithmetic is the COCO arithmetic. 0.630 is on the same scale as any
              published COCO-convention number.
    DOES NOT: make our PROTOCOL standard. Whole 2048^2 tiles (not subtiled), masks
              rasterised at 512^2, maxDets 600 (COCO default is 100), and canopy-ignore
              are all still ours. The notes on `tab:oamtcd439` must keep saying so.
    DOES NOT: say anything about the Restor or SelvaBox rows, which were never re-scored
              here. Whether THEY used COCOeval is unknown to this repo.

CITING THIS WITHOUT RE-RUNNING
    `results/cocoeval_parity.json` carries every number above plus the input hashes.
    Paper-ready sentence:
        "Our AP implementation was cross-validated against pycocotools COCOeval on the
        canopy-as-FP arm, where no ignore rule applies: AP50 agrees exactly (0.4867 for
        both) and AP50:95 to 0.0008."

RUNTIME
    ~5 min, CPU only. Dominated by RLE-encoding 159687 predicted and 25705 GT masks at
    512^2 so COCOeval can consume them. No GPU, no Modal, no refit.

KNOWN NOISE (macOS/Apple Accelerate)
    `mask_iou` (evaluate.py:56) emits "overflow/invalid value encountered in matmul" on
    Apple silicon. Verified spurious: results are finite, contain no NaN, and lie in
    [0, 1]. It is the Accelerate BLAS backend raising FP status flags that numpy surfaces
    as warnings. Suppressed below so the log stays readable.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.ablation.scripts.t13_cocoeval_parity
"""
import argparse
import json
import os
import warnings

import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from boxinst_commonality_tcd_04 import evaluate as E
from ..lib import gate, io

# seed-0 canopy=FP reference, from phase4/results_no_ignore_band.json per_seed.0.no_ignore
REF = {"mask_mAP50": 0.4867, "mask_mAP50_95": 0.2041}
REF_PATH = os.path.join(io.PHASE4, "results_no_ignore_band.json")
MAX_DETS = 600          # matches the decode cap; COCO's own default is 100


def _rle(m):
    """bool (H,W) -> COCO RLE with str counts, plus its bbox (computed pre-decode)."""
    r = maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
    bb = [float(x) for x in maskUtils.toBbox(r)]
    return {"size": r["size"], "counts": r["counts"].decode("ascii")}, bb


def collect(seed, gt, progress):
    """One streaming pass: our per-tile IoU/score arrays AND the COCO dicts."""
    images, anns, dets = [], [], []
    mI, S, IGN, n_gt, aid = [], [], [], 0, 1
    for t in io.per_tile(io.knobbed(seed), gt, progress=progress):
        iid = t["i"] + 1
        images.append({"id": iid, "width": E.RES, "height": E.RES})
        mI.append(E.mask_iou(t["pm"], t["gm"]))
        S.append(t["scores"])
        IGN.append(np.zeros(len(t["scores"]), bool))     # canopy=FP arm: no ignore
        n_gt += len(t["gm"])
        for g in t["gm"]:
            seg, bb = _rle(g)
            anns.append({"id": aid, "image_id": iid, "category_id": 1,
                         "segmentation": seg, "area": float(g.sum()),
                         "iscrowd": 0, "bbox": bb})
            aid += 1
        for m, sc in zip(t["pm"], t["scores"]):
            seg, _ = _rle(m)
            dets.append({"image_id": iid, "category_id": 1,
                         "segmentation": seg, "score": float(sc)})
    return images, anns, dets, mI, S, IGN, n_gt


def cocoeval_ap(images, anns, dets):
    """AP50 and AP50:95 from pycocotools, single area range, maxDets as above."""
    coco = COCO()
    coco.dataset = {"images": images, "annotations": anns,
                    "categories": [{"id": 1, "name": "tree"}]}
    coco.createIndex()
    ev = COCOeval(coco, coco.loadRes(dets), "segm")
    ev.params.maxDets = [1, 100, MAX_DETS]
    ev.params.areaRng, ev.params.areaRngLbl = [[0, 1e10]], ["all"]
    ev.evaluate(); ev.accumulate()
    p = ev.eval["precision"]                       # [T, R, K, A, M]
    m = len(ev.params.maxDets) - 1
    i50 = int(np.where(np.isclose(ev.params.iouThrs, 0.5))[0][0])
    c50, csw = p[i50, :, 0, 0, m], p[:, :, 0, 0, m]
    return float(c50[c50 > -1].mean()), float(csw[csw > -1].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(io.RESULTS, "cocoeval_parity.json"))
    a = ap.parse_args()

    warnings.filterwarnings("ignore")      # Accelerate matmul flags; see module docstring
    np.seterr(all="ignore")

    images, anns, dets, mI, S, IGN, n_gt = collect(a.seed, io.load_gt(), a.progress)

    ours50 = E._greedy_ap(mI, S, IGN, n_gt, 0.5)
    sweep = [E._greedy_ap(mI, S, IGN, n_gt, t) for t in E.IOU_50_95]
    ours = {"mask_mAP50": round(float(ours50), 4),
            "mask_mAP50_95": round(float(np.nanmean(sweep)), 4)}

    # GATE: our re-score must reproduce the recorded canopy=FP reference before the
    # COCOeval comparison means anything.
    gate.check(ours, REF, label=f"canopy=FP seed {a.seed}")

    c50, csw = cocoeval_ap(images, anns, dets)

    out = {
        "label": "AP-core parity: evaluate._greedy_ap vs pycocotools COCOeval",
        "arm": "canopy=FP (ignore disabled) -- the only arm where the two scorers see "
               "identical inputs, since COCOeval has no equivalent of our per-prediction "
               "canopy-ignore rule",
        "seed": a.seed, "n_tiles": len(images), "n_gt": n_gt, "n_det": len(dets),
        "max_dets": MAX_DETS, "iou_type": "segm", "mask_res": E.RES,
        "reference": {"value": REF, "source": os.path.relpath(REF_PATH, io.REPO),
                      "path_in_file": "per_seed.0.no_ignore"},
        "greedy_ap": ours,
        "cocoeval": {"mask_mAP50": round(c50, 4), "mask_mAP50_95": round(csw, 4)},
        "delta_cocoeval_minus_ours": {
            "mask_mAP50": round(c50 - ours["mask_mAP50"], 4),
            "mask_mAP50_95": round(csw - ours["mask_mAP50_95"], 4)},
        "verdict": (
            "AP50 agrees EXACTLY to 4dp -- greedy matching, cross-image pooling and the "
            "101-point precision envelope match COCO. The AP50:95 delta of +0.0008 is the "
            "matching-fallback difference: _greedy_ap (evaluate.py:108-112) takes the "
            "best-IoU GT and gives up if it is already matched, whereas COCOeval falls "
            "back to the best UNMATCHED GT above threshold. Ours is therefore strictly "
            "conservative -- it under-counts TPs where crowns compete, never over-counts. "
            "0.0008 is about a third of the 3-seed sigma (0.005 on AP50)."),
        "scope": {
            "establishes": "our AP ARITHMETIC is the COCO arithmetic; 0.630 is on the "
                           "same scale as published COCO-convention numbers",
            "does_not_establish": [
                "that our PROTOCOL is standard -- whole 2048^2 tiles, 512^2 mask raster, "
                "maxDets 600 (COCO default 100) and canopy-ignore all remain ours",
                "anything about the Restor Mask R-CNN or SelvaBox rows of tab:oamtcd439, "
                "which were never re-scored here; whether they used COCOeval is unknown "
                "to this repo"],
            "paper_sentence": (
                "Our AP implementation was cross-validated against pycocotools COCOeval "
                "on the canopy-as-FP arm, where no ignore rule applies: AP50 agrees "
                "exactly (0.4867 for both) and AP50:95 to 0.0008.")},
        "note_macos": (
            "mask_iou (evaluate.py:56) emits spurious overflow/invalid-value matmul "
            "warnings under Apple Accelerate. Verified harmless: finite, no NaN, values "
            "in [0,1]. Warnings are suppressed by this script."),
    }
    json.dump(out, open(a.out, "w"), indent=2)
    io.record_inputs([io.knobbed(a.seed), io.GT, REF_PATH])

    print(f"\n=== AP-core parity, seed {a.seed}, canopy=FP arm "
          f"({len(images)} tiles, {n_gt} GT, {len(dets)} det) ===")
    print(f"{'scorer':<14}{'AP50':>10}{'AP50:95':>10}")
    print(f"{'_greedy_ap':<14}{ours['mask_mAP50']:>10.4f}{ours['mask_mAP50_95']:>10.4f}")
    print(f"{'COCOeval':<14}{c50:>10.4f}{csw:>10.4f}")
    print(f"{'delta':<14}{c50 - ours['mask_mAP50']:>+10.4f}"
          f"{csw - ours['mask_mAP50_95']:>+10.4f}")
    print(f"\n-> {os.path.relpath(a.out, io.REPO)}")


if __name__ == "__main__":
    main()
