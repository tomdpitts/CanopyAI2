"""How much of our 439 score comes from the canopy-ignore rule? (protocol sensitivity)

Our scorer treats a prediction sitting >50% inside an OAM-TCD `canopy` region as IGNORED
rather than a false positive: canopy is real tree cover whose individual crowns were never
delineated, so a detection there is valid-but-unlabelled. Several published baselines
(SelvaMask/CanopyRS in particular) apply NO such rule and score plain COCO mask AP.

That makes their numbers un-quotable against ours until the gap is measured. This script
measures it at the HEADLINE configuration (knobbed alpha=0.3 / kappa x1.6, 3 seeds) by
scoring the identical saved predictions twice -- once with the ignore mask, once with it
forced off. Nothing else changes: same preds, same GT, same n_gt, same AP core.

It is a SENSITIVITY measurement, not a restatement. The published numbers remain the
canopy-ignore ones. A reproduction gate asserts the with-ignore pass matches the recorded
per-seed results before any no-ignore figure is reported -- if that fails, the scoring path
is wrong and the no-ignore number would be meaningless too.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.no_ignore_sensitivity
"""
import argparse
import json
import os

import numpy as np
from pycocotools import mask as maskUtils

from boxinst_commonality_tcd_04 import evaluate as E

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
PREDS = os.path.join(HERE, "preds")
GT = os.path.join(PKG, "test_gt.json")
OUT = os.path.join(HERE, "results_no_ignore_band.json")
SEEDS = (0, 1, 2)


def _decode(r):
    c = r["counts"]
    return maskUtils.decode({"size": r["size"],
                             "counts": c.encode("ascii") if isinstance(c, str) else c}
                            ).astype(bool)


def per_tile(preds_path, gt):
    """Stream tiles -> per-tile (mask IoU matrix, scores, ignore flags). Masks are freed as
    we go: 160k predicted masks at 512x512 would be ~40 GB held at once."""
    preds = json.load(open(preds_path))["preds"]
    mI, S, IGN = [], [], []
    n_gt = n_pred = n_ign = 0
    tids = sorted(gt)
    for k, tid in enumerate(tids):
        p = preds[tid]
        pm = (np.stack([_decode(r) for r in p["masks_rle"]]) if p["masks_rle"]
              else np.zeros((0, E.RES, E.RES), bool))
        sc = np.array(p["scores"], np.float32)
        gm = np.array(E.raster(gt[tid]["trees"]))
        can = np.array(E.raster(gt[tid]["canopy"]))
        can = can.any(0) if len(can) else np.zeros((E.RES, E.RES), bool)
        # the rule under test, identical to phase4_lib_tcd.eval_4p_selfmask
        ign = (np.array([bool(m.sum()) and (m & can).sum() / m.sum() > 0.5 for m in pm])
               if len(pm) else np.zeros(0, bool))
        mI.append(E.mask_iou(pm, gm)); S.append(sc); IGN.append(ign)
        n_gt += len(gm); n_pred += len(sc); n_ign += int(ign.sum())
        del pm, gm, can
        if (k + 1) % 100 == 0 or k + 1 == len(tids):
            print(f"    {k + 1}/{len(tids)}", flush=True)
    return mI, S, IGN, n_gt, n_pred, n_ign


def score(mI, S, IGN, n_gt, ignore):
    """AP at 0.5 and the 50:95 sweep. `ignore=False` forces every ignore flag off, so a
    canopy prediction becomes a plain false positive -- the no-canopy-ignore protocol."""
    ig = IGN if ignore else [np.zeros(len(s), bool) for s in S]
    per_iou = [E._greedy_ap(mI, S, ig, n_gt, t) for t in E.IOU_50_95]
    return {"mask_mAP50": round(E._greedy_ap(mI, S, ig, n_gt, 0.5), 4),
            "mask_mAP50_95": round(float(np.nanmean(per_iou)), 4),
            "per_iou": [round(float(x), 4) for x in per_iou]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", action="append", default=None,
                    help="predictions json (repeat for a multi-seed band). "
                         "Default: the knobbed 3-seed 439 run.")
    ap.add_argument("--ref", action="append", default=None,
                    help="recorded results json per --preds, for the reproduction gate. "
                         "Omit to skip the gate (then the run is UNVERIFIED).")
    ap.add_argument("--label", default="ours knobbed 3-seed")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    preds_paths = a.preds or [os.path.join(PREDS, f"knobbed_s{s}.json") for s in SEEDS]
    refs = a.ref or ([os.path.join(PREDS, f"ref_knobbed_s{s}.json") for s in SEEDS]
                     if not a.preds else [None] * len(preds_paths))
    assert len(refs) == len(preds_paths), "--ref count must match --preds"

    gt = json.load(open(GT))
    runs = {}
    for s, (pp, rp) in enumerate(zip(preds_paths, refs)):
        print(f"[run {s}] {os.path.basename(pp)}", flush=True)
        mI, S, IGN, n_gt, n_pred, n_ign = per_tile(pp, gt)
        with_ig, no_ig = score(mI, S, IGN, n_gt, True), score(mI, S, IGN, n_gt, False)

        # GATE: the with-ignore pass must reproduce the recorded result for this run.
        if rp:
            ref = json.load(open(rp))
            for k in ("mask_mAP50", "mask_mAP50_95"):
                assert abs(with_ig[k] - ref[k]) < 1e-9, \
                    f"GATE FAILED run {s} {k}: recomputed {with_ig[k]} vs recorded {ref[k]}"
            print(f"  gate OK (reproduces {ref['mask_mAP50']}/{ref['mask_mAP50_95']})",
                  flush=True)
        else:
            print("  no --ref given: reproduction gate SKIPPED (unverified)", flush=True)
        # sanity: removing the rule can only ADD false positives, never help
        assert all(x <= y + 1e-12 for x, y in zip(no_ig["per_iou"], with_ig["per_iou"])), \
            f"run {s}: no-ignore beat with-ignore at some IoU — impossible"
        print(f"  {n_ign}/{n_pred} preds ({100 * n_ign / n_pred:.1f}%) in canopy", flush=True)

        runs[s] = {"with_ignore": with_ig, "no_ignore": no_ig, "n_gt": n_gt,
                   "n_pred": n_pred, "n_canopy_ignored": n_ign,
                   "pct_ignored": round(100 * n_ign / n_pred, 1)}
        del mI, S, IGN

    ids = sorted(runs)
    band = {}
    for proto in ("with_ignore", "no_ignore"):
        # ddof=1 (SAMPLE std) — the repo convention for seed bands: the published
        # 0.630 +/- 0.005 / 0.257 +/- 0.001 reproduce only with ddof=1. numpy's default
        # ddof=0 would report 0.004 / 0.000 from the identical per-seed values.
        band[proto] = {m: [round(float(np.mean([runs[i][proto][m] for i in ids])), 4),
                           round(float(np.std([runs[i][proto][m] for i in ids], ddof=1)), 4)]
                       for m in ("mask_mAP50", "mask_mAP50_95")}
    out = {"label": a.label, "preds": preds_paths,
           "note": ("SENSITIVITY ONLY. Published numbers are the with_ignore ones. no_ignore "
                    "removes the >50%-in-canopy ignore rule so canopy predictions count as FP, "
                    "which is what baselines lacking a canopy class (e.g. SelvaMask/CanopyRS) "
                    "implicitly do. It does NOT reproduce their full protocol (tiling, per-tile "
                    "vs pooled AP, maxDets, mask resolution all still differ)."),
           "per_seed": runs, "band_mean_std": band}
    json.dump(out, open(a.out, "w"), indent=2)

    print(f"\n=== {a.label} ===")
    print(f"{'':8}{'mask mAP50':>22}{'mask mAP50-95':>24}")
    print(f"{'run':<8}{'ignore':>11}{'no-ign':>11}{'ignore':>12}{'no-ign':>12}{'delta50-95':>12}")
    for s in ids:
        aa, b = runs[s]["with_ignore"], runs[s]["no_ignore"]
        print(f"{s:<8}{aa['mask_mAP50']:>11.4f}{b['mask_mAP50']:>11.4f}"
              f"{aa['mask_mAP50_95']:>12.4f}{b['mask_mAP50_95']:>12.4f}"
              f"{b['mask_mAP50_95'] - aa['mask_mAP50_95']:>+12.4f}")
    w, nn = band["with_ignore"], band["no_ignore"]
    print(f"{'mean':<8}{w['mask_mAP50'][0]:>11.4f}{nn['mask_mAP50'][0]:>11.4f}"
          f"{w['mask_mAP50_95'][0]:>12.4f}{nn['mask_mAP50_95'][0]:>12.4f}"
          f"{nn['mask_mAP50_95'][0] - w['mask_mAP50_95'][0]:>+12.4f}")
    print(f"{'std':<8}{w['mask_mAP50'][1]:>11.4f}{nn['mask_mAP50'][1]:>11.4f}"
          f"{w['mask_mAP50_95'][1]:>12.4f}{nn['mask_mAP50_95'][1]:>12.4f}")
    print(f"\n-> {os.path.relpath(a.out, os.path.dirname(PKG))}")


if __name__ == "__main__":
    main()
