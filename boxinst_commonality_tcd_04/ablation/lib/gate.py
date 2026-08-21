"""Reproduction gate.

Every script here re-scores predictions that were produced elsewhere. Before any NEW
number is reported, the gate re-scores the SAME predictions under the SAME protocol and
asserts it reproduces the recorded result. If that fails, the scoring path is wrong and
the new number would be meaningless -- so the gate raises rather than warns.

Pattern lifted from modal_tcd_multiseed/phase4/no_ignore_sensitivity.py:104-111.
"""
import json

import numpy as np

from boxinst_commonality_tcd_04 import evaluate as E
from . import io

TOL = 1e-4          # recorded refs are rounded to 4dp


class GateFailure(AssertionError):
    pass


def score_ap(mI, S, IGN, n_gt):
    """mask AP50 and the 50:95 sweep, identical core to the published eval."""
    per_iou = [E._greedy_ap(mI, S, IGN, n_gt, t) for t in E.IOU_50_95]
    return {"mask_mAP50": round(E._greedy_ap(mI, S, IGN, n_gt, 0.5), 4),
            "mask_mAP50_95": round(float(np.nanmean(per_iou)), 4),
            "per_iou": [round(float(x), 4) for x in per_iou]}


def check(got, ref_path, keys=("mask_mAP50", "mask_mAP50_95"), tol=TOL, label=""):
    """Assert `got` reproduces the recorded reference. Returns the ref for logging."""
    ref = json.load(open(ref_path)) if isinstance(ref_path, str) else ref_path
    for k in keys:
        if k not in ref:
            continue
        if abs(got[k] - ref[k]) > tol:
            raise GateFailure(
                f"GATE FAILED {label} {k}: recomputed {got[k]} vs recorded {ref[k]} "
                f"(|d|={abs(got[k] - ref[k]):.2e} > {tol:.0e}). The scoring path is wrong; "
                f"no downstream number from this run is trustworthy.")
    print(f"  gate OK {label}: reproduces "
          f"{ref.get('mask_mAP50')}/{ref.get('mask_mAP50_95')}", flush=True)
    return ref


def rescore_knobbed(seed, gt=None, progress=100):
    """Full re-score of one knobbed seed. Returns (metrics, per-tile IoU/score/ignore).

    The per-tile lists are returned so callers (T0.1 boundary, T0.4 strata) reuse the same
    single pass rather than decoding 160k masks twice.
    """
    gt = gt if gt is not None else io.load_gt()
    mI, S, IGN, n_gt = [], [], [], 0
    for t in io.per_tile(io.knobbed(seed), gt, progress=progress):
        mI.append(E.mask_iou(t["pm"], t["gm"]))
        S.append(t["scores"]); IGN.append(t["ignore"])
        n_gt += len(t["gm"])
    return score_ap(mI, S, IGN, n_gt), {"iou": mI, "scores": S, "ignore": IGN, "n_gt": n_gt}
