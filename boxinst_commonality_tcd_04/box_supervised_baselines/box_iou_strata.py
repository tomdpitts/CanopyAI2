"""How much mask-quality signal survives once detection quality is controlled for?

The paper's claim is about the box->MASK module, not the detector. The clean way to isolate
that is to hand every method the SAME boxes -- but only promptable modules (SAM, MAL, LACE's
EM) accept a box as an input; CondInst/SOLOv2-family box-supervised methods (BoxInst,
DiscoBox, BoxLevelSet, Box2Mask) condition their mask head on an FPN LOCATION and never see
box coordinates at inference, so they cannot be prompted at all.

This script provides the instrument that works for every method WITHOUT prompting: pair each
prediction with the GT crown it matched, and record BOTH its box IoU and its mask IoU. Then
compare mask IoU *within* box-IoU strata. Within a stratum the competing methods started from
boxes of equal quality, so the residual difference is the masker. In the top stratum
(box IoU >= 0.9) "its own box" is nearly the GT box, which reconstructs the GT-box experiment
observationally -- available even to methods that cannot be handed a box.

The instrument is only worth building if the top strata are populated, which is what this
script measures. It reports, per row, the matched-instance count per box-IoU bin and the mean
mask IoU in each -- so an under-powered stratum is visible as a small n, never hidden inside
an average.

Matching is intentionally box-IoU-based and greedy by score: mask IoU must be the DEPENDENT
variable, so it can play no part in deciding which prediction pairs with which crown. Canopy
predictions are dropped by the project's >50%-in-canopy rule before matching.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.box_supervised_baselines.box_iou_strata
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from pycocotools import mask as maskUtils

from boxinst_commonality_tcd_04 import evaluate as E

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(PKG, "test_gt.json")
RES = 512
SCALE = 2048.0 / RES
BINS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]

# EVERY row listed here enters the common-subset intersection, so adding one moves every
# published number. Change this dict and regenerate the whole mask-IoU column at once -- never
# one cell. All three LACE seeds are present so the column can carry a real band; they were
# recovered from tcd04-phase4-vol on 2026-09-02 (confidence/PREDICTIONS.md).
ROWS = {
    "LACE s0 (product)": os.path.join(PKG, "confidence", "preds_knobbed_s0_product.json"),
    "LACE s1 (product)": os.path.join(PKG, "confidence", "preds_knobbed_s1_product.json"),
    "LACE s2 (product)": os.path.join(PKG, "confidence", "preds_knobbed_s2_product.json"),
    "Restor MRCNN rpn1000": os.path.join(PKG, "restor_baseline", "preds_restor_rpn1000.json"),
    "DetecTree2 s0": os.path.join(PKG, "detectree2_baseline", "preds_dt2_s0v2.json"),
    "SelvaBox -> SAM 3": os.path.join(PKG, "selvabox_baseline", "preds_selvabox_bench.json"),
    "Box2Mask R-50": os.path.join(PKG, "box_supervised_baselines", "preds_box2mask_s0_test.json"),
    "Box2Mask Swin-L": os.path.join(PKG, "box_supervised_baselines",
                                    "preds_box2mask_swin-l_s0_test.json"),
}
ROWS = {k: v for k, v in ROWS.items() if os.path.exists(v)}   # skip arms not yet run


def _as_bytes(r):
    c = r["counts"]
    return {"size": list(r["size"]), "counts": c.encode("ascii") if isinstance(c, str) else c}


def _boxes_from_rles(rles):
    """Tight box of each mask, in RASTER coords."""
    if not rles:
        return np.zeros((0, 4), np.float64)
    xywh = np.asarray([maskUtils.toBbox(r) for r in rles], np.float64).reshape(-1, 4)
    return np.stack([xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]], 1)


def _det_boxes(rec, n):
    """The DETECTION boxes, in raster coords -- the box the masker was actually handed.

    This must be the stratifying variable, not the tight box of the predicted mask. Mask-
    derived boxes are partly determined by the mask, so stratifying on them would leak the
    dependent variable into the stratifier: a masker that under-segments would be scored as
    having had a worse box, which is precisely backwards. GT boxes stay mask-derived (the
    tight box of the GT polygon) because that is the true crown extent, not an estimate."""
    b = np.asarray(rec["boxes_2048"], np.float64).reshape(-1, 4) / SCALE
    assert len(b) == n, f"boxes ({len(b)}) and masks ({n}) misaligned"
    return b


def _box_iou(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def pairs_for_row(preds, gt, min_box_iou=0.5):
    """-> {(tile, gt_index): (box_iou, mask_iou)} for every prediction matched to a GT crown.

    Keyed by the GT crown rather than returned as bare arrays so the rows can be intersected:
    strata equalise BOX quality but not CROWN difficulty, and the methods do not match the
    same crowns (LACE matches ~2.9k more than Restor). Comparing raw per-row means would
    therefore charge the higher-recall method for the extra hard crowns only it found."""
    out = {}
    for tid in sorted(gt):
        rec = preds.get(tid)
        if not rec or not rec.get("masks_rle"):
            continue
        p_rles = [_as_bytes(r) for r in rec["masks_rle"]]
        assert tuple(p_rles[0]["size"]) == (RES, RES), f"{tid}: masks not at {RES}"
        sc = np.asarray(rec["scores"], np.float32)
        p_boxes = _det_boxes(rec, len(p_rles))

        g_masks = E.raster(gt[tid]["trees"], res=RES, scale=SCALE)
        if not len(g_masks):
            continue
        g_rles = [maskUtils.encode(np.asfortranarray(m.astype(np.uint8))) for m in g_masks]
        can = E.raster(gt[tid]["canopy"], res=RES, scale=SCALE)
        can_rle = (maskUtils.merge([maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
                                    for m in can]) if len(can) else None)

        # drop canopy-landing predictions first: the project ignores them everywhere else
        if can_rle is not None:
            ar = np.asarray([maskUtils.area(r) for r in p_rles], np.float64)
            inter = np.asarray([maskUtils.area(maskUtils.merge([r, can_rle], intersect=1))
                                for r in p_rles], np.float64)
            keep = np.flatnonzero(np.divide(inter, ar, out=np.zeros_like(inter), where=ar > 0) <= 0.5)
            p_rles = [p_rles[i] for i in keep]; sc = sc[keep]; p_boxes = p_boxes[keep]
        if not p_rles:
            continue

        bi = _box_iou(p_boxes, _boxes_from_rles(g_rles))
        mi = np.asarray(maskUtils.iou(p_rles, g_rles, [0] * len(g_rles)),
                        np.float64).reshape(len(p_rles), len(g_rles))
        taken = np.zeros(len(g_rles), bool)
        for i in np.argsort(-sc, kind="mergesort"):        # greedy by score, as AP does
            cand = np.where(taken, -1.0, bi[i])
            j = int(np.argmax(cand))
            if cand[j] < min_box_iou:
                continue
            taken[j] = True
            out[(tid, j)] = (float(bi[i, j]), float(mi[i, j]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PKG, "results_439", "box_iou_strata.json"))
    a = ap.parse_args()
    gt = json.load(open(GT))

    matched = {}
    for label, fp in ROWS.items():
        if not os.path.exists(fp):
            print(f"[strata] SKIP {label}: {fp} not found", flush=True)
            continue
        d = json.load(open(fp))
        matched[label] = pairs_for_row(d.get("preds", d), gt)
        print(f"[strata] {label}: {len(matched[label])} matched", flush=True)

    common = set.intersection(*(set(v) for v in matched.values()))
    print(f"[strata] common subset (matched by ALL {len(matched)} rows): {len(common)} crowns",
          flush=True)

    def _bin_table(keys):
        out = {}
        for label, pairs in matched.items():
            b = np.asarray([pairs[k][0] for k in keys if k in pairs])
            m = np.asarray([pairs[k][1] for k in keys if k in pairs])
            bins = []
            for lo, hi in BINS:
                sel = (b >= lo) & (b < hi)
                bins.append({"box_iou": f"[{lo:.1f},{hi:.1f})", "n": int(sel.sum()),
                             "mean_mask_iou": round(float(m[sel].mean()), 4) if sel.any() else None})
            out[label] = {"n_matched": int(len(b)),
                          "mean_box_iou": round(float(b.mean()), 4) if len(b) else None,
                          "mean_mask_iou": round(float(m.mean()), 4) if len(m) else None,
                          "bins": bins}
        return out

    report = {"all_matches": _bin_table(set().union(*(set(v) for v in matched.values()))),
              "common_subset": _bin_table(common),
              "n_common": len(common)}
    json.dump(report, open(a.out, "w"), indent=2)

    # One table, each cell "mean mask IoU (n)". The n belongs beside the mean, not in a
    # separate table: a stratum's mean is only as trustworthy as the count behind it, and the
    # counts differ ~6x across rows within a column because the methods' box quality differs.
    w = max(len(l) for l in report["common_subset"])
    cell = 16
    hdr = ("box IoU".ljust(w) + " | "
           + " | ".join(f"{lo:.1f}-{hi:.1f}".center(cell) for lo, hi in BINS)
           + " | " + "all".center(cell))
    for arm, title in (("common_subset",
                        f"COMMON SUBSET -- the {len(common)} crowns EVERY row found (paired)"),
                       ("all_matches", "ALL MATCHES (confounded: rows match different crowns)")):
        print(f"\n=== {title} ===")
        print("mean mask IoU (n)   -- compare DOWN each column, never across\n" + hdr)
        print("-" * len(hdr))
        for label, row in report[arm].items():
            cells = [("None".center(cell) if b["mean_mask_iou"] is None
                      else f"{b['mean_mask_iou']:.4f} ({b['n']})".center(cell))
                     for b in row["bins"]]
            cells.append(f"{row['mean_mask_iou']:.4f} ({row['n_matched']})".center(cell))
            print(label.ljust(w) + " | " + " | ".join(cells))
        print("mean box IoU: " + "  ".join(
            f"{l}={r['mean_box_iou']:.4f}" for l, r in report[arm].items()))
    print(f"\n-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
