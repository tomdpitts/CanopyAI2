"""Python reimplementation of the NeonTreeEvaluation box scorer (Weinstein et al.
2021, PLOS Comput Biol; benchmark protocol for the image-annotated crown partition).

NO R / no `NeonTreeEvaluation` R package is used — this is a from-scratch port of the
documented protocol:

  - True positive  : a predicted box with IoU >= `iou_thr` (default 0.4) to a GT box.
  - Matching       : one-to-one, GLOBAL-GREEDY by highest IoU. Repeatedly take the
                     highest-IoU (pred, gt) pair with IoU >= thr, lock both out, and
                     continue. Each GT matches its single best available prediction;
                     each prediction matches at most one GT.
  - Unmatched preds = false positives; unmatched GT = false negatives.
  - Per IMAGE      : precision = TP/(TP+FP), recall = TP/(TP+FN).
  - Benchmark score: MACRO-average — the MEAN of the per-image precision and recall,
                     NOT pooled over all boxes ("mean precision and recall per image
                     rather than pooling results across sites", Weinstein et al.).
  - All geometry is in image PIXEL coordinates (xmin, ymin, xmax, ymax).

The scored plot set is fixed by the caller (the intersection of eval tiles and
annotation XMLs). Every scored image has >=1 GT box, so per-image recall is always
defined. An image may have 0 predictions -> precision denominator 0; the convention
is configurable via `nan_precision` and validated against the DeepForest
reproduction in Step 3 (default 'zero': found-nothing-on-an-annotated-tile = 0).
"""
from __future__ import annotations

import numpy as np


def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Axis-aligned IoU between pred (N,4) and gt (M,4), xyxy pixel coords -> (N,M)."""
    pred = np.asarray(pred, np.float64).reshape(-1, 4)
    gt = np.asarray(gt, np.float64).reshape(-1, 4)
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), np.float64)
    # areas
    pa = (pred[:, 2] - pred[:, 0]).clip(0) * (pred[:, 3] - pred[:, 1]).clip(0)
    ga = (gt[:, 2] - gt[:, 0]).clip(0) * (gt[:, 3] - gt[:, 1]).clip(0)
    # pairwise intersection
    x0 = np.maximum(pred[:, None, 0], gt[None, :, 0])
    y0 = np.maximum(pred[:, None, 1], gt[None, :, 1])
    x1 = np.minimum(pred[:, None, 2], gt[None, :, 2])
    y1 = np.minimum(pred[:, None, 3], gt[None, :, 3])
    inter = (x1 - x0).clip(0) * (y1 - y0).clip(0)
    union = pa[:, None] + ga[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def match_image(pred_boxes, gt_boxes, iou_thr: float = 0.4):
    """Global-greedy one-to-one matching. Returns (tp, fp, fn, matched_pairs) where
    matched_pairs is a list of (pred_idx, gt_idx, iou)."""
    pred_boxes = np.asarray(pred_boxes, np.float64).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, np.float64).reshape(-1, 4)
    n, m = len(pred_boxes), len(gt_boxes)
    if n == 0 or m == 0:
        return 0, n, m, []
    iou = iou_matrix(pred_boxes, gt_boxes)
    # candidate (pred, gt) pairs above threshold, sorted by descending IoU
    pi, gi = np.where(iou >= iou_thr)
    if len(pi) == 0:
        return 0, n, m, []
    order = np.argsort(-iou[pi, gi])
    pred_used = np.zeros(n, bool)
    gt_used = np.zeros(m, bool)
    pairs = []
    for k in order:
        p, g = int(pi[k]), int(gi[k])
        if pred_used[p] or gt_used[g]:
            continue
        pred_used[p] = gt_used[g] = True
        pairs.append((p, g, float(iou[p, g])))
    tp = len(pairs)
    fp = n - tp                      # unmatched predictions
    fn = m - tp                      # unmatched ground truth
    return tp, fp, fn, pairs


def score_image(pred_boxes, gt_boxes, iou_thr: float = 0.4,
                nan_precision: str = "zero"):
    """Per-image precision/recall. `nan_precision` sets precision when there are no
    predictions (0/0): 'zero' -> 0.0, 'one' -> 1.0, 'nan' -> np.nan (drop from mean)."""
    tp, fp, fn, _ = match_image(pred_boxes, gt_boxes, iou_thr)
    if tp + fp == 0:
        precision = {"zero": 0.0, "one": 1.0, "nan": np.nan}[nan_precision]
    else:
        precision = tp / (tp + fp)
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def _filter_by_score(boxes, scores, score_thr):
    if scores is None:
        return np.asarray(boxes, np.float64).reshape(-1, 4)
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    scores = np.asarray(scores, np.float64).reshape(-1)
    keep = scores >= score_thr
    return boxes[keep]


def evaluate(predictions: dict, ground_truth: dict, iou_thr: float = 0.4,
             score_thr: float = 0.0, nan_precision: str = "zero"):
    """Macro-average benchmark score over the scored plot list.

    predictions  : {plot_name: {"boxes": (N,4), "scores": (N,) or None}}
    ground_truth : {plot_name: (M,4)}   -- defines the scored set (its keys)

    Returns dict with mean_precision, mean_recall (macro over images), the pooled
    micro totals, and the per-image table. Images with NaN precision (policy 'nan')
    are dropped from the precision mean only.
    """
    per_image = {}
    tp_tot = fp_tot = fn_tot = 0
    for plot, gt in ground_truth.items():
        pred = predictions.get(plot, {})
        pb = _filter_by_score(pred.get("boxes", np.zeros((0, 4))),
                              pred.get("scores"), score_thr)
        r = score_image(pb, gt, iou_thr, nan_precision)
        per_image[plot] = r
        tp_tot += r["tp"]; fp_tot += r["fp"]; fn_tot += r["fn"]
    precs = np.array([r["precision"] for r in per_image.values()], np.float64)
    recs = np.array([r["recall"] for r in per_image.values()], np.float64)
    mean_p = float(np.nanmean(precs)) if len(precs) else float("nan")
    mean_r = float(np.nanmean(recs)) if len(recs) else float("nan")
    micro_p = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) else float("nan")
    micro_r = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) else float("nan")
    return {
        "iou_thr": iou_thr, "score_thr": score_thr, "n_images": len(ground_truth),
        "mean_precision": mean_p, "mean_recall": mean_r,       # <-- the benchmark score
        "micro_precision": micro_p, "micro_recall": micro_r,   # pooled, for reference
        "tp": tp_tot, "fp": fp_tot, "fn": fn_tot,
        "per_image": per_image,
    }


def pr_curve(predictions: dict, ground_truth: dict, iou_thr: float = 0.4,
             thresholds=None, nan_precision: str = "zero"):
    """Sweep the confidence threshold; return list of {score_thr, mean_precision,
    mean_recall} points (macro-average at each threshold)."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.0, 1.0, 0.02), 3)
    out = []
    for t in thresholds:
        e = evaluate(predictions, ground_truth, iou_thr, float(t), nan_precision)
        out.append({"score_thr": float(t), "mean_precision": e["mean_precision"],
                    "mean_recall": e["mean_recall"]})
    return out
