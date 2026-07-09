"""Detection metrics on decoded boxes: COCO-style AP, P/R/F1, count-error,
isolated-vs-touching recall, bootstrap CIs.

Self-contained (greedy matching + 101-point AP) so we can stratify and bootstrap
freely; cross-checkable against pycocotools on a sanity case. All boxes xyxy px.
"""
from __future__ import annotations

import numpy as np

IOU_50_95 = np.arange(0.5, 0.96, 0.05)
# COCO area ranges on box area (px^2): small <32^2, medium 32^2-96^2, large >96^2
AREA_RANGES = {"all": (0, 1e12), "small": (0, 32 ** 2),
               "medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, 1e12)}


def iou_matrix(a, b):
    """a:(N,4) b:(M,4) xyxy -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _box_area(b):
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) if len(b) else np.zeros(0)


def greedy_match(pred_boxes, pred_scores, gt_boxes, iou_thr):
    """Return (is_tp bool[Npred] in score order, n_gt). Standard greedy COCO match."""
    order = np.argsort(-pred_scores)
    pb = pred_boxes[order]
    ious = iou_matrix(pb, gt_boxes)
    matched = np.zeros(len(gt_boxes), bool)
    tp = np.zeros(len(pb), bool)
    for i in range(len(pb)):
        if ious.shape[1] == 0:
            break
        j = np.argmax(ious[i])
        if ious[i, j] >= iou_thr and not matched[j]:
            matched[j] = True
            tp[i] = True
    return tp, order, len(gt_boxes)


def average_precision(preds, gts, iou_thr, area=None):
    """preds/gts: lists per tile of (boxes,scores) / boxes. Returns COCO 101-pt AP.

    area: optional (lo,hi) px^2 filter applied to BOTH gt and predictions.
    """
    all_scores, all_tp, n_gt_total = [], [], 0
    for (pb, ps), gb in zip(preds, gts):
        if area is not None:
            gb = gb[(_box_area(gb) >= area[0]) & (_box_area(gb) < area[1])]
            m = (_box_area(pb) >= area[0]) & (_box_area(pb) < area[1])
            pb, ps = pb[m], ps[m]
        tp, order, n_gt = greedy_match(pb, ps, gb, iou_thr)
        all_scores.append(ps[order])
        all_tp.append(tp)
        n_gt_total += n_gt
    if n_gt_total == 0:
        return float("nan")
    scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
    tp = np.concatenate(all_tp) if all_tp else np.zeros(0, bool)
    if len(scores) == 0:
        return 0.0
    o = np.argsort(-scores)
    tp = tp[o]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(~tp)
    recall = tp_cum / n_gt_total
    precision = tp_cum / (tp_cum + fp_cum)
    # 101-point interpolation
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        p = precision[recall >= r].max() if np.any(recall >= r) else 0.0
        ap += p / 101
    return ap


def prf_at_threshold(preds, gts, score_thr, iou_thr=0.5):
    """Precision/recall/F1 and count-error at a fixed operating score threshold."""
    tp = fp = fn = 0
    count_err = []
    for (pb, ps), gb in zip(preds, gts):
        m = ps >= score_thr
        pb2, ps2 = pb[m], ps[m]
        is_tp, _, n_gt = greedy_match(pb2, ps2, gb, iou_thr)
        tp += int(is_tp.sum())
        fp += int((~is_tp).sum())
        fn += n_gt - int(is_tp.sum())
        count_err.append(abs(len(pb2) - len(gb)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "count_mae": float(np.mean(count_err)) if count_err else 0.0}


def touching_flags(gt_boxes):
    """bool[M]: a crown is 'touching' if it overlaps (IoU>0) another crown in tile."""
    if len(gt_boxes) < 2:
        return np.zeros(len(gt_boxes), bool)
    iou = iou_matrix(gt_boxes, gt_boxes)
    np.fill_diagonal(iou, 0)
    return iou.max(axis=1) > 0


def recall_by_stratum(preds, gts, score_thr, iou_thr=0.5):
    """Recall on isolated vs touching GT crowns."""
    hit = {"isolated": [0, 0], "touching": [0, 0]}   # [tp, total]
    for (pb, ps), gb in zip(preds, gts):
        m = ps >= score_thr
        pb2 = pb[m]
        touch = touching_flags(gb)
        ious = iou_matrix(pb2, gb)
        gt_hit = (ious >= iou_thr).any(axis=0) if len(pb2) else np.zeros(len(gb), bool)
        for j in range(len(gb)):
            key = "touching" if touch[j] else "isolated"
            hit[key][1] += 1
            hit[key][0] += int(gt_hit[j])
    return {k: (v[0] / v[1] if v[1] else float("nan"), v[1]) for k, v in hit.items()}


def pick_threshold(preds, gts, iou_thr=0.5, grid=None):
    """Score threshold maximizing F1 (chosen on val)."""
    grid = grid if grid is not None else np.linspace(0.05, 0.9, 18)
    best_t, best_f1 = 0.2, -1.0
    for t in grid:
        f1 = prf_at_threshold(preds, gts, t, iou_thr)["f1"]
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def full_report(preds, gts, score_thr, iou_thr=0.5):
    ap50 = average_precision(preds, gts, 0.5)
    ap5095 = float(np.nanmean([average_precision(preds, gts, t) for t in IOU_50_95]))
    ap_small = average_precision(preds, gts, 0.5, AREA_RANGES["small"])
    rep = {"mAP50": ap50, "mAP50_95": ap5095, "AP_small": ap_small,
           "score_thr": score_thr}
    rep.update(prf_at_threshold(preds, gts, score_thr, iou_thr))
    rep["recall_strata"] = recall_by_stratum(preds, gts, score_thr, iou_thr)
    return rep


def bootstrap_ci(preds, gts, metric_fn, n=1000, seed=0, alpha=0.05):
    """Resample tiles with replacement; return (lo, hi) percentile CI of metric_fn."""
    rng = np.random.default_rng(seed)
    N = len(preds)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        vals.append(metric_fn([preds[i] for i in idx], [gts[i] for i in idx]))
    vals = np.array([v for v in vals if not np.isnan(v)])
    return float(np.percentile(vals, 100 * alpha / 2)), \
        float(np.percentile(vals, 100 * (1 - alpha / 2)))
