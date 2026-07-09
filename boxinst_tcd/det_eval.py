"""Canopy-aware detection eval for ITC on TCD.

Individual trees (cat 2) are the positives. Canopy (cat 1) is neither positive nor
negative: a predicted box that lands in canopy and does NOT match a labelled tree is
a VALID-but-unlabelled detection, so it is IGNORED (removed from the score list), not
counted as a false positive. This mirrors COCO iscrowd/ignore handling and matches the
training-side canopy ignore mask.

Only the FP bookkeeping changes; TP/FN against labelled trees are unchanged. Falls
back to the plain dapt metrics when no canopy mask is supplied.
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw

from dapt.eval import iou_matrix
from boxinst_tcd.prepare import OUT, RES


def canopy_pixel_mask(canopy_polys, res=RES):
    """Deprecated polygon-only path (drops RLE); kept for callers passing polys.
    Prefer build_canopy.load_canopy_mask(tile_path), which includes RLE canopy."""
    m = Image.new("L", (res, res), 0)
    d = ImageDraw.Draw(m)
    for poly in canopy_polys:
        if poly and len(poly) >= 6:
            d.polygon([tuple(v) for v in np.array(poly).reshape(-1, 2)], fill=1)
    return np.asarray(m, bool)


def _centers_in_canopy(boxes, canopy):
    if canopy is None or len(boxes) == 0:
        return np.zeros(len(boxes), bool)
    cx = ((boxes[:, 0] + boxes[:, 2]) / 2).clip(0, RES - 1).astype(int)
    cy = ((boxes[:, 1] + boxes[:, 3]) / 2).clip(0, RES - 1).astype(int)
    return canopy[cy, cx]


def _iop_canopy(boxes, canopy, tau=0.5):
    """IoP = |box ∩ canopy| / |box|. True where >= tau (prediction sits on tree
    cover). Size-invariant: a small ITC fully inside a huge canopy scores 1."""
    if canopy is None or len(boxes) == 0:
        return np.zeros(len(boxes), bool)
    out = np.zeros(len(boxes), bool)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(RES, int(x1)), min(RES, int(y1))
        a = (x1 - x0) * (y1 - y0)
        if a <= 0:
            continue
        out[i] = canopy[y0:y1, x0:x1].sum() / a >= tau
    return out


def match_ignore(pred_boxes, pred_scores, gt_boxes, canopy, iou_thr, iop_tau=0.5):
    """Greedy match. Returns (tp, ignore) bool arrays in score order, and n_gt.
    ignore[i] = unmatched prediction with IoP-in-canopy >= iop_tau (on tree cover,
    valid-but-unlabelled) — size-invariant, so a small ITC inside big canopy counts."""
    order = np.argsort(-pred_scores)
    pb = pred_boxes[order]
    ious = iou_matrix(pb, gt_boxes)
    in_can = _iop_canopy(pb, canopy, iop_tau)
    matched = np.zeros(len(gt_boxes), bool)
    tp = np.zeros(len(pb), bool)
    ign = np.zeros(len(pb), bool)
    for i in range(len(pb)):
        j = int(np.argmax(ious[i])) if ious.shape[1] else -1
        if j >= 0 and ious[i, j] >= iou_thr and not matched[j]:
            matched[j] = True; tp[i] = True
        elif in_can[i]:
            ign[i] = True                       # valid-but-unlabelled canopy hit
    return tp, ign, order, len(gt_boxes)


def average_precision(preds, gts, canopies, iou_thr):
    scores, tps, n_gt = [], [], 0
    for (pb, ps), gb, can in zip(preds, gts, canopies):
        tp, ign, order, ng = match_ignore(pb, ps, gb, can, iou_thr)
        n_gt += ng
        keep = ~ign
        scores.append(ps[order][keep]); tps.append(tp[keep])
    if n_gt == 0:
        return float("nan")
    s = np.concatenate(scores) if scores else np.zeros(0)
    tp = np.concatenate(tps) if tps else np.zeros(0, bool)
    if len(s) == 0:
        return 0.0
    o = np.argsort(-s); tp = tp[o]
    tpc, fpc = np.cumsum(tp), np.cumsum(~tp)
    rec, prec = tpc / n_gt, tpc / (tpc + fpc)
    return float(sum((prec[rec >= r].max() if np.any(rec >= r) else 0.0)
                     for r in np.linspace(0, 1, 101)) / 101)


def prf(preds, gts, canopies, score_thr, iou_thr=0.5):
    tp = fp = fn = 0
    for (pb, ps), gb, can in zip(preds, gts, canopies):
        m = ps >= score_thr
        t, ign, _, ng = match_ignore(pb[m], ps[m], gb, can, iou_thr)
        tp += int(t.sum()); fp += int((~t & ~ign).sum()); fn += ng - int(t.sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def pick_threshold(preds, gts, canopies, iou_thr=0.5):
    best_t, best_f1 = 0.2, -1.0
    for t in np.linspace(0.05, 0.9, 18):
        f1 = prf(preds, gts, canopies, t, iou_thr)["f1"]
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def full_report(preds, gts, canopies, score_thr, iou_thr=0.5):
    rep = {"mAP50": average_precision(preds, gts, canopies, 0.5)}
    rep.update(prf(preds, gts, canopies, score_thr, iou_thr))
    return rep


def load_canopies(paths):
    """path-list -> list of (RES,RES) bool canopy masks (complete: polygon + RLE)."""
    from boxinst_tcd.build_canopy import load_canopy_mask
    return [load_canopy_mask(p) for p in paths]
