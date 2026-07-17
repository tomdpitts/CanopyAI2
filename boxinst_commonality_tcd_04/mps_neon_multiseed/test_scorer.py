"""Unit tests for scorer.py — run: .venv/bin/python -m boxinst_commonality_tcd_04.mps_neon_multiseed.test_scorer
(or plain `python test_scorer.py` from this dir). Pure asserts, no pytest needed."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from scorer import (evaluate, iou_matrix, match_image, pr_curve,  # noqa: E402
                    score_image)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_iou_basic():
    # identical boxes -> IoU 1
    a = np.array([[0, 0, 10, 10]])
    assert approx(iou_matrix(a, a)[0, 0], 1.0)
    # half-overlap along x: boxes [0,0,10,10] and [5,0,15,10] -> inter 50, union 150
    b = np.array([[5, 0, 15, 10]])
    assert approx(iou_matrix(a, b)[0, 0], 50 / 150)
    # disjoint -> 0
    c = np.array([[100, 100, 110, 110]])
    assert approx(iou_matrix(a, c)[0, 0], 0.0)
    print("ok test_iou_basic")


def test_worked_example():
    """Paper's worked example: 10 GT, 9 preds, all 9 matched (IoU>=0.4)
    -> recall 9/10 = 0.9, precision 9/9 = 1.0."""
    gt = np.array([[i * 20, 0, i * 20 + 10, 10] for i in range(10)], float)
    pred = gt[:9].copy()                       # 9 exact matches, GT #10 missed
    r = score_image(pred, gt, iou_thr=0.4)
    assert r["tp"] == 9 and r["fp"] == 0 and r["fn"] == 1, r
    assert approx(r["recall"], 0.9), r
    assert approx(r["precision"], 1.0), r
    print("ok test_worked_example (recall 0.9, precision 1.0)")


def test_greedy_one_to_one():
    """Two preds overlapping one GT: only the higher-IoU pred is a TP, the other
    is an FP (one-to-one). GT count 1."""
    gt = np.array([[0, 0, 10, 10]], float)
    pred = np.array([[0, 0, 10, 10],          # IoU 1.0  -> TP
                     [1, 1, 11, 11]], float)  # IoU lower -> FP (GT already taken)
    tp, fp, fn, pairs = match_image(pred, gt, 0.4)
    assert tp == 1 and fp == 1 and fn == 0, (tp, fp, fn)
    assert pairs[0][0] == 0, pairs                # the exact-overlap pred won
    print("ok test_greedy_one_to_one")


def test_greedy_picks_highest_iou():
    """One pred, two GT: pred must match the GT it overlaps MORE, leaving the other
    as FN."""
    pred = np.array([[0, 0, 10, 10]], float)
    gt = np.array([[0, 0, 10, 10],            # IoU 1.0
                   [5, 0, 15, 10]], float)    # IoU 1/3
    tp, fp, fn, pairs = match_image(pred, gt, 0.4)
    assert tp == 1 and fp == 0 and fn == 1, (tp, fp, fn)
    assert pairs[0][1] == 0, pairs                # matched the higher-IoU GT
    print("ok test_greedy_picks_highest_iou")


def test_threshold_excludes_low_iou():
    # IoU exactly 1/3 < 0.4 -> not a match
    pred = np.array([[5, 0, 15, 10]], float)
    gt = np.array([[0, 0, 10, 10]], float)
    tp, fp, fn, _ = match_image(pred, gt, 0.4)
    assert tp == 0 and fp == 1 and fn == 1, (tp, fp, fn)
    # but at thr 0.3 it matches
    tp2, _, _, _ = match_image(pred, gt, 0.3)
    assert tp2 == 1
    print("ok test_threshold_excludes_low_iou")


def test_macro_average():
    """Macro-average = mean of per-image P/R, NOT pooled. Image A: perfect (P1,R1).
    Image B: 1 GT, 0 preds (P policy 'zero' -> 0, R 0). Macro P=0.5, R=0.5; pooled
    micro differs."""
    gt = {"A": np.array([[0, 0, 10, 10]], float),
          "B": np.array([[0, 0, 10, 10]], float)}
    preds = {"A": {"boxes": np.array([[0, 0, 10, 10]], float), "scores": np.array([0.9])},
             "B": {"boxes": np.zeros((0, 4)), "scores": np.zeros(0)}}
    e = evaluate(preds, gt, iou_thr=0.4)
    assert approx(e["mean_precision"], 0.5), e
    assert approx(e["mean_recall"], 0.5), e
    # micro: tp=1 fp=0 fn=1 -> micro_p=1.0, micro_r=0.5 -> differs from macro P
    assert approx(e["micro_precision"], 1.0) and approx(e["micro_recall"], 0.5), e
    print("ok test_macro_average (macro P/R 0.5/0.5 != micro P 1.0)")


def test_score_threshold_filters():
    gt = {"A": np.array([[0, 0, 10, 10]], float)}
    preds = {"A": {"boxes": np.array([[0, 0, 10, 10]], float),
                   "scores": np.array([0.3])}}
    # thr below score -> matched
    assert evaluate(preds, gt, score_thr=0.2)["mean_recall"] == 1.0
    # thr above score -> filtered out -> recall 0
    assert evaluate(preds, gt, score_thr=0.5)["mean_recall"] == 0.0
    print("ok test_score_threshold_filters")


def test_pr_curve_monotone_recall():
    """As confidence threshold rises, recall must be non-increasing."""
    gt = {"A": np.array([[i * 20, 0, i * 20 + 10, 10] for i in range(5)], float)}
    boxes = gt["A"].copy()
    scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    preds = {"A": {"boxes": boxes, "scores": scores}}
    curve = pr_curve(preds, gt, thresholds=[0.0, 0.4, 0.6, 0.8, 0.95])
    recs = [p["mean_recall"] for p in curve]
    assert all(recs[i] >= recs[i + 1] - 1e-9 for i in range(len(recs) - 1)), recs
    print("ok test_pr_curve_monotone_recall")


def test_nan_precision_policies():
    gt = {"A": np.array([[0, 0, 10, 10]], float)}
    empty = {"A": {"boxes": np.zeros((0, 4)), "scores": np.zeros(0)}}
    assert evaluate(empty, gt, nan_precision="zero")["mean_precision"] == 0.0
    assert evaluate(empty, gt, nan_precision="one")["mean_precision"] == 1.0
    assert np.isnan(evaluate(empty, gt, nan_precision="nan")["mean_precision"])
    print("ok test_nan_precision_policies")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nALL {len(fns)} SCORER TESTS PASSED")
