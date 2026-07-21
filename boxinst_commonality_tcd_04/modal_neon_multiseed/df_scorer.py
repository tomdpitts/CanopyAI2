"""THE scorer for every reported NEON number: the benchmark authors' own
`deepforest.evaluate_boxes` (macro per-image precision/recall, greedy IoU matching), swept
over confidence thresholds to build the PR curve. On NeonTreeEvaluation we use only the
authors' provided evaluation code.

Runs in the isolated DeepForest venv (CPU, on saved predictions — device-independent):
    ../.venv_df/bin/python df_scorer.py <pred_json> <gt_json> [out_json]
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from deepforest.evaluate import evaluate_boxes    # noqa: E402


def _ground_df(gt):
    return pd.DataFrame([{"image_path": p, "xmin": float(b[0]), "ymin": float(b[1]),
                          "xmax": float(b[2]), "ymax": float(b[3]), "label": "Tree"}
                         for p, boxes in gt.items() for b in boxes])


def _pred_df(preds, thr):
    rows = []
    for p, v in preds.items():
        bx = np.array(v["boxes"], float).reshape(-1, 4)
        sc = np.array(v["scores"], float)
        m = sc >= thr
        for b, s in zip(bx[m], sc[m]):
            rows.append({"image_path": p, "xmin": b[0], "ymin": b[1], "xmax": b[2],
                         "ymax": b[3], "label": "Tree", "score": float(s)})
    return pd.DataFrame(rows, columns=["image_path", "xmin", "ymin", "xmax", "ymax",
                                       "label", "score"])


def score(pred_json, gt_json, out_json=None, thresholds=None, iou=0.4):
    gt = json.load(open(gt_json))
    preds = json.load(open(pred_json))
    gdf = _ground_df(gt)
    if thresholds is None:
        thresholds = np.round(np.arange(0.0, 0.95, 0.02), 3)
    curve = []
    for t in thresholds:
        pdf = _pred_df(preds, float(t))
        if len(pdf) == 0:
            curve.append({"score_thr": float(t), "mean_precision": 0.0,
                          "mean_recall": 0.0})
            continue
        r = evaluate_boxes(pdf, gdf, iou_threshold=iou)   # AUTHORS' matcher, macro P/R
        curve.append({"score_thr": float(t),
                      "mean_precision": float(r["box_precision"]),
                      "mean_recall": float(r["box_recall"])})

    def f1(x):
        p, r = x["mean_precision"], x["mean_recall"]
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)
    best = max(curve, key=f1)
    res = {"iou_thr": iou, "scorer": "deepforest.evaluate_boxes (NEON benchmark authors)",
           "n_plots": len(gt), "mean_precision": round(best["mean_precision"], 4),
           "mean_recall": round(best["mean_recall"], 4),
           "best_f1_point": {"score_thr": best["score_thr"],
                             "P": round(best["mean_precision"], 4),
                             "R": round(best["mean_recall"], 4)},
           "target_paper_P": 0.659, "target_paper_R": 0.790, "pr_curve": curve}
    if out_json:
        json.dump(res, open(out_json, "w"), indent=2)
    return res


if __name__ == "__main__":
    out = sys.argv[3] if len(sys.argv) > 3 else None
    r = score(sys.argv[1], sys.argv[2], out)
    bf = r["best_f1_point"]
    print(f"[df_scorer] {os.path.basename(sys.argv[1])}: best-F1 "
          f"P={bf['P']} R={bf['R']} @thr{bf['score_thr']}  "
          f"(deepforest.evaluate_boxes)")
