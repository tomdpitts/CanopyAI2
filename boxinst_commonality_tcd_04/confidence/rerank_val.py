"""Fit the EM-evidence reranker on the HELD-OUT VAL split, apply to the 439 test set.

No test data touches the fit. The 108 val tiles are held out from the detector's 792
training tiles (train_tiles_gt.json `partition`), so they are clean for calibration.

SUPERVISION: the fitting target is BOX IoU >= 0.5 against val GT BOXES. Never mask IoU --
fitting on mask overlap would import the mask supervision the paper claims not to use.

SCORE SCALE: the reranked score is a probability on a different scale from the heatmap
score, so applying the 0.05 floor to it would cut a DIFFERENT set of detections and the
comparison would not be like-for-like. The detection SET is therefore frozen (everything
that passed the 0.05 floor at decode, which is what the saved preds already contain) and
only the ORDERING changes -- so the reranked file must be scored with --score_floor 0.
The baseline scored at floor 0 is identical to it at floor 0.05, since nothing in the
saved preds is below 0.05.

    .venv/bin/python -m boxinst_commonality_tcd_04.confidence.rerank_val
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
P4 = os.path.join(PKG, "modal_tcd_multiseed", "phase4")
VAL_DIAG = os.path.join(HERE, "em_diag_val_s0.json")
VAL_GT = os.path.join(P4, "val_gt.json")
TEST_DIAG = os.path.join(HERE, "em_diag_s0.json")
TEST_PREDS = os.path.join(P4, "preds", "knobbed_s0.json")
OUT_PREDS = os.path.join(HERE, "preds_knobbed_s0_reranked.json")

# EM evidence only. The cross-fit ablation showed geometry (fill / mask-box IoU / box area)
# adds nothing once EM evidence is present (+0.0346 vs +0.0351), and geometry needs masks,
# which the val extraction deliberately does not compute.
FEATS = ["logit_s", "a_mean", "a_std", "a_p90", "a_max", "bimod", "pfg_mean", "log_cells"]


def _feats(scores, dg):
    s = np.asarray(scores, np.float64)
    return np.c_[np.log(s / (1 - s + 1e-9) + 1e-12),
                 dg["a_mean"], dg["a_std"], dg["a_p90"], dg["a_max"],
                 dg["bimod"], dg["pfg_mean"],
                 np.log(np.asarray(dg["n_cells"], np.float64) + 1.0)]


def _box_iou_label(boxes_2048, gt_polys, thr=0.5):
    """1 if the predicted box hits ANY GT box at IoU>=thr. Box supervision only."""
    b = np.asarray(boxes_2048, np.float64).reshape(-1, 4)
    if not gt_polys or not len(b):
        return np.zeros(len(b))
    g = np.asarray([[*np.asarray(p).reshape(-1, 2).min(0),
                     *np.asarray(p).reshape(-1, 2).max(0)] for p in gt_polys],
                   np.float64).reshape(-1, 4)
    a0 = np.maximum(b[:, None, :2], g[None, :, :2])
    a1 = np.minimum(b[:, None, 2:], g[None, :, 2:])
    wh = np.clip(a1 - a0, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    iou = inter / (ab[:, None] + ag[None] - inter + 1e-9)
    return (iou.max(1) >= thr).astype(float)


def fit_logreg(X, y, l2=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]

    def obj(w):
        t = np.clip(Z @ w, -50, 50)
        ll = np.sum(np.logaddexp(0, t) - y * t) / len(y)
        g = Z.T @ (1 / (1 + np.exp(-t)) - y) / len(y)
        return ll + l2 * w[1:] @ w[1:], g + 2 * l2 * np.r_[0.0, w[1:]]

    w = minimize(obj, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B",
                 options={"maxiter": 500}).x
    return (lambda A: 1 / (1 + np.exp(-np.clip(
        np.c_[np.ones(len(A)), (A - mu) / sd] @ w, -50, 50)))), w


def main():
    vd = json.load(open(VAL_DIAG))
    vg = json.load(open(VAL_GT))
    X, y = [], []
    for t, r in vd.items():
        if not r["scores"]:
            continue
        X.append(_feats(r["scores"], r))
        y.append(_box_iou_label(r["boxes_2048"], vg[t]["trees"]))
    X = np.vstack(X); y = np.concatenate(y)
    print(f"[fit] VAL: {len(vd)} tiles, {len(y)} boxes, "
          f"{y.mean():.1%} positive (box IoU>=0.5 vs GT boxes)")
    f, w = fit_logreg(X, y)
    print(f"[fit] weights (standardised):")
    for nm, wi in zip(["bias"] + FEATS, w):
        print(f"        {nm:11} {wi:+.4f}")

    td = json.load(open(TEST_DIAG))
    P = json.load(open(TEST_PREDS))
    preds = P["preds"]
    n_used = 0
    for t, r in preds.items():
        if t not in td or not r["scores"]:
            continue
        p = f(_feats(r["scores"], td[t]))
        r["scores"] = [round(float(v), 6) for v in p]
        n_used += len(p)
    P["meta"]["model"] = (P["meta"].get("model", "") +
                          " + EM-evidence rerank (fit on 108 held-out val tiles, "
                          "box-IoU target)")
    P["meta"]["rerank"] = {"features": FEATS, "fit_split": "val(108)",
                           "target": "box IoU>=0.5 vs GT boxes",
                           "weights_standardised": dict(zip(["bias"] + FEATS,
                                                            [round(float(v), 5) for v in w]))}
    json.dump(P, open(OUT_PREDS, "w"))
    print(f"\n[apply] rescored {n_used} test boxes -> {os.path.relpath(OUT_PREDS)}")
    print("Score it with:  .venv/bin/python -m boxinst_commonality_tcd_04.score_coco \\")
    print(f"    --preds {os.path.relpath(OUT_PREDS)} --res 512 --score_floor 0")


if __name__ == "__main__":
    main()
