"""Re-rank LACE predictions using the EM masker's discarded box-level evidence.

WHAT THIS TESTS
    LACE ranks masks by the CenterNet heatmap peak alone. AP depends ONLY on score
    ORDERING, so no monotone recalibration (Platt/isotonic/temperature) can move it --
    the only thing that helps is NEW information that reorders. The EM masker is an
    independent estimator (training-free, sees feature commonality, never the heatmap),
    and `estep` computes box-level evidence `A.mean()` then discards it. This asks whether
    putting it back is worth anything.

SUPERVISION DISCIPLINE
    The reranker's target is BOX IoU against GT boxes, never mask IoU -- fitting on mask
    overlap would smuggle in the mask supervision the paper claims not to use. Fitting is
    2-fold cross-fit by TILE: no prediction is ever scored by a model that saw its own
    tile's labels. That is an effect-size estimate, not a publishable protocol -- for the
    paper this must be refit on a held-out val split (see NOTE at the bottom).

    .venv/bin/python -m boxinst_commonality_tcd_04.confidence.rerank
"""
from __future__ import annotations

import json
import os

import numpy as np
from pycocotools import mask as M
from scipy.optimize import minimize

from boxinst_commonality_tcd_04 import evaluate as E

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
PREDS = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "preds", "knobbed_s0.json")
DIAG = os.path.join(HERE, "em_diag_s0.json")
GT = os.path.join(PKG, "test_gt.json")

FEATS = ["logit_s", "a_mean", "a_std", "a_p90", "a_max", "bimod", "pfg_mean",
         "log_cells", "fill", "mb_iou", "log_barea"]


def _logreg(X, y, l2=1e-3):
    """Plain L2 logistic regression via L-BFGS on standardised features."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]

    def nll(w):
        t = Z @ w
        ll = np.sum(np.logaddexp(0, t) - y * t)
        g = Z.T @ (1 / (1 + np.exp(-t)) - y)
        return ll / len(y) + l2 * w[1:] @ w[1:], g / len(y) + 2 * l2 * np.r_[0, w[1:]]

    w = minimize(nll, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
    return lambda A: 1 / (1 + np.exp(-(np.c_[np.ones(len(A)), (A - mu) / sd] @ w))), w


def build():
    """-> per-tile dict of features, mask-IoU matrix (for scoring), canopy-ignore, box-IoU
    label (for FITTING). Feature/label separation is the whole point: labels are box-only."""
    gt = json.load(open(GT))
    preds = json.load(open(PREDS))["preds"]
    diag = json.load(open(DIAG))
    tids = sorted(t for t in gt if t in preds and t in diag)
    D, n_gt = {}, 0
    for t in tids:
        r, dg = preds[t], diag[t]
        sc = np.asarray(r["scores"], np.float64)
        if not len(sc):
            continue
        rl = [{"size": x["size"], "counts": x["counts"].encode()} for x in r["masks_rle"]]
        bx = np.asarray(r["boxes_2048"], np.float64).reshape(-1, 4) / 4.0
        gm = [M.encode(np.asfortranarray(m.astype(np.uint8)))
              for m in E.raster(gt[t]["trees"], res=512, scale=4.0)]
        cm = [M.encode(np.asfortranarray(m.astype(np.uint8)))
              for m in E.raster(gt[t]["canopy"], res=512, scale=4.0)]
        can = M.merge(cm) if cm else M.encode(np.asfortranarray(np.zeros((512, 512), np.uint8)))
        n_gt += len(gm)

        ar = np.asarray([float(M.area(x)) for x in rl])
        mb = np.asarray([M.toBbox(x) for x in rl]).reshape(-1, 4)
        ba = np.clip((bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1]), 1e-6, None)
        ix0 = np.maximum(bx[:, 0], mb[:, 0]); iy0 = np.maximum(bx[:, 1], mb[:, 1])
        ix1 = np.minimum(bx[:, 2], mb[:, 0] + mb[:, 2])
        iy1 = np.minimum(bx[:, 3], mb[:, 1] + mb[:, 3])
        it = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
        ma = np.clip(mb[:, 2] * mb[:, 3], 1e-6, None)

        F = np.c_[np.log(sc / (1 - sc + 1e-9) + 1e-12),
                  dg["a_mean"], dg["a_std"], dg["a_p90"], dg["a_max"],
                  dg["bimod"], dg["pfg_mean"], np.log(np.asarray(dg["n_cells"]) + 1.0),
                  ar / ba, it / (ba + ma - it), np.log(ba)]

        miou = (np.asarray(M.iou(rl, gm, [0] * len(gm)), np.float64).reshape(len(rl), len(gm))
                if gm else np.zeros((len(rl), 0)))
        inter = np.asarray([float(M.area(M.merge([x, can], intersect=1))) for x in rl])
        ign = np.divide(inter, ar, out=np.zeros(len(rl)), where=ar > 0) > 0.5

        # FITTING LABEL: box IoU vs GT BOXES (box supervision only, never mask overlap)
        gb = np.asarray([[*np.asarray(p).reshape(-1, 2).min(0),
                          *np.asarray(p).reshape(-1, 2).max(0)]
                         for p in gt[t]["trees"]], np.float64).reshape(-1, 4) / 4.0
        if len(gb):
            a = np.maximum(bx[:, None, :2], gb[None, :, :2])
            b = np.minimum(bx[:, None, 2:], gb[None, :, 2:])
            wh = np.clip(b - a, 0, None); inter_b = wh[..., 0] * wh[..., 1]
            areas_g = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
            biou = inter_b / (ba[:, None] + areas_g[None] - inter_b + 1e-9)
            y = (biou.max(1) >= 0.5).astype(float) if biou.size else np.zeros(len(bx))
        else:
            y = np.zeros(len(bx))
        D[t] = dict(F=F, sc=sc, miou=miou, ign=ign, y=y)
    return D, tids, n_gt


def ap_with(D, tids, n_gt, score_of, iou_thr=0.5):
    I, S, G = [], [], []
    for t in tids:
        if t not in D:
            continue
        s = score_of(t)
        o = np.argsort(-s, kind="mergesort")
        I.append(D[t]["miou"][o]); S.append(s[o]); G.append(D[t]["ign"][o])
    return E._greedy_ap(I, S, G, n_gt, iou_thr)


def main():
    D, tids, n_gt = build()
    keys = [t for t in tids if t in D]
    base = ap_with(D, keys, n_gt, lambda t: D[t]["sc"])
    base5095 = float(np.nanmean([ap_with(D, keys, n_gt, lambda t: D[t]["sc"], th)
                                 for th in np.linspace(0.5, 0.95, 10)]))
    print(f"n_gt {n_gt}   boxes {sum(len(D[t]['sc']) for t in keys)}")
    print(f"\nBASELINE (detector heatmap only)   AP50 {base:.4f}   AP50:95 {base5095:.4f}")

    # single-feature AUCs, to see which signals carry independent information
    Xa = np.vstack([D[t]["F"] for t in keys]); ya = np.concatenate([D[t]["y"] for t in keys])
    print(f"\n{'feature':12}{'AUC (box-TP)':>14}{'corr w/ det score':>20}")
    for i, nm in enumerate(FEATS):
        x = Xa[:, i]
        r = np.argsort(np.argsort(x)); n1 = ya.sum(); n0 = len(ya) - n1
        auc = (r[ya > 0].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)
        print(f"{nm:12}{auc:>14.4f}{np.corrcoef(x, Xa[:, 0])[0, 1]:>20.3f}")

    h = len(keys) // 2
    for label, cols in (("EM evidence only", [0, 1, 2, 3, 4, 5, 6, 7]),
                        ("geometry only", [0, 8, 9, 10]),
                        ("EM + geometry", list(range(len(FEATS))))):
        new = {}
        for tr, te in ((keys[:h], keys[h:]), (keys[h:], keys[:h])):
            X = np.vstack([D[t]["F"][:, cols] for t in tr])
            y = np.concatenate([D[t]["y"] for t in tr])
            f, _ = _logreg(X, y)
            for t in te:
                new[t] = f(D[t]["F"][:, cols])
        a50 = ap_with(D, keys, n_gt, lambda t: new[t])
        a5095 = float(np.nanmean([ap_with(D, keys, n_gt, lambda t: new[t], th)
                                  for th in np.linspace(0.5, 0.95, 10)]))
        print(f"\n{label:20} AP50 {a50:.4f} ({a50-base:+.4f})   "
              f"AP50:95 {a5095:.4f} ({a5095-base5095:+.4f})")
    print("\nreference: Restor topk2000 AP50 0.6605 / AP50:95 0.2887; "
          "LACE oracle-rank ceiling 0.8020")
    print("NOTE: 2-fold cross-fit BY TILE on the test set. No prediction is scored by a "
          "model that saw its own tile, but the fit still uses test-set boxes. For "
          "publication refit on a held-out val split.")


if __name__ == "__main__":
    main()
