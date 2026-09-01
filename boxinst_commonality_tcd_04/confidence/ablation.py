"""Ablation over the confidence reranker: which evidence, which model form, how many
fitted numbers.

WHAT IS HELD FIXED
    The detection SET is frozen -- every row rescores the same 159,687 seed-0 boxes and
    only the ORDERING changes. That makes the mask-IoU matrix, the canopy-ignore flags and
    n_gt score-invariant, so they are rasterised once into `_ap_cache_s0.pkl` and every row
    after that is a `evaluate._greedy_ap` over cached arrays (~0.25 s). The AP50 this
    reproduces is byte-identical to `score_coco`'s canopy_neutral_legacy arm, which in turn
    equals canopy_neutral_crowd at AP50.

WHAT IS BEING ASKED
    1  feature subsets   -- all 128 subsets of the 7 EM statistics (detector score always in)
    2  model form        -- fitted logistic regression vs an unfitted product
    3  confounds         -- how much of the gain is a plain box-SIZE prior, no masker needed
    4  selection         -- val box-AP50 on the 108 held-out tiles, the only test-free proxy
    5  significance      -- paired tile bootstrap, 2000 resamples

    .venv/bin/python -m boxinst_commonality_tcd_04.confidence.ablation [--full]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle

import numpy as np
from scipy.optimize import minimize

from boxinst_commonality_tcd_04 import evaluate as E

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
P4 = os.path.join(PKG, "modal_tcd_multiseed", "phase4")
CACHE = os.path.join(HERE, "_ap_cache_s0.pkl")
RES, SCALE = 512, 4.0

EM = ["a_mean", "a_std", "a_p90", "a_max", "bimod", "pfg_mean", "log_cells"]
FEATS = ["logit_s"] + EM
I = {n: i for i, n in enumerate(FEATS)}


# ------------------------------------------------------------------ features / fitting
def feats(scores, dg):
    s = np.asarray(scores, np.float64)
    return np.c_[np.log(s / (1 - s + 1e-9) + 1e-12),
                 dg["a_mean"], dg["a_std"], dg["a_p90"], dg["a_max"],
                 dg["bimod"], dg["pfg_mean"],
                 np.log(np.asarray(dg["n_cells"], np.float64) + 1.0)]


def gt_boxes(polys):
    return (np.asarray([[*np.asarray(p).reshape(-1, 2).min(0),
                         *np.asarray(p).reshape(-1, 2).max(0)] for p in polys],
                       np.float64).reshape(-1, 4) if polys else np.zeros((0, 4)))


def box_iou(b, g):
    if not len(g) or not len(b):
        return np.zeros((len(b), len(g)))
    a0 = np.maximum(b[:, None, :2], g[None, :, :2])
    a1 = np.minimum(b[:, None, 2:], g[None, :, 2:])
    wh = np.clip(a1 - a0, 0, None)
    it = wh[..., 0] * wh[..., 1]
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return it / (ab[:, None] + ag[None] - it + 1e-9)


def fit_logreg(X, y, l2=1e-3):
    """L2 logistic regression on standardised features -- same fit as rerank_val.py."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]

    def obj(w):
        t = np.clip(Z @ w, -50, 50)
        g = Z.T @ (1 / (1 + np.exp(-t)) - y) / len(y)
        return (np.sum(np.logaddexp(0, t) - y * t) / len(y) + l2 * w[1:] @ w[1:],
                g + 2 * l2 * np.r_[0.0, w[1:]])

    w = minimize(obj, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B",
                 options={"maxiter": 500}).x
    return (lambda A: 1 / (1 + np.exp(-np.clip(
        np.c_[np.ones(len(A)), (A - mu) / sd] @ w, -50, 50)))), w


# ------------------------------------------------------------------ ranking functions
def log_s(X):
    return X[:, 0] - np.log1p(np.exp(X[:, 0]))          # log sigmoid(logit_s)


def product(X):
    """The zero-parameter reranker: log(s * bimod * pfg_mean)."""
    return log_s(X) + np.log(X[:, I["bimod"]]) + np.log(X[:, I["pfg_mean"]])


def size_block(X):
    lc = X[:, I["log_cells"]]
    return np.c_[lc, lc ** 2, lc ** 3]


# ------------------------------------------------------------------ cached test scorer
def build_cache():
    from pycocotools import mask as M
    gt = json.load(open(os.path.join(PKG, "test_gt.json")))
    P = json.load(open(os.path.join(P4, "preds", "knobbed_s0.json")))["preds"]
    D, n_gt = {}, 0
    for i, t in enumerate(sorted(gt)):
        trs = [M.encode(np.asfortranarray(m.astype(np.uint8)))
               for m in E.raster(gt[t]["trees"], res=RES, scale=SCALE)]
        n_gt += len(trs)
        r = P.get(t)
        if not r or not r["scores"]:
            continue
        rl = [{"size": x["size"], "counts": x["counts"].encode()} for x in r["masks_rle"]]
        can = [M.encode(np.asfortranarray(m.astype(np.uint8)))
               for m in E.raster(gt[t]["canopy"], res=RES, scale=SCALE)]
        canu = (M.merge(can) if can
                else M.encode(np.asfortranarray(np.zeros((RES, RES), np.uint8))))
        ar = np.asarray([float(M.area(x)) for x in rl])
        inter = np.asarray([float(M.area(M.merge([x, canu], intersect=1))) for x in rl])
        iou = (np.asarray(M.iou(rl, trs, [0] * len(trs)), np.float64)
               .reshape(len(rl), len(trs)) if trs else np.zeros((len(rl), 0)))
        D[t] = dict(iou=iou.astype(np.float32),
                    ign=np.divide(inter, ar, out=np.zeros(len(rl)), where=ar > 0) > 0.5,
                    sc=np.asarray(r["scores"], np.float64))
        if (i + 1) % 100 == 0:
            print(f"  cache {i+1}/{len(gt)}", flush=True)
    pickle.dump(dict(D=D, n_gt=n_gt), open(CACHE, "wb"))
    return D, n_gt


class Test:
    def __init__(self):
        if not os.path.exists(CACHE):
            print(f"[cache] rasterising 439 GT tiles at {RES}^2 (once, ~2 min) ...", flush=True)
            self.D, self.n_gt = build_cache()
        else:
            c = pickle.load(open(CACHE, "rb"))
            self.D, self.n_gt = c["D"], c["n_gt"]
        self.tids = sorted(self.D)
        diag = json.load(open(os.path.join(HERE, "em_diag_s0.json")))
        self.X = {t: feats(self.D[t]["sc"], diag[t]) for t in self.tids}

    def ap(self, fn, thr=0.5):
        Iou, S, G = [], [], []
        for t in self.tids:
            s = np.asarray(fn(self.X[t]), np.float64)
            o = np.argsort(-s, kind="mergesort")
            Iou.append(self.D[t]["iou"][o]); S.append(s[o]); G.append(self.D[t]["ign"][o])
        return E._greedy_ap(Iou, S, G, self.n_gt, thr)

    def ap5095(self, fn):
        return float(np.nanmean([self.ap(fn, t) for t in np.linspace(0.5, 0.95, 10)]))

    def matched(self, fn, thr=0.5):
        """Per-tile (kept scores, tp, n_gt) -- matching is tile-local, so a bootstrap over
        tiles is a concatenate + sort over these."""
        out = []
        for t in self.tids:
            s = np.asarray(fn(self.X[t]), np.float64)
            o = np.argsort(-s, kind="mergesort")
            iou, ign = self.D[t]["iou"][o], self.D[t]["ign"][o]
            _, ps, tp, keep, _ = E.match_tile(iou, s[o], ign, thr)
            out.append((ps[keep], tp[keep], iou.shape[1]))
        return out


def ap_from_matched(cached, idx):
    s = np.concatenate([cached[i][0] for i in idx])
    tp = np.concatenate([cached[i][1] for i in idx])
    n = sum(cached[i][2] for i in idx)
    if not n or not len(s):
        return 0.0
    o = np.argsort(-s, kind="mergesort"); tp = tp[o]
    tpc, fpc = np.cumsum(tp), np.cumsum(~tp)
    rec, prec = tpc / n, tpc / (tpc + fpc + 1e-9)
    return float(sum((prec[rec >= r].max() if np.any(rec >= r) else 0.0)
                     for r in np.linspace(0, 1, 101)) / 101)


# ------------------------------------------------------------------ val split
class Val:
    """The 108 held-out val tiles. GT BOXES only -- fitting or selecting on mask overlap
    would import the mask supervision the paper claims not to use."""

    def __init__(self):
        vd = json.load(open(os.path.join(HERE, "em_diag_val_s0.json")))
        vg = json.load(open(os.path.join(P4, "val_gt.json")))
        self.tiles, X, y, self.n_gt = [], [], [], 0
        for t, r in sorted(vd.items()):
            g = gt_boxes(vg[t]["trees"])
            self.n_gt += len(g)
            if not r["scores"]:
                continue
            F = feats(r["scores"], r)
            iou = box_iou(np.asarray(r["boxes_2048"], np.float64).reshape(-1, 4), g)
            self.tiles.append((iou, F))
            X.append(F)
            y.append((iou.max(1) >= 0.5).astype(float) if iou.size else np.zeros(len(F)))
        self.X, self.y = np.vstack(X), np.concatenate(y)

    def ap(self, fn, thr=0.5):
        """Val BOX AP. AUC is the wrong proxy here: it weights every pair equally, AP
        weights the head of the ranking, which is the thing being changed."""
        Iou, S, G = [], [], []
        for iou, F in self.tiles:
            s = fn(F); o = np.argsort(-s, kind="mergesort")
            Iou.append(iou[o]); S.append(s[o]); G.append(np.zeros(len(s), bool))
        return E._greedy_ap(Iou, S, G, self.n_gt, thr)


# ------------------------------------------------------------------ the ablation
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also sweep all 128 EM feature subsets (~1 min)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(HERE, "ablation_results.json"))
    a = ap.parse_args()

    T, V = Test(), Val()
    base = T.ap(lambda X: X[:, 0])
    R = {"baseline_cn_ap50": round(base, 4), "n_boxes": int(sum(len(T.X[t]) for t in T.tids)),
         "n_gt": T.n_gt, "val_boxes": int(len(V.y)), "val_pos_rate": round(float(V.y.mean()), 4)}
    fit8, w8 = fit_logreg(V.X, V.y)
    print(f"BASELINE (heatmap score alone)   CN AP50 {base:.4f}   "
          f"AP50:95 {T.ap5095(lambda X: X[:, 0]):.4f}")
    print(f"PUBLISHED 8-feature logreg       CN AP50 {T.ap(fit8):.4f}   "
          f"AP50:95 {T.ap5095(fit8):.4f}   ({len(w8)-1} fitted weights + bias)\n")
    R["logreg8"] = {"cn_ap50": round(T.ap(fit8), 4), "ap50_95": round(T.ap5095(fit8), 4),
                    "ap75": round(T.ap(fit8, 0.75), 4), "fitted_numbers": len(w8)}

    # --- 1. feature subsets ------------------------------------------------------------
    if a.full:
        print("=== 1. ALL 128 EM FEATURE SUBSETS (detector score always in, val-fitted) ===")
        rows = []
        for r in range(8):
            for c in itertools.combinations(range(1, 8), r):
                cols = [0] + list(c)
                f, _ = fit_logreg(V.X[:, cols], V.y)
                rows.append(dict(names=[FEATS[i] for i in c],
                                 ap=round(T.ap(lambda X, cc=cols, ff=f: ff(X[:, cc])), 4)))
        aps = np.array([r["ap"] for r in rows])
        print(f"  test CN AP50 over 128 subsets: min {aps.min():.4f}  "
              f"median {np.median(aps):.4f}  max {aps.max():.4f}")
        for k in range(1, 9):
            b = max([r for r in rows if len(r["names"]) == k - 1], key=lambda d: d["ap"])
            print(f"  best k={k}: {b['ap']:.4f} ({b['ap']-base:+.4f})  "
                  f"{'+'.join(b['names']) or '(score only)'}")
        R["subsets"] = rows
        print()

    # --- 2. model form -----------------------------------------------------------------
    print("=== 2. MODEL FORM: fitted logreg vs an unfitted product ===")
    print(f"{'ranking function':38}{'fitted':>7}{'AP50':>8}{'delta':>9}{'AP75':>8}{'AP50:95':>9}")
    forms = {
        "s  (baseline)": (lambda X: X[:, 0], 0),
        "s * bimod": (lambda X: log_s(X) + np.log(X[:, I["bimod"]]), 0),
        "s * pfg_mean": (lambda X: log_s(X) + np.log(X[:, I["pfg_mean"]]), 0),
        "bimod * pfg_mean  (no detector)":
            (lambda X: np.log(X[:, I["bimod"]] * X[:, I["pfg_mean"]]), 0),
        "s * bimod * pfg_mean": (product, 0),
        "logreg(s, bimod, pfg_mean)": (None, 3),
        "logreg(s, log bimod, log pfg)": (None, 3),
        "logreg, published 8 features": (fit8, 9),
    }
    blocks = {"logreg(s, bimod, pfg_mean)":
              lambda X: np.c_[X[:, 0], X[:, I["bimod"]], X[:, I["pfg_mean"]]],
              "logreg(s, log bimod, log pfg)":
              lambda X: np.c_[X[:, 0], np.log(X[:, I["bimod"]]), np.log(X[:, I["pfg_mean"]])]}
    R["forms"] = {}
    for nm, (fn, k) in forms.items():
        if fn is None:
            B = blocks[nm]
            g, _ = fit_logreg(B(V.X), V.y)
            fn = lambda X, bb=B, gg=g: gg(bb(X))
        v = dict(fitted=k, cn_ap50=round(T.ap(fn), 4), ap75=round(T.ap(fn, 0.75), 4),
                 ap50_95=round(T.ap5095(fn), 4), val_box_ap50=round(V.ap(fn), 4))
        R["forms"][nm] = v
        print(f"{nm:38}{k:>7}{v['cn_ap50']:>8.4f}{v['cn_ap50']-base:>+9.4f}"
              f"{v['ap75']:>8.4f}{v['ap50_95']:>9.4f}")

    print("\n  exponent surface  s * bimod^q * pfg^r  (AP is scale-free: s exponent fixed 1)")
    qs = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    print("        " + "".join(f"r={r:<7.2f}" for r in qs))
    surf = {}
    for q in qs:
        row = []
        for r in qs:
            v = T.ap(lambda X, x=q, z=r: log_s(X) + x * np.log(X[:, I["bimod"]])
                     + z * np.log(X[:, I["pfg_mean"]]))
            surf[f"{q},{r}"] = round(v, 4); row.append(v)
        print(f"  q={q:<4.2f}" + "".join(f"{v:<9.4f}" for v in row))
    R["exponent_surface"] = surf

    # --- 3. size confound ---------------------------------------------------------------
    print("\n=== 3. HOW MUCH IS JUST A BOX-SIZE PRIOR? (SIZE = cubic in log n_cells) ===")
    S_ = lambda X: X[:, 0:1]
    EMlog = lambda X: np.c_[np.log(X[:, I["bimod"]]), np.log(X[:, I["pfg_mean"]])]
    EMall = lambda X: X[:, 1:7]
    R["size_decomposition"] = {}
    for nm, blk in [("s + SIZE                  [no masker at all]", [S_, size_block]),
                    ("s + log bimod, log pfg    [masker only]", [S_, EMlog]),
                    ("s + SIZE + log bimod, log pfg", [S_, size_block, EMlog]),
                    ("s + 6 EM stats (published minus log_cells)", [S_, EMall]),
                    ("s + 6 EM stats + SIZE", [S_, EMall, size_block]),
                    ("s * bimod * pfg  + val-fitted SIZE correction",
                     [lambda X: product(X)[:, None], size_block])]:
        B = lambda X, bb=blk: np.c_[tuple(f(X) for f in bb)]
        g, _ = fit_logreg(B(V.X), V.y)
        v = T.ap(lambda X, bb=B, gg=g: gg(bb(X)))
        R["size_decomposition"][nm.split("  ")[0].strip()] = round(v, 4)
        print(f"  {nm:46}{v:>8.4f}{v-base:>+9.4f}")
    p = T.ap(product)
    print(f"  {'s * bimod * pfg  (0 fitted numbers)':46}{p:>8.4f}{p-base:>+9.4f}")

    # --- 4. val-side selection ----------------------------------------------------------
    print("\n=== 4. VAL-SIDE SELECTION (108 held-out tiles, box AP -- no test data) ===")
    cands = {"s": lambda X: X[:, 0], "s*bimod": lambda X: log_s(X) + np.log(X[:, I["bimod"]]),
             "s*pfg": lambda X: log_s(X) + np.log(X[:, I["pfg_mean"]]),
             "s*bimod*pfg": product,
             "s*bimod^0.5*pfg": lambda X: log_s(X) + .5 * np.log(X[:, I["bimod"]])
             + np.log(X[:, I["pfg_mean"]]),
             "s*bimod*pfg^0.5": lambda X: log_s(X) + np.log(X[:, I["bimod"]])
             + .5 * np.log(X[:, I["pfg_mean"]]),
             "s*bimod^2*pfg": lambda X: log_s(X) + 2 * np.log(X[:, I["bimod"]])
             + np.log(X[:, I["pfg_mean"]]),
             "s*bimod*pfg^2": lambda X: log_s(X) + np.log(X[:, I["bimod"]])
             + 2 * np.log(X[:, I["pfg_mean"]])}
    print(f"{'zero-parameter form':22}{'VAL boxAP50':>12}{'VAL boxAP50:95':>16}"
          f"{'| TEST AP50':>13}{'delta':>9}")
    R["val_selection"] = {}
    for nm, f in sorted(cands.items(), key=lambda kv: -V.ap(kv[1])):
        v50, v95, t50 = V.ap(f), float(np.nanmean([V.ap(f, t) for t in
                                                   np.linspace(0.5, 0.95, 10)])), T.ap(f)
        R["val_selection"][nm] = dict(val_box_ap50=round(v50, 4),
                                      val_box_ap5095=round(v95, 4), test_cn_ap50=round(t50, 4))
        print(f"{nm:22}{v50:>12.4f}{v95:>16.4f}{t50:>13.4f}{t50-base:>+9.4f}")

    # --- 5. significance ----------------------------------------------------------------
    print(f"\n=== 5. PAIRED TILE BOOTSTRAP ({a.boot} resamples of the 439 tiles) ===")
    rng = np.random.default_rng(0)
    n = len(T.tids)
    B = rng.integers(0, n, (a.boot, n))
    R["bootstrap"] = {}
    for thr, lab in ((0.5, "AP50"), (0.75, "AP75")):
        C = {k: T.matched(f, thr) for k, f in
             (("baseline", lambda X: X[:, 0]), ("product", product), ("logreg8", fit8))}
        d = {k: np.array([ap_from_matched(C[k], b) - ap_from_matched(C["baseline"], b)
                          for b in B]) for k in ("product", "logreg8")}
        print(f"  {lab}:")
        for k, v in d.items():
            R["bootstrap"][f"{lab}_{k}"] = [round(float(v.mean()), 5),
                                            round(float(np.percentile(v, 2.5)), 5),
                                            round(float(np.percentile(v, 97.5)), 5)]
            print(f"    {k:9} vs baseline {v.mean():+.4f}  95% CI "
                  f"[{np.percentile(v,2.5):+.4f}, {np.percentile(v,97.5):+.4f}]")
        dd = d["logreg8"] - d["product"]
        R["bootstrap"][f"{lab}_logreg8_minus_product"] = [
            round(float(dd.mean()), 5), round(float(np.percentile(dd, 2.5)), 5),
            round(float(np.percentile(dd, 97.5)), 5), round(float((dd > 0).mean()), 3)]
        print(f"    logreg8 - product      {dd.mean():+.5f}  95% CI "
              f"[{np.percentile(dd,2.5):+.5f}, {np.percentile(dd,97.5):+.5f}]  "
              f"P(logreg8 better) {(dd>0).mean():.3f}")

    json.dump(R, open(a.out, "w"), indent=1)
    print(f"\n-> {os.path.relpath(a.out)}")


if __name__ == "__main__":
    main()
