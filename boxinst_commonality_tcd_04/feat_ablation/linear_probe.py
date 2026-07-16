"""Stage-1 early peek: NO training. Linear separability of in-box vs clear
background cells under feature compression.

Cell labels at the 128x128 feature grid (16px cells) from train_tiles_gt.json,
same conventions as the detector targets: positive = cell CENTER inside any ITC
box; negative = clear background (center in no box AND cell not canopy by the
pipeline's canopy_cell_mask >50% rule); canopy cells are dropped entirely.
Balanced per-tile sample (<=100 pos + 100 neg). One IO pass loads the parent
4096-dim features; pca256 / block1024 are derived from the SAME sampled vectors
(pca256 via the saved basis, block1024 = last 1024 channels), so the three
probes see identical cells.

Probe: standardized features -> logistic regression (torch LBFGS, CPU,
deterministic, L2 1e-4). Fit on train-partition tiles, AUC on val-partition
tiles (the pipeline's own 88/12 tile split — no cell-level leakage).

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.feat_ablation.linear_probe
"""
import json
import os
import time

import numpy as np
import torch

from boxinst_commonality_tcd_04.cache_train_tiles import cache_dir as train_cache
from boxinst_commonality_tcd_04.detector import canopy_cell_mask
from boxinst_commonality_tcd_04.train_detector_tiles import OUT, canopy_px
from boxinst_commonality_tcd_04.feat_ablation.variants import BLOCK_LO

HERE = os.path.abspath(os.path.dirname(__file__))
PCA_NPZ = os.path.join(HERE, "pca256.npz")
G = 128
PER = 100                                    # per-tile cap per class
SEED = 0


def cell_labels(v):
    """gt entry -> (pos, neg) bool (128,128): center-in-box vs clear background."""
    c = (np.arange(G, dtype=np.float32) + 0.5) * (2048 / G)
    pos = np.zeros((G, G), bool)
    for x0, y0, x1, y1 in np.asarray(v["boxes"], np.float32).reshape(-1, 4):
        pos |= ((c[None, :] >= x0) & (c[None, :] < x1)
                & (c[:, None] >= y0) & (c[:, None] < y1))
    canopy = canopy_cell_mask(canopy_px(v["canopy"]), G)
    return pos & ~canopy, ~pos & ~canopy


def collect():
    cdir = train_cache("web")
    gt = json.load(open(os.path.join(OUT, "train_tiles_gt.json")))
    gt = {t: v for t, v in gt.items()
          if os.path.exists(os.path.join(cdir, t + ".npy"))}
    rng = np.random.default_rng(SEED)
    X, y, part = [], [], []
    t0 = time.time()
    for i, (t, v) in enumerate(sorted(gt.items())):
        pos, neg = cell_labels(v)
        pi, ni = np.flatnonzero(pos.ravel()), np.flatnonzero(neg.ravel())
        if len(pi) == 0 or len(ni) == 0:
            continue
        pi = rng.choice(pi, min(PER, len(pi)), replace=False)
        ni = rng.choice(ni, min(PER, len(ni)), replace=False)
        cols = np.concatenate([pi, ni])
        x2d = np.load(os.path.join(cdir, t + ".npy")).reshape(4096, -1)
        X.append(x2d[:, cols].T.astype(np.float32))
        y.append(np.r_[np.ones(len(pi)), np.zeros(len(ni))])
        part.append(np.full(len(cols), v["partition"] == "train"))
        if (i + 1) % 100 == 0 or i + 1 == len(gt):
            print(f"[collect] {i+1}/{len(gt)} tiles  n={sum(len(a) for a in y)}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    return (np.concatenate(X), np.concatenate(y).astype(np.float32),
            np.concatenate(part))


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    npos, nneg = labels.sum(), (1 - labels).sum()
    return float((ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def probe(Xtr, ytr, Xva, yva, name, l2=1e-4, iters=200):
    torch.manual_seed(SEED)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = torch.from_numpy((Xtr - mu) / sd)
    Xva = torch.from_numpy((Xva - mu) / sd)
    ytr_t = torch.from_numpy(ytr)
    w = torch.zeros(Xtr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            Xtr @ w + b, ytr_t) + l2 * w.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        a_tr = auc((Xtr @ w + b).numpy(), ytr)
        a_va = auc((Xva @ w + b).numpy(), yva)
    print(f"[probe] {name:10s} dim={Xtr.shape[1]:4d}  "
          f"train AUC={a_tr:.4f}  VAL AUC={a_va:.4f}", flush=True)
    return {"dim": int(Xtr.shape[1]), "train_auc": round(a_tr, 4),
            "val_auc": round(a_va, 4)}


def main():
    z = np.load(PCA_NPZ)
    W, mean = z["W"], z["mean"].astype(np.float32)
    ev = z["eigenvalues"].astype(np.float64)
    evr256 = float(ev[:256].sum() / ev.sum())
    X, y, is_tr = collect()
    print(f"[data] n={len(y)}  train={int(is_tr.sum())}  val={int((~is_tr).sum())}  "
          f"pos_rate={y.mean():.3f}", flush=True)
    Xc = X - mean
    feats = {"full4096": X, "block1024": X[:, BLOCK_LO:].copy(),
             "pca256": Xc @ W.T}
    res = {"explained_var_top256": round(evr256, 4),
           "n_train": int(is_tr.sum()), "n_val": int((~is_tr).sum()),
           "seed": SEED, "per_tile_per_class": PER, "probes": {}}
    for name, F in feats.items():
        res["probes"][name] = probe(F[is_tr], y[is_tr], F[~is_tr], y[~is_tr], name)
    out = os.path.join(HERE, "results", "linear_probe.json")
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
