"""Per-patch crown probe: PCA-reduced frozen features ⊕ shadow → ridge probe.

DINOv3 features are PCA-reduced to k dims once (unsupervised, on the train pool;
labels never seen, confirmation split untouched) for speed + small-N regularisation.
Shadow channels are appended *un-reduced* so their weight is preserved. The probe
is closed-form ridge (stable, deterministic, ~ms/fit); AP measures patch ranking.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from shadow_prior.shadow_feature import compute_shadow_feature  # noqa: E402
from data import CACHE, GRID  # noqa: E402


def _pool16(a):
    return a.reshape(GRID, 16, GRID, 16).mean((1, 3))


def shadow_patches(rgb01, azimuth, cfg):
    feat = compute_shadow_feature(rgb01, azimuth, cfg)              # (k,512,512)
    pooled = np.stack([_pool16(feat[c]) for c in range(feat.shape[0])], 0)
    return pooled.reshape(pooled.shape[0], -1).T.astype(np.float32)  # (1024,k)


def _raw_patches(scene):
    """Per-patch mean+std of RGB -> (P, 6); a deliberately weak 'low-level' base."""
    rgb = np.load(os.path.join(CACHE, "rgb512", scene + ".npy")).astype(np.float32) / 255.0
    mean = np.stack([_pool16(rgb[..., c]) for c in range(3)], 0)
    sq = np.stack([_pool16(rgb[..., c] ** 2) for c in range(3)], 0)
    std = np.sqrt(np.clip(sq - mean ** 2, 0, None))
    return np.concatenate([mean.reshape(3, -1).T, std.reshape(3, -1).T], 1)


def load_base(recs, feat_tag, include_raw):
    """scene -> full (P, D) base patch matrix.

    feat_tag 'web'/'sat' -> DINOv3 features (+ raw if include_raw); 'raw' -> the
    6-dim low-level base only (for the delta-of-deltas: does shadow help on a weak
    base but not on DINOv3?).
    """
    out = {}
    for r in recs:
        if feat_tag == "raw":
            out[r.scene] = _raw_patches(r.scene)
            continue
        F = np.load(os.path.join(CACHE, f"feat_{feat_tag}", r.scene + ".npy")).astype(np.float32)
        base = F.reshape(F.shape[0], -1).T
        if include_raw:
            base = np.concatenate([base, _raw_patches(r.scene)], 1)
        out[r.scene] = base
    return out


def fit_pca(base_by_scene, scenes, k=128):
    X = np.concatenate([base_by_scene[s] for s in scenes], 0)
    mean = X.mean(0)
    Xc = X - mean
    C = Xc.T @ Xc / Xc.shape[0]
    vals, vecs = np.linalg.eigh(C)
    comp = vecs[:, ::-1][:, :min(k, vecs.shape[1])]   # top-k eigenvectors
    return {"mean": mean, "comp": comp}


def apply_pca(base, pca):
    return (base - pca["mean"]) @ pca["comp"]


def build_scene_cache(recs, base_by_scene, pca, cfg, az_c, az_s, target="occ"):
    out = {}
    for r in recs:
        red = apply_pca(base_by_scene[r.scene], pca).astype(np.float32)
        rgb01 = np.load(os.path.join(CACHE, "rgb512", r.scene + ".npy")).astype(np.float32) / 255.0
        out[r.scene] = {
            "base": red,
            "sc": shadow_patches(rgb01, az_c[r.scene], cfg),
            "ss": shadow_patches(rgb01, az_s[r.scene], cfg),
            "y": np.load(os.path.join(CACHE, target, r.scene + ".npy")).reshape(-1).astype(np.float32),
            "acq": r.acq,
        }
    return out


def assemble(sc_cache, scenes, rung):
    Xs, ys = [], []
    for s in scenes:
        d = sc_cache[s]
        if rung == "r1":
            X = d["base"]
        elif rung == "r2":
            X = np.concatenate([d["base"], d["sc"]], 1)
        else:  # r3
            X = np.concatenate([d["base"], d["ss"]], 1)
        Xs.append(X); ys.append(d["y"])
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)


def fit_ridge(X, y, l2=1.0):
    """Closed-form standardised ridge (least-squares classification), float64."""
    X = X.astype(np.float64)
    mu = X.mean(0); sd = X.std(0) + 1e-6
    Xs = (X - mu) / sd
    d = Xs.shape[1]
    A = Xs.T @ Xs + l2 * np.eye(d)
    rhs = Xs.T @ (y.astype(np.float64) - y.mean())
    w = np.linalg.solve(A, rhs)
    return {"w": w, "mu": mu, "sd": sd, "b": float(y.mean())}


def probe_scores(pr, X):
    return ((X.astype(np.float64) - pr["mu"]) / pr["sd"]) @ pr["w"] + pr["b"]


def average_precision(scores, labels):
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    l = labels[order]
    tp = np.cumsum(l); fp = np.cumsum(1 - l)
    prec = tp / (tp + fp)
    rec = tp / labels.sum()
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return float(np.sum((rec - rec_prev) * prec))
