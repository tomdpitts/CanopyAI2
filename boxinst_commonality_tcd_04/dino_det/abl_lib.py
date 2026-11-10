"""Shared renderer for pre-registration ABL-D1 (README.md, 2026-11-10): lace4px_simple post4 with a chosen masker npz, E-step
recentring flag and inference scalars (alpha, kappa_scale), EDGE_FILL on. Mirrors native_raster/hunt/btmask/btmask.render_boxes
exactly (GA._prob at 4 px / origin 8, post4 c_peak 0.33, s' = s * mean|2r-1| * mean(r)); only the masker object and its
`contrast` flag can differ. The deployed setting (em_model_4p_fix.npz, contrast True, 0.3 / 1.6) must reproduce the cached
edge-fixed masks byte for byte (gated by the callers).
"""
import functools, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE)
FC = "/Users/tompitts/dphil/feat_cache"
DEPLOYED = f"{FC}/em_model_4p_fix.npz"
C_PEAK = 0.33
_W = {}


def masker(npz, contrast):
    key = (npz, bool(contrast))
    if key not in _W:
        from boxinst_commonality_tcd_04 import evaluate as E
        m = E.TCDMasker(npz); m.s, m.origin = 4, 8.0
        if not contrast:
            m.box_mask = functools.partial(type(m).box_mask, m, contrast=False)
        _W[key] = m
    return _W[key]


def _mods():
    if "GA" not in _W:
        from boxinst_commonality_tcd_04.native_raster import guided_ablate_modal as GA
        from boxinst_commonality_tcd_04.native_raster.hunt import render as R
        GA.EDGE_FILL = True
        _W.update(GA=GA, R=R)
    return _W["GA"], _W["R"]


def render(split, t, boxes, scores, npz=DEPLOYED, contrast=True, alpha=0.3, ks=1.6, as_rle=True):
    """-> (masks [rle dict or 2048 bool], s_prime list). boxes xyxy in 2048 px; scores = detector s (None -> s' None)."""
    from boxinst_commonality_tcd_04.native_raster.hunt.btmask import btmask as BM
    GA, R = _mods(); m = masker(npz, contrast)
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    if not len(boxes):
        return [], []
    ctx = {"rgbf": BM._rgb(split, t), "GA": GA, "m4": m, "g4": 512}
    ctx["zn4"] = np.load(f"{BM.SPLIT[split]['z4']}/{t}.npy").astype(np.float32).reshape(-1, 128)
    M, SP = [], []
    for i, b in enumerate(boxes):
        post = GA._prob(m, ctx["zn4"], 512, b, alpha, ks)
        det = {"i": i, "box": b, "cache": {("prob4", alpha, ks): post}}
        r = post[2]
        SP.append(round(float(scores[i] * np.abs(2 * r - 1).mean() * r.mean()), 8) if scores is not None else None)
        mk = R.post4(ctx, det, {"kind": "post4", "c_peak": C_PEAK, "alpha": alpha, "kappa": ks})
        M.append(GA._rle(mk) if as_rle else mk)
    return M, SP


def r3_heads():
    """The frozen R3 heads, refit exactly as rerank3.final (deterministic; callers gate against published R3 scores)."""
    from boxinst_commonality_tcd_04.dino_det import rerank as R, rerank3 as R3
    ch = json.load(open(os.path.join(HERE, "results", "rerankR3_final.json")))["chosen"]; c = {**ch, "cols": tuple(ch["cols"])}
    R3._init(); W = R3._W
    fits = [f for s in range(5) for f in R3.oof_fits(c, s, "rr", W["X"], W["y"], W["soft"], W["tile"], W["F"])[1]]; assert len(fits) == 25
    cols = [R.NAMES.index(x) for x in c["cols"]]

    def score(b, s, sp):
        b = np.asarray(b, np.float64).reshape(-1, 4); s = np.asarray(s, np.float64); sp = np.asarray(sp, np.float64)
        if not len(s):
            return np.zeros(0)
        X = np.zeros((len(s), R.NF)); h = np.maximum(b[:, 3] - b[:, 1], 1)
        X[:, 0], X[:, 1], X[:, 2], X[:, 4] = s, sp, np.log(np.maximum(sp, 1e-6) / np.maximum(s, 1e-6)), np.log(h)
        X[:, 12] = np.where(s[None, :] > s[:, None], R._box_iou(b, b), 0).max(1)
        return np.mean([f(X[:, cols]) for f in fits], 0)
    return score
