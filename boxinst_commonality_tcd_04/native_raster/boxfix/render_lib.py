"""Render variants on cached per-crown windows (cache_windows.py). Variant = dict of knobs:

  r        guided-filter radius: int, or "half" = min(12, max(1, floor(c_r * short side)))
  c_r      fraction of the box short side for r="half" (default 0.5)
  eps      guided-filter eps (default 1e-4)
  tau      absolute threshold (default 0.35)
  c_tau    if set: tau_box = min(tau, c_tau * peak of the filtered q inside the clip)
  knob     index into cache KNOBS (0 = deployed 0.3/1.6)
  knob_elong  knob index used when box aspect > elong_thr (default: same as knob)
  elong_thr   default 2.0
  rmax     cap for r="half" (default 12)
  fill     True: fill holes in the final mask (scipy binary_fill_holes)
  largest  True: keep the largest 8-connected component only
  se_p     if set: intersect with the box-inscribed superellipse |2dx/w|^p + |2dy/h|^p <= 1 (p=2 ellipse)

Deployed = {r: 12, eps: 1e-4, tau: 0.35, knob: 0}; must reproduce gtbox_lace_guided IoUs.
"""
import glob, os
import numpy as np

FC = "/Users/tompitts/dphil/feat_cache"
GF_R, GF_EPS, GF_TAU = 12, 1e-4, 0.35


def color_guided_filter(I, p, r, eps):
    from boxinst_commonality_tcd_04.native_raster.guided_ablate_modal import color_guided_filter as f
    return f(I, p, r, eps)


def load_tile(fp):
    d = np.load(fp)
    n = len(d["box"])
    crowns = []
    for i in range(n):
        wy0, wy1, wx0, wx1 = d["win"][i]
        h, w = wy1 - wy0, wx1 - wx0
        gw = np.unpackbits(d[f"gt_{i}"])[:h * w].reshape(h, w).astype(bool)
        crowns.append(dict(box=d["box"][i], clip=d["clip"][i], win=d["win"][i], valid=bool(d["valid"][i]),
                           gt_area=int(d["gt_area"][i]), gt_out=int(d["gt_out"][i]), gt=gw,
                           rgb=d[f"rgb_{i}"].astype(np.float32) / 255.0,
                           p=[d[f"p{k}_{i}"] for k in range(len(d["knobs"]))]))
    return crowns


def render(c, v):
    by0, by1, bx0, bx1 = c["clip"]; wy0, _, wx0, _ = c["win"]
    x0, y0, x1, y1 = [float(t) for t in c["box"]]
    bw, bh = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    knob = v.get("knob", 0)
    if max(bw, bh) / min(bw, bh) > v.get("elong_thr", 2.0):
        knob = v.get("knob_elong", knob)
    p = c["p"][knob]
    r = v.get("r", GF_R)
    if r == "half":
        r = int(min(v.get("rmax", GF_R), max(1, np.floor(v.get("c_r", 0.5) * min(bw, bh)))))
    eps, tau = v.get("eps", GF_EPS), v.get("tau", GF_TAU)
    q = p if r == 0 else color_guided_filter(c["rgb"], p, r, eps)
    cm = np.zeros(q.shape, bool); cm[by0 - wy0:by1 - wy0, bx0 - wx0:bx1 - wx0] = True
    if v.get("c_tau") is not None:
        pk = float(q[cm].max()) if cm.any() else 0.0
        tau = min(tau, v["c_tau"] * pk)
    m = (q >= tau) & cm
    if v.get("largest") and m.any():
        from scipy.ndimage import label
        lab, n = label(m, structure=np.ones((3, 3)))
        if n > 1:
            m = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    if v.get("fill"):
        from scipy.ndimage import binary_fill_holes
        m = binary_fill_holes(m)
    if v.get("se_p") is not None:
        ys = np.arange(m.shape[0])[:, None] + wy0 + 0.5; xs = np.arange(m.shape[1])[None] + wx0 + 0.5
        p_ = v["se_p"]
        m = m & ((np.abs(2 * (xs - (x0 + x1) / 2) / bw) ** p_ + np.abs(2 * (ys - (y0 + y1) / 2) / bh) ** p_) <= 1)
    return m


def iou(c, m):
    inter = float((m & c["gt"]).sum()); union = float(m.sum()) + c["gt_area"] - inter
    return inter / union if union > 0 else 0.0


def tile_files(split):
    return sorted(glob.glob(os.path.join(FC, f"boxfix_{split}", "*.npz")))


def render_full(rgbf, prob, clip, box, v, margin=16):
    """Full-raster entry point: rgbf (2048,2048,3) float in [0,1], prob (2048,2048) posterior from GA._prob,
    clip (y0,y1,x0,x1). Builds the same window as GA._guided and calls render(); returns a (2048,2048) bool mask."""
    by0, by1, bx0, bx1 = clip
    wy0, wy1 = max(0, by0 - margin), min(2048, by1 + margin)
    wx0, wx1 = max(0, bx0 - margin), min(2048, bx1 + margin)
    c = {"clip": clip, "win": (wy0, wy1, wx0, wx1), "box": np.asarray(box, np.float32),
         "rgb": rgbf[wy0:wy1, wx0:wx1], "p": [prob[wy0:wy1, wx0:wx1]]}
    m = np.zeros((2048, 2048), bool); m[wy0:wy1, wx0:wx1] = render(c, v)
    return m
