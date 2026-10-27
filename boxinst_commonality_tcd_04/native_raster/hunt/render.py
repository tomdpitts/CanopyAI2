"""Render kinds for hunt/arms.py. Each: f(ctx, det, spec) -> (2048, 2048) bool mask.
ctx: rgbf (2048,2048,3) float, m (TCDMasker), PL (PixelLACE), GA, zn, g, Z.  det: box, clip (y0,y1,x0,x1), prob
(2048^2 bilinear cell posterior, deployed _prob), r (cell posteriors), cache (per-detection memo shared by arms).

  deployed     GA._guided (r12 / eps 1e-4 / tau 0.35)                          -- the gate
  guided       colour-guided filter of a seed posterior, params r, eps, tau, seed in {cell, px}
  colour_em    per-box 2-class Gaussian colour model seeded by the posterior (GrabCut-like, no graph cut):
               fg weights = seed inside the box clip, bg = (1 - seed) inside + ring outside (window margin 16);
               q = sigmoid(beta * logit(seed) + gamma * LLR_colour), refit n_iter times; optional Gaussian
               smoothing of the logit (sigma_s); mask = q >= tau, clipped to the box.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

MARGIN = 16


def _win(clip):
    by0, by1, bx0, bx1 = clip
    return max(0, by0 - MARGIN), min(2048, by1 + MARGIN), max(0, bx0 - MARGIN), min(2048, bx1 + MARGIN)


def _full(win, clip, mw):
    wy0, wy1, wx0, wx1 = win; by0, by1, bx0, bx1 = clip
    m = np.zeros((2048, 2048), bool); m[wy0:wy1, wx0:wx1] = mw
    c = np.zeros((2048, 2048), bool); c[by0:by1, bx0:bx1] = True
    return m & c


def seed_window(ctx, det, which):
    """Seed posterior over the window: 'cell' = deployed bilinear prob; 'px' = PixelLACE inside the clip, 0 outside."""
    key = ("seed", which)
    if key in det["cache"]:
        return det["cache"][key]
    win = _win(det["clip"]); wy0, wy1, wx0, wx1 = win
    if which == "cell":
        s = det["prob"][wy0:wy1, wx0:wx1]
    else:
        PL, Z, g = (ctx["PL4"], ctx["Z4"], 512) if which.startswith("px4") else (ctx["PL"], ctx["Z"], ctx["g"])
        sc = None
        if which.startswith("px4_jbu"):                       # 'px4_jbu<sigma>': colour-joint feature interpolation
            sc = float(which[len("px4_jbu"):])
            if "cell_rgb4" not in ctx:
                from boxinst_commonality_tcd_04.native_raster.hunt.pxlace import _cell_rgb_mean
                ctx["cell_rgb4"] = _cell_rgb_mean(ctx["rgbf"], 4, 8, 512)
        (py0, py1, px0, px1), P = PL.pixel_post(Z, g, det["box"], sigma_c=sc, rgbf=ctx["rgbf"], cell_rgb=ctx.get("cell_rgb4"))
        s = np.zeros((wy1 - wy0, wx1 - wx0), np.float32)
        if P.size:
            s[py0 - wy0:py1 - wy0, px0 - wx0:px1 - wx0] = P
    det["cache"][key] = (win, s)
    return win, s


def deployed(ctx, det, spec):
    return ctx["GA"]._guided(ctx["rgbf"], det["clip"], det["prob"])


def guided(ctx, det, spec):
    win, s = seed_window(ctx, det, spec.get("seed", "cell"))
    wy0, wy1, wx0, wx1 = win
    r = spec.get("r", 12)
    q = s if r == 0 else ctx["GA"].color_guided_filter(ctx["rgbf"][wy0:wy1, wx0:wx1], s, r, spec.get("eps", 1e-4))
    tau = spec.get("tau", 0.35)
    if "c_peak" in spec:
        by0, by1, bx0, bx1 = det["clip"]; qb = q[by0 - wy0:by1 - wy0, bx0 - wx0:bx1 - wx0]
        tau = spec["c_peak"] * (float(qb.max()) if qb.size else 0.0)
    return _full(win, det["clip"], q >= tau)


def _gauss_ll(X, w):
    sw = w.sum()
    if sw < 1e-6:
        return None
    mu = (w[:, None] * X).sum(0) / sw
    d = X - mu
    cov = (w[:, None, None] * (d[:, :, None] * d[:, None, :])).sum(0) / sw + 1e-4 * np.eye(X.shape[1])
    ic = np.linalg.inv(cov); _, ld = np.linalg.slogdet(cov)
    return -0.5 * (np.einsum("ni,ij,nj->n", d, ic, d) + ld)


def colour_em(ctx, det, spec):
    win, s = seed_window(ctx, det, spec.get("seed", "cell"))
    wy0, wy1, wx0, wx1 = win; by0, by1, bx0, bx1 = det["clip"]
    rgb = ctx["rgbf"][wy0:wy1, wx0:wx1]
    if spec.get("pre_r"):                                    # optional guided pre-filter of the seed
        s = ctx["GA"].color_guided_filter(rgb, s, spec["pre_r"], 1e-4)
    H, W = s.shape
    inb = np.zeros((H, W), bool); inb[by0 - wy0:by1 - wy0, bx0 - wx0:bx1 - wx0] = True
    X = rgb.reshape(-1, 3).astype(np.float64)
    sc = np.clip(s, 0.02, 0.98).astype(np.float64)
    prior = spec.get("beta", 1.0) * np.log(sc / (1 - sc)).ravel()
    q = sc.ravel().copy(); q[~inb.ravel()] = 0.0
    for _ in range(spec.get("n_iter", 2)):
        wf = np.where(inb.ravel(), q, 0.0); wb = np.where(inb.ravel(), 1 - q, spec.get("ring_w", 1.0))
        lf, lb = _gauss_ll(X, wf), _gauss_ll(X, wb)
        if lf is None or lb is None:
            break
        logit = prior + spec.get("gamma", 1.0) * (lf - lb)
        if spec.get("sigma_s"):
            logit = gaussian_filter(logit.reshape(H, W), spec["sigma_s"]).ravel()
        q = 1 / (1 + np.exp(-np.clip(logit, -30, 30))); q[~inb.ravel()] = 0.0
    return _full(win, det["clip"], q.reshape(H, W) >= spec.get("tau", 0.5))


def _boxfill(mask, clip):
    by0, by1, bx0, bx1 = clip
    a = max((by1 - by0) * (bx1 - bx0), 1)
    return float(mask[by0:by1, bx0:bx1].sum()) / a


def cond(ctx, det, spec):
    """Deployed mask unless it under-fills its box (fill < theta), then the `alt` render (any kind).
    Optional `max_short`: only boxes with short side < max_short are eligible."""
    key = "deployed_mask"
    if key not in det["cache"]:
        det["cache"][key] = deployed(ctx, det, spec)
    dm = det["cache"][key]
    x0, y0, x1, y1 = [float(v) for v in det["box"]]
    if min(x1 - x0, y1 - y0) >= spec.get("max_short", 1e9) or _boxfill(dm, det["clip"]) >= spec["theta"]:
        return dm
    alt = spec["alt"]
    return RENDER[alt["kind"]](ctx, det, alt)


def rhalf(ctx, det, spec):
    """Colour-guided render with radius min(12, max(1, floor(c_r * short side))), eps 1e-4, tau (default 0.35)."""
    x0, y0, x1, y1 = [float(v) for v in det["box"]]
    r = int(min(12, max(1, np.floor(spec.get("c_r", 0.4) * min(x1 - x0, y1 - y0)))))
    return guided(ctx, det, {"seed": "cell", "r": r, "eps": 1e-4, "tau": spec.get("tau", 0.35)})


def ellipse(ctx, det, spec):
    """Box-inscribed superellipse |2dx/w|^p + |2dy/h|^p <= 1 (p=2 ellipse), clipped to the box."""
    x0, y0, x1, y1 = [float(v) for v in det["box"]]; by0, by1, bx0, bx1 = det["clip"]; by1, bx1 = min(by1, 2048), min(bx1, 2048)
    w, h, p = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6), spec.get("p", 2.0)
    ys = np.arange(by0, by1)[:, None] + 0.5; xs = np.arange(bx0, bx1)[None] + 0.5
    m = np.zeros((2048, 2048), bool)
    m[by0:by1, bx0:bx1] = np.abs(2 * (xs - (x0 + x1) / 2) / w) ** p + np.abs(2 * (ys - (y0 + y1) / 2) / h) ** p <= 1
    return m


def quantile(ctx, det, spec):
    """Fill a fixed fraction `f` of the box with its highest colour-guided posterior pixels (radius rule c_r)."""
    x0, y0, x1, y1 = [float(v) for v in det["box"]]
    r = int(min(12, max(1, np.floor(spec.get("c_r", 0.4) * min(x1 - x0, y1 - y0)))))
    win, s = seed_window(ctx, det, "cell"); wy0, wy1, wx0, wx1 = win; by0, by1, bx0, bx1 = det["clip"]
    q = ctx["GA"].color_guided_filter(ctx["rgbf"][wy0:wy1, wx0:wx1], s, r, 1e-4)
    qb = q[by0 - wy0:by1 - wy0, bx0 - wx0:bx1 - wx0]
    if not qb.size:
        return np.zeros((2048, 2048), bool)
    thr = np.quantile(qb, 1 - spec["f"])
    return _full(win, det["clip"], q >= thr)


def post(ctx, det, spec):
    """Re-run the cell posterior with masker knobs (alpha, kappa_scale) for boxes with short side < max_short (others:
    deployed), then colour-guided render: r (int or "half" -> min(12, floor(c_r*short))), eps 1e-4, tau.
    Optional `theta`: only when the deployed mask fills < theta of the box (as `cond`)."""
    x0, y0, x1, y1 = [float(v) for v in det["box"]]; short = min(x1 - x0, y1 - y0)
    if "deployed_mask" not in det["cache"]:
        det["cache"]["deployed_mask"] = deployed(ctx, det, spec)
    dm = det["cache"]["deployed_mask"]
    if short >= spec.get("max_short", 1e9) or ("theta" in spec and _boxfill(dm, det["clip"]) >= spec["theta"]):
        return dm
    key = ("prob", spec.get("alpha", 0.3), spec.get("kappa", 1.6))
    if key not in det["cache"]:
        det["cache"][key] = ctx["GA"]._prob(ctx["m"], ctx["zn"], ctx["g"], det["box"], key[1], key[2])[1]
    prob = det["cache"][key]
    r = spec.get("r", 12)
    if r == "half":
        r = int(min(12, max(1, np.floor(spec.get("c_r", 0.4) * short))))
    win = _win(det["clip"]); wy0, wy1, wx0, wx1 = win
    q = prob[wy0:wy1, wx0:wx1] if r == 0 else ctx["GA"].color_guided_filter(ctx["rgbf"][wy0:wy1, wx0:wx1], prob[wy0:wy1, wx0:wx1], r, 1e-4)
    return _full(win, det["clip"], q >= spec.get("tau", 0.35))


def sam(ctx, det, spec):
    """Reference arm: SAM 3 zs per-subtile mask for this detection (hunt_sam_val_dets)."""
    from pycocotools import mask as mu
    r = ctx["sam"][det["i"]]
    return mu.decode({"size": r["size"], "counts": r["counts"].encode()}).astype(bool)


def post4(ctx, det, spec):
    """LACE posterior on the real-4px lattice (same masker npz, s=4, origin=8; knobs alpha/kappa default 0.3/1.6),
    bilinear to 2048 as _prob, then colour-guided render (r int or "half"; eps; tau). Optional `theta` / `max_short`
    gating against the deployed mask exactly as `post`. Optional `seed_mix` w: q = w*prob4 + (1-w)*prob8.
    Optional `guide` (controls, as guided_control_val.py): "blur" = constant guide, q = mean(mean(p)) with the filter's
    box (2r+1); "wrong" = colour guide from a different tile (ctx["rgbf_wrong"], next val tile in sorted order)."""
    x0, y0, x1, y1 = [float(v) for v in det["box"]]; short = min(x1 - x0, y1 - y0)
    if "theta" in spec or "max_short" in spec:
        if "deployed_mask" not in det["cache"]:
            det["cache"]["deployed_mask"] = deployed(ctx, det, spec)
        dm = det["cache"]["deployed_mask"]
        if short >= spec.get("max_short", 1e9) or ("theta" in spec and _boxfill(dm, det["clip"]) >= spec["theta"]):
            return dm
    key = ("prob4", spec.get("alpha", 0.3), spec.get("kappa", 1.6))
    if key not in det["cache"]:
        det["cache"][key] = ctx["GA"]._prob(ctx["m4"], ctx["zn4"], ctx["g4"], det["box"], key[1], key[2])
    clip, prob, _ = det["cache"][key]
    if "seed_mix" in spec:
        prob = spec["seed_mix"] * prob + (1 - spec["seed_mix"]) * det["prob"]
    r = spec.get("r", 12)
    if r == "half":
        r = int(min(spec.get("rmax", 12), max(1, np.floor(spec.get("c_r", 0.4) * short))))
    win = _win(clip); wy0, wy1, wx0, wx1 = win
    guide = spec.get("guide", "rgb")
    if r == 0:
        q = prob[wy0:wy1, wx0:wx1]
    elif guide == "blur":
        k = 2 * r + 1
        q = uniform_filter(uniform_filter(prob[wy0:wy1, wx0:wx1], size=k, mode="reflect"), size=k, mode="reflect")
    else:
        I = ctx["rgbf"] if guide == "rgb" else ctx["rgbf_wrong"]
        q = ctx["GA"].color_guided_filter(I[wy0:wy1, wx0:wx1], prob[wy0:wy1, wx0:wx1], r, spec.get("eps", 1e-4))
    tau = spec.get("tau", 0.35)
    if "c_peak" in spec:                                     # two-way box-relative threshold (as relpeak)
        by0, by1, bx0, bx1 = clip
        qb = q[by0 - wy0:by1 - wy0, bx0 - wx0:bx1 - wx0]
        tau = float(np.clip(spec["c_peak"] * (float(qb.max()) if qb.size else 0.0), spec.get("lo", 0.0), spec.get("hi", 1.0)))
    return _full(win, clip, q >= tau)


def recentre(ctx, det, spec):
    """Cell posterior with contrast centre spec['centre'] (recentre.py; 8 px cells, or 4 px with lattice=4), then
    colour-guided render r / eps / tau. Optional theta gating vs the deployed mask."""
    from boxinst_commonality_tcd_04.native_raster.hunt import recentre as RC
    if "theta" in spec:
        if "deployed_mask" not in det["cache"]:
            det["cache"]["deployed_mask"] = deployed(ctx, det, spec)
        dm = det["cache"]["deployed_mask"]
        if _boxfill(dm, det["clip"]) >= spec["theta"]:
            return dm
    four = spec.get("lattice", 8) == 4
    m, zn, g = (ctx["m4"], ctx["zn4"], ctx["g4"]) if four else (ctx["m"], ctx["zn"], ctx["g"])
    key = ("rc", spec.get("centre"), spec.get("ring_px", 16), spec.get("w", 0.5), four)
    if key not in det["cache"]:
        idx, r = RC.posterior(m, zn, g, det["box"], spec.get("centre", "box"), spec.get("ring_px", 16), spec.get("w", 0.5))
        det["cache"][key] = RC.prob_from_cells(m, g, idx, r, det["box"])
    clip, prob = det["cache"][key]
    x0, y0, x1, y1 = [float(v) for v in det["box"]]
    r = spec.get("r", 12)
    if r == "half":
        r = int(min(12, max(1, np.floor(spec.get("c_r", 0.4) * min(x1 - x0, y1 - y0)))))
    win = _win(clip); wy0, wy1, wx0, wx1 = win
    q = prob[wy0:wy1, wx0:wx1] if r == 0 else ctx["GA"].color_guided_filter(ctx["rgbf"][wy0:wy1, wx0:wx1], prob[wy0:wy1, wx0:wx1], r, 1e-4)
    return _full(win, clip, q >= spec.get("tau", 0.35))


def select(ctx, det, spec):
    """Per-detection tau selection (selectors_lace.py): q = colour-guided filter (r, eps 1e-4) of the deployed posterior;
    candidates q >= tau for tau in spec['taus']; pick by spec['score'] (otsu|stab|edge|colour); stab uses +-spec['d']."""
    from boxinst_commonality_tcd_04.native_raster.hunt import selectors_lace as SEL
    win, s = seed_window(ctx, det, "cell"); wy0, wy1, wx0, wx1 = win; by0, by1, bx0, bx1 = det["clip"]
    key = ("q", spec.get("r", 12))
    if key not in det["cache"]:
        det["cache"][key] = ctx["GA"].color_guided_filter(ctx["rgbf"][wy0:wy1, wx0:wx1], s, spec.get("r", 12), 1e-4)
    q = det["cache"][key]
    rgb = ctx["rgbf"][wy0:wy1, wx0:wx1]
    inb = np.zeros(q.shape, bool); inb[by0 - wy0:by1 - wy0, bx0 - wx0:bx1 - wx0] = True
    grad = None
    if spec["score"] == "edge":
        if "grad" not in det["cache"]:
            det["cache"]["grad"] = np.sqrt(sum(SEL.sobel(rgb[..., c], 0) ** 2 + SEL.sobel(rgb[..., c], 1) ** 2 for c in range(3)))
        grad = det["cache"]["grad"]
    best, bs = None, -np.inf
    taus = spec["taus"]; d = spec.get("d", 0.05)
    for tau in taus:
        m = (q >= tau) & inb
        extra = ((q >= tau - d) & inb, (q >= tau + d) & inb) if spec["score"] == "stab" else None
        sc = SEL.score(spec["score"], m, q, inb, rgb, grad, extra)
        if sc > bs + 1e-12 or (abs(sc - bs) <= 1e-12 and abs(tau - 0.35) < 1e-9):
            best, bs = m, sc
    if best is None:
        best = (q >= 0.35) & inb
    return _full(win, det["clip"], best)


def areamatch(ctx, det, spec):
    """Area-calibrated threshold: q = colour-guided filter (r, eps 1e-4) of a posterior (source 'cell' bilinear prob,
    or 'q' = q itself); target area = gain * sum over the box clip of that source (the posterior's expected crown
    area); tau* = the q value at which the in-box mask reaches the target area; tau* clamped to [lo, hi]."""
    win, s = seed_window(ctx, det, "cell"); wy0, wy1, wx0, wx1 = win; by0, by1, bx0, bx1 = det["clip"]
    key = ("q", spec.get("r", 12))
    if key not in det["cache"]:
        det["cache"][key] = ctx["GA"].color_guided_filter(ctx["rgbf"][wy0:wy1, wx0:wx1], s, spec.get("r", 12), 1e-4)
    q = det["cache"][key]
    qb = q[by0 - wy0:by1 - wy0, bx0 - wx0:bx1 - wx0]
    src = s[by0 - wy0:by1 - wy0, bx0 - wx0:bx1 - wx0] if spec.get("source", "cell") == "cell" else qb
    if not qb.size:
        return np.zeros((2048, 2048), bool)
    n = int(round(spec.get("gain", 1.0) * float(np.clip(src, 0, 1).sum())))
    v = np.sort(qb.ravel())[::-1]
    tau = v[min(max(n, 1), v.size) - 1] if n > 0 else np.inf
    tau = float(np.clip(tau, spec.get("lo", -np.inf), spec.get("hi", np.inf)))
    return _full(win, det["clip"], q >= tau)


def relpeak(ctx, det, spec):
    """Box-relative threshold, both directions: tau = clip(c * peak(q in box), lo, hi), q = colour-guided filter
    (r, eps 1e-4) of the deployed posterior. Optional theta-gated alternative as `cond` is done via kind cond."""
    win, s = seed_window(ctx, det, "cell"); wy0, wy1, wx0, wx1 = win; by0, by1, bx0, bx1 = det["clip"]
    key = ("q", spec.get("r", 12))
    if key not in det["cache"]:
        det["cache"][key] = ctx["GA"].color_guided_filter(ctx["rgbf"][wy0:wy1, wx0:wx1], s, spec.get("r", 12), 1e-4)
    q = det["cache"][key]
    qb = q[by0 - wy0:by1 - wy0, bx0 - wx0:bx1 - wx0]
    pk = float(qb.max()) if qb.size else 0.0
    tau = float(np.clip(spec["c"] * pk, spec.get("lo", 0.0), spec.get("hi", 1.0)))
    return _full(win, det["clip"], q >= tau)


def _disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def shape(ctx, det, spec):
    """Shape regularisation applied to a base render (spec['base'], any kind), inside the box clip:
      close     binary closing with a disk of radius max(1, round(c_close * box short side))
      fill      fill holes
      largest   keep the largest 8-connected component
      hull      convex hull of the (largest component of the) mask, rasterised, clipped to the box
      smooth    Gaussian blur (sigma = c_smooth * short) of the mask, re-thresholded at 0.5
    Steps run in the order listed in spec['ops']."""
    from scipy.ndimage import binary_closing, binary_fill_holes, label
    m = RENDER[spec["base"]["kind"]](ctx, det, spec["base"])
    by0, by1, bx0, bx1 = det["clip"]; by1, bx1 = min(by1, 2048), min(bx1, 2048)
    x0, y0, x1, y1 = [float(v) for v in det["box"]]; short = min(x1 - x0, y1 - y0)
    P = 8
    wy0, wy1, wx0, wx1 = max(0, by0 - P), min(2048, by1 + P), max(0, bx0 - P), min(2048, bx1 + P)
    w = m[wy0:wy1, wx0:wx1].copy()
    if not w.any():
        return m
    for op in spec["ops"]:
        if op == "close":
            r = max(1, int(round(spec.get("c_close", 0.1) * short)))
            w = binary_closing(np.pad(w, r), structure=_disk(r))[r:-r, r:-r]
        elif op == "fill":
            w = binary_fill_holes(w)
        elif op == "largest":
            lab, n = label(w, structure=np.ones((3, 3)))
            if n > 1:
                w = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
        elif op == "hull":
            from scipy.spatial import ConvexHull
            from PIL import Image, ImageDraw
            ys, xs = np.nonzero(w)
            if len(ys) >= 3:
                pts = np.stack([xs, ys], 1).astype(float)
                pts = np.concatenate([pts + d for d in ([0, 0], [1, 0], [0, 1], [1, 1])])
                try:
                    hv = pts[ConvexHull(pts).vertices]
                    img = Image.new("L", (w.shape[1], w.shape[0]), 0)
                    ImageDraw.Draw(img).polygon([tuple(p) for p in hv], fill=1)
                    w = w | np.asarray(img, bool)
                except Exception:
                    pass
        elif op == "smooth":
            w = gaussian_filter(w.astype(np.float32), max(0.5, spec.get("c_smooth", 0.05) * short)) >= 0.5
    out = np.zeros((2048, 2048), bool); out[wy0:wy1, wx0:wx1] = w
    c = np.zeros((2048, 2048), bool); c[by0:by1, bx0:bx1] = True
    return out & c


def tta(ctx, det, spec):
    """Per-tile test-time EM (tta.py) on this tile's detections (adapted masker cached per tile and config), then a
    base render on the adapted posterior: spec['render'] in {'deployed','relpeak'} (+ 'close' c_close optional)."""
    from boxinst_commonality_tcd_04.native_raster.hunt import tta as TTA
    cfg = (spec.get("lam", 200.0), spec.get("n_iter", 2), spec.get("top"), spec.get("adapt_bg", True))
    store = ctx.setdefault("tta_models", {})
    if cfg not in store:
        store[cfg] = TTA.adapt(ctx["m"], ctx["zn"], ctx["g"], ctx["boxes"], ctx["scores"], lam=cfg[0], n_iter=cfg[1], top=cfg[2], adapt_bg=cfg[3])
    mm = store[cfg]
    key = ("tta_prob", cfg)
    if key not in det["cache"]:
        det["cache"][key] = ctx["GA"]._prob(mm, ctx["zn"], ctx["g"], det["box"], None, None)
    clip, prob, _ = det["cache"][key]
    win = _win(clip); wy0, wy1, wx0, wx1 = win; by0, by1, bx0, bx1 = clip
    q = ctx["GA"].color_guided_filter(ctx["rgbf"][wy0:wy1, wx0:wx1], prob[wy0:wy1, wx0:wx1], 12, 1e-4)
    tau = 0.35
    if spec.get("render") == "relpeak":
        qb = q[by0 - wy0:by1 - wy0, bx0 - wx0:bx1 - wx0]
        tau = spec.get("c", 0.4) * (float(qb.max()) if qb.size else 0.0)
    return _full(win, clip, q >= tau)


def size_gate(ctx, det, spec):
    """spec['small'] render for boxes with short side < T (px), spec['large'] otherwise."""
    x0, y0, x1, y1 = [float(v) for v in det["box"]]
    sub = spec["small"] if min(x1 - x0, y1 - y0) < spec["T"] else spec["large"]
    return RENDER[sub["kind"]](ctx, det, sub)


def ensemble(ctx, det, spec):
    """Training-free ensemble of LACE renders (spec['members'], any kinds), inside the box clip:
      vote    pixel majority: mean of member masks >= spec.get('frac', 0.5)
      medoid  the member mask with the highest summed IoU to the other members"""
    ms = [RENDER[m["kind"]](ctx, det, m) for m in spec["members"]]
    by0, by1, bx0, bx1 = det["clip"]; by1, bx1 = min(by1, 2048), min(bx1, 2048)
    P = 4
    wy0, wy1, wx0, wx1 = max(0, by0 - P), min(2048, by1 + P), max(0, bx0 - P), min(2048, bx1 + P)
    W = [m[wy0:wy1, wx0:wx1] for m in ms]
    if spec["mode"] == "vote":
        w = np.mean(W, 0) >= spec.get("frac", 0.5)
    else:
        def iou(a, b):
            u = (a | b).sum(); return (a & b).sum() / u if u else 1.0
        sc = [sum(iou(W[i], W[j]) for j in range(len(W)) if j != i) for i in range(len(W))]
        w = W[int(np.argmax(sc))]
    out = np.zeros((2048, 2048), bool); out[wy0:wy1, wx0:wx1] = w
    return out


def rebox(ctx, det, spec):
    """Self-consistent box: render mask0 with spec['base'] on the detector box; refit the box to mask0's extent,
    expanded by e_frac * (w, h) + e_px on every side (image-clipped; if spec['within'] also intersected with the
    detector box); re-run spec['final'] (default = base) on the refit box. n_iter rounds. Empty mask0 -> base mask.
    The final mask is clipped to the refit box (and so may extend past the detector box unless 'within')."""
    base = spec["base"]; final = spec.get("final", base)
    m = RENDER[base["kind"]](ctx, det, base)
    b0 = np.asarray(det["box"], np.float64); cur = det
    for _ in range(spec.get("n_iter", 1)):
        by0, by1, bx0, bx1 = cur["clip"]; by1, bx1 = min(by1, 2048), min(bx1, 2048)
        sub = m[by0:by1, bx0:bx1]
        if not sub.any():
            return m
        ys, xs = np.nonzero(sub)
        x0, x1, y0, y1 = xs.min() + bx0, xs.max() + 1 + bx0, ys.min() + by0, ys.max() + 1 + by0
        w, h = x1 - x0, y1 - y0
        ex, ey = spec.get("e_frac", 0.1) * w + spec.get("e_px", 2), spec.get("e_frac", 0.1) * h + spec.get("e_px", 2)
        nb = np.array([max(0, x0 - ex), max(0, y0 - ey), min(2048, x1 + ex), min(2048, y1 + ey)])
        if spec.get("within"):
            nb = np.array([max(nb[0], b0[0]), max(nb[1], b0[1]), min(nb[2], b0[2]), min(nb[3], b0[3])])
        if nb[2] - nb[0] < 2 or nb[3] - nb[1] < 2:
            return m
        nb = nb.astype(np.float32)
        clip, prob, r = ctx["GA"]._prob(ctx["m"], ctx["zn"], ctx["g"], nb, ctx["GA"].ALPHA, ctx["GA"].KAPPA)
        cur = {"i": det["i"], "box": nb, "clip": clip, "prob": prob, "r": r, "cache": {}}
        m = RENDER[final["kind"]](ctx, cur, final)
    return m


def combine(ctx, det, spec):
    """Set algebra over member renders: op 'union' or 'inter' (left fold over spec['members'])."""
    ms = [RENDER[m["kind"]](ctx, det, m) for m in spec["members"]]
    out = ms[0].copy()
    for m in ms[1:]:
        out = (out | m) if spec["op"] == "union" else (out & m)
    return out


class _PadMasker:
    """TCDMasker view whose box_mask also scores cells up to `extra` px OUTSIDE the box (default pad s/2 + extra);
    the spatial prior is clamped to the box-edge bins there (as box_mask already does for its pad cells); the
    contrast mean is still taken over ALL scored cells (box_mask semantics)."""
    def __init__(self, m, extra):
        self.m, self.extra = m, extra
        self.s, self.origin = m.s, m.origin

    def box_mask(self, zn, g, box, contrast=True, prior_weight=None, kappa_scale=None):
        from boxinst_commonality.em import estep
        m = self.m
        prior_weight = m.prior_weight if prior_weight is None else prior_weight
        kappa_scale = m.kappa_scale if kappa_scale is None else kappa_scale
        cy, cx = np.mgrid[0:g, 0:g] * m.s + m.origin
        x0, y0, x1, y1 = box; pad = m.s / 2.0 + self.extra
        sel = (cx >= x0 - pad) & (cx < x1 + pad) & (cy >= y0 - pad) & (cy < y1 + pad)
        idx = np.flatnonzero(sel.ravel())
        if len(idx) == 0:
            d2 = (cx - (x0 + x1) / 2) ** 2 + (cy - (y0 + y1) / 2) ** 2
            idx = np.array([int(d2.ravel().argmin())])
        u = np.clip((cx.ravel()[idx] - x0) / max(x1 - x0, 1), 0, 1); v = np.clip((cy.ravel()[idx] - y0) / max(y1 - y0, 1), 0, 1)
        bu = np.minimum((u * m.NB).astype(int), m.NB - 1); bv = np.minimum((v * m.NB).astype(int), m.NB - 1)
        k = m.kappa * kappa_scale
        pfg, _ = estep(zn[idx], m.pi[m.size_bin(box)][:, bv, bu], k, m.C, m.bg_ll(zn[idx], kappa=k), contrast, prior_weight=prior_weight)
        return idx, pfg


def post4pad(ctx, det, spec):
    """4 px posterior scored `extra` px beyond the box (prior clamped at the edge), colour-guided r12, tau = c*peak
    (peak over the ORIGINAL box clip), mask clipped to the box grown by `grow` px."""
    pm = _PadMasker(ctx["m4"], spec["extra"])
    key = ("p4pad", spec["extra"])
    if key not in det["cache"]:
        det["cache"][key] = ctx["GA"]._prob(pm, ctx["zn4"], ctx["g4"], det["box"], 0.3, 1.6)
    clip0, prob, _ = det["cache"][key]
    gpx = spec.get("grow", spec["extra"])
    clip = (max(0, clip0[0] - gpx), min(2048, clip0[1] + gpx), max(0, clip0[2] - gpx), min(2048, clip0[3] + gpx))
    win = _win(clip); wy0, wy1, wx0, wx1 = win
    q = ctx["GA"].color_guided_filter(ctx["rgbf"][wy0:wy1, wx0:wx1], prob[wy0:wy1, wx0:wx1], spec.get("r", 12), 1e-4)
    by0, by1, bx0, bx1 = clip0; qb = q[by0 - wy0:by1 - wy0, bx0 - wx0:bx1 - wx0]
    tau = spec.get("c_peak", 0.33) * (float(qb.max()) if qb.size else 0.0)
    return _full(win, clip, q >= tau)


RENDER = {"deployed": deployed, "guided": guided, "colour_em": colour_em, "cond": cond, "rhalf": rhalf, "ellipse": ellipse, "quantile": quantile, "post": post, "sam": sam, "post4": post4, "recentre": recentre, "select": select, "areamatch": areamatch, "relpeak": relpeak, "shape": shape, "tta": tta, "size_gate": size_gate, "ensemble": ensemble, "rebox": rebox, "combine": combine, "post4pad": post4pad}
