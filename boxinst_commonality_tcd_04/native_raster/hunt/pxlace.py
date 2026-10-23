"""LACE posterior at PIXEL resolution (training-free, same masker npz, same knobs).

Deployed LACE: E-step per 8 px feature cell -> cell posterior -> bilinear upsample -> colour-guided filter -> tau.
Here: bilinearly interpolate the PROJECTED, whitened, unit-normalised features (128-d) to every pixel of the box
clip, renormalise, and run LACE's own E-step per pixel (vMF prototypes C, background mixture Gbg, the box-normalised
spatial prior pi at the pixel's (u,v) bin, prior weight alpha, kappa*kappa_scale). The within-box contrast
centring uses the mean appearance log-ratio of the box's CELLS, exactly as box_mask, so a pixel lying on a cell
centre reproduces that cell's posterior. Optional colour-joint weighting of the interpolation (`sigma_c`): the four
neighbouring cells are weighted by bilinear weight x exp(-|rgb_pixel - rgb_cell|^2 / 2 sigma_c^2), rgb_cell = the
mean colour of the cell's 8x8 footprint (joint-bilateral feature upsampling).

pixel_post(...) -> (clip (y0,y1,x0,x1), P (clip-shaped float32 posterior; 0 outside the box clip)).
"""
import numpy as np
from scipy.special import logsumexp


def _cell_rgb_mean(rgbf, s, origin, g):
    """Mean RGB over each cell's s x s footprint centred at (i*s + origin) (continuous coords) -> (g, g, 3)."""
    from scipy.ndimage import uniform_filter
    k = s
    m = np.stack([uniform_filter(rgbf[..., c], size=k, mode="nearest") for c in range(3)], -1)
    # pixel index of a cell centre at continuous coord i*s + origin is the pixel whose span contains it
    ix = np.clip((np.arange(g) * s + origin).astype(int), 0, rgbf.shape[0] - 1)
    return m[ix][:, ix]


class PixelLACE:
    def __init__(self, masker):
        self.m = masker
        self.k = masker.kappa * masker.kappa_scale
        self.logwbg = np.log(masker.wbg)

    def appearance(self, z, pis, k):
        zc = k * (z @ self.m.C.T)
        psum = np.clip(pis.sum(0), 1e-4, 1 - 1e-4)
        lw = zc + np.log((pis / psum).T + 1e-9)
        bgll = logsumexp(self.logwbg[None] + k * (z @ self.m.Gbg.T), 1)
        return logsumexp(lw, 1) - bgll, psum

    def pixel_post(self, Z, g, box, alpha=None, kappa_scale=None, sigma_c=None, rgbf=None, cell_rgb=None):
        m = self.m
        alpha = m.prior_weight if alpha is None else alpha
        k = m.kappa * (m.kappa_scale if kappa_scale is None else kappa_scale)
        s, o, NB = m.s, m.origin, m.NB
        x0, y0, x1, y1 = [float(v) for v in box]
        bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
        sb = m.size_bin(box)
        # --- cell-level appearance mean (identical cell set to box_mask)
        cy, cx = np.mgrid[0:g, 0:g] * s + o
        pad = s / 2.0
        sel = (cx >= x0 - pad) & (cx < x1 + pad) & (cy >= y0 - pad) & (cy < y1 + pad)
        idx = np.flatnonzero(sel.ravel())
        if len(idx) == 0:
            d2 = (cx - (x0 + x1) / 2) ** 2 + (cy - (y0 + y1) / 2) ** 2
            idx = np.array([int(d2.ravel().argmin())])
        u = np.clip((cx.ravel()[idx] - x0) / bw, 0, 1); v = np.clip((cy.ravel()[idx] - y0) / bh, 0, 1)
        bu = np.minimum((u * NB).astype(int), NB - 1); bv = np.minimum((v * NB).astype(int), NB - 1)
        Zf = Z.reshape(g * g, -1)
        Ac, _ = self.appearance(Zf[idx], m.pi[sb][:, bv, bu], k)
        Abar = Ac.mean() if len(Ac) > 1 else 0.0
        # --- pixels of the box clip
        py0, py1, px0, px1 = max(0, int(y0)), int(np.ceil(y1)), max(0, int(x0)), int(np.ceil(x1))
        py1, px1 = min(py1, 2048), min(px1, 2048)
        H, W = py1 - py0, px1 - px0
        if H <= 0 or W <= 0:
            return (py0, py1, px0, px1), np.zeros((max(H, 0), max(W, 0)), np.float32)
        yc = np.arange(py0, py1) + 0.5; xc = np.arange(px0, px1) + 0.5
        fy = np.clip((yc - o) / s, 0, g - 1); fx = np.clip((xc - o) / s, 0, g - 1)
        iy0 = np.minimum(np.floor(fy).astype(int), g - 2); ix0 = np.minimum(np.floor(fx).astype(int), g - 2)
        wy = (fy - iy0)[:, None]; wx = (fx - ix0)[None, :]
        Y0, X0 = iy0[:, None], ix0[None, :]
        corners = [(Y0, X0, (1 - wy) * (1 - wx)), (Y0, X0 + 1, (1 - wy) * wx),
                   (Y0 + 1, X0, wy * (1 - wx)), (Y0 + 1, X0 + 1, wy * wx)]
        D = Z.shape[-1]
        zp = np.zeros((H, W, D), np.float32); wsum = np.zeros((H, W), np.float32)
        if sigma_c is not None:
            pix = rgbf[py0:py1, px0:px1]
        for (Yi, Xi, w) in corners:
            Yb, Xb = np.broadcast_to(Yi, (H, W)), np.broadcast_to(Xi, (H, W))
            w = np.broadcast_to(w, (H, W)).astype(np.float32)
            if sigma_c is not None:
                dc = ((pix - cell_rgb[Yb, Xb]) ** 2).sum(-1)
                w = w * np.exp(-dc / (2 * sigma_c ** 2)).astype(np.float32)
            zp += w[..., None] * Z[Yb, Xb]; wsum += w
        zp /= np.maximum(wsum, 1e-12)[..., None]
        zp = zp.reshape(H * W, D)
        zp /= np.linalg.norm(zp, axis=1, keepdims=True) + 1e-8
        uu = np.clip((xc - x0) / bw, 0, 1); vv = np.clip((yc - y0) / bh, 0, 1)
        bu = np.minimum((uu * NB).astype(int), NB - 1); bv = np.minimum((vv * NB).astype(int), NB - 1)
        BV, BU = np.meshgrid(bv, bu, indexing="ij")
        pis = m.pi[sb][:, BV.ravel(), BU.ravel()]
        A, psum = self.appearance(zp, pis, k)
        P = 1.0 / (1.0 + np.exp(-((A - Abar) + alpha * np.log(psum / (1 - psum)))))
        return (py0, py1, px0, px1), P.reshape(H, W).astype(np.float32)
