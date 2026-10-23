"""Training-free per-detection mask SELECTION among a ladder of LACE renders (no GT).

The val oracle (results/b07_oracle.json) shows the LACE family contains SAM-beating masks per detection; this module
picks one per box with an intrinsic score. Candidates: colour-guided render of the deployed posterior (r, eps 1e-4)
thresholded at each tau in `taus`. Scores (higher = chosen), computed inside the box clip on the window:
  otsu    between-class variance of the filtered posterior q split by the mask: w1 w0 (mu1 - mu0)^2
  stab    stability: IoU(mask at tau - d, mask at tau + d)   (SAM's stability score, on LACE's posterior)
  edge    mean RGB gradient magnitude on the mask's inner boundary pixels
  colour  || mean RGB of a 2-px band inside the boundary - mean RGB of a 2-px band outside || / in-box RGB std
Ties -> the deployed tau (0.35) if present, else the first.
"""
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, sobel


def _boundary_bands(m, k=2):
    inner = m & ~binary_erosion(m, iterations=k)
    outer = binary_dilation(m, iterations=k) & ~m
    return inner, outer


def score(name, m, q, inb, rgb, grad, extra):
    mb = m & inb
    if mb.sum() == 0 or mb.sum() == inb.sum() and name != "stab":
        return -np.inf if mb.sum() == 0 else 0.0
    if name == "otsu":
        x = q[inb]; f = m[inb]
        w1 = f.mean(); w0 = 1 - w1
        return float(w1 * w0 * (x[f].mean() - x[~f].mean()) ** 2) if 0 < w1 < 1 else 0.0
    if name == "stab":
        lo, hi = extra
        u = (lo | hi).sum()
        return float((lo & hi).sum() / u) if u else 0.0
    inner, outer = _boundary_bands(mb)
    if name == "edge":
        return float(grad[inner].mean()) if inner.any() else 0.0
    if name == "colour":
        outer &= inb | ~inb
        if not inner.any() or not outer.any():
            return 0.0
        sd = rgb[inb].std(0).mean() + 1e-3
        return float(np.linalg.norm(rgb[inner].mean(0) - rgb[outer].mean(0)) / sd)
    raise ValueError(name)
