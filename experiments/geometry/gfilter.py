"""The deterministic crown->shadow geometric filter (zero learned parameters).

Given a candidate at (cx,cy) and the image-frame shadow-displacement direction
u = azimuth_to_vector(azimuth) = (u_row, u_col), the filter samples standardised
luminance along the anti-solar ray (crown -> shadow, +u) and the solar ray
(crown -> sun, -u) over a frozen offset bracket, staying inside the valid mask.

Two paired cues, combined into one signed contrast:
  * shadow-presence  : darkness along +u  (a real crown casts a dark shadow there)
  * self-shadow      : darkness along -u  (a cast-shadow FP has its dark crown there)
The signed contrast

    delta = mean_lum(+u) - mean_lum(-u)

is < 0 for a real crown (shadow ahead, lit behind) and > 0 for a shadow-FP (crown
behind, lit soil ahead) -- domain-agnostic, so it also holds for arid tiles where
BOTH crown and shadow are dark on bright soil. The geometry keep-probability is

    g = expit(-K * delta)  in (0,1)

and re-scoring is multiplicative: final = raw_score * g. The NONE arm uses g=1.
Nothing here is fit to labels; the bracket and K are frozen constants below.
"""
from __future__ import annotations

import os, sys
import numpy as np
from scipy.special import expit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.geometry.gdata import luminance
from shadow_prior.geometry import azimuth_to_vector

# Frozen filter constants (set once; never tuned on the arm outcomes).
D_MIN, D_MAX, D_STEPS = 15.0, 60.0, 10   # anti-solar/solar ray offset bracket (px)
K_GAIN = 2.0                              # logistic gain mapping delta -> keep-prob
MIN_VALID_FRAC = 0.4                      # need this frac of offsets valid on a side


def standardize(f):
    med = np.median(f)
    mad = np.median(np.abs(f - med)) * 1.4826
    scale = mad if mad > 1e-6 else (f.std() + 1e-6)
    return (f - med) / scale


def _sample_ray(zlum, valid, cx, cy, u_row, u_col, offs):
    """Mean standardised luminance along +offs*u from (cx,cy), valid pixels only.
    Returns (mean, frac_valid)."""
    H, W = zlum.shape
    vals = []
    for d in offs:
        rr, cc = cy + d * u_row, cx + d * u_col
        r0, c0 = int(np.floor(rr)), int(np.floor(cc))
        if r0 < 0 or c0 < 0 or r0 + 1 >= H or c0 + 1 >= W:
            continue
        if not (valid[r0, c0] and valid[r0 + 1, c0] and valid[r0, c0 + 1] and valid[r0 + 1, c0 + 1]):
            continue
        fr, fc = rr - r0, cc - c0
        v = (zlum[r0, c0] * (1 - fr) * (1 - fc) + zlum[r0 + 1, c0] * fr * (1 - fc)
             + zlum[r0, c0 + 1] * (1 - fr) * fc + zlum[r0 + 1, c0 + 1] * fr * fc)
        vals.append(v)
    frac = len(vals) / len(offs)
    return (float(np.mean(vals)) if vals else np.nan), frac


class ShadowGeometry:
    """Precompute the standardised luminance field once per image, then score many
    candidates cheaply against a chosen azimuth."""

    def __init__(self, rgb, valid):
        self.z = standardize(luminance(rgb))
        self.valid = valid
        self.offs = np.linspace(D_MIN, D_MAX, D_STEPS)

    def score(self, cx, cy, azimuth):
        """Return dict(delta, ahead, behind, g, ok). g in (0,1] keep-probability.

        ok=False when neither ray has enough valid samples (candidate too close to
        padding/edge); caller should treat g as neutral (1.0) so the filter never
        *invents* a rejection from missing data."""
        u_row, u_col = azimuth_to_vector(azimuth)
        ahead, fa = _sample_ray(self.z, self.valid, cx, cy, u_row, u_col, self.offs)   # +u anti-solar
        behind, fb = _sample_ray(self.z, self.valid, cx, cy, -u_row, -u_col, self.offs)  # -u solar
        ok = (fa >= MIN_VALID_FRAC) and (fb >= MIN_VALID_FRAC)
        if not ok or np.isnan(ahead) or np.isnan(behind):
            return {"delta": np.nan, "ahead": ahead, "behind": behind, "g": 1.0, "ok": False}
        delta = ahead - behind
        g = float(expit(-K_GAIN * delta))
        return {"delta": float(delta), "ahead": float(ahead), "behind": float(behind),
                "g": g, "ok": True}
