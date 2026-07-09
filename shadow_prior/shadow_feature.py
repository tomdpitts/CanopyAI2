"""Swept directional matched filter -- the directional shadow prior.

This module implements design decision #1: because we have **azimuth only** (no
solar elevation, no timestamps), the shadow displacement
``d = h / tan(elevation)`` is unknown. We therefore treat crown height as a
nuisance and *marginalise over shadow length by sweeping the offset* rather than
computing it. Per pixel the response aggregates, over a bracket of offsets ``s``
along the annotated azimuth ``φ``, a "bright-here AND dark-at-offset-s" score.

Everything here is a pure function of ``(rgb, azimuth_rad, cfg)`` with no I/O, so
it is fully unit-testable and -- critically -- cheap to recompute from the rotated
azimuth at augmentation time (decision #2: never rotate a precomputed raster).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import shift as ndi_shift
from scipy.special import logsumexp, expit

from .config import ShadowFeatureConfig
from .geometry import azimuth_to_vector

_MAD_TO_SIGMA = 1.4826  # MAD -> std for a normal distribution


# --------------------------------------------------------------------------- #
# Brightness proxies
# --------------------------------------------------------------------------- #
def _luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.601 luma. ``rgb`` is HxWx3, any float range."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _greenness(rgb: np.ndarray) -> np.ndarray:
    """Normalised excess green: (2G - R - B). Higher over lit canopy.

    Offered as an alternative "is this canopy lit?" field because for vegetation
    the lit/shaded contrast can be stronger in a greenness channel than in luma.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 2.0 * g - r - b


_PROXIES = {"luminance": _luminance, "greenness": _greenness}


def brightness_z(rgb: np.ndarray, cfg: ShadowFeatureConfig) -> np.ndarray:
    """Robustly **standardise** the brightness proxy to a clipped z-score field.

    Centre = median, scale = MAD (normal-consistent), with a std fallback when the
    MAD is degenerate (a near-flat tile whose majority pixels share one value, so
    >50% have zero deviation). We standardise rather than min-max/percentile rescale
    because a rescale collapses to a constant whenever the bright/dark features
    cover a small fraction of the tile -- exactly the small-crown regime of the
    sparse-rangeland target, where the feature must still fire. The result is
    clipped to ``+/- cfg.z_clip`` so a few specular/black pixels cannot dominate.
    Returns all-zeros for a genuinely constant tile (no contrast to exploit).
    """
    field = _PROXIES[cfg.brightness_proxy](rgb.astype(np.float64, copy=False))
    median = np.median(field)
    mad = np.median(np.abs(field - median)) * _MAD_TO_SIGMA
    scale = mad if mad > 1e-8 else float(field.std())
    if scale <= 1e-8:
        return np.zeros_like(field)
    z = (field - median) / scale
    return np.clip(z, -cfg.z_clip, cfg.z_clip)


# --------------------------------------------------------------------------- #
# Offset displacement
# --------------------------------------------------------------------------- #
def _shift_along(
    field: np.ndarray, s: float, u_row: float, u_col: float, cfg: ShadowFeatureConfig
) -> np.ndarray:
    """Return ``field`` sampled at ``p + s*(u_row, u_col)`` for every pixel ``p``.

    Uses :func:`scipy.ndimage.shift`, whose convention is
    ``out[idx] = in[idx - shift]``; so to read the value *ahead* by ``+s*u`` we
    pass ``shift = -s*u``. Sub-pixel offsets are interpolated (``cfg.shift_order``).
    """
    return ndi_shift(
        field,
        shift=(-s * u_row, -s * u_col),
        order=cfg.shift_order,
        mode=cfg.shift_mode,
    )


def _offsets(cfg: ShadowFeatureConfig) -> np.ndarray:
    return np.linspace(cfg.offset_min, cfg.offset_max, cfg.offset_steps)


def _aggregate(scores: np.ndarray, cfg: ShadowFeatureConfig) -> np.ndarray:
    """Collapse the per-offset score stack (axis 0) into one response map.

    ``"max"``        -> MAP over displacement (best single offset).
    ``"logsumexp"``  -> soft marginalisation over the nuisance displacement
                        (decision #1), temperature ``1/cfg.logsumexp_beta``.
    """
    if cfg.aggregation == "max":
        return scores.max(axis=0)
    # logsumexp_beta * scores then logsumexp / beta keeps the result on the same
    # scale as `scores` and recovers `max` as beta -> infinity.
    beta = cfg.logsumexp_beta
    return logsumexp(beta * scores, axis=0) / beta


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def compute_shadow_feature(
    rgb: np.ndarray, azimuth_rad: float, cfg: ShadowFeatureConfig
) -> np.ndarray:
    """Compute the directional shadow-prior feature stack.

    Parameters
    ----------
    rgb:
        ``HxWx3`` array (any float range; ints accepted and cast). Channel order
        R, G, B.
    azimuth_rad:
        Direction along which shadows are displaced, in the convention of
        :mod:`shadow_prior.geometry`. This is the (possibly rotation-updated)
        annotated azimuth ``φ`` -- never tuned, never per-pixel.
    cfg:
        :class:`~shadow_prior.config.ShadowFeatureConfig`.

    Returns
    -------
    np.ndarray
        ``C x H x W`` float64 stack with ``C == cfg.n_channels`` (1 or 2):

        * channel 0 -- **crown** response: ``lit(p) * dark(p + s*φ̂)`` aggregated
          over ``s``, where ``lit = expit(gain*z)`` and ``dark = expit(-gain*z)``
          are soft thresholds on the standardised brightness ``z``. High where a
          pixel is lit *and* a pixel an offset ``s`` ahead along ``φ`` is dark --
          the crown+cast-shadow signature.
        * channel 1 (optional) -- **shadow** response: the dual filter
          ``dark(p) * lit(p - s*φ̂)``. High where a pixel is dark *and* a pixel an
          offset behind (up-sun) is lit -- i.e. this pixel is itself in a cast
          shadow. We emit this rather than the arg-max offset, which would amount
          to computing shadow length and so violate decision #1.

    Notes
    -----
    The only loop is over ``cfg.offset_steps`` offsets; each iteration is a fully
    vectorised array shift (no per-pixel Python).
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be HxWx3, got shape {rgb.shape}")

    z = brightness_z(rgb, cfg)  # HxW standardised brightness
    lit = expit(cfg.contrast_gain * z)   # soft "is lit"  in (0, 1)
    dark = 1.0 - lit                     # soft "is dark" = expit(-gain*z)

    # Map the annotated azimuth to the *shadow displacement* direction. If the
    # annotation points at the sun, the shadow falls the opposite way (rotate by pi).
    # Adding pi commutes with the rotation update, so the recompute invariant holds.
    phi = azimuth_rad if cfg.azimuth_points_to == "shadow" else azimuth_rad + np.pi
    u_row, u_col = azimuth_to_vector(phi)
    offsets = _offsets(cfg)

    # Stack of "lit here AND dark ahead" scores, one per offset.
    crown_scores = np.empty((offsets.size, *z.shape), dtype=np.float64)
    want_shadow = cfg.n_channels == 2
    if want_shadow:
        shadow_scores = np.empty_like(crown_scores)

    for i, s in enumerate(offsets):
        dark_ahead = _shift_along(dark, s, u_row, u_col, cfg)   # dark(p + s*u)
        crown_scores[i] = lit * dark_ahead                      # lit here, dark ahead
        if want_shadow:
            lit_behind = _shift_along(lit, -s, u_row, u_col, cfg)  # lit(p - s*u)
            shadow_scores[i] = dark * lit_behind                   # dark here, lit behind

    crown = _aggregate(crown_scores, cfg)
    if not want_shadow:
        return crown[np.newaxis, :, :]
    shadow = _aggregate(shadow_scores, cfg)
    return np.stack([crown, shadow], axis=0)


# --------------------------------------------------------------------------- #
# Rung-3 control: within-acquisition azimuth shuffle
# --------------------------------------------------------------------------- #
def shuffle_azimuths(
    azimuths: np.ndarray,
    acquisition_ids: np.ndarray,
    seed: int,
    avoid_global_identity: bool = True,
) -> np.ndarray:
    """Permute azimuth labels **within each acquisition group** (decision #3, rung 3).

    The rung-3 control must supply the same channel count and the same azimuth
    *distribution* as rung 2 but with the **direction de-correlated from the
    scene**. The permutation is stratified by acquisition so that a model cannot
    recover the true direction by first inferring acquisition identity (a global
    shuffle would leave that backdoor open -- see README "Why within-acquisition").

    Parameters
    ----------
    azimuths:
        1-D array of per-sample azimuths (radians).
    acquisition_ids:
        1-D array, same length, giving each sample's acquisition group.
    seed:
        RNG seed (logged for reproducibility).
    avoid_global_identity:
        If True, reject a draw whose *overall* result equals the input and redraw
        (bounded retries). This guards the common failure where the permutation is
        a no-op. NOTE: groups of size 1 are unshufflable -- those entries are
        necessarily unchanged regardless of this flag; that is a documented
        weakening of the control for singleton acquisitions, not a bug.

    Returns
    -------
    np.ndarray
        Permuted azimuths, same shape/order as ``azimuths``.
    """
    azimuths = np.asarray(azimuths)
    acquisition_ids = np.asarray(acquisition_ids)
    if azimuths.shape != acquisition_ids.shape or azimuths.ndim != 1:
        raise ValueError("azimuths and acquisition_ids must be 1-D and same length")

    rng = np.random.default_rng(seed)

    def _one_draw() -> np.ndarray:
        out = azimuths.copy()
        for acq in np.unique(acquisition_ids):
            idx = np.flatnonzero(acquisition_ids == acq)
            out[idx] = azimuths[idx][rng.permutation(idx.size)]
        return out

    out = _one_draw()
    if avoid_global_identity:
        shufflable = _has_shufflable_group(acquisition_ids)
        retries = 0
        while shufflable and np.array_equal(out, azimuths) and retries < 100:
            out = _one_draw()
            retries += 1
    return out


def _has_shufflable_group(acquisition_ids: np.ndarray) -> bool:
    """True if at least one acquisition has >= 2 members (so a non-identity exists)."""
    _, counts = np.unique(acquisition_ids, return_counts=True)
    return bool((counts >= 2).any())
