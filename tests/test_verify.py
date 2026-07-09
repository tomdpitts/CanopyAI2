"""Tests for the empirical azimuth-convention verifier."""

from __future__ import annotations

import math

import numpy as np

from shadow_prior.config import ShadowFeatureConfig
from shadow_prior.verify import recommend_convention, crown_response_in_mask


def _tile_and_crown_mask(size=64, azimuth_rad=0.0, offset=10.0, blob_radius=3.0):
    """Lit blob (the crown) + a dark shadow patch displaced anti-solar by `offset`.
    Returns (rgb, crown_mask) where crown_mask marks the lit blob."""
    img = np.full((size, size, 3), 0.5, dtype=np.float64)
    rr, cc = np.ogrid[:size, :size]
    cy = cx = size / 2.0
    crown_mask = (rr - cy) ** 2 + (cc - cx) ** 2 <= blob_radius**2
    img[crown_mask] = 1.0
    py = cy + offset * math.sin(azimuth_rad)
    px = cx + offset * math.cos(azimuth_rad)
    img[(rr - py) ** 2 + (cc - px) ** 2 <= 16] = 0.0
    return img, crown_mask


def test_recommend_convention_picks_shadow_for_antisolar_data():
    cfg = ShadowFeatureConfig(n_channels=1)
    samples = []
    for az in [0.0, 0.9, 2.3, 4.0]:
        rgb, mask = _tile_and_crown_mask(azimuth_rad=az)
        samples.append((rgb, az, mask))

    report = recommend_convention(samples, cfg)
    # The data was generated with the shadow anti-solar, so "shadow" must win.
    assert report.recommended == "shadow"
    assert report.score_shadow > report.score_sun
    assert report.n_samples == 4
    assert report.margin > 0.0


def test_recommend_convention_flips_when_annotation_points_at_sun():
    """If the *annotation* points at the sun (shadow falls opposite the labelled
    vector), the verifier should recommend 'sun'."""
    cfg = ShadowFeatureConfig(n_channels=1)
    samples = []
    for az in [0.0, 0.9, 2.3, 4.0]:
        # Place the shadow at az+pi but label the sample with az (sun-pointing label).
        rgb, mask = _tile_and_crown_mask(azimuth_rad=az + math.pi)
        samples.append((rgb, az, mask))

    report = recommend_convention(samples, cfg)
    assert report.recommended == "sun"


def test_crown_response_in_mask_empty_is_nan():
    cfg = ShadowFeatureConfig(n_channels=1)
    rgb, _ = _tile_and_crown_mask()
    empty = np.zeros(rgb.shape[:2], dtype=bool)
    assert math.isnan(crown_response_in_mask(rgb, 0.0, empty, cfg))
