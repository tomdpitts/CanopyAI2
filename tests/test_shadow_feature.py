"""Tests for the swept directional matched filter.

Covers the three properties called out in the task:

* a known bright-blob / dark-patch geometry yields a peak response at the blob
  (and, for the dual channel, at the shadow patch);
* the **recompute-not-rotate invariant** (decision #2): rotating tile *and*
  azimuth together gives the rotation of the original response (rotation
  equivariance);
* a regression lock that **fails if the feature raster is rotated instead of
  recomputed** -- in two ways: forgetting to update the azimuth, and the
  interpolation leak from rotating a precomputed raster.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.ndimage import rotate as ndi_rotate

from shadow_prior.config import ShadowFeatureConfig
from shadow_prior.geometry import rotate_azimuth, vector_to_azimuth, azimuth_to_vector
from shadow_prior.shadow_feature import compute_shadow_feature


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_tile_with_shadow(
    size: int = 64,
    blob_center=(32.0, 32.0),
    azimuth_rad: float = 0.0,
    offset: float = 10.0,
    blob_radius: float = 3.0,
    patch_radius: float = 4.0,
    bg: float = 0.5,
    blob_val: float = 1.0,
    patch_val: float = 0.0,
):
    """Grey tile with a bright circular blob and a dark patch displaced by
    ``offset`` pixels along ``azimuth_rad`` (the crown + cast-shadow signature).

    Returns ``(rgb, blob_center_rc, patch_center_rc)``.
    """
    img = np.full((size, size, 3), bg, dtype=np.float64)
    rr, cc = np.ogrid[:size, :size]
    by, bx = blob_center
    img[(rr - by) ** 2 + (cc - bx) ** 2 <= blob_radius**2] = blob_val
    u_row, u_col = math.sin(azimuth_rad), math.cos(azimuth_rad)
    py, px = by + offset * u_row, bx + offset * u_col
    img[(rr - py) ** 2 + (cc - px) ** 2 <= patch_radius**2] = patch_val
    return img, (by, bx), (py, px)


def rotate_tile(rgb: np.ndarray, deg: float) -> np.ndarray:
    """Rotate an HxWx3 tile in the (row, col) plane, the convention
    :func:`rotate_azimuth` is derived against."""
    return ndi_rotate(rgb, deg, axes=(0, 1), reshape=False, order=1, mode="nearest")


def argmax_rc(field: np.ndarray):
    return np.unravel_index(int(np.argmax(field)), field.shape)


def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# --------------------------------------------------------------------------- #
# Peak-response geometry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("azimuth_deg", [0.0, 35.0, 90.0, 200.0])
@pytest.mark.parametrize("aggregation", ["max", "logsumexp"])
def test_peak_response_at_known_location(azimuth_deg, aggregation):
    az = math.radians(azimuth_deg)
    cfg = ShadowFeatureConfig(
        offset_min=2.0, offset_max=20.0, offset_steps=8,
        aggregation=aggregation, n_channels=2,
    )
    rgb, blob_c, patch_c = make_tile_with_shadow(azimuth_rad=az, offset=10.0)
    feat = compute_shadow_feature(rgb, az, cfg)

    assert feat.shape == (2, 64, 64)
    crown, shadow = feat[0], feat[1]

    # Crown (bright-here AND dark-ahead) peaks on the lit blob...
    assert dist(argmax_rc(crown), blob_c) <= 4.0
    # ...and the dual shadow channel peaks on the dark patch.
    assert dist(argmax_rc(shadow), patch_c) <= 4.0


@pytest.mark.parametrize("angle_deg", [18.821, 2.864, 90.0, 200.0, 359.0])
def test_vector_to_azimuth_matches_csv_convention(angle_deg):
    """CSV gives shadow_x=sin(angle) [East/col], shadow_y=cos(angle) [North, y-up];
    vector_to_azimuth must reproduce the row-down displacement (row=-shadow_y,
    col=shadow_x) -- North points toward decreasing rows."""
    sx, sy = math.sin(math.radians(angle_deg)), math.cos(math.radians(angle_deg))
    phi = vector_to_azimuth(sx, sy)
    u_row, u_col = azimuth_to_vector(phi)
    assert u_col == pytest.approx(sx, abs=1e-9)
    assert u_row == pytest.approx(-sy, abs=1e-9)


def test_sun_convention_equals_shadow_with_pi_offset():
    """`azimuth_points_to="sun"` must be exactly the anti-solar convention with the
    azimuth rotated by pi (the sign flip is a pure pi rotation, nothing else)."""
    az = 0.7
    rgb, _, _ = make_tile_with_shadow(azimuth_rad=az, offset=10.0)
    f_sun = compute_shadow_feature(
        rgb, az, ShadowFeatureConfig(azimuth_points_to="sun", n_channels=2)
    )
    f_shadow_pi = compute_shadow_feature(
        rgb, az + math.pi, ShadowFeatureConfig(azimuth_points_to="shadow", n_channels=2)
    )
    np.testing.assert_allclose(f_sun, f_shadow_pi, atol=1e-9)


def test_wrong_azimuth_gives_weaker_crown_peak():
    """A filter pointed 90 degrees off the true shadow direction should not light
    up the crown as strongly: this is the whole point of *directional* prior."""
    cfg = ShadowFeatureConfig(n_channels=1, aggregation="max")
    az = 0.0
    rgb, blob_c, _ = make_tile_with_shadow(azimuth_rad=az, offset=10.0)

    aligned = compute_shadow_feature(rgb, az, cfg)[0]
    orthogonal = compute_shadow_feature(rgb, az + math.pi / 2, cfg)[0]

    by, bx = int(blob_c[0]), int(blob_c[1])
    assert aligned[by, bx] > orthogonal[by, bx] + 0.2


# --------------------------------------------------------------------------- #
# Recompute-not-rotate invariant (rotation equivariance) -- decision #2
# --------------------------------------------------------------------------- #
def test_recompute_not_rotate_invariant_90deg():
    """compute(rot(tile), rot_az(az)) == rot(compute(tile, az)) at 90 degrees.

    A 90-degree rotation lands pixel centres on pixel centres, so equivariance
    should hold to interpolation precision. This is the positive statement of
    decision #2: recomputing from the rotated azimuth reproduces the rotated
    response, so the feature is genuine rotation-equivariant directional evidence.
    """
    cfg = ShadowFeatureConfig(n_channels=1)
    az = 0.6
    rgb, _, _ = make_tile_with_shadow(azimuth_rad=az, offset=10.0)

    base = compute_shadow_feature(rgb, az, cfg)[0]
    rgb_rot = rotate_tile(rgb, 90.0)
    recomputed = compute_shadow_feature(rgb_rot, rotate_azimuth(az, 90.0), cfg)[0]
    rotated_raster = ndi_rotate(base, 90.0, reshape=False, order=1, mode="nearest")

    interior = (slice(8, -8), slice(8, -8))
    a, b = recomputed[interior], rotated_raster[interior]
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.99
    assert np.max(np.abs(a - b)) < 0.05


# --------------------------------------------------------------------------- #
# Lock decision #2: rotating the raster is NOT a substitute for recomputing
# --------------------------------------------------------------------------- #
def test_recompute_required_when_azimuth_not_updated():
    """Decisive lock: rotate the tile but reuse the OLD azimuth (the exact thing a
    'rotate the precomputed raster' shortcut bakes in) and the feature is wrong.

    At 90 degrees the directional filter is pointed a quarter turn off, so the
    crown response collapses where it should be strong. If this assertion ever
    starts passing as 'close', someone has decoupled the feature from the azimuth.
    """
    cfg = ShadowFeatureConfig(n_channels=1, aggregation="max")
    az = 0.0
    rgb, _, _ = make_tile_with_shadow(azimuth_rad=az, offset=10.0)
    rgb_rot = rotate_tile(rgb, 90.0)

    correct = compute_shadow_feature(rgb_rot, rotate_azimuth(az, 90.0), cfg)[0]
    stale_azimuth = compute_shadow_feature(rgb_rot, az, cfg)[0]  # forgot to update

    assert not np.allclose(correct, stale_azimuth, atol=1e-3)
    assert np.max(np.abs(correct - stale_azimuth)) > 0.2


def test_rotating_precomputed_raster_leaks_interpolation():
    """At a non-axis angle, rotating a precomputed feature raster diverges from a
    fresh recompute by more than floating-point noise -- the interpolation 'leak'
    decision #2 warns about. The two paths are therefore NOT interchangeable, which
    is exactly why the pipeline must recompute (not rotate) the feature.
    """
    cfg = ShadowFeatureConfig(n_channels=1)
    az = 0.0
    rgb, _, _ = make_tile_with_shadow(azimuth_rad=az, offset=10.0)

    base = compute_shadow_feature(rgb, az, cfg)[0]
    rgb_rot = rotate_tile(rgb, 37.0)
    recomputed = compute_shadow_feature(rgb_rot, rotate_azimuth(az, 37.0), cfg)[0]
    rotated_raster = ndi_rotate(base, 37.0, reshape=False, order=1, mode="nearest")

    interior = (slice(10, -10), slice(10, -10))
    diff = np.abs(recomputed[interior] - rotated_raster[interior])
    # Materially different, not just FP: substituting would change the model's input.
    assert diff.max() > 0.02
