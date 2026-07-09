"""Tests for the dataset: recompute-not-rotate, rung selection, leakage guard."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.ndimage import rotate as ndi_rotate

from shadow_prior.config import ShadowFeatureConfig, DatasetConfig
from shadow_prior.geometry import rotate_azimuth
from shadow_prior.shadow_feature import compute_shadow_feature
from shadow_prior.dataset import CrownTileDataset, TileRecord, assign_folds_by_scene


def make_tile(size=64, azimuth_rad=0.0, offset=10.0):
    img = np.full((size, size, 3), 0.5, dtype=np.float64)
    rr, cc = np.ogrid[:size, :size]
    cy = cx = size / 2
    img[(rr - cy) ** 2 + (cc - cx) ** 2 <= 9] = 1.0
    py = cy + offset * math.sin(azimuth_rad)
    px = cx + offset * math.cos(azimuth_rad)
    img[(rr - py) ** 2 + (cc - px) ** 2 <= 16] = 0.0
    return img


def _rec(scene, acq, az, crowns, fold=None):
    return TileRecord(
        rgb=make_tile(azimuth_rad=az), azimuth_rad=az,
        scene_id=scene, acquisition_id=acq, crown_ids=crowns, fold=fold,
    )


# --------------------------------------------------------------------------- #
# Recompute-not-rotate at the dataset level (decision #2)
# --------------------------------------------------------------------------- #
def test_dataset_recomputes_feature_from_rotated_tile_and_azimuth():
    shadow_cfg = ShadowFeatureConfig(n_channels=1)
    ds_cfg = DatasetConfig(
        rung="correct", augment=True, rotation_choices_deg=(90.0,), seed=0
    )
    az = 0.3
    rec = _rec("s0", "A", az, ("c0",), fold=0)
    ds = CrownTileDataset([rec], shadow_cfg, ds_cfg, keep_folds=[0])

    x, meta = ds[0]
    assert x.shape[0] == 4  # RGB (0-2) + 1 shadow channel
    assert meta["rotation_deg"] == 90.0

    rgb_rot = ndi_rotate(
        rec.rgb, 90.0, axes=(0, 1), reshape=False, order=1, mode="nearest"
    )
    expected = compute_shadow_feature(rgb_rot, rotate_azimuth(az, 90.0), shadow_cfg)[0]
    np.testing.assert_allclose(x[3].numpy(), expected, atol=1e-5)

    # The classic bug -- not updating the azimuth -- must give a DIFFERENT channel.
    stale = compute_shadow_feature(rgb_rot, az, shadow_cfg)[0]
    assert not np.allclose(x[3].numpy(), stale, atol=1e-3)


def test_rgb_channels_are_first():
    """RGB must be channels 0-2 so a pretrained 3-channel stem can be inflated."""
    shadow_cfg = ShadowFeatureConfig(n_channels=2)
    ds_cfg = DatasetConfig(rung="correct", augment=False)
    rec = _rec("s0", "A", 0.0, ("c0",), fold=0)
    ds = CrownTileDataset([rec], shadow_cfg, ds_cfg, keep_folds=[0])
    x, _ = ds[0]
    assert x.shape[0] == 5  # 3 RGB + 2 shadow
    rgb_expected = np.transpose(make_tile(azimuth_rad=0.0), (2, 0, 1))
    np.testing.assert_allclose(x[:3].numpy(), rgb_expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# Rung selection (decision #3)
# --------------------------------------------------------------------------- #
def test_rung_channel_counts():
    shadow_cfg = ShadowFeatureConfig(n_channels=2)
    rec = _rec("s0", "A", 0.0, ("c0",), fold=0)

    rgb_ds = CrownTileDataset(
        [rec], shadow_cfg, DatasetConfig(rung="rgb", augment=False), keep_folds=[0]
    )
    assert rgb_ds.n_channels == 3
    assert rgb_ds[0][0].shape[0] == 3


def test_shuffled_rung_uses_within_acquisition_permuted_azimuth():
    shadow_cfg = ShadowFeatureConfig(n_channels=1)
    # Two scenes in one acquisition with different azimuths -> a real shuffle.
    recs = [
        _rec("s0", "A", 0.0, ("c0",), fold=0),
        _rec("s1", "A", 1.2, ("c1",), fold=0),
    ]
    ds_cfg = DatasetConfig(rung="shuffled", augment=False, seed=0)
    ds = CrownTileDataset(recs, shadow_cfg, ds_cfg, keep_folds=[0])

    base_used = {ds[i][1]["scene_id"]: ds[i][1]["base_azimuth_rad"] for i in range(2)}
    originals = {"s0": 0.0, "s1": 1.2}
    # Each used azimuth is still one of the acquisition's azimuths (within-group)...
    assert set(np.round(list(base_used.values()), 6)) <= {0.0, 1.2}
    # ...and the pair has been swapped relative to the originals (size-2 group).
    assert base_used != originals


# --------------------------------------------------------------------------- #
# Leakage guard (decision #4)
# --------------------------------------------------------------------------- #
def test_crown_spanning_two_folds_raises():
    shadow_cfg = ShadowFeatureConfig()
    ds_cfg = DatasetConfig(rung="rgb", augment=False)
    # Same crown "c0" present in tiles assigned to fold 0 and fold 1 -> leakage.
    recs = [
        _rec("s0", "A", 0.0, ("c0",), fold=0),
        _rec("s1", "A", 0.0, ("c0",), fold=1),
    ]
    with pytest.raises(ValueError, match="leakage"):
        CrownTileDataset(recs, shadow_cfg, ds_cfg)


def test_missing_fold_raises():
    shadow_cfg = ShadowFeatureConfig()
    ds_cfg = DatasetConfig(rung="rgb", augment=False)
    rec = _rec("s0", "A", 0.0, ("c0",), fold=None)
    with pytest.raises(ValueError, match="needs a fold"):
        CrownTileDataset([rec], shadow_cfg, ds_cfg)


def test_assign_folds_by_scene_populates_records():
    recs = [_rec("s0", "A", 0.0, ("c0",)), _rec("s1", "A", 0.0, ("c1",))]
    assign_folds_by_scene(recs, {"s0": 0, "s1": 1})
    assert [r.fold for r in recs] == [0, 1]
