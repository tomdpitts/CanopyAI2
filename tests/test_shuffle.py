"""Tests for the within-acquisition azimuth shuffle (rung-3 control, decision #3)."""

from __future__ import annotations

import numpy as np
import pytest

from shadow_prior.shadow_feature import shuffle_azimuths


def _grouped_azimuths():
    # Three acquisitions of sizes 4, 3, 1 with distinct azimuth ranges so we can
    # detect any leakage across groups by value.
    az = np.array([0.0, 0.1, 0.2, 0.3,  1.0, 1.1, 1.2,  2.5])
    acq = np.array(["A", "A", "A", "A", "B", "B", "B", "C"])
    return az, acq


@pytest.mark.parametrize("seed", range(8))
def test_permutation_stays_within_acquisition(seed):
    az, acq = _grouped_azimuths()
    out = shuffle_azimuths(az, acq, seed=seed)

    for group in np.unique(acq):
        idx = acq == group
        # Every shuffled value in a group must come from that same group...
        assert set(np.round(out[idx], 9)) <= set(np.round(az[idx], 9))
        # ...and the group is a permutation of itself (multiset preserved).
        assert sorted(np.round(out[idx], 9)) == sorted(np.round(az[idx], 9))


def test_singleton_acquisition_is_unchanged():
    az, acq = _grouped_azimuths()
    out = shuffle_azimuths(az, acq, seed=3)
    # Group "C" has one member: it cannot be shuffled and must be identical.
    assert out[acq == "C"] == az[acq == "C"]


def test_avoid_global_identity_changes_something():
    az, acq = _grouped_azimuths()
    out = shuffle_azimuths(az, acq, seed=0, avoid_global_identity=True)
    # With shufflable groups present, the result must differ from the input.
    assert not np.array_equal(out, az)


def test_all_singletons_returns_identity_without_hanging():
    az = np.array([0.0, 1.0, 2.0])
    acq = np.array(["A", "B", "C"])
    out = shuffle_azimuths(az, acq, seed=0, avoid_global_identity=True)
    # Nothing is shufflable; identity is the only possibility (must not loop).
    assert np.array_equal(out, az)
