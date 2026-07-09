"""Tests for the paired statistics (Nadeau-Bengio, permutation, seed variance)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from shadow_prior.stats import (
    corrected_resampled_ttest,
    permutation_test_paired,
    seed_variance,
    cohens_dz,
)


def test_corrected_test_is_more_conservative_than_plain_ttest():
    """The Nadeau-Bengio correction inflates the variance, so for the same deltas
    it must yield a LARGER p-value than the naive paired t-test (the whole point:
    overlapping training sets make the naive test anti-conservative)."""
    deltas = np.array([0.03, 0.05, 0.02, 0.04, 0.06])
    n_train, n_test = 40, 10  # test/train ratio drives the correction

    res = corrected_resampled_ttest(deltas, n_train=n_train, n_test=n_test)
    naive_t = deltas.mean() / (deltas.std(ddof=1) / np.sqrt(deltas.size))
    naive_p = 2 * sps.t.sf(abs(naive_t), deltas.size - 1)

    assert res.p_value > naive_p
    # CI is consistent with the reported mean.
    assert res.ci_low < res.mean_delta < res.ci_high
    assert res.effect_size_dz == pytest.approx(cohens_dz(deltas))


def test_permutation_exact_for_small_n_and_symmetric_null():
    # Symmetric-about-zero deltas -> mean ~0 -> large p (no effect).
    deltas = np.array([0.1, -0.1, 0.2, -0.2])
    res = permutation_test_paired(deltas, seed=0)
    assert res.note == "exact"  # 2**4 = 16 enumerated
    assert res.p_value > 0.5


def test_permutation_detects_consistent_positive_effect():
    deltas = np.array([0.05, 0.06, 0.04, 0.07, 0.05, 0.06])
    res = permutation_test_paired(deltas, seed=0)
    # All positive -> the all-plus sign assignment is the most extreme -> small p.
    assert res.p_value < 0.05


def test_seed_variance_mde_scales_with_noise_and_n():
    low = seed_variance([0.50, 0.51, 0.49, 0.50], alpha=0.05, power=0.8)
    high = seed_variance([0.50, 0.60, 0.40, 0.55], alpha=0.05, power=0.8)
    # More seed noise -> larger minimum detectable effect.
    assert high.minimum_detectable_effect > low.minimum_detectable_effect
    # Paired-difference std is sqrt(2) * seed std.
    assert high.paired_diff_std == pytest.approx(np.sqrt(2) * high.seed_std)
