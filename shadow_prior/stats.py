"""Paired statistics for the ablation, done the honest way (decision #4).

We report **effect size + confidence interval**, never a bare p-value, because the
declared target is a credible estimate (a precise null is a valid result), not a
hunt for p < 0.05 (README "Objective").

Primary test: the **Nadeau & Bengio (2003) corrected resampled t-test**. In
cross-validation the training sets overlap across folds, so the per-fold
differences are positively correlated; a plain paired t-test ignores this and
inflates significance. Nadeau & Bengio correct the variance by
``(1/J + n_test/n_train)`` instead of ``1/J``.

Backup test: a sign-flipping **permutation test** over the paired deltas
(assumption-light; exact enumeration when the number of folds is small).

Before either: a **seed-variance** estimate. If the shadow effect is smaller than
seed-to-seed noise, no test rescues it -- we report the minimum detectable effect.

All inputs are per-fold (or per-acquisition) paired deltas of a scalar metric, and
N is a *scene/acquisition* count, never tiles or rotations.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Sequence
import math

import numpy as np
from scipy import stats as sps


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class TestResult:
    name: str
    mean_delta: float
    ci_low: float
    ci_high: float
    effect_size_dz: float   # Cohen's d_z = mean / sd of paired deltas
    p_value: float
    n_folds: int
    statistic: Optional[float] = None
    df: Optional[float] = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SeedVarianceResult:
    mean_metric: float
    seed_std: float
    n_seeds: int
    paired_diff_std: float       # sd of a difference of two seed-noisy configs
    minimum_detectable_effect: float
    alpha: float
    power: float

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def cohens_dz(deltas: np.ndarray) -> float:
    """Standardised paired effect size: mean / sd of the paired deltas."""
    d = np.asarray(deltas, dtype=float)
    sd = d.std(ddof=1)
    return float("nan") if sd == 0 else float(d.mean() / sd)


# --------------------------------------------------------------------------- #
# Primary test: Nadeau & Bengio corrected resampled t-test
# --------------------------------------------------------------------------- #
def corrected_resampled_ttest(
    deltas: Sequence[float],
    n_train: int,
    n_test: int,
    alpha: float = 0.05,
    name: str = "nadeau-bengio",
) -> TestResult:
    """Corrected resampled t-test for CV folds with shared training data.

    Parameters
    ----------
    deltas:
        Paired per-fold differences (metric of config A minus config B), e.g.
        rung-2 minus rung-3 per fold. Length J = number of folds/resamples.
    n_train, n_test:
        Sizes of the training and test partitions **in the unit of analysis**
        (distinct scenes, decision #4). Use
        :meth:`shadow_prior.folds.FoldAssignment.train_test_scene_counts`.
    alpha:
        Two-sided significance level used for the reported CI.

    Notes
    -----
    Statistic ``t = mean(d) / sqrt((1/J + n_test/n_train) * var(d, ddof=1))`` with
    ``df = J - 1`` (Nadeau & Bengio 2003, Eq. for the corrected resampled t). The
    same corrected standard error is used for the CI, so the CI and the test agree.
    """
    d = np.asarray(deltas, dtype=float)
    J = d.size
    if J < 2:
        raise ValueError("need >= 2 folds for a paired test")
    if n_train <= 0 or n_test <= 0:
        raise ValueError("n_train and n_test must be positive scene counts")

    mean = float(d.mean())
    var = float(d.var(ddof=1))
    correction = 1.0 / J + n_test / n_train  # Nadeau & Bengio variance inflation
    se = math.sqrt(correction * var)
    df = J - 1
    t_crit = float(sps.t.ppf(1.0 - alpha / 2.0, df))

    if se == 0.0:
        # All deltas identical: degenerate. p=0 if non-zero mean, else 1.0.
        p = 0.0 if mean != 0.0 else 1.0
        ci_low = ci_high = mean
        stat = float("inf") if mean != 0.0 else 0.0
    else:
        stat = mean / se
        p = float(2.0 * sps.t.sf(abs(stat), df))
        ci_low = mean - t_crit * se
        ci_high = mean + t_crit * se

    return TestResult(
        name=name,
        mean_delta=mean,
        ci_low=ci_low,
        ci_high=ci_high,
        effect_size_dz=cohens_dz(d),
        p_value=p,
        n_folds=J,
        statistic=stat,
        df=df,
        note=f"correction_factor={correction:.4f} (1/J + n_test/n_train)",
    )


# --------------------------------------------------------------------------- #
# Backup test: sign-flip permutation over paired deltas
# --------------------------------------------------------------------------- #
def permutation_test_paired(
    deltas: Sequence[float],
    n_perm: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
    name: str = "permutation",
) -> TestResult:
    """Two-sided sign-flip permutation test on paired deltas (assumption-light).

    Under the null the sign of each paired delta is exchangeable. We enumerate all
    ``2**J`` sign assignments exactly when ``J`` is small, else sample ``n_perm`` of
    them. The CI reported is a bootstrap percentile interval on the mean delta (the
    permutation null has no natural CI; the bootstrap quantifies the same estimate's
    uncertainty for the assumption-light branch).
    """
    d = np.asarray(deltas, dtype=float)
    J = d.size
    if J < 2:
        raise ValueError("need >= 2 folds for a paired test")
    obs = float(d.mean())
    rng = np.random.default_rng(seed)

    if J <= 22 and (1 << J) <= max(n_perm, 1 << 20):
        # Exact enumeration of all sign flips.
        signs = ((np.arange(1 << J)[:, None] >> np.arange(J)) & 1) * 2 - 1
        perm_means = (signs * d).mean(axis=1)
    else:
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, J))
        perm_means = (signs * d).mean(axis=1)

    p = float((np.count_nonzero(np.abs(perm_means) >= abs(obs)) + 1) / (perm_means.size + 1))

    # Bootstrap percentile CI on the mean delta.
    boot = rng.choice(d, size=(2000, J), replace=True).mean(axis=1)
    ci_low, ci_high = (float(x) for x in np.quantile(boot, [alpha / 2, 1 - alpha / 2]))

    return TestResult(
        name=name,
        mean_delta=obs,
        ci_low=ci_low,
        ci_high=ci_high,
        effect_size_dz=cohens_dz(d),
        p_value=p,
        n_folds=J,
        statistic=obs,
        df=None,
        note=("exact" if perm_means.size == (1 << J) else f"sampled n_perm={perm_means.size}"),
    )


# --------------------------------------------------------------------------- #
# Seed variance & minimum detectable effect
# --------------------------------------------------------------------------- #
def seed_variance(
    scores: Sequence[float],
    n_seeds_paired: Optional[int] = None,
    alpha: float = 0.05,
    power: float = 0.8,
) -> SeedVarianceResult:
    """Seed-to-seed noise floor and the minimum detectable effect (MDE).

    ``scores`` are the metric from running one fixed config under different random
    seeds (everything else held constant). We report the seed std and the smallest
    true mean difference an A/B comparison could detect at ``alpha``/``power``,
    assuming paired runs over ``n_seeds_paired`` seeds and that each config carries
    independent seed noise (so the paired-difference std is ``sqrt(2)*seed_std``).

    Run this *before* trusting any rung comparison: if the shadow effect is smaller
    than the MDE, the comparison is underpowered and no test rescues it.
    """
    s = np.asarray(scores, dtype=float)
    if s.size < 2:
        raise ValueError("need >= 2 seed runs to estimate seed variance")
    seed_std = float(s.std(ddof=1))
    n = int(n_seeds_paired) if n_seeds_paired is not None else int(s.size)
    paired_diff_std = math.sqrt(2.0) * seed_std
    z_alpha = float(sps.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(sps.norm.ppf(power))
    mde = (z_alpha + z_power) * paired_diff_std / math.sqrt(n)
    return SeedVarianceResult(
        mean_metric=float(s.mean()),
        seed_std=seed_std,
        n_seeds=int(s.size),
        paired_diff_std=paired_diff_std,
        minimum_detectable_effect=float(mde),
        alpha=alpha,
        power=power,
    )
