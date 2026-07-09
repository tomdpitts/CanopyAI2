"""Tests for scene-clustered folds and the effective-N bookkeeping (decision #4)."""

from __future__ import annotations

import numpy as np
import pytest

from shadow_prior.folds import (
    make_scene_clustered_folds,
    leave_one_acquisition_out,
)


def _scene_tile_table():
    """4 scenes, 3 tiles each (12 items). Scenes nest under 2 acquisitions."""
    scenes, acqs = [], []
    scene_to_acq = {"s0": "A", "s1": "A", "s2": "B", "s3": "B"}
    for s in ["s0", "s1", "s2", "s3"]:
        for _ in range(3):
            scenes.append(s)
            acqs.append(scene_to_acq[s])
    return np.array(scenes), np.array(acqs)


def test_no_scene_leakage_across_folds():
    scenes, acqs = _scene_tile_table()
    fa = make_scene_clustered_folds(scenes, n_splits=2, acquisition_ids=acqs, seed=0)
    # Every tile of a scene shares one fold (the guard raises otherwise).
    fa.assert_no_scene_leakage()
    for s in np.unique(scenes):
        assert np.unique(fa.fold_of_item[scenes == s]).size == 1


def test_effective_n_is_scene_count_not_tile_count():
    scenes, acqs = _scene_tile_table()
    fa = make_scene_clustered_folds(scenes, n_splits=2, acquisition_ids=acqs, seed=0)
    eff = fa.effective_n()
    # 4 distinct scenes total, not 12 tiles.
    assert sum(eff.values()) == 4
    assert sum(eff.values()) != len(scenes)
    # train/test scene counts sum to the total scene count.
    for fold in range(fa.n_folds):
        n_train, n_test = fa.train_test_scene_counts(fold)
        assert n_train + n_test == 4


def test_too_few_scenes_raises():
    scenes = np.array(["s0", "s0", "s1"])
    with pytest.raises(ValueError):
        make_scene_clustered_folds(scenes, n_splits=5)


def test_scene_spanning_two_acquisitions_raises():
    scenes = np.array(["s0", "s0"])
    acqs = np.array(["A", "B"])  # same scene, two acquisitions -> bad nesting
    with pytest.raises(ValueError):
        make_scene_clustered_folds(scenes, n_splits=2, acquisition_ids=acqs)


def test_leave_one_acquisition_out():
    scenes, acqs = _scene_tile_table()
    fa = leave_one_acquisition_out(scenes, acqs)
    assert fa.n_folds == 2  # two acquisitions
    # The test fold for acquisition A holds exactly A's items.
    fold_A = fa.labels.index("A")
    assert set(acqs[fa.test_indices(fold_A)]) == {"A"}
    assert set(acqs[fa.train_indices(fold_A)]) == {"B"}
    # Effective N for a held-out acquisition = its distinct scenes (2 here).
    assert fa.effective_n()[fold_A] == 2
