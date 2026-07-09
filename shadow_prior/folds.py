"""Scene-clustered cross-validation splitting.

Design decision #4: the unit of analysis is the **source scene**, not the crown
and not the tile. Effective N is the number of distinct scenes, and every tile /
rotation of a scene must live in a single fold -- otherwise the same scene appears
in train and test (pseudoreplication / rotation-inflated N, the named failure
modes). This module produces fold assignments at the scene level and exposes the
effective-N so it can be persisted as a reproducibility artifact.

Scenes are required to nest under acquisitions (each scene belongs to exactly one
acquisition). That nesting is what makes leave-one-acquisition-out expressible as a
coarsening of the same grouping (the generalisation number in the README protocol).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import json

import numpy as np


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class FoldAssignment:
    """Per-item fold labels plus the scene-level bookkeeping that defines N.

    ``fold_of_item[i]`` is the fold (test fold) that item ``i`` belongs to.
    ``labels`` names what each fold index means: a fold id for k-fold, or the held
    out acquisition id for leave-one-acquisition-out.
    """

    fold_of_item: np.ndarray              # int, shape (n_items,)
    scene_ids: np.ndarray                 # shape (n_items,)
    acquisition_ids: Optional[np.ndarray] # shape (n_items,) or None
    n_folds: int
    labels: List = field(default_factory=list)  # human-readable id per fold index

    # -- index access -------------------------------------------------------- #
    def test_indices(self, fold: int) -> np.ndarray:
        return np.flatnonzero(self.fold_of_item == fold)

    def train_indices(self, fold: int) -> np.ndarray:
        return np.flatnonzero(self.fold_of_item != fold)

    # -- effective-N (unit of analysis = scene) ------------------------------ #
    def scenes_in_fold(self, fold: int) -> np.ndarray:
        return np.unique(self.scene_ids[self.test_indices(fold)])

    def effective_n(self) -> Dict[int, int]:
        """Distinct **scenes** per test fold. This is the sample size that the
        statistics treat as N -- not tile or rotation counts (decision #4)."""
        return {f: int(self.scenes_in_fold(f).size) for f in range(self.n_folds)}

    def train_test_scene_counts(self, fold: int) -> Tuple[int, int]:
        """(n_train_scenes, n_test_scenes) -- the sizes the Nadeau-Bengio
        corrected resampled t-test needs (see :mod:`shadow_prior.stats`)."""
        n_test = self.scenes_in_fold(fold).size
        n_total = np.unique(self.scene_ids).size
        return int(n_total - n_test), int(n_test)

    def assert_no_scene_leakage(self) -> None:
        """Raise if any scene's items span more than one fold (hard leakage guard)."""
        for scene in np.unique(self.scene_ids):
            folds = np.unique(self.fold_of_item[self.scene_ids == scene])
            if folds.size > 1:
                raise ValueError(
                    f"scene {scene!r} spans folds {folds.tolist()} -- "
                    "leakage; all items of a scene must share one fold"
                )

    # -- persistence (reproducibility artifact) ------------------------------ #
    def to_dict(self) -> dict:
        return {
            "n_folds": self.n_folds,
            "labels": [str(x) for x in self.labels],
            "fold_of_item": self.fold_of_item.tolist(),
            "scene_ids": [str(x) for x in self.scene_ids.tolist()],
            "effective_n": self.effective_n(),
        }

    def save_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _validate_nesting(scene_ids: np.ndarray, acquisition_ids: np.ndarray) -> None:
    """Each scene must belong to exactly one acquisition (decision #4 nesting)."""
    for scene in np.unique(scene_ids):
        acqs = np.unique(acquisition_ids[scene_ids == scene])
        if acqs.size > 1:
            raise ValueError(
                f"scene {scene!r} maps to multiple acquisitions {acqs.tolist()}; "
                "scenes must nest under acquisitions"
            )


# --------------------------------------------------------------------------- #
# k-fold clustered by scene
# --------------------------------------------------------------------------- #
def make_scene_clustered_folds(
    scene_ids: Sequence,
    n_splits: int = 5,
    acquisition_ids: Optional[Sequence] = None,
    seed: int = 0,
) -> FoldAssignment:
    """GroupKFold-style split with the **scene** as the group key.

    All items sharing a scene id land in the same fold (no scene leakage). Folds
    are balanced greedily by item count -- scenes are shuffled (seeded) then the
    largest are placed first into the currently-smallest fold -- which keeps test
    sizes comparable without ever splitting a scene.

    If ``acquisition_ids`` are given, scene/acquisition nesting is validated so the
    same data also supports :func:`leave_one_acquisition_out`.
    """
    scene_ids = np.asarray(scene_ids)
    if scene_ids.ndim != 1:
        raise ValueError("scene_ids must be 1-D")
    acq = np.asarray(acquisition_ids) if acquisition_ids is not None else None
    if acq is not None:
        if acq.shape != scene_ids.shape:
            raise ValueError("acquisition_ids must match scene_ids length")
        _validate_nesting(scene_ids, acq)

    unique_scenes, counts = np.unique(scene_ids, return_counts=True)
    if unique_scenes.size < n_splits:
        raise ValueError(
            f"cannot make {n_splits} folds from {unique_scenes.size} scenes; "
            "effective N (distinct scenes) is below n_splits"
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(unique_scenes.size)
    # Largest scenes first (descending count), shuffled tie-break, for balance.
    order = order[np.argsort(-counts[order], kind="stable")]

    fold_load = np.zeros(n_splits, dtype=np.int64)
    fold_of_scene: Dict = {}
    for idx in order:
        target = int(np.argmin(fold_load))
        fold_of_scene[unique_scenes[idx]] = target
        fold_load[target] += counts[idx]

    fold_of_item = np.array([fold_of_scene[s] for s in scene_ids], dtype=np.int64)
    assignment = FoldAssignment(
        fold_of_item=fold_of_item,
        scene_ids=scene_ids,
        acquisition_ids=acq,
        n_folds=n_splits,
        labels=list(range(n_splits)),
    )
    assignment.assert_no_scene_leakage()
    return assignment


# --------------------------------------------------------------------------- #
# Leave-one-acquisition-out (the generalisation number)
# --------------------------------------------------------------------------- #
def leave_one_acquisition_out(
    scene_ids: Sequence,
    acquisition_ids: Sequence,
) -> FoldAssignment:
    """One fold per acquisition: test = that acquisition's scenes, train = the rest.

    This is the across-acquisition / external-validity axis (README). Each held-out
    acquisition contributes one paired estimate, so the effective N for the
    generalisation comparison is the number of acquisitions, not scenes or tiles.
    """
    scene_ids = np.asarray(scene_ids)
    acquisition_ids = np.asarray(acquisition_ids)
    if scene_ids.shape != acquisition_ids.shape or scene_ids.ndim != 1:
        raise ValueError("scene_ids and acquisition_ids must be 1-D and same length")
    _validate_nesting(scene_ids, acquisition_ids)

    unique_acq = np.unique(acquisition_ids)
    if unique_acq.size < 2:
        raise ValueError("need >= 2 acquisitions for leave-one-acquisition-out")

    acq_to_fold = {acq: i for i, acq in enumerate(unique_acq)}
    fold_of_item = np.array([acq_to_fold[a] for a in acquisition_ids], dtype=np.int64)
    assignment = FoldAssignment(
        fold_of_item=fold_of_item,
        scene_ids=scene_ids,
        acquisition_ids=acquisition_ids,
        n_folds=unique_acq.size,
        labels=list(unique_acq),
    )
    assignment.assert_no_scene_leakage()
    return assignment
