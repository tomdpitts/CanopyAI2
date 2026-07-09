"""PyTorch dataset that emits the per-rung input stack for the ablation.

Three things this module guarantees, each tied to a design decision:

* **Recompute, never rotate the feature** (decision #2). Each sample applies
  rotation augmentation to the tile, updates the azimuth scalar by the *same*
  rotation, and only *then* computes the shadow feature from the rotated tile and
  rotated azimuth. The order is enforced (`_featurize` refuses to run on a tile
  that has not been through the augmentation stage).
* **Rung selection** (decision #3). ``"rgb"`` emits RGB only; ``"correct"`` appends
  the true-direction feature; ``"shuffled"`` appends the feature computed from a
  within-acquisition permuted azimuth -- the load-bearing control.
* **No scene/crown leakage across folds** (decision #4). Construction raises if any
  crown's records span more than one fold.

The emitted tensor is RGB-first: channels 0-2 are RGB, any shadow channels follow.
That ordering is deliberate -- it lets a pretrained 3-channel stem (e.g. DeepForest
RetinaNet weights) be reused by *inflating* the first conv with extra zero-init
input channels for the shadow feature, leaving the RGB response unchanged at init.
The detector itself is out of scope: we stop at the ``(C, H, W)`` stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import math

import numpy as np
import torch
from scipy.ndimage import rotate as ndi_rotate
from torch.utils.data import Dataset

from .config import ShadowFeatureConfig, DatasetConfig
from .geometry import rotate_azimuth
from .shadow_feature import compute_shadow_feature, shuffle_azimuths

ArrayOrLoader = Union[np.ndarray, Callable[[], np.ndarray]]


# --------------------------------------------------------------------------- #
# Sample record
# --------------------------------------------------------------------------- #
@dataclass
class TileRecord:
    """One annotated tile.

    ``rgb`` may be an ``HxWx3`` array or a zero-arg callable returning one (for lazy
    loading from disk). ``crown_ids`` lists the instance ids in the tile; the fold
    leakage guard is expressed over crowns. ``fold`` is the scene-level fold this
    tile belongs to (populate with :func:`assign_folds_by_scene`).
    """

    rgb: ArrayOrLoader
    azimuth_rad: float
    scene_id: object
    acquisition_id: object
    crown_ids: Tuple = ()
    instance_masks: Optional[np.ndarray] = None  # KxHxW or HxW label map
    fold: Optional[int] = None

    def load_rgb(self) -> np.ndarray:
        return self.rgb() if callable(self.rgb) else self.rgb


# --------------------------------------------------------------------------- #
# Fold helpers
# --------------------------------------------------------------------------- #
def assign_folds_by_scene(
    records: Sequence[TileRecord], fold_of_scene: Dict
) -> None:
    """Populate ``record.fold`` from a scene->fold mapping (in place)."""
    for rec in records:
        if rec.scene_id not in fold_of_scene:
            raise KeyError(f"scene {rec.scene_id!r} missing from fold mapping")
        rec.fold = int(fold_of_scene[rec.scene_id])


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CrownTileDataset(Dataset):
    def __init__(
        self,
        records: Sequence[TileRecord],
        shadow_cfg: ShadowFeatureConfig,
        ds_cfg: DatasetConfig,
        keep_folds: Optional[Sequence[int]] = None,
        shuffled_azimuth_of_scene: Optional[Dict] = None,
    ):
        self.shadow_cfg = shadow_cfg
        self.ds_cfg = ds_cfg
        self._all_records = list(records)

        self._assert_no_crown_fold_leakage(self._all_records)

        if keep_folds is None:
            self.records = list(self._all_records)
        else:
            keep = set(int(f) for f in keep_folds)
            self.records = [r for r in self._all_records if r.fold in keep]
        if not self.records:
            raise ValueError("no records selected (check keep_folds)")

        # Rung-3 control: a within-acquisition azimuth permutation, keyed by scene.
        # Built from ALL records (so the permutation is over the full acquisition,
        # not just the kept fold) and frozen for the dataset's lifetime.
        if ds_cfg.rung == "shuffled":
            self.shuffled_azimuth_of_scene = (
                shuffled_azimuth_of_scene
                if shuffled_azimuth_of_scene is not None
                else self._build_scene_shuffle(self._all_records, ds_cfg.seed)
            )
        else:
            self.shuffled_azimuth_of_scene = None

        # Deterministic per-index rotation choice (reproducible augmentation).
        self._rng = np.random.default_rng(ds_cfg.seed)
        self._rotation_index = self._rng.integers(
            0, len(ds_cfg.rotation_choices_deg), size=len(self.records)
        )

    # -- guards ------------------------------------------------------------- #
    @staticmethod
    def _assert_no_crown_fold_leakage(records: Sequence[TileRecord]) -> None:
        """Decision #4: a crown (with all its rotations) must live in one fold.

        Rotation augmentation is applied at runtime, so a crown's rotations are
        trivially in the same fold *by construction*; this guard catches the
        upstream error where folds were assigned per tile/crown inconsistently and
        the same crown landed in two folds. Raise -- never warn.
        """
        crown_to_folds: Dict[object, set] = {}
        for rec in records:
            if rec.fold is None:
                raise ValueError(
                    "every record needs a fold before constructing the dataset; "
                    "use assign_folds_by_scene()"
                )
            for cid in rec.crown_ids:
                crown_to_folds.setdefault(cid, set()).add(int(rec.fold))
        leaking = {c: sorted(f) for c, f in crown_to_folds.items() if len(f) > 1}
        if leaking:
            raise ValueError(
                f"crowns span multiple folds (leakage): {leaking}. "
                "All rotations/tiles of a crown must be in one fold."
            )

    @staticmethod
    def _build_scene_shuffle(records: Sequence[TileRecord], seed: int) -> Dict:
        """Permute azimuths within acquisition at the scene level (decision #3)."""
        scene_az: Dict = {}
        scene_acq: Dict = {}
        for rec in records:
            if rec.scene_id in scene_az and not math.isclose(
                scene_az[rec.scene_id], rec.azimuth_rad
            ):
                raise ValueError(
                    f"scene {rec.scene_id!r} has inconsistent azimuths; one azimuth "
                    "per scene is required for the within-acquisition shuffle"
                )
            scene_az[rec.scene_id] = rec.azimuth_rad
            scene_acq[rec.scene_id] = rec.acquisition_id
        scenes = list(scene_az.keys())
        az = np.array([scene_az[s] for s in scenes], dtype=float)
        acq = np.array([scene_acq[s] for s in scenes], dtype=object)
        shuffled = shuffle_azimuths(az, acq, seed=seed)
        return {s: float(v) for s, v in zip(scenes, shuffled)}

    # -- length / item ------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.records)

    def _rotation_deg(self, index: int) -> float:
        if not self.ds_cfg.augment:
            return 0.0
        return float(self.ds_cfg.rotation_choices_deg[self._rotation_index[index]])

    def _to_float_rgb(self, rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"tile rgb must be HxWx3, got {rgb.shape}")
        rgb = rgb.astype(np.float64, copy=False)
        # uint8-style imagery -> [0, 1]; already-float [0,1] left as-is.
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        return rgb

    def _rotate(
        self, rgb: np.ndarray, masks: Optional[np.ndarray], deg: float
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if deg == 0.0:
            return rgb, masks
        rgb_rot = ndi_rotate(
            rgb, deg, axes=(0, 1), reshape=False,
            order=self.ds_cfg.rotation_order, mode=self.ds_cfg.rotation_mode,
        )
        masks_rot = masks
        if masks is not None:
            # order=0 to keep integer instance labels; outside -> background (0).
            axes = (1, 2) if masks.ndim == 3 else (0, 1)
            masks_rot = ndi_rotate(
                masks, deg, axes=axes, reshape=False, order=0, mode="constant", cval=0
            )
        return rgb_rot, masks_rot

    def _featurize(
        self, rgb: np.ndarray, azimuth_rad: float, *, post_rotation: bool
    ) -> np.ndarray:
        """Compute the shadow feature. ``post_rotation`` MUST be True.

        This flag is the teeth of decision #2's order requirement: the feature is
        only ever computed inside the augmentation stage, after the tile has been
        rotated and the azimuth updated. A refactor that tried to featurize a
        pre-rotation tile would have to pass ``post_rotation=False`` and trip here.
        """
        assert post_rotation, (
            "feature must be computed AFTER rotation from the rotation-updated "
            "azimuth (decision #2: recompute, never rotate a precomputed raster)"
        )
        return compute_shadow_feature(rgb, azimuth_rad, self.shadow_cfg)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict]:
        rec = self.records[index]
        rgb = self._to_float_rgb(rec.load_rgb())
        deg = self._rotation_deg(index)

        # Order matters (decision #2): rotate tile -> update azimuth -> featurize.
        rgb_rot, masks_rot = self._rotate(rgb, rec.instance_masks, deg)

        if self.ds_cfg.rung == "shuffled":
            base_az = self.shuffled_azimuth_of_scene[rec.scene_id]
        else:
            base_az = rec.azimuth_rad
        az_used = rotate_azimuth(base_az, deg)

        rgb_chw = np.transpose(rgb_rot, (2, 0, 1))  # HxWx3 -> 3xHxW, RGB first
        if self.ds_cfg.rung == "rgb":
            stack = rgb_chw
        else:
            feat = self._featurize(rgb_rot, az_used, post_rotation=True)
            stack = np.concatenate([rgb_chw, feat], axis=0)

        meta = {
            "scene_id": rec.scene_id,
            "acquisition_id": rec.acquisition_id,
            "crown_ids": list(rec.crown_ids),
            "fold": rec.fold,
            "rung": self.ds_cfg.rung,
            "rotation_deg": deg,
            "base_azimuth_rad": float(base_az),
            "azimuth_used_rad": float(az_used),
        }
        if masks_rot is not None:
            meta["instance_masks"] = torch.as_tensor(np.ascontiguousarray(masks_rot))

        return torch.as_tensor(stack, dtype=torch.float32), meta

    # -- introspection ------------------------------------------------------ #
    @property
    def n_channels(self) -> int:
        """Channel count of the emitted stack for the configured rung."""
        return 3 if self.ds_cfg.rung == "rgb" else 3 + self.shadow_cfg.n_channels
