"""Configuration dataclasses for the directional shadow-prior experiment.

Every tunable lives here so there are no magic numbers inline in the algorithms
(project convention) and so a run's full configuration can be serialised as a
reproducibility artifact (README "Reproducibility").

The offset bracket (``offset_min``/``offset_max``) encodes a *site prior* on the
plausible range of shadow displacement in pixels. Per design decision #1 it is
passed in as config and must NEVER be tuned on test data or per-acquisition --
doing so would leak test information into the feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Tuple
import json


@dataclass(frozen=True)
class ShadowFeatureConfig:
    """Parameters of the swept directional matched filter (design decision #1).

    Attributes
    ----------
    offset_min, offset_max:
        Inclusive bracket, in **pixels**, over which the shadow displacement
        ``s`` is swept. This is the "marginalise over shadow length" range: we do
        not know solar elevation, so crown height (hence displacement
        ``d = h / tan(elevation)``) is a nuisance we integrate over rather than
        compute (decision #1). Derived from site priors and frozen across the
        whole experiment.
    offset_steps:
        Number of discrete offsets sampled in ``[offset_min, offset_max]``
        (linspace, endpoints inclusive). This is the only Python-level loop in the
        feature; it is over offsets (~handful), never over pixels.
    aggregation:
        How the per-offset scores are combined into one response. ``"max"`` is the
        MAP-style choice (best single displacement); ``"logsumexp"`` is the soft
        marginalisation over the nuisance displacement. Switchable by design.
    logsumexp_beta:
        Inverse-temperature for the ``logsumexp`` aggregation. Higher -> closer to
        ``max``; lower -> flatter average over offsets. Ignored for ``"max"``.
    brightness_proxy:
        ``"luminance"`` (Rec.601 luma) or ``"greenness"`` (normalised excess
        green). The matched filter needs a scalar "is this canopy lit?" field; the
        proxy is configurable because the informative quantity is *contrast along
        the azimuth*, not the specific photometric channel.
    azimuth_points_to:
        Convention for the annotated azimuth vector. ``"shadow"`` (default) means it
        points along the **shadow displacement** (anti-solar) -- the feature samples
        "dark ahead along phi" directly. ``"sun"`` means it points toward the sun, so
        shadows fall the opposite way and the effective azimuth is rotated by pi.
        This is a one-flag switch precisely so the convention can be **verified
        against real crowns** (see :func:`shadow_prior.verify.recommend_convention`)
        rather than assumed; getting the sign wrong silently inverts the prior.
    n_channels:
        1 or 2 extra channels. 1 -> crown response only. 2 -> ``[crown, shadow]``
        where the shadow channel is the dual filter ("dark here AND bright up-sun").
        We deliberately do NOT emit the arg-max offset as a second channel: that
        would be *computing* shadow length, which decision #1 forbids.
    shift_mode, shift_order:
        Passed to :func:`scipy.ndimage.shift` for the sub-pixel displacement along
        the azimuth. ``mode="nearest"`` avoids fabricating dark borders (which a
        constant-0 pad would, creating false "shadow" responses at edges).
        ``order=1`` is linear interpolation (offsets are generally fractional).
    contrast_gain:
        Logistic gain applied to the standardised brightness ``z`` when forming the
        soft "is-lit" / "is-dark" terms (``expit(+/- gain * z)``). Higher -> harder
        threshold. The score is built from a *standardised* field (median/MAD, with
        a std fallback for near-flat tiles) rather than a min-max rescale, because
        a min-max/percentile rescale collapses to zero on tiles whose bright/dark
        features cover only a small area -- precisely the small-crown regime of the
        sparse-rangeland target.
    z_clip:
        Symmetric clamp on the standardised brightness before the logistic, so a
        handful of specular highlights or black shadows cannot dominate.
    """

    offset_min: float = 2.0
    offset_max: float = 20.0
    offset_steps: int = 8
    aggregation: str = "max"  # "max" | "logsumexp"
    logsumexp_beta: float = 4.0
    brightness_proxy: str = "luminance"  # "luminance" | "greenness"
    azimuth_points_to: str = "shadow"  # "shadow" (anti-solar) | "sun"
    n_channels: int = 1  # 1 | 2
    shift_mode: str = "nearest"
    shift_order: int = 1
    contrast_gain: float = 1.5
    z_clip: float = 6.0

    def __post_init__(self) -> None:
        if self.offset_steps < 1:
            raise ValueError("offset_steps must be >= 1")
        if not (self.offset_min <= self.offset_max):
            raise ValueError("offset_min must be <= offset_max")
        if self.aggregation not in ("max", "logsumexp"):
            raise ValueError(f"unknown aggregation {self.aggregation!r}")
        if self.brightness_proxy not in ("luminance", "greenness"):
            raise ValueError(f"unknown brightness_proxy {self.brightness_proxy!r}")
        if self.azimuth_points_to not in ("shadow", "sun"):
            raise ValueError(f"unknown azimuth_points_to {self.azimuth_points_to!r}")
        if self.n_channels not in (1, 2):
            raise ValueError("n_channels must be 1 or 2")
        if self.contrast_gain <= 0:
            raise ValueError("contrast_gain must be > 0")
        if self.z_clip <= 0:
            raise ValueError("z_clip must be > 0")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass(frozen=True)
class FoldConfig:
    """Cross-validation configuration (see :mod:`shadow_prior.folds`)."""

    n_splits: int = 5
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")


@dataclass(frozen=True)
class DatasetConfig:
    """Augmentation/runtime configuration for :mod:`shadow_prior.dataset`.

    ``rotation_choices_deg`` are the discrete rotation angles the augmenter may
    apply. Rotation is the *synthetic* direction-variance axis (README: "Direction
    variance has two sources"): the scene is held fixed while the azimuth is
    rotated, giving a clean-mechanism test free of capture-context confounds.
    """

    rung: str = "correct"  # "rgb" | "correct" | "shuffled"
    augment: bool = True
    rotation_choices_deg: Tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
    rotation_order: int = 1
    rotation_mode: str = "nearest"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.rung not in ("rgb", "correct", "shuffled"):
            raise ValueError(f"unknown rung {self.rung!r}")
        if len(self.rotation_choices_deg) == 0:
            raise ValueError("rotation_choices_deg must be non-empty")
