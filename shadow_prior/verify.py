"""Empirically check the azimuth convention against annotated crowns.

Answer #2 in the design handoff was "anti-solar shadow direction, but we'll need to
double-check". This module operationalises that double-check so it is data-driven,
not assumed: the correct convention should make the crown response *fire on real
crowns* (a lit pixel with its cast shadow an offset ahead), so the mean crown
response inside true crown masks is higher under the correct sign than under the
flipped one. We try both and report which wins, with a margin.

Run this on a handful of annotated tiles before committing ``azimuth_points_to`` for
the experiment. Pure functions, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, field
from typing import Iterable, List, Tuple

import numpy as np

from .config import ShadowFeatureConfig
from .shadow_feature import compute_shadow_feature

Sample = Tuple[np.ndarray, float, np.ndarray]  # (rgb HxWx3, azimuth_rad, crown_mask HxW)


@dataclass
class ConventionReport:
    recommended: str          # "shadow" | "sun"
    score_shadow: float       # mean in-crown crown-response, anti-solar convention
    score_sun: float          # mean in-crown crown-response, sun-pointing convention
    margin: float             # |score_shadow - score_sun|
    n_samples: int
    per_sample: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "recommended": self.recommended,
            "score_shadow": self.score_shadow,
            "score_sun": self.score_sun,
            "margin": self.margin,
            "n_samples": self.n_samples,
        }


def crown_response_in_mask(
    rgb: np.ndarray, azimuth_rad: float, crown_mask: np.ndarray, cfg: ShadowFeatureConfig
) -> float:
    """Mean crown-channel response inside ``crown_mask`` (NaN if mask is empty)."""
    feat = compute_shadow_feature(rgb, azimuth_rad, replace(cfg, n_channels=1))[0]
    m = np.asarray(crown_mask, dtype=bool)
    if m.shape != feat.shape:
        raise ValueError(f"mask shape {m.shape} != feature shape {feat.shape}")
    if not m.any():
        return float("nan")
    return float(feat[m].mean())


def recommend_convention(
    samples: Iterable[Sample], cfg: ShadowFeatureConfig
) -> ConventionReport:
    """Recommend ``azimuth_points_to`` by comparing in-crown response under each sign.

    ``samples`` is an iterable of ``(rgb, azimuth_rad, crown_mask)``. The convention
    whose feature responds more strongly inside true crowns is recommended. A small
    ``margin`` means the data does not clearly distinguish the two -- investigate
    (low contrast, wrong masks, or an offset bracket mismatched to the shadows)
    rather than trusting the pick.
    """
    cfg_shadow = replace(cfg, azimuth_points_to="shadow")
    cfg_sun = replace(cfg, azimuth_points_to="sun")
    per: List[Tuple[float, float]] = []
    for rgb, az, mask in samples:
        ss = crown_response_in_mask(rgb, az, mask, cfg_shadow)
        us = crown_response_in_mask(rgb, az, mask, cfg_sun)
        per.append((ss, us))
    if not per:
        raise ValueError("no samples provided")

    arr = np.array(per, dtype=float)
    score_shadow = float(np.nanmean(arr[:, 0]))
    score_sun = float(np.nanmean(arr[:, 1]))
    recommended = "shadow" if score_shadow >= score_sun else "sun"
    return ConventionReport(
        recommended=recommended,
        score_shadow=score_shadow,
        score_sun=score_sun,
        margin=abs(score_shadow - score_sun),
        n_samples=len(per),
        per_sample=per,
    )
