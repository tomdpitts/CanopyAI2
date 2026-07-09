"""Directional shadow-prior feature + ablation harness for tree-crown segmentation.

See README.md for the experimental design. This package emits a clean ``(C,H,W)``
feature stack and the validation/statistics scaffolding around it; the detector
itself is out of scope (design conventions).
"""

from .config import ShadowFeatureConfig, FoldConfig, DatasetConfig
from .shadow_feature import compute_shadow_feature, shuffle_azimuths
from .geometry import azimuth_to_vector, rotate_azimuth, vector_to_azimuth
from .verify import recommend_convention, crown_response_in_mask

__all__ = [
    "ShadowFeatureConfig",
    "FoldConfig",
    "DatasetConfig",
    "compute_shadow_feature",
    "shuffle_azimuths",
    "azimuth_to_vector",
    "rotate_azimuth",
    "vector_to_azimuth",
    "recommend_convention",
    "crown_response_in_mask",
]
