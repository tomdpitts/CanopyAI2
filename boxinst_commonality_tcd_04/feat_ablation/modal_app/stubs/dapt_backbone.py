"""Modal stub for dapt.backbone — features are precomputed, so the DINOv3
backbone (and its transformers dependency) is never needed at train/eval time.
Only pick_device is actually called; FrozenDinoV3Features exists so that
cache_test/cache_train_tiles (imported for their cache_dir/OUT constants)
import cleanly, and raises if anything tries to extract features."""
from __future__ import annotations

import torch

LAYERS = (3, 6, 9, 12)
GRID = 512
MODEL_IDS = {}


def pick_device(pref=None):
    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_tile(*a, **k):
    raise RuntimeError("dapt.backbone is stubbed on Modal (precomputed features)")


class FrozenDinoV3Features:
    def __init__(self, *a, **k):
        raise RuntimeError(
            "dapt.backbone is stubbed on Modal (precomputed features)")
