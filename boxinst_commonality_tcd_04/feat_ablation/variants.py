"""Variant feature access shared by the train/eval wrappers.

pca256    : materialized fp16 cache under feat_ablation/cache/pca256/.
block1024 : the LAST 1024 of the parent 4096 channels (deepest ViT block, layer
            24 of LAYERS=(21,22,23,24)). Channels-first .npy means the slice is
            contiguous on disk, so an mmap slice reads exactly the bytes a
            materialized 1024-dim cache would — no new storage, parent untouched.
full4096  : the parent cache as-is (baseline sanity checks).
"""
import os

import numpy as np

from boxinst_commonality_tcd_04.cache_train_tiles import cache_dir as train_cache
from boxinst_commonality_tcd_04.cache_test import cache_dir as test_cache

HERE = os.path.abspath(os.path.dirname(__file__))
VARIANTS = ("full4096", "block1024", "pca256")
BLOCK_LO = 3072                                   # 4096 - 1024


def feat_dir(variant, split):                     # split: feat_traintile|feat_test
    if variant == "pca256":
        return os.path.join(HERE, "cache", "pca256", split)
    return train_cache("web") if split == "feat_traintile" else test_cache("web")


def load_feat(variant, split, tid):
    """-> (C,128,128) fp16 ndarray for the variant."""
    fp = os.path.join(feat_dir(variant, split), tid + ".npy")
    if variant == "block1024":
        return np.ascontiguousarray(np.load(fp, mmap_mode="r")[BLOCK_LO:])
    return np.load(fp)
