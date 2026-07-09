"""Assemble cached features + encoded targets into per-partition tensors.

Reads the frozen feature cache (dapt/cache/<arm>) and the frozen target encoder
(dapt.targets), keyed by the split (dapt/data/split.json). Supports the
label-efficiency subsampling (25/50/100% of train tiles, seeded).
"""
import json
import os

import numpy as np
import torch

from dapt.cache_features import cache_key
from dapt.data.cohort import REPO, load_boxes
from dapt.targets import CFG, encode


class CohortData:
    def __init__(self, arm, split_path="dapt/data/split.json", cache_root="dapt/cache"):
        sp = split_path if os.path.isabs(split_path) else os.path.join(REPO, split_path)
        self.split = json.load(open(sp))
        self.arm = arm
        self.cache_dir = os.path.join(REPO, cache_root, arm)
        self.boxes, _ = load_boxes(self.split["csv"])
        # per-tile records
        self.tiles = {}
        for path, info in self.split["tiles"].items():
            self.tiles[path] = {**info, "boxes": self.boxes.get(path,
                                np.zeros((0, 4), np.float32))}

    def partition(self, name):
        return [p for p, t in self.split["tiles"].items() if t["partition"] == name]

    def _load_feat(self, path):
        f = np.load(os.path.join(self.cache_dir, cache_key(path) + ".npy"))
        return torch.from_numpy(f.astype(np.float32))          # (C,32,32)

    def batch(self, paths):
        """Stack features + targets for the given tiles into a dict of tensors."""
        G = CFG.grid
        feats, hm, off, size, mask = [], [], [], [], []
        for p in paths:
            feats.append(self._load_feat(p))
            enc = encode(self.tiles[p]["boxes"])
            hm.append(torch.from_numpy(enc["heatmap"]))
            off.append(torch.from_numpy(enc["offset"]))
            size.append(torch.from_numpy(enc["size"]))
            mask.append(torch.from_numpy(enc["reg_mask"]))
        return {
            "feat": torch.stack(feats),                        # (B,C,G,G)
            "heatmap": torch.stack(hm),                        # (B,1,G,G)
            "offset": torch.stack(off),                        # (B,2,G,G)
            "size": torch.stack(size),                         # (B,2,G,G)
            "reg_mask": torch.stack(mask),                     # (B,G,G) bool
            "paths": list(paths),
        }

    def train_subset(self, frac, seed):
        """Label-efficiency: a seeded fraction of train tiles (stratified by domain)."""
        from collections import defaultdict
        rng = np.random.default_rng(seed)
        by_dom = defaultdict(list)
        for p in self.partition("train"):
            by_dom[self.tiles[p]["domain"]].append(p)
        chosen = []
        for dom in sorted(by_dom):
            ps = sorted(by_dom[dom])
            rng.shuffle(ps)
            k = max(1, round(len(ps) * frac))
            chosen.extend(ps[:k])
        return chosen
