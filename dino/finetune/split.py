"""Leakage-safe train/val split for TCD fine-tuning.

OAM-TCD train tiles cluster spatially (up to ~29 tiles within a 1 km bin), so a
random *tile* split leaks near-duplicate neighbours into val and inflates it (we
measured val 0.9235 vs a true 0.864 on the geographically-separated 439 holdout).

Fix: group tiles by a spatial grid bin (from each tile's projected bounds) and hold
out **whole bins** for val, so val tiles are not spatial neighbours of train tiles.
This mirrors how the official test split is separated, so val tracks test.

NEVER uses the 439 test tiles — selection stays entirely within train.
"""
import json

import numpy as np


def fold_split(tiles, val_fold=0):
    """Official Restor split: val = tiles whose meta `validation_fold` == val_fold,
    train = the other 4 folds. Purpose-built (geographically grouped) so val is
    leakage-safe. The 439 test holdout (no validation_fold) is never included here.
    """
    train, val = [], []
    for t in tiles:
        f = json.load(open(t.meta_path)).get("validation_fold")
        (val if f == val_fold else train).append(t)
    return train, val, {"val_fold": val_fold, "n_train": len(train), "n_val": len(val)}


def tile_bin(tile, km):
    b = json.load(open(tile.meta_path))["bounds"]     # [minx,miny,maxx,maxy] in metres (EPSG:3395)
    cx = (b[0] + b[2]) / 2.0
    cy = (b[1] + b[3]) / 2.0
    return (int(cx // (km * 1000)), int(cy // (km * 1000)))


def grouped_split(tiles, val_frac=0.15, km=2.0, seed=0):
    """Hold out whole spatial bins for val until ~val_frac of tiles is reached."""
    bins = {}
    for t in tiles:
        bins.setdefault(tile_bin(t, km), []).append(t)
    keys = sorted(bins)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    target = int(val_frac * len(tiles))
    val_keys, val = set(), []
    for k in keys:
        if len(val) >= target:
            break
        val_keys.add(k)
        val += bins[k]
    train = [t for k in keys if k not in val_keys for t in bins[k]]
    return train, val, {"n_bins": len(bins), "val_bins": len(val_keys),
                        "km": km, "n_train": len(train), "n_val": len(val)}
