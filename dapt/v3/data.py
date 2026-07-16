"""v3 cohort: arid-only (WON+BRU), basename-keyed, folder-based leakage-safe split.

data/finetune/v3/{train,test}/*.png + annotations.csv (basename,xyxy,label,domain).
- test/  = 81 tiles DEFINITIVELY EXCLUDED from the DAPT SSL orthos -> the only tiles
  valid to EVALUATE the DAPT arm on (leakage-safe). (Was 62; the 19 BRU tiles once
  in train/ come from the L/R strips outside the pool's center-80 crop -> moved
  2026-07-14.)
- train/ = 16 WON tiles that overlap the SSL pool (WON right50) -> probe-TRAIN only,
  never DAPT-arm eval.
- Rows whose basename is in neither folder are stale (259 of them) -> dropped on load.

Features cache to dapt/v3/cache/<arm>/<basename>.npy (reuses the frozen extractor).
Targets reuse dapt.targets.encode; tiles load+pad via dapt.backbone.load_tile.
"""
import csv
import os
from collections import defaultdict

import numpy as np
import torch

from dapt.targets import CFG, encode

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
V3_DIR = os.path.join(REPO, "data/finetune/v3")
CACHE = os.path.join(REPO, "dapt/v3/cache")


def _folder_index():
    """basename -> 'train'|'test' for every image actually present on disk."""
    idx = {}
    for folder in ("train", "test"):
        d = os.path.join(V3_DIR, folder)
        for f in os.listdir(d):
            if f.lower().endswith((".png", ".tif", ".jpg")):
                idx[f] = folder
    return idx


def load_v3():
    """basename -> {folder, domain, boxes (N,4)}. Stale rows (not on disk) dropped."""
    idx = _folder_index()
    boxes = defaultdict(list)
    domain = {}
    dropped = 0
    with open(os.path.join(V3_DIR, "annotations.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            b = os.path.basename(row["image_path"].strip())
            if b not in idx:
                dropped += 1
                continue
            domain[b] = row["domain"].strip()
            if row["xmin"].strip() != "":
                boxes[b].append([float(row["xmin"]), float(row["ymin"]),
                                 float(row["xmax"]), float(row["ymax"])])
    tiles = {}
    for b, folder in idx.items():
        tiles[b] = {"folder": folder, "domain": domain.get(b, "?"),
                    "boxes": np.asarray(boxes.get(b, []), np.float32).reshape(-1, 4)}
    return tiles, dropped


class V3Data:
    """Feature + target access for one arm over the v3 tiles (mirrors dapt.dataset)."""

    def __init__(self, arm):
        self.arm = arm
        self.tiles, self.dropped = load_v3()
        self.cache_dir = os.path.join(CACHE, arm)

    def leakage_safe(self):
        return sorted(b for b, t in self.tiles.items() if t["folder"] == "test")

    def overlap(self):
        return sorted(b for b, t in self.tiles.items() if t["folder"] == "train")

    def img_path(self, b):
        return os.path.join(V3_DIR, self.tiles[b]["folder"], b)

    def _feat(self, b):
        f = np.load(os.path.join(self.cache_dir, os.path.splitext(b)[0] + ".npy"))
        return torch.from_numpy(f.astype(np.float32))

    def batch(self, names):
        G = CFG.grid
        feats, hm, off, size, mask = [], [], [], [], []
        for b in names:
            feats.append(self._feat(b))
            enc = encode(self.tiles[b]["boxes"])
            hm.append(torch.from_numpy(enc["heatmap"]))
            off.append(torch.from_numpy(enc["offset"]))
            size.append(torch.from_numpy(enc["size"]))
            mask.append(torch.from_numpy(enc["reg_mask"]))
        return {"feat": torch.stack(feats), "heatmap": torch.stack(hm),
                "offset": torch.stack(off), "size": torch.stack(size),
                "reg_mask": torch.stack(mask), "names": list(names)}


def cache_features_v3(arm, device=None):
    """Extract + cache frozen features for every v3 tile (idempotent). ~free/local."""
    from dapt.backbone import FrozenDinoV3Features, load_tile
    tiles, _ = load_v3()
    net = FrozenDinoV3Features(arm, device=device)
    out = os.path.join(CACHE, arm)
    os.makedirs(out, exist_ok=True)
    todo = [b for b in tiles
            if not os.path.exists(os.path.join(out, os.path.splitext(b)[0] + ".npy"))]
    print(f"[{arm}] caching {len(todo)}/{len(tiles)} v3 tiles -> {out}", flush=True)
    for i, b in enumerate(todo):
        folder = tiles[b]["folder"]
        x, _ = load_tile(os.path.join(V3_DIR, folder, b))
        feat = net.extract(x)[0].to(torch.float16).cpu().numpy()
        np.save(os.path.join(out, os.path.splitext(b)[0] + ".npy"), feat)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(todo)}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", nargs="*", help="arms to cache features for")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        tiles, dropped = load_v3()
        from collections import Counter
        safe = [b for b, t in tiles.items() if t["folder"] == "test"]
        ov = [b for b, t in tiles.items() if t["folder"] == "train"]
        nb = lambda S: sum(len(tiles[b]["boxes"]) for b in S)
        print(f"tiles: {len(tiles)} | leakage-safe(test) {len(safe)} "
              f"({nb(safe)} boxes) | overlap(train) {len(ov)} ({nb(ov)} boxes)")
        print("safe domains:", Counter(tiles[b]["domain"] for b in safe))
        print("stale rows dropped:", dropped)
    for arm in (a.cache or []):
        cache_features_v3(arm)
