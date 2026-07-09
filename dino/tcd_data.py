"""OAM-TCD tile adapter: `<stem>_meta.json` -> semantic mask + crown instances.

Backbone-independent. Each meta carries `coco_annotations` as a *JSON-encoded
string* (must `json.loads` it a second time). Categories: **1 = canopy regions,
2 = individual tree crowns**. The `segmentation` field is a mix of polygon lists
and uncompressed RLE, so decoding goes through pycocotools (a polygon-only
rasteriser silently drops the RLE regions -- see data/tcd/experimental/sparse).

This module deliberately handles labels only; pixel reading lives in the run
script so the data layer can be validated with no image I/O and no model weights.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pycocotools.mask as mask_util

CAT_CANOPY = 1
CAT_CROWN = 2


def _ann_to_rle(seg, h, w):
    """COCO segmentation (polygon list | uncompressed RLE | compressed RLE) -> RLE."""
    if isinstance(seg, list):                       # polygon(s)
        return mask_util.merge(mask_util.frPyObjects(seg, h, w))
    if isinstance(seg.get("counts"), list):         # uncompressed RLE
        return mask_util.frPyObjects(seg, h, w)
    return seg                                       # already compressed RLE


def ann_to_mask(seg, h, w):
    """Decode a single annotation's segmentation to a boolean HxW mask."""
    return mask_util.decode(_ann_to_rle(seg, h, w)).astype(bool)


@dataclass
class TcdTile:
    stem: str
    image_path: str
    meta_path: str
    width: int
    height: int
    biome: int
    anns: list = field(repr=False)

    @property
    def crowns(self):
        return [a for a in self.anns if a["category_id"] == CAT_CROWN]

    @property
    def canopy(self):
        return [a for a in self.anns if a["category_id"] == CAT_CANOPY]

    def semantic_mask(self, categories=(CAT_CANOPY, CAT_CROWN)):
        """Union of decoded masks over `categories` -> boolean HxW tree-cover mask."""
        m = np.zeros((self.height, self.width), dtype=bool)
        for a in self.anns:
            if a["category_id"] in categories:
                m |= ann_to_mask(a["segmentation"], self.height, self.width)
        return m

    def canopy_frac(self):
        """cat=1 union area / tile area -- reproduces the recorded provenance metric."""
        return float(self.semantic_mask((CAT_CANOPY,)).mean())


def load_tile(meta_path):
    m = json.load(open(meta_path))
    stem = os.path.basename(meta_path).replace("_meta.json", "")
    return TcdTile(
        stem=stem,
        image_path=meta_path.replace("_meta.json", ".tif"),
        meta_path=meta_path,
        width=m["width"],
        height=m["height"],
        biome=m.get("biome", -1),
        anns=json.loads(m["coco_annotations"]),
    )


def load_split(meta_dir, stems=None, warn=True):
    """Load all tiles under `meta_dir`; skip dangling symlinks / missing tifs.

    Optionally restrict to `stems` (iterable). Returns only tiles whose meta and
    .tif both resolve on disk (experimental subsets are symlink farms that can
    dangle, e.g. dryland -> ../by_id which may be absent).
    """
    paths = sorted(glob.glob(os.path.join(meta_dir, "*_meta.json")))
    kept, skipped = [], 0
    for p in paths:
        tif = p.replace("_meta.json", ".tif")
        if os.path.exists(p) and os.path.exists(tif):   # follows symlinks; False if dangling
            kept.append(load_tile(p))
        else:
            skipped += 1
    if skipped and warn:
        print(f"[load_split] {meta_dir}: skipped {skipped}/{len(paths)} unresolved tiles")
    if stems is not None:
        keep = set(stems)
        kept = [t for t in kept if t.stem in keep]
    return kept
