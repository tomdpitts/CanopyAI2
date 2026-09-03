"""Derive a BOX-ONLY COCO from the DetecTree2 baseline's already-built datasets.

The point of this file is that the box-supervised baselines train on the *identical* data
the DetecTree2 row trained on -- same 792/108 tiles, same 1024@50% subtile crops, the same
PNG files on `tcd-detectree2-vol` -- with the crown polygons removed and nothing else
touched. Nothing is re-tiled and no image is re-encoded, so "same training dataset" is a
property of the file system rather than a claim about two builders agreeing.

WHAT IS REMOVED
    `segmentation` is replaced by the 4-point RECTANGLE of the annotation's own `bbox`, and
    `area` by the rectangle's area. The polygon is not merely dropped: mmdet's CocoDataset
    requires the field, and a rectangle is the standard box-supervised stand-in. Replacing
    rather than deleting also makes the box-only property *checkable* -- `verify()` below
    asserts every mask in the emitted file is its own bounding rectangle, so a config that
    silently fell back to mask supervision cannot produce a good number by accident.

WHAT IS KEPT
    `bbox` (tight box of the crown clipped to the subtile, as built by
    `detectree2_baseline/build_coco.py`), and `iscrowd=1` on canopy so that closed-canopy
    regions stay IGNORE rather than becoming background negatives -- the same treatment
    LACE, the scorer, and the DetecTree2 row all give canopy. Getting this wrong would
    teach a baseline that canopy is background and then charge it for the consequences.

MODEL SELECTION
    Do NOT select on segm AP against this file: the "masks" are rectangles, so segm AP here
    measures nothing. Select on val **box** AP50, which is what LACE's own detector selects
    on (`HYPERPARAMS.md` §3, "Model selection | best val box mAP50"). Final masks are scored
    only after stitching to 2048, through `score_coco.py`.

Usage (locally, on pulled JSONs, or on Modal against the volume):
    python -m boxinst_commonality_tcd_04.box_supervised_baselines.make_box_only_coco \
        --data_dir /vol/data --splits train val
"""
from __future__ import annotations

import argparse
import copy
import json
import os

SPLITS = ("train", "val")          # test is scored from stitched 2048 preds, never trained on


def box_only_ann(a):
    """One COCO annotation -> the same annotation with its mask replaced by its box."""
    x, y, w, h = [float(v) for v in a["bbox"]]
    out = copy.deepcopy(a)
    out["segmentation"] = [[x, y, x + w, y, x + w, y + h, x, y + h]]
    out["area"] = w * h
    return out


def convert(coco):
    d = {"images": coco["images"], "categories": coco["categories"],
         "annotations": [box_only_ann(a) for a in coco["annotations"]]}
    d["info"] = {"supervision": "boxes only",
                 "note": "segmentation is the bbox rectangle; no crown polygon is present. "
                         "canopy retains iscrowd=1 (ignore, never background)."}
    return d


def verify(coco):
    """Assert no mask information survives: every polygon must BE its own bounding box.

    Cheap, and it is the guard that lets the paper say 'box-only' without hedging."""
    n_crowd = 0
    for a in coco["annotations"]:
        n_crowd += int(a.get("iscrowd", 0) == 1)
        seg = a["segmentation"]
        assert isinstance(seg, list) and len(seg) == 1 and len(seg[0]) == 8, \
            f"ann {a['id']}: segmentation is not a single 4-point polygon"
        xs, ys = seg[0][0::2], seg[0][1::2]
        x, y, w, h = a["bbox"]
        assert (min(xs), min(ys), max(xs), max(ys)) == (x, y, x + w, y + h), \
            f"ann {a['id']}: polygon is not its own bbox -- mask information survived"
    return {"n_images": len(coco["images"]), "n_anns": len(coco["annotations"]),
            "n_ignore_canopy": n_crowd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="dir holding {split}/coco.json (DetecTree2's built datasets)")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--out_name", default="coco_boxonly.json")
    a = ap.parse_args()

    for split in a.splits:
        src = os.path.join(a.data_dir, split, "coco.json")
        dst = os.path.join(a.data_dir, split, a.out_name)
        d = convert(json.load(open(src)))
        stats = verify(d)
        json.dump(d, open(dst, "w"))
        print(f"[box-only] {split}: {stats['n_images']} subtiles · {stats['n_anns']} anns "
              f"({stats['n_ignore_canopy']} canopy ignore) -> {dst}", flush=True)


if __name__ == "__main__":
    main()
