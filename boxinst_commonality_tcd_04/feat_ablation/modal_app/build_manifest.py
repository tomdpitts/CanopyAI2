"""Build the tile<->image_id join manifest for the 900+439 baseline tiles.

The Modal extraction reads pixels from the HF restor/tcd dataset (already on the
canopyai-deepforest-data volume), NOT by filename but by image_id — the stable
key in each tile's meta.json AND in the HF rows. This manifest pins exactly which
tiles the 0.555 baseline used and lets the Modal side assert an exact join:

  per tile: image_id, width/height, n_cat2 (ITC box count from meta's
  coco_annotations), and — for a sample — a sha1 of the decoded RGB pixels so the
  Modal side can prove HF image == local tile, not merely image_id-equal.

Writes manifest.json next to this file (small, safe to upload).

Usage:
    .venv/bin/python boxinst_commonality_tcd_04/feat_ablation/modal_app/build_manifest.py
"""
import hashlib
import json
import os

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
REPO = "/Users/tompitts/dphil/CanopyAI2"
PKG = os.path.join(REPO, "boxinst_commonality_tcd_04")
HERE = os.path.abspath(os.path.dirname(__file__))
SAMPLE = 20                                       # pixel-hash the first N of each split


def n_cat2(meta):
    return sum(a["category_id"] == 2
               for a in json.loads(meta["coco_annotations"]))


def build(split, tiles_dir, tids):
    out = {}
    for i, tid in enumerate(sorted(tids)):
        meta = json.load(open(os.path.join(tiles_dir, tid + "_meta.json")))
        rec = {"image_id": int(meta["image_id"]),
               "width": int(meta["width"]), "height": int(meta["height"]),
               "n_cat2": n_cat2(meta)}
        if i < SAMPLE:
            arr = np.asarray(Image.open(
                os.path.join(tiles_dir, tid + ".tif")).convert("RGB"))
            rec["rgb_sha1"] = hashlib.sha1(arr.tobytes()).hexdigest()
            rec["rgb_shape"] = list(arr.shape)
        out[tid] = rec
    return out


def main():
    train_gt = json.load(open(os.path.join(PKG, "train_tiles_gt.json")))
    test_gt = json.load(open(os.path.join(PKG, "test_gt.json")))
    man = {
        "feat_traintile": build("train", os.path.join(REPO, "data/tcd/train"),
                                list(train_gt)),
        "feat_test": build("test", os.path.join(REPO, "data/tcd/test"),
                           list(test_gt)),
    }
    # image_id uniqueness within and across splits (join must be 1:1)
    ids_tr = [r["image_id"] for r in man["feat_traintile"].values()]
    ids_te = [r["image_id"] for r in man["feat_test"].values()]
    assert len(ids_tr) == len(set(ids_tr)), "duplicate image_id in train"
    assert len(ids_te) == len(set(ids_te)), "duplicate image_id in test"
    dup = set(ids_tr) & set(ids_te)
    print(f"train tiles={len(ids_tr)} test tiles={len(ids_te)} "
          f"train/test image_id overlap={len(dup)}")
    out = os.path.join(HERE, "manifest.json")
    json.dump(man, open(out, "w"))
    print(f"-> {out}  ({os.path.getsize(out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
