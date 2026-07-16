"""Cache stitched 128x128 features + GT for a sample of full 2048 TRAIN tiles.

Train on whole tiles, exactly like the 439 test — no windowing, no crop filters
(those filters were the distribution gap), no train/test grid-size mismatch.
Each train tile is cached the same way as a test tile: 2x2 of 1024 windows ->
(4096,128,128) fp16. GT = all ITC boxes (category 2 poly extents) + canopy polys
(category 1, ignore). Sample N tiles (default 900 ~ 121GB) from the ITC-bearing
train tiles.

Products:
  cache/<arm>/feat_traintile/<tid>.npy
  train_tiles_gt.json  {tid: {boxes:[xyxy@2048], canopy:[poly@2048], partition}}

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.cache_train_tiles --n 900
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

from dapt.backbone import FrozenDinoV3Features
from boxinst.cache_feats import LAYERS
from boxinst_commonality_tcd_04.cache_test import tile_feature
from boxinst_commonality_tcd_04.prepare_test import OUT, REPO

TCD_TRAIN = os.path.join(REPO, "data/tcd/train")
Image.MAX_IMAGE_PIXELS = None


def cache_dir(arm):
    return os.path.join(OUT, "cache", arm, "feat_traintile")


def anns(meta, cat):
    return [a for a in json.loads(meta["coco_annotations"])
            if a["category_id"] == cat]


def poly_ring(seg):
    if isinstance(seg, dict) or not seg:
        return []
    return [round(float(v), 1) for v in
            np.asarray(max(seg, key=len), np.float32).reshape(-1)]


def build_gt(n, seed):
    """Sample n ITC-bearing train tiles, 88/12 train/val by tile. Returns gt dict."""
    metas = sorted(f for f in os.listdir(TCD_TRAIN) if f.endswith("_meta.json"))
    rng = np.random.default_rng(seed)
    rng.shuffle(metas)
    gt = {}
    for mf in metas:
        if len(gt) >= n:
            break
        meta = json.load(open(os.path.join(TCD_TRAIN, mf)))
        if meta.get("width") != 2048:
            continue
        trees = [a for a in anns(meta, 2) if a.get("iscrowd", 0) == 0]
        if not trees:
            continue
        tid = mf.replace("_meta.json", "")
        boxes = [[round(a["bbox"][0], 1), round(a["bbox"][1], 1),
                  round(a["bbox"][0] + a["bbox"][2], 1),
                  round(a["bbox"][1] + a["bbox"][3], 1)] for a in trees]
        canopy = [p for p in (poly_ring(a["segmentation"]) for a in anns(meta, 1))
                  if len(p) >= 6]
        gt[tid] = {"boxes": boxes, "canopy": canopy}
    tids = sorted(gt)
    rng2 = np.random.default_rng(seed + 1)
    rng2.shuffle(tids)
    n_val = int(0.12 * len(tids))
    for i, t in enumerate(tids):
        gt[t]["partition"] = "val" if i < n_val else "train"
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="web")
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    gt_path = os.path.join(OUT, "train_tiles_gt.json")
    if os.path.exists(gt_path):
        gt = json.load(open(gt_path))
    else:
        gt = build_gt(args.n, args.seed)
        json.dump(gt, open(gt_path, "w"))
    import collections
    print(f"train tiles: {len(gt)} "
          f"({collections.Counter(v['partition'] for v in gt.values())}), "
          f"{sum(len(v['boxes']) for v in gt.values())} ITC boxes", flush=True)

    cdir = cache_dir(args.arm)
    os.makedirs(cdir, exist_ok=True)
    todo = [t for t in sorted(gt) if not os.path.exists(os.path.join(cdir, t + ".npy"))]
    print(f"[cache_train_tiles:{args.arm}] {len(todo)}/{len(gt)} to extract",
          flush=True)
    if not todo:
        return
    net = FrozenDinoV3Features(args.arm, layers=LAYERS, device=args.device)
    for i, tid in enumerate(todo):
        img = Image.open(os.path.join(TCD_TRAIN, tid + ".tif")).convert("RGB")
        np.save(os.path.join(cdir, tid + ".npy"), tile_feature(net, img, net.device))
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            print(f"  {i+1}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
