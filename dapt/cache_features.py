"""Cache frozen DINOv3 multi-layer features once per (arm, tile).

Features are identical inputs to every probe/head fit, so compute them once. Stored
float16 to halve disk (feature values are already L2-normed, fp16 is ample).

Usage:
    .venv/bin/python -m dapt.cache_features --arm web
    .venv/bin/python -m dapt.cache_features --arm sat
"""
import argparse
import json
import os

import numpy as np
import torch

from dapt.backbone import FrozenDinoV3Features, load_tile
from dapt.data.cohort import REPO


def cache_key(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def cache_arm(arm: str, split_path: str, out_root: str, device=None):
    split = json.load(open(split_path))
    net = FrozenDinoV3Features(arm, device=device)
    out_dir = os.path.join(out_root, arm)
    os.makedirs(out_dir, exist_ok=True)
    paths = list(split["tiles"])
    print(f"[{arm}] {len(paths)} tiles -> {os.path.relpath(out_dir, REPO)} "
          f"(device={net.device}, out_dim={net.out_dim})")
    for i, rel in enumerate(paths):
        dst = os.path.join(out_dir, cache_key(rel) + ".npy")
        if os.path.exists(dst):
            continue
        abspath = rel if os.path.isabs(rel) else os.path.join(REPO, rel)
        x, _ = load_tile(abspath)
        feat = net.extract(x)[0].to(torch.float16).cpu().numpy()   # (C,32,32)
        np.save(dst, feat)
        if (i + 1) % 25 == 0 or i + 1 == len(paths):
            print(f"  {i+1}/{len(paths)}")
    # record provenance
    meta = {"arm": arm, "model_id": net.__class__.__module__, "out_dim": net.out_dim,
            "layers": list(net.layers), "n_tiles": len(paths)}
    json.dump(meta, open(os.path.join(out_dir, "_meta.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--split", default="dapt/data/split.json")
    ap.add_argument("--out", default="dapt/cache")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    split_path = args.split if os.path.isabs(args.split) else os.path.join(REPO, args.split)
    out_root = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    cache_arm(args.arm, split_path, out_root, args.device)


if __name__ == "__main__":
    main()
