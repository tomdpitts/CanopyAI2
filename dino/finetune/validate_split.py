"""Validate that the spatially-grouped val tracks the 439 test.

Eval the saved FROZEN 1x1 probe (known 439-test F1 = 0.874) on the grouped-val,
excluding the probe's own 300 training tiles. Representative val -> ~0.874.
A reading near 0.92 would mean the split still leaks.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
ROOT = os.path.join(HERE, "..", "..")
from dinov3_seg import Dinov3Seg  # noqa: E402
from tcd_data import load_split  # noqa: E402
from run_semantic import read_rgb, eval_tile  # noqa: E402
from eval import aggregate_semantic, semantic_counts  # noqa: E402
from split import fold_split  # noqa: E402

CHECK_N = 120


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    all_train = load_split(os.path.join(ROOT, "data/tcd/train"))
    probe_train = set(t.stem for t in all_train[:300])          # frozen probe's train tiles
    train, val, info = fold_split(all_train, val_fold=0)        # Restor official fold 0 as val
    print(f"[split] {info}", flush=True)
    val_clean = [t for t in val if t.stem not in probe_train]
    rng = np.random.default_rng(1)
    rng.shuffle(val_clean)
    check = val_clean[:CHECK_N]
    print(f"[split] val_clean={len(val_clean)} (excluded {len(val)-len(val_clean)} probe-train); "
          f"evaluating frozen probe on {len(check)}", flush=True)

    model = Dinov3Seg("facebook/dinov3-vitl16-pretrain-lvd1689m", device=dev)
    ck = torch.load(os.path.join(HERE, "..", "efficiency", "ckpt", "probe_web_1x1.pt"), map_location=dev)
    model.head.load_state_dict(ck["head"])
    counts = []
    for i, t in enumerate(check):
        pred = eval_tile(model, read_rgb(t.image_path), 512, dev)
        counts.append(semantic_counts(pred, t.semantic_mask()))
        if i % 40 == 0:
            print(f"[split] {i}/{len(check)}", flush=True)
    f1 = aggregate_semantic(counts)["micro_f1"]
    print(f"\n>>> frozen probe on grouped-val: F1={f1:.4f}  (439-test was 0.874; "
          f"random-tile-val leaked to ~0.92)", flush=True)


if __name__ == "__main__":
    main()
