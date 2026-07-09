"""Train the linear head on TCD train crops, eval on a TCD split -> area F1/IoU.

Frozen DINOv3 + linear head; sliding-window inference stitched back to tile res.
First runnable experiment of the dino/ thread. Instance mAP50 (run_instance.py)
is the next rung and reuses dino/eval.py:coco_map50.

NOTE: untested until gated DINOv3 weights are available (see dino/README.md).

Usage (once HF_TOKEN is set and license accepted):
  .venv/bin/python dino/run_semantic.py \
      --model facebook/dinov3-vitl16-pretrain-sat493m \
      --eval-split test --train-tiles 300 --steps 800
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tcd_data import load_split  # noqa: E402
from eval import aggregate_semantic, semantic_counts  # noqa: E402
from dinov3_seg import Dinov3Seg  # noqa: E402

SPLIT_DIR = {"test": "data/tcd/test", "dryland": "data/tcd/experimental/dryland",
             "sparse": "data/tcd/experimental/sparse"}


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0  # HxWx3


def rand_crop(rgb, mask, size, rng):
    H, W = mask.shape
    y = rng.integers(0, max(1, H - size)); x = rng.integers(0, max(1, W - size))
    return (rgb[y:y + size, x:x + size], mask[y:y + size, x:x + size])


def train_head(model, tiles, size, steps, lr, device, seed=0):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    model.head.train()
    for it in range(steps):
        t = tiles[rng.integers(len(tiles))]
        rgb = read_rgb(t.image_path); m = t.semantic_mask().astype(np.int64)
        cr, cm = rand_crop(rgb, m, size, rng)
        x = torch.from_numpy(cr.transpose(2, 0, 1))[None].to(device)
        y = torch.from_numpy(cm)[None].to(device)
        logits = model(x)
        loss = lossf(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0:
            print(f"[train] step {it:4d}/{steps} loss={loss.item():.4f}", flush=True)


@torch.no_grad()
def eval_tile(model, rgb, size, device):
    """Sliding-window logits, stitched -> argmax pred mask (HxW)."""
    H, W = rgb.shape[:2]
    acc = np.zeros((2, H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    ys = list(range(0, max(1, H - size) + 1, size)) or [0]
    xs = list(range(0, max(1, W - size) + 1, size)) or [0]
    if ys[-1] != H - size: ys.append(H - size)
    if xs[-1] != W - size: xs.append(W - size)
    model.head.eval()
    for y in ys:
        for x in xs:
            cr = rgb[y:y + size, x:x + size]
            xt = torch.from_numpy(cr.transpose(2, 0, 1))[None].to(device)
            lg = model(xt)[0].cpu().numpy()
            acc[:, y:y + size, x:x + size] += lg
            cnt[y:y + size, x:x + size] += 1
    return (acc[1] > acc[0]).astype(bool) & (cnt > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-sat493m")
    ap.add_argument("--eval-split", default="test", choices=list(SPLIT_DIR))
    ap.add_argument("--train-tiles", type=int, default=300)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-limit", type=int, default=0, help="cap eval tiles (0=all); for quick checks")
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.time()

    model = Dinov3Seg(a.model, device=device)
    train_tiles = load_split("data/tcd/train")[: a.train_tiles]
    eval_tiles = load_split(SPLIT_DIR[a.eval_split])
    if a.eval_limit:
        eval_tiles = eval_tiles[: a.eval_limit]
    print(f"[data] train={len(train_tiles)} eval({a.eval_split})={len(eval_tiles)} device={device}", flush=True)

    train_head(model, train_tiles, a.crop, a.steps, a.lr, device)

    counts = []
    for i, t in enumerate(eval_tiles):
        pred = eval_tile(model, read_rgb(t.image_path), a.crop, device)
        counts.append(semantic_counts(pred, t.semantic_mask()))
        if i % 50 == 0:
            print(f"[eval] {i}/{len(eval_tiles)}", flush=True)
    res = aggregate_semantic(counts)

    out = {"model": a.model, "eval_split": a.eval_split, "crop": a.crop,
           "train_tiles": len(train_tiles), "steps": a.steps,
           "micro_f1": res["micro_f1"], "micro_iou": res["micro_iou"],
           "wall_clock_s": round(time.time() - t0, 1)}
    os.makedirs("dino/artifacts", exist_ok=True)
    name = a.model.split("/")[-1]
    json.dump(out, open(f"dino/artifacts/semantic_{name}_{a.eval_split}.json", "w"), indent=2)
    print(f"\n>>> {a.model} on {a.eval_split}: F1={res['micro_f1']:.4f} IoU={res['micro_iou']:.4f} "
          f"({out['wall_clock_s']}s)", flush=True)


if __name__ == "__main__":
    main()
