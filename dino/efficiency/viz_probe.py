"""Visualise the frozen DINOv3 web ViT-L 1x1 linear probe on TCD holdout tiles.

Retrains the same 1x1 semantic probe as the bake-off (web ViT-L, 300 tiles/800
steps -> ~0.874 F1), then renders N holdout tiles as RGB | prediction | GT panels
with per-tile F1. PIL-only (no matplotlib).
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))          # dino/
ROOT = os.path.join(HERE, "..", "..")
import torch  # noqa: E402
from dinov3_seg import Dinov3Seg  # noqa: E402
from tcd_data import load_split  # noqa: E402
from run_semantic import read_rgb, train_head, eval_tile  # noqa: E402
from eval import semantic_counts  # noqa: E402

PANEL = 380
PAD = 6
N = 10
CROP = 512


def overlay(rgb, mask, color, a=0.45):
    out = rgb.astype(np.float32).copy()
    out[mask] = (1 - a) * out[mask] + a * np.array(color, np.float32)
    return out.astype(np.uint8)


def small(arr):
    return np.asarray(Image.fromarray(arr).resize((PANEL, PANEL), Image.BILINEAR))


def f1(pred, gt):
    c = semantic_counts(pred, gt)
    return 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"] + 1e-9)


MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
CKPT = os.path.join(HERE, "ckpt", "probe_web_1x1.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load", action="store_true", help="load saved head instead of training")
    a = p.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Dinov3Seg(MODEL_ID, device=dev)
    if a.load and os.path.exists(CKPT):
        model.head.load_state_dict(torch.load(CKPT, map_location=dev)["head"])
        print(f"[viz] loaded saved head from {CKPT}", flush=True)
    else:
        train = load_split(os.path.join(ROOT, "data/tcd/train"))[:300]
        print(f"[viz] training 1x1 probe on {len(train)} tiles ({dev})...", flush=True)
        train_head(model, train, CROP, 800, 1e-3, dev)
        os.makedirs(os.path.dirname(CKPT), exist_ok=True)
        torch.save({"head": model.head.state_dict(), "model_id": MODEL_ID,
                    "crop": CROP, "in_channels": 3, "num_classes": 2,
                    "train_tiles": len(train), "steps": 800}, CKPT)
        print(f"[viz] saved trained head -> {CKPT}", flush=True)

    # pick N holdout tiles with meaningful tree coverage, spread across the split
    test = load_split(os.path.join(ROOT, "data/tcd/test"))
    picks = []
    for t in test[::7]:
        gt = t.semantic_mask()
        fg = gt.mean()
        if 0.10 <= fg <= 0.75:
            picks.append((t, gt))
        if len(picks) == N:
            break
    print(f"[viz] selected {len(picks)} tiles", flush=True)

    rows = []
    for i, (t, gt) in enumerate(picks):
        rgb = (read_rgb(t.image_path) * 255).astype(np.uint8)
        pred = eval_tile(model, rgb.astype(np.float32) / 255.0, CROP, dev)
        sc = f1(pred, gt)
        rgbS = small(rgb)
        predS = small(overlay(rgb, pred, (255, 40, 40)))
        gtS = small(overlay(rgb, gt, (40, 220, 40)))
        sep = np.full((PANEL, PAD, 3), 255, np.uint8)
        row = np.concatenate([rgbS, sep, predS, sep, gtS], 1)
        rows.append((row, t.stem, sc))
        print(f"[viz] {i+1}/{N} {t.stem} F1={sc:.3f}", flush=True)

    hsep = np.full((PAD, rows[0][0].shape[1], 3), 255, np.uint8)
    stacked = []
    for row, stem, sc in rows:
        stacked.append(row); stacked.append(hsep)
    canvas = np.concatenate(stacked[:-1], 0)
    img = Image.fromarray(canvas)
    d = ImageDraw.Draw(img)
    y = 2
    for row, stem, sc in rows:
        d.text((4, y + 2), f"{stem[:34]}  F1={sc:.2f}", fill=(255, 255, 0))
        d.text((4, y + 14), "RGB", fill=(255, 255, 0))
        d.text((PANEL + PAD + 4, y + 14), "PRED (red)", fill=(255, 255, 0))
        d.text((2 * (PANEL + PAD) + 4, y + 14), "GT (green)", fill=(255, 255, 0))
        y += PANEL + PAD
    os.makedirs(os.path.join(HERE, "viz"), exist_ok=True)
    out = os.path.join(HERE, "viz", "probe_preds.png")
    img.save(out)
    print(f"\n>>> saved {out}  mean F1 (these {len(rows)} tiles)="
          f"{np.mean([r[2] for r in rows]):.3f}", flush=True)


if __name__ == "__main__":
    main()
