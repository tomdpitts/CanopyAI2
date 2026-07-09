"""Calibrate the SOTA bar: run restor SegFormer through MY eval protocol.

"Beat 0.902" is only meaningful in a matched protocol. This scores the published
restor/tcd-segformer (the model that reports ~0.902 area-F1 for mit-b5) with the
exact same foreground definition (cat1 u cat2) and sliding-window-512 inference I
use for the DINOv3 probe -- so the DINOv3 number and the SegFormer number live on
one ruler. Prints F1/IoU; a gap here is a real gap.
"""
import argparse
import os
import sys

import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))        # dino/
from tcd_data import load_split  # noqa: E402
from eval import aggregate_semantic, semantic_counts  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@torch.no_grad()
def seg_tile(model, rgb, tree_idx, size, device):
    H, W = rgb.shape[:2]
    acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    ys = sorted(set(list(range(0, max(1, H - size) + 1, size)) + [max(0, H - size)]))
    xs = sorted(set(list(range(0, max(1, W - size) + 1, size)) + [max(0, W - size)]))
    for y in ys:
        for x in xs:
            cr = (rgb[y:y + size, x:x + size].astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            xt = torch.from_numpy(cr.transpose(2, 0, 1))[None].to(device)
            logits = model(pixel_values=xt).logits            # (1,C,h/4,w/4)
            up = torch.nn.functional.interpolate(logits, size=cr.shape[:2], mode="bilinear",
                                                 align_corners=False)[0]
            prob = torch.softmax(up, 0)[tree_idx].cpu().numpy()
            acc[y:y + size, x:x + size] += prob
            cnt[y:y + size, x:x + size] += 1
    return (acc / np.maximum(cnt, 1)) > 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="restor/tcd-segformer-mit-b5")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SegformerForSemanticSegmentation.from_pretrained(a.model).eval().to(dev)
    id2label = model.config.id2label
    tree_idx = next((i for i, l in id2label.items() if "tree" in l.lower() or "canopy" in l.lower()), 1)
    tree_idx = int(tree_idx)
    print(f"[calib] {a.model} labels={id2label} tree_idx={tree_idx} dev={dev}", flush=True)

    tiles = load_split(os.path.join(HERE, "..", "..", "..", "data/tcd/test"))
    if a.limit:
        tiles = tiles[: a.limit]
    from PIL import Image
    counts = []
    for i, t in enumerate(tiles):
        rgb = np.asarray(Image.open(t.image_path).convert("RGB"))
        pred = seg_tile(model, rgb, tree_idx, a.size, dev)
        counts.append(semantic_counts(pred, t.semantic_mask()))
        if i % 50 == 0:
            print(f"[calib] {i}/{len(tiles)}", flush=True)
    res = aggregate_semantic(counts)
    print(f"\n>>> {a.model} in MY protocol ({len(tiles)} tiles): "
          f"F1={res['micro_f1']:.4f} IoU={res['micro_iou']:.4f}", flush=True)


if __name__ == "__main__":
    main()
