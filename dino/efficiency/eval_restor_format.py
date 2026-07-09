"""Report the DINOv3 1x1 probe on TCD-439 in Restor's format: area F1/IoU + mAP50.

F1/IoU are native to the semantic probe. mAP50 is NOT: the probe outputs a canopy
*area* mask, so we derive instances by connected components (score = mean prob) and
run COCO mask AP@0.50 against the GT crowns (cat2; canopy cat1 as iscrowd ignore).
This CC-mAP50 is an honest LOWER BOUND / illustration -- it under-segments touching
crowns and is NOT comparable to a trained instance model (Restor Mask R-CNN 43.2).
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
import pycocotools.mask as mask_util

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))          # dino/
ROOT = os.path.join(HERE, "..", "..")
from dinov3_seg import Dinov3Seg  # noqa: E402
from tcd_data import load_split  # noqa: E402
from eval import (aggregate_semantic, build_coco_gt, coco_map50,  # noqa: E402
                  semantic_counts)

MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
CKPT = os.path.join(HERE, "ckpt", "probe_web_1x1.pt")
CROP = 512
MIN_AREA = 25   # drop CC specks below this many px


@torch.no_grad()
def prob_tile(model, rgb01, size, device):
    H, W = rgb01.shape[:2]
    acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    ys = sorted(set(list(range(0, max(1, H - size) + 1, size)) + [max(0, H - size)]))
    xs = sorted(set(list(range(0, max(1, W - size) + 1, size)) + [max(0, W - size)]))
    for y in ys:
        for x in xs:
            cr = rgb01[y:y + size, x:x + size]
            xt = torch.from_numpy(cr.transpose(2, 0, 1))[None].to(device)
            p = torch.softmax(model(xt)[0], 0)[1].cpu().numpy()   # P(tree)
            acc[y:y + size, x:x + size] += p
            cnt[y:y + size, x:x + size] += 1
    return acc / np.maximum(cnt, 1)


def cc_instances(prob, img_id):
    """Connected components of the >0.5 mask -> COCO detections (score=mean prob)."""
    binary = prob > 0.5
    lab, n = ndimage.label(binary)
    preds = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < MIN_AREA:
            continue
        rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        preds.append({"image_id": img_id, "category_id": 1, "score": float(prob[m].mean()),
                      "segmentation": rle, "bbox": mask_util.toBbox(rle).tolist()})
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Dinov3Seg(MODEL_ID, device=dev)
    model.head.load_state_dict(torch.load(CKPT, map_location=dev)["head"])
    print(f"[eval] loaded head {CKPT}", flush=True)

    tiles = load_split(os.path.join(ROOT, "data/tcd/test"))
    if a.limit:
        tiles = tiles[: a.limit]
    gt = build_coco_gt(tiles)

    counts, preds = [], []
    for img_id, t in enumerate(tiles):
        rgb = np.asarray(Image.open(t.image_path).convert("RGB"), np.float32) / 255.0
        prob = prob_tile(model, rgb, CROP, dev)
        pred = prob > 0.5
        counts.append(semantic_counts(pred, t.semantic_mask()))
        preds += cc_instances(prob, img_id)
        if img_id % 50 == 0:
            print(f"[eval] {img_id}/{len(tiles)}", flush=True)

    sem = aggregate_semantic(counts)
    m_segm = coco_map50(gt, preds, "segm")
    m_bbox = coco_map50(gt, preds, "bbox")
    print("\n==================== TCD-439 (Restor format) ====================")
    print(f"SEMANTIC   area-F1 = {sem['micro_f1']:.4f}   IoU = {sem['micro_iou']:.4f}")
    print(f"INSTANCE   mAP50(segm, CC-derived) = {100*m_segm['AP50']:.2f}   "
          f"mAP50(bbox) = {100*m_bbox['AP50']:.2f}   [{len(preds)} pred instances]")
    print(f"           (AP@[.5:.95] segm = {100*m_segm['AP']:.2f})")
    print("NOTE: CC-mAP50 is a semantic->instance lower bound, NOT a trained instance model.")


if __name__ == "__main__":
    main()
