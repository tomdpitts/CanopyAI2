"""TCD metrics: semantic area F1/IoU (canopy) + instance mAP50 (tree crowns).

Semantic: micro-averaged pixel F1 and IoU of the binary tree-cover mask.

Instance: COCO mask AP at IoU=0.50. Crowns (cat=2) are the detection targets
(iscrowd=0); canopy regions (cat=1) are folded in as iscrowd=1 *ignore* zones
under the crown category, so a prediction landing on dense closed canopy is
neither rewarded nor penalised -- this matches the OAM-TCD "canopy iscrowd
ignore" convention (data/tcd/experimental/sparse/README.md).
"""
from __future__ import annotations

import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from tcd_data import CAT_CANOPY, CAT_CROWN, _ann_to_rle

CROWN_CAT = 1  # single-category eval space for crowns


# ---- semantic ------------------------------------------------------------- #
def semantic_counts(pred_mask, gt_mask):
    p, g = pred_mask.astype(bool), gt_mask.astype(bool)
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    return {"tp": tp, "fp": fp, "fn": fn}


def aggregate_semantic(per_tile_counts):
    tp = sum(d["tp"] for d in per_tile_counts)
    fp = sum(d["fp"] for d in per_tile_counts)
    fn = sum(d["fn"] for d in per_tile_counts)
    return {
        "micro_iou": tp / (tp + fp + fn + 1e-9),
        "micro_f1": 2 * tp / (2 * tp + fp + fn + 1e-9),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ---- instance (COCO mAP50) ------------------------------------------------ #
def build_coco_gt(tiles):
    """COCO GT dict over crowns (target) + canopy (iscrowd ignore), one category."""
    images, annotations, ann_id = [], [], 1
    for img_id, t in enumerate(tiles):
        images.append({"id": img_id, "file_name": os.path.basename(t.image_path),
                       "width": t.width, "height": t.height})
        for a in t.anns:
            rle = _ann_to_rle(a["segmentation"], t.height, t.width)
            from pycocotools import mask as mask_util
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": CROWN_CAT,
                "segmentation": rle, "bbox": a["bbox"],
                "area": float(mask_util.area(rle)),
                "iscrowd": 0 if a["category_id"] == CAT_CROWN else 1,  # canopy -> ignore
            })
            ann_id += 1
    return {"images": images, "annotations": annotations,
            "categories": [{"id": CROWN_CAT, "name": "tree_crown"}]}


def coco_map50(gt_dict, predictions, iou_type="segm"):
    """predictions: list of {image_id, category_id=1, score, segmentation|bbox}.

    Returns AP@0.50 (the requested map50) plus COCO AP@[.5:.95] for context.
    """
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()
    if not predictions:
        return {"AP50": 0.0, "AP": 0.0, "n_pred": 0}
    coco_dt = coco_gt.loadRes(predictions)
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"AP50": float(ev.stats[1]), "AP": float(ev.stats[0]), "n_pred": len(predictions)}
