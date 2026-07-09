"""Predicted-mask renders, coloured by detection outcome (boundary only, no fill).

Detection comes from a trained boxinst checkpoint; masks are the contrastive-EM
posteriors (this package) prompted by each box. Predicted boxes are greedy-matched
to GT at IoU 0.5 (COCO-style, score order); every drawn polygon is a 0.5-level
mask contour coloured by outcome:
  GREEN  true positive   (predicted box matched a GT)
  RED    false positive  (predicted box, no GT match)
  TEAL   false negative   (GT box missed; mask prompted from the GT box)

Masks have no GT, so colour reflects DETECTION correctness, not mask correctness.

Usage:
    .venv/bin/python -m boxinst_commonality.render_masks --n 15
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dapt.backbone import pick_device
from dapt.cache_features import cache_key
from dapt.data.cohort import REPO
from dapt.decode import decode
from dapt.eval import iou_matrix
from boxinst.cache_feats import FEAT_DIR
from boxinst.commonality import load_split
from boxinst.model import BoxInstHead
from boxinst_commonality.em import OUT, EMModel
from boxinst_commonality.evaluate import MASK_RES, MASK_STRIDE, prob_map_128

TP, FP, FN = "#22c55e", "#ef4444", "#14b8a6"      # green / red / teal


def match(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5):
    """COCO greedy match -> (tp bool[Npred] orig order, gt_matched bool[Ngt])."""
    order = np.argsort(-pred_scores)
    tp = np.zeros(len(pred_boxes), bool)
    gt_matched = np.zeros(len(gt_boxes), bool)
    if len(gt_boxes) and len(pred_boxes):
        ious = iou_matrix(pred_boxes[order], gt_boxes)
        for rank, pi in enumerate(order):
            j = int(np.argmax(ious[rank]))
            if ious[rank, j] >= iou_thr and not gt_matched[j]:
                gt_matched[j] = True
                tp[pi] = True
    return tp, gt_matched


def contour(ax, prob, H0, W0, color):
    rs = np.array(Image.fromarray(prob).resize(
        (MASK_RES * MASK_STRIDE,) * 2, Image.BILINEAR))[:H0, :W0]
    if (rs >= 0.5).any():
        ax.contour(rs, levels=[0.5], colors=[color], linewidths=1.3)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="boxinst/artifacts/boxinst_s0.pt")
    ap.add_argument("--model", default=None, help="EM npz (default: primary)")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    device = pick_device(args.device)

    ck = torch.load(os.path.join(REPO, args.ckpt), map_location="cpu",
                    weights_only=False)
    cfg = ck["cfg"]
    det_model = BoxInstHead(
        cfg["in_dim"], commonality_ch=1 if cfg.get("commonality_channel") else 0,
    ).to(device).eval()
    det_model.load_state_dict(ck["state"])
    rep = json.load(open(os.path.join(REPO, args.ckpt.replace(".pt", ".json"))))
    map50 = rep["test"]["mAP50"]
    em = EMModel(args.model) if args.model else EMModel()
    split, boxes_all = load_split()

    # 15 test tiles: round-robin across domains, densest first
    test = [p for p, t in split["tiles"].items()
            if t["partition"] == "test" and t["n_boxes"] > 0]
    by_dom = {}
    for p in sorted(test, key=lambda p: -split["tiles"][p]["n_boxes"]):
        by_dom.setdefault(split["tiles"][p]["domain"], []).append(p)
    chosen, i = [], 0
    while len(chosen) < min(args.n, len(test)):
        for dom in by_dom:
            if i < len(by_dom[dom]) and len(chosen) < args.n:
                chosen.append(by_dom[dom][i])
        i += 1

    os.makedirs(OUT, exist_ok=True)
    cols = 5
    rows = (len(chosen) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.2), dpi=115)
    axs = np.atleast_1d(axs).ravel()
    tot_tp = tot_fp = tot_fn = 0
    for ax, p in zip(axs, chosen):
        feat = np.load(os.path.join(FEAT_DIR, cache_key(p) + ".npy"))
        det, _, _ = det_model(
            torch.from_numpy(feat.astype(np.float32))[None].to(device))
        pboxes, scores = decode(det.cpu(), score_thr=cfg["score_thr"])
        pboxes, scores = pboxes.numpy(), scores.numpy()
        gt = np.asarray(boxes_all.get(p, np.zeros((0, 4))), np.float32)
        tp, gt_matched = match(pboxes, scores, gt)
        zn = em.project_tile(p)
        img = np.asarray(Image.open(p).convert("RGB"))
        H0, W0 = img.shape[:2]
        ax.imshow(img)
        for b, is_tp in zip(pboxes, tp):
            contour(ax, prob_map_128(em, zn, b, H0, W0), H0, W0,
                    TP if is_tp else FP)
        for b, m in zip(gt, gt_matched):
            if not m:
                contour(ax, prob_map_128(em, zn, b, H0, W0), H0, W0, FN)
        ntp, nfp = int(tp.sum()), int((~tp).sum())
        nfn = int((~gt_matched).sum())
        tot_tp += ntp; tot_fp += nfp; tot_fn += nfn
        t = split["tiles"][p]
        ax.set_title(f"{t['domain']}/{t['site']}  TP {ntp} / FP {nfp} / FN {nfn}",
                     fontsize=8)
        ax.axis("off")
    for ax in axs[len(chosen):]:
        ax.axis("off")
    prec = tot_tp / max(tot_tp + tot_fp, 1)
    rec = tot_tp / max(tot_tp + tot_fn, 1)
    fig.suptitle(
        f"Contrastive-EM predicted masks (boundary @0.5) — "
        f"GREEN TP / RED FP / TEAL FN(missed GT).   "
        f"detector test mAP50={map50:.3f} (Restor)  |  "
        f"these {len(chosen)} tiles: P={prec:.2f} R={rec:.2f}  |  "
        f"pixel area-F1: N/A (no mask GT on dryland)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(OUT, "predmasks_tpfpfn.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"rendered {out}  ({len(chosen)} tiles, "
          f"TP {tot_tp} FP {tot_fp} FN {tot_fn})", flush=True)


if __name__ == "__main__":
    main()
