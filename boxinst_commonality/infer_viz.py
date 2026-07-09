"""Full pipeline viz: boxinst detection -> EM posterior instance masks.

Detection comes from a trained boxinst checkpoint (its mask branch is unused);
masks are the hand-computed EM posteriors prompted by the predicted boxes.
Renders test tiles to claude_outputs/boxinst_commonality/.

Usage:
    .venv/bin/python -m boxinst_commonality.infer_viz \
        --ckpt boxinst/artifacts/boxinst_s0.pt
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
from boxinst_commonality.em import (EMModel, draw_active_cells, draw_grid,
                                    out_dir)
from boxinst_commonality.evaluate import (MASK_RES, MASK_STRIDE, prob_map_128,
                                          resolve_overlaps)

C_TP, C_FP, C_FN = "#22dd22", "#ff3333", "#00e5e5"   # green / red / teal


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="boxinst/artifacts/boxinst_s0.pt")
    ap.add_argument("--model", default=None,
                    help="EM model npz (default boxinst_commonality/artifacts/"
                         "em_model.npz)")
    ap.add_argument("--per_domain", type=int, default=2)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    device = pick_device(args.device)

    ck = torch.load(os.path.join(REPO, args.ckpt), map_location="cpu",
                    weights_only=False)
    cfg = ck["cfg"]
    det_model = BoxInstHead(
        cfg["in_dim"],
        commonality_ch=1 if cfg.get("commonality_channel") else 0,
    ).to(device).eval()
    det_model.load_state_dict(ck["state"])
    rep = json.load(open(os.path.join(
        REPO, args.ckpt.replace(".pt", ".json"))))
    map50 = rep["test"]["mAP50"]
    em = EMModel(args.model) if args.model else EMModel()
    split, boxes_all = load_split()

    chosen = []
    test = [p for p, t in split["tiles"].items() if t["partition"] == "test"]
    for dom in ("WON", "BRU", "NEON"):
        cand = [p for p in test if split["tiles"][p]["domain"] == dom]
        cand.sort(key=lambda p: -split["tiles"][p]["n_boxes"])
        chosen += cand[:args.per_domain]

    for p in chosen:
        # detection features must match the checkpoint (web_last4); the EM
        # projects from its own recorded feature cache
        feat = np.load(os.path.join(FEAT_DIR, cache_key(p) + ".npy"))
        det, _, _ = det_model(
            torch.from_numpy(feat.astype(np.float32))[None].to(device))
        pboxes, scores = decode(det.cpu(), score_thr=cfg["score_thr"])
        pb, sc = pboxes.numpy(), scores.numpy()
        gt = np.asarray(boxes_all.get(p, np.zeros((0, 4))), np.float32)
        zn = em.project_tile(p)
        img = np.asarray(Image.open(p).convert("RGB"))
        H0, W0 = img.shape[:2]

        # greedy IoU-0.5 match (score order) -> TP flags + which GT are missed
        ious = iou_matrix(pb, gt)
        tp = np.zeros(len(pb), bool)
        gt_hit = np.zeros(len(gt), bool)
        for i in np.argsort(-sc):
            if not len(gt):
                break
            j = int(np.argmax(ious[i]))
            if ious[i, j] >= 0.5 and not gt_hit[j]:
                gt_hit[j] = True
                tp[i] = True

        fig, axs = plt.subplots(1, 2, figsize=(15, 7.8 * H0 / W0), dpi=140)
        axs[0].imshow(img)
        for b in gt:
            axs[0].add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0],
                                           b[3] - b[1], fill=False,
                                           ec="white", lw=1.0))
        axs[0].set_title(f"RGB + GT boxes (n={len(gt)})", fontsize=9)
        axs[1].imshow(img)
        draw_grid(axs[1], H0, W0, em.s)
        # instance masks: upsample each box posterior, then make masks mutually
        # exclusive (pb is NMS output = descending score, so ties go to the
        # stronger detection) — overlapping detections otherwise draw crossing
        # figure-8 boundaries
        rss = []
        for b in pb:
            prob = prob_map_128(em, zn, b, H0, W0)
            rss.append(np.array(Image.fromarray(prob).resize(
                (MASK_RES * MASK_STRIDE,) * 2, Image.BILINEAR))[:H0, :W0])
        rss = resolve_overlaps(rss)
        for j, b in enumerate(pb):
            idx, r = em.box_posterior(zn, b, H0, W0)
            draw_active_cells(axs[1], idx, r, em.grid, em.s)
            if (rss[j] >= 0.5).any():
                axs[1].contour(rss[j], levels=[0.5],
                               colors=[C_TP if tp[j] else C_FP],
                               linewidths=1.5)
        for b in gt[~gt_hit]:
            axs[1].add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0],
                                           b[3] - b[1], fill=False, ec=C_FN,
                                           lw=1.4, linestyle="--"))
        axs[1].set_title(f"EM mask boundaries at det thr={cfg['score_thr']:.2f}"
                         "  |  green=TP mask  red=FP mask  teal dashed=missed GT"
                         f"  |  purple=active {em.s}px cells", fontsize=8)
        for a in axs:
            a.axis("off")
        name = cache_key(p)
        t = split["tiles"][p]
        fig.suptitle(
            f"{t['domain']}/{t['site']}  {name}   |   detector test mAP50="
            f"{map50:.3f} (Restor cohort)   |   pixel area-F1: N/A "
            "(no mask GT on dryland)", fontsize=9)
        fig.tight_layout()
        tag = os.path.basename(args.model).replace("em_model", "").replace(
            ".npz", "") if args.model else ""
        od = out_dir(tag)
        fig.savefig(os.path.join(od, f"pipeline_{name}.png"),
                    bbox_inches="tight")
        plt.close(fig)
        print(f"rendered {od}/pipeline_{name}.png", flush=True)


if __name__ == "__main__":
    main()
