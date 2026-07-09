"""One-pass inference + visualization on held-out tiles.

centre-heatmap peak NMS -> instances; per instance the controller's dynamic conv is
run over F_mask (+ rel-coords) -> mask logits -> upsample to image res -> threshold
0.5 -> contourpy polygon. Draws per-instance coloured masks + polygons + (dashed)
predicted boxes over RGB, with GT boxes in white for reference. Saves PNGs +
per-tile counts JSON to claude_outputs/boxinst/.

Usage:
    .venv/bin/python -m boxinst.infer_viz                 # all test tiles
    .venv/bin/python -m boxinst.infer_viz --domains BRU WON --limit 6
"""
import argparse
import json
import os

import contourpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import nms

from dapt.backbone import pick_device
from dapt.cache_features import cache_key
from dapt.data.cohort import REPO, load_boxes
from boxinst.cache_feats import FEAT_DIR
from boxinst.model import (GRID, MASK_STRIDE, STRIDE, BoxInstHead, dynamic_masks,
                           signature_cosine)
from boxinst.train import ART_DIR

OUT_DIR = os.path.join(REPO, "claude_outputs/boxinst")
CMAP = plt.get_cmap("tab20")


@torch.no_grad()
def detect_instances(model, feat, score_thr, nms_iou=0.5, topk=200, lda=None,
                     z=None):
    """feat:(1,C,32,32) -> boxes(M,4) px, scores(M), masks (M,512,512) prob.
    z:(1,D,32,32) required iff the model uses the signature channel."""
    det, ctrl, fmask = model(feat, lda)
    hm_logit, off, size = det[:, :1], det[:, 1:3], det[:, 3:5]
    hm = torch.sigmoid(hm_logit)
    keep = (F.max_pool2d(hm, 3, stride=1, padding=1) == hm).float()
    hm = (hm * keep)[0, 0]
    scores, idx = hm.flatten().topk(min(topk, GRID * GRID))
    m = scores > score_thr
    scores, idx = scores[m], idx[m]
    if idx.numel() == 0:
        return (torch.zeros(0, 4), torch.zeros(0),
                torch.zeros(0, GRID * STRIDE, GRID * STRIDE))
    gy, gx = idx // GRID, idx % GRID
    ox, oy = off[0, 0, gy, gx], off[0, 1, gy, gx]
    w, h = size[0, 0, gy, gx].exp(), size[0, 1, gy, gx].exp()
    cx, cy = (gx.float() + ox) * STRIDE, (gy.float() + oy) * STRIDE
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    k = nms(boxes, scores, nms_iou)
    boxes, scores, gy, gx = boxes[k], scores[k], gy[k], gx[k]
    cx, cy = cx[k], cy[k]

    params = ctrl[0, :, gy, gx].T                              # (M,n_dyn)
    centers = torch.stack([cx, cy], 1) / MASK_STRIDE           # mask-grid px
    sig = None
    if getattr(model, "sig_ch", 0) and z is not None:
        img_idx = torch.zeros(len(k), dtype=torch.long, device=feat.device)
        sig = signature_cosine(z.expand(1, -1, -1, -1), img_idx, boxes)
    logits = dynamic_masks(fmask.expand(len(k), -1, -1, -1), centers, params, sig=sig)
    probs = torch.sigmoid(F.interpolate(
        logits[:, None], size=(GRID * STRIDE, GRID * STRIDE),
        mode="bilinear", align_corners=False))[:, 0]
    return boxes.cpu(), scores.cpu(), probs.cpu()


def mask_polygon(prob: np.ndarray, level=0.5):
    """Largest closed contour of the prob map at `level` -> (K,2) xy, or None."""
    gen = contourpy.contour_generator(z=prob)
    lines = gen.lines(level)
    if not lines:
        return None
    return max(lines, key=lambda l: len(l))


def render_tile(img, gt_boxes, boxes, scores, probs, mask_thr, title, out_path):
    H0, W0 = img.shape[:2]
    fig, ax = plt.subplots(figsize=(8, 8 * H0 / W0), dpi=140)
    ax.imshow(img)
    order = np.argsort(scores)                                 # high score on top
    overlay = np.zeros((H0, W0, 4), np.float32)
    polys = 0
    for rank, i in enumerate(order):
        color = np.array(CMAP(int(i) % 20))
        m = probs[i, :H0, :W0]
        binm = m >= mask_thr
        if binm.sum() < 4:
            continue
        overlay[binm] = [*color[:3], 0.45]
        poly = mask_polygon(m, mask_thr)
        if poly is not None:
            ax.plot(poly[:, 0], poly[:, 1], color=color[:3], lw=1.4)
            polys += 1
        x0, y0, x1, y1 = boxes[i]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   ec=color[:3], lw=0.7, ls="--", alpha=0.8))
    ax.imshow(overlay)
    for (x0, y0, x1, y1) in gt_boxes:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   ec="white", lw=1.0, alpha=0.9))
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, W0)
    ax.set_ylim(H0, 0)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path)
    plt.close(fig)
    return polys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ART_DIR, "boxinst_s0.pt"))
    ap.add_argument("--partition", default="test")
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--out_sub", default="", help="subdirectory under claude_outputs/boxinst")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    out_dir = os.path.join(OUT_DIR, args.out_sub) if args.out_sub else OUT_DIR

    device = pick_device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    comm_ch = 1 if cfg.get("commonality_channel") else 0
    model = BoxInstHead(cfg["in_dim"], commonality_ch=comm_ch).to(device).eval()
    model.load_state_dict(ck["state"])
    score_thr, nms_iou, mask_thr = cfg["score_thr"], cfg["nms_iou"], cfg["mask_thr"]

    split = json.load(open(os.path.join(REPO, "dapt/data/split.json")))
    gt_boxes, _ = load_boxes(split["csv"])
    tiles = [p for p, t in split["tiles"].items()
             if t["partition"] == args.partition
             and (args.domains is None or t["domain"] in args.domains)]
    tiles = sorted(tiles, key=lambda p: -split["tiles"][p]["n_boxes"])
    if args.limit:
        tiles = tiles[:args.limit]
    os.makedirs(out_dir, exist_ok=True)

    counts = []
    for p in tiles:
        feat = torch.from_numpy(np.load(
            os.path.join(FEAT_DIR, cache_key(p) + ".npy"))).float()[None].to(device)
        lda = None
        if comm_ch:
            lda = torch.from_numpy(np.load(os.path.join(
                REPO, "dapt/cache/commonality_last4", cache_key(p) + ".npz"))
                ["lda"]).float()[None].to(device)
        boxes, scores, probs = detect_instances(model, feat, score_thr, nms_iou,
                                                lda=lda)
        img = np.asarray(Image.open(p).convert("RGB"))
        gb = gt_boxes.get(p, np.zeros((0, 4)))
        info = split["tiles"][p]
        name = cache_key(p)
        title = (f"{info['domain']}/{info['site']}  {name}\n"
                 f"pred={len(boxes)} crowns (GT boxes={len(gb)}, white)  "
                 f"score_thr={score_thr:.2f} mask_thr={mask_thr}")
        out = os.path.join(out_dir, f"{args.partition}_{name}.png")
        render_tile(img, gb, boxes.numpy(), scores.numpy(), probs.numpy(),
                    mask_thr, title, out)
        counts.append({"tile": name, "domain": info["domain"], "site": info["site"],
                       "pred": len(boxes), "gt": len(gb)})
        print(f"{name}: pred={len(boxes)} gt={len(gb)} -> {os.path.relpath(out, REPO)}")
    json.dump({"cfg": {k: cfg[k] for k in
                       ("score_thr", "nms_iou", "mask_thr", "tau", "seed")},
               "tiles": counts},
              open(os.path.join(out_dir, f"counts_{args.partition}.json"), "w"),
              indent=2)


if __name__ == "__main__":
    main()
