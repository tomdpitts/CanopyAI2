"""Proxy mask metrics without mask GT: GT-box-prompted, per checkpoint.

For every test-partition GT box, generate the instance mask (prompted at the GT
centre cell, like training) and measure, at mask res inside/around the box:
  fill      mask area inside box / box area          (box-filling -> ~1.0)
  corner    mean prob in the four corner zones (outer 25% of u AND v)
  centre    mean prob in the central zone (middle 50% x 50%)
  leak      mask area outside a 1.2x box / total mask area
Crowns are blobs: good masks -> centre high, corner low, moderate fill. These are
PROXIES (a mask could game them without being a crown) — read with the renders.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from dapt.backbone import pick_device
from dapt.cache_features import cache_key
from dapt.data.cohort import REPO, load_boxes
from boxinst.cache_feats import FEAT_DIR
from boxinst.losses import gather_instances
from boxinst.model import MASK_RES, MASK_STRIDE, BoxInstHead, dynamic_masks


@torch.no_grad()
def eval_ckpt(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    comm_ch = 1 if cfg.get("commonality_channel") else 0
    model = BoxInstHead(cfg["in_dim"], commonality_ch=comm_ch).to(device).eval()
    model.load_state_dict(ck["state"])
    split = json.load(open(os.path.join(REPO, "dapt/data/split.json")))
    boxes_all, _ = load_boxes(split["csv"])
    stats = {"fill": [], "corner": [], "centre": [], "leak": []}
    r = torch.arange(MASK_RES, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    for p, t in split["tiles"].items():
        if t["partition"] != "test" or t["n_boxes"] == 0:
            continue
        feat = torch.from_numpy(np.load(
            os.path.join(FEAT_DIR, cache_key(p) + ".npy"))).float()[None].to(device)
        lda = None
        if comm_ch:
            lda = torch.from_numpy(np.load(os.path.join(
                REPO, "dapt/cache/commonality_last4", cache_key(p) + ".npz"))
                ["lda"]).float()[None].to(device)
        det, ctrl, fmask = model(feat, lda)
        inst = gather_instances([boxes_all[p]], device)
        img_idx, bx, cells, centers = inst
        params = ctrl[0, :, cells[:, 0], cells[:, 1]].T
        logits = dynamic_masks(fmask.expand(len(bx), -1, -1, -1), centers, params)
        prob = torch.sigmoid(logits)
        for i in range(len(bx)):
            x0, y0, x1, y1 = (bx[i] / MASK_STRIDE)
            w, h = (x1 - x0).clamp(min=1), (y1 - y0).clamp(min=1)
            inb = ((xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)).float()
            u, v = (xx - x0) / w, (yy - y0) / h
            corner = (((u < .25) | (u > .75)) & ((v < .25) | (v > .75))).float() * inb
            centre = ((u > .25) & (u < .75) & (v > .25) & (v < .75)).float() * inb
            big = ((xx >= x0 - .1 * w) & (xx < x1 + .1 * w) &
                   (yy >= y0 - .1 * h) & (yy < y1 + .1 * h)).float()
            m = prob[i]
            mb = (m >= 0.5).float()
            stats["fill"].append(((mb * inb).sum() / inb.sum().clamp(min=1)).item())
            stats["corner"].append(((m * corner).sum() /
                                    corner.sum().clamp(min=1)).item())
            stats["centre"].append(((m * centre).sum() /
                                    centre.sum().clamp(min=1)).item())
            stats["leak"].append(((mb * (1 - big)).sum() /
                                  mb.sum().clamp(min=1)).item())
    return {k: round(float(np.mean(v)), 3) for k, v in stats.items()}, len(stats["fill"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = pick_device(args.device)
    print(f"{'ckpt':38s} {'fill':>6} {'corner':>7} {'centre':>7} {'leak':>6}")
    for c in args.ckpts:
        s, n = eval_ckpt(c, device)
        name = os.path.basename(c).replace(".pt", "")
        print(f"{name:38s} {s['fill']:6.3f} {s['corner']:7.3f} "
              f"{s['centre']:7.3f} {s['leak']:6.3f}   (n={n})")


if __name__ == "__main__":
    main()
