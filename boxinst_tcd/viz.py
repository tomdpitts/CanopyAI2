"""Visualize predicted instance masks vs held-out GT polygons on TCD test tiles.

Left: RGB + GT polygons (yellow). Right: predicted instance masks (coloured) +
their contours. GT polygons read for DISPLAY ONLY. One row per tile, one column
block per config, saved to claude_outputs/boxinst_tcd/.

Usage:
    .venv/bin/python -m boxinst_tcd.viz --ckpts tcd_A tcd_B tcd_C --limit 4
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
from PIL import Image

from dapt.backbone import pick_device
from boxinst.infer_viz import detect_instances, mask_polygon
from boxinst.model import BoxInstHead
from boxinst_tcd.cache import COMM, FEAT, key
from boxinst_tcd.eval_masks import raster_polys
from boxinst_tcd.prepare import OUT, RES

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                      "claude_outputs/boxinst_tcd")
CMAP = plt.get_cmap("tab20")


def load(ckpt, device):
    ck = torch.load(os.path.join(OUT, "artifacts", ckpt + ".pt"),
                    map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    cc = 1 if cfg.get("commonality_channel") else 0
    m = BoxInstHead(cfg["in_dim"], commonality_ch=cc).to(device).eval()
    m.load_state_dict(ck["state"])
    return m, cfg, cc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=["tcd_A", "tcd_B", "tcd_C"])
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = pick_device(args.device)
    split = json.load(open(os.path.join(OUT, "split.json")))
    polys = json.load(open(os.path.join(OUT, "gt_polys.json")))     # DISPLAY ONLY
    test = sorted((p for p, t in split["tiles"].items() if t["partition"] == "test"),
                  key=lambda p: -split["tiles"][p]["n_boxes"])[:args.limit]
    models = {c: load(c, device) for c in args.ckpts}
    os.makedirs(OUTDIR, exist_ok=True)
    ncol = 1 + len(args.ckpts)
    for p in test:
        img = np.asarray(Image.open(p).convert("RGB"))
        feat = torch.from_numpy(np.load(os.path.join(FEAT, key(p) + ".npy"))
                                ).float()[None].to(device)
        fig, axs = plt.subplots(1, ncol, figsize=(5.2 * ncol, 5.4), dpi=115)
        axs[0].imshow(img)
        gm = raster_polys(polys[p])
        for poly in polys[p]:
            if poly:
                pa = np.array(poly).reshape(-1, 2)
                axs[0].plot(pa[:, 0], pa[:, 1], "y-", lw=1.1)
        axs[0].set_title(f"GT polygons (n={len(gm)})", fontsize=9)
        for ax, c in zip(axs[1:], args.ckpts):
            model, cfg, cc = models[c]
            lda = None
            if cc:
                lda = torch.from_numpy(np.load(os.path.join(COMM, key(p) + ".npz"))
                                       ["lda"]).float()[None].to(device)
            boxes, scores, probs = detect_instances(model, feat, cfg["score_thr"],
                                                    cfg["nms_iou"], lda=lda)
            ax.imshow(img)
            ov = np.zeros((RES, RES, 4), np.float32)
            for i in range(len(boxes)):
                col = np.array(CMAP(i % 20))
                m = probs[i].numpy()
                ov[m >= cfg["mask_thr"]] = [*col[:3], 0.5]
                pl = mask_polygon(m, cfg["mask_thr"])
                if pl is not None:
                    ax.plot(pl[:, 0], pl[:, 1], color=col[:3], lw=1.2)
            ax.imshow(ov)
            ax.set_title(f"{c}: pred={len(boxes)}", fontsize=9)
        for a in axs:
            a.axis("off")
        fig.suptitle(f"{split['tiles'][p]['tile_id']}  "
                     f"({split['tiles'][p].get('biome','?')})", fontsize=10)
        fig.tight_layout()
        out = os.path.join(OUTDIR, f"cmp_{split['tiles'][p]['tile_id']}.png")
        fig.savefig(out); plt.close(fig)
        print(out)


if __name__ == "__main__":
    main()
