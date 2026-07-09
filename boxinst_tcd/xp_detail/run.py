"""THROWAWAY experiment: does fusing an RGB detail branch into F_mask tighten masks?

Apples-to-apples on TCD (720 train / 160 test, complete canopy). Trains a standalone
CondInst mask head on GT-box prompts (isolates mask quality from detection), box-only
losses (projection + DINO-affinity pairwise). Two arms, identical except the branch:
  base  : F_mask = neck(DINO features)                        (8 ch)
  detail: F_mask = neck(DINO features) ⊕ small-conv(RGB image) (8 + D ch)

Reports GT-prompted mask mIoU AND boundary-F1 (mIoU is near-blind on tight boxes;
boundary-F1 is the tightness metric that matters).

Usage:
    .venv/bin/python -m boxinst_tcd.xp_detail.run --epochs 200
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from dapt.backbone import pick_device
from boxinst.losses import gather_instances, pairwise_loss, projection_loss
from boxinst.model import MASK_RES, REL_NORM
from boxinst_tcd.cache import FEAT, key
from boxinst_tcd.eval_masks import mask_iou_matrix, raster_polys
from boxinst_tcd.prepare import OUT, RES
from boxinst_tcd.train import Data


def _gn(c):
    return nn.GroupNorm(8 if c % 8 == 0 else 1, c)


class MaskHead(nn.Module):
    """Standalone CondInst mask head (+ optional RGB detail branch)."""

    def __init__(self, in_dim, detail_ch=0, width=256, mask_ch=8):
        super().__init__()
        self.detail_ch = detail_ch
        self.mask_ch = mask_ch
        self.stem = nn.Sequential(nn.Conv2d(in_dim, width, 1), _gn(width), nn.ReLU(True))
        self.tower = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), _gn(width), nn.ReLU(True),
            nn.Conv2d(width, width, 3, padding=1), _gn(width), nn.ReLU(True))
        self.neck = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width + 2, 128, 3, padding=1), _gn(128), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), _gn(64), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), _gn(64), nn.ReLU(True),
                          nn.Conv2d(64, mask_ch, 1))])
        if detail_ch:
            # RGB (3ch, 512) -> detail_ch @128 (stride 4). Sees real image edges.
            self.detail = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1), _gn(16), nn.ReLU(True),   # 256
                nn.Conv2d(16, 32, 3, stride=2, padding=1), _gn(32), nn.ReLU(True),  # 128
                nn.Conv2d(32, 32, 3, padding=1), _gn(32), nn.ReLU(True),
                nn.Conv2d(32, detail_ch, 1), _gn(detail_ch))    # GN: match F_mask scale
        din = mask_ch + detail_ch + 2
        self.dyn = [(din, mask_ch), (mask_ch, mask_ch), (mask_ch, 1)]
        n_dyn = sum(i * o + o for i, o in self.dyn)
        self.ctrl = nn.Conv2d(width, n_dyn, 1)
        nn.init.normal_(self.ctrl.weight, std=0.01); nn.init.constant_(self.ctrl.bias, 0.0)

    def fmap(self, feat, img):
        s = self.stem(feat); t = self.tower(s)
        r = torch.linspace(-1, 1, feat.shape[-1], device=feat.device)
        yy, xx = torch.meshgrid(r, r, indexing="ij")
        coords = torch.stack([xx, yy])[None].expand(feat.shape[0], -1, -1, -1)
        m = self.neck[0](torch.cat([s, coords], 1))
        m = F.interpolate(m, scale_factor=2, mode="bilinear", align_corners=False)
        m = self.neck[1](m)
        m = F.interpolate(m, scale_factor=2, mode="bilinear", align_corners=False)
        fmask = self.neck[2](m)                                  # (B,8,128,128)
        if self.detail_ch:
            fmask = torch.cat([fmask, self.detail(img)], 1)      # (B,8+D,128,128)
        return fmask, self.ctrl(t)

    def masks(self, fmask, ctrl, cells, centers, img_idx):
        params = ctrl[img_idx, :, cells[:, 0], cells[:, 1]]      # (N, n_dyn)
        fi = fmask[img_idx]                                      # (N, C, R, R)
        R = fi.shape[-1]; dev = fi.device
        r = torch.arange(R, device=dev, dtype=torch.float32)
        yy, xx = torch.meshgrid(r, r, indexing="ij")
        rx = (xx[None] - centers[:, 0, None, None]) / REL_NORM
        ry = (yy[None] - centers[:, 1, None, None]) / REL_NORM
        x = torch.cat([fi, rx[:, None], ry[:, None]], 1).flatten(2)
        i = 0
        for li, (cin, cout) in enumerate(self.dyn):
            w = params[:, i:i + cin * cout].reshape(-1, cout, cin); i += cin * cout
            b = params[:, i:i + cout]; i += cout
            x = torch.bmm(w, x) + b[:, :, None]
            if li < len(self.dyn) - 1:
                x = F.relu(x)
        return x.reshape(len(img_idx), R, R)


def load_img(p, dev):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(dev)


def dilate(x, d):
    return F.max_pool2d(x, 2 * d + 1, stride=1, padding=d)


def boundary_f1(pred, gt, d=3):
    """Contour F-measure: how well mask boundaries align within d px. pred/gt bool (R,R)."""
    p = torch.from_numpy(pred.astype(np.float32))[None, None]
    g = torch.from_numpy(gt.astype(np.float32))[None, None]
    bp = (p - (1 - dilate(1 - p, 1))).clamp(0, 1)               # inner rim of pred
    bg = (g - (1 - dilate(1 - g, 1))).clamp(0, 1)
    if bp.sum() == 0 or bg.sum() == 0:
        return 0.0
    prec = (bp * dilate(bg, d)).sum() / bp.sum()
    rec = (bg * dilate(bp, d)).sum() / bg.sum()
    return float(2 * prec * rec / (prec + rec + 1e-9))


def train(data, tr, detail_ch, dev, epochs, seed=0, bs=6):
    torch.manual_seed(seed)
    in_dim = data.tiles[tr[0]]["feat"].shape[0]
    head = MaskHead(in_dim, detail_ch).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed); it = 0
    for ep in range(epochs):
        head.train(); order = list(tr); rng.shuffle(order)
        for i in range(0, len(order), bs):
            paths = order[i:i + bs]
            b = data.batch(paths, dev)
            imgs = torch.cat([load_img(p, dev) for p in paths], 0) if detail_ch else None
            fmask, ctrl = head.fmap(b["feat"], imgs)
            inst = gather_instances(b["boxes"], dev)
            if inst is None:
                continue
            img_idx, boxes_px, cells, centers = inst
            logits = head.masks(fmask, ctrl, cells, centers, img_idx)
            l_proj = projection_loss(logits, boxes_px)
            l_pair = pairwise_loss(logits, boxes_px, b["sims"][img_idx], 0.975, [4, 5, 6, 7])
            loss = l_proj + min(1.0, it / 800) * l_pair
            opt.zero_grad(); loss.backward(); opt.step(); it += 1
    return head


@torch.no_grad()
def evaluate(head, data, dev, thr=0.5):
    head.eval()
    polys = json.load(open(os.path.join(OUT, "gt_polys.json")))
    ious, bfs = [], []
    for p in data.partition("test"):
        gbx = data.tiles[p]["boxes"]
        if len(gbx) == 0:
            continue
        feat = torch.from_numpy(np.load(os.path.join(FEAT, key(p) + ".npy"))).float()[None].to(dev)
        img = load_img(p, dev) if head.detail_ch else None
        fmask, ctrl = head.fmap(feat, img)
        inst = gather_instances([gbx], dev)
        img_idx, boxes_px, cells, centers = inst
        logits = head.masks(fmask, ctrl, cells, centers, img_idx)
        prob = torch.sigmoid(F.interpolate(logits[:, None], size=(RES, RES),
                             mode="bilinear", align_corners=False))[:, 0].cpu().numpy()
        gm = np.array(raster_polys(polys[p]))
        pm = prob >= thr
        iou = mask_iou_matrix(pm, gm)
        for i in range(min(len(pm), len(gm))):
            ious.append(float(iou[i, i])); bfs.append(boundary_f1(pm[i], gm[i]))
    return np.mean(ious), np.mean(bfs), len(ious)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--detail_ch", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = pick_device(args.device)
    data = Data(); tr = data.partition("train")
    print(f"apples-to-apples TCD: train={len(tr)} test={len(data.partition('test'))}")
    for dch, name in [(0, "base (F_mask only)"), (args.detail_ch, f"+detail RGB ({args.detail_ch}ch)")]:
        head = train(data, tr, dch, dev, args.epochs)
        miou, bf1, n = evaluate(head, data, dev)
        torch.save(head.state_dict(), os.path.join(
            OUT, "xp_detail", f"mask_detail{dch}.pt"))
        print(f"{name:26s}: GT-prompted mIoU={miou:.3f}  boundary-F1={bf1:.3f}  (n={n})")


if __name__ == "__main__":
    main()
