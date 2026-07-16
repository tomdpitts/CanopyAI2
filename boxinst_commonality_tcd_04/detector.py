"""8px-grid CenterNet detector on frozen DINO features (individual-tree crowns).

Motivation (decided, not re-litigated): the dominant TCD failure is small/adjacent
crowns collapsing into one 16px feature cell in closed canopy. The head therefore
UPSAMPLES the 16px features to an 8px grid and predicts CenterNet there — halving
cell collisions — while the DINO backbone stays frozen at patch-16.

Fully convolutional, so the same weights run on 512 crops (32->64 grid) at train
time and on stitched 2048 test tiles (128->256 grid) at eval. Targets come from
dapt.targets.encode with an 8px TargetConfig; losses are the grid-agnostic dapt
focal + masked-smooth-L1 + GIoU. Trained on BOXES ONLY, canopy cells ignored.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss

from dapt.head import _masked_smooth_l1, focal_heatmap_loss
from dapt.targets import TargetConfig

STRIDE8 = 8
CFG8 = TargetConfig(grid=64, stride=8)          # per 512 crop; grid scales with input


def _gn(c):
    return nn.GroupNorm(32 if c % 32 == 0 else 8, c)


class Detector8(nn.Module):
    """(B,C,g,g) frozen feats -> (B,5,2g,2g) det maps at 8px (hm, off2, logsize2)."""

    def __init__(self, in_dim: int, width: int = 256, tower: int = 3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_dim, width, 1), _gn(width),
                                  nn.ReLU(inplace=True))
        t = []
        for _ in range(tower):
            t += [nn.Conv2d(width, width, 3, padding=1), _gn(width),
                  nn.ReLU(inplace=True)]
        self.tower = nn.Sequential(*t)
        # 16px -> 8px: learned-context refine after bilinear x2
        self.up = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), _gn(width),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(width, width, 3, padding=1), _gn(width),
                                nn.ReLU(inplace=True))
        self.hm = nn.Conv2d(width, 1, 1)
        self.reg = nn.Conv2d(width, 4, 1)          # offset(2) + logsize(2)
        nn.init.constant_(self.hm.bias, -2.19)     # focal prior p~0.1
        for i, v in enumerate((0.5, 0.5, 3.0, 3.0)):
            nn.init.constant_(self.reg.bias[i], v)  # log ~ 20px crown at 8px stride

    def forward(self, feat):
        x = self.tower(self.stem(feat))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up(x)
        return torch.cat([self.hm(x), self.reg(x)], dim=1)


def det_loss(det, tgt, w_size=0.1):
    """det:(B,5,G,G); tgt: stacked encode() maps at 8px + 'ignore' (B,G,G) canopy.
    Returns (l_hm, l_off, l_size, l_giou)."""
    hm, off, size = det[:, :1], det[:, 1:3], det[:, 3:5]
    l_hm = focal_heatmap_loss(hm, tgt["heatmap"], tgt.get("ignore"))
    l_off = _masked_smooth_l1(off, tgt["offset"], tgt["reg_mask"])
    l_size = _masked_smooth_l1(size, tgt["size"], tgt["reg_mask"])
    b, gy, gx = tgt["reg_mask"].nonzero(as_tuple=True)
    if len(b):
        def to_box(o, s):
            cx = (gx.float() + o[b, 0, gy, gx]) * STRIDE8
            cy = (gy.float() + o[b, 1, gy, gx]) * STRIDE8
            w = s[b, 0, gy, gx].clamp(max=8).exp()
            h = s[b, 1, gy, gx].clamp(max=8).exp()
            return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        l_giou = generalized_box_iou_loss(to_box(off, size),
                                          to_box(tgt["offset"], tgt["size"]),
                                          reduction="mean")
    else:
        l_giou = det.sum() * 0.0
    return l_hm, l_off, l_size * w_size, l_giou


def canopy_cell_mask(canopy_px, g):
    """(512,512) bool canopy pixels -> (g,g) bool: cell is canopy if >50% covered.
    Grid-agnostic (g=64 for a 512 crop at 8px)."""
    if canopy_px is None or not np.any(canopy_px):
        return np.zeros((g, g), bool)
    s = canopy_px.shape[0] // g
    a = np.asarray(canopy_px, bool)[:g * s, :g * s].reshape(g, s, g, s)
    return a.mean((1, 3)) > 0.5
