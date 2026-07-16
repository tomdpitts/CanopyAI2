"""ISOLATED, DELETABLE experiment: one model, two heads (multiclass-ish).

`MultiHead8` = the exact `Detector8` trunk (frozen-feat stem + tower + 16→8px up)
with TWO heads sharing it:
  - det head  (5ch): the ITC CenterNet (heatmap + offset + log-size), unchanged;
  - sem head  (1ch): a semantic TREE-COVER logit = P(pixel is tree, incl. canopy).

The ITC head is trained exactly as `det_t8` (boxes only, canopy-ignore) so instance
mAP50 is preserved; the sem head is trained on (ITC-box ∪ canopy) foreground so it
learns "canopy = tree cover" — the area-F1 lever the ignore label currently blocks.
Shared trunk = one forward, one model; the sem loss is low-weighted so it perturbs
the ITC features minimally (measured). Delete the `tcd04_semseg/` folder to discard.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from boxinst_commonality_tcd_04.detector import _gn


class MultiHead8(nn.Module):
    def __init__(self, in_dim: int, width: int = 256, tower: int = 3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_dim, width, 1), _gn(width),
                                  nn.ReLU(inplace=True))
        t = []
        for _ in range(tower):
            t += [nn.Conv2d(width, width, 3, padding=1), _gn(width),
                  nn.ReLU(inplace=True)]
        self.tower = nn.Sequential(*t)
        self.up = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), _gn(width),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(width, width, 3, padding=1), _gn(width),
                                nn.ReLU(inplace=True))
        self.hm = nn.Conv2d(width, 1, 1)
        self.reg = nn.Conv2d(width, 4, 1)
        self.sem = nn.Conv2d(width, 1, 1)              # tree-cover logit
        nn.init.constant_(self.hm.bias, -2.19)
        for i, v in enumerate((0.5, 0.5, 3.0, 3.0)):
            nn.init.constant_(self.reg.bias[i], v)
        nn.init.constant_(self.sem.bias, 0.0)

    def forward(self, feat):
        """(B,C,g,g) -> det (B,5,2g,2g), sem (B,1,2g,2g)."""
        x = self.tower(self.stem(feat))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up(x)
        det = torch.cat([self.hm(x), self.reg(x)], dim=1)
        return det, self.sem(x)

    def load_det_from(self, det8_state):
        """Warm-start the shared trunk + det head from a trained Detector8."""
        self.load_state_dict(det8_state, strict=False)
