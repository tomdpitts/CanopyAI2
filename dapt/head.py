"""CenterNet probe heads (the fixed probe) + losses.

Capacity ladder, all consuming the SAME frozen features and emitting the SAME 5
channels (heatmap 1 + offset 2 + size 2) at the 32x32 grid:
  L1 'linear' : a single 1x1 conv (per-patch linear). No hidden layers. PRIMARY.
  L2 'mlp'    : 1-2 hidden 1x1-conv layers + ReLU.
Localization inside a cell comes from the offset channels; the fixed bilinear
upsample lives in decode (deterministic, no learnable params), not here.

Loss config (frozen, identical across arms): CenterNet penalty-reduced focal for the
heatmap; smooth-L1 for offset and (log-)size at positive cells only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_OUT = 5   # heatmap(1) + offset(2) + size(2)
# frozen loss weights (see PLAN §4); identical across all arms
W_HM, W_OFF, W_SIZE = 1.0, 1.0, 0.1
FOCAL_ALPHA, FOCAL_BETA = 2.0, 4.0


class ProbeHead(nn.Module):
    def __init__(self, in_dim: int, capacity: str = "linear", hidden: int = 256,
                 n_hidden: int = 1):
        super().__init__()
        self.capacity = capacity
        if capacity == "linear":
            self.net = nn.Conv2d(in_dim, N_OUT, 1)             # pure per-patch linear
        elif capacity == "mlp":
            layers, c = [], in_dim
            for _ in range(n_hidden):
                layers += [nn.Conv2d(c, hidden, 1), nn.ReLU(inplace=True)]
                c = hidden
            layers += [nn.Conv2d(c, N_OUT, 1)]
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(f"capacity must be 'linear' or 'mlp', got {capacity!r}")

    def forward(self, feat):                                   # (B,C,G,G) -> (B,5,G,G)
        return self.net(feat)

    @staticmethod
    def split(out):
        """(B,5,G,G) -> hm_logit(B,1,G,G), offset(B,2,G,G), size(B,2,G,G)."""
        return out[:, :1], out[:, 1:3], out[:, 3:5]


def focal_heatmap_loss(hm_logit, gt, ignore=None):
    """CenterNet penalty-reduced focal. gt in [0,1] with peaks==1.

    ignore: optional (B,1,H,W) or (B,H,W) mask of cells to exclude from the NEGATIVE
    loss (e.g. canopy for ITC — a peak there is valid, just unlabelled, so it must not
    be penalised as a false positive). Positives are always supervised. Default None =
    original behaviour."""
    p = torch.sigmoid(hm_logit).clamp(1e-6, 1 - 1e-6)
    pos = (gt == 1).float()
    neg = 1.0 - pos
    if ignore is not None:
        neg = neg * (1.0 - ignore.view_as(neg).float())
    pos_loss = ((1 - p) ** FOCAL_ALPHA) * torch.log(p) * pos
    neg_loss = ((1 - gt) ** FOCAL_BETA) * (p ** FOCAL_ALPHA) * torch.log(1 - p) * neg
    n_pos = pos.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / n_pos


def _masked_smooth_l1(pred, target, mask):
    """mask: (B,G,G) bool -> smooth-L1 averaged over positive cells (per-channel)."""
    m = mask.unsqueeze(1).float()                              # (B,1,G,G)
    n = m.sum().clamp(min=1)
    loss = F.smooth_l1_loss(pred * m, target * m, reduction="none")
    return loss.sum() / n


def probe_loss(out, targets):
    hm_logit, off, size = ProbeHead.split(out)
    l_hm = focal_heatmap_loss(hm_logit, targets["heatmap"])
    l_off = _masked_smooth_l1(off, targets["offset"], targets["reg_mask"])
    l_size = _masked_smooth_l1(size, targets["size"], targets["reg_mask"])
    total = W_HM * l_hm + W_OFF * l_off + W_SIZE * l_size
    return total, {"hm": l_hm.item(), "off": l_off.item(), "size": l_size.item(),
                   "total": total.item()}
