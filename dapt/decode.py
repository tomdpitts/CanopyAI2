"""Decode probe output -> scored boxes. peak-pick -> offset -> size -> NMS.

Peak-picking is at the 32x32 grid (3x3 max-pool keeps local maxima, the CenterNet
NMS-free trick); the offset channels place the centre inside its 16 px cell; size is
exp() of the log-size channels. torchvision IoU-NMS is applied as a final guard.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.ops import nms

from dapt.head import ProbeHead
from dapt.targets import CFG


def _local_max(hm, k=3):
    keep = (F.max_pool2d(hm, k, stride=1, padding=k // 2) == hm).float()
    return hm * keep


@torch.no_grad()
def decode(out, score_thr=0.2, topk=200, nms_iou=0.5, stride=CFG.stride):
    """out:(1,5,G,G) -> boxes (M,4) xyxy px, scores (M,). Single tile."""
    hm_logit, off, size = ProbeHead.split(out)
    hm = _local_max(torch.sigmoid(hm_logit))[0, 0]             # (G,G)
    G = hm.shape[0]
    scores, idx = hm.flatten().topk(min(topk, G * G))
    keep = scores > score_thr
    scores, idx = scores[keep], idx[keep]
    if idx.numel() == 0:
        return torch.zeros((0, 4)), torch.zeros((0,))
    gy, gx = idx // G, idx % G
    ox, oy = off[0, 0, gy, gx], off[0, 1, gy, gx]
    w, h = size[0, 0, gy, gx].exp(), size[0, 1, gy, gx].exp()
    cx = (gx.float() + ox) * stride
    cy = (gy.float() + oy) * stride
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
    k = nms(boxes, scores, nms_iou)
    return boxes[k], scores[k]
