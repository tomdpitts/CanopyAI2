"""Frozen DINOv3 backbone + linear segmentation head (canonical dense linear probe).

The backbone is frozen; only the 1x1-conv head trains. This is the standard
DINOv3 dense-eval protocol and the cheapest fair way to compare the satellite
(SAT-493M) vs web (LVD-1689M) pretraining at 0.1 m drone/aerial resolution.

NOTE: untested until gated DINOv3 weights are available (see dino/README.md).
The two spots most likely to need adjustment against the real `transformers`
DINOv3 API are flagged inline: the patch-token slice and the processor mean/std.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


class Dinov3Seg(nn.Module):
    def __init__(self, model_id: str, num_classes: int = 2, device: str = "mps"):
        super().__init__()
        self.device = device
        self.backbone = AutoModel.from_pretrained(model_id).eval().to(device)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        cfg = self.backbone.config
        self.patch = getattr(cfg, "patch_size", 16)
        C = getattr(cfg, "hidden_size")
        self.head = nn.Conv2d(C, num_classes, kernel_size=1).to(device)
        proc = AutoImageProcessor.from_pretrained(model_id)
        self.register_buffer("mean", torch.tensor(proc.image_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(proc.image_std).view(1, 3, 1, 1))
        self.to(device)

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) in [0,1] with H,W divisible by patch -> (B,C,h,w) grid."""
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x).last_hidden_state  # (B, seq, C)
        B, _, C = out.shape
        h, w = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        # patch tokens are the trailing h*w tokens (after CLS + register tokens);
        # slicing from the end is robust to the exact prefix-token count.
        patch_tokens = out[:, -(h * w):, :]
        return patch_tokens.transpose(1, 2).reshape(B, C, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.features(x))               # (B,K,h,w), backbone frozen
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
