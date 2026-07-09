"""Frozen DINOv3 multi-layer patch-feature extractor — shared by all arms.

Identical across web / sat / dapt: only the checkpoint differs. Extracts patch
tokens from several transformer blocks, L2-norms each, and concatenates them into a
per-patch feature grid. See dapt/PLAN.md §3.

Tiles are loaded at native 0.1 m/px (no downsampling) and zero-padded bottom-right to
a fixed 512x512 (→ 32x32 patch grid at patch=16), so mixed 400/500 px tiles share one
grid. The original (H,W) is returned so targets/decode can map back and ignore pad.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_IDS = {
    "web": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "sat": "facebook/dinov3-vitl16-pretrain-sat493m",
}

# Local DAPT checkpoints (Modal-exported HF AutoModel dirs) register in
# dapt/ssl/checkpoints.json as {arm_name: repo_relative_or_abs_path}. Merged here so
# `--arm dapt` (or `dapt_sNNNN` for checkpoint selection) loads an adapted backbone
# with zero pipeline change. Each dir MUST include preprocessor_config.json with the
# arid mean/std it was DAPT'd with (else features aren't normalised as trained).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REGISTRY = os.path.join(_HERE, "ssl", "checkpoints.json")
if os.path.exists(_REGISTRY):
    import json as _json
    for _name, _p in _json.load(open(_REGISTRY)).items():
        MODEL_IDS[_name] = _p if os.path.isabs(_p) else os.path.abspath(os.path.join(_HERE, "..", _p))

LAYERS = (3, 6, 9, 12)   # transformer blocks to aggregate (of ViT-L's 24)
GRID = 512               # padded tile size; 512/16 = 32x32 patches


def pick_device(pref: str | None = None) -> str:
    if pref:
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_tile(path: str, size: int = GRID):
    """Return (x, (H0,W0)): x is (1,3,size,size) float[0,1], zero-padded bottom-right.

    H0,W0 are the tile's true pixel dims (the valid, non-padded region).
    """
    img = Image.open(path).convert("RGB")
    W0, H0 = img.size
    if W0 > size or H0 > size:
        raise ValueError(f"tile {path} is {W0}x{H0}, larger than pad size {size}")
    arr = np.asarray(img, dtype=np.float32) / 255.0          # (H0,W0,3)
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    canvas[:H0, :W0] = arr
    x = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0)  # (1,3,size,size)
    return x, (H0, W0)


class FrozenDinoV3Features(nn.Module):
    def __init__(self, arm: str, layers=LAYERS, device: str | None = None):
        super().__init__()
        if arm not in MODEL_IDS:
            raise ValueError(f"arm must be one of {list(MODEL_IDS)} or a model id; "
                             f"got {arm!r}")
        model_id = MODEL_IDS[arm]
        self.arm = arm
        self.device = pick_device(device)
        self.layers = tuple(layers)
        self.backbone = AutoModel.from_pretrained(model_id).eval().to(self.device)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        cfg = self.backbone.config
        self.patch = getattr(cfg, "patch_size", 16)
        self.hidden = getattr(cfg, "hidden_size")
        self.out_dim = self.hidden * len(self.layers)
        proc = AutoImageProcessor.from_pretrained(model_id)
        self.register_buffer("mean", torch.tensor(proc.image_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(proc.image_std).view(1, 3, 1, 1))
        self.to(self.device)

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """x:(B,3,H,W) in [0,1], H,W % patch == 0 -> (B, out_dim, h, w) grid."""
        x = x.to(self.device)
        x = (x - self.mean) / self.std
        hs = self.backbone(pixel_values=x, output_hidden_states=True).hidden_states
        h, w = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        feats = []
        for li in self.layers:
            tok = hs[li][:, -(h * w):, :]                    # trailing patch tokens
            tok = F.normalize(tok, dim=-1)                   # L2-norm per token
            feats.append(tok.transpose(1, 2).reshape(x.shape[0], self.hidden, h, w))
        return torch.cat(feats, dim=1)                       # (B, C*L, h, w)


if __name__ == "__main__":
    # Smoke test: extract features for one tile per domain, print shapes.
    import argparse
    from dapt.data.build_split import REPO

    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="web")
    ap.add_argument("--tiles", nargs="*", default=[
        "data/finetune/phase5_tiles/bru_tile_1256_0_rot16.tif",
        "data/finetune/phase5_tiles/won_tile_12_rot230.tif",
    ])
    args = ap.parse_args()
    net = FrozenDinoV3Features(args.arm)
    print(f"arm={args.arm} device={net.device} out_dim={net.out_dim} "
          f"patch={net.patch} layers={net.layers}")
    for rel in args.tiles:
        x, (H0, W0) = load_tile(os.path.join(REPO, rel))
        g = net.extract(x)
        print(f"  {os.path.basename(rel):32s} tile={W0}x{H0} -> feat {tuple(g.shape)}")
