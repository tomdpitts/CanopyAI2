"""Single head for box-supervised instance segmentation on frozen DINOv3 features.

One forward pass produces, from the cached (C,32,32) patch features:
  - CenterNet detection maps at the 32x32 grid: heatmap(1) + offset(2) + logsize(2)
  - a CondInst controller map (169 ch): dynamic-conv weights per location
  - a shared mask-feature map F_mask (8 ch) at the 128x128 (stride-4) grid

Instance masks come from running each instance's 3-layer/8-channel dynamic conv over
F_mask concatenated with that instance's relative-coordinate channels. Everything
here is trainable; the backbone never appears (features are cached).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = 32            # detection grid (= DINOv3 patch grid at 512 px)
STRIDE = 16          # image px per detection cell
MASK_RES = 128       # mask-feature resolution (stride 4)
MASK_STRIDE = 4
MASK_CH = 8          # dynamic-conv width
REL_NORM = 64.0      # rel-coord divisor (mask px); 64 = half the mask grid

# dynamic conv: (MASK_CH + 2 coord + sig_ch -> 8) -> (8 -> 8) -> (8 -> 1).
# sig_ch=0 -> 169 params (baseline); sig_ch=1 -> 177 (per-instance signature channel).
def dyn_shapes(sig_ch: int = 0):
    return [((MASK_CH + 2 + sig_ch), MASK_CH), (MASK_CH, MASK_CH), (MASK_CH, 1)]


def n_dyn(sig_ch: int = 0):
    return sum(i * o + o for i, o in dyn_shapes(sig_ch))


DYN_SHAPES = dyn_shapes(0)
N_DYN = n_dyn(0)                                 # 169 (baseline default)


def _gn(c):
    return nn.GroupNorm(32 if c % 32 == 0 else 8, c)


class BoxInstHead(nn.Module):
    def __init__(self, in_dim: int, width: int = 256, commonality_ch: int = 0,
                 det_tower: int = 2, use_sig: bool = False):
        """commonality_ch: extra input maps (e.g. the LDA "treeness" channel)
        concatenated to the mask neck input. 0 = baseline (no commonality).
        det_tower: number of 3x3 conv layers before the detection heads. 0 = a
        lean per-patch probe (stem 1x1 -> heads), lower capacity / less overfit.
        use_sig: add a per-instance cosine-to-box-signature channel to the dynamic
        mask conv (the controller then emits n_dyn(1)=177 params)."""
        super().__init__()
        self.commonality_ch = commonality_ch
        self.sig_ch = 1 if use_sig else 0
        self.stem = nn.Sequential(nn.Conv2d(in_dim, width, 1), _gn(width),
                                  nn.ReLU(inplace=True))
        tower = []
        for _ in range(det_tower):
            tower += [nn.Conv2d(width, width, 3, padding=1), _gn(width),
                      nn.ReLU(inplace=True)]
        self.tower = nn.Sequential(*tower) if tower else nn.Identity()
        self.hm = nn.Conv2d(width, 1, 1)
        self.reg = nn.Conv2d(width, 4, 1)          # offset(2) + logsize(2)
        self.ctrl = nn.Conv2d(width, n_dyn(self.sig_ch), 1)
        # mask neck: (width+2 coord+commonality) @32 -> x2 -> x2 -> 8ch @128
        self.neck = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width + 2 + commonality_ch, 128, 3, padding=1),
                          _gn(128), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), _gn(64),
                          nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), _gn(64),
                          nn.ReLU(inplace=True), nn.Conv2d(64, MASK_CH, 1)),
        ])
        self._init()

    def _init(self):
        nn.init.constant_(self.hm.bias, -2.19)     # focal prior p~0.1
        nn.init.normal_(self.ctrl.weight, std=0.01)
        nn.init.constant_(self.ctrl.bias, 0.0)
        # offset targets live in [0,1); log-size ~ log(30px)
        nn.init.constant_(self.reg.bias[0], 0.5)
        nn.init.constant_(self.reg.bias[1], 0.5)
        nn.init.constant_(self.reg.bias[2], 3.4)
        nn.init.constant_(self.reg.bias[3], 3.4)

    @staticmethod
    def coord_grid(size, device):
        """(2,size,size) x/y coords in [-1,1]."""
        r = torch.linspace(-1, 1, size, device=device)
        y, x = torch.meshgrid(r, r, indexing="ij")
        return torch.stack([x, y])

    def forward(self, feat, commonality=None):
        """feat:(B,C,32,32), commonality:(B,commonality_ch,32,32) or None
        -> det (B,5,32,32), ctrl (B,169,32,32), F_mask (B,8,128,128)."""
        B = feat.shape[0]
        s = self.stem(feat)
        t = self.tower(s)
        det = torch.cat([self.hm(t), self.reg(t)], dim=1)
        ctrl = self.ctrl(t)
        coords = self.coord_grid(GRID, feat.device).expand(B, -1, -1, -1)
        neck_in = [s, coords]
        if self.commonality_ch:
            neck_in.append(commonality)
        m = self.neck[0](torch.cat(neck_in, dim=1))
        m = F.interpolate(m, scale_factor=2, mode="bilinear", align_corners=False)
        m = self.neck[1](m)
        m = F.interpolate(m, scale_factor=2, mode="bilinear", align_corners=False)
        fmask = self.neck[2](m)
        return det, ctrl, fmask


def split_dynamic_params(params: torch.Tensor, sig_ch: int = 0):
    """params:(N,n_dyn) -> list of (weight (N,out,in), bias (N,out)) per layer."""
    out, i = [], 0
    for cin, cout in dyn_shapes(sig_ch):
        w = params[:, i:i + cin * cout].reshape(-1, cout, cin)
        i += cin * cout
        b = params[:, i:i + cout]
        i += cout
        out.append((w, b))
    return out


def signature_cosine(z_batch: torch.Tensor, img_idx: torch.Tensor,
                     boxes_px: torch.Tensor, res: int = MASK_RES) -> torch.Tensor:
    """Per-instance cosine-to-box-signature map.

    z_batch: (B, D, G, G) L2-normed per-cell PCA DINO features.
    For each instance, pool z over the cells inside its box (unweighted), normalize
    -> the box "signature"; cos(each cell, signature) -> (G,G) -> upsample to res.
    Returns (N, res, res) in [-1, 1].
    """
    B, D, G, _ = z_batch.shape
    dev = z_batch.device
    r = torch.arange(G, device=dev, dtype=torch.float32)
    cyc, cxc = torch.meshgrid(r, r, indexing="ij")
    cxp, cyp = (cxc + 0.5) * STRIDE, (cyc + 0.5) * STRIDE          # cell centres px
    maps = []
    for i in range(len(img_idx)):
        x0, y0, x1, y1 = boxes_px[i]
        inbox = ((cxp >= x0) & (cxp < x1) & (cyp >= y0) & (cyp < y1)).flatten()
        z = z_batch[img_idx[i]].reshape(D, -1)                    # (D, G*G)
        w = inbox.float()
        sig = (z * w).sum(1) / w.sum().clamp(min=1)               # (D,)
        sig = F.normalize(sig, dim=0)
        cos = (z * sig[:, None]).sum(0).reshape(1, 1, G, G)       # z already normed
        maps.append(F.interpolate(cos, size=(res, res), mode="bilinear",
                                  align_corners=False)[0, 0])
    return torch.stack(maps) if maps else z_batch.new_zeros((0, res, res))


def dynamic_masks(fmask_per_inst: torch.Tensor, centers: torch.Tensor,
                  params: torch.Tensor, sig: torch.Tensor = None) -> torch.Tensor:
    """Run each instance's dynamic conv (1x1, so a per-pixel MLP) over its features.

    fmask_per_inst: (N, MASK_CH, R, R) — F_mask of the instance's image
    centers:        (N, 2) instance centre (x, y) in mask-grid px
    params:         (N, n_dyn) controller outputs at the instance's cell
    sig:            (N, R, R) optional cosine-to-signature channel (from
                    signature_cosine); its presence must match the model's use_sig.
    returns mask logits (N, R, R)
    """
    N, _, R, _ = fmask_per_inst.shape
    dev = fmask_per_inst.device
    sig_ch = 1 if sig is not None else 0
    r = torch.arange(R, device=dev, dtype=torch.float32)
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    rel_x = (xx[None] - centers[:, 0, None, None]) / REL_NORM     # (N,R,R)
    rel_y = (yy[None] - centers[:, 1, None, None]) / REL_NORM
    chans = [fmask_per_inst, rel_x[:, None], rel_y[:, None]]
    if sig is not None:
        chans.append(sig[:, None])
    x = torch.cat(chans, dim=1).flatten(2)                       # (N, 10|11, R*R)
    layers = split_dynamic_params(params, sig_ch)
    for li, (w, b) in enumerate(layers):
        x = torch.bmm(w, x) + b[:, :, None]
        if li < len(layers) - 1:
            x = F.relu(x)
    return x.reshape(N, R, R)
