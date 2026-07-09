"""BoxInst losses: detection (focal + L1 + GIoU) and box-supervised mask terms.

Mask supervision uses bounding boxes ONLY:
  - projection: max over rows / cols of the sigmoid mask must match the box's 1-D
    extent (dice on the two 1-D profiles) — forces the mask to fill its box;
  - pairwise: neighbouring pixels inside the box whose DINO-feature cosine
    similarity exceeds tau are pushed to share a label (−log P(same)).

Detection reuses the frozen dapt CenterNet losses (penalty-reduced focal +
masked smooth-L1 on offset/log-size) and adds GIoU on the decoded box at each
positive cell.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss

from dapt.head import _masked_smooth_l1, focal_heatmap_loss
from boxinst.model import (GRID, MASK_RES, MASK_STRIDE, STRIDE, dynamic_masks,
                           signature_cosine)

# cached pairwise-map offsets (must match boxinst.cache_feats.OFFSETS)
PAIR_OFFSETS = [(0, 2), (2, 0), (2, 2), (2, -2),
                (0, 4), (4, 0), (4, 4), (4, -4)]


def det_loss(det, targets):
    """det:(B,5,G,G); targets from dapt.targets.encode (stacked).

    targets may carry 'ignore' (B,G,G): canopy cells excluded from the negative
    focal loss (valid-but-unlabelled), for ITC on TCD."""
    hm_logit, off, size = det[:, :1], det[:, 1:3], det[:, 3:5]
    l_hm = focal_heatmap_loss(hm_logit, targets["heatmap"], targets.get("ignore"))
    l_off = _masked_smooth_l1(off, targets["offset"], targets["reg_mask"])
    l_size = _masked_smooth_l1(size, targets["size"], targets["reg_mask"])
    # GIoU at positive cells: decode pred + target boxes from (offset, log-size)
    b, gy, gx = targets["reg_mask"].nonzero(as_tuple=True)
    if len(b):
        def to_box(o, s):
            cx = (gx.float() + o[b, 0, gy, gx]) * STRIDE
            cy = (gy.float() + o[b, 1, gy, gx]) * STRIDE
            w = s[b, 0, gy, gx].clamp(max=8).exp()
            h = s[b, 1, gy, gx].clamp(max=8).exp()
            return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        l_giou = generalized_box_iou_loss(
            to_box(off, size), to_box(targets["offset"], targets["size"]),
            reduction="mean")
    else:
        l_giou = det.sum() * 0.0
    return l_hm, l_off, l_size, l_giou


def gather_instances(boxes_list, device):
    """Per-image GT boxes -> flat instance tensors.

    Returns img_idx (N,), boxes_px (N,4), cells (N,2)=(gy,gx), centers_mask (N,2)=
    (cx,cy) in mask-grid px. Centre cell matches dapt.targets.encode exactly.
    """
    img_idx, boxes, cells, centers = [], [], [], []
    for i, bx in enumerate(boxes_list):
        for (x0, y0, x1, y1) in bx:
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            gx = min(int(cx // STRIDE), GRID - 1)
            gy = min(int(cy // STRIDE), GRID - 1)
            img_idx.append(i)
            boxes.append((x0, y0, x1, y1))
            cells.append((gy, gx))
            centers.append((cx / MASK_STRIDE, cy / MASK_STRIDE))
    if not img_idx:
        return None
    return (torch.tensor(img_idx, device=device),
            torch.tensor(boxes, dtype=torch.float32, device=device),
            torch.tensor(cells, dtype=torch.long, device=device),
            torch.tensor(centers, dtype=torch.float32, device=device))


def _box_1d_targets(boxes_px, res=MASK_RES):
    """(N,4) px -> x-profile (N,res), y-profile (N,res) {0,1} at mask res."""
    N = boxes_px.shape[0]
    dev = boxes_px.device
    r = torch.arange(res, device=dev, dtype=torch.float32)
    bx = boxes_px / MASK_STRIDE
    tx = ((r[None] >= bx[:, 0:1].floor()) & (r[None] < bx[:, 2:3].ceil())).float()
    ty = ((r[None] >= bx[:, 1:2].floor()) & (r[None] < bx[:, 3:4].ceil())).float()
    return tx, ty


def _box_2d_masks(boxes_px, res=MASK_RES):
    tx, ty = _box_1d_targets(boxes_px, res)
    return ty[:, :, None] * tx[:, None, :]                    # (N,res,res)


def projection_loss(mask_logits, boxes_px):
    """Dice between axis-max of sigmoid mask and the box's 1-D extents."""
    p = torch.sigmoid(mask_logits)                            # (N,R,R)
    px, py = p.max(dim=1).values, p.max(dim=2).values         # over rows / cols
    tx, ty = _box_1d_targets(boxes_px)

    def dice(a, b):
        num = 2 * (a * b).sum(1)
        den = (a * a).sum(1) + (b * b).sum(1) + 1e-5
        return (1 - num / den).mean()

    return dice(px, tx) + dice(py, ty)


def pairwise_loss(mask_logits, boxes_px, sims_per_inst, tau, dil_idx):
    """-log P(same label) over high-affinity neighbour pairs anchored in the box.

    Gating is ONE-endpoint (the anchor pixel must be in the box; the neighbour may
    be outside), as in BoxInst. This is load-bearing: out-of-box pixels are driven
    to 0 by the projection term, so ties across the box border are the only force
    that carves background out of box corners. Gating BOTH endpoints in-box makes
    "fill the whole box" a global minimum of proj+pairwise (observed empirically:
    masks converge to rounded rectangles).

    sims_per_inst: (N,8,R,R) cached DINO cosine maps for each instance's image.
    dil_idx: which of the 8 cached (direction, dilation) offsets to use.
    """
    lp = F.logsigmoid(mask_logits)                            # log p      (N,R,R)
    ln = F.logsigmoid(-mask_logits)                           # log (1-p)
    inbox = _box_2d_masks(boxes_px)                           # (N,R,R)
    total, weight = 0.0, 0.0
    for k in dil_idx:
        dy, dx = PAIR_OFFSETS[k]
        lp_j = torch.roll(lp, shifts=(-dy, -dx), dims=(1, 2))
        ln_j = torch.roll(ln, shifts=(-dy, -dx), dims=(1, 2))
        sim = sims_per_inst[:, k]
        w = ((sim >= tau) & (inbox > 0)).float()
        # zero the wrapped border
        if dy > 0:
            w[:, MASK_RES - dy:, :] = 0
        if dx > 0:
            w[:, :, MASK_RES - dx:] = 0
        elif dx < 0:
            w[:, :, :(-dx)] = 0
        log_same = torch.logaddexp(lp + lp_j, ln + ln_j)
        total = total - (w * log_same).sum()
        weight = weight + w.sum()
    return total / weight.clamp(min=1.0)


def prototype_loss(logits, boxes_px, img_idx, z_batch, c_ema, margin=0.3):
    """Cross-box commonality: mask-weighted box embedding -> shared crown prototype.

    logits:(N,R,R) instance masks; z_batch:(B,D,G,G) L2-normed PCA features;
    c_ema:(D,) running crown prototype (detached, updated outside). Pools each
    instance's mask-weighted embedding over its box's 32-grid cells and pulls it
    toward c_ema (1-cos). Gradient flows through the mask -> the head learns to
    select the sub-region of each box whose latent matches the population crown.
    Returns (loss, z_b detached (N,D) for the EMA update).
    """
    N, R, _ = logits.shape
    dev = logits.device
    D, G = z_batch.shape[1], z_batch.shape[2]
    prob32 = F.avg_pool2d(torch.sigmoid(logits)[:, None], R // G)[:, 0]  # (N,G,G)
    cy, cx = torch.meshgrid(torch.arange(G, device=dev), torch.arange(G, device=dev),
                            indexing="ij")
    cxp, cyp = (cx + 0.5) * (R // G) * MASK_STRIDE, (cy + 0.5) * (R // G) * MASK_STRIDE
    zb = []
    for i in range(N):
        x0, y0, x1, y1 = boxes_px[i]
        inbox = (cxp >= x0) & (cxp < x1) & (cyp >= y0) & (cyp < y1)
        w = (prob32[i] * inbox).flatten()                      # (G*G,)
        z = z_batch[img_idx[i]].reshape(D, -1)                 # (D,G*G)
        zb.append((z * w).sum(1) / w.sum().clamp(min=1e-3))
    zb = torch.stack(zb)                                       # (N,D)
    zb_n = F.normalize(zb, dim=1)
    cos = zb_n @ F.normalize(c_ema, dim=0)
    loss = (1 - cos).mean()
    return loss, zb_n.detach()


def mask_losses(ctrl, fmask, sims, boxes_list, tau, dil_idx, max_inst=192,
                z_batch=None):
    """Assemble per-instance dynamic masks for all GT boxes and score them.

    ctrl:(B,n_dyn,G,G)  fmask:(B,8,R,R)  sims:(B,8,R,R)  boxes_list: list of (n,4).
    z_batch:(B,D,G,G) is required iff the head uses the signature channel.
    Returns (l_proj, l_pair, n_instances, extras) where extras carries the mask
    logits + per-instance (img_idx, boxes_px) for an optional prototype loss.
    """
    inst = gather_instances(boxes_list, fmask.device)
    if inst is None:
        z = fmask.sum() * 0.0
        return z, z, 0, None
    img_idx, boxes_px, cells, centers = inst
    if len(img_idx) > max_inst:                               # cap batch memory
        keep = torch.randperm(len(img_idx), device=img_idx.device)[:max_inst]
        img_idx, boxes_px, cells, centers = (img_idx[keep], boxes_px[keep],
                                             cells[keep], centers[keep])
    params = ctrl[img_idx, :, cells[:, 0], cells[:, 1]]       # (N,n_dyn)
    sig = (signature_cosine(z_batch, img_idx, boxes_px)
           if z_batch is not None else None)
    logits = dynamic_masks(fmask[img_idx], centers, params, sig=sig)   # (N,R,R)
    l_proj = projection_loss(logits, boxes_px)
    l_pair = pairwise_loss(logits, boxes_px, sims[img_idx], tau, dil_idx)
    extras = {"logits": logits, "img_idx": img_idx, "boxes_px": boxes_px}
    return l_proj, l_pair, len(img_idx), extras
