"""FCOS-style head on LACE's frozen 8 px L24 features (pre-registration: dethead/README.md).

Trunk = Detector4Phase's modules verbatim (stem, tower, up); only the heads, targets and losses differ.
Anchor point of cell (X, Y) = pixel (8X + 8, 8Y + 8) (the 4-phase grid's true cell centre, HYPERPARAMS.md).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss, nms, sigmoid_focal_loss

from boxinst_commonality_tcd_04.detector import Detector8

STRIDE, G, ORIGIN = 8, 256, 8.0
CENTER_RADIUS = 1.5                      # strides (FCOS centre sampling)
PRE_THRESH, PRE_TOPK, NMS_IOU, MAX_DET = 0.05, 4000, 0.6, 600


class FCOS8(Detector8):
    """(B,1024,256,256) -> (B,6,256,256): cls logit, ltrb (log, units of stride), ctr logit."""

    def __init__(self, in_dim: int, width: int = 256, tower: int = 3):
        super().__init__(in_dim, width, tower)
        del self.hm, self.reg
        self.cls = nn.Conv2d(width, 1, 1)
        self.box = nn.Conv2d(width, 4, 1)
        self.ctr = nn.Conv2d(width, 1, 1)
        nn.init.normal_(self.cls.weight, std=0.01); nn.init.constant_(self.cls.bias, -math.log(99))   # p = 0.01
        nn.init.normal_(self.box.weight, std=0.01); nn.init.constant_(self.box.bias, math.log(2.25))  # ~18 px half-size
        nn.init.normal_(self.ctr.weight, std=0.01); nn.init.constant_(self.ctr.bias, 0.0)

    def forward(self, feat):
        x = self.up(self.tower(self.stem(feat)))            # = Detector4Phase trunk (no interpolate)
        return torch.cat([self.cls(x), self.box(x), self.ctr(x)], dim=1)


def anchors(g=G):
    a = np.arange(g, dtype=np.float32) * STRIDE + ORIGIN
    return a                                               # same for x and y


def encode(boxes: np.ndarray, g=G):
    """GT boxes (N,4) xyxy px -> pos (g,g) bool, ltrb (4,g,g) px, ctr (g,g), n_fallback."""
    a = anchors(g)
    area_map = np.full((g, g), np.inf, np.float32)
    ltrb = np.zeros((4, g, g), np.float32)
    pos = np.zeros((g, g), bool)
    nfb = 0
    order = np.argsort(-((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))) if len(boxes) else []
    for i in order:                                         # largest first -> smallest wins ties
        x0, y0, x1, y1 = (float(v) for v in boxes[i])
        area = max(x1 - x0, 1e-3) * max(y1 - y0, 1e-3)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rad = CENTER_RADIUS * STRIDE
        lo_x, hi_x = max(x0, cx - rad), min(x1, cx + rad)
        lo_y, hi_y = max(y0, cy - rad), min(y1, cy + rad)
        xs = np.nonzero((a > lo_x) & (a < hi_x))[0]
        ys = np.nonzero((a > lo_y) & (a < hi_y))[0]
        if len(xs) == 0 or len(ys) == 0:                    # tiny box: nearest anchor to the centre
            xs = np.array([int(np.clip(np.round((cx - ORIGIN) / STRIDE), 0, g - 1))])
            ys = np.array([int(np.clip(np.round((cy - ORIGIN) / STRIDE), 0, g - 1))])
            nfb += 1
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        win = area < area_map[Y, X]
        Y, X = Y[win], X[win]
        if not len(Y):
            continue
        ax, ay = a[X], a[Y]
        area_map[Y, X] = area
        pos[Y, X] = True
        ltrb[0, Y, X] = np.maximum(ax - x0, 1.0); ltrb[1, Y, X] = np.maximum(ay - y0, 1.0)
        ltrb[2, Y, X] = np.maximum(x1 - ax, 1.0); ltrb[3, Y, X] = np.maximum(y1 - ay, 1.0)
    l, t, r, b = ltrb
    ctr = np.zeros((g, g), np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        ctr[pos] = np.sqrt((np.minimum(l, r) / np.maximum(l, r) * np.minimum(t, b) / np.maximum(t, b))[pos])
    return pos, ltrb, ctr, nfb


def _boxes(ltrb_px, ys, xs):
    a = torch.as_tensor(anchors(), device=ltrb_px.device)
    ax, ay = a[xs], a[ys]
    return torch.stack([ax - ltrb_px[0], ay - ltrb_px[1], ax + ltrb_px[2], ay + ltrb_px[3]], 1)


def loss(out, tgt):
    """out (B,6,g,g); tgt: pos (B,g,g) bool, ltrb (B,4,g,g) px, ctr (B,g,g), ignore (B,g,g) bool canopy."""
    cls, box, ctr = out[:, 0], out[:, 1:5], out[:, 5]
    pos, ign = tgt["pos"], tgt["ignore"]
    npos = pos.sum().clamp(min=1).float()
    valid = pos | ~ign                                      # canopy cells: no negative loss (LACE's rule)
    l_cls = sigmoid_focal_loss(cls[valid], pos[valid].float(), alpha=0.25, gamma=2.0, reduction="sum") / npos
    b, ys, xs = pos.nonzero(as_tuple=True)
    if len(b):
        pr = (box[b, :, ys, xs].clamp(max=8).exp() * STRIDE).T          # (4, P) px
        gt = tgt["ltrb"][b, :, ys, xs].T
        w = tgt["ctr"][b, ys, xs]
        l_box = (generalized_box_iou_loss(_boxes(pr, ys, xs), _boxes(gt, ys, xs), reduction="none") * w).sum() / w.sum().clamp(min=1e-6)
        l_ctr = F.binary_cross_entropy_with_logits(ctr[b, ys, xs], w, reduction="sum") / npos
    else:
        l_box = l_ctr = out.sum() * 0.0
    return l_cls, l_box, l_ctr


@torch.no_grad()
def decode(out):
    """out (1,6,g,g) -> boxes (M,4) xyxy px, scores (M,) on CPU, M <= MAX_DET."""
    out = out[0].float().cpu()
    s = torch.sqrt(torch.sigmoid(out[0]) * torch.sigmoid(out[5]))
    flat = s.flatten()
    keep = (flat > PRE_THRESH).nonzero(as_tuple=True)[0]
    if keep.numel() == 0:
        return torch.zeros((0, 4)), torch.zeros((0,))
    sc = flat[keep]
    if sc.numel() > PRE_TOPK:
        sc, k = sc.topk(PRE_TOPK); keep = keep[k]
    g = s.shape[0]
    ys, xs = keep // g, keep % g
    ltrb = (out[1:5, ys, xs].clamp(max=8).exp() * STRIDE)
    bx = _boxes(ltrb, ys, xs).clamp(0, 2048)
    k = nms(bx, sc, NMS_IOU)[:MAX_DET]
    return bx[k], sc[k]


# ================================================================== pyramid (pre-registration 2, README.md)
LEVELS = ((8, 256, 0.0, 64.0), (16, 128, 64.0, 128.0), (32, 64, 128.0, float("inf")))   # stride, grid, m range


class FCOS8FPN(FCOS8):
    """FCOS8 + P4/P5 by stride-2 convs on the unchanged P3 trunk map; shared 1x1 heads; per-level box scale."""

    def __init__(self, in_dim: int, width: int = 256, tower: int = 3):
        super().__init__(in_dim, width, tower)
        from boxinst_commonality_tcd_04.detector import _gn
        self.p4 = nn.Sequential(nn.Conv2d(width, width, 3, stride=2, padding=1), _gn(width), nn.ReLU(inplace=True))
        self.p5 = nn.Sequential(nn.Conv2d(width, width, 3, stride=2, padding=1), _gn(width), nn.ReLU(inplace=True))
        self.scales = nn.Parameter(torch.ones(len(LEVELS)))

    def forward(self, feat):
        p3 = self.up(self.tower(self.stem(feat))); p4 = self.p4(p3); p5 = self.p5(p4)
        return [torch.cat([self.cls(p), self.box(p) * self.scales[i], self.ctr(p)], dim=1) for i, p in enumerate((p3, p4, p5))]


def anchors_l(stride, g):
    return np.arange(g, dtype=np.float32) * stride + ORIGIN


def encode_fpn(boxes: np.ndarray, levels=LEVELS):
    """-> list per level of (pos (g,g), ltrb (4,g,g) px, ctr (g,g)), n_fallback (P3 only)."""
    out, nfb = [], 0
    order = np.argsort(-((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))) if len(boxes) else []
    for li, (stride, g, lo, hi) in enumerate(levels):
        a = anchors_l(stride, g)
        area_map = np.full((g, g), np.inf, np.float32); ltrb = np.zeros((4, g, g), np.float32); pos = np.zeros((g, g), bool)
        for i in order:
            x0, y0, x1, y1 = (float(v) for v in boxes[i])
            area = max(x1 - x0, 1e-3) * max(y1 - y0, 1e-3); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rad = CENTER_RADIUS * stride
            xs = np.nonzero((a > max(x0, cx - rad)) & (a < min(x1, cx + rad)))[0]
            ys = np.nonzero((a > max(y0, cy - rad)) & (a < min(y1, cy + rad)))[0]
            fb = False
            if (len(xs) == 0 or len(ys) == 0):
                if li != 0:
                    continue
                xs = np.array([int(np.clip(np.round((cx - ORIGIN) / stride), 0, g - 1))])
                ys = np.array([int(np.clip(np.round((cy - ORIGIN) / stride), 0, g - 1))]); fb = True
            Y, X = np.meshgrid(ys, xs, indexing="ij"); ax, ay = a[X], a[Y]
            l, t, r, b = ax - x0, ay - y0, x1 - ax, y1 - ay
            m = np.maximum(np.maximum(l, t), np.maximum(r, b))
            ok = ((m > lo) & (m <= hi)) | (fb & (li == 0))
            win = ok & (area < area_map[Y, X])
            if fb and win.any():
                nfb += 1
            Y, X = Y[win], X[win]
            if not len(Y):
                continue
            ax, ay = a[X], a[Y]
            area_map[Y, X] = area; pos[Y, X] = True
            ltrb[0, Y, X] = np.maximum(ax - x0, 1.0); ltrb[1, Y, X] = np.maximum(ay - y0, 1.0)
            ltrb[2, Y, X] = np.maximum(x1 - ax, 1.0); ltrb[3, Y, X] = np.maximum(y1 - ay, 1.0)
        l, t, r, b = ltrb
        ctr = np.zeros((g, g), np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            ctr[pos] = np.sqrt((np.minimum(l, r) / np.maximum(l, r) * np.minimum(t, b) / np.maximum(t, b))[pos])
        out.append((pos, ltrb, ctr))
    return out, nfb


def _boxes_l(ltrb_px, ys, xs, stride, g):
    a = torch.as_tensor(anchors_l(stride, g), device=ltrb_px.device)
    ax, ay = a[xs], a[ys]
    return torch.stack([ax - ltrb_px[0], ay - ltrb_px[1], ax + ltrb_px[2], ay + ltrb_px[3]], 1)


def loss_fpn(outs, tgt, levels=LEVELS):
    """outs: list per level (B,6,g,g); tgt keys pos{l}, ltrb{l}, ctr{l}, ignore{l}."""
    cls_p, cls_t, gi_num, gi_den, ct_sum = [], [], 0.0, 0.0, 0.0
    npos = sum(tgt[f"pos{l}"].sum() for l in range(len(levels))).clamp(min=1).float()
    for l, (stride, g, _, _) in enumerate(levels):
        out = outs[l]; pos, ign = tgt[f"pos{l}"], tgt[f"ignore{l}"]
        valid = pos | ~ign
        cls_p.append(out[:, 0][valid]); cls_t.append(pos[valid].float())
        b, ys, xs = pos.nonzero(as_tuple=True)
        if len(b):
            pr = (out[:, 1:5][b, :, ys, xs].clamp(max=8).exp() * stride).T
            gt = tgt[f"ltrb{l}"][b, :, ys, xs].T; w = tgt[f"ctr{l}"][b, ys, xs]
            gi_num = gi_num + (generalized_box_iou_loss(_boxes_l(pr, ys, xs, stride, g), _boxes_l(gt, ys, xs, stride, g), reduction="none") * w).sum()
            gi_den = gi_den + w.sum()
            ct_sum = ct_sum + F.binary_cross_entropy_with_logits(out[:, 5][b, ys, xs], w, reduction="sum")
    l_cls = sigmoid_focal_loss(torch.cat(cls_p), torch.cat(cls_t), alpha=0.25, gamma=2.0, reduction="sum") / npos
    zero = outs[0].sum() * 0.0
    l_box = gi_num / gi_den.clamp(min=1e-6) if torch.is_tensor(gi_den) else zero
    l_ctr = ct_sum / npos if torch.is_tensor(ct_sum) else zero
    return l_cls, l_box, l_ctr


@torch.no_grad()
def decode_fpn(outs, nms_iou=None, levels=LEVELS, return_level=False, qfl=False):
    nms_iou = NMS_IOU if nms_iou is None else nms_iou
    allb, alls, alll = [], [], []
    for l, (stride, g, _, _) in enumerate(levels):
        out = outs[l][0].float().cpu()
        s = torch.sigmoid(out[0]) if qfl else torch.sqrt(torch.sigmoid(out[0]) * torch.sigmoid(out[5])); flat = s.flatten()   # qfl: C2 (prereg 6)
        keep = (flat > PRE_THRESH).nonzero(as_tuple=True)[0]
        if keep.numel() == 0:
            continue
        sc = flat[keep]
        if sc.numel() > PRE_TOPK:
            sc, k = sc.topk(PRE_TOPK); keep = keep[k]
        ys, xs = keep // g, keep % g
        allb.append(_boxes_l(out[1:5, ys, xs].clamp(max=8).exp() * stride, ys, xs, stride, g).clamp(0, 2048)); alls.append(sc)
        alll.append(torch.full((sc.numel(),), l, dtype=torch.long))
    if not allb:
        return (torch.zeros((0, 4)), torch.zeros((0,))) + ((torch.zeros((0,), dtype=torch.long),) if return_level else ())
    bx, sc, lv = torch.cat(allb), torch.cat(alls), torch.cat(alll)
    k = nms(bx, sc, nms_iou)[:MAX_DET]
    return (bx[k], sc[k], lv[k]) if return_level else (bx[k], sc[k])


# ================================================================== arm C (pre-registration 4, README.md)
LEVELS_C = ((4, 512, 0.0, 32.0), (8, 256, 32.0, 64.0), (16, 128, 64.0, 128.0), (32, 64, 128.0, float("inf")))
RGB_MEAN, RGB_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _cgr(cin, cout, stride=1):
    from boxinst_commonality_tcd_04.detector import _gn
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1), _gn(cout), nn.ReLU(inplace=True))


class FCOSC(FCOS8FPN):
    """Arm C: FCOS8FPN on the 1024-d 8 px features + P2 (stride 4) from the 128-d 4 px features + a pixel branch on RGB.
    forward(feat8 (B,1024,256,256), z4 (B,128,512,512), rgb (B,3,2048,2048) in [0,1]) -> list of 4 level maps (P2..P5).
    The pixel branch enters P3 and P2 through zero-initialised 1x1 projections (training starts at the P2-extended head)."""

    def __init__(self, in_dim: int = 1024, z_dim: int = 128, width: int = 256, tower: int = 3):
        super().__init__(in_dim, width, tower)
        from boxinst_commonality_tcd_04.detector import _gn
        self.scales = nn.Parameter(torch.ones(len(LEVELS_C)))
        self.pix2 = nn.Sequential(_cgr(3, 32, 2), _cgr(32, 32))                 # stride 2
        self.pix4 = nn.Sequential(_cgr(32, 64, 2), _cgr(64, 64))                # stride 4 -> P2
        self.pix8 = _cgr(64, 128, 2)                                            # stride 8 -> P3
        self.proj4 = nn.Conv2d(64, width, 1); self.proj8 = nn.Conv2d(128, width, 1)
        for m in (self.proj4, self.proj8):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        self.lat2 = nn.Sequential(nn.Conv2d(z_dim, width, 1), _gn(width), nn.ReLU(inplace=True))
        self.p2 = _cgr(width, width)
        self.register_buffer("rgb_mean", torch.tensor(RGB_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("rgb_std", torch.tensor(RGB_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, feat8, z4, rgb):
        x = (rgb - self.rgb_mean) / self.rgb_std
        s4 = self.pix4(self.pix2(x)); s8 = self.pix8(s4)
        p3 = self.up(self.tower(self.stem(feat8))) + self.proj8(s8)
        p4 = self.p4(p3); p5 = self.p5(p4)
        p2 = self.p2(self.lat2(z4) + F.interpolate(p3, scale_factor=2, mode="nearest") + self.proj4(s4))
        return [torch.cat([self.cls(p), self.box(p) * self.scales[i], self.ctr(p)], dim=1) for i, p in enumerate((p2, p3, p4, p5))]


# ================================================================== arm C2 (pre-registration 6, README.md)
class FCOSC2(FCOSC):
    """Arm C + a 3-layer P2 stack (mirrors the 8 px trunk) and no centre-ness: outputs (B,5,g,g) = cls logit, ltrb."""

    def __init__(self, in_dim: int = 1024, z_dim: int = 128, width: int = 256, tower: int = 3):
        super().__init__(in_dim, z_dim, width, tower)
        self.p2 = nn.Sequential(*[_cgr(width, width) for _ in range(3)])
        del self.ctr

    def forward(self, feat8, z4, rgb):
        x = (rgb - self.rgb_mean) / self.rgb_std
        s4 = self.pix4(self.pix2(x)); s8 = self.pix8(s4)
        p3 = self.up(self.tower(self.stem(feat8))) + self.proj8(s8)
        p4 = self.p4(p3); p5 = self.p5(p4)
        p2 = self.p2(self.lat2(z4) + F.interpolate(p3, scale_factor=2, mode="nearest") + self.proj4(s4))
        return [torch.cat([self.cls(p), self.box(p) * self.scales[i]], dim=1) for i, p in enumerate((p2, p3, p4, p5))]


def loss_qfl(outs, tgt, levels=LEVELS_C, beta=2.0):
    """GFL: quality focal loss (target = IoU(pred, GT) detached on positives, 0 on negatives; canopy-ignored cells excluded)
    / #pos, and GIoU on positives weighted by the detached predicted score / its sum. outs: list (B,5,g,g)."""
    from torchvision.ops import box_iou
    npos = sum(tgt[f"pos{l}"].sum() for l in range(len(levels))).clamp(min=1).float()
    q_sum, gi_num, gi_den = 0.0, 0.0, 0.0
    for l, (stride, g, _, _) in enumerate(levels):
        out = outs[l]; pos, ign = tgt[f"pos{l}"], tgt[f"ignore{l}"]
        logit = out[:, 0]; q = torch.zeros_like(logit)
        b, ys, xs = pos.nonzero(as_tuple=True)
        if len(b):
            pr = (out[:, 1:5][b, :, ys, xs].clamp(max=8).exp() * stride).T
            gt = tgt[f"ltrb{l}"][b, :, ys, xs].T
            pb, gb = _boxes_l(pr, ys, xs, stride, g), _boxes_l(gt, ys, xs, stride, g)
            iou = box_iou(pb, gb).diagonal().detach().clamp(0, 1)
            q[b, ys, xs] = iou
            w = torch.sigmoid(logit[b, ys, xs]).detach()
            gi_num = gi_num + (generalized_box_iou_loss(pb, gb, reduction="none") * w).sum(); gi_den = gi_den + w.sum()
        valid = pos | ~ign
        x, t = logit[valid], q[valid]; p = torch.sigmoid(x)
        q_sum = q_sum + (F.binary_cross_entropy_with_logits(x, t, reduction="none") * (p - t).abs().pow(beta)).sum()
    zero = outs[0].sum() * 0.0
    l_q = q_sum / npos
    l_box = gi_num / gi_den.clamp(min=1e-6) if torch.is_tensor(gi_den) else zero
    return l_q, l_box
