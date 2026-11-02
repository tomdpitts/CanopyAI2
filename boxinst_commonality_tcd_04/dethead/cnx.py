"""Frozen DINOv3 ConvNeXt-L + ATSS / GFL head (pre-registration 3, README.md).

Feature cache: feat_cache/feat_cnxl/s{1..4}/<tid>.npy, fp16, strides 4/8/16/32 (192/384/768/1536 ch).
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.cnx extract --split val
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.cnx extract --split train
"""
from __future__ import annotations

import argparse, json, math, os, shutil, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, generalized_box_iou_loss, nms

MODEL = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
FCC = "/Users/tompitts/dphil/feat_cache"
CACHE = f"{FCC}/feat_cnxl"
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); REPO = os.path.dirname(PKG)
STRIDES = (4, 8, 16, 32); GRIDS = (512, 256, 128, 64); CH = (192, 384, 768, 1536)
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
ATSS_TOPK, ANCHOR_SCALE = 9, 8
PRE_THRESH, PRE_TOPK, MAX_DET = 0.05, 1000, 600
WIDTH = 192
MIN_FREE_GB = 15


# ------------------------------------------------------------------ backbone
def load_backbone(device):
    from transformers import AutoModel
    return AutoModel.from_pretrained(MODEL).eval().to(device)


@torch.no_grad()
def extract(model, rgb_uint8, device):
    """(2048,2048,3) uint8 -> list of 4 (C,g,g) fp16 CPU tensors."""
    x = (rgb_uint8.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(x.transpose(2, 0, 1).copy())[None].to(device)
    hs = model(pixel_values=x, output_hidden_states=True).hidden_states[1:5]
    out = [h[0].to(torch.float16).cpu() for h in hs]
    assert [tuple(o.shape) for o in out] == [(c, g, g) for c, g in zip(CH, GRIDS)], [tuple(o.shape) for o in out]
    return out


def rgb_trainval(t):
    """Same pixels as LACE: local tif verified against the phase4 manifest; tile 1878 from the Tier B HF-staged PNG."""
    from PIL import Image
    from boxinst_commonality_tcd_04.native_raster import guided_rgb_ext_val as GV
    if t == "tcd_tile_1878":
        p = f"{FCC}/tierb_rgb_tcd_tile_1878.png"
        if not os.path.exists(p):
            import subprocess
            subprocess.run([os.path.join(REPO, ".venv", "bin", "modal"), "volume", "get", "--force", "tcd04-tierb-vol",
                            "rgb/tcd_tile_1878.png", p], check=True)
        arr = np.asarray(Image.open(p).convert("RGB")); assert arr.shape == (2048, 2048, 3)
        return arr
    man = json.load(open(GV.MANIFEST))["feat_traintile"]
    return GV.load_rgb_verified(t, man)


def rgb_test(t):
    from boxinst_commonality_tcd_04.native_raster import guided_ablate_modal as GA
    man = json.load(open(os.path.join(PKG, "modal_tcd_multiseed", "phase4", "manifest.json")))["feat_test"]
    return (GA._rgb(os.path.join(REPO, "data", "tcd", "test"), t, man) * 255.0 + 0.5).astype(np.uint8)


def cmd_extract(split, reverse=False):
    gt = json.load(open(os.path.join(PKG, "train_tiles_gt.json")))
    tids = sorted((t for t, v in gt.items() if v["partition"] == split), reverse=reverse)   # 2nd worker: reverse order
    for k in range(4):
        os.makedirs(f"{CACHE}/s{k+1}", exist_ok=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    m = load_backbone(dev); t0 = time.time(); n = 0
    for i, t in enumerate(tids, 1):
        if all(os.path.exists(f"{CACHE}/s{k+1}/{t}.npy") for k in range(4)):
            continue
        free = shutil.disk_usage(FCC).free / 2 ** 30
        assert free > MIN_FREE_GB, f"disk guard: {free:.0f} GB free < {MIN_FREE_GB}; stopped before {t} ({i-1}/{len(tids)} done)"
        for k, f in enumerate(extract(m, rgb_trainval(t), dev)):
            p = f"{CACHE}/s{k+1}/{t}.npy"; tmp = f"{p}.{os.getpid()}.tmp.npy"; np.save(tmp, f.numpy()); os.replace(tmp, p)   # pid: 2 workers
        n += 1
        if n % 25 == 0:
            print(f"  [{split}] {i}/{len(tids)} {time.time()-t0:.0f}s free {free:.0f} GB", flush=True)
    print(f"[{split}] done {len(tids)} tiles ({n} new) {time.time()-t0:.0f}s", flush=True)


# ------------------------------------------------------------------ ATSS targets
def centres(stride, g):
    return np.arange(g, dtype=np.float32) * stride + stride / 2.0


def encode_atss(boxes: np.ndarray):
    """-> list per level of int16 (g,g) GT-index maps (-1 = negative)."""
    N = len(boxes)
    idx_maps = [np.full((g, g), -1, np.int16) for g in GRIDS]
    best_iou = [np.zeros((g, g), np.float32) for g in GRIDS]
    for n in range(N):
        x0, y0, x1, y1 = (float(v) for v in boxes[n])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        cand = []                                            # (level, iy, ix, iou, inside)
        for l, (s, g) in enumerate(zip(STRIDES, GRIDS)):
            c = centres(s, g)
            jx = int(np.clip(np.floor(cx / s), 0, g - 1)); jy = int(np.clip(np.floor(cy / s), 0, g - 1))
            xs = np.arange(max(0, jx - 2), min(g, jx + 3)); ys = np.arange(max(0, jy - 2), min(g, jy + 3))
            Y, X = np.meshgrid(ys, xs, indexing="ij"); Y, X = Y.ravel(), X.ravel()
            d = (c[X] - cx) ** 2 + (c[Y] - cy) ** 2
            o = np.argsort(d, kind="stable")[:ATSS_TOPK]; Y, X = Y[o], X[o]
            half = ANCHOR_SCALE * s / 2.0
            ax0, ay0, ax1, ay1 = c[X] - half, c[Y] - half, c[X] + half, c[Y] + half
            iw = np.clip(np.minimum(ax1, x1) - np.maximum(ax0, x0), 0, None); ih = np.clip(np.minimum(ay1, y1) - np.maximum(ay0, y0), 0, None)
            inter = iw * ih; iou = inter / ((ax1 - ax0) * (ay1 - ay0) + (x1 - x0) * (y1 - y0) - inter)
            inside = (c[X] > x0) & (c[X] < x1) & (c[Y] > y0) & (c[Y] < y1)
            cand += [(l, int(Y[k]), int(X[k]), float(iou[k]), bool(inside[k])) for k in range(len(Y))]
        ious = np.array([c[3] for c in cand])
        thr = ious.mean() + ious.std()
        for l, iy, ix, iou, inside in cand:
            if iou >= thr and inside and iou > best_iou[l][iy, ix]:
                best_iou[l][iy, ix] = iou; idx_maps[l][iy, ix] = n
    return idx_maps


# ------------------------------------------------------------------ model
def _gn(c):
    return nn.GroupNorm(32 if c % 32 == 0 else 8, c)


class LayerNorm2d(nn.Module):
    """Channel LayerNorm at every location (ConvNeXt's channels_first LN), learnable affine."""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(c, eps=eps)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


STATS = f"{CACHE}/chan_stats_train200.npz"      # per-channel mean/std, 200 train tiles (every 4th), spatial stride ~8


class ChanStd(nn.Module):
    """Frozen per-channel standardisation with training-set statistics (affine-free; as MAE's linear-probe BN)."""

    def __init__(self, mu, sd):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu).view(1, -1, 1, 1)); self.register_buffer("sd", torch.as_tensor(sd).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mu) / self.sd


class CNXHead(nn.Module):
    def __init__(self, width=WIDTH, tower=3, stage_norm="chanstd"):
        super().__init__()
        # amendment 3a (2026-11-02): normalise every stage before the lateral 1x1 -- the raw stage outputs carry outlier
        # channels (s3/s4: top-5 channels ~50% of variance, max|x| ~300). LayerNorm alone left s4's channel-std ratio at
        # 31x, so frozen per-channel standardisation is used. stage_norm=None = the stopped cnxl_s0_noLN run.
        if stage_norm == "chanstd":
            st = np.load(STATS)
            self.norm = nn.ModuleList([ChanStd(st[f"mu{k}"], st[f"sd{k}"]) for k in range(4)])
        elif stage_norm == "ln":
            self.norm = nn.ModuleList([LayerNorm2d(c) for c in CH])
        else:
            self.norm = nn.ModuleList([nn.Identity() for _ in CH])
        self.lat = nn.ModuleList([nn.Sequential(nn.Conv2d(c, width, 1), _gn(width), nn.ReLU(inplace=True)) for c in CH])
        t = []
        for _ in range(tower):
            t += [nn.Conv2d(width, width, 3, padding=1), _gn(width), nn.ReLU(inplace=True)]
        self.tower = nn.Sequential(*t)
        self.cls = nn.Conv2d(width, 1, 1); self.box = nn.Conv2d(width, 4, 1)
        self.scales = nn.Parameter(torch.ones(len(CH)))
        nn.init.normal_(self.cls.weight, std=0.01); nn.init.constant_(self.cls.bias, -math.log(99))
        nn.init.normal_(self.box.weight, std=0.01); nn.init.constant_(self.box.bias, 1.0)

    def forward(self, feats):
        outs = []
        for l, f in enumerate(feats):
            x = self.tower(self.lat[l](self.norm[l](f)))
            outs.append(torch.cat([self.cls(x), self.box(x) * self.scales[l]], 1))
        return outs


def _decode_boxes(raw4, ys, xs, l):
    s = STRIDES[l]; c = torch.as_tensor(centres(s, GRIDS[l]), device=raw4.device)
    d = raw4.clamp(max=8).exp() * s                          # (4, P)
    return torch.stack([c[xs] - d[0], c[ys] - d[1], c[xs] + d[2], c[ys] + d[3]], 1)


def gfl_loss(outs, tgt):
    """tgt: idx{l} (B,g,g) int64 (-1 neg), ignore{l} (B,g,g) bool, gtb list of (N,4) float tensors per image."""
    xs_all, ys_all, npos = [], [], 0
    box_num, box_den = outs[0].sum() * 0.0, outs[0].sum() * 0.0
    for l, out in enumerate(outs):
        idx, ign = tgt[f"idx{l}"], tgt[f"ignore{l}"]
        pos = idx >= 0
        y = torch.zeros_like(out[:, 0])
        b, ys, xs = pos.nonzero(as_tuple=True)
        if len(b):
            pred = _decode_boxes(out[:, 1:5][b, :, ys, xs].T, ys, xs, l)
            gt = torch.stack([tgt["gtb"][int(bi)][int(gi)] for bi, gi in zip(b.tolist(), idx[b, ys, xs].tolist())])
            q = box_iou(pred.detach(), gt).diagonal().clamp(min=0)
            y[b, ys, xs] = q
            box_num = box_num + (generalized_box_iou_loss(pred, gt, reduction="none") * q).sum(); box_den = box_den + q.sum()
            npos += len(b)
        valid = pos | ~ign
        xs_all.append(out[:, 0][valid]); ys_all.append(y[valid])
    x, y = torch.cat(xs_all), torch.cat(ys_all)
    p = torch.sigmoid(x)
    qfl = (F.binary_cross_entropy_with_logits(x, y, reduction="none") * (p - y).abs().pow(2.0)).sum() / max(npos, 1)
    return qfl, box_num / box_den.clamp(min=1e-6)


@torch.no_grad()
def decode(outs, nms_iou=0.5):
    allb, alls = [], []
    for l, out in enumerate(outs):
        o = out[0].float().cpu(); s = torch.sigmoid(o[0]).flatten()
        keep = (s > PRE_THRESH).nonzero(as_tuple=True)[0]
        if keep.numel() == 0:
            continue
        sc = s[keep]
        if sc.numel() > PRE_TOPK:
            sc, k = sc.topk(PRE_TOPK); keep = keep[k]
        g = GRIDS[l]; ys, xs = keep // g, keep % g
        allb.append(_decode_boxes(o[1:5, ys, xs], ys, xs, l).clamp(0, 2048)); alls.append(sc)
    if not allb:
        return torch.zeros((0, 4)), torch.zeros((0,))
    bx, sc = torch.cat(allb), torch.cat(alls)
    k = nms(bx, sc, nms_iou)[:MAX_DET]
    return bx[k], sc[k]


def load_feats(t):
    return [torch.from_numpy(np.load(f"{CACHE}/s{k+1}/{t}.npy")) for k in range(4)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["extract"]); ap.add_argument("--split", required=True)
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args(); cmd_extract(a.split, a.reverse)
