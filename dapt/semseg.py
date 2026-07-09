"""Semantic-segmentation readout: canopy area-F1 from the SAME frozen features.

Per-pixel canopy probe on the cached backbone features, scored in Restor-style
AREA F1 (pixel F1 of the canopy class, aggregated over all test pixels).

GT caveat (state in any write-up): the dryland cohort has NO mask annotations, so
the pixel GT here is BOX-FILL occupancy (GT boxes rasterized). Identical pseudo-GT
across arms ⇒ fair for RANKING backbones; absolute F1 is not true canopy F1.

Capacities mirror the detection ladder on the 32x32 feature grid:
  L1 'linear': 1x1 conv -> 1 logit/cell -> FIXED bilinear x16 -> pixel BCE
  L2 'mlp'   : 1-2 hidden 1x1-conv layers, same fixed upsample
Threshold chosen on val (max F1), reported once on test. Padding (tiles are
400/500 px in the 512 canvas) is excluded from loss and metrics.

Usage:
    .venv/bin/python -m dapt.semseg --arms web sat dapt_s42_i4999 --capacity linear
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dapt.data.cohort import REPO
from dapt.dataset import CohortData
from dapt.head import ProbeHead  # reuse the same probe bodies (N_OUT ignored here)
from dapt.train import set_seed

GRID, STRIDE = 32, 16


def _tile_hw(path):
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    with Image.open(p) as im:
        return im.size[1], im.size[0]           # (H, W)


def boxfill_mask(boxes, H, W, size=GRID * STRIDE):
    """(N,4) xyxy px -> (size,size) float {0,1} canopy mask + (size,size) valid."""
    m = np.zeros((size, size), np.float32)
    for x0, y0, x1, y1 in boxes:
        m[max(0, int(y0)):min(size, int(np.ceil(y1))),
          max(0, int(x0)):min(size, int(np.ceil(x1)))] = 1.0
    valid = np.zeros((size, size), np.float32)
    valid[:H, :W] = 1.0
    return m, valid


class SemsegProbe(torch.nn.Module):
    """Same bodies as the detection ladder, 1 output channel, fixed x16 upsample."""

    def __init__(self, in_dim, capacity="linear", hidden=256, n_hidden=1):
        super().__init__()
        if capacity == "linear":
            self.net = torch.nn.Conv2d(in_dim, 1, 1)
        elif capacity == "mlp":
            layers, c = [], in_dim
            for _ in range(n_hidden):
                layers += [torch.nn.Conv2d(c, hidden, 1), torch.nn.ReLU(inplace=True)]
                c = hidden
            layers += [torch.nn.Conv2d(c, 1, 1)]
            self.net = torch.nn.Sequential(*layers)
        else:
            raise ValueError(capacity)

    def forward(self, feat):                                   # (B,C,32,32)
        logit = self.net(feat)
        return F.interpolate(logit, scale_factor=STRIDE, mode="bilinear",
                             align_corners=False)              # (B,1,512,512)


def _prep(data, paths):
    masks, valids = [], []
    for p in paths:
        H, W = _tile_hw(p)
        m, v = boxfill_mask(data.tiles[p]["boxes"], H, W)
        masks.append(torch.from_numpy(m))
        valids.append(torch.from_numpy(v))
    return torch.stack(masks), torch.stack(valids)


@torch.no_grad()
def _probs(head, data, paths, device, bs=8):
    out = []
    head.eval()
    for i in range(0, len(paths), bs):
        b = data.batch(paths[i:i + bs])
        out.append(torch.sigmoid(head(b["feat"].to(device)))[:, 0].cpu())
    return torch.cat(out)                                      # (N,512,512)


def area_f1(probs, masks, valids, thr):
    pred = (probs >= thr).float() * valids
    gt = masks * valids
    tp = (pred * gt).sum().item()
    fp = (pred * (1 - gt) * valids).sum().item()
    fn = ((1 - pred) * gt * valids).sum().item()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"area_f1": f1, "precision": prec, "recall": rec, "iou": iou}


def train_semseg(arm, capacity="linear", seed=0, epochs=40, lr=1e-3, wd=1e-4, bs=8,
                 device=None):
    from dapt.backbone import pick_device
    set_seed(seed)
    device = pick_device(device)
    data = CohortData(arm)
    tr, va, te = (data.partition(p) for p in ("train", "val", "test"))
    in_dim = data.batch([tr[0]])["feat"].shape[1]
    head = SemsegProbe(in_dim, capacity).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr_masks, tr_valids = _prep(data, tr)

    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        head.train()
        order = rng.permutation(len(tr))
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            b = data.batch([tr[j] for j in idx])
            logit = head(b["feat"].to(device))[:, 0]
            m = tr_masks[idx].to(device)
            v = tr_valids[idx].to(device)
            loss = (F.binary_cross_entropy_with_logits(logit, m, reduction="none")
                    * v).sum() / v.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    # threshold on val, report once on test
    va_masks, va_valids = _prep(data, va)
    te_masks, te_valids = _prep(data, te)
    va_probs = _probs(head, data, va, device)
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.1, 0.9, 17):
        f1 = area_f1(va_probs, va_masks, va_valids, thr)["area_f1"]
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    te_probs = _probs(head, data, te, device)
    rep = area_f1(te_probs, te_masks, te_valids, best_thr)
    rep.update({"arm": arm, "capacity": capacity, "seed": seed, "thr": best_thr,
                "val_area_f1": best_f1})
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--capacity", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default="dapt/artifacts/semseg/semseg_summary.json")
    args = ap.parse_args()

    summary = {}
    for arm in args.arms:
        f1s, reps = [], []
        for s in args.seeds:
            r = train_semseg(arm, args.capacity, s, args.epochs)
            f1s.append(r["area_f1"])
            reps.append(r)
            print(f"[{arm} {args.capacity} s{s}] test areaF1={r['area_f1']:.3f} "
                  f"P={r['precision']:.3f} R={r['recall']:.3f} IoU={r['iou']:.3f}",
                  flush=True)
        summary[arm] = {"capacity": args.capacity,
                        "area_f1_mean": float(np.mean(f1s)),
                        "area_f1_std": float(np.std(f1s)), "runs": reps}
        print(f"==> {arm} {args.capacity}: test areaF1 "
              f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}\n", flush=True)
    out_path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(summary, open(out_path, "w"), indent=2)
    print(f"wrote {os.path.relpath(out_path, REPO)}")


if __name__ == "__main__":
    main()
