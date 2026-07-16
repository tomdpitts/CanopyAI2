"""Train ONLY the semantic tree-cover head on frozen det_t8 features.

Warm-starts MultiHead8 from det_t8 and freezes stem/tower/up/hm/reg, so the det
output stays byte-identical to det_t8 -> ITC mask mAP50 is preserved *exactly* (0
risk), and only the 1-conv sem head learns. Target = (ITC-box ∪ canopy) rasterised
at the 256 grid; BCE over all pixels (canopy is now POSITIVE, not ignore). The sem
operating threshold is picked on the held-out val train-tiles by tree-cover F1.

Reuses the main package's feat_traintile cache + train_tiles_gt.json. Fast (one
conv on frozen feats). Delete tcd04_semseg/ to discard.

Usage:
    .venv/bin/python -m tcd04_semseg.train --tag sem --epochs 30
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from dapt.backbone import pick_device
from boxinst_commonality_tcd_04.cache_train_tiles import cache_dir
from tcd04_semseg.model import MultiHead8

MAIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "boxinst_commonality_tcd_04")
HERE = os.path.abspath(os.path.dirname(__file__))
SEM_GRID = 256                         # 2048 / 8


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def treecover_target(boxes, canopy, g=SEM_GRID, tile=2048):
    """(ITC boxes ∪ canopy polys) -> (g,g) float {0,1} at 8px stride."""
    s = tile / g
    im = Image.new("L", (g, g), 0)
    d = ImageDraw.Draw(im)
    for x0, y0, x1, y1 in boxes:
        d.rectangle([x0 / s, y0 / s, x1 / s, y1 / s], fill=1)
    for poly in canopy:
        if poly and len(poly) >= 6:
            d.polygon([tuple(v) for v in (np.asarray(poly).reshape(-1, 2) / s)],
                      fill=1)
    return np.asarray(im, np.float32)


class Data:
    def __init__(self, arm):
        self.cdir = cache_dir(arm)
        gt = json.load(open(os.path.join(MAIN, "train_tiles_gt.json")))
        self.gt = {t: v for t, v in gt.items()
                   if os.path.exists(os.path.join(self.cdir, t + ".npy"))}
        self.tgt = {}
        for t, v in self.gt.items():
            bx = np.array(v["boxes"], np.float32).reshape(-1, 4)
            self.tgt[t] = torch.from_numpy(treecover_target(bx, v["canopy"]))

    def partition(self, name):
        return sorted(t for t, v in self.gt.items() if v["partition"] == name)

    def _feat(self, t):
        return torch.from_numpy(np.load(os.path.join(self.cdir, t + ".npy")))

    def batch(self, tids, device):
        f = torch.stack([self._feat(t) for t in tids]).float().to(device)
        y = torch.stack([self.tgt[t] for t in tids]).to(device)
        return f, y


@torch.no_grad()
def pick_sem_thr(model, data, val, device):
    """Threshold maximising tree-cover F1 on val train-tiles."""
    tp = np.zeros(19); fp = np.zeros(19); fn = np.zeros(19)
    thrs = np.linspace(0.05, 0.95, 19)
    model.eval()
    for t in val:
        f, y = data.batch([t], device)
        _, sem = model(f)
        p = torch.sigmoid(sem)[0, 0].cpu().numpy()
        g = y[0].cpu().numpy().astype(bool)
        for i, th in enumerate(thrs):
            pm = p >= th
            tp[i] += (pm & g).sum(); fp[i] += (pm & ~g).sum(); fn[i] += (~pm & g).sum()
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return float(thrs[f1.argmax()]), float(f1.max())


def train(args):
    set_seed(args.seed)
    device = pick_device(args.device)
    data = Data(args.arm)
    tr, va = data.partition("train"), data.partition("val")
    in_dim = data._feat(tr[0]).shape[0]
    ck = torch.load(os.path.join(MAIN, "artifacts", "det_t8.pt"),
                    map_location="cpu", weights_only=False)
    dcfg = ck["cfg"]
    model = MultiHead8(in_dim, width=dcfg["width"], tower=dcfg["tower"]).to(device)
    model.load_det_from(ck["state"])
    # freeze everything except the sem head -> det output == det_t8 (mAP50 unchanged)
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("sem.")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=args.lr, weight_decay=args.wd)
    print(f"[{args.tag}] device={device} train/val={len(tr)}/{len(va)} "
          f"sem-head-only (trunk+det frozen from det_t8) seed={args.seed}",
          flush=True)

    rng = np.random.default_rng(args.seed)
    for ep in range(args.epochs):
        model.train()
        order = list(tr); rng.shuffle(order)
        losses = []
        for i in range(0, len(order), args.bs):
            f, y = data.batch(order[i:i + args.bs], device)
            _, sem = model(f)
            loss = F.binary_cross_entropy_with_logits(sem[:, 0], y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            thr, f1 = pick_sem_thr(model, data, va, device)
            print(f"  ep{ep+1:3d} bce={np.mean(losses):.4f} | val treecover "
                  f"F1={f1:.3f}@{thr:.2f}", flush=True)
    thr, f1 = pick_sem_thr(model, data, va, device)
    os.makedirs(os.path.join(HERE, "artifacts"), exist_ok=True)
    cfg = {"tag": args.tag, "seed": args.seed, "in_dim": in_dim,
           "width": dcfg["width"], "tower": dcfg["tower"],
           "det_score_thr": dcfg["score_thr"], "sem_thr": round(thr, 3),
           "val_treecover_F1": round(f1, 4), "warm_from": "det_t8"}
    torch.save({"state": model.state_dict(), "cfg": cfg},
               os.path.join(HERE, "artifacts", f"mh_{args.tag}.pt"))
    print(f"[{args.tag}] val treecover F1={f1:.3f} sem_thr={thr:.2f} "
          f"-> mh_{args.tag}.pt", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sem")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=3)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--arm", default="web")
    ap.add_argument("--device", default=None)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
