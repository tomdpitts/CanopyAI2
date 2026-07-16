"""Train the 8px CenterNet detector on TCD individual-tree boxes (frozen DINO).

Reuses the boxinst_tcd 720-crop ITC split + web features (boxes only; canopy is an
ignore region). Targets and decoding are at 8px (dapt.targets with CFG8); val
selection is canopy-aware box mAP50 (boxinst_tcd.det_eval). Masks are NOT trained
here — they come from the training-free EM (em.py).

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.train_detector --tag d8 --epochs 200
"""
import argparse
import json
import os
import random

import numpy as np
import torch

from dapt.backbone import pick_device
from dapt.decode import decode
from dapt.targets import encode
from boxinst_tcd import det_eval
from boxinst_tcd.build_canopy import load_canopy_mask
from boxinst_tcd.cache import FEAT, key
from boxinst_tcd.prepare import OUT as TCD_OUT
from boxinst_commonality_tcd_04.detector import (CFG8, STRIDE8, Detector8,
                                                 canopy_cell_mask, det_loss)

OUT = os.path.abspath(os.path.dirname(__file__))
ART = os.path.join(OUT, "artifacts")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class Data:
    def __init__(self, feat_dir):
        self.split = json.load(open(os.path.join(TCD_OUT, "split.json")))
        boxes = json.load(open(os.path.join(TCD_OUT, "boxes.json")))
        self.tiles = {}
        for p, info in self.split["tiles"].items():
            bx = np.array(boxes[p], np.float32).reshape(-1, 4)
            enc = encode(bx, CFG8)                        # 8px targets
            g = CFG8.grid
            ignore = canopy_cell_mask(load_canopy_mask(p), g)
            self.tiles[p] = {
                **info, "boxes": bx,
                "feat": torch.from_numpy(np.load(os.path.join(feat_dir, key(p) + ".npy"))),
                "enc": {k: torch.from_numpy(enc[k]) for k in
                        ("heatmap", "offset", "size")} |
                       {"reg_mask": torch.from_numpy(enc["reg_mask"]),
                        "ignore": torch.from_numpy(ignore)}}

    def partition(self, name):
        return sorted(p for p, t in self.tiles.items() if t["partition"] == name)

    def batch(self, paths, device):
        t = [self.tiles[p] for p in paths]
        st = lambda k: torch.stack([x["enc"][k] for x in t]).to(device)
        return {"feat": torch.stack([x["feat"] for x in t]).float().to(device),
                "heatmap": st("heatmap"), "offset": st("offset"), "size": st("size"),
                "reg_mask": st("reg_mask"), "ignore": st("ignore"),
                "boxes": [x["boxes"] for x in t]}


@torch.no_grad()
def infer(model, data, paths, device, score_thr=0.05):
    model.eval()
    preds, gts = [], []
    for p in paths:
        b = data.batch([p], device)
        det = model(b["feat"])
        bx, sc = decode(det.cpu(), score_thr=score_thr, stride=STRIDE8)
        preds.append((bx.numpy(), sc.numpy()))
        gts.append(data.tiles[p]["boxes"])
    return preds, gts


def train(args):
    set_seed(args.seed)
    device = pick_device(args.device)
    data = Data(args.feat_dir)
    tr, va = data.partition("train"), data.partition("val")
    if args.n_train and args.n_train < len(tr):
        tr = list(np.random.default_rng(args.seed).permutation(tr)[:args.n_train])
    in_dim = data.tiles[tr[0]]["feat"].shape[0]
    model = Detector8(in_dim, width=args.width, tower=args.tower).to(device)
    npar = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    val_canopies = det_eval.load_canopies(va)
    print(f"[{args.tag}] device={device} in_dim={in_dim} params={npar:.2f}M "
          f"8px-grid tower={args.tower} train/val={len(tr)}/{len(va)} seed={args.seed}",
          flush=True)

    rng = np.random.default_rng(args.seed)
    best = {"mAP50": -1, "state": None, "epoch": -1, "thr": 0.2}
    for ep in range(args.epochs):
        model.train()
        order = list(tr); rng.shuffle(order)
        losses = []
        for i in range(0, len(order), args.bs):
            b = data.batch(order[i:i + args.bs], device)
            det = model(b["feat"])
            l_hm, l_off, l_size, l_giou = det_loss(det, b)
            loss = l_hm + l_off + l_size + l_giou
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append([l_hm.item(), l_off.item(), l_size.item(), l_giou.item()])
        sched.step()
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            m = np.mean(losses, 0)
            preds, gts = infer(model, data, va, device)
            thr, _ = det_eval.pick_threshold(preds, gts, val_canopies)
            rep = det_eval.full_report(preds, gts, val_canopies, thr)
            print(f"  ep{ep+1:3d} hm={m[0]:.3f} off={m[1]:.3f} size={m[2]:.3f} "
                  f"giou={m[3]:.3f} | val boxAP50={rep['mAP50']:.3f} "
                  f"F1={rep['f1']:.3f}@{thr:.2f}", flush=True)
            if rep["mAP50"] > best["mAP50"]:
                best = {"mAP50": rep["mAP50"], "epoch": ep + 1, "thr": thr,
                        "state": {k: v.cpu().clone() for k, v in
                                  model.state_dict().items()}}
    if best["state"] is None:
        best["state"] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    os.makedirs(ART, exist_ok=True)
    cfg = {"tag": args.tag, "seed": args.seed, "in_dim": in_dim, "grid_stride": STRIDE8,
           "width": args.width, "tower": args.tower, "best_epoch": best["epoch"],
           "score_thr": best["thr"], "nms_iou": 0.5, "n_train": len(tr),
           "feat_dir": args.feat_dir, "val_boxAP50": round(best["mAP50"], 4)}
    torch.save({"state": best["state"], "cfg": cfg},
               os.path.join(ART, f"det_{args.tag}.pt"))
    print(f"[{args.tag}] best ep{best['epoch']} val boxAP50={best['mAP50']:.3f} "
          f"thr={best['thr']:.2f} -> det_{args.tag}.pt", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--tower", type=int, default=3)
    ap.add_argument("--n_train", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--feat_dir", default=FEAT)
    ap.add_argument("--device", default=None)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
