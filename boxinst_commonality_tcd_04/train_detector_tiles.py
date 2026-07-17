"""Train Detector8 on full 2048 TRAIN tiles (128x128 feats -> 256x256 8px grid).

No windowing, no crop filters, and the SAME grid size as the 439 test — train and
test distributions are now identical. Reads the stitched tile features cached by
cache_train_tiles.py per batch (they don't fit in RAM). All ITC boxes per tile are
targets; canopy is ignored. Backbone frozen; only the decoder trains.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.train_detector_tiles --tag t8 --epochs 120
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from PIL import Image, ImageDraw

from dapt.backbone import pick_device
from dapt.decode import decode
from dapt.eval import full_report, pick_threshold
from dapt.targets import TargetConfig, encode
from boxinst_commonality_tcd_04.cache_train_tiles import cache_dir
from boxinst_commonality_tcd_04.detector import (STRIDE8, Detector8,
                                                 canopy_cell_mask, det_loss)

OUT = os.path.abspath(os.path.dirname(__file__))
ART = os.path.join(OUT, "artifacts")
CFG_TILE = TargetConfig(grid=256, stride=8)     # 2048 tile at 8px (TCD default)
on_best_save = None                             # optional hook(ckpt_path) per new best


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def canopy_px(polys, res=2048):
    m = Image.new("L", (res, res), 0)
    d = ImageDraw.Draw(m)
    for poly in polys:
        if poly and len(poly) >= 6:
            d.polygon([tuple(v) for v in np.asarray(poly).reshape(-1, 2)], fill=1)
    return np.asarray(m, bool)


class TileData:
    def __init__(self, arm, canvas=2048, gt_path=None):
        """canvas = padded tile side (px); the 8px target grid is canvas//8, and
        the canopy raster is at `canvas`. Defaults reproduce TCD (2048 -> grid
        256). gt_path overrides train_tiles_gt.json for other datasets/folds."""
        self.cdir = cache_dir(arm)
        gt = json.load(open(gt_path or os.path.join(OUT, "train_tiles_gt.json")))
        self.gt = {t: v for t, v in gt.items()
                   if os.path.exists(os.path.join(self.cdir, t + ".npy"))}
        cfg = TargetConfig(grid=canvas // 8, stride=8)
        self.enc, self.ign, self.boxes = {}, {}, {}
        for t, v in self.gt.items():
            bx = np.array(v["boxes"], np.float32).reshape(-1, 4)
            e = encode(bx, cfg)
            self.enc[t] = {k: torch.from_numpy(e[k]) for k in
                           ("heatmap", "offset", "size", "reg_mask")}
            # canopy ignore at the target grid (empty list -> all-False, box-only)
            self.ign[t] = torch.from_numpy(
                canopy_cell_mask(canopy_px(v.get("canopy", []), res=canvas),
                                 cfg.grid))
            self.boxes[t] = bx

    def partition(self, name):
        return sorted(t for t, v in self.gt.items() if v["partition"] == name)

    def _feat(self, t):
        return torch.from_numpy(np.load(os.path.join(self.cdir, t + ".npy")))

    def batch(self, tids, device):
        f = torch.stack([self._feat(t) for t in tids]).float().to(device)
        st = lambda k: torch.stack([self.enc[t][k] for t in tids]).to(device)
        return {"feat": f, "heatmap": st("heatmap"), "offset": st("offset"),
                "size": st("size"), "reg_mask": st("reg_mask"),
                "ignore": torch.stack([self.ign[t] for t in tids]).to(device),
                "boxes": [self.boxes[t] for t in tids]}


@torch.no_grad()
def infer(model, data, tids, device, score_thr=0.05, topk=600):
    model.eval()
    preds, gts = [], []
    for t in tids:
        det = model(data._feat(t).float()[None].to(device))
        bx, sc = decode(det.cpu(), score_thr=score_thr, stride=STRIDE8, topk=topk)
        preds.append((bx.numpy(), sc.numpy())); gts.append(data.boxes[t])
    return preds, gts


def train(args):
    set_seed(args.seed)
    device = pick_device(args.device)
    data = TileData(args.arm, canvas=getattr(args, "canvas", 2048),
                    gt_path=getattr(args, "gt_path", None))
    tr, va = data.partition("train"), data.partition("val")
    in_dim = data._feat(tr[0]).shape[0]
    model = Detector8(in_dim, width=args.width, tower=args.tower).to(device)
    npar = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # LR schedule: ReduceLROnPlateau on val boxAP50 (opt-in) or cosine (default).
    # Plateau anneals only when val stalls, which cosine-with-early-stop never
    # reaches (the 0.504 run's flaw). All extras are getattr-guarded so the plain
    # CLI keeps its original behaviour.
    use_plateau = getattr(args, "plateau", False)
    if use_plateau:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=getattr(args, "plateau_factor", 0.3),
            patience=getattr(args, "plateau_patience", 2))
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    es = getattr(args, "early_stop", False)
    min_epochs = getattr(args, "min_epochs", 12)
    es_patience = getattr(args, "es_patience", 4)
    es_min_delta = getattr(args, "es_min_delta", 0.005)
    grid = getattr(args, "canvas", 2048) // 8
    print(f"[{args.tag}] device={device} in_dim={in_dim} params={npar:.2f}M "
          f"grid={grid} train/val={len(tr)}/{len(va)} seed={args.seed} "
          f"tower={args.tower} wd={args.wd} "
          f"{'plateau' if use_plateau else 'cosine'}"
          f"{' early-stop' if es else ''}", flush=True)

    rng = np.random.default_rng(args.seed)
    best = {"mAP50": -1, "state": None, "epoch": -1, "thr": 0.2}
    no_improve = 0
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
        if not use_plateau:
            sched.step()
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            m = np.mean(losses, 0)
            preds, gts = infer(model, data, va, device)
            thr, _ = pick_threshold(preds, gts)
            rep = full_report(preds, gts, thr)
            if use_plateau:
                sched.step(rep["mAP50"])
            lr_now = opt.param_groups[0]["lr"]
            print(f"  ep{ep+1:3d} hm={m[0]:.3f} off={m[1]:.3f} size={m[2]:.3f} "
                  f"giou={m[3]:.3f} | val boxAP50={rep['mAP50']:.3f} "
                  f"F1={rep['f1']:.3f}@{thr:.2f} lr={lr_now:.1e}", flush=True)
            improved = rep["mAP50"] > best["mAP50"] + es_min_delta
            if rep["mAP50"] > best["mAP50"]:
                best = {"mAP50": rep["mAP50"], "epoch": ep + 1, "thr": thr,
                        "state": {k: v.cpu().clone() for k, v in
                                  model.state_dict().items()}}
                # checkpoint incrementally so a mid-run stop keeps the best model
                os.makedirs(ART, exist_ok=True)
                cfg = {"tag": args.tag, "seed": args.seed, "in_dim": in_dim,
                       "grid_stride": STRIDE8, "width": args.width,
                       "tower": args.tower, "best_epoch": best["epoch"],
                       "score_thr": best["thr"], "nms_iou": 0.5, "n_train": len(tr),
                       "arm": args.arm, "val_boxAP50": round(best["mAP50"], 4),
                       "data": "full 2048 train tiles, no windowing"}
                ckpt_path = os.path.join(ART, f"det_{args.tag}.pt")
                torch.save({"state": best["state"], "cfg": cfg}, ckpt_path)
                print(f"    ^ saved best ep{best['epoch']} "
                      f"boxAP50={best['mAP50']:.3f}", flush=True)
                # optional durable-persist hook (Modal: copy to Volume + commit),
                # fired on EVERY new best so a mid-training crash keeps it
                if on_best_save is not None:
                    on_best_save(ckpt_path)
            no_improve = 0 if improved else no_improve + 1
            if es and ep + 1 >= min_epochs and no_improve >= es_patience:
                print(f"    early-stop: {no_improve} evals without >"
                      f"{es_min_delta} improvement (best ep{best['epoch']})",
                      flush=True)
                break
    if best["state"] is None:
        torch.save({"state": {k: v.cpu().clone() for k, v in
                              model.state_dict().items()},
                    "cfg": {"tag": args.tag, "in_dim": in_dim, "width": args.width,
                            "tower": args.tower, "score_thr": 0.2, "nms_iou": 0.5,
                            "arm": args.arm}},
                   os.path.join(ART, f"det_{args.tag}.pt"))
    print(f"[{args.tag}] best ep{best['epoch']} val boxAP50={best['mAP50']:.3f} "
          f"-> det_{args.tag}.pt", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--tower", type=int, default=3)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--arm", default="web")
    ap.add_argument("--device", default=None)
    # improved recipe (all optional; defaults reproduce the original behaviour)
    ap.add_argument("--plateau", action="store_true",
                    help="ReduceLROnPlateau on val boxAP50 instead of cosine")
    ap.add_argument("--plateau_factor", type=float, default=0.3)
    ap.add_argument("--plateau_patience", type=int, default=2)
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--min_epochs", type=int, default=12)
    ap.add_argument("--es_patience", type=int, default=4)
    ap.add_argument("--es_min_delta", type=float, default=0.005)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
