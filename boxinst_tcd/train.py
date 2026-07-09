"""Train the boxinst head from scratch on the TCD individual-tree sample.

BOX-SUPERVISED ONLY: reads boxes.json (never gt_polys.json). Reuses boxinst.model
+ boxinst.losses unchanged (frozen DINOv3 features cached by boxinst_tcd.cache).
Configs A/B/C via the same flags as the dryland ablation:
  A  (default)                          proj + pairwise
  B  --commonality_channel              + LDA treeness channel in mask neck
  C  --commonality_channel --proto_loss + cross-box prototype-consistency loss

Selection: best val detection mAP50 among post-warmup epochs (masks have no boxes
to select on either; consistent with the dryland protocol). Seeded, recorded.

Usage:
    .venv/bin/python -m boxinst_tcd.train --tag A --epochs 300
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from dapt.backbone import pick_device
from dapt.decode import decode
from dapt.targets import encode
from boxinst_tcd import det_eval          # canopy-aware detection metrics
from boxinst.losses import det_loss, mask_losses, prototype_loss
from boxinst.model import BoxInstHead
from boxinst_tcd.cache import COMM, FEAT, PAIR, key
from boxinst_tcd.prepare import OUT

ART = os.path.join(OUT, "artifacts")
W_HM, W_OFF, W_SIZE, W_GIOU, W_PROJ, W_PAIR = 1.0, 1.0, 0.1, 1.0, 1.0, 1.0


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class Data:
    def __init__(self, feat_dir=FEAT, pair_dir=PAIR, comm_dir=COMM):
        from boxinst_tcd.cache import canopy_cell_mask
        from boxinst_tcd.build_canopy import load_canopy_mask
        self.split = json.load(open(os.path.join(OUT, "split.json")))
        boxes = json.load(open(os.path.join(OUT, "boxes.json")))    # TRAIN input
        self.tiles = {}
        for p, info in self.split["tiles"].items():
            bx = np.array(boxes[p], np.float32).reshape(-1, 4)
            enc = encode(bx)
            comm = np.load(os.path.join(comm_dir, key(p) + ".npz"))
            ignore = canopy_cell_mask(load_canopy_mask(p))   # complete: polygon+RLE
            self.tiles[p] = {
                **info, "boxes": bx,
                "feat": torch.from_numpy(np.load(os.path.join(feat_dir, key(p) + ".npy"))),
                "sims": torch.from_numpy(np.load(os.path.join(pair_dir, key(p) + ".npy"))),
                "lda": torch.from_numpy(comm["lda"]),
                "z": torch.from_numpy(comm["z"]),
                "ignore": torch.from_numpy(ignore),
                "enc": {k: torch.from_numpy(enc[k]) for k in
                        ("heatmap", "offset", "size")} |
                       {"reg_mask": torch.from_numpy(enc["reg_mask"])}}

    def partition(self, name):
        return sorted(p for p, t in self.tiles.items() if t["partition"] == name)

    def batch(self, paths, device):
        t = [self.tiles[p] for p in paths]
        st = lambda k: torch.stack([x[k] for x in t]).float().to(device)
        return {"feat": st("feat"), "lda": st("lda"), "z": st("z"), "sims": st("sims"),
                "heatmap": torch.stack([x["enc"]["heatmap"] for x in t]).to(device),
                "offset": torch.stack([x["enc"]["offset"] for x in t]).to(device),
                "size": torch.stack([x["enc"]["size"] for x in t]).to(device),
                "reg_mask": torch.stack([x["enc"]["reg_mask"] for x in t]).to(device),
                "ignore": torch.stack([x["ignore"] for x in t]).to(device),
                "boxes": [x["boxes"] for x in t]}


@torch.no_grad()
def infer(model, data, paths, device, comm_ch, score_thr=0.05):
    model.eval()
    preds, gts = [], []
    for p in paths:
        b = data.batch([p], device)
        det, _, _ = model(b["feat"], b["lda"] if comm_ch else None)
        bx, sc = decode(det.cpu(), score_thr=score_thr)
        preds.append((bx.numpy(), sc.numpy()))
        gts.append(data.tiles[p]["boxes"])
    return preds, gts


def train(args):
    set_seed(args.seed)
    device = pick_device(args.device)
    feat_dir = args.feat_dir or FEAT
    pair_dir = args.pair_dir or PAIR
    comm_dir = args.comm_dir or COMM
    data = Data(feat_dir, pair_dir, comm_dir)
    tr, va, te = data.partition("train"), data.partition("val"), data.partition("test")
    if args.n_train and args.n_train < len(tr):
        rng0 = np.random.default_rng(args.seed)
        tr = list(rng0.permutation(tr)[:args.n_train])
    in_dim = data.tiles[tr[0]]["feat"].shape[0]
    comm_ch = 1 if args.commonality_channel else 0
    model = BoxInstHead(in_dim, commonality_ch=comm_ch, det_tower=args.det_tower,
                        use_sig=args.signature).to(device)
    D = data.tiles[tr[0]]["z"].shape[0]
    c_ema = F.normalize(torch.randn(D, device=device), dim=0)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    dil_idx = [int(x) for x in args.pair_dils.split(",")]
    iters_per_ep = (len(tr) + args.bs - 1) // args.bs
    val_canopies = det_eval.load_canopies(va)    # canopy-aware val selection
    print(f"[{args.tag}] device={device} in_dim={in_dim} det_tower={args.det_tower} "
          f"comm_ch={comm_ch} proto={args.proto_loss} feat={os.path.basename(feat_dir)} "
          f"train/val/test={len(tr)}/{len(va)}/{len(te)} seed={args.seed}")

    rng = np.random.default_rng(args.seed)
    best = {"mAP50": -1, "state": None, "epoch": -1, "thr": 0.2}
    it = 0
    for ep in range(args.epochs):
        model.train()
        order = list(tr); rng.shuffle(order)
        parts = []
        for i in range(0, len(order), args.bs):
            b = data.batch(order[i:i + args.bs], device)
            det, ctrl, fmask = model(b["feat"], b["lda"] if comm_ch else None)
            l_hm, l_off, l_size, l_giou = det_loss(det, b)
            l_proj, l_pair, n_inst, ex = mask_losses(
                ctrl, fmask, b["sims"], b["boxes"], args.tau, dil_idx,
                z_batch=b["z"] if args.signature else None)
            w_pair = W_PAIR * min(1.0, it / max(args.warmup_iters, 1))
            l_proto = det.new_zeros(())
            if args.proto_loss and ex is not None:
                w_proto = args.proto_loss * min(1.0, it / max(args.warmup_iters, 1))
                l_proto, zb = prototype_loss(ex["logits"], ex["boxes_px"],
                                             ex["img_idx"], b["z"], c_ema)
                c_ema = F.normalize(0.99 * c_ema + 0.01 * zb.mean(0), dim=0)
            else:
                w_proto = 0.0
            loss = (W_HM * l_hm + W_OFF * l_off + W_SIZE * l_size + W_GIOU * l_giou +
                    W_PROJ * l_proj + w_pair * l_pair + w_proto * l_proto)
            opt.zero_grad(); loss.backward(); opt.step()
            it += 1
            parts.append([x.item() for x in (l_hm, l_proj, l_pair, l_proto)])
        sched.step()
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            m = np.mean(parts, axis=0)
            preds, gts = infer(model, data, va, device, comm_ch)
            thr, _ = det_eval.pick_threshold(preds, gts, val_canopies)
            rep = det_eval.full_report(preds, gts, val_canopies, thr)
            elig = (ep + 1) * iters_per_ep >= args.warmup_iters
            print(f"  ep{ep+1:3d} hm={m[0]:.3f} proj={m[1]:.3f} pair={m[2]:.3f} "
                  f"proto={m[3]:.3f} | val boxAP50={rep['mAP50']:.3f} "
                  f"F1={rep['f1']:.3f}@{thr:.2f}{'' if elig else ' (pre-warmup)'}")
            if elig and rep["mAP50"] > best["mAP50"]:
                best = {"mAP50": rep["mAP50"], "epoch": ep + 1, "thr": thr,
                        "state": {k: v.cpu().clone() for k, v in
                                  model.state_dict().items()}}
    if best["state"] is None:
        best["state"] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    os.makedirs(ART, exist_ok=True)
    cfg = {"tag": args.tag, "seed": args.seed, "in_dim": in_dim,
           "commonality_channel": bool(comm_ch), "proto_loss": args.proto_loss,
           "tau": args.tau, "pair_dils": dil_idx, "best_epoch": best["epoch"],
           "score_thr": best["thr"], "nms_iou": 0.5, "mask_thr": 0.5,
           "det_tower": args.det_tower, "n_train": len(tr),
           "use_sig": bool(args.signature),
           "feat_dir": feat_dir, "pair_dir": pair_dir, "comm_dir": comm_dir}
    torch.save({"state": best["state"], "cfg": cfg},
               os.path.join(ART, f"tcd_{args.tag}.pt"))
    # report val box detection (mask metrics come from eval_masks)
    preds, gts = infer(model_load(model, best), data, va, device, comm_ch)
    rep = det_eval.full_report(preds, gts, val_canopies, best["thr"])
    print(f"[{args.tag}] best ep{best['epoch']} val boxAP50={rep['mAP50']:.3f} "
          f"F1={rep['f1']:.3f} thr={best['thr']:.2f} -> tcd_{args.tag}.pt")


def model_load(model, best):
    model.load_state_dict(best["state"]); return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--commonality_channel", action="store_true")
    ap.add_argument("--proto_loss", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.975)
    ap.add_argument("--pair_dils", default="4,5,6,7")
    ap.add_argument("--warmup_iters", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--det_tower", type=int, default=2,
                    help="3x3 conv layers before detection heads (0=lean probe)")
    ap.add_argument("--n_train", type=int, default=0, help="subsample train (0=all)")
    ap.add_argument("--signature", action="store_true",
                    help="add per-instance cosine-to-box-signature channel to mask")
    ap.add_argument("--feat_dir", default=None)
    ap.add_argument("--pair_dir", default=None)
    ap.add_argument("--comm_dir", default=None)
    ap.add_argument("--device", default=None)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
