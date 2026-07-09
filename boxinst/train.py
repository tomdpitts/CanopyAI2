"""Train the box-supervised instance-seg head on cached frozen DINOv3 features.

Boxes-only supervision: focal centre heatmap + smooth-L1 offset/log-size + GIoU
(detection), BoxInst projection dice + DINO-affinity pairwise (masks). No mask
labels anywhere. Selection on val detection mAP50 (masks have no GT to select on);
the operating score threshold is frozen on val. Seeds numpy+torch from one --seed
and records it in the checkpoint + result JSON.

Usage:
    .venv/bin/python -m boxinst.train --epochs 300 --seed 0
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from dapt.backbone import pick_device
from dapt.cache_features import cache_key
from dapt.data.cohort import REPO, load_boxes
from dapt.decode import decode
from dapt.eval import full_report, pick_threshold
from dapt.targets import encode
from boxinst.cache_feats import FEAT_DIR, PAIR_DIR
from boxinst.losses import det_loss, mask_losses, prototype_loss
from boxinst.model import BoxInstHead

COMM_DIR = os.path.join(REPO, "dapt/cache/commonality_last4")

ART_DIR = os.path.join(REPO, "boxinst/artifacts")

# loss weights (untuned defaults; hm/off/size follow the frozen dapt protocol)
W_HM, W_OFF, W_SIZE, W_GIOU, W_PROJ, W_PAIR = 1.0, 1.0, 0.1, 1.0, 1.0, 1.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Data:
    """All cached tensors in RAM (fp16), stacked to fp32 per batch."""

    def __init__(self, split_path="dapt/data/split.json"):
        self.split = json.load(open(os.path.join(REPO, split_path)))
        boxes, _ = load_boxes(self.split["csv"])
        self.tiles = {}
        for p, info in self.split["tiles"].items():
            bx = boxes.get(p, np.zeros((0, 4), np.float32))
            enc = encode(bx)
            self.tiles[p] = {
                **info, "boxes": bx,
                "feat": torch.from_numpy(np.load(
                    os.path.join(FEAT_DIR, cache_key(p) + ".npy"))),
                "sims": torch.from_numpy(np.load(
                    os.path.join(PAIR_DIR, cache_key(p) + ".npy"))),
                "lda": torch.from_numpy(np.load(
                    os.path.join(COMM_DIR, cache_key(p) + ".npz"))["lda"]),
                "z": torch.from_numpy(np.load(
                    os.path.join(COMM_DIR, cache_key(p) + ".npz"))["z"]),
                "enc": {k: torch.from_numpy(enc[k]) for k in
                        ("heatmap", "offset", "size")} |
                       {"reg_mask": torch.from_numpy(enc["reg_mask"])},
            }

    def partition(self, name):
        return sorted(p for p, t in self.tiles.items() if t["partition"] == name)

    def batch(self, paths, device):
        t = [self.tiles[p] for p in paths]
        return {
            "feat": torch.stack([x["feat"] for x in t]).float().to(device),
            "lda": torch.stack([x["lda"] for x in t]).float().to(device),
            "z": torch.stack([x["z"] for x in t]).float().to(device),
            "sims": torch.stack([x["sims"] for x in t]).float().to(device),
            "heatmap": torch.stack([x["enc"]["heatmap"] for x in t]).to(device),
            "offset": torch.stack([x["enc"]["offset"] for x in t]).to(device),
            "size": torch.stack([x["enc"]["size"] for x in t]).to(device),
            "reg_mask": torch.stack([x["enc"]["reg_mask"] for x in t]).to(device),
            "boxes": [x["boxes"] for x in t],
        }


@torch.no_grad()
def infer_partition(model, data, paths, device, score_thr=0.05):
    model.eval()
    preds, gts = [], []
    for p in paths:
        b = data.batch([p], device)
        det, _, _ = model(b["feat"], b["lda"] if model.commonality_ch else None)
        boxes, scores = decode(det.cpu(), score_thr=score_thr)
        preds.append((boxes.numpy(), scores.numpy()))
        gts.append(data.tiles[p]["boxes"])
    return preds, gts


def train(args):
    set_seed(args.seed)
    device = pick_device(args.device)
    data = Data()
    train_paths = data.partition("train")
    val_paths = data.partition("val")
    test_paths = data.partition("test")
    in_dim = data.tiles[train_paths[0]]["feat"].shape[0]

    comm_ch = 1 if args.commonality_channel else 0
    model = BoxInstHead(in_dim, commonality_ch=comm_ch).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    D = data.tiles[train_paths[0]]["z"].shape[0]
    c_ema = F.normalize(torch.randn(D, device=device), dim=0)   # crown prototype
    print(f"device={device} in_dim={in_dim} head_params={n_par/1e6:.2f}M "
          f"train/val/test={len(train_paths)}/{len(val_paths)}/{len(test_paths)} "
          f"tau={args.tau} comm_ch={comm_ch} proto={args.proto_loss} "
          f"center={not args.no_center} warmup={args.warmup_iters}it seed={args.seed}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    dil_idx = [int(x) for x in args.pair_dils.split(",")]
    rng = np.random.default_rng(args.seed)
    # Selection rule (boxes-only, untuned on masks): a checkpoint is only eligible
    # once ALL loss terms are fully active, i.e. after the pairwise warmup —
    # otherwise val detection mAP picks an early epoch whose mask branch has
    # effectively not trained yet. The pre-warmup detection peak is still logged.
    iters_per_ep = (len(train_paths) + args.bs - 1) // args.bs
    best = {"mAP50": -1.0, "state": None, "epoch": -1}
    best_pre = {"mAP50": -1.0, "epoch": -1}
    curve = []
    it = 0
    for ep in range(args.epochs):
        model.train()
        order = list(train_paths)
        rng.shuffle(order)
        parts = []
        for i in range(0, len(order), args.bs):
            b = data.batch(order[i:i + args.bs], device)
            det, ctrl, fmask = model(b["feat"], b["lda"] if comm_ch else None)
            l_hm, l_off, l_size, l_giou = det_loss(det, b)
            l_proj, l_pair, n_inst, ex = mask_losses(
                ctrl, fmask, b["sims"], b["boxes"], args.tau, dil_idx)
            w_pair = W_PAIR * min(1.0, it / max(args.warmup_iters, 1))
            # center objective ablation: optionally drop the heatmap focal term
            w_hm = 0.0 if args.no_center else W_HM
            l_proto = det.new_zeros(())
            if args.proto_loss and ex is not None:
                w_proto = args.proto_loss * min(1.0, it / max(args.warmup_iters, 1))
                l_proto, zb = prototype_loss(ex["logits"], ex["boxes_px"],
                                             ex["img_idx"], b["z"], c_ema)
                c_ema = F.normalize(0.99 * c_ema + 0.01 * zb.mean(0), dim=0)
            else:
                w_proto = 0.0
            loss = (w_hm * l_hm + W_OFF * l_off + W_SIZE * l_size +
                    W_GIOU * l_giou + W_PROJ * l_proj + w_pair * l_pair +
                    w_proto * l_proto)
            opt.zero_grad()
            loss.backward()
            opt.step()
            it += 1
            parts.append([x.item() for x in
                          (l_hm, l_off, l_size, l_giou, l_proj, l_pair, l_proto)])
        sched.step()

        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            m = np.mean(parts, axis=0)
            preds, gts = infer_partition(model, data, val_paths, device)
            thr, _ = pick_threshold(preds, gts)
            rep = full_report(preds, gts, thr)
            eligible = (ep + 1) * iters_per_ep >= args.warmup_iters
            print(f"ep{ep+1:3d} hm={m[0]:.3f} off={m[1]:.3f} size={m[2]:.3f} "
                  f"giou={m[3]:.3f} proj={m[4]:.3f} pair={m[5]:.3f} proto={m[6]:.3f} "
                  f"| val mAP50={rep['mAP50']:.3f} F1={rep['f1']:.3f}@{thr:.2f}"
                  f"{'' if eligible else ' (pre-warmup, ineligible)'}")
            curve.append({"epoch": ep + 1, "mAP50": round(rep["mAP50"], 4),
                          "f1": round(rep["f1"], 4), "eligible": eligible})
            if not eligible:
                if rep["mAP50"] > best_pre["mAP50"]:
                    best_pre = {"mAP50": rep["mAP50"], "epoch": ep + 1}
            elif rep["mAP50"] > best["mAP50"]:
                best = {"mAP50": rep["mAP50"], "epoch": ep + 1, "thr": thr,
                        "state": {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}}

    if best["state"] is None:                # no eligible epoch scored > -1 (e.g.
        best = {"state": {k: v.cpu().clone()  # detection collapsed under --no_center)
                          for k, v in model.state_dict().items()},
                "epoch": args.epochs, "thr": 0.2, "mAP50": 0.0}
    model.load_state_dict(best["state"])
    val_preds, val_gts = infer_partition(model, data, val_paths, device)
    thr, _ = pick_threshold(val_preds, val_gts)          # frozen on val
    val_rep = full_report(val_preds, val_gts, thr)
    test_preds, test_gts = infer_partition(model, data, test_paths, device)
    test_rep = full_report(test_preds, test_gts, thr)
    for rep in (val_rep, test_rep):
        rep["strata"] = {k: {"recall": round(v[0], 3), "n": v[1]}
                         for k, v in rep.pop("recall_strata").items()}

    os.makedirs(ART_DIR, exist_ok=True)
    cfg = {"seed": args.seed, "epochs": args.epochs, "lr": args.lr, "wd": args.wd,
           "bs": args.bs, "tau": args.tau, "pair_dils": dil_idx,
           "warmup_iters": args.warmup_iters, "best_epoch": best["epoch"],
           "best_pre_warmup": best_pre,
           "score_thr": thr, "nms_iou": 0.5, "mask_thr": 0.5,
           "loss_weights": {"hm": W_HM, "off": W_OFF, "size": W_SIZE,
                            "giou": W_GIOU, "proj": W_PROJ, "pair": W_PAIR},
           "commonality_channel": bool(comm_ch), "proto_loss": args.proto_loss,
           "no_center": args.no_center,
           "in_dim": in_dim, "feat_layers": [21, 22, 23, 24]}
    tag = args.tag or f"boxinst_s{args.seed}"
    torch.save({"state": best["state"], "cfg": cfg},
               os.path.join(ART_DIR, tag + ".pt"))
    json.dump({"cfg": cfg, "val": val_rep, "test": test_rep, "val_curve": curve},
              open(os.path.join(ART_DIR, tag + ".json"), "w"), indent=2)
    print(f"\nbest ep{best['epoch']}  VAL mAP50={val_rep['mAP50']:.3f} "
          f"F1={val_rep['f1']:.3f}  |  TEST mAP50={test_rep['mAP50']:.3f} "
          f"mAP50-95={test_rep['mAP50_95']:.3f} F1={test_rep['f1']:.3f} "
          f"countMAE={test_rep['count_mae']:.2f}  (thr={thr:.2f} frozen on val)")
    print(f"TEST strata {test_rep['strata']}")
    print(f"wrote boxinst/artifacts/{tag}.pt and .json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.975,
                    help="pairwise affinity threshold; default = p75 of the "
                         "dilation-4 DINO cosine distribution (feature stats only, "
                         "never tuned on masks)")
    ap.add_argument("--pair_dils", default="4,5,6,7",
                    help="indices into the 8 cached (dir,dilation) offsets; "
                         "4-7 = dilation 4 (16 image px = one DINO patch)")
    ap.add_argument("--commonality_channel", action="store_true",
                    help="append the train-fit LDA treeness channel to the mask neck")
    ap.add_argument("--proto_loss", type=float, default=0.0,
                    help="weight for the cross-box prototype-consistency loss (0=off)")
    ap.add_argument("--no_center", action="store_true",
                    help="ablation: drop the centre heatmap focal objective")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--warmup_iters", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
