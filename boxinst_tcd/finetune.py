"""Light backbone fine-tune: unfreeze the last N DINOv3 blocks and train end-to-end.

The frozen-feature ceiling for ITC detection is ~0.46 box mAP50. This is the one lever
left: let gradients into the top of the backbone. Early blocks stay frozen (their
activations aren't stored, so memory ~ N unfrozen blocks); the head warm-starts from
the frozen-feature detector (tcd_itc_A). Same losses, same canopy-ignore. Cached
pairwise/commonality (auxiliary mask targets) are reused as-is.

Usage:
    .venv/bin/python -m boxinst_tcd.finetune --ft_blocks 2 --epochs 60 --lr_bb 1e-5
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from dapt.backbone import FrozenDinoV3Features, load_tile, pick_device
from boxinst.cache_feats import LAYERS
from boxinst.losses import det_loss, mask_losses
from boxinst.model import BoxInstHead
from boxinst_tcd import det_eval
from boxinst_tcd.prepare import OUT
from boxinst_tcd.train import Data

ART = os.path.join(OUT, "artifacts")


class FTBackbone(torch.nn.Module):
    """DINOv3 with the last `ft_blocks` transformer blocks trainable; grad-enabled
    multi-layer feature extraction (concat of LAYERS, L2-normed)."""

    def __init__(self, ft_blocks, device):
        super().__init__()
        self.net = FrozenDinoV3Features("web", layers=LAYERS, device=device)
        self.device = self.net.device
        self.patch = self.net.patch
        self.out_dim = self.net.out_dim
        bb = self.net.backbone
        for p in bb.parameters():
            p.requires_grad_(False)
        n = len(bb.model.layer)
        self.trainable = []
        for i in range(n - ft_blocks, n):
            for p in bb.model.layer[i].parameters():
                p.requires_grad_(True); self.trainable.append(p)
        for p in bb.norm.parameters():
            p.requires_grad_(True); self.trainable.append(p)
        bb.eval()                      # LayerNorm-only; eval is fine, grads still flow

    def extract(self, x):
        net = self.net
        x = x.to(self.device)
        x = (x - net.mean) / net.std
        hs = net.backbone(pixel_values=x, output_hidden_states=True).hidden_states
        h, w = x.shape[-2] // net.patch, x.shape[-1] // net.patch
        feats = []
        for li in net.layers:
            tok = hs[li][:, -(h * w):, :]
            tok = F.normalize(tok, dim=-1)
            feats.append(tok.transpose(1, 2).reshape(x.shape[0], net.hidden, h, w))
        return torch.cat(feats, dim=1)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


@torch.no_grad()
def infer(bb, head, data, paths, device, batch_targets):
    from dapt.decode import decode
    head.eval(); preds, gts = [], []
    for p in paths:
        x, _ = load_tile(p)
        feat = bb.extract(x)
        det, _, _ = head(feat)
        bx, sc = decode(det.cpu(), score_thr=0.02)
        preds.append((bx.numpy(), sc.numpy()))
        gts.append(data.tiles[p]["boxes"])
    return preds, gts


def train(args):
    set_seed(args.seed)
    dev = pick_device(args.device)
    data = Data()
    tr, va = data.partition("train"), data.partition("val")
    if args.n_train:
        tr = list(np.random.default_rng(args.seed).permutation(tr)[:args.n_train])
    bb = FTBackbone(args.ft_blocks, dev)
    head = BoxInstHead(bb.out_dim, det_tower=2).to(dev)
    if args.warm_start:
        ck = torch.load(os.path.join(ART, args.warm_start), map_location="cpu",
                        weights_only=False)
        head.load_state_dict(ck["state"]); print(f"warm-started head from {args.warm_start}")
    opt = torch.optim.Adam([
        {"params": head.parameters(), "lr": args.lr_head},
        {"params": bb.trainable, "lr": args.lr_bb}], weight_decay=1e-4)
    dil = [int(x) for x in "4,5,6,7".split(",")]
    val_can = det_eval.load_canopies(va)
    n_bb = sum(p.numel() for p in bb.trainable) / 1e6
    print(f"FT: ft_blocks={args.ft_blocks} ({n_bb:.1f}M trainable bb) head_lr={args.lr_head} "
          f"bb_lr={args.lr_bb} train={len(tr)} bs={args.bs}")
    rng = np.random.default_rng(args.seed)
    best = {"mAP50": -1, "bb": None, "head": None, "epoch": -1, "thr": 0.15}
    it = 0
    for ep in range(args.epochs):
        head.train(); order = list(tr); rng.shuffle(order); losses = []
        for i in range(0, len(order), args.bs):
            paths = order[i:i + args.bs]
            xs = torch.cat([load_tile(p)[0] for p in paths], 0)
            feat = bb.extract(xs)
            b = {k: data.batch(paths, dev)[k] for k in
                 ("heatmap", "offset", "size", "reg_mask", "ignore", "sims", "boxes")}
            det, ctrl, fmask = head(feat)
            l_hm, l_off, l_size, l_giou = det_loss(det, b)
            l_proj, l_pair, n_inst, ex = mask_losses(
                ctrl, fmask, b["sims"], b["boxes"], 0.975, dil)
            w_pair = min(1.0, it / 800)
            loss = l_hm + l_off + 0.1 * l_size + l_giou + l_proj + w_pair * l_pair
            opt.zero_grad(); loss.backward(); opt.step(); it += 1
            losses.append(l_hm.item())
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            preds, gts = infer(bb, head, data, va, dev, None)
            thr, _ = det_eval.pick_threshold(preds, gts, val_can)
            rep = det_eval.full_report(preds, gts, val_can, thr)
            print(f"  ep{ep+1:3d} hm={np.mean(losses):.3f} | val boxAP50={rep['mAP50']:.3f} "
                  f"F1={rep['f1']:.3f}@{thr:.2f}")
            if rep["mAP50"] > best["mAP50"]:
                best = {"mAP50": rep["mAP50"], "epoch": ep + 1, "thr": thr,
                        "head": {k: v.cpu().clone() for k, v in head.state_dict().items()},
                        "bb": {k: v.detach().cpu().clone()
                               for k, v in bb.net.backbone.state_dict().items()}}
    os.makedirs(ART, exist_ok=True)
    cfg = {"in_dim": bb.out_dim, "det_tower": 2, "score_thr": best["thr"],
           "nms_iou": 0.5, "mask_thr": 0.5, "ft_blocks": args.ft_blocks,
           "best_epoch": best["epoch"], "commonality_channel": False}
    torch.save({"head": best["head"], "bb": best["bb"], "cfg": cfg},
               os.path.join(ART, f"tcd_{args.tag}.pt"))
    print(f"[{args.tag}] best ep{best['epoch']} val boxAP50={best['mAP50']:.3f} "
          f"thr={best['thr']:.2f} -> tcd_{args.tag}.pt (frozen ref: 0.41 val / 0.46 test)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="itc_ft")
    ap.add_argument("--ft_blocks", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--lr_head", type=float, default=5e-4)
    ap.add_argument("--lr_bb", type=float, default=1e-5)
    ap.add_argument("--warm_start", default="tcd_itc_A.pt")
    ap.add_argument("--n_train", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
