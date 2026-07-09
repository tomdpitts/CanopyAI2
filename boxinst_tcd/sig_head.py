"""Learned signature-ONLY mask head + parameter sweep.

No controller, no F_mask, no dynamic filter: the per-box cosine-to-signature map is
already instance-specific, so a small STATIC conv stack over [cosine, rel_x, rel_y]
is the whole mask decoder. Trained box-only (projection + DINO-affinity pairwise);
detection untouched. The mask head trains standalone on GT-box signatures.

Selection is HONEST: gt_polys.json is used for eval/selection only, never training.
The head config AND the mask threshold are picked on VAL polygons; TEST is reported
once at the val-selected setting. Baselines: box-fill 0.644, full CondInst head 0.648,
training-free cosine threshold 0.691 (all GT-prompted mIoU).

Sweep axes: head width (hid), depth, signature pooling (unweighted vs centre-weighted
to de-weight contaminated box corners), and eval threshold.

Usage:
    .venv/bin/python -m boxinst_tcd.sig_head --sweep
    .venv/bin/python -m boxinst_tcd.sig_head --epochs 200 --hid 16   # single run
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dapt.backbone import pick_device
from boxinst.losses import gather_instances, pairwise_loss, projection_loss
from boxinst.model import GRID, MASK_RES, REL_NORM, STRIDE
from boxinst_tcd.cache import COMM, key
from boxinst_tcd.eval_masks import mask_iou_matrix, raster_polys
from boxinst_tcd.prepare import OUT, RES
from boxinst_tcd.train import Data

ART = os.path.join(OUT, "artifacts")
THRESHOLDS = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]


def box_signature_cosine(z_batch, img_idx, boxes_px, pooling="mean", res=MASK_RES):
    """Per-instance cosine-to-signature map, with pooling mode (fully vectorized).

    pooling='mean'   : unweighted mean of in-box cells (contaminated by corners).
    pooling='center' : Gaussian weight toward the box centre (sigma=0.3 of box),
                       so corner/background cells contribute less -> purer signature.
    """
    N = len(img_idx)
    if N == 0:
        return z_batch.new_zeros((0, res, res))
    B, D, G, _ = z_batch.shape
    dev = z_batch.device
    r = torch.arange(G, device=dev, dtype=torch.float32)
    cyc, cxc = torch.meshgrid(r, r, indexing="ij")
    cxp = ((cxc + 0.5) * STRIDE).flatten()                        # (P,)
    cyp = ((cyc + 0.5) * STRIDE).flatten()
    bx = boxes_px.to(dev).float()                                 # (N,4)
    inbox = ((cxp[None] >= bx[:, 0:1]) & (cxp[None] < bx[:, 2:3]) &
             (cyp[None] >= bx[:, 1:2]) & (cyp[None] < bx[:, 3:4])).float()   # (N,P)
    if pooling == "center":
        u = (cxp[None] - bx[:, 0:1]) / (bx[:, 2:3] - bx[:, 0:1]).clamp(min=1)
        v = (cyp[None] - bx[:, 1:2]) / (bx[:, 3:4] - bx[:, 1:2]).clamp(min=1)
        inbox = inbox * torch.exp(-(((u - .5) ** 2 + (v - .5) ** 2) / (2 * .3 ** 2)))
    zf = z_batch.reshape(B, D, -1)[img_idx]                       # (N,D,P)
    w = inbox[:, None]                                            # (N,1,P)
    sig = (zf * w).sum(-1) / w.sum(-1).clamp(min=1e-6)            # (N,D)
    sig = F.normalize(sig, dim=1)
    cos = torch.einsum("ndp,nd->np", zf, sig).reshape(N, 1, G, G)
    return F.interpolate(cos, size=(res, res), mode="bilinear",
                         align_corners=False)[:, 0]


class SigHead(nn.Module):
    def __init__(self, hid=16, depth=2, use_rel=True):
        super().__init__()
        self.use_rel = use_rel
        cin = 1 + (2 if use_rel else 0)
        layers, c = [], cin
        for _ in range(depth):
            layers += [nn.Conv2d(c, hid, 3, padding=1), nn.GroupNorm(8, hid),
                       nn.ReLU(True)]
            c = hid
        layers += [nn.Conv2d(hid, 1, 1)]
        self.net = nn.Sequential(*layers)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, sig, centers, R=MASK_RES):
        dev = sig.device
        chans = [sig[:, None]]
        if self.use_rel:
            r = torch.arange(R, device=dev, dtype=torch.float32)
            yy, xx = torch.meshgrid(r, r, indexing="ij")
            chans += [((xx[None] - centers[:, 0, None, None]) / REL_NORM)[:, None],
                      ((yy[None] - centers[:, 1, None, None]) / REL_NORM)[:, None]]
        return self.net(torch.cat(chans, dim=1))[:, 0]


def train_head(data, tr, cfg, dev, tau=0.975, dils=(4, 5, 6, 7), epochs=200,
               warmup=500, bs=8, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    head = SigHead(cfg["hid"], cfg["depth"], cfg["use_rel"]).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    it = 0
    for ep in range(epochs):
        head.train(); order = list(tr); rng.shuffle(order)
        for i in range(0, len(order), bs):
            b = data.batch(order[i:i + bs], dev)
            inst = gather_instances(b["boxes"], dev)
            if inst is None:
                continue
            img_idx, boxes_px, cells, centers = inst
            sig = box_signature_cosine(b["z"], img_idx, boxes_px, cfg["pooling"])
            logits = head(sig, centers)
            l_proj = projection_loss(logits, boxes_px)
            l_pair = pairwise_loss(logits, boxes_px, b["sims"][img_idx], tau, list(dils))
            loss = l_proj + min(1.0, it / max(warmup, 1)) * l_pair
            opt.zero_grad(); loss.backward(); opt.step(); it += 1
    return head


@torch.no_grad()
def eval_part(head, data, part, pooling, dev, thresholds=THRESHOLDS):
    head.eval()
    polys = json.load(open(os.path.join(OUT, "gt_polys.json")))
    probs_by_tile, gm_by_tile = [], []
    for p in data.partition(part):
        gbx = data.tiles[p]["boxes"]
        if len(gbx) == 0:
            continue
        z = torch.from_numpy(np.load(os.path.join(COMM, key(p) + ".npz"))["z"]
                             ).float()[None].to(dev)
        idx = torch.zeros(len(gbx), dtype=torch.long, device=dev)
        boxes = torch.tensor(gbx, dtype=torch.float32, device=dev)
        centers = torch.stack([(boxes[:, 0] + boxes[:, 2]) / 8,
                               (boxes[:, 1] + boxes[:, 3]) / 8], 1)
        sig = box_signature_cosine(z, idx, boxes, pooling)
        prob = torch.sigmoid(F.interpolate(head(sig, centers)[:, None],
                             size=(RES, RES), mode="bilinear",
                             align_corners=False))[:, 0].cpu().numpy()
        probs_by_tile.append(prob)
        gm_by_tile.append(np.array(raster_polys(polys[p])))
    out = {}
    for thr in thresholds:
        ious, tp, fp, fn = [], 0, 0, 0
        for prob, gm in zip(probs_by_tile, gm_by_tile):
            pm = prob >= thr
            iou = mask_iou_matrix(pm, gm)
            for i in range(min(len(pm), len(gm))):
                ious.append(float(iou[i, i]))
            pf, gf = pm.any(0), gm.any(0)
            tp += int((pf & gf).sum()); fp += int((pf & ~gf).sum())
            fn += int((~pf & gf).sum())
        ious = np.array(ious)
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        out[thr] = {"mIoU": float(ious.mean()), "IoU50": float(np.mean(ious >= 0.5)),
                    "areaF1": float(2 * prec * rec / (prec + rec + 1e-9))}
    return out


def sweep(args):
    dev = pick_device(args.device)
    data = Data()
    tr = data.partition("train")
    grid = list(itertools.product(args.hids, args.depths, args.poolings, [True]))
    print(f"sweep: {len(grid)} configs x {len(THRESHOLDS)} thr, train={len(tr)}")
    print(f"baselines (GT-prompted test mIoU): box-fill 0.644 | full head 0.648 | "
          f"training-free cos-thresh 0.691\n")
    results = []
    for hid, depth, pooling, use_rel in grid:
        cfg = {"hid": hid, "depth": depth, "pooling": pooling, "use_rel": use_rel}
        head = train_head(data, tr, cfg, dev, tau=args.tau, epochs=args.epochs,
                          seed=args.seed)
        val = eval_part(head, data, "val", pooling, dev)
        best_thr = max(val, key=lambda t: val[t]["mIoU"])
        test = eval_part(head, data, "test", pooling, dev, [best_thr])[best_thr]
        row = {**cfg, "val_thr": best_thr, "val_mIoU": round(val[best_thr]["mIoU"], 3),
               "test_mIoU": round(test["mIoU"], 3),
               "test_IoU50": round(test["IoU50"], 3),
               "test_areaF1": round(test["areaF1"], 3),
               "state": head.state_dict()}
        results.append(row)
        print(f"hid={hid} depth={depth} pool={pooling:8s} | val_thr={best_thr} "
              f"val_mIoU={row['val_mIoU']} -> TEST mIoU={row['test_mIoU']} "
              f"IoU>.5={row['test_IoU50']} areaF1={row['test_areaF1']}")
    best = max(results, key=lambda r: r["val_mIoU"])
    print(f"\nBEST (by val): hid={best['hid']} depth={best['depth']} "
          f"pool={best['pooling']} thr={best['val_thr']} -> "
          f"TEST mIoU={best['test_mIoU']} IoU>.5={best['test_IoU50']} "
          f"areaF1={best['test_areaF1']}")
    torch.save({"state": best.pop("state"), "cfg": best},
               os.path.join(ART, "sighead_best.pt"))
    json.dump([{k: v for k, v in r.items() if k != "state"} for r in results],
              open(os.path.join(OUT, "sighead_sweep.json"), "w"), indent=2)
    print(f"wrote boxinst_tcd/sighead_sweep.json + sighead_best.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--hids", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--poolings", nargs="+", default=["mean", "center"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--tau", type=float, default=0.975)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    sweep(args)


if __name__ == "__main__":
    main()
