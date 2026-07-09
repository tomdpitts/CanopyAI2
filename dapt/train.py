"""Train a fixed probe head on frozen features; SELECT on val; REPORT on test.

Discipline (see PLAN §7 / methodology): val is the selection surface (operating
threshold + best-epoch checkpoint + any HP). test is the held-out comparison surface,
scored once with the val-frozen threshold and never used to drive design choices.

Identical protocol across arms (only --arm / cached features differ). Seeds numpy +
torch from one --seed and records it. Target-encoder + loss config are frozen in
dapt.targets / dapt.head and never tuned per-arm.

Usage:
    .venv/bin/python -m dapt.train --arm web --capacity linear --epochs 60 --seed 0
"""
import argparse
import hashlib
import json
import os
import random

import numpy as np
import torch

from dapt.backbone import pick_device
from dapt.data.cohort import REPO
from dapt.dataset import CohortData
from dapt.decode import decode
from dapt.eval import full_report, pick_threshold
from dapt.head import ProbeHead, probe_loss

TKEYS = ["feat", "heatmap", "offset", "size", "reg_mask"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def infer_partition(head, data, paths, device, score_thr=0.05):
    """preds [(boxes,scores) numpy], gts [boxes numpy], per tile."""
    head.eval()
    preds, gts = [], []
    for p in paths:
        out = head(data.batch([p])["feat"].to(device))
        boxes, scores = decode(out.cpu(), score_thr=score_thr)
        preds.append((boxes.numpy(), scores.numpy()))
        gts.append(data.tiles[p]["boxes"])
    return preds, gts


def train_probe(arm, capacity="linear", frac=1.0, seed=0, epochs=60, lr=1e-3,
                wd=1e-4, bs=8, device=None, eval_every=10, verbose=True):
    """Train one probe. Returns dict with val+test reports and the test preds/gts
    (for downstream bootstrap), plus the best checkpoint state."""
    set_seed(seed)
    device = pick_device(device)
    data = CohortData(arm)
    train_paths = data.train_subset(frac, seed)
    val_paths = data.partition("val")
    test_paths = data.partition("test")
    in_dim = data.batch([train_paths[0]])["feat"].shape[1]

    head = ProbeHead(in_dim, capacity=capacity).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    if verbose:
        print(f"arm={arm} cap={capacity} frac={frac} seed={seed} device={device} "
              f"in_dim={in_dim} train={len(train_paths)} val={len(val_paths)} "
              f"test={len(test_paths)}")

    best = {"mAP50": -1.0, "state": None, "thr": 0.2, "epoch": -1}
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        head.train()
        order = list(train_paths)
        rng.shuffle(order)
        losses = []
        for i in range(0, len(order), bs):
            b = data.batch(order[i:i + bs])
            b = {k: b[k].to(device) for k in TKEYS}
            out = head(b["feat"])
            loss, parts = probe_loss(out, b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(parts["total"])
        sched.step()

        if (ep + 1) % eval_every == 0 or ep + 1 == epochs:
            preds, gts = infer_partition(head, data, val_paths, device)
            thr, _ = pick_threshold(preds, gts)
            rep = full_report(preds, gts, thr)
            if verbose:
                print(f"  ep{ep+1:3d} loss={np.mean(losses):.3f} "
                      f"val mAP50={rep['mAP50']:.3f} F1={rep['f1']:.3f}@{thr:.2f}")
            if rep["mAP50"] > best["mAP50"]:
                best = {"mAP50": rep["mAP50"], "thr": thr, "epoch": ep + 1,
                        "state": {k: v.cpu().clone()
                                  for k, v in head.state_dict().items()}}

    # freeze selection: best checkpoint + val threshold -> report val AND test
    head.load_state_dict(best["state"])
    val_preds, val_gts = infer_partition(head, data, val_paths, device)
    test_preds, test_gts = infer_partition(head, data, test_paths, device)
    val_rep = full_report(val_preds, val_gts, best["thr"])
    test_rep = full_report(test_preds, test_gts, best["thr"])
    for rep in (val_rep, test_rep):
        rep["strata"] = {k: {"recall": round(v[0], 3), "n": v[1]}
                         for k, v in rep.pop("recall_strata").items()}
    return {"arm": arm, "capacity": capacity, "frac": frac, "seed": seed,
            "best_epoch": best["epoch"], "val": val_rep, "test": test_rep,
            "state": best["state"], "thr": best["thr"],
            "test_preds": test_preds, "test_gts": test_gts,
            "test_paths": list(test_paths),
            "test_domains": [data.tiles[p]["domain"] for p in test_paths]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--capacity", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--out", default="dapt/artifacts")
    args = ap.parse_args()

    r = train_probe(args.arm, args.capacity, args.frac, args.seed, args.epochs,
                    args.lr, args.wd, args.bs, args.device, args.eval_every)
    cfg_hash = hashlib.md5(json.dumps(
        {"cap": args.capacity, "lr": args.lr, "wd": args.wd, "bs": args.bs,
         "epochs": args.epochs}, sort_keys=True).encode()).hexdigest()[:8]
    tag = f"{args.arm}_{args.capacity}_frac{int(args.frac*100)}_s{args.seed}_{cfg_hash}"
    out_dir = os.path.join(REPO, args.out)
    os.makedirs(out_dir, exist_ok=True)
    dump = {k: r[k] for k in ("arm", "capacity", "frac", "seed", "best_epoch",
                              "thr", "val", "test")}
    dump["cfg_hash"] = cfg_hash
    json.dump(dump, open(os.path.join(out_dir, tag + ".json"), "w"), indent=2)
    torch.save(r["state"], os.path.join(out_dir, tag + ".pt"))
    t, v = r["test"], r["val"]
    print(f"\nTEST mAP50={t['mAP50']:.3f} mAP50-95={t['mAP50_95']:.3f} "
          f"AP_small={t['AP_small']:.3f} F1={t['f1']:.3f} countMAE={t['count_mae']:.2f}"
          f"   (val mAP50={v['mAP50']:.3f}, thr={r['thr']:.2f} frozen from val)")
    print(f"TEST strata {t['strata']}")
    print(f"wrote {os.path.relpath(os.path.join(out_dir, tag + '.json'), REPO)}")


if __name__ == "__main__":
    main()
