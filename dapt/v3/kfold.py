"""Repeated k-fold over the leakage-safe v3 tiles; pooled paired-gap DAPT vs web.

For each arm and probe seed: repeated K-fold over the 81 leakage-safe tiles. Each fold
trains the probe on the other folds (+ an inner val slice for threshold selection) and
predicts the held-out fold -> out-of-fold (OOF) predictions covering all 81 tiles.
Averaging OOF over folds within a repeat, then over probe seeds, gives each arm a
per-tile prediction set; paired-gap bootstrap (reused from run_baseline) then tests
DAPT-web on the SAME resampled tiles.

Zero DAPT leakage (eval only on pool-excluded tiles), full 81-tile coverage, ~65
train tiles/fold (no starvation). See dapt/ssl/SPEC.md `## v3 study`.

Usage:
    .venv/bin/python -m dapt.v3.kfold --arms web sat dapt_v3_s101_i1000 \
        --k 5 --repeats 3 --seeds 0 1 2 3 4 --capacity mlp

Pooling: join member arms with '+' (e.g. --arms dapt_v3_s101_i999+dapt_v3_s201_i999
web) — the group's OOF runs are the concatenation of every member's runs (the v1/v2
pooled-seed estimator). OOF runs are disk-cached per (arm, capacity, seed, protocol)
under dapt/v3/cache/oof/ so repeat invocations (mlp vs linear, other pairings) reuse
them.
"""
import argparse
import itertools
import json
import os
import pickle

import numpy as np
import torch

from dapt.backbone import pick_device
from dapt.decode import decode
from dapt.eval import full_report, pick_threshold
from dapt.head import ProbeHead, probe_loss
from dapt.run_baseline import _draw_map, paired_gap_bootstrap
from dapt.train import TKEYS, set_seed
from dapt.v3.data import REPO, V3Data

SUBSETS = {"FULL": None, "WON": {"WON"}, "BRU": {"BRU"}}


def _folds(items, k, rng):
    idx = rng.permutation(len(items))
    return [[items[i] for i in idx[f::k]] for f in range(k)]   # stratified-ish stripes


def train_fold(data, train_names, val_names, capacity, seed, epochs, lr, wd, bs,
               device):
    """Train probe on train_names, select thr on val_names; return (head, thr)."""
    set_seed(seed)
    head = ProbeHead(data.batch([train_names[0]])["feat"].shape[1], capacity).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        head.train()
        order = list(train_names)
        rng.shuffle(order)
        for i in range(0, len(order), bs):
            b = data.batch(order[i:i + bs])
            b = {k: b[k].to(device) for k in TKEYS}
            loss, _ = probe_loss(head(b["feat"]), b)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    # threshold on inner val
    preds, gts = _infer(head, data, val_names, device)
    thr, _ = pick_threshold(preds, gts)
    return head, thr


@torch.no_grad()
def _infer(head, data, names, device, score_thr=0.05):
    head.eval()
    preds, gts = [], []
    for b in names:
        out = head(data.batch([b])["feat"].to(device))
        boxes, scores = decode(out.cpu(), score_thr=score_thr)
        preds.append((boxes.numpy(), scores.numpy()))
        gts.append(data.tiles[b]["boxes"])
    return preds, gts


def oof_predictions(arm, capacity, probe_seed, k, repeats, epochs, lr, wd, bs, device):
    """Return {name: (boxes,scores)} averaged is not needed — last-repeat OOF per tile.

    We keep OOF per (repeat): for each repeat, every tile is predicted exactly once
    (by the fold that holds it out). Returns list over repeats of {name:(boxes,scores)}.
    """
    data = V3Data(arm)
    safe = data.leakage_safe()
    extra = data.overlap()   # NOTE: not used in headline (see SPEC); folds are safe-only
    per_repeat = []
    for rep in range(repeats):
        rng = np.random.default_rng(1000 * probe_seed + rep)
        folds = _folds(safe, k, rng)
        oof = {}
        for f in range(k):
            test_names = folds[f]
            rest = [n for j in range(k) if j != f for n in folds[j]]
            # inner val = 1/(k-1) slice of rest for threshold; rest-of-rest trains
            n_val = max(2, len(rest) // (k - 1))
            val_names, train_names = rest[:n_val], rest[n_val:]
            head, thr = train_fold(data, train_names, val_names, capacity,
                                   probe_seed, epochs, lr, wd, bs, device)
            preds, _ = _infer(head, data, test_names, device)
            for name, pr in zip(test_names, preds):
                oof[name] = pr
        per_repeat.append(oof)
    return data, safe, per_repeat


def cached_oof(arm, capacity, probe_seed, k, repeats, epochs, lr, wd, bs, device):
    """oof_predictions per-repeat dicts for ONE concrete arm+seed, disk-cached."""
    tag = (f"{arm}_{capacity}_s{probe_seed}_k{k}r{repeats}"
           f"e{epochs}lr{lr}wd{wd}bs{bs}")
    cpath = os.path.join(REPO, "dapt/v3/cache/oof", tag + ".pkl")
    if os.path.exists(cpath):
        with open(cpath, "rb") as fh:
            return pickle.load(fh)
    _, _, per_repeat = oof_predictions(arm, capacity, probe_seed, k, repeats,
                                       epochs, lr, wd, bs, device)
    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    with open(cpath, "wb") as fh:
        pickle.dump(per_repeat, fh)
    return per_repeat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="arm[0] vs arm[1] paired; '+'-join members to pool a group")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--capacity", default="mlp", choices=["linear", "mlp"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", default="dapt/v3/artifacts/kfold.json")
    args = ap.parse_args()
    device = pick_device()

    # tile metadata is arm-independent; take it from the first concrete member
    ref = V3Data(args.arms[0].split("+")[0])
    safe = ref.leakage_safe()
    gts_ref = [ref.tiles[n]["boxes"] for n in safe]
    domains = [ref.tiles[n]["domain"] for n in safe]
    idx_all = np.arange(len(safe))

    # collect OOF prediction runs: one per (member, probe seed, repeat) over all
    # safe tiles; a '+'-group pools every member's runs under the group key
    runs = {a: [] for a in args.arms}
    for arm in args.arms:
        for member in arm.split("+"):
            for s in args.seeds:
                per_repeat = cached_oof(member, args.capacity, s, args.k,
                                        args.repeats, args.epochs, args.lr,
                                        args.wd, args.bs, device)
                these = [[oof[n] for n in safe] for oof in per_repeat]
                runs[arm].extend(these)
                m = np.mean([_draw_map([run], gts_ref, idx_all, 0.5)
                             for run in these])
                print(f"[{member} s{s}] OOF mAP50 {m:.3f}", flush=True)
    out = {"k": args.k, "repeats": args.repeats, "seeds": args.seeds,
           "capacity": args.capacity, "n_safe_tiles": len(safe),
           "arm_map50": {a: float(np.mean([_draw_map([r], gts_ref, idx_all, 0.5)
                                           for r in runs[a]]))
                         for a in args.arms},
           "paired": {}}
    for a, b in itertools.combinations(args.arms, 2):
        pair = f"{a} - {b}"
        out["paired"][pair] = {}
        for name, doms in SUBSETS.items():
            idx = (list(range(len(safe))) if doms is None
                   else [i for i, d in enumerate(domains) if d in doms])
            gsub = [gts_ref[i] for i in idx]
            pa = [[pr[i] for i in idx] for pr in runs[a]]
            pb = [[pr[i] for i in idx] for pr in runs[b]]
            glo, gmed, ghi, p = paired_gap_bootstrap(pa, pb, gsub)
            res = glo > 0 or ghi < 0
            out["paired"][pair][name] = {"n_tiles": len(idx), "gap": gmed,
                                         "ci95": [glo, ghi], "p_gt0": p,
                                         "resolved": res}
            print(f"PAIRED {a}-{b} [{name:4s} {len(idx):2d}t] gap={gmed:+.3f} "
                  f"CI[{glo:+.3f},{ghi:+.3f}] P>0={p:.2f} -> "
                  f"{'RESOLVED' if res else 'not resolved'}", flush=True)

    out_path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
