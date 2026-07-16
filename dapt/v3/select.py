"""v3 per-SSL-seed rung selection — INNER-VAL ONLY, never touches held-out folds.

For each candidate arm (checkpoint rung), runs the same fold machinery as
dapt.v3.kfold but scores mAP50 on the inner-val slice of each fold's TRAINING side
(the slice train_fold already uses for threshold selection). Held-out fold tiles are
never predicted here, so the step-7 paired test gap stays unread (SPEC `## v3 study`
step 6; dapt.select_checkpoint uses the OLD v1/v2 split — do NOT use it for v3).

Usage:
    .venv/bin/python -m dapt.v3.select --arms dapt_v3_s101_i499 dapt_v3_s101_i999 \
        --seeds 0 1 --capacity mlp
"""
import argparse
import json
import os

import numpy as np

from dapt.backbone import pick_device
from dapt.eval import average_precision
from dapt.v3.data import REPO, V3Data
from dapt.v3.kfold import _folds, _infer, train_fold


def val_score(arm, capacity, probe_seed, k, epochs, lr, wd, bs, device):
    """Mean inner-val mAP50 over one repeat of k folds."""
    data = V3Data(arm)
    safe = data.leakage_safe()
    rng = np.random.default_rng(1000 * probe_seed)   # repeat-0 partition, as kfold
    folds = _folds(safe, k, rng)
    scores = []
    for f in range(k):
        rest = [n for j in range(k) if j != f for n in folds[j]]
        n_val = max(2, len(rest) // (k - 1))
        val_names, train_names = rest[:n_val], rest[n_val:]
        head, _ = train_fold(data, train_names, val_names, capacity,
                             probe_seed, epochs, lr, wd, bs, device)
        preds, gts = _infer(head, data, val_names, device)
        scores.append(average_precision(preds, gts, 0.5))
    return float(np.nanmean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--capacity", default="mlp", choices=["linear", "mlp"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", default="dapt/v3/artifacts/val_select.json")
    args = ap.parse_args()
    device = pick_device()

    results = {}
    for arm in args.arms:
        per_seed = [val_score(arm, args.capacity, s, args.k, args.epochs,
                              args.lr, args.wd, args.bs, device)
                    for s in args.seeds]
        results[arm] = {"val_map50": float(np.mean(per_seed)),
                        "per_seed": per_seed}
        print(f"[{arm}] inner-val mAP50 {results[arm]['val_map50']:.3f} "
              f"(per-seed {['%.3f' % v for v in per_seed]})", flush=True)

    out_path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # flock so concurrent capacities can't clobber each other's read-modify-write
    import fcntl
    with open(out_path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        existing = json.load(open(out_path)) if os.path.exists(out_path) else {}
        existing.update({f"{arm}|{args.capacity}": {**results[arm],
                                                    "capacity": args.capacity,
                                                    "seeds": args.seeds}
                         for arm in results})
        tmp = out_path + ".tmp"
        json.dump(existing, open(tmp, "w"), indent=2)
        os.replace(tmp, out_path)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
