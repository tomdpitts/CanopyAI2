"""Select the DAPT checkpoint on VAL — never on test (pre-registered protocol).

For each registered dapt arm (dapt/ssl/checkpoints.json), caches features if needed,
trains the L2 probe over the given seeds, and reports FULL-VAL mAP50 mean ± std.
Prints the val ranking and the winner. Deliberately prints NO test numbers — the
winner (and only the winner) then gets the one-shot test report via:

    .venv/bin/python -m dapt.run_baseline --arms <winner> web --capacity mlp \
        --seeds 0 1 2 3 4

Degradation alarm: if later-step checkpoints rank below web on val, features are
drifting — expect the winner to be an earlier step.

Usage:
    .venv/bin/python -m dapt.select_checkpoint            # all arms in checkpoints.json
    .venv/bin/python -m dapt.select_checkpoint --arms dapt_s1000 dapt_s2000 web
"""
import argparse
import json
import os

import numpy as np

from dapt.cache_features import cache_arm
from dapt.data.cohort import REPO
from dapt.train import train_probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None,
                    help="default: all arms in checkpoints.json + web reference")
    ap.add_argument("--capacity", default="mlp", choices=["linear", "mlp"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="dapt/artifacts/selection/checkpoint_selection.json")
    args = ap.parse_args()

    arms = args.arms
    if not arms:
        reg = json.load(open(os.path.join(REPO, "dapt/ssl/checkpoints.json")))
        arms = sorted(reg) + ["web"]          # web = the do-not-degrade reference
    split_path = os.path.join(REPO, "dapt/data/split.json")

    n_tiles = len(json.load(open(split_path))["tiles"])
    rows = []
    for arm in arms:
        cache_dir = os.path.join(REPO, "dapt/cache", arm)
        n_cached = len([f for f in os.listdir(cache_dir) if f.endswith(".npy")]) \
            if os.path.isdir(cache_dir) else 0
        if n_cached < n_tiles:   # empty OR partial (e.g. an interrupted earlier run)
            print(f"[{arm}] caching features ({n_cached}/{n_tiles} present)...",
                  flush=True)
            cache_arm(arm, split_path, os.path.join(REPO, "dapt/cache"))
        vals = []
        for s in args.seeds:
            r = train_probe(arm, args.capacity, 1.0, s, args.epochs, verbose=False)
            vals.append(r["val"]["mAP50"])     # VAL ONLY — test stays unseen here
        rows.append({"arm": arm, "val_mAP50_mean": float(np.mean(vals)),
                     "val_mAP50_std": float(np.std(vals)), "seeds": args.seeds})
        print(f"[{arm}] val mAP50 = {np.mean(vals):.3f} ± {np.std(vals):.3f}",
              flush=True)

    rows.sort(key=lambda r: -r["val_mAP50_mean"])
    winner = rows[0]["arm"]
    out = {"capacity": args.capacity, "ranking": rows, "winner": winner}
    out_path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nVAL RANKING: " + " > ".join(f"{r['arm']}({r['val_mAP50_mean']:.3f})"
                                          for r in rows))
    print(f"WINNER (val-selected): {winner}")
    if winner == "web":
        print("NOTE: web outranks every DAPT checkpoint on val — adaptation degraded "
              "features; the DAPT≈web/DAPT<web outcome is already likely.")
    print(f"next: .venv/bin/python -m dapt.run_baseline --arms {winner} web "
          f"--capacity mlp --seeds {' '.join(map(str, args.seeds))}")
    print(f"wrote {os.path.relpath(out_path, REPO)}")


if __name__ == "__main__":
    main()
