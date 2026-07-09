"""Label-efficiency curve: TEST mAP50 vs # labelled train tiles, per arm.

The headline axis for DAPT value ("same accuracy, fewer labels"). Here it establishes
the web-vs-sat baseline curve; DAPT is added as another arm later. Selection on val,
report on test, paired gap per fraction.

Usage:
    .venv/bin/python -m dapt.label_efficiency --arms web sat --capacity mlp \
        --fracs 0.25 0.5 1.0 --seeds 0 1 2 3 4
"""
import argparse
import json
import os

import numpy as np

from dapt.data.cohort import REPO
from dapt.run_baseline import multiseed_bootstrap, paired_gap_bootstrap
from dapt.train import train_probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["web", "sat"])
    ap.add_argument("--capacity", default="mlp", choices=["linear", "mlp"])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="dapt/artifacts/labeff/label_efficiency.json")
    args = ap.parse_args()

    curve = {arm: [] for arm in args.arms}
    for frac in args.fracs:
        preds_store = {}
        for arm in args.arms:
            map50s, preds_by_seed, gts = [], [], None
            for s in args.seeds:
                r = train_probe(arm, args.capacity, frac, s, args.epochs,
                                verbose=False)
                map50s.append(r["test"]["mAP50"])
                preds_by_seed.append(r["test_preds"])
                gts = r["test_gts"]
            lo, med, hi = multiseed_bootstrap(preds_by_seed, gts)
            preds_store[arm] = (preds_by_seed, gts)
            pt = {"frac": frac, "n_train": None,
                  "mAP50_mean": float(np.mean(map50s)),
                  "mAP50_std": float(np.std(map50s)), "ci95": [lo, hi]}
            curve[arm].append(pt)
            print(f"[frac {frac:.2f} {arm}] test mAP50 "
                  f"{np.mean(map50s):.3f}+/-{np.std(map50s):.3f} 95%CI[{lo:.3f},{hi:.3f}]")
        if len(args.arms) >= 2:
            a, b = args.arms[0], args.arms[1]
            glo, gmed, ghi, pgt0 = paired_gap_bootstrap(
                preds_store[a][0], preds_store[b][0], preds_store[a][1])
            print(f"  paired {a}-{b} gap={gmed:+.3f} CI[{glo:+.3f},{ghi:+.3f}] "
                  f"P>0={pgt0:.2f}\n")

    out = {"capacity": args.capacity, "fracs": args.fracs, "seeds": args.seeds,
           "curve": curve}
    out_path = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"wrote {os.path.relpath(out_path, REPO)}")


if __name__ == "__main__":
    main()
