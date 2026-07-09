"""Multi-seed L1/L2 baseline sweep: web vs sat, reported on TEST with bootstrap CIs.

For each (arm, capacity) we train over several seeds (training-init variance),
evaluate each on the held-out test set with its val-frozen threshold, and combine
two uncertainty sources:
  - seed variance: mean +/- std of test mAP50 across seeds
  - tile bootstrap: resample test tiles, average the metric across seeds per draw ->
    a 95% CI that reflects the small (33-tile) test set.

Usage:
    .venv/bin/python -m dapt.run_baseline --arms web sat --capacity linear \
        --seeds 0 1 2 3 4 --epochs 60
"""
import argparse
import json
import os

import numpy as np

from dapt.data.cohort import REPO
from dapt.eval import average_precision
from dapt.train import train_probe


def _draw_map(preds_by_seed, gts, idx, iou_thr):
    """Mean-over-seeds mAP50 on the tile subset `idx`."""
    gsub = [gts[i] for i in idx]
    return np.nanmean([average_precision([ps[i] for i in idx], gsub, iou_thr)
                       for ps in preds_by_seed])


def multiseed_bootstrap(preds_by_seed, gts, iou_thr=0.5, n=1000, seed=0):
    """Resample test tiles; per draw average mAP50 across seeds; return (lo, med, hi)."""
    rng = np.random.default_rng(seed)
    N = len(gts)
    vals = [_draw_map(preds_by_seed, gts, rng.integers(0, N, N), iou_thr)
            for _ in range(n)]
    vals = np.array([v for v in vals if not np.isnan(v)])
    return (float(np.percentile(vals, 2.5)), float(np.median(vals)),
            float(np.percentile(vals, 97.5)))


def paired_gap_bootstrap(preds_a, preds_b, gts, iou_thr=0.5, n=1000, seed=0):
    """Paired bootstrap of gap (a - b) on the SAME resampled tiles. This cancels
    tile-to-tile noise, the powerful test for whether arm a beats arm b.
    Returns (gap_lo, gap_med, gap_hi, frac_gap_gt_0)."""
    rng = np.random.default_rng(seed)
    N = len(gts)
    gaps = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        gaps.append(_draw_map(preds_a, gts, idx, iou_thr)
                    - _draw_map(preds_b, gts, idx, iou_thr))
    gaps = np.array([g for g in gaps if not np.isnan(g)])
    return (float(np.percentile(gaps, 2.5)), float(np.median(gaps)),
            float(np.percentile(gaps, 97.5)), float((gaps > 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["web", "sat"])
    ap.add_argument("--capacity", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="dapt/artifacts/baseline_summary.json")
    args = ap.parse_args()

    summary = {}
    preds_store = {}
    domains = None
    for arm in args.arms:
        map50s, f1s, cmaes, preds_by_seed, gts = [], [], [], [], None
        for s in args.seeds:
            r = train_probe(arm, args.capacity, args.frac, s, args.epochs,
                            verbose=False)
            t = r["test"]
            map50s.append(t["mAP50"])
            f1s.append(t["f1"])
            cmaes.append(t["count_mae"])
            preds_by_seed.append(r["test_preds"])
            gts = r["test_gts"]
            domains = r["test_domains"]
            print(f"[{arm} {args.capacity} s{s}] test mAP50={t['mAP50']:.3f} "
                  f"F1={t['f1']:.3f} countMAE={t['count_mae']:.2f}")
        lo, med, hi = multiseed_bootstrap(preds_by_seed, gts)
        preds_store[arm] = (preds_by_seed, gts)
        summary[arm] = {
            "capacity": args.capacity, "seeds": args.seeds,
            "mAP50_mean": float(np.mean(map50s)), "mAP50_std": float(np.std(map50s)),
            "mAP50_ci95": [lo, hi], "mAP50_ci_median": med,
            "f1_mean": float(np.mean(f1s)), "count_mae_mean": float(np.mean(cmaes))}
        print(f"==> {arm} {args.capacity}: test mAP50 "
              f"{np.mean(map50s):.3f}+/-{np.std(map50s):.3f}  "
              f"95%CI[{lo:.3f},{hi:.3f}]  F1={np.mean(f1s):.3f}\n")

    # paired gap between the first two arms (the powerful, noise-cancelling test).
    # PRIMARY = full test set (endpoint amended 2026-07-07); arid/NEON subsets are the
    # always-reported mechanism readout (arid-concentrated gain => arid-specific
    # adaptation; uniform incl. NEON => generic extra SSL).
    if len(args.arms) >= 2:
        a, b = args.arms[0], args.arms[1]
        subsets = {"FULL": None,
                   "arid": {"WON", "BRU"},
                   "NEON": {"NEON"}}
        summary["_paired_gap"] = {"a": a, "b": b}
        for name, doms in subsets.items():
            if doms is None:
                idx = list(range(len(domains)))
            else:
                idx = [i for i, d in enumerate(domains) if d in doms]
            pa = [[ps[i] for i in idx] for ps in preds_store[a][0]]
            pb = [[ps[i] for i in idx] for ps in preds_store[b][0]]
            g = [preds_store[a][1][i] for i in idx]
            glo, gmed, ghi, pgt0 = paired_gap_bootstrap(pa, pb, g)
            resolved = glo > 0 or ghi < 0
            summary["_paired_gap"][name] = {
                "n_tiles": len(idx), "gap_median": gmed, "gap_ci95": [glo, ghi],
                "frac_gap_gt_0": pgt0, "resolved": resolved}
            tag = "PRIMARY" if name == "FULL" else "secondary"
            print(f"PAIRED {a}-{b} [{name:4s} {len(idx):2d}t {tag:9s}] gap = "
                  f"{gmed:+.3f}  95%CI[{glo:+.3f},{ghi:+.3f}]  P(gap>0)={pgt0:.2f}"
                  f"  -> {'RESOLVED' if resolved else 'not resolved'}")

    out_path = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    json.dump(summary, open(out_path, "w"), indent=2)
    print(f"wrote {os.path.relpath(out_path, REPO)}")


if __name__ == "__main__":
    main()
