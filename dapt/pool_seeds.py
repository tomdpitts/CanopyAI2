"""Pooled multi-SSL-seed test: is DAPT > web when pooling N independent SSL runs?

Pools the val-selected winner arm from EACH SSL seed (identical short protocol) and
tests the pooled paired gap vs web: per tile-bootstrap draw, the DAPT value is the
mean mAP50 over all (SSL seed x probe seed) runs, web's is the mean over its probe
seeds — same paired, noise-cancelling construction as run_baseline, but the
treatment estimate now averages SSL-seed variability instead of riding one run.
Also reports each SSL seed's own paired gap (mini forest plot) and the across-seed
spread of gaps.

Usage:
    .venv/bin/python -m dapt.pool_seeds \
        --dapt-arms dapt_s43_p1k_i999 dapt_s124_p1k_i999 ... --baseline web
"""
import argparse
import json
import os

import numpy as np

from dapt.data.cohort import REPO
from dapt.run_baseline import _draw_map, paired_gap_bootstrap
from dapt.train import train_probe

SUBSETS = {"FULL": None, "arid": {"WON", "BRU"}, "NEON": {"NEON"}}


def pooled_gap_bootstrap(preds_group, preds_base, gts, iou_thr=0.5, n=1000, seed=0):
    """preds_group: list over runs (each = per-tile preds); pooled mean vs baseline."""
    rng = np.random.default_rng(seed)
    N = len(gts)
    gaps = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        gaps.append(_draw_map(preds_group, gts, idx, iou_thr)
                    - _draw_map(preds_base, gts, idx, iou_thr))
    gaps = np.array([g for g in gaps if not np.isnan(g)])
    return (float(np.percentile(gaps, 2.5)), float(np.median(gaps)),
            float(np.percentile(gaps, 97.5)), float((gaps > 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dapt-arms", nargs="+", required=True)
    ap.add_argument("--baseline", default="web")
    ap.add_argument("--capacity", default="mlp", choices=["linear", "mlp"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="dapt/artifacts/detection/pooled_seeds.json")
    args = ap.parse_args()

    runs = {}          # arm -> list over probe seeds of per-tile preds
    gts = domains = None
    for arm in [args.baseline] + args.dapt_arms:
        runs[arm] = []
        maps = []
        for s in args.seeds:
            r = train_probe(arm, args.capacity, 1.0, s, args.epochs, verbose=False)
            runs[arm].append(r["test_preds"])
            maps.append(r["test"]["mAP50"])
            gts, domains = r["test_gts"], r["test_domains"]
        print(f"[{arm}] test mAP50 {np.mean(maps):.3f} ± {np.std(maps):.3f}",
              flush=True)

    out = {"baseline": args.baseline, "dapt_arms": args.dapt_arms,
           "capacity": args.capacity, "probe_seeds": args.seeds, "per_seed": {},
           "pooled": {}}
    dapt_all = [p for arm in args.dapt_arms for p in runs[arm]]   # N_ssl x N_probe

    for name, doms in SUBSETS.items():
        idx = (list(range(len(domains))) if doms is None
               else [i for i, d in enumerate(domains) if d in doms])
        gsub = [gts[i] for i in idx]
        sub = lambda group: [[ps[i] for i in idx] for ps in group]

        # per-SSL-seed paired gaps (mini forest plot)
        seed_gaps = []
        for arm in args.dapt_arms:
            lo, med, hi, p = paired_gap_bootstrap(sub(runs[arm]),
                                                  sub(runs[args.baseline]), gsub)
            seed_gaps.append(med)
            out["per_seed"].setdefault(arm, {})[name] = {
                "gap": med, "ci95": [lo, hi], "p_gt0": p,
                "resolved": lo > 0 or hi < 0}
        # pooled
        lo, med, hi, p = pooled_gap_bootstrap(sub(dapt_all),
                                              sub(runs[args.baseline]), gsub)
        out["pooled"][name] = {
            "gap": med, "ci95": [lo, hi], "p_gt0": p, "resolved": lo > 0 or hi < 0,
            "gap_mean_over_seeds": float(np.mean(seed_gaps)),
            "gap_std_over_seeds": float(np.std(seed_gaps))}
        tag = "PRIMARY" if name == "FULL" else "secondary"
        print(f"POOLED({len(args.dapt_arms)} SSL seeds)-{args.baseline} "
              f"[{name:4s} {len(idx):2d}t {tag:9s}] gap = {med:+.3f}  "
              f"95%CI[{lo:+.3f},{hi:+.3f}]  P(gap>0)={p:.2f}  "
              f"-> {'RESOLVED' if lo > 0 or hi < 0 else 'not resolved'}   "
              f"(per-seed gaps {np.mean(seed_gaps):+.3f} ± {np.std(seed_gaps):.3f})",
              flush=True)

    out_path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"wrote {os.path.relpath(out_path, REPO)}")


if __name__ == "__main__":
    main()
