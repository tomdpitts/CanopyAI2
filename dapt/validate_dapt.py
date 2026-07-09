"""Probe every registered dapt* checkpoint vs web and print a selection table.

One command. For each arm named `dapt*` in dapt/ssl/checkpoints.json, caches its
frozen features (if not already) and runs the paired-gap bootstrap vs web at the
given capacity, then prints checkpoints sorted by gap so you pick the best one.
Selection cost = feature extraction only (no retraining) — this is the cheap,
anti-forgetting checkpoint-selection lever from dapt/ssl/SPEC.md.

NOTE: reports the gap on the FULL test set. The SPEC headlines the *arid (WON+BRU)*
subset; add a `--subset` domain filter to run_baseline to select on that instead.

Usage:
    .venv/bin/python -m dapt.validate_dapt --capacity linear --seeds 0 1 2 3 4
"""
import argparse
import json
import os
import subprocess
import sys

from dapt.data.cohort import REPO

REG = os.path.join(REPO, "dapt/ssl/checkpoints.json")


def run(mod, *cliargs):
    subprocess.run([sys.executable, "-m", mod, *cliargs], check=True, cwd=REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()

    reg = json.load(open(REG)) if os.path.exists(REG) else {}
    dapt_arms = sorted(k for k in reg if k.startswith("dapt"))
    if not dapt_arms:
        print(f"No dapt* checkpoints registered in {REG}. Add e.g. "
              f'{{"dapt": "dapt/ckpt/dapt_hf"}} and re-run.')
        return

    for arm in ["web", *dapt_arms]:                       # cache features once per arm
        run("dapt.cache_features", "--arm", arm)

    rows = []
    for arm in dapt_arms:
        out = f"dapt/artifacts/validate_{arm}_{a.capacity}.json"
        run("dapt.run_baseline", "--arms", arm, "web", "--capacity", a.capacity,
            "--seeds", *map(str, a.seeds), "--epochs", str(a.epochs), "--out", out)
        s = json.load(open(os.path.join(REPO, out)))
        g = s["_paired_gap"]
        rows.append((arm, s[arm]["mAP50_mean"], g["gap_median"], g["gap_ci95"],
                     g["frac_gap_gt_0"], g["resolved"]))

    rows.sort(key=lambda r: -r[2])
    print(f"\n=== DAPT checkpoint selection  (paired gap vs web, capacity={a.capacity}, "
          f"full test) ===")
    print(f"{'checkpoint':18s} {'mAP50':>7s} {'gap':>8s} {'gap_CI95':>18s} {'P(>0)':>6s} resolved")
    for arm, m, gm, ci, p, res in rows:
        print(f"{arm:18s} {m:7.3f} {gm:+8.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] {p:6.2f}  {res}")
    best = rows[0]
    print(f"\n>>> best checkpoint: {best[0]}  gap={best[2]:+.3f}  resolved={best[5]}  "
          f"(web baseline mAP50 in dapt/artifacts/baseline_summary.json)")


if __name__ == "__main__":
    main()
