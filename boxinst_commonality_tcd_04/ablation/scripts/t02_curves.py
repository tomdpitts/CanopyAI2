"""T0.2 -- AP across IoU thresholds: LACE vs DetecTree2, 3 seeds.

mAP50 is generous. Reporting AP at every IoU threshold from 0.50 to 0.95 shows where a
method actually lives, and box-supervised methods are usually assumed to fall away fastest.
Plot only -- both curves are already on disk:
  ours: phase4/results_no_ignore_band.json         (3 seeds x {with,no} ignore x 10 IoUs)
  dt2:  detectree2_baseline/results_dt2_s0_fullcov_noignore.json  (1 seed)
Published numbers are the WITH-ignore ones; the no-ignore variants are drawn faintly to
show the protocol's sensitivity in the same frame.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.ablation.scripts.t02_curves
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from boxinst_commonality_tcd_04 import evaluate as E
from ..lib import io

OURS = os.path.join(io.PHASE4, "results_no_ignore_band.json")
DT2 = os.path.join(io.DT2, "results_dt2_s0_fullcov_noignore.json")


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out", default=os.path.join(io.FIGURES, "per_iou.pdf"))
    a = ap_.parse_args()

    o = json.load(open(OURS)); d = json.load(open(DT2))
    x = np.array(E.IOU_50_95)

    def band(src, proto):
        v = np.array([src["per_seed"][s][proto]["per_iou"] for s in sorted(src["per_seed"])])
        return v.mean(0), (v.std(0, ddof=1) if len(v) > 1 else np.zeros(v.shape[1])), len(v)

    om, osd, on = band(o, "with_ignore")
    onm, _, _ = band(o, "no_ignore")
    dm, _, dn = band(d, "with_ignore")
    dnm, _, _ = band(d, "no_ignore")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(x, om, "-o", ms=4, lw=2, color="#1b6ca8",
            label=f"LACE (ours), {on} seeds, canopy-ignore")
    ax.fill_between(x, om - osd, om + osd, color="#1b6ca8", alpha=0.18, lw=0)
    ax.plot(x, dm, "-s", ms=4, lw=2, color="#c1440e",
            label=f"DetecTree2 (mask-supervised), {dn} seed")
    ax.plot(x, onm, "--", lw=1.2, color="#1b6ca8", alpha=0.65, label="LACE, no canopy-ignore")
    ax.plot(x, dnm, "--", lw=1.2, color="#c1440e", alpha=0.65, label="DetecTree2, no canopy-ignore")

    ax.set_xlabel("mask IoU threshold"); ax.set_ylabel("AP")
    ax.set_title("OAM-TCD 439: AP across IoU thresholds", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels([f"{t:.2f}" for t in x], fontsize=8)
    ax.grid(alpha=0.25, lw=0.6); ax.set_ylim(0, None)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(a.out); fig.savefig(a.out.replace(".pdf", ".svg"))

    summ = {"label": "AP vs IoU threshold, 439 OAM-TCD",
            "sources": {"ours": os.path.relpath(OURS, io.REPO),
                        "dt2": os.path.relpath(DT2, io.REPO)},
            "iou_thresholds": [round(float(t), 2) for t in x],
            "ours_with_ignore_mean": [round(float(v), 4) for v in om],
            "ours_with_ignore_sd": [round(float(v), 4) for v in osd],
            "dt2_with_ignore": [round(float(v), 4) for v in dm],
            "ours_minus_dt2": [round(float(v), 4) for v in om - dm],
            "crossover_iou": next((round(float(t), 2) for t, g in zip(x, om - dm) if g <= 0), None),
            "note": ("published numbers are the canopy-ignore curves; the dashed no-ignore "
                     "curves show protocol sensitivity. DetecTree2 is a single seed.")}
    json.dump(summ, open(os.path.join(io.RESULTS, "per_iou.json"), "w"), indent=2)
    io.record_inputs([OURS, DT2])

    print(f"{'IoU':>6}{'ours':>9}{'sd':>8}{'dt2':>9}{'ours-dt2':>10}")
    for t, m, s_, dd in zip(x, om, osd, dm):
        print(f"{t:>6.2f}{m:>9.4f}{s_:>8.4f}{dd:>9.4f}{m-dd:>+10.4f}")
    print(f"\ncrossover (ours falls below dt2): IoU {summ['crossover_iou']}")
    print(f"-> {os.path.relpath(a.out, io.REPO)}")


if __name__ == "__main__":
    main()
