"""PR-curve figure (Step-4 deliverable): our detector's precision-recall curve at
IoU 0.4 on the 194 NEON tiles, with DeepForest's curve + operating point and the
Weinstein 2021 Table 3 paper point overlaid. Isolated output -> modal_neon_multiseed/.

Usage: .venv_df/bin/python make_pr_plot.py [results_neon_s0.json]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.abspath(os.path.dirname(__file__))
PAPER_P, PAPER_R = 0.659, 0.790          # Table 3 image-annotated RGB @IoU0.4
DF_OP_P, DF_OP_R = 0.617, 0.765          # our DeepForest reproduction @thr0.10


def curve_xy(pr_curve):
    """PR points sorted by recall for a clean line (precision vs recall)."""
    pts = sorted(((p["mean_recall"], p["mean_precision"]) for p in pr_curve))
    return [r for r, _ in pts], [p for _, p in pts]


def main():
    res_fp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, "results_neon_s0.json")
    res = json.load(open(res_fp))
    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    # ours
    r, p = curve_xy(res["pr_curve"])
    ax.plot(r, p, "-", color="#2a6f97", lw=2.2, label="Ours (DINOv3-web, box-only)")
    bf = res["best_f1_point"]
    ax.plot(bf["R"], bf["P"], "o", color="#2a6f97", ms=9,
            label=f"Ours best-F1 (P{bf['P']:.2f}/R{bf['R']:.2f})")

    # DeepForest reproduction curve (if available)
    df_fp = os.path.join(HERE, "deepforest_repro.json")
    if os.path.exists(df_fp):
        df = json.load(open(df_fp))
        if "pr_curve" in df:
            dr, dp = curve_xy(df["pr_curve"])
            ax.plot(dr, dp, "--", color="#e07a5f", lw=1.8,
                    label="DeepForest (our repro)")
    ax.plot(DF_OP_R, DF_OP_P, "s", color="#e07a5f", ms=9,
            label=f"DeepForest @0.10 (P{DF_OP_P:.2f}/R{DF_OP_R:.2f})")

    # paper Table 3 point
    ax.plot(PAPER_R, PAPER_P, "*", color="#d62828", ms=18,
            label=f"Weinstein 2021 Table 3 (P{PAPER_P:.3f}/R{PAPER_R:.3f})")
    ax.axvline(PAPER_R, color="#d62828", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(PAPER_P, color="#d62828", ls=":", lw=0.8, alpha=0.5)

    ax.set_xlabel("Recall (macro over 194 images)")
    ax.set_ylabel("Precision (macro over 194 images)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("NEON image-annotated crowns, RGB-only, IoU 0.4\n"
                 "box-to-box: ours vs DeepForest vs paper")
    ax.grid(alpha=0.25); ax.legend(loc="lower left", fontsize=8)
    out = os.path.join(HERE, "pr_curve_neon.png")
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
