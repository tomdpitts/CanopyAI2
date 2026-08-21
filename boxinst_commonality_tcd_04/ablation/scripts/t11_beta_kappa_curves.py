"""T0.11 -- The collapse/accuracy figure: AP50 against beta, kappa, and effective rank.

THE RESULT THIS FIGURE EXISTS TO SHOW
    Prototype collapse is not simply good or bad. Its sign depends on BOX PRECISION.
      GT boxes      (masker_lab 16px, 4 betas):  rank 13.0 -> 1.7,  AP50 0.969 -> 0.771  (HURTS)
      predicted box (deployed 8px, 2 betas):     rank 10.8 -> 1.7,  AP50 0.579 -> 0.620  (HELPS)
    Reading: the collapsed prototype is a single "not-background" direction, which is robust
    to a sloppy box that admits extra background. Diverse prototypes are free to match that
    background instead, so they only win when the box is already tight.

CONFOUND, STATED UP FRONT
    The GT-box arm is 16px/local and the predicted-box arm is 8px/Modal, so stride and stack
    differ alongside box precision. This figure DOCUMENTS the sign flip; it does not isolate
    its cause. Breaking that confound is the proposed experiment (see README).

SOURCES -- all pre-existing, nothing recomputed here except effective rank:
    masker_lab/sweep_results.json + em_L24_16px_b*.npz      (GT-box beta sweep)
    phase4/results_b{0,05}_fix_thr025_439.json + models     (predicted-box, 439)
    phase4/sweep_val_knobs_s0.json                          (alpha x kappa, 108 val tiles)

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.ablation.scripts.t11_beta_kappa_curves
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..lib import io

LAB = io.MASKER_LAB
BLUE, RED, GREY = "#1b6ca8", "#c1440e", "#666666"


def eff_rank(C):
    Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
    sv = np.linalg.svd(Cn, compute_uv=False)
    ps = sv / sv.sum()
    return float(np.exp(-(ps * np.log(ps + 1e-12)).sum()))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out", default=os.path.join(io.FIGURES, "collapse_beta_kappa.pdf"))
    a = ap_.parse_args()

    # --- GT-box beta sweep (16px lab) -------------------------------------------------
    lab = json.load(open(os.path.join(LAB, "sweep_results.json")))
    gt_rows = []
    for r in lab:
        if not r["variant"].startswith("L24-16px"):
            continue
        b = float(r["variant"].split("b=")[1])
        # the sweep wrote b0.0 / b0.1 / b0.25 / b0.5 -- %g would render 0.0 as "0"
        cands = [f"em_L24_16px_b{b}.npz", "em_L24_16px_b%g.npz" % b]
        f = next(os.path.join(LAB, c) for c in cands if os.path.exists(os.path.join(LAB, c)))
        C = np.load(f)["C"]
        gt_rows.append({"beta": b, "K": r["K"], "eff_rank": round(eff_rank(C), 3),
                        "proto_cos": round(r["proto_cos"], 4),
                        "mask_ap50": r["mask_ap50"], "mean_iou": r["mean_iou"]})
    gt_rows.sort(key=lambda r: r["beta"])

    # --- predicted-box arm (8px deployed, 439) -----------------------------------------
    pr_rows = []
    for b, mdl, res in ((0.0, "em_model_4p_b0_fix.npz", "results_b0_fix_thr025_439.json"),
                        (0.5, "em_model_4p_fix.npz", "results_b05_fix_thr025_439.json")):
        C = np.load(os.path.join(io.PHASE4, mdl), allow_pickle=True)["C"]
        d = json.load(open(os.path.join(io.PHASE4, res)))
        pr_rows.append({"beta": b, "K": len(C), "eff_rank": round(eff_rank(C), 3),
                        "mask_ap50": d["mask_mAP50"], "mask_ap50_95": d["mask_mAP50_95"]})

    # --- alpha x kappa surface (108 val tiles, deployed masker) -------------------------
    sv = json.load(open(os.path.join(io.PHASE4, "sweep_val_knobs_s0.json")))
    g = [c for c in sv["grid"] if abs(c["mask_thr"] - 0.25) < 1e-9]
    al = sorted({c["prior_weight"] for c in g}); ks = sorted({c["kappa_scale"] for c in g})
    M = np.full((len(al), len(ks)), np.nan)
    for c in g:
        M[al.index(c["prior_weight"]), ks.index(c["kappa_scale"])] = c["mask_mAP50"]

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))

    # (a) AP50 vs beta, both regimes, effective rank on the twin axis
    ax = axes[0]
    ax.plot([r["beta"] for r in gt_rows], [r["mask_ap50"] for r in gt_rows], "-o", ms=4,
            color=BLUE, label="GT boxes (16 px)")
    ax.plot([r["beta"] for r in pr_rows], [r["mask_ap50"] for r in pr_rows], "-s", ms=5,
            color=RED, label="predicted boxes (8 px)")
    ax.set_xlabel(r"contrastive repel $\beta$"); ax.set_ylabel("mask AP50")
    ax.set_title("(a) beta vs accuracy, two\nNON-COMPARABLE configurations", fontsize=9)
    ax.grid(alpha=.25, lw=.6); ax.legend(fontsize=7, frameon=False, loc="center left")
    ax2 = ax.twinx()
    ax2.plot([r["beta"] for r in gt_rows], [r["eff_rank"] for r in gt_rows], ":", lw=1.2,
             color=GREY)
    ax2.plot([r["beta"] for r in pr_rows], [r["eff_rank"] for r in pr_rows], ":", lw=1.2,
             color=GREY)
    ax2.set_ylabel("effective rank (dotted)", fontsize=8, color=GREY)
    ax2.tick_params(axis="y", labelsize=7, colors=GREY)

    # (b) the mediator: AP50 against effective rank directly
    ax = axes[1]
    ax.plot([r["eff_rank"] for r in gt_rows], [r["mask_ap50"] for r in gt_rows], "-o", ms=4,
            color=BLUE)
    ax.plot([r["eff_rank"] for r in pr_rows], [r["mask_ap50"] for r in pr_rows], "-s", ms=5,
            color=RED)
    for r in gt_rows + pr_rows:
        ax.annotate(r"$\beta$=%g" % r["beta"], (r["eff_rank"], r["mask_ap50"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points", color=GREY)
    ax.axvline(1.0, color=GREY, lw=.6, ls="--")
    ax.set_xlabel("effective rank of FG prototypes"); ax.set_ylabel("mask AP50")
    ax.set_title("(b) rank vs accuracy\n(confounded: stride + box source)", fontsize=9)
    ax.grid(alpha=.25, lw=.6)

    # (c) alpha x kappa, inference-time, deployed masker
    ax = axes[2]
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(ks))); ax.set_xticklabels([f"{k:g}" for k in ks], fontsize=7)
    ax.set_yticks(range(len(al))); ax.set_yticklabels([f"{x:g}" for x in al], fontsize=7)
    ax.set_xlabel(r"$\kappa$ scale"); ax.set_ylabel(r"$\alpha$ (prior weight)")
    ax.set_title("(c) inference knobs, 108 val tiles\n(deployed masker, mask AP50)", fontsize=9)
    j, i = np.unravel_index(np.nanargmax(M), M.shape)
    ax.plot(i, j, "r*", ms=11)
    for p in range(M.shape[0]):
        for q in range(M.shape[1]):
            if not np.isnan(M[p, q]):
                ax.text(q, p, f"{M[p,q]:.3f}", ha="center", va="center", fontsize=5.4,
                        color="w")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04).ax.tick_params(labelsize=6)

    fig.tight_layout()
    fig.savefig(a.out); fig.savefig(a.out.replace(".pdf", ".svg"))

    out = {"label": "Collapse vs accuracy vs box precision; alpha/kappa surface",
           "SUPERSEDED": {
               "status": "HEADLINE RETRACTED -- the data below is valid, the conclusion was not.",
               "retracted_claim": ("The sign of the collapse effect flips with box precision; "
                                   "collapse buys box-robustness and only pays when boxes are "
                                   "imprecise."),
               "why": ("This file inferred a sign flip by comparing a 16px/local/GT-box arm with "
                       "an 8px/Modal/predicted-box arm, confounding box precision with stride and "
                       "stack. phase4_collapse Stage 2 tested it at ONE configuration and found NO "
                       "flip: beta=0 -> beta=0.5 raises mean crown IoU by +0.0483 with oracle boxes "
                       "and +0.0476 with predicted boxes, agreeing to 0.0007."),
               "superseded_by": [
                   "modal_tcd_multiseed/phase4_collapse/results/stage2_gate.json",
                   "modal_tcd_multiseed/phase4_collapse/results/factorial_summary.json"],
               "still_usable": ("gt_box_16px is the only multi-beta evidence in the project and its "
                                "monotone rank-vs-beta trend is real, but it is a 16px local GT-box "
                                "arm whose AP levels are not comparable to the deployed 8px numbers. "
                                "alpha_kappa_val and predicted_box_8px are unaffected."),
           },
           "headline": "RETRACTED -- see SUPERSEDED. Retained for provenance only.",
           "confound": (
               "The GT-box arm is 16px/local, the predicted-box arm is 8px/Modal. Stride and "
               "stack vary alongside box precision, so this documents the flip without "
               "isolating its cause. Also: the GT-box AP50 values (~0.77-0.97) are not "
               "comparable in level to the predicted-box ones (~0.58-0.62) -- perfect boxes "
               "inflate AP. Only the SLOPES are comparable."),
           "gt_box_16px": gt_rows, "predicted_box_8px": pr_rows,
           "alpha_kappa_val": {"alpha": al, "kappa": ks,
                               "mask_ap50": [[None if np.isnan(v) else round(float(v), 4)
                                              for v in row] for row in M],
                               "argmax": {"alpha": al[j], "kappa": ks[i],
                                          "mask_ap50": round(float(M[j, i]), 4)}}}
    json.dump(out, open(os.path.join(io.RESULTS, "collapse_beta_kappa.json"), "w"), indent=2)
    io.record_inputs([os.path.join(LAB, "sweep_results.json"),
                      os.path.join(io.PHASE4, "sweep_val_knobs_s0.json")])

    print(f"{'regime':<22}{'beta':>6}{'K':>4}{'eff rank':>10}{'AP50':>9}")
    for r in gt_rows:
        print(f"{'GT box 16px':<22}{r['beta']:>6.2f}{r['K']:>4}{r['eff_rank']:>10.2f}"
              f"{r['mask_ap50']:>9.4f}")
    for r in pr_rows:
        print(f"{'predicted box 8px':<22}{r['beta']:>6.2f}{r['K']:>4}{r['eff_rank']:>10.2f}"
              f"{r['mask_ap50']:>9.4f}")
    print(f"\nalpha/kappa val argmax: a={al[j]:g} k={ks[i]:g} -> {M[j,i]:.4f}")
    print(f"-> {os.path.relpath(a.out, io.REPO)} (+ .svg)")


if __name__ == "__main__":
    main()
