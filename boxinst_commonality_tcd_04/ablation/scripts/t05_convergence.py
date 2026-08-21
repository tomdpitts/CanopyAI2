"""T0.5 -- EM convergence and degeneracy diagnostics for the DEPLOYED masker.

Two things a careful reviewer will ask, and the honest answers:

1. CONVERGENCE. There is no convergence criterion in the code. Both fit loops are
   `for it in range(args.iters)` with no tolerance and no early break
   (boxinst_commonality/em.py:403, boxinst_commonality_tcd_04/em.py:245); iters=30 fixed.
   The only convergence evidence is the per-iteration mean P(fg) trace. This script
   computes a POST-HOC convergence statistic from the traces that exist.

2. DEGENERACY. Pruning is by responsibility share (em.py:431-438) and is therefore blind
   to REDUNDANCY: prototypes that duplicate each other all keep healthy shares and none is
   pruned. masker_lab measured pairwise prototype cosine 0.964 at 8px/beta=0.5. This
   script measures it directly on the deployed prototypes.

PROVENANCE PROBLEM, STATED PLAINLY
   The deployed model is `em_model_4p_fix.npz`. No fit log or fit report for it survives
   in phase4/: `em_fit_report_4p.json` and `fit.log` describe `em_model_4p.npz`, a
   DIFFERENT fit (every learned array differs; the fix model is a genuine refit, and is
   the older file of the two). So no per-iteration trace exists for the shipped model.
   What this script does: derive every STATIC diagnostic directly from the deployed npz,
   and report the surviving traces as same-recipe lineage evidence, explicitly labelled as
   not the deployed fit. It does not invent a trace.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.ablation.scripts.t05_convergence
"""
import argparse
import json
import os
import re

import numpy as np

from ..lib import io

PHASE4 = io.PHASE4
DEPLOYED = os.path.join(PHASE4, "em_model_4p_fix.npz")


def parse_trace(log_path):
    """Per-iteration mean P(fg) from a fit log ('  it 5: mean P(fg)=0.677')."""
    if not os.path.exists(log_path):
        return None
    txt = open(log_path, errors="ignore").read()
    it = [(int(a), float(b))
          for a, b in re.findall(r"it\s*(\d+):\s*mean P\(fg\)=([\d.]+)", txt)]
    return sorted(it) or None


def convergence_stats(trace):
    """Post-hoc: has the trace stopped moving by the time the fixed loop ends?"""
    if not trace or len(trace) < 2:
        return None
    it, v = np.array([t[0] for t in trace], float), np.array([t[1] for t in trace], float)
    d = np.abs(np.diff(v))
    first_below = None
    for k in range(len(d)):
        if np.all(d[k:] < 1e-3):
            first_below = int(it[k + 1]); break
    return {"iters_logged": [int(x) for x in it],
            "fg_mass": [round(float(x), 4) for x in v],
            "final_fg_mass": round(float(v[-1]), 4),
            "abs_delta_last_step": round(float(d[-1]), 5),
            "total_drift_after_it10": round(float(abs(v[-1] - v[it >= 10][0])), 5)
            if np.any(it >= 10) else None,
            "converged_by_iter_tol1e-3": first_below,
            "note": "logged every 5th iteration; 'converged' = all later steps move <1e-3"}


def prototype_stats(C, label):
    """Redundancy among prototypes. Pruning cannot see this -- it prunes on responsibility
    share, so duplicated-but-busy prototypes survive."""
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    G = Cn @ Cn.T
    iu = np.triu_indices(len(C), 1)
    pc = G[iu]
    # Effective rank of the prototype set: exp(entropy of the normalised singular-value
    # spectrum). K counts rows; this counts DIRECTIONS. If the contrastive update subtracts
    # a near-common background centroid from every prototype, the set becomes rank-1 and
    # this lands near 1 regardless of K.
    sv = np.linalg.svd(Cn, compute_uv=False)
    ps = sv / max(sv.sum(), 1e-12)
    eff_rank = float(np.exp(-(ps * np.log(ps + 1e-12)).sum()))
    return {"K": int(len(C)),
            "effective_rank": round(eff_rank, 3),
            "top_singular_frac": round(float(sv[0] / max(sv.sum(), 1e-12)), 4),
            "pairwise_cos_mean": round(float(pc.mean()), 4),
            "pairwise_cos_max": round(float(pc.max()), 4),
            "pairwise_cos_p90": round(float(np.percentile(pc, 90)), 4),
            "frac_pairs_over_0.9": round(float((pc > 0.9).mean()), 4),
            "frac_pairs_over_0.99": round(float((pc > 0.99).mean()), 4),
            "n_pairs": int(len(pc)),
            "was_L2_normalised": bool(np.allclose(np.linalg.norm(C, axis=1), 1, atol=1e-6)),
            "label": label}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out", default=os.path.join(io.RESULTS, "convergence.json"))
    ap_.add_argument("--report-out",
                     default=os.path.join(io.RESULTS, "em_fit_report_4p_fix.json"))
    a = ap_.parse_args()

    d = np.load(DEPLOYED, allow_pickle=True)
    C, Gbg, wbg, pi = d["C"], d["Gbg"], d["wbg"], d["pi"]

    proto = prototype_stats(C, "deployed em_model_4p_fix.npz (beta=0.5)")
    bg = prototype_stats(Gbg, "deployed background mixture (frozen after init)")

    # Is the collapse specific to the deployed fit, or systematic? Measure every model in
    # phase4/ under the identical statistic (which is also masker_lab/sweep.py:66's).
    cohort = {}
    for fn in sorted(os.listdir(PHASE4)):
        if fn.startswith("em_model") and fn.endswith(".npz"):
            try:
                cohort[fn] = prototype_stats(np.load(os.path.join(PHASE4, fn),
                                                     allow_pickle=True)["C"], fn)
            except Exception as e:
                cohort[fn] = {"error": str(e)}

    # whitening spectrum: how much work is whitening actually doing?
    sc = np.asarray(d["scale"], float)
    whiten = {"pca_dim": int(len(sc)),
              "singular_value_scale_max": round(float(sc.max()), 5),
              "singular_value_scale_min": round(float(sc.min()), 5),
              "condition_number": round(float(sc.max() / max(sc.min(), 1e-12)), 2),
              "participation_ratio": round(float(sc.sum() ** 2 / (sc ** 2).sum()), 2),
              "note": ("condition number is the anisotropy whitening removes: 1.0 would mean "
                       "whitening is a no-op. participation ratio is the effective number of "
                       "directions out of pca_dim.")}

    # The M-step pins `pi[s].sum(0).mean()` -- the per-(u,v)-bin total fg mass, averaged
    # over bins -- to FG_PRIOR_MASS = pi/4 (em.py:422). Summing over bins as well would
    # multiply by the 64 bins and mean nothing; the pinned quantity is the one below.
    fg_per_band = pi.sum(axis=1).mean(axis=(1, 2))
    fg_mass = float(fg_per_band.mean())
    FG_PRIOR_MASS = float(np.pi / 4)
    traces = {}
    for lab, log in (("beta0.5 lineage (em_model_4p.npz -- NOT the deployed fit)", "fit.log"),
                     ("beta0 lineage (em_model_4p_b0.npz)", "fit_b0.log")):
        t = parse_trace(os.path.join(PHASE4, log))
        if t:
            traces[lab] = {"log": log, **convergence_stats(t)}

    out = {
        "label": "EM convergence + degeneracy diagnostics, deployed TCD masker",
        "deployed_model": os.path.relpath(DEPLOYED, io.REPO),
        "provenance_gap": (
            "No fit log or fit report survives for em_model_4p_fix.npz. em_fit_report_4p.json "
            "and fit.log describe em_model_4p.npz, a DIFFERENT fit -- every learned array "
            "differs between the two files, so the fix model is a genuine refit rather than a "
            "field rewrite, and it is the OLDER of the two on disk. The traces below are "
            "same-recipe lineage evidence, NOT the deployed fit's own trace."),
        "no_convergence_criterion": (
            "The fit loop is `for it in range(args.iters)` with no tolerance and no early "
            "break (boxinst_commonality_tcd_04/em.py:245); iters=30 is fixed a priori. "
            "Convergence below is assessed post hoc from the logged fg-mass trace."),
        "static_from_deployed_npz": {
            "fg_prototypes": proto,
            "bg_mixture": bg,
            "bg_weights": {"n": int(len(wbg)),
                           "min": round(float(wbg.min()), 4),
                           "max": round(float(wbg.max()), 4),
                           "entropy_nats": round(float(-(wbg * np.log(wbg + 1e-12)).sum()), 4),
                           "max_entropy_nats": round(float(np.log(len(wbg))), 4)},
            "spatial_prior": {
                "shape": list(pi.shape), "axes": "(size band, K, u bin, v bin)",
                "mean_fg_mass_per_bin": round(fg_mass, 4),
                "per_size_band": [round(float(x), 4) for x in fg_per_band],
                "pinned_target_FG_PRIOR_MASS": round(FG_PRIOR_MASS, 4),
                "shortfall_vs_target": round(FG_PRIOR_MASS - fg_mass, 4),
                "bin_max": round(float(pi.max()), 4),
                "note": ("each M-step pins pi[s].sum(0).mean() to pi/4 (em.py:422), then caps "
                         "any bin at BIN_CAP=0.95 (em.py:423-426); the cap is why the realised "
                         "mass sits slightly below the target")},
            "whitening": whiten,
            "knobs": {k: float(d[k]) for k in ("prior_weight", "kappa_scale", "kappa",
                                               "cell_origin", "s_px") if k in d.files},
        },
        "prototype_collapse_cohort": cohort,
        "collapse_reading": (
            "Pruning keys on responsibility SHARE (em.py:431-438), which is blind to "
            "redundancy: near-duplicate prototypes each keep a healthy share, so none is "
            "pruned and K stays at its initial value. The reported K is therefore nominal, "
            "not an effective count. This also predicts a k_init sweep would be nearly flat, "
            "and it does not by itself imply worse masks -- the mask comes from the fg-vs-bg "
            "log-ratio, so a degenerate fg mixture still yields a usable posterior."),
        "traces_same_recipe_not_deployed": traces,
    }
    json.dump(out, open(a.out, "w"), indent=2)

    # the reconstruction, written HERE (never into phase4/)
    rep = {"RECONSTRUCTED": True,
           "what": ("Static fit report for the deployed em_model_4p_fix.npz, derived from the "
                    "npz itself by ablation/scripts/t05_convergence.py. The original fit's "
                    "report and log are absent from phase4/; this is NOT a recovered file and "
                    "contains no per-iteration trace, because none survives."),
           "model": os.path.relpath(DEPLOYED, io.REPO),
           "k_effective": proto["K"], "k_bg": bg["K"],
           "pca": whiten["pca_dim"], "whiten": True,
           "kappa": float(d["kappa"]), "s_px": int(d["s_px"]),
           "cell_origin_px": float(d["cell_origin"]),
           "prior_weight": float(d["prior_weight"]), "kappa_scale": float(d["kappa_scale"]),
           "size_edges_px": [round(float(x), 2) for x in d["size_edges"]],
           "bins": int(pi.shape[-1]), "size_bands": int(pi.shape[0]),
           "mean_fg_mass_per_bin": round(fg_mass, 4),
           "prototype_pairwise_cos_mean": proto["pairwise_cos_mean"]}
    json.dump(rep, open(a.report_out, "w"), indent=2)
    io.record_inputs([DEPLOYED, os.path.join(PHASE4, "fit.log"),
                      os.path.join(PHASE4, "fit_b0.log")])

    print("=== deployed masker, static diagnostics ===")
    print(f"  K (fg prototypes)        {proto['K']}   L2-normalised: {proto['was_L2_normalised']}")
    print(f"  pairwise cosine          mean {proto['pairwise_cos_mean']}  "
          f"max {proto['pairwise_cos_max']}  p90 {proto['pairwise_cos_p90']}")
    print(f"  redundant pairs          {100*proto['frac_pairs_over_0.9']:.1f}% >0.9   "
          f"{100*proto['frac_pairs_over_0.99']:.1f}% >0.99   (n={proto['n_pairs']})")
    print(f"  bg mixture K={bg['K']}  weight entropy {out['static_from_deployed_npz']['bg_weights']['entropy_nats']}"
          f" / {out['static_from_deployed_npz']['bg_weights']['max_entropy_nats']} nats")
    print(f"  whitening condition no.  {whiten['condition_number']}  "
          f"(participation ratio {whiten['participation_ratio']}/{whiten['pca_dim']})")
    print(f"  fg mass per bin          {fg_mass:.4f}  (pinned target pi/4 = {FG_PRIOR_MASS:.4f})")
    print("\n=== prototype collapse across every model in phase4/ ===")
    print(f"  {'model':30s}{'K':>4}{'cos mean':>10}{'cos max':>9}{'>0.9':>8}{'>0.99':>8}{'eff rank':>9}")
    for fn, c in cohort.items():
        if "error" in c:
            print(f"  {fn:30s}  ERROR {c['error'][:40]}"); continue
        print(f"  {fn:30s}{c['K']:>4}{c['pairwise_cos_mean']:>10.4f}"
              f"{c['pairwise_cos_max']:>9.4f}{100*c['frac_pairs_over_0.9']:>7.0f}%"
              f"{100*c['frac_pairs_over_0.99']:>7.0f}%{c['effective_rank']:>9.2f}")
    print("\n=== convergence (same recipe, NOT the deployed fit) ===")
    for lab, t in traces.items():
        print(f"  {lab}")
        print(f"    fg mass {t['fg_mass']}  final {t['final_fg_mass']}")
        print(f"    |delta| last step {t['abs_delta_last_step']}   "
              f"converged by it{t['converged_by_iter_tol1e-3']} (tol 1e-3)")
    print(f"\n-> {os.path.relpath(a.out, io.REPO)}")
    print(f"-> {os.path.relpath(a.report_out, io.REPO)}")


if __name__ == "__main__":
    main()
