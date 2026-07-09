"""Three-arm evaluation of the deterministic crown->shadow filter.

Arms (identical candidates, identical filter, only the azimuth scalar differs):
  NONE     -> g=1 for every candidate (raw candidate scores)
  CORRECT  -> filter with the true per-image shadow azimuth
  WRONG_*  -> filter with a corrupted azimuth:
                flip   = az + 180 deg  (points the search up-sun: clean for ALL sites)
                rot90  = az + 90 deg
                shuffle= within-acquisition azimuth permutation (weak for NEON,
                         whose true azimuths are clustered -- reported, not primary)

Because all arms re-score ONE candidate pool through a ZERO-parameter filter, the
CORRECT-vs-WRONG asymmetry cannot be manufactured by the candidate generator.

Decision rule: CORRECT > NONE (precision up / FP down at matched recall) AND
WRONG < NONE, with the CORRECT-minus-NONE gap clearing a scene-bootstrap CI.
Outputs: experiments/geometry/report.json + PR/stratification plots in
claude_outputs/geometry/.
"""
from __future__ import annotations

import os, sys, json, pickle, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.geometry.gdata import load_records, cohort, load_rgb, valid_mask
from experiments.geometry.candidates import generate, match_to_gt
from experiments.geometry.gfilter import ShadowGeometry

SEED = 0
CACHE = "experiments/geometry/_cand_cache.pkl"
REPORT = "experiments/geometry/report.json"
OUTDIR = "claude_outputs/geometry"
RECALL_TARGETS = [0.4, 0.5, 0.6, 0.7, 0.8]
REF_RECALL = 0.6                 # operating point for FP-removed / TP-removed accounting
N_BOOT = 1000
ARMS = ["NONE", "CORRECT", "WRONG_flip", "WRONG_rot90", "WRONG_shuffle"]


# --------------------------------------------------------------------------- #
# Build / cache the candidate table with per-arm geometry keep-probabilities
# --------------------------------------------------------------------------- #
def build_table(recs):
    rng = np.random.default_rng(SEED)
    az = np.array([r.azimuth for r in recs])
    dom = np.array([r.domain for r in recs])
    shuf = az.copy()
    for d in np.unique(dom):
        idx = np.flatnonzero(dom == d)
        shuf[idx] = az[idx][rng.permutation(idx.size)]

    rows = []  # each: dict with scene, domain, is_tp, size, raw, g[arm]
    n_gt = {}
    for i, r in enumerate(recs):
        rgb = load_rgb(r.path); vm = valid_mask(rgb)
        cands = generate(rgb, vm)
        is_tp, _ = match_to_gt(cands, r.boxes)
        n_gt[r.scene] = n_gt.get(r.scene, 0) + len(r.boxes)
        geo = ShadowGeometry(rgb, vm)
        az_arm = {"NONE": None, "CORRECT": r.azimuth,
                  "WRONG_flip": r.azimuth + np.pi,
                  "WRONG_rot90": r.azimuth + np.pi / 2,
                  "WRONG_shuffle": float(shuf[i])}
        for c, tp in zip(cands, is_tp):
            g = {}
            for arm in ARMS:
                if arm == "NONE":
                    g[arm] = 1.0
                else:
                    g[arm] = geo.score(c.cx, c.cy, az_arm[arm])["g"]
            rows.append({"scene": r.scene, "domain": r.domain, "is_tp": bool(tp),
                         "size": float(c.sigma), "raw": float(c.score), "g": g})
    return rows, n_gt


def load_or_build():
    recs = cohort(load_records())
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            d = pickle.load(f)
        if d.get("seed") == SEED and d.get("n_recs") == len(recs):
            return d["rows"], d["n_gt"]
    rows, n_gt = build_table(recs)
    with open(CACHE, "wb") as f:
        pickle.dump({"seed": SEED, "n_recs": len(recs), "rows": rows, "n_gt": n_gt}, f)
    return rows, n_gt


# --------------------------------------------------------------------------- #
# PR machinery (pooled over a set of candidates)
# --------------------------------------------------------------------------- #
def pr_curve(final_scores, is_tp, n_gt_total):
    order = np.argsort(-final_scores)
    tp = is_tp[order].astype(np.float64)
    ctp = np.cumsum(tp); cfp = np.cumsum(1 - tp)
    recall = ctp / max(n_gt_total, 1)
    precision = ctp / np.maximum(ctp + cfp, 1)
    return recall, precision, ctp, cfp


def prec_at_recall(recall, precision, R):
    """Actual operating point: first index reaching recall>=R; returns (precision,
    index) or (nan, None) if R unreachable."""
    idx = np.searchsorted(recall, R, side="left")
    if idx >= len(recall):
        return np.nan, None
    return float(precision[idx]), int(idx)


def average_precision(recall, precision):
    mrec = np.concatenate([[0], recall, [recall[-1]]])
    mpre = np.concatenate([[0], precision, [0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def arm_scores(rows, arm):
    return np.array([r["raw"] * r["g"][arm] for r in rows])


def eval_subset(rows, n_gt_total, arms=ARMS):
    is_tp = np.array([r["is_tp"] for r in rows])
    out = {}
    for arm in arms:
        fs = arm_scores(rows, arm)
        rec, prec, ctp, cfp = pr_curve(fs, is_tp, n_gt_total)
        pa = {f"P@R{R}": prec_at_recall(rec, prec, R)[0] for R in RECALL_TARGETS}
        fp = {}
        for R in RECALL_TARGETS:
            _, idx = prec_at_recall(rec, prec, R)
            fp[f"FP@R{R}"] = float(cfp[idx]) if idx is not None else np.nan
        out[arm] = {"AP": average_precision(rec, prec), **pa, **fp}
    return out


# --------------------------------------------------------------------------- #
# Scene bootstrap for the key deltas
# --------------------------------------------------------------------------- #
def scene_bootstrap(rows, n_gt, metric_fn, n=N_BOOT, seed=0):
    scenes = sorted(set(r["scene"] for r in rows))
    by_scene = {s: [] for s in scenes}
    for r in rows:
        by_scene[r["scene"]].append(r)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(len(scenes), len(scenes), replace=True)
        sub, ngt = [], 0
        for k in pick:
            s = scenes[k]
            sub.extend(by_scene[s]); ngt += n_gt.get(s, 0)
        vals.append(metric_fn(sub, ngt))
    vals = np.array([v for v in vals if np.isfinite(v)])
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def delta_prec_at_R(arm_a, arm_b, R):
    """metric = P@R(arm_a) - P@R(arm_b) on a candidate subset."""
    def f(sub, ngt):
        is_tp = np.array([r["is_tp"] for r in sub])
        ra, pa, *_ = pr_curve(arm_scores(sub, arm_a), is_tp, ngt)
        rb, pb, *_ = pr_curve(arm_scores(sub, arm_b), is_tp, ngt)
        va = prec_at_recall(ra, pa, R)[0]
        vb = prec_at_recall(rb, pb, R)[0]
        return va - vb
    return f


# --------------------------------------------------------------------------- #
# FP-removed / TP-removed accounting at the reference recall
# --------------------------------------------------------------------------- #
def selection_at_recall(rows, n_gt_total, arm, R):
    is_tp = np.array([r["is_tp"] for r in rows])
    fs = arm_scores(rows, arm)
    rec, prec, *_ = pr_curve(fs, is_tp, n_gt_total)
    _, idx = prec_at_recall(rec, prec, R)
    if idx is None:
        return None
    order = np.argsort(-fs)
    sel = np.zeros(len(rows), dtype=bool)
    sel[order[:idx + 1]] = True
    return sel, is_tp


def fp_tp_accounting(rows, n_gt_total, arm_ref, arm_new, R):
    a = selection_at_recall(rows, n_gt_total, arm_ref, R)
    b = selection_at_recall(rows, n_gt_total, arm_new, R)
    if a is None or b is None:
        return {}
    sel_ref, is_tp = a; sel_new, _ = b
    fp_removed = int(np.sum(sel_ref & ~sel_new & ~is_tp))
    tp_removed = int(np.sum(sel_ref & ~sel_new & is_tp))
    fp_added = int(np.sum(~sel_ref & sel_new & ~is_tp))
    tp_added = int(np.sum(~sel_ref & sel_new & is_tp))
    return {"ref_sel": int(sel_ref.sum()), "new_sel": int(sel_new.sum()),
            "fp_removed": fp_removed, "tp_removed": tp_removed,
            "fp_added": fp_added, "tp_added": tp_added}


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    rows, n_gt = load_or_build()
    n_gt_total = sum(n_gt.values())
    print(f"[data] {len(rows)} candidates | GT total={n_gt_total} | "
          f"pool recall(max)={np.mean([r['is_tp'] for r in rows])*len(rows)/n_gt_total:.2f}",
          flush=True)

    report = {"config": {"seed": SEED, "recall_targets": RECALL_TARGETS,
                         "ref_recall": REF_RECALL, "n_boot": N_BOOT, "arms": ARMS,
                         "filter": {"D_MIN": 15.0, "D_MAX": 60.0, "D_STEPS": 10,
                                    "K_GAIN": 2.0}},
              "n_candidates": len(rows), "n_gt_total": n_gt_total}

    # overall + per-domain metrics
    report["overall"] = eval_subset(rows, n_gt_total)
    report["per_domain"] = {}
    for d in ("WON", "BRU", "NEON"):
        sub = [r for r in rows if r["domain"] == d]
        ngt = sum(v for s, v in n_gt.items()
                  if any(rr["scene"] == s and rr["domain"] == d for rr in sub[:1]) or True)
        ngt = sum(n_gt[s] for s in set(r["scene"] for r in sub))
        report["per_domain"][d] = eval_subset(sub, ngt)

    # crown-size strata (small vs large by sigma) overall
    report["by_size"] = {}
    sizes = np.array([r["size"] for r in rows])
    med = float(np.median(sizes))
    for tag, mask in (("small", sizes <= med), ("large", sizes > med)):
        sub = [r for r, m in zip(rows, mask) if m]
        # n_gt for a size stratum is ill-defined; use recall vs *matched* TPs proxy:
        ngt = int(np.sum([r["is_tp"] for r in sub]))  # max recoverable in stratum
        report["by_size"][tag] = eval_subset(sub, max(ngt, 1))

    # bootstrap the key gaps at each recall target
    report["bootstrap"] = {}
    for R in RECALL_TARGETS:
        m_cn, lo_cn, hi_cn = scene_bootstrap(rows, n_gt, delta_prec_at_R("CORRECT", "NONE", R))
        m_nf, lo_nf, hi_nf = scene_bootstrap(rows, n_gt, delta_prec_at_R("NONE", "WRONG_flip", R))
        m_cf, lo_cf, hi_cf = scene_bootstrap(rows, n_gt, delta_prec_at_R("CORRECT", "WRONG_flip", R))
        report["bootstrap"][f"R{R}"] = {
            "CORRECT_minus_NONE": {"mean": m_cn, "ci": [lo_cn, hi_cn]},
            "NONE_minus_WRONGflip": {"mean": m_nf, "ci": [lo_nf, hi_nf]},
            "CORRECT_minus_WRONGflip": {"mean": m_cf, "ci": [lo_cf, hi_cf]},
        }

    # FP/TP accounting at ref recall
    report["accounting_at_refR"] = {
        "CORRECT_vs_NONE": fp_tp_accounting(rows, n_gt_total, "NONE", "CORRECT", REF_RECALL),
        "WRONGflip_vs_NONE": fp_tp_accounting(rows, n_gt_total, "NONE", "WRONG_flip", REF_RECALL),
    }

    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)

    _plots(rows, n_gt, n_gt_total)
    _print_summary(report)
    report["wall_clock_s"] = round(time.time() - t0, 1)
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] {report['wall_clock_s']}s | report: {REPORT}", flush=True)


def _plots(rows, n_gt, n_gt_total):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    is_tp = np.array([r["is_tp"] for r in rows])
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    panels = [("ALL", rows, n_gt_total)]
    for d in ("WON", "BRU", "NEON"):
        sub = [r for r in rows if r["domain"] == d]
        ngt = sum(n_gt[s] for s in set(r["scene"] for r in sub))
        panels.append((d, sub, ngt))
    colors = {"NONE": "black", "CORRECT": "green", "WRONG_flip": "red",
              "WRONG_rot90": "orange", "WRONG_shuffle": "purple"}
    for ax, (tag, sub, ngt) in zip(axes, panels):
        itp = np.array([r["is_tp"] for r in sub])
        for arm in ARMS:
            rec, prec, *_ = pr_curve(arm_scores(sub, arm), itp, ngt)
            ax.plot(rec, prec, color=colors[arm], lw=1.6,
                    label=f"{arm} (AP={average_precision(rec,prec):.3f})",
                    alpha=0.9 if arm in ("NONE", "CORRECT", "WRONG_flip") else 0.5)
        ax.set_title(f"{tag}  (n_gt={ngt})"); ax.set_xlabel("recall"); ax.set_ylabel("precision")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("Three-arm PR: deterministic crown->shadow filter (NONE / CORRECT / WRONG)")
    fig.tight_layout()
    p = os.path.join(OUTDIR, "pr_three_arm.png")
    fig.savefig(p, dpi=90, bbox_inches="tight"); plt.close(fig)
    print("wrote", p, flush=True)


def _print_summary(report):
    print("\n===== THREE-ARM SUMMARY =====", flush=True)
    ov = report["overall"]
    print(f"{'arm':14} {'AP':>6} " + " ".join(f"P@R{R:>3}" for R in RECALL_TARGETS), flush=True)
    for arm in ARMS:
        print(f"{arm:14} {ov[arm]['AP']:6.3f} " +
              " ".join(f"{ov[arm][f'P@R{R}']:6.3f}" if np.isfinite(ov[arm][f'P@R{R}']) else "   nan"
                       for R in RECALL_TARGETS), flush=True)
    print("\nper-domain AP:", flush=True)
    for d in ("WON", "BRU", "NEON"):
        print(f"  {d:5} " + " ".join(f"{arm}={report['per_domain'][d][arm]['AP']:.3f}"
                                      for arm in ("NONE", "CORRECT", "WRONG_flip")), flush=True)
    print("\nbootstrap gaps (scene-resampled 95% CI):", flush=True)
    for R in RECALL_TARGETS:
        b = report["bootstrap"][f"R{R}"]
        cn = b["CORRECT_minus_NONE"]; nf = b["NONE_minus_WRONGflip"]
        print(f"  R={R}: CORRECT-NONE={cn['mean']:+.3f} CI({cn['ci'][0]:+.3f},{cn['ci'][1]:+.3f})"
              f"  | NONE-WRONGflip={nf['mean']:+.3f} CI({nf['ci'][0]:+.3f},{nf['ci'][1]:+.3f})",
              flush=True)
    acc = report["accounting_at_refR"]["CORRECT_vs_NONE"]
    print(f"\naccounting @R={REF_RECALL} (CORRECT vs NONE): {acc}", flush=True)

    # verdict
    print("\n>>> VERDICT", flush=True)
    helped = hurt = 0; clears = 0
    for R in RECALL_TARGETS:
        b = report["bootstrap"][f"R{R}"]
        cn = b["CORRECT_minus_NONE"]; nf = b["NONE_minus_WRONGflip"]
        if cn["mean"] > 0: helped += 1
        if nf["mean"] > 0: hurt += 1
        if cn["ci"][0] > 0: clears += 1
    print(f"  CORRECT>NONE at {helped}/{len(RECALL_TARGETS)} recall points "
          f"({clears} with CI excluding 0); WRONGflip<NONE at {hurt}/{len(RECALL_TARGETS)}", flush=True)
    if clears >= 3 and hurt >= 3:
        print("  -> ASYMMETRY PRESENT: geometry is real and usable (greenlight learned conditioning)", flush=True)
    elif helped == 0 and hurt == 0:
        print("  -> NULL: no asymmetry; do NOT build conditioning", flush=True)
    else:
        print("  -> PARTIAL/AMBIGUOUS: inspect per-domain + CIs before deciding", flush=True)


if __name__ == "__main__":
    main()
