"""Fusion-independent diagnostics for the crown->shadow geometry cue.

The three-arm PR test showed CORRECT >> WRONG (signal is directional) but
CORRECT <= NONE (the penalty-only multiplicative fusion doesn't beat the raw
detector). Two questions remain, both independent of how you fuse the cue:

(A) DISCRIMINATION: does the signed contrast delta separate true crowns from
    (shadow) false positives at all? Measured as TP-vs-FP AUC of the crown-like
    score (-delta). NONE has no cue -> AUC=0.5 by construction. If correct gives
    AUC>0.5, shuffle ~0.5, flip <0.5, the geometry carries *usable, directional*
    discriminative information regardless of fusion -- which is exactly what a
    learned conditioning head would consume.

(B) FUSION SENSITIVITY: replace the penalty-only g with a symmetric log-linear
    re-score final = raw * exp(-lambda*delta) (can promote AND demote) and sweep
    lambda. If some lambda>0 lets CORRECT beat NONE while WRONG still collapses,
    the CORRECT<=NONE result was a fusion artifact, not absence of signal. Reported
    as a curve; any single lambda is post-hoc and labelled as such.
"""
from __future__ import annotations

import os, sys, json, pickle, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.geometry.gdata import load_records, cohort, load_rgb, valid_mask, luminance
from experiments.geometry.candidates import generate, match_to_gt
from experiments.geometry.gfilter import ShadowGeometry, standardize
from shadow_prior.geometry import azimuth_to_vector

SEED = 0
DCACHE = "experiments/geometry/_delta_cache.pkl"
OUT = "experiments/geometry/diagnostics.json"
OUTDIR = "claude_outputs/geometry"
CONDS = ["correct", "shuffle", "flip", "rot90"]


def build():
    recs = cohort(load_records())
    rng = np.random.default_rng(SEED)
    az = np.array([r.azimuth for r in recs]); dom = np.array([r.domain for r in recs])
    shuf = az.copy()
    for d in np.unique(dom):
        idx = np.flatnonzero(dom == d)
        shuf[idx] = az[idx][rng.permutation(idx.size)]
    rows = []
    for i, r in enumerate(recs):
        rgb = load_rgb(r.path); vm = valid_mask(rgb)
        cands = generate(rgb, vm)
        is_tp, _ = match_to_gt(cands, r.boxes)
        geo = ShadowGeometry(rgb, vm)
        az_c = {"correct": r.azimuth, "shuffle": float(shuf[i]),
                "flip": r.azimuth + np.pi, "rot90": r.azimuth + np.pi / 2}
        # centre-pixel darkness (confusable = dark candidate)
        z = standardize(luminance(rgb))
        for c, tp in zip(cands, is_tp):
            yy, xx = int(round(c.cy)), int(round(c.cx))
            zc = float(z[yy, xx]) if (0 <= yy < z.shape[0] and 0 <= xx < z.shape[1]) else 0.0
            rec = {"domain": r.domain, "scene": r.scene, "is_tp": bool(tp),
                   "raw": float(c.score), "zc": zc, "delta": {}, "ok": {}}
            for cond in CONDS:
                s = geo.score(c.cx, c.cy, az_c[cond])
                rec["delta"][cond] = s["delta"]; rec["ok"][cond] = s["ok"]
            rows.append(rec)
    return rows


def load_or_build():
    if os.path.exists(DCACHE):
        d = pickle.load(open(DCACHE, "rb"))
        if d.get("seed") == SEED:
            return d["rows"]
    rows = build()
    pickle.dump({"seed": SEED, "rows": rows}, open(DCACHE, "wb"))
    return rows


def auc(scores, labels):
    """Mann-Whitney AUC of `scores` for positive `labels` (bool)."""
    labels = np.asarray(labels); scores = np.asarray(scores)
    npos = labels.sum(); nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def pr_ap(final, is_tp, ngt):
    order = np.argsort(-final); tp = is_tp[order].astype(float)
    ctp = np.cumsum(tp); cfp = np.cumsum(1 - tp)
    rec = ctp / max(ngt, 1); pre = ctp / np.maximum(ctp + cfp, 1)
    mrec = np.concatenate([[0], rec, [rec[-1]]]); mpre = np.concatenate([[0], pre, [0]])
    for k in range(len(mpre) - 1, 0, -1):
        mpre[k - 1] = max(mpre[k - 1], mpre[k])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main():
    t0 = time.time()
    rows = load_or_build()
    doms = np.array([r["domain"] for r in rows])
    is_tp = np.array([r["is_tp"] for r in rows])
    raw = np.array([r["raw"] for r in rows])
    zc = np.array([r["zc"] for r in rows])
    n_all = len(rows)
    print(f"[diag] {n_all} candidates ({is_tp.sum()} TP)\n", flush=True)

    out = {"config": {"seed": SEED, "conds": CONDS}}

    # ---- (A) discrimination AUC (crown-like score = -delta), ok-only ----------
    print("=== (A) TP-vs-FP AUC of geometry cue (-delta); NONE=0.500 by construction ===", flush=True)
    out["auc"] = {}
    for scope, mask in [("ALL", np.ones(n_all, bool)),
                        ("WON", doms == "WON"), ("BRU", doms == "BRU"),
                        ("NEON", doms == "NEON"),
                        ("ALL_dark(zc<-0.3)", zc < -0.3)]:
        line = {}
        for cond in CONDS:
            delta = np.array([r["delta"][cond] for r in rows])
            ok = np.array([r["ok"][cond] for r in rows])
            sel = mask & ok & np.isfinite(delta)
            line[cond] = auc(-delta[sel], is_tp[sel])
        out["auc"][scope] = line
        print(f"  {scope:18} " + "  ".join(f"{c}={line[c]:.3f}" for c in CONDS)
              + f"   [n={int((mask).sum())}]", flush=True)

    # ---- (B) symmetric fusion lambda-sweep ------------------------------------
    print("\n=== (B) symmetric re-score final=raw*exp(-lambda*delta): AP vs lambda ===", flush=True)
    lambdas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    out["lambda_sweep"] = {}
    ngt_by = {"ALL": 1685, "WON": 1333, "BRU": 160, "NEON": 192}
    for scope in ("ALL", "WON", "BRU", "NEON"):
        mask = np.ones(n_all, bool) if scope == "ALL" else (doms == scope)
        ngt = ngt_by[scope]
        rows_s = [r for r, m in zip(rows, mask) if m]
        itp = is_tp[mask]; rw = raw[mask]
        d_cor = np.array([r["delta"]["correct"] if r["ok"]["correct"] else 0.0 for r in rows_s])
        d_flp = np.array([r["delta"]["flip"] if r["ok"]["flip"] else 0.0 for r in rows_s])
        curve = {"lambda": lambdas, "correct": [], "flip": [], "none": pr_ap(rw, itp, ngt)}
        for lam in lambdas:
            curve["correct"].append(pr_ap(rw * np.exp(-lam * d_cor), itp, ngt))
            curve["flip"].append(pr_ap(rw * np.exp(-lam * d_flp), itp, ngt))
        out["lambda_sweep"][scope] = curve
        best = max(range(len(lambdas)), key=lambda k: curve["correct"][k])
        print(f"  {scope:5} NONE AP={curve['none']:.3f} | "
              f"CORRECT AP@lambda: " + " ".join(f"{l}:{v:.3f}" for l, v in zip(lambdas, curve['correct']))
              + f"  (best lambda={lambdas[best]}, +{curve['correct'][best]-curve['none']:+.3f} vs NONE)",
              flush=True)
        print(f"        {'':0}      | FLIP    AP@lambda: "
              + " ".join(f"{l}:{v:.3f}" for l, v in zip(lambdas, curve['flip'])), flush=True)

    json.dump(out, open(OUT, "w"), indent=2)
    _plot(out)
    print(f"\n[done] {round(time.time()-t0,1)}s | {OUT}", flush=True)


def _plot(out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, scope in zip(axes, ("ALL", "WON", "BRU", "NEON")):
        c = out["lambda_sweep"][scope]
        ax.axhline(c["none"], color="black", ls="--", label=f"NONE ({c['none']:.3f})")
        ax.plot(c["lambda"], c["correct"], "-o", color="green", label="CORRECT")
        ax.plot(c["lambda"], c["flip"], "-o", color="red", label="FLIP")
        ax.set_title(scope); ax.set_xlabel("lambda (fusion strength)"); ax.set_ylabel("AP")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Symmetric fusion sweep: does correct-azimuth geometry beat NONE at some lambda?")
    fig.tight_layout()
    p = os.path.join(OUTDIR, "lambda_sweep.png")
    fig.savefig(p, dpi=90, bbox_inches="tight"); plt.close(fig)
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
