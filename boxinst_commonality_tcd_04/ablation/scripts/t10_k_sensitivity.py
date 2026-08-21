"""T0.10 -- Would a different K change anything? A free, parameter-space prediction.

An AP-level K sweep needs Modal (refit + eval). This asks the cheaper question that mostly
determines the answer: given the DEPLOYED prototypes, how much does the E-step's foreground
score actually move when K is reduced?

MECHANISM BEING TESTED
    estep computes  zc = kappa * (z @ C.T);  A = logsumexp(zc + log(pi/psum)) - bg_ll,
    then pfg = sigmoid(A + alpha*logit(psum)). If the prototypes are near-identical, the
    logsumexp over K collapses toward  max_k(...) + log(K_eff)  -- i.e. dropping prototypes
    shifts A by an almost CONSTANT offset. A constant offset in A is absorbed by the existing
    alpha / mask_thr knobs, which means K is inert up to a re-tune of levers you already have.
    The diagnostic is therefore not the size of the shift but its VARIABILITY: the standard
    deviation of (A_k - A_16) across probe directions is the part no knob can absorb.

PROBE DIRECTIONS
    Real unit vectors from the model itself: the 16 fg prototypes, the 12 background
    components, and interpolations between them. The fg->bg axis is where the decision
    boundary lives, so this is the segment that matters, not random 128-d directions (which
    are near-orthogonal to everything and would understate the effect).

WHAT THIS CANNOT DO
    Predict AP. Real cells are not exactly these directions, and the spatial prior pi and the
    bg log-likelihood both vary per cell. Treat the output as a strong prior on the sweep's
    outcome, not a substitute for it.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.ablation.scripts.t10_k_sensitivity
"""
import argparse
import json
import os

import numpy as np

from ..lib import io

DEPLOYED = os.path.join(io.PHASE4, "em_model_4p_fix.npz")


def logsumexp(a, axis=-1):
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def probes(C, Gbg, n_interp=9):
    """Unit directions spanning the fg->bg axis, built from the model's own vectors."""
    P = [C, Gbg]
    cbar = C.mean(0); cbar /= np.linalg.norm(cbar)
    for j in range(len(Gbg)):
        for t in np.linspace(0.1, 0.9, n_interp):
            v = (1 - t) * cbar + t * Gbg[j]
            P.append((v / np.linalg.norm(v))[None])
    Z = np.vstack(P)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def fg_score(Z, C, w, kappa):
    """logsumexp_k(kappa * z.Ck + log w_k) -- the fg term of A, up to the shared bg_ll."""
    return logsumexp(kappa * (Z @ C.T) + np.log(w + 1e-12)[None], axis=1)


def reduce_k(C, w, k, mode="mass"):
    """Reduce to k prototypes, renormalising weights (the M-step pins total fg mass anyway)."""
    if k >= len(C):
        return C, w
    if mode == "mass":
        idx = np.argsort(-w)[:k]
    else:                                   # spread: greedy farthest-point, the kindest case
        idx = [int(np.argmax(w))]
        while len(idx) < k:
            d = 1 - (C @ C[idx].T).max(1)
            d[idx] = -1
            idx.append(int(np.argmax(d)))
        idx = np.array(idx)
    Ck, wk = C[idx], w[idx]
    return Ck, wk / wk.sum()


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out", default=os.path.join(io.RESULTS, "k_sensitivity.json"))
    a = ap_.parse_args()

    d = np.load(DEPLOYED, allow_pickle=True)
    C, Gbg, pi = d["C"], d["Gbg"], d["pi"]
    kappa = float(d["kappa"]) * float(d["kappa_scale"])          # deployed effective kappa
    w = pi.sum(axis=(0, 2, 3)); w = w / w.sum()                  # per-prototype prior mass

    Z = probes(C, Gbg)
    ref = fg_score(Z, C, w, kappa)

    rows = []
    for mode in ("mass", "spread"):
        for k in (1, 2, 3, 4, 6, 8, 12, 16):
            Ck, wk = reduce_k(C, w, k, mode)
            s = fg_score(Z, Ck, wk, kappa)
            dlt = s - ref
            rows.append({"mode": mode, "k": k,
                         "mean_shift_nats": round(float(dlt.mean()), 4),
                         "sd_shift_nats": round(float(dlt.std(ddof=1)), 4),
                         "max_abs_residual_after_offset": round(
                             float(np.abs(dlt - dlt.mean()).max()), 4),
                         "equiv_sigmoid_shift_at_A0": round(
                             float(1 / (1 + np.exp(-dlt.mean())) - 0.5), 4)})

    base = [r for r in rows if r["mode"] == "mass"]
    k2 = next(r for r in base if r["k"] == 2)
    k1 = next(r for r in base if r["k"] == 1)

    out = {
        "label": "Does K matter? Parameter-space prediction from the deployed prototypes",
        "deployed": os.path.relpath(DEPLOYED, io.REPO),
        "fit_data": "120 4-phase L24 train tiles (fit.log); NO test data enters this analysis",
        "effective_kappa": kappa,
        "prototype_mass_gini": round(float(1 - (np.sort(w) * np.arange(1, len(w) + 1)).sum()
                                           * 2 / len(w) + 1 / len(w)), 4),
        "how_to_read": (
            "mean_shift is absorbable by the existing alpha / mask_thr knobs -- a constant "
            "offset in A just moves the sigmoid's operating point. sd_shift (and "
            "max_abs_residual_after_offset) is the part NO knob can absorb; that is the only "
            "quantity that can change the mask shape, and therefore AP."),
        "verdict_inputs": {"k1_sd": k1["sd_shift_nats"], "k2_sd": k2["sd_shift_nats"]},
        "rows": rows,
    }
    json.dump(out, open(a.out, "w"), indent=2)
    io.record_inputs([DEPLOYED])

    print(f"deployed effective kappa = {kappa:g}   (kappa {float(d['kappa']):g} x "
          f"kappa_scale {float(d['kappa_scale']):g})")
    print(f"prototype prior mass: min {w.min():.4f} max {w.max():.4f} "
          f"(uniform would be {1/len(w):.4f})")
    print(f"\n{'sel':<8}{'K':>4}{'mean shift':>12}{'sd shift':>11}{'max resid':>11}"
          f"{'sigmoid dP':>12}")
    for r in rows:
        print(f"{r['mode']:<8}{r['k']:>4}{r['mean_shift_nats']:>12.4f}"
              f"{r['sd_shift_nats']:>11.4f}{r['max_abs_residual_after_offset']:>11.4f}"
              f"{r['equiv_sigmoid_shift_at_A0']:>+12.4f}")
    print(f"\n-> {os.path.relpath(a.out, io.REPO)}")


if __name__ == "__main__":
    main()
