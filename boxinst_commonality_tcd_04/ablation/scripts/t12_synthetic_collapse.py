"""T0.12 -- Is prototype collapse a property of the UPDATE RULE, or of our data?

The deployed masker collapses (effective rank 1.66 of 16). That could be a fact about
DINOv3 crown features, about OAM-TCD, or about the update itself. This isolates the third
possibility by running the EXACT update on synthetic vectors with no imagery, no DINO, no
spatial prior and no background model:

    a     = softmax(kappa * (negZ @ C.T))        # negatives soft-assigned per prototype
    neg_k = normalise(sum_m a[m,k] negZ[m])      # that prototype's own negative centroid
    C_k   = normalise(pos_k - beta * neg_k)

`pos` and `negZ` are i.i.d. random unit vectors. If the prototypes collapse anyway, the
mechanism is the rule, not the domain.

Reported: effective rank = exp(entropy of the normalised singular-value spectrum of the
prototype matrix) -- the same statistic used on the fitted maskers, so numbers are directly
comparable to the real ones.

    .venv/bin/python -m boxinst_commonality_tcd_04.ablation.scripts.t12_synthetic_collapse
"""
import argparse
import json
import os

import numpy as np

from ..lib import io


def eff_rank(C):
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    sv = np.linalg.svd(Cn, compute_uv=False)
    ps = sv / max(sv.sum(), 1e-12)
    return float(np.exp(-(ps * np.log(ps + 1e-12)).sum()))


def mean_cos(C):
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    G = Cn @ Cn.T
    return float(G[~np.eye(len(C), dtype=bool)].mean())


def run(beta, conc=8.0, K=16, D=128, M=4096, kappa=16.0, iters=30, seed=0, track=False):
    """Iterate the real update on synthetic unit vectors.

    `conc` sets how strongly the vectors share a dominant direction. conc=0 is isotropic --
    in 128-d that makes them near-orthogonal, which is NOT what whitened ViT cells look like
    (the deployed masker's whitening leaves condition number 8.55 and participation ratio
    86/128, i.e. substantial residual anisotropy). Concentration turns out to be the
    condition that decides whether the update collapses; see main().
    """
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=D); mu /= np.linalg.norm(mu)
    def draw(n):
        v = conc * mu + rng.normal(size=(n, D))
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    C, N = draw(K), draw(M)
    hist = []
    for _ in range(iters):
        pos = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        if beta > 0:
            s = kappa * (N @ pos.T)                       # (M,K)
            s -= s.max(1, keepdims=True)
            a = np.exp(s); a /= a.sum(1, keepdims=True)
            neg = (a.T @ N) / (a.sum(0)[:, None] + 1e-8)
            neg /= (np.linalg.norm(neg, axis=1, keepdims=True) + 1e-12)
            C = pos - beta * neg
            C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        else:
            C = pos
        if track:
            hist.append(round(eff_rank(C), 3))
    return eff_rank(C), mean_cos(C), hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(io.RESULTS, "synthetic_collapse.json"))
    ap.add_argument("--conc", type=float, default=8.0)
    a = ap.parse_args()

    betas = [0.0, 0.25, 0.5, 0.75, 1.0]
    concs = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]
    grid = []
    for c in concs:
        row = {"concentration": c}
        for b in betas:
            r = [run(b, conc=c, seed=s)[0] for s in range(5)]
            row[f"beta_{b:g}"] = round(float(np.mean(r)), 3)
            row[f"beta_{b:g}_sd"] = round(float(np.std(r, ddof=1)), 3)
        grid.append(row)
    rows = [{"beta": b,
             "eff_rank_mean": grid[4][f"beta_{b:g}"],
             "eff_rank_sd": grid[4][f"beta_{b:g}_sd"]} for b in betas]   # conc=8 slice
    _, _, trace = run(0.5, conc=8.0, seed=0, track=True)

    print("K=16 synthetic unit vectors in R^128, 4096 negatives, kappa=16, 30 iters, 5 seeds")
    print("effective rank (16 = fully diverse, 1 = total collapse)\n")
    print(f"{'concentration':>14}" + "".join(f"{'b=' + f'{b:g}':>9}" for b in betas))
    for row in grid:
        print(f"{row['concentration']:>14.0f}"
              + "".join(f"{row[f'beta_{b:g}']:>9.2f}" for b in betas))
    print(f"\nbeta=0.5 (conc 8) rank by iteration: {trace[:10]} ...")
    print("\nreal fitted maskers:  beta=0 -> 11.43   beta=0.5 -> 1.66")

    json.dump({"label": "synthetic prototype collapse -- update rule in isolation",
               "setup": {"K": 16, "D": 128, "M_negatives": 4096, "kappa": 16.0,
                         "iters": 30, "seeds": 5,
                         "data": "i.i.d. random unit vectors; no imagery, no DINO"},
               "rows": rows, "grid_concentration_x_beta": grid, "beta05_rank_trace": trace,
               "real_reference": {"beta0_eff_rank": 11.43, "beta05_eff_rank": 1.66,
                                  "whitening_condition_number": 8.55,
                                  "whitening_participation_ratio": 86.02},
               "reading": (
                   "Collapse reproduces with no imagery and no DINO, so it is a property of the "
                   "update rule. The enabling condition is CONCENTRATION: once the vectors share "
                   "a dominant direction, every beta>0 drives the prototypes to effective rank "
                   "~1 while beta=0 stays diverse. Isotropic vectors (concentration 0) do NOT "
                   "collapse cleanly -- in 128-d they are near-orthogonal, an artefact rather "
                   "than a model of whitened ViT cells, whose residual anisotropy the deployed "
                   "masker measures at condition number 8.55 and participation ratio 86/128.")},
              open(a.out, "w"), indent=2)
    print(f"\n-> {os.path.relpath(a.out, io.REPO)}")


if __name__ == "__main__":
    main()
