"""Confirmation: run ONE pre-specified config on the locked CONFIRMATION scenes.

All config selection happened on EXPLORE (sweep.py). This touches the 40 held-out
scenes for the first time and reports the R2-vs-R3 effect with Nadeau-Bengio +
sign-flip permutation + the seed/draw MDE floor, overall and split dryland
(WON+BRU) vs temperate (NEON). Only this result is "the effect".
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from shadow_prior.config import ShadowFeatureConfig  # noqa: E402
from shadow_prior.stats import (corrected_resampled_ttest, permutation_test_paired,  # noqa: E402
                                seed_variance)
from data import confirm_explore_split, load_cohort, subsample_scenes  # noqa: E402
from probe import (assemble, average_precision, build_scene_cache, fit_pca,  # noqa: E402
                   fit_ridge, load_base, probe_scores)
from run_efficiency import az_dicts, fixed_test_pool  # noqa: E402


def evaluate(recs, feat, cfg, n_grid, n_draws, n_test, tag):
    az_c, az_s = az_dicts(recs)
    test_recs, train_pool = fixed_test_pool(recs, n_test, seed=11)
    base = load_base(recs, feat, False)
    pca = fit_pca(base, [r.scene for r in train_pool], k=min(128, base[recs[0].scene].shape[1]))
    sc = build_scene_cache(recs, base, pca, cfg, az_c, az_s, target="ctr")
    test_scenes = [r.scene for r in test_recs]
    Xte = {rg: assemble(sc, test_scenes, rg) for rg in ("r1", "r2", "r3")}

    res = []
    for N in n_grid:
        if N > len(train_pool):
            continue
        ap = {"r1": [], "r2": [], "r3": []}
        for d in range(n_draws):
            tr = [r.scene for r in subsample_scenes(train_pool, N, seed=2000 + d)]
            for rg in ("r1", "r2", "r3"):
                Xtr, ytr = assemble(sc, tr, rg)
                ap[rg].append(average_precision(probe_scores(fit_ridge(Xtr, ytr), Xte[rg][0]), Xte[rg][1]))
        ap = {k: np.array(v) for k, v in ap.items()}
        d23 = ap["r2"] - ap["r3"]
        rec = {"N": N, "tag": tag,
               "ap_mean": {k: float(v.mean()) for k, v in ap.items()},
               "d23_mean": float(d23.mean()),
               "nb_23": corrected_resampled_ttest(d23, N, n_test).to_dict(),
               "perm_23": permutation_test_paired(d23).to_dict(),
               "mde": seed_variance(ap["r2"]).to_dict()}
        res.append(rec)
        nb = rec["nb_23"]; pm = rec["perm_23"]
        print(f"[{tag}] N={N:3d} AP r1/r2/r3={rec['ap_mean']['r1']:.3f}/{rec['ap_mean']['r2']:.3f}/"
              f"{rec['ap_mean']['r3']:.3f} d23={rec['d23_mean']:+.4f} "
              f"CI=({nb['ci_low']:+.4f},{nb['ci_high']:+.4f}) p_nb={nb['p_value']:.4f} "
              f"p_perm={pm['p_value']:.4f} MDE={rec['mde']['minimum_detectable_effect']:.4f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="raw")
    ap.add_argument("--omin", type=float, default=2.0)
    ap.add_argument("--omax", type=float, default=20.0)
    ap.add_argument("--nch", type=int, default=1)
    ap.add_argument("--draws", type=int, default=25)
    a = ap.parse_args()
    cfg = ShadowFeatureConfig(offset_min=a.omin, offset_max=a.omax, offset_steps=8,
                              aggregation="max", n_channels=a.nch, brightness_proxy="greenness")
    _, confirm = confirm_explore_split(load_cohort())
    dryland = [r for r in confirm if r.acq in ("WON", "BRU")]
    print(f"CONFIRM scenes={len(confirm)} (dryland WON+BRU={len(dryland)})  feat={a.feat} cfg=greenness "
          f"offset={a.omin}-{a.omax} nch={a.nch}\n")
    out = {"feat": a.feat, "cfg": cfg.to_dict(),
           "all": evaluate(confirm, a.feat, cfg, [6, 12, 20], a.draws, 12, "all"),
           "dryland": evaluate(dryland, a.feat, cfg, [6, 12], a.draws, 8, "dryland")}
    json.dump(out, open(os.path.join(HERE, "artifacts", f"confirm_{a.feat}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
