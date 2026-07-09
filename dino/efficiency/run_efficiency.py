"""Label-efficiency curve for the shadow prior on frozen DINOv3 features.

One config -> efficiency curve over N train scenes x 3 rungs, repeated draws on a
fixed test pool within the EXPLORE split. Reports per-N Nadeau-Bengio R2-vs-R3 and
the seed/draw-noise MDE floor. Sweep.py calls this over the variant grid; the
CONFIRMATION split is never touched here.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from shadow_prior.config import ShadowFeatureConfig  # noqa: E402
from shadow_prior.shadow_feature import shuffle_azimuths  # noqa: E402
from shadow_prior.stats import corrected_resampled_ttest, seed_variance  # noqa: E402
from data import confirm_explore_split, load_cohort, subsample_scenes  # noqa: E402
from probe import (assemble, average_precision, build_scene_cache, fit_pca,  # noqa: E402
                   fit_ridge, load_base, probe_scores)


def az_dicts(recs, seed=0):
    scenes = [r.scene for r in recs]
    az = np.array([r.azimuth for r in recs])
    acq = np.array([r.acq for r in recs], dtype=object)
    shuf = shuffle_azimuths(az, acq, seed=seed)
    return ({s: float(a) for s, a in zip(scenes, az)},
            {s: float(a) for s, a in zip(scenes, shuf)})


def fixed_test_pool(recs, n_test, seed=7):
    """Stratified held-out test scenes within explore; rest is the train pool."""
    test = subsample_scenes(recs, n_test, seed=seed)
    tset = set(r.scene for r in test)
    train_pool = [r for r in recs if r.scene not in tset]
    return test, train_pool


def run(feat_tag, cfg, include_raw, label, n_grid, n_draws, n_test, out_path, pca_k=128, target="occ"):
    recs, _ = confirm_explore_split(load_cohort())     # EXPLORE only
    az_c, az_s = az_dicts(recs)
    test_recs, train_pool = fixed_test_pool(recs, n_test)

    base = load_base(recs, feat_tag, include_raw)
    pca = fit_pca(base, [r.scene for r in train_pool], k=pca_k)   # unsupervised, train pool
    sc = build_scene_cache(recs, base, pca, cfg, az_c, az_s, target=target)

    test_scenes = [r.scene for r in test_recs]
    Xte = {rg: assemble(sc, test_scenes, rg) for rg in ("r1", "r2", "r3")}

    curve = []
    for N in n_grid:
        if N > len(train_pool):
            continue
        ap = {"r1": [], "r2": [], "r3": []}
        for d in range(n_draws):
            tr = [r.scene for r in subsample_scenes(train_pool, N, seed=1000 + d)]
            for rg in ("r1", "r2", "r3"):
                Xtr, ytr = assemble(sc, tr, rg)
                pr = fit_ridge(Xtr, ytr)
                s = probe_scores(pr, Xte[rg][0])
                ap[rg].append(average_precision(s, Xte[rg][1]))
        ap = {k: np.array(v) for k, v in ap.items()}
        d23 = ap["r2"] - ap["r3"]
        d21 = ap["r2"] - ap["r1"]
        rec = {"N": N,
               "ap_mean": {k: float(v.mean()) for k, v in ap.items()},
               "ap_std": {k: float(v.std(ddof=1)) for k, v in ap.items()},
               "d23_mean": float(d23.mean()), "d21_mean": float(d21.mean())}
        try:
            nb = corrected_resampled_ttest(d23, n_train=N, n_test=n_test)
            rec["nb_23"] = nb.to_dict()
        except Exception as e:
            rec["nb_23_err"] = str(e)
        try:
            rec["mde"] = seed_variance(ap["r2"]).to_dict()
        except Exception as e:
            rec["mde_err"] = str(e)
        curve.append(rec)
        v = rec.get("nb_23", {})
        print(f"[{label}] N={N:3d} AP r1/r2/r3={rec['ap_mean']['r1']:.3f}/"
              f"{rec['ap_mean']['r2']:.3f}/{rec['ap_mean']['r3']:.3f}  "
              f"d23={rec['d23_mean']:+.4f} p={v.get('p_value', float('nan')):.3f} "
              f"MDE={rec.get('mde', {}).get('minimum_detectable_effect', float('nan')):.4f}", flush=True)

    out = {"label": label, "feat_tag": feat_tag, "include_raw": include_raw,
           "cfg": cfg.to_dict(), "n_grid": n_grid, "n_draws": n_draws,
           "n_test_scenes": n_test, "curve": curve}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="web")
    ap.add_argument("--label", default="primary_web")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--agg", default="max")
    ap.add_argument("--nch", type=int, default=1)
    ap.add_argument("--bright", default="luminance")
    ap.add_argument("--omin", type=float, default=2.0)
    ap.add_argument("--omax", type=float, default=20.0)
    ap.add_argument("--draws", type=int, default=25)
    ap.add_argument("--ntest", type=int, default=30)
    ap.add_argument("--grid", default="6,12,24,48")
    ap.add_argument("--target", default="occ", choices=["occ", "ctr"])
    a = ap.parse_args()
    cfg = ShadowFeatureConfig(offset_min=a.omin, offset_max=a.omax, offset_steps=8,
                              aggregation=a.agg, n_channels=a.nch, brightness_proxy=a.bright)
    n_grid = [int(x) for x in a.grid.split(",")]
    run(a.feat, cfg, a.raw, a.label, n_grid, a.draws, a.ntest,
        os.path.join(HERE, "artifacts", f"eff_{a.label}.json"), target=a.target)


if __name__ == "__main__":
    main()
