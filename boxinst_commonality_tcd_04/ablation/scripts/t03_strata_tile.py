"""T0.3 -- Where does LACE degrade? Tile-level stratification by canopy closure and density.

The method's premise is foreground/background COMMONALITY within a box: it assumes the box
is mostly one crown against a distinguishable background. That assumption should weaken as
canopy closes (no background left) and as crowns crowd (neighbours inside the box). This
measures both, on the deployed knobbed masker, 3 seeds.

Strata come from modal_sparse_tcd_multiseed/tile_index.json:
  canopy_frac  fraction of the tile under canopy  -> closure
  n_crowns     labelled crowns in the tile        -> density
Join verified: all 439 test tids resolve, all split=test/seen=False, no nulls, and
sum(n_crowns) == 25705 == the GT tree count.

AP is POOLED WITHIN each bin against that bin's own GT count -- the cut convention already
used by subsets_sparse_band.json -- so bins are internally normalised and comparable.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.ablation.scripts.t03_strata_tile
"""
import argparse
import json
import os

import numpy as np

from ..lib import gate, io
from ..lib import strata as S


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--seeds", default="0,1,2")
    ap_.add_argument("--out", default=os.path.join(io.RESULTS, "strata_tile.json"))
    a = ap_.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    gt = io.load_gt()
    meta = S.tile_meta(sorted(gt))
    assert len(meta) == len(gt), f"tile_index misses {len(gt) - len(meta)} test tids"
    assert all(m["split"] == "test" and not m["seen"] for m in meta.values())
    assert sum(m["n_crowns"] for m in meta.values()) == 25705, "n_crowns != GT tree count"

    axes = {
        "closure_canopy_frac": (S.CLOSURE_EDGES, lambda m: m["canopy_frac"], "{:g}"),
        "density_n_crowns": (S.DENSITY_EDGES, lambda m: m["n_crowns"], "{:g}"),
    }
    groups = {}
    for name, (edges, key, fmt) in axes.items():
        g = {}
        for i in range(len(edges) - 1):
            g[S.bin_label(edges, i, fmt)] = [t for t, m in meta.items()
                                             if S.bin_of(key(m), edges) == i]
        groups[name] = g
    # biome, for the cross-site read; only biomes with enough tiles to mean anything
    byb = {}
    for t, m in meta.items():
        byb.setdefault(f"{m['biome']} {m['biome_name']}", []).append(t)
    groups["biome"] = {k: v for k, v in sorted(byb.items()) if len(v) >= 15}

    per_seed = {}
    for s in seeds:
        print(f"[s{s}] scoring 439 tiles", flush=True)
        per = S.collect(io.knobbed(s), gt)
        overall = S.pooled_ap(per, sorted(gt))[0]
        gate.check({"mask_mAP50": round(overall, 4)}, io.ref_knobbed(s),
                   keys=("mask_mAP50",), label=f"s{s} pooled-all")
        res = {}
        for axis, g in groups.items():
            res[axis] = {}
            for lab, tids in g.items():
                ap50, n_gt = S.pooled_ap(per, tids, 0.5)
                bap50, _ = S.pooled_ap(per, tids, 0.5, key="box_iou")
                res[axis][lab] = {"n_tiles": len(tids), "n_gt": n_gt,
                                  "mask_mAP50": round(float(ap50), 4),
                                  "mask_mAP50_95": round(S.pooled_ap_5095(per, tids), 4),
                                  "box_mAP50": round(float(bap50), 4),
                                  "mask_minus_box_AP50": round(float(ap50 - bap50), 4)}
        per_seed[s] = res
        del per

    bands = {}
    for axis in groups:
        bands[axis] = {}
        for lab in groups[axis]:
            bands[axis][lab] = {
                "n_tiles": per_seed[seeds[0]][axis][lab]["n_tiles"],
                "n_gt": per_seed[seeds[0]][axis][lab]["n_gt"],
                "mask_mAP50": io.band([per_seed[s][axis][lab]["mask_mAP50"] for s in seeds]),
                "mask_mAP50_95": io.band([per_seed[s][axis][lab]["mask_mAP50_95"] for s in seeds]),
                "box_mAP50": io.band([per_seed[s][axis][lab]["box_mAP50"] for s in seeds]),
                "mask_minus_box_AP50": io.band(
                    [per_seed[s][axis][lab]["mask_minus_box_AP50"] for s in seeds]),
            }
    # monotonicity read: Spearman of bin midpoint rank vs AP, on the ordered axes
    trend = {}
    for axis in ("closure_canopy_frac", "density_n_crowns"):
        labs = list(groups[axis])
        y = np.array([bands[axis][l]["mask_mAP50"][0] for l in labs])
        x = np.arange(len(labs), dtype=float)
        ok = ~np.isnan(y)
        if ok.sum() > 2:
            xr, yr = x[ok], y[ok]
            trend[axis] = round(float(np.corrcoef(
                np.argsort(np.argsort(xr)), np.argsort(np.argsort(yr)))[0, 1]), 3)

    out = {"label": "Tile-level stratification, 439 OAM-TCD, knobbed masker, 3 seeds",
           "protocol": {
               "ap": "pooled within bin against that bin's own n_gt (subsets_sparse_band convention)",
               "source": "modal_sparse_tcd_multiseed/tile_index.json",
               "join_check": "439/439 tids, all split=test & seen=False, sum(n_crowns)=25705=n_gt",
               "closure_edges": S.CLOSURE_EDGES, "density_edges": S.DENSITY_EDGES,
               "biome_min_tiles": 15},
           "reading": (
               "AP is LOWER on sparse/open tiles, the opposite of the naive commonality "
               "prediction. That is mostly a DETECTION effect: box AP moves the same way, so "
               "the mask-minus-box column is the masker-specific quantity. A flat "
               "mask-minus-box across strata means the masker is not what degrades. Pooled AP "
               "in low-n_gt bins is also inherently unstable -- the [0,10) crowns/tile bin "
               "holds 147 tiles but only ~362 GT crowns, so a few false positives dominate it. "
               "The assumption-relevant test is the instance-level touching split in "
               "results/strata_instance.json, not this tile-level proxy."),
           "spearman_bin_rank_vs_ap50": trend,
           "band_mean_std": bands, "per_seed": per_seed}
    json.dump(out, open(a.out, "w"), indent=2)
    io.record_inputs([io.GT, S.TILE_INDEX] + [io.knobbed(s) for s in seeds])

    for axis in groups:
        print(f"\n=== {axis} ===")
        print(f"{'bin':<34}{'tiles':>7}{'n_gt':>8}{'mask50':>9}{'sd':>8}{'box50':>9}{'mask-box':>10}")
        for lab, b in bands[axis].items():
            print(f"{lab:<34}{b['n_tiles']:>7}{b['n_gt']:>8}"
                  f"{b['mask_mAP50'][0]:>9.4f}{b['mask_mAP50'][1]:>8.4f}"
                  f"{b['box_mAP50'][0]:>9.4f}{b['mask_minus_box_AP50'][0]:>+10.4f}")
        if axis in trend:
            print(f"  Spearman(bin rank, AP50) = {trend[axis]:+.3f}")
    print(f"\n-> {os.path.relpath(a.out, io.REPO)}")


if __name__ == "__main__":
    main()
