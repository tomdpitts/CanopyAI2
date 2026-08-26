"""T1.4 -- Closure axis rebuilt on TOTAL tree cover (canopy UNION individual crowns).

t03's closure axis used `canopy_frac` from tile_index.json, which is the coverage of the
OAM-TCD canopy class ALONE. That mislabels a tile packed with individually-annotated
crowns and no canopy polygons as open: its crowns contribute nothing to the fraction.
Canopy closure should be the union of both annotated classes.

Coverage is recomputed here from test_gt.json at the E.RES scoring raster, as
    cover_frac = |union(canopy polys) U union(tree polys)| / RES^2
which is overlap-safe by construction (boolean OR of rasters), unlike canopy_frac +
crown_frac. Scoring is byte-identical to t03: same S.collect / S.pooled_ap, same seeds,
same pooled-within-bin convention, same CLOSURE_EDGES.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.ablation.scripts.t14_closure_cover
"""
import argparse
import json
import os

import numpy as np

from boxinst_commonality_tcd_04 import evaluate as E
from ..lib import gate, io
from ..lib import strata as S


def cover_fracs(gt):
    """{tid: (cover_frac, canopy_only_frac, crown_only_frac)} at the scoring raster."""
    out = {}
    for i, (tid, g) in enumerate(sorted(gt.items())):
        can = np.array(E.raster(g["canopy"]))
        tre = np.array(E.raster(g["trees"]))
        cu = can.any(0) if len(can) else np.zeros((E.RES, E.RES), bool)
        tu = tre.any(0) if len(tre) else np.zeros((E.RES, E.RES), bool)
        n = float(E.RES * E.RES)
        out[tid] = (float((cu | tu).sum()) / n, float(cu.sum()) / n, float(tu.sum()) / n)
        if i % 100 == 0:
            print(f"  cover {i}/{len(gt)}", flush=True)
    return out


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--seeds", default="0,1,2")
    ap_.add_argument("--out", default=os.path.join(io.RESULTS, "closure_cover.json"))
    a = ap_.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    gt = io.load_gt()
    print(f"[cover] rasterising {len(gt)} tiles at {E.RES}px", flush=True)
    cov = cover_fracs(gt)

    # old axis, for the side-by-side migration table
    meta = S.tile_meta(sorted(gt))
    assert sum(m["n_crowns"] for m in meta.values()) == 25705, "n_crowns != GT tree count"

    axes = {
        "closure_cover_frac": lambda t: cov[t][0],      # NEW: canopy U crowns
        "closure_canopy_frac": lambda t: meta[t]["canopy_frac"],  # OLD: canopy class only
    }
    groups = {}
    for name, key in axes.items():
        groups[name] = {S.bin_label(S.CLOSURE_EDGES, i, "{:g}"):
                        [t for t in sorted(gt) if S.bin_of(key(t), S.CLOSURE_EDGES) == i]
                        for i in range(len(S.CLOSURE_EDGES) - 1)}

    # how many tiles move, and where
    moves = {}
    for t in sorted(gt):
        o = S.bin_of(meta[t]["canopy_frac"], S.CLOSURE_EDGES)
        n = S.bin_of(cov[t][0], S.CLOSURE_EDGES)
        if o != n:
            moves.setdefault(f"{o}->{n}", []).append(t)

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
                "box_mAP50": io.band([per_seed[s][axis][lab]["box_mAP50"] for s in seeds]),
                "mask_minus_box_AP50": io.band(
                    [per_seed[s][axis][lab]["mask_minus_box_AP50"] for s in seeds]),
            }

    out = {"label": "Closure axis on total tree cover (canopy U crowns), 439 OAM-TCD, 3 seeds",
           "protocol": {
               "cover_frac": "|union(canopy) U union(trees)| / RES^2, RES=%d" % E.RES,
               "why": ("t03's canopy_frac counts the canopy CLASS only, so a tile dense with "
                       "individually-annotated crowns reads as open. Closure must be the union "
                       "of both classes."),
               "ap": "pooled within bin against that bin's own n_gt (identical to t03)",
               "closure_edges": S.CLOSURE_EDGES},
           "n_tiles_moved": sum(len(v) for v in moves.values()),
           "moves": {k: len(v) for k, v in sorted(moves.items())},
           "cover_summary": {
               "cover_frac_median": round(float(np.median([cov[t][0] for t in cov])), 4),
               "canopy_only_median": round(float(np.median([cov[t][1] for t in cov])), 4),
               "crown_only_median": round(float(np.median([cov[t][2] for t in cov])), 4)},
           "band_mean_std": bands, "per_seed": per_seed,
           "per_tile_cover": {t: round(cov[t][0], 5) for t in sorted(cov)}}
    json.dump(out, open(a.out, "w"), indent=2)
    io.record_inputs([io.GT, S.TILE_INDEX] + [io.knobbed(s) for s in seeds])

    print(f"\ntiles moving bin: {out['n_tiles_moved']}/439  {out['moves']}")
    for axis in groups:
        print(f"\n=== {axis} ===")
        print(f"{'bin':<20}{'tiles':>7}{'n_gt':>8}{'mask50':>9}{'sd':>8}{'box50':>9}{'m-b':>9}")
        for lab, b in bands[axis].items():
            print(f"{lab:<20}{b['n_tiles']:>7}{b['n_gt']:>8}"
                  f"{b['mask_mAP50'][0]:>9.4f}{b['mask_mAP50'][1]:>8.4f}"
                  f"{b['box_mAP50'][0]:>9.4f}{b['mask_minus_box_AP50'][0]:>9.4f}")


if __name__ == "__main__":
    main()
