"""Rule-based tile choice for the proposed main-text qualitative comparison grid (2026-11-13, Tom: CVPR analysis item 3).

Rule (fixed before any tile was viewed): candidates are OAM-TCD 439 holdout tiles with >= MIN_CROWNS GT crowns, excluding
tile 122 (already Figure 1). Tiles are stratified by labelled-canopy fraction (tile_index.json canopy_frac, the
Table S7 / tab:strata variable) into three bands: open [0, 0.10), intermediate [0.10, 0.50), closed [0.50, 1]. Per-tile mask
AP50 (select_qual_tile.ap50: COCO 101-point, greedy score-ordered at IoU >= 0.5, predictions > 50 % in canopy ignored) is
computed for LACE and Restor (the headline mask-supervised competitor, as Figure 1). In each band the chosen tile is the
one whose LACE - Restor margin is closest to that band's median margin. A fourth "failure" row is the tile at the
FAIL_Q quantile of the margin over all candidates (LACE well behind Restor), chosen to show LACE's typical failure, not
its single worst tile. Crops are the script's auto-crop (most GT centroids in a 512 px window), never by eye.

Usage: .venv/bin/python boxinst_commonality_tcd_04/ablation/results/figs/select_qual_grid.py
Writes qualitative_grid_select_tcd439.json next to this script.
"""
import json, os, sys
import numpy as np
from pycocotools import mask as mu

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from select_qual_tile import ap50, rles, ROOT, MODELS  # noqa: E402

MIN_CROWNS, FAIL_Q, EXCLUDE = 10, 0.05, {"tcd_val_tile_122"}
BANDS = [("open", 0.0, 0.10), ("intermediate", 0.10, 0.50), ("closed", 0.50, 1.01)]
GT = os.path.join(ROOT, "test_gt.json"); TI = os.path.join(ROOT, "modal_sparse_tcd_multiseed", "tile_index.json")


def main():
    gt = json.load(open(GT)); ti = {x["tid"]: x for x in json.load(open(TI))["tiles"] if x["split"] == "test"}
    assert set(ti) == set(gt), "tile_index / test_gt mismatch"
    preds = {k: json.load(open(v))["preds"] for k, v in MODELS.items()}
    rows = []
    for t, g in gt.items():
        tr = rles(g["trees"])
        if t in EXCLUDE or len(tr) < MIN_CROWNS:
            continue
        cn = rles(g["canopy"]); canopy = mu.merge(cn, intersect=False) if cn else None
        a = {k: ap50(tr, canopy, preds[k][t]["masks_rle"], preds[k][t]["scores"]) for k in MODELS}
        rows.append(dict(tile=t, biome=ti[t]["biome_name"], canopy_frac=ti[t]["canopy_frac"], crowns=len(tr),
                         ap50_lace=round(a["LACE"], 4), ap50_restor=round(a["Restor"], 4), margin=round(a["LACE"] - a["Restor"], 4)))
    out = dict(rule=__doc__.strip().split("\n\n")[1], n_candidates=len(rows), bands={}, models=MODELS)
    for name, lo, hi in BANDS:
        c = [r for r in rows if lo <= r["canopy_frac"] < hi]; m = np.array([r["margin"] for r in c]); med = float(np.median(m))
        c = sorted(c, key=lambda r: abs(r["margin"] - med))
        out["bands"][name] = dict(edges=[lo, hi], n=len(c), median_margin=round(med, 4), frac_lace_ahead=round(float((m > 0).mean()), 4),
                                  chosen=c[0], next3=c[1:4])
        print(f"{name:13s} n={len(c):3d} median margin {med:+.4f} LACE ahead {float((m > 0).mean()):.0%} -> {c[0]}")
    m = np.array([r["margin"] for r in rows]); q = float(np.quantile(m, FAIL_Q))
    f = sorted(rows, key=lambda r: abs(r["margin"] - q))
    out["failure"] = dict(quantile=FAIL_Q, margin_at_q=round(q, 4), chosen=f[0], next3=f[1:4])
    print(f"failure q={FAIL_Q} margin {q:+.4f} -> {f[0]}")
    json.dump(out, open(os.path.join(HERE, "qualitative_grid_select_tcd439.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
