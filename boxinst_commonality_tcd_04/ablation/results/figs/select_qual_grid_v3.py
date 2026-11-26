"""Re-pick of Figure 5 (fig:qualgrid) with open-ground crops only (2026-11-25, Tom: "we need images without (or very little)
closed canopy"). Supersedes select_qual_grid_v2.py's band picks 363/336/56.

Why: v2 stratified by TILE-level canopy_frac, so even its "open" row (tile 363) showed a large canopy-class patch in the
auto-crop; the "intermediate"/"closed" rows are mostly canopy wash. The paper's setting is open ground, so the figure is not.

Rule v3 (fixed before any new tile was viewed): candidates, per-tile LACE - Restor AP50 margin, auto-crop and
display-eligibility (>= 10 GT crowns in the crop, LACE in-crop F1 >= 0.8 at its best-F1 display threshold) are exactly
v2's, read from qualitative_grid_select_tcd439_v2.json. A candidate is OPEN-CROP if the OAM-TCD canopy class covers
< MAX_CROP_CANOPY of its 512 px auto-crop. Canopy-density bands are dropped. Three rows are picked from the open-crop
tiles at the 25th, 50th and 75th percentile of the margin over ALL open-crop candidates (eligible or not): for each
quantile, the display-eligible open-crop tile with margin closest to it, no tile used twice. The failure row is v1/v2's
tile 341 (5th-percentile margin over all candidates), unchanged and exempt from eligibility, labelled as a failure.
The eligibility filter conditions on LACE's own in-crop F1; the caption must say so.

Usage: .venv/bin/python boxinst_commonality_tcd_04/ablation/results/figs/select_qual_grid_v3.py
Writes qualitative_grid_select_tcd439_v3.json next to this script; progress to stdout.
"""
import json, os, sys
import numpy as np
from pycocotools import mask as mu

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from select_qual_tile import rles  # noqa: E402
from select_qual_grid import GT  # noqa: E402

MAX_CROP_CANOPY, QS, CROP = 0.02, (0.25, 0.50, 0.75), 512


def main():
    v2 = json.load(open(os.path.join(HERE, "qualitative_grid_select_tcd439_v2.json")))
    gt = json.load(open(GT)); rows = v2["all_rows"]
    for n, r in enumerate(rows):
        cn = rles(gt[r["tile"]]["canopy"]); x0, y0 = r["crop_xy"]
        r["crop_canopy_frac"] = round(float(mu.decode(mu.merge(cn, intersect=False))[y0:y0 + CROP, x0:x0 + CROP].mean()), 4) if cn else 0.0
        r["open_crop"] = r["crop_canopy_frac"] < MAX_CROP_CANOPY
        if (n + 1) % 50 == 0:
            print(f"canopy-in-crop {n + 1}/{len(rows)}", flush=True)
    op = [r for r in rows if r["open_crop"]]; el = [r for r in op if r["eligible"]]
    m = np.array([r["margin"] for r in op])
    print(f"open-crop {len(op)}/{len(rows)}, display-eligible {len(el)}; LACE ahead on {float((m > 0).mean()):.0%} of open-crop", flush=True)
    out = dict(rule=__doc__.strip().split("\n\n")[2], max_crop_canopy=MAX_CROP_CANOPY, n_candidates=len(rows),
               n_open_crop=len(op), n_open_crop_eligible=len(el), frac_lace_ahead_open_crop=round(float((m > 0).mean()), 4),
               median_margin_open_crop=round(float(np.median(m)), 4), rows={})
    used = set()
    for q in QS:
        tq = float(np.quantile(m, q)); ranked = sorted([r for r in el if r["tile"] not in used], key=lambda r: abs(r["margin"] - tq))
        used.add(ranked[0]["tile"])
        out["rows"][f"q{int(q * 100)}"] = dict(quantile=q, margin_at_q=round(tq, 4), chosen=ranked[0], next3=ranked[1:4])
        print(f"q{q:.2f} margin {tq:+.4f} -> {ranked[0]['tile']} margin {ranked[0]['margin']:+.4f} "
              f"crop canopy {ranked[0]['crop_canopy_frac']:.3f} F1 {ranked[0]['crop_f1']} crowns {ranked[0]['crop_crowns']} {ranked[0]['biome']}", flush=True)
    f = next(r for r in rows if r["tile"] == v2["failure"]["chosen"]["tile"])
    out["failure"] = dict(v2["failure"], chosen=f)
    print(f"failure {f['tile']} crop canopy {f['crop_canopy_frac']:.3f}", flush=True)
    out["all_rows"] = rows
    json.dump(out, open(os.path.join(HERE, "qualitative_grid_select_tcd439_v3.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
