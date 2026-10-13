"""Box2Mask's 100-instance cap: AP50 with and without the tiles that contain an over-cap subtile.

Paper note d (Table 1) says removing every tile containing a 1024^2 subtile holding more than 100
ground-truth crowns improves Box2Mask Swin-L by +0.008 AP50 (512 raster, source not recorded).
This recomputes it with an explicit definition and two gates, then at 2048.

Definition: subtiles are the 3x3 grid of 1024^2 windows at origins {0, 512, 1024}^2 (the
prediction grid); a crown belongs to a subtile if its GT box centre lies inside it.
Gate 1: the count of over-cap test subtiles must equal PLAN.md's 46 of 3,951.
Gate 2: at 512 the subset delta must reproduce +0.008 (published preds file).

    .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.b2m_cap_subset
"""
import json
import os
import tempfile

import numpy as np

from boxinst_commonality_tcd_04 import score_coco as S

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
OUT = os.path.join(HERE, "b2m_cap_subset.json")
PREDS = {512: os.path.join(PKG, "box_supervised_baselines", "preds_box2mask_swin-l_s0_test.json"),
         2048: os.path.join(HERE, "preds_box2mask_swinl_s0_r2048.json")}
CAP = 100


def main():
    assert not os.path.exists(OUT), f"{OUT} exists -- refusing to overwrite"
    gt = json.load(open(os.path.join(PKG, "test_gt.json")))
    tids = sorted(gt)
    n_sub = n_over = 0
    flagged = []
    for t in tids:
        polys = gt[t]["trees"]
        if not polys:
            n_sub += 9
            continue
        c = np.array([np.asarray(p, float).reshape(-1, 2).min(0) / 2 + np.asarray(p, float).reshape(-1, 2).max(0) / 2
                      for p in polys])
        hit = False
        for y0 in (0, 512, 1024):
            for x0 in (0, 512, 1024):
                n_sub += 1
                k = int(((c[:, 0] >= x0) & (c[:, 0] < x0 + 1024) & (c[:, 1] >= y0) & (c[:, 1] < y0 + 1024)).sum())
                if k > CAP:
                    n_over += 1; hit = True
        if hit:
            flagged.append(t)
    print(f"gate1: over-cap subtiles {n_over} of {n_sub} (PLAN.md: 46 of 3,951); {len(flagged)} tiles flagged")
    out = {"definition": "1024^2 subtiles at origins {0,512,1024}^2; crown in subtile if GT box centre inside; cap 100",
           "n_subtiles": n_sub, "n_over_cap": n_over, "gate1_pass": (n_over, n_sub) == (46, 3951),
           "flagged_tiles": flagged, "res": {}}
    keep = [t for t in tids if t not in set(flagged)]
    for res, fp in PREDS.items():
        P = json.load(open(fp)); preds = P.get("preds", P)
        row = {}
        for name, ts in (("all", tids), ("without_flagged", keep)):
            sub = {t: gt[t] for t in ts}
            gt_fp, tid2img, tree_rles, canopy_union = S.build_gt(sub, res, with_canopy=False)
            gt_cr, _, _, _ = S.build_gt(sub, res, with_canopy=True)
            fd, tmp = tempfile.mkstemp(suffix=".json"); os.close(fd)
            json.dump({"meta": P.get("meta", {"mask_res": res}), "preds": {t: preds[t] for t in ts if t in preds}},
                      open(tmp, "w"))
            r = S._score_one(tmp, (gt_fp, gt_cr, tid2img, tree_rles, canopy_union, ts), res, S.MAX_DETS,
                             S.SCORE_FLOOR)["canopy_neutral_crowd"]["mask"]
            os.remove(tmp)
            row[name] = r
        row["delta_AP50"] = round(row["without_flagged"]["AP50"] - row["all"]["AP50"], 4)
        out["res"][str(res)] = row
        print(res, row["all"]["AP50"], row["without_flagged"]["AP50"], row["delta_AP50"], flush=True)
    out["gate2_pass"] = abs(out["res"]["512"]["delta_AP50"] - 0.008) <= 0.0005
    json.dump(out, open(OUT, "w"), indent=2)
    print("gate2 (512 delta == +0.008):", out["gate2_pass"], "->", OUT)


if __name__ == "__main__":
    main()
