"""Val diagnostic: per-crown precision (inter/pred) and recall (inter/GT) of one render variant vs SAM IoU, by area stratum.
  PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.boxfix.diag_prec_rec VARIANT_JSON OUT.json"""
import json, os, sys, multiprocessing as mp
import numpy as np
from boxinst_commonality_tcd_04.native_raster.boxfix import render_lib as R
from boxinst_commonality_tcd_04.native_raster.boxfix.compare import sam_iou, EDGES
V = json.loads(sys.argv[1])

def _t(fp):
    t = os.path.basename(fp)[:-4]; s = sam_iou("val", t); out = []
    for i, c in enumerate(R.load_tile(fp)):
        if not c["valid"]: continue
        m = R.render(c, V); inter = float((m & c["gt"]).sum()); pa = float(m.sum())
        x0, y0, x1, y1 = c["box"]; boxa = (x1 - x0) * (y1 - y0)
        out.append([c["gt_area"], inter / max(pa, 1), inter / c["gt_area"], inter / (pa + c["gt_area"] - inter) if pa + c["gt_area"] else 0, s[i], c["gt_area"] / max(boxa, 1), pa / max(boxa, 1)])
    return out

if __name__ == "__main__":
    assert not os.path.exists(sys.argv[2])
    with mp.get_context("fork").Pool(8) as p: rows = np.array([r for t in p.map(_t, R.tile_files("val")) for r in t])
    res = {}
    for st, lo, hi in (("small", 0, EDGES[0]), ("mid", EDGES[0], EDGES[1]), ("large", EDGES[1], 1e12)):
        k = (rows[:, 0] >= lo) & (rows[:, 0] < hi); x = rows[k]; lose = x[:, 4] - x[:, 3] > 0.1
        res[st] = {"n": int(k.sum()), "prec": round(x[:, 1].mean(), 4), "rec": round(x[:, 2].mean(), 4), "iou": round(x[:, 3].mean(), 4), "sam": round(x[:, 4].mean(), 4),
                   "gt_fill": round(x[:, 5].mean(), 4), "pred_fill": round(x[:, 6].mean(), 4), "n_lose>0.1": int(lose.sum()),
                   "lose_prec": round(x[lose, 1].mean(), 4) if lose.any() else None, "lose_rec": round(x[lose, 2].mean(), 4) if lose.any() else None,
                   "lose_gt_fill": round(x[lose, 5].mean(), 4) if lose.any() else None, "lose_pred_fill": round(x[lose, 6].mean(), 4) if lose.any() else None}
    json.dump({"variant": V, "strata": res}, open(sys.argv[2], "w"), indent=1); print(json.dumps(res, indent=1))
