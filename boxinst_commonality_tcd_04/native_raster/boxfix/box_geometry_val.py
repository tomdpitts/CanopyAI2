"""Val diagnostic: geometry of detected boxes vs their box-matched GT crown (pairs from pair_iou_val_s0.json, product order).
Per pair: det/GT box width & height ratio, centre offset / GT size, fraction of GT crown pixels OUTSIDE the det box clip
(lost to the box clip in every render), fraction of det box covered by GT; plus deployed-render mask recall/precision.
-> boxfix/box_geometry_val_s0.json"""
import json, os
import numpy as np
from pycocotools import mask as mu
from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as G
from boxinst_commonality_tcd_04.native_raster.oracle_boxes_val import VAL_GT
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "box_geometry_val_s0.json"); assert not os.path.exists(OUT)
gt = json.load(open(VAL_GT)); PI = json.load(open(os.path.join(HERE, "pair_iou_val_s0.json")))["tiles"]
P = json.load(open(os.path.join(HERE, "preds_val_s0_r2048_deployed_product.json")))["preds"]
rows = []
for r in PI:
    t = r["tile"]; polys = gt[t]["trees"]; gb = G.gt_boxes_2048(polys); bx = np.asarray(P[t]["boxes_2048"]).reshape(-1, 4)
    for i, j, a in zip(r["det"], r["crown"], r["area"]):
        d, g = bx[i], gb[j]
        gr = mu.encode(np.asfortranarray(np.asarray(__import__("boxinst_commonality_tcd_04.evaluate", fromlist=["raster"]).raster([polys[j]], res=2048, scale=1.0)[0]).astype(np.uint8)))
        gm = mu.decode(gr).astype(bool)
        y0, y1, x0, x1 = max(0, int(d[1])), int(np.ceil(d[3])), max(0, int(d[0])), int(np.ceil(d[2]))
        inbox = gm[y0:y1, x0:x1].sum(); boxa = max((y1 - y0) * (x1 - x0), 1)
        pm = mu.decode({"size": [2048, 2048], "counts": P[t]["masks_rle"][i]["counts"].encode()}).astype(bool)
        inter = (pm & gm).sum()
        gw, gh = g[2] - g[0], g[3] - g[1]
        rows.append([a, (d[2] - d[0]) / max(gw, 1), (d[3] - d[1]) / max(gh, 1),
                     np.hypot((d[0] + d[2] - g[0] - g[2]) / 2, (d[1] + d[3] - g[1] - g[3]) / 2) / max(np.sqrt(gw * gh), 1),
                     1 - inbox / max(gm.sum(), 1), inbox / boxa, inter / max(gm.sum(), 1), inter / max(pm.sum(), 1)])
R = np.array(rows); E = (622.0, 2456.3); out = {"n": len(R), "cols": ["gt_area", "w_ratio", "h_ratio", "centre_off_rel", "gt_outside_box", "box_fill_by_gt", "mask_recall", "mask_precision"], "strata": {}}
for st, lo, hi in (("all", 0, 1e12), ("small", 0, E[0]), ("mid", E[0], E[1]), ("large", E[1], 1e12)):
    k = (R[:, 0] >= lo) & (R[:, 0] < hi); x = R[k]
    out["strata"][st] = {"n": int(k.sum()), **{c: {"median": round(float(np.median(x[:, n])), 3), "mean": round(float(x[:, n].mean()), 3)} for n, c in enumerate(out["cols"]) if n},
                         "frac_box_smaller_both": round(float(((x[:, 1] < 1) & (x[:, 2] < 1)).mean()), 3), "frac_box_larger_both": round(float(((x[:, 1] > 1) & (x[:, 2] > 1)).mean()), 3)}
json.dump(out, open(OUT, "w"), indent=1); print(json.dumps(out["strata"], indent=0))
