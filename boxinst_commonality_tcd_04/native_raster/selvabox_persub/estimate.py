"""Paired estimate on the 88-tile sample (every 5th test tile): same boxes and detector scores, masks
from (a) the seam-fixed whole-tile SAM run vs (b) per-subtile SAM (selvabox_persub_local.py). Ranking
by detector score and by sqrt(det x SAM). Projection = full-set seam-fixed AP + paired sample delta.
Writes estimate_sample88.json (refuses to overwrite).

  PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.selvabox_persub.estimate
"""
import json
import os
import tempfile

import numpy as np

from boxinst_commonality_tcd_04 import score_coco as S

HERE = os.path.dirname(os.path.abspath(__file__))
NR = os.path.dirname(HERE)
PKG = os.path.dirname(NR)
OUT = os.path.join(HERE, "estimate_sample88.json")


def score(preds, gt, tids, floor):
    sub = {t: gt[t] for t in tids}
    gt_fp, tid2img, tree_rles, cu = S.build_gt(sub, 2048, with_canopy=False)
    gt_cr, _, _, _ = S.build_gt(sub, 2048, with_canopy=True)
    fd, tmp = tempfile.mkstemp(suffix=".json"); os.close(fd)
    json.dump({"meta": {"mask_res": 2048}, "preds": preds}, open(tmp, "w"))
    c = S._score_one(tmp, (gt_fp, gt_cr, tid2img, tree_rles, cu, tids), 2048, S.MAX_DETS, floor)
    os.remove(tmp)
    c = c["canopy_neutral_crowd"]
    return {"AP50": c["mask"]["AP50"], "AP50_95": c["mask"]["AP50_95"], "box_AP50": c["box"]["AP50"]}


def gmean(p):
    det = np.asarray(p["scores"], float); sam = np.clip(np.asarray(p["sam_scores"], float), 0, None)
    return dict(p, scores=np.round(np.sqrt(det * sam), 6).tolist())


def main():
    assert not os.path.exists(OUT), OUT
    gt = json.load(open(os.path.join(PKG, "test_gt.json")))
    tids = sorted(gt)[::5]
    whole = json.load(open(os.path.join(NR, "seamfix", "preds_selvabox_seamfix_on_r2048.json")))["preds"]
    per = {}
    for t in tids:
        r = json.load(open(os.path.join(HERE, "tiles_mps", t + ".json")))
        assert r["boxes_2048"] == whole[t]["boxes_2048"] and r["scores"] == whole[t]["scores"], t
        per[t] = {k: r[k] for k in ("boxes_2048", "scores", "sam_scores", "masks_rle")}
    W = {t: {k: whole[t][k] for k in ("boxes_2048", "scores", "sam_scores", "masks_rle")} for t in tids}
    res = {"n_tiles": len(tids)}
    res["whole_det"] = score(W, gt, tids, 0.05)
    res["persub_det"] = score(per, gt, tids, 0.05)
    res["whole_gmean"] = score({t: gmean(p) for t, p in W.items()}, gt, tids, 0.0)
    res["persub_gmean"] = score({t: gmean(p) for t, p in per.items()}, gt, tids, 0.0)
    full = {"det": (0.5984, 0.2207), "gmean": (0.6141, 0.2502)}   # seam-fixed, all 439, maxDets 600
    for k in ("det", "gmean"):
        d50 = res[f"persub_{k}"]["AP50"] - res[f"whole_{k}"]["AP50"]
        d95 = res[f"persub_{k}"]["AP50_95"] - res[f"whole_{k}"]["AP50_95"]
        res[f"projected_full_{k}"] = {"AP50": round(full[k][0] + d50, 4), "AP50_95": round(full[k][1] + d95, 4),
                                      "delta_AP50_sample": round(d50, 4), "delta_AP50_95_sample": round(d95, 4)}
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
