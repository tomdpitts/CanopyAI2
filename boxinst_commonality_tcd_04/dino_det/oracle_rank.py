"""Ranking ceiling for LACE (D1 variant), test 439 (2026-11-03). Diagnostic only, never a table row.

The detection set and masks are frozen (preds_dino_swinl_s0_test_sprime_r2048.json). Only the scores change:
  oracle_tp : per-tile Hungarian det<->tree assignment on mask IoU >= 0.5. Assigned dets get 2 + IoU,
              all others s'/10. That puts every TP above every FP across all tiles, so it gives the AP50 ceiling for this set.
  oracle_iou: score = max mask IoU with any tree, so duplicates are NOT penalised. This is the ceiling for a perfect
              per-detection quality head with no NMS.
Recall ceiling R50 = assigned / n_gt. Under 101-point interpolation with perfect ranking, AP50 ~= R50.
Scored with the frozen score_coco internals (2048, maxDets 600, floor 0, crowd arm). The GT is rasterised once.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.oracle_rank
"""
import json, os, time
import numpy as np
from pycocotools import mask as maskUtils
from scipy.optimize import linear_sum_assignment

from boxinst_commonality_tcd_04 import score_coco as SC

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); REPO = os.path.dirname(PKG)
OUT = os.path.join(HERE, "results"); TAG = "dino_swinl_s0"
SRC = os.path.join(OUT, f"preds_{TAG}_test_sprime_r2048.json")


def main():
    t0 = time.time()
    gt = json.load(open(os.path.join(PKG, "test_gt.json"))); tids = sorted(gt)
    print("[oracle] rasterising GT at 2048 (x2) ...", flush=True)
    gt_fp, tid2img, tree_rles, canopy_union = SC.build_gt(gt, 2048, with_canopy=False)
    gt_cr, _, _, _ = SC.build_gt(gt, 2048, with_canopy=True)
    print(f"[oracle] GT done {time.time()-t0:.0f}s", flush=True)
    P = json.load(open(SRC))["preds"]
    otp, oiou, n_gt, n_hit, n_dup, n_det = {}, {}, 0, 0, 0, 0
    for k, t in enumerate(tids, 1):
        r = P[t]; d = [SC._as_bytes(x) for x in r["masks_rle"]]; g = tree_rles[t]
        s = np.asarray(r["scores"], np.float64); n_gt += len(g); n_det += len(d)
        iou = (np.asarray(maskUtils.iou(d, g, [0] * len(g)), np.float64).reshape(len(d), len(g))
               if d and g else np.zeros((len(d), len(g))))
        a = s / 10.0
        if iou.size:
            ri, ci = linear_sum_assignment(-np.where(iou >= 0.5, iou, 0.0))
            ok = iou[ri, ci] >= 0.5; ri, ci = ri[ok], ci[ok]
            a[ri] = 2.0 + iou[ri, ci]; n_hit += len(ri)
            n_dup += int(((iou >= 0.5).any(1)).sum()) - len(ri)
            b = iou.max(1)
        else:
            b = np.zeros(len(d))
        otp[t] = {**r, "scores": a.tolist()}; oiou[t] = {**r, "scores": b.tolist()}
        if k % 50 == 0:
            print(f"  iou {k}/439 {time.time()-t0:.0f}s", flush=True)
    stats = {"n_gt": n_gt, "n_det": n_det, "assigned_tp50": n_hit, "R50_ceiling": round(n_hit / n_gt, 4),
             "dup_dets_iou50_unassigned": n_dup}
    print(json.dumps(stats), flush=True)
    ctx = (gt_fp, gt_cr, tid2img, tree_rles, canopy_union, tids)
    os.chdir(REPO); rows = {}
    for name, preds in (("oracle_tp", otp), ("oracle_iou", oiou)):
        fp = os.path.join(OUT, f"preds_{TAG}_test_{name}_r2048.json")
        json.dump({"meta": {"mask_res": 2048, "model": f"D1 frozen dets/masks, {name} scores (DIAGNOSTIC)"}, "preds": preds},
                  open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
        c = SC._score_one(fp, ctx, 2048, 600, 0.0)["canopy_neutral_crowd"]
        rows[name] = {"mask": {k: c["mask"][k] for k in ("AP50", "AP75", "AP50_95")}, "box": {k: c["box"][k] for k in ("AP50", "AP75")}}
        print(f"[oracle] {name}: {rows[name]} {time.time()-t0:.0f}s", flush=True)
    out = {"src": os.path.relpath(SRC), "stats": stats, "rows": rows, "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(out, open(os.path.join(OUT, f"oracle_rank_{TAG}_test.json"), "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
