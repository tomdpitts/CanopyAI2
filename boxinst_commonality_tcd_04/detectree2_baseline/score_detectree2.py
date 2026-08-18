"""Score DetecTree2 predictions with the SAME metric as our result: evaluate.py COCO-101pt
mask/box AP over IOU_50_95 + the >50%-in-canopy ignore rule (via evaluate._greedy_ap +
phase4_lib._instance_pr, the exact functions behind our 0.615). Runs in the main .venv.

Streams tile-by-tile — decodes each tile's masks, computes the (Npred,Ngt) IoU matrices, then
DISCARDS the masks — so 147k predicted masks never sit in RAM at once (they'd be ~38 GB).

    .venv/bin/python -m boxinst_commonality_tcd_04.detectree2_baseline.score_detectree2 \
        --preds .../preds_dt2_s0.json --gt boxinst_commonality_tcd_04/test_gt.json
"""
import argparse
import json

import numpy as np
from pycocotools import mask as maskUtils

from dapt.eval import iou_matrix
from boxinst_commonality_tcd_04 import evaluate as E
from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.phase4_lib_tcd import _instance_pr


def _decode(rle):
    c = rle["counts"]
    return maskUtils.decode({"size": rle["size"],
                             "counts": c.encode("ascii") if isinstance(c, str) else c}
                            ).astype(bool)


def score(preds_path, gt_path, op_thr=0.5, out_json=None):
    P = json.load(open(preds_path))
    preds = P["preds"]
    gt = json.load(open(gt_path))
    # Score EVERY GT tile. A tile with no prediction entry means the model output nothing
    # there -- zero recall on its crowns, not a tile to omit. The old `if t in preds` filter
    # shrank the GT denominator instead, which would hide a partial prediction run.
    tids = sorted(gt)
    missing = [t for t in tids if t not in preds]
    if missing:
        print(f"[score] WARNING: {len(missing)}/{len(tids)} GT tiles have NO predictions; "
              f"scored as zero-prediction (recall penalised): {missing[:5]}", flush=True)
    print(f"[score] {len(tids)} tiles (gt {len(gt)}, pred {len(preds)})", flush=True)

    mI, bI = [], []                      # per-tile IoU matrices (small); masks discarded
    P_scores, Ign_mask, Ign_box = [], [], []
    n_mask = n_box = 0
    for k, tid in enumerate(tids):
        rec = preds.get(tid, {"masks_rle": [], "scores": [], "boxes_2048": []})
        pm = (np.stack([_decode(r) for r in rec["masks_rle"]])
              if rec["masks_rle"] else np.zeros((0, E.RES, E.RES), bool))
        sc = np.array(rec["scores"], np.float32)
        pbox = np.array(rec["boxes_2048"], np.float32).reshape(-1, 4) / E.SCALE
        gm = np.array(E.raster(gt[tid]["trees"]))
        can = np.array(E.raster(gt[tid]["canopy"]))
        can = can.any(0) if len(can) else np.zeros((E.RES, E.RES), bool)
        gbox = (np.array([[*(np.asarray(t).reshape(-1, 2).min(0)),
                           *(np.asarray(t).reshape(-1, 2).max(0))]
                          for t in gt[tid]["trees"]], np.float32).reshape(-1, 4) / E.SCALE
                if gt[tid]["trees"] else np.zeros((0, 4), np.float32))
        # canopy-ignore, identical rule to eval_4p_selfmask
        ign_m = np.array([bool(m.sum()) and (m & can).sum() / m.sum() > 0.5
                          for m in pm]) if len(pm) else np.zeros(0, bool)
        ign_b = np.zeros(len(pbox), bool)
        for i, (x0, y0, x1, y1) in enumerate(pbox):
            sub = can[slice(max(0, int(y0)), int(np.ceil(y1))),
                      slice(max(0, int(x0)), int(np.ceil(x1)))]
            ign_b[i] = sub.size > 0 and sub.mean() > 0.5
        mI.append(E.mask_iou(pm, gm))                    # (Npred,Ngt); masks freed after
        bI.append(iou_matrix(pbox, gbox))
        P_scores.append(sc); Ign_mask.append(ign_m); Ign_box.append(ign_b)
        n_mask += len(gm); n_box += len(gbox)
        del pm, gm, can
        if (k + 1) % 50 == 0 or k + 1 == len(tids):
            print(f"  scored {k+1}/{len(tids)} tiles", flush=True)

    def ap(Is, Ign, n, thr):
        return E._greedy_ap(Is, P_scores, Ign, n, thr)
    mask_ap50 = round(ap(mI, Ign_mask, n_mask, 0.5), 4)
    mask_ap5095 = round(float(np.nanmean([ap(mI, Ign_mask, n_mask, t) for t in E.IOU_50_95])), 4)
    box_ap50 = round(ap(bI, Ign_box, n_box, 0.5), 4)
    box_ap40 = round(ap(bI, Ign_box, n_box, 0.4), 4)
    box_ap5095 = round(float(np.nanmean([ap(bI, Ign_box, n_box, t) for t in E.IOU_50_95])), 4)
    mask_pr = _instance_pr(mI, P_scores, Ign_mask, n_mask, 0.5, op_thr)
    box_pr40 = _instance_pr(bI, P_scores, Ign_box, n_box, 0.4, op_thr)
    box_pr50 = _instance_pr(bI, P_scores, Ign_box, n_box, 0.5, op_thr)

    res = {"model": P["meta"].get("model", "DetecTree2"), "n_tiles": len(tids),
           "n_gt_trees": n_mask, "n_pred": int(sum(len(s) for s in P_scores)),
           "mask_mAP50": mask_ap50, "mask_mAP50_95": mask_ap5095,
           "box_mAP50": box_ap50, "box_mAP40": box_ap40, "box_mAP50_95": box_ap5095,
           "mask_bestF1@0.5": mask_pr["best"], "box_bestF1@0.4": box_pr40["best"],
           "box_bestF1@0.5": box_pr50["best"]}
    print(json.dumps(res, indent=2), flush=True)
    print(f"\n>>> DetecTree2 mask mAP50 = {mask_ap50}  (mAP50-95 {mask_ap5095})", flush=True)
    print(f">>> OURS (β=0.5-fix, 5-seed) mask mAP50 = 0.615 ± 0.011 (seed-0 0.620)", flush=True)
    print(f">>> Restor Mask R-CNN (paper) = 0.432", flush=True)
    if out_json:
        json.dump(res, open(out_json, "w"), indent=2)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", default="boxinst_commonality_tcd_04/test_gt.json")
    ap.add_argument("--op_thr", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    score(a.preds, a.gt, a.op_thr, a.out)
