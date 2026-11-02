"""VAL-only seam diagnostic. LACE's DINOv3 features are extracted per 1024 window (2x2 per 2048 tile, stitched), so
cells either side of x=1024 / y=1024 come from different ViT passes. Recall at IoU 0.5 (COCOeval bbox, crowd arm,
maxDets 600) for GT crowns that STRADDLE a seam (box crosses x=1024 or y=1024), lie NEAR one (box edge within 32 px,
not crossing) or lie FAR from both; plus the same split of detections that are FPs. Rows as diag_box.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.diag_seam
"""
import json, os, time
import numpy as np

from boxinst_commonality_tcd_04.dethead.diag_box import BT_VAL, FCC, HERE, VAL_GT

OUT = os.path.join(HERE, "results", "diag_seam_val.json")
SEAM, NEAR = 1024.0, 32.0


def cat(b):
    x0, y0, x1, y1 = b
    if (x0 < SEAM < x1) or (y0 < SEAM < y1):
        return "straddle"
    d = min(abs(x0 - SEAM), abs(x1 - SEAM), abs(y0 - SEAM), abs(y1 - SEAM))
    return "near" if d <= NEAR else "far"


def analyse(S, gd, t2i, floor):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    dets = [{"image_id": t2i[t], "category_id": 1, "score": float(s), "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]]}
            for t, (B, sc) in S.items() for b, s in zip(B, sc) if s >= floor]
    cg = COCO(); cg.dataset = gd; cg.createIndex(); cd = cg.loadRes([dict(d) for d in dets])
    ev = COCOeval(cg, cd, "bbox"); ev.params.iouThrs = np.array([0.5]); ev.params.recThrs = SL.REC_THRS
    ev.params.maxDets = [600]; ev.params.areaRng = [[0, 1e10]]; ev.params.areaRngLbl = ["all"]
    ev.evaluate()
    g = {"straddle": [], "near": [], "far": []}; fp = {"straddle": 0, "near": 0, "far": 0}; nd = {"straddle": 0, "near": 0, "far": 0}
    for e in ev.evalImgs:
        if e is None:
            continue
        for a, ig, m in zip(cg.loadAnns(e["gtIds"]), e["gtIgnore"], e["gtMatches"][0]):
            if ig:
                continue
            x, y, w, h = a["bbox"]; g[cat((x, y, x + w, y + h))].append(m > 0)
        for a, ig, m in zip(cd.loadAnns(e["dtIds"]), e["dtIgnore"][0], e["dtMatches"][0]):
            if ig:
                continue
            x, y, w, h = a["bbox"]; c = cat((x, y, x + w, y + h)); nd[c] += 1; fp[c] += m == 0
    return {c: {"n_gt": len(v), "recall": round(float(np.mean(v)), 4), "n_det": nd[c], "fp_frac": round(fp[c] / max(nd[c], 1), 4)}
            for c, v in g.items()}


def main():
    assert not os.path.exists(OUT), OUT
    import torch
    from boxinst_commonality_tcd_04.dethead import fcos as FC
    from boxinst_commonality_tcd_04.dethead.test import load_model
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    vg = json.load(open(VAL_GT)); tids = sorted(vg); gd, t2i = SL.build_gt_crowd(vg, 2048)
    m = load_model(os.path.join(HERE, "runs", "fcos_s0", "best.pt"), "fcos"); f5 = {}
    FC.NMS_IOU = 0.5
    with torch.no_grad():
        for t in tids:
            b, s = FC.decode(m(torch.from_numpy(np.load(f"{FCC}/feat_4p_val/{t}.npy")).float()[None]))
            f5[t] = (b.numpy().astype(float).tolist(), s.numpy().astype(float).tolist())
    cn = {t: (lambda d: (d["boxes_2048"], d["scores_product"]))(json.load(open(f"{FCC}/boxfix_ap_val_tiles/{t}.json"))) for t in tids}
    bt = json.load(open(BT_VAL))["preds"]; btS = {t: (bt[t]["boxes_2048"], bt[t]["scores"]) for t in tids}
    rows = {"lace_centernet_s0": analyse(cn, gd, t2i, 0.0), "fcos_nms05": analyse(f5, gd, t2i, 0.05), "boxteacher": analyse(btS, gd, t2i, 0.05)}
    for k, v in rows.items():
        print(k, json.dumps(v), flush=True)
    json.dump({"seam_px": SEAM, "near_px": NEAR, "rows": rows, "date": time.strftime("%Y-%m-%d %H:%M")}, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
