"""VAL-only false-positive breakdown, arm C vs BoxTeacher (diagnostic after arm C's val selection; any fix it motivates
needs its own pre-registration and test run). score_lite crowd arm, bbox, IoU 0.5, maxDets 600, floor 0.05.

Per source: detections ranked above the score at which recall reaches 0.6 ("top"), and all detections. For TP and FP:
counts by detection size (sqrt box area px: <24, 24-48, 48-96, >=96), by canopy contact (fraction of the box covered by
canopy pixels: 0, (0, 0.5), >=0.5 but not crowd-ignored), and FP score quartiles. Precision per size bin = TP / (TP + FP).
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.diag_fp
"""
import json, os, time
import numpy as np

from boxinst_commonality_tcd_04.dethead.diag_maskbox import HERE, BT, VAL_GT

SIZES = [(0, 24), (24, 48), (48, 96), (96, 1e9)]; SL_ = ["<24", "24-48", "48-96", ">=96"]


def run(name, preds, gd, t2i, canopy, floor=0.05):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    dets = SL.build_dt(preds, t2i, 2048, floor, with_masks=False)
    cg = COCO(); cg.dataset = gd; cg.createIndex(); cd = cg.loadRes([dict(d) for d in dets])
    ev = COCOeval(cg, cd, "bbox"); ev.params.iouThrs = np.array([0.5]); ev.params.recThrs = SL.REC_THRS
    ev.params.maxDets = [600]; ev.params.areaRng = [[0, 1e10]]; ev.params.areaRngLbl = ["all"]
    ev.evaluate(); ev.accumulate()
    p = ev.eval["precision"][0, :, 0, 0, 0]; ap = float(p[p > -1].mean())
    i2t = {v: k for k, v in t2i.items()}
    rows = []                                                  # (score, status, size, canopy_frac)
    for e in ev.evalImgs:
        if e is None:
            continue
        C = canopy[i2t[e["image_id"]]]
        for k, did in enumerate(e["dtIds"]):
            if e["dtIgnore"][0][k]:
                continue
            x, y, w, h = dets[did - 1]["bbox"]
            x0, y0, x1, y1 = int(max(x, 0)), int(max(y, 0)), int(min(x + w, 2048)), int(min(y + h, 2048))
            cf = float(C[y0:y1, x0:x1].mean()) if (C is not None and x1 > x0 and y1 > y0) else 0.0
            rows.append((e["dtScores"][k], "TP" if e["dtMatches"][0][k] > 0 else "FP", np.sqrt(w * h), cf))
    rows.sort(key=lambda r: -r[0])
    st = np.array([r[1] for r in rows]); sz = np.array([r[2] for r in rows]); cf = np.array([r[3] for r in rows]); sc = np.array([r[0] for r in rows])
    npig = sum(1 for a in gd["annotations"] if not a["iscrowd"])
    cut = int(np.searchsorted(np.cumsum(st == "TP"), 0.6 * npig)) + 1
    out = {"box_AP50": round(ap, 4), "n_det": len(rows), "cut_rank_recall0.6": cut, "score_at_cut": round(float(sc[min(cut, len(sc)) - 1]), 4)}
    for part, sel in (("top", np.arange(len(st)) < cut), ("all", np.ones(len(st), bool))):
        d = {}
        for (lo, hi), l in zip(SIZES, SL_):
            m = sel & (sz >= lo) & (sz < hi); tp = int((m & (st == "TP")).sum()); fp = int((m & (st == "FP")).sum())
            d[l] = {"TP": tp, "FP": fp, "precision": round(tp / max(tp + fp, 1), 3)}
        f = sel & (st == "FP")
        d["FP_canopy"] = {"none": int((f & (cf == 0)).sum()), "touching_<0.5": int((f & (cf > 0) & (cf < 0.5)).sum()), ">=0.5": int((f & (cf >= 0.5)).sum())}
        d["FP_score_quartiles"] = [round(float(q), 3) for q in np.percentile(sc[f], [25, 50, 75])] if f.any() else []
        out[part] = d
    print(f"[{name}] {json.dumps(out)}", flush=True)
    return out


def main():
    out = os.path.join(HERE, "results", "diag_fp_val.json"); assert not os.path.exists(out), out
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    from boxinst_commonality_tcd_04.train_detector_tiles import canopy_px
    vg = json.load(open(VAL_GT)); tids = sorted(vg); gd, t2i = SL.build_gt_crowd(vg, 2048)
    canopy = {t: (canopy_px(vg[t]["canopy"], 2048) if vg[t].get("canopy") else None) for t in tids}
    C = json.load(open(os.path.join(HERE, "results", "modal", "fpnC_s0_val_nms0.5.json")))["preds"]
    B = json.load(open(os.path.join(BT, "preds_boxteacher_sub_val_r2048.json")))["preds"]
    rows = {"armC_nms05_detscore": run("armC", {t: C[t] for t in tids}, gd, t2i, canopy),
            "boxteacher": run("boxteacher", {t: B[t] for t in tids}, gd, t2i, canopy)}
    json.dump({"split": "val", "protocol": "score_lite crowd bbox IoU 0.5, maxDets 600, floor 0.05", "rows": rows,
               "date": time.strftime("%Y-%m-%d %H:%M")}, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
