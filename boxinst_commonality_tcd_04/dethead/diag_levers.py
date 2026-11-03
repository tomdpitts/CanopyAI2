"""Mask-AP50 levers on the 439 TEST tiles (TIDE-style oracle fixes, Bolya et al. 2020). Test-informed DIAGNOSTIC, asked for
by Tom; no setting is chosen from it without a new pre-registration. score_lite crowd arm, segm, IoU 0.5, maxDets 600.

Each fix is applied alone to the row's per-detection statuses and AP50 recomputed (fixes interact; the sum is not the gap):
  FN by GT size:        missed (unmatched, non-crowd) GTs in the stratum removed from the denominator (TIDE "missed")
  FP by det size:       FPs whose detection box is in the stratum become ignored
  FP by type:           background (max mask IoU with any GT < 0.1), localisation (0.1-0.5), duplicate/other (>= 0.5)
  ranking oracle:       the same detections re-ordered TPs-first (each group keeps its order): AP of a perfect ranking
Size = sqrt(box area) px: <24, 24-48, 48-96, >=96.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.diag_levers
"""
import contextlib, io, json, os, time
import numpy as np

from boxinst_commonality_tcd_04.dethead.diag_maskbox import HERE, BT, TEST_GT, ap_from

SIZES = [(0, 24), (24, 48), (48, 96), (96, 1e9)]; LBL = ["<24", "24-48", "48-96", ">=96"]


def run(name, preds, gd, t2i, floor):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    dm = SL.build_dt(preds, t2i, 2048, floor, with_masks=True)
    cg = COCO(); cg.dataset = gd; cg.createIndex()
    with contextlib.redirect_stdout(io.StringIO()):
        cd = cg.loadRes([dict(d) for d in dm])
        ev = COCOeval(cg, cd, "segm"); ev.params.iouThrs = np.array([0.5]); ev.params.recThrs = SL.REC_THRS
        ev.params.maxDets = [600]; ev.params.areaRng = [[0, 1e10]]; ev.params.areaRngLbl = ["all"]
        ev.evaluate(); ev.accumulate()
    p = ev.eval["precision"][0, :, 0, 0, 0]; ap = float(p[p > -1].mean())
    gtb = {a["id"]: a["bbox"] for a in gd["annotations"]}
    sc, st, dsz, dio, missed = [], [], [], [], []
    for e in ev.evalImgs:
        if e is None:
            continue
        I = np.asarray(ev.ious[(e["image_id"], 1)]); gign = [g["iscrowd"] for g in ev._gts[e["image_id"], 1]]
        for k, did in enumerate(e["dtIds"]):
            ig = e["dtIgnore"][0][k]; m = e["dtMatches"][0][k]
            sc.append(e["dtScores"][k]); st.append("IGN" if ig else ("TP" if m > 0 else "FP"))
            b = dm[did - 1]["bbox"]; dsz.append(np.sqrt(b[2] * b[3]))
            row = I[k] if I.size else np.zeros(0)                # ious rows follow the evaluated (score-sorted) dt order
            nc = [j for j, c in enumerate(gign) if not c]
            dio.append(float(row[nc].max()) if (row.size and nc) else 0.0)
        for k, gid in enumerate(e["gtIds"]):
            if not e["gtIgnore"][k] and int(e["gtMatches"][0][k]) == 0:
                w, h = gtb[gid][2], gtb[gid][3]; missed.append(np.sqrt(w * h))
    o = np.argsort(-np.asarray(sc), kind="mergesort")
    st = np.array(st)[o]; dsz = np.array(dsz)[o]; dio = np.array(dio)[o]; missed = np.array(missed)
    npig = sum(1 for a in gd["annotations"] if not a["iscrowd"])
    gsz = np.array([np.sqrt(a["bbox"][2] * a["bbox"][3]) for a in gd["annotations"] if not a["iscrowd"]])
    base = ap_from(None, st, npig)
    rec = {"mask_AP50": round(ap, 4), "parity": abs(base - ap) < 1e-6, "n_gt": npig, "n_missed": int(len(missed)),
           "n_fp": int((st == "FP").sum()), "n_tp": int((st == "TP").sum())}

    def d(H, n=npig):
        return round(ap_from(None, H, n) - base, 4)

    rec["FN_by_gt_size"] = {}; rec["FP_by_det_size"] = {}
    for (lo, hi), l in zip(SIZES, LBL):
        nm = int(((missed >= lo) & (missed < hi)).sum()); ng = int(((gsz >= lo) & (gsz < hi)).sum())
        rec["FN_by_gt_size"][l] = {"n_gt": ng, "recall": round(1 - nm / max(ng, 1), 3), "n_missed": nm, "dAP50": d(st, npig - nm)}
        m = (st == "FP") & (dsz >= lo) & (dsz < hi); H = st.copy(); H[m] = "IGN"
        rec["FP_by_det_size"][l] = {"n_fp": int(m.sum()), "dAP50": d(H)}
    rec["FP_by_type"] = {}
    for l, m in (("background_iou<0.1", (st == "FP") & (dio < 0.1)), ("localisation_0.1-0.5", (st == "FP") & (dio >= 0.1) & (dio < 0.5)),
                 ("duplicate_or_other_>=0.5", (st == "FP") & (dio >= 0.5))):
        H = st.copy(); H[m] = "IGN"; rec["FP_by_type"][l] = {"n_fp": int(m.sum()), "dAP50": d(H)}
    H = st.copy(); H[st == "FP"] = "IGN"; rec["all_FP_fix"] = d(H)
    rec["all_FN_fix"] = d(st, npig - len(missed))
    keep = st != "IGN"; s2 = st[keep]; rec["ranking_oracle"] = round(ap_from(None, np.concatenate([s2[s2 == "TP"], s2[s2 == "FP"]]), npig) - base, 4)
    print(f"[{name}] {json.dumps(rec)}", flush=True)
    return rec


def main():
    out = os.path.join(HERE, "results", "diag_levers_test.json"); assert not os.path.exists(out), out
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    g = json.load(open(TEST_GT)); tids = sorted(g); assert len(tids) == 439
    gd, t2i = SL.build_gt_crowd(g, 2048)
    C = json.load(open(os.path.join(HERE, "results", "preds_fpnC2_s0_nms05_sprime_r2048.json")))["preds"]
    B = json.load(open(os.path.join(BT, "preds_boxteacher_sub_test_r2048.json")))["preds"]
    rows = {"C2_sprime": run("C2_sprime", {t: C[t] for t in tids}, gd, t2i, 0.0),
            "boxteacher": run("boxteacher", {t: B[t] for t in tids}, gd, t2i, 0.05)}
    json.dump({"split": "test", "test_informed_diagnostic": True, "rows": rows, "date": time.strftime("%Y-%m-%d %H:%M")},
              open(out + ".tmp", "w"), indent=1); os.replace(out + ".tmp", out); print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
