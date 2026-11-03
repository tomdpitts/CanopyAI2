"""Pre-registration 5 + amendment 5a: per-level Platt calibration of arm C's detector score (local, $0).

Level (proxy, 5a) from box max side: P2 <= 64, P3 (64,128], P4 (128,256], P5 > 256 px. s_cal = sigmoid(a_l logit(s) + b_l),
fitted (unregularised logistic regression, L-BFGS) on the val 108 detections at NMS 0.5: label = TP in the crowd-arm MASK
eval at IoU 0.5 (score_lite segm, maxDets 600, detector-s ranking, floor 0.05), crowd-ignored detections excluded.
Levels with < 50 val detections keep the identity. Applied after NMS to the fixed detection set (raw s >= 0.05).
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.calibC val     # fit + in-sample val
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.calibC test    # single test run
"""
import argparse, json, os, time
import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); HERE = os.path.join(PKG, "dethead"); REPO = os.path.dirname(PKG)
FC_ = "/Users/tompitts/dphil/feat_cache"
OUT = os.path.join(HERE, "results")
VAL_GT = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "val_gt.json")
EDGES = (64.0, 128.0, 256.0)                      # max side -> level 0..3 (P2..P5)
FIT = os.path.join(OUT, "calib_fpnC_s0_nms05.json")


def level(boxes):
    b = np.asarray(boxes, float).reshape(-1, 4)
    return np.searchsorted(EDGES, np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]), side="left")


def logit(s):
    s = np.clip(np.asarray(s, float), 1e-6, 1 - 1e-6); return np.log(s / (1 - s))


def apply(params, boxes, scores):
    lv = level(boxes); x = logit(scores); out = np.empty_like(x)
    for l in range(4):
        a, b = params[str(l)]["a"], params[str(l)]["b"]; m = lv == l
        out[m] = 1 / (1 + np.exp(-(a * x[m] + b)))
    return out


def mask_labels(preds, gd, t2i):
    """-> per tile list of labels (1 TP, 0 FP, -1 crowd-ignored / below floor / beyond maxDets) in the preds' own order."""
    import contextlib, io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    dets = SL.build_dt(preds, t2i, 2048, 0.05, with_masks=True)
    src = []                                                # det index -> (tile, position in the tile's list)
    for t, rec in preds.items():
        for i in np.flatnonzero(np.asarray(rec["scores"], np.float32) >= 0.05):
            src.append((t, int(i)))
    assert len(src) == len(dets)
    cg = COCO(); cg.dataset = gd; cg.createIndex(); cd = cg.loadRes([dict(d) for d in dets])
    ev = COCOeval(cg, cd, "segm"); ev.params.iouThrs = np.array([0.5]); ev.params.recThrs = SL.REC_THRS
    ev.params.maxDets = [600]; ev.params.areaRng = [[0, 1e10]]; ev.params.areaRngLbl = ["all"]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
    lab = {t: np.full(len(r["scores"]), -1, int) for t, r in preds.items()}
    for e in ev.evalImgs:
        if e is None:
            continue
        for k, did in enumerate(e["dtIds"]):
            if e["dtIgnore"][0][k]:
                continue
            t, i = src[did - 1]; lab[t][i] = 1 if e["dtMatches"][0][k] > 0 else 0
    return lab


def fit(x, y):
    from scipy.optimize import minimize
    def nll(p):
        z = p[0] * x + p[1]
        return float(np.sum(np.logaddexp(0, z) - y * z))
    def grad(p):
        r = 1 / (1 + np.exp(-(p[0] * x + p[1]))) - y
        return np.array([np.sum(r * x), np.sum(r)])
    res = minimize(nll, np.array([1.0, 0.0]), jac=grad, method="L-BFGS-B")
    assert res.success, res.message
    return float(res.x[0]), float(res.x[1])


def val():
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    from boxinst_commonality_tcd_04.dethead.diag_fp import run as fp_run
    from boxinst_commonality_tcd_04.train_detector_tiles import canopy_px
    assert not os.path.exists(FIT), FIT
    vg = json.load(open(VAL_GT)); tids = sorted(vg); gd, t2i = SL.build_gt_crowd(vg, 2048)
    V = json.load(open(f"{FC_}/dethead_fpnC_s0_val_nms05_mask.json")); assert sorted(V) == tids
    raw = {t: {"boxes_2048": V[t]["boxes_2048"], "scores": V[t]["scores"], "masks_rle": V[t]["masks_rle"]} for t in tids}
    lab = mask_labels(raw, gd, t2i)
    X, Y, Lv = [], [], []
    for t in tids:
        m = lab[t] >= 0; X.append(logit(np.asarray(V[t]["scores"])[m])); Y.append(lab[t][m]); Lv.append(level(V[t]["boxes_2048"])[m])
    X, Y, Lv = np.concatenate(X), np.concatenate(Y), np.concatenate(Lv)
    params = {}
    for l in range(4):
        m = Lv == l
        a, b = fit(X[m], Y[m].astype(float)) if m.sum() >= 50 else (1.0, 0.0)
        params[str(l)] = {"a": round(a, 6), "b": round(b, 6), "n": int(m.sum()), "pos_rate": round(float(Y[m].mean()), 4) if m.any() else None}
    print("[calib] params", json.dumps(params), flush=True)
    rows = {}
    for name, sc_of in (("detscore", lambda t: V[t]["scores"]), ("calibrated", lambda t: apply(params, V[t]["boxes_2048"], V[t]["scores"]).tolist())):
        preds = {t: {"boxes_2048": V[t]["boxes_2048"], "scores": sc_of(t), "masks_rle": V[t]["masks_rle"]} for t in tids}
        r = SL.score(preds, gd, t2i, masks=True, floor=0.0)          # detection set already raw s >= 0.05 upstream
        rows[name] = {"mask": {k: r["mask"][k] for k in ("AP50", "AP75", "AP50_95")}, "box": {k: r["box"][k] for k in ("AP50", "AP75")}}
        print(f"[val in-sample] {name}: {rows[name]}", flush=True)
    canopy = {t: (canopy_px(vg[t]["canopy"], 2048) if vg[t].get("canopy") else None) for t in tids}
    Vc = {t: {"boxes_2048": V[t]["boxes_2048"], "scores": apply(params, V[t]["boxes_2048"], V[t]["scores"]).tolist()} for t in tids}
    fp = fp_run("armC_calibrated", Vc, gd, t2i, canopy, floor=0.0)
    rec = {"prereg": "README pre-registration 5 + amendment 5a", "edges_max_side_px": EDGES, "params": params,
           "val_in_sample": rows, "val_fp_breakdown_calibrated": fp, "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(rec, open(FIT, "w"), indent=1); print(f"wrote {FIT}", flush=True)


def test():
    from boxinst_commonality_tcd_04 import score_coco as SC
    params = json.load(open(FIT))["params"]
    sfp = os.path.join(OUT, "summary_fpnC_s0_nms05_cal.json"); assert not os.path.exists(sfp), sfp
    D = json.load(open(os.path.join(OUT, "modal", "fpnC_s0_test_nms0.5.json")))["preds"]; tids = sorted(D); assert len(tids) == 439
    mdir = f"{FC_}/dethead_fpnC_s0_nms05_mask"
    fp = os.path.join(OUT, "preds_fpnC_s0_nms05_cal_r2048.json")
    if not os.path.exists(fp):
        preds = {}
        for t in tids:
            M = json.load(open(os.path.join(mdir, t + ".json"))); s = np.asarray(D[t]["scores"])
            assert (s >= 0.05).all() and len(M["masks_rle"]) == len(s), t
            preds[t] = {"boxes_2048": D[t]["boxes_2048"], "scores": apply(params, D[t]["boxes_2048"], s).tolist(), "masks_rle": M["masks_rle"]}
        json.dump({"meta": {"mask_res": 2048, "model": "dethead arm C fpnC_s0 NMS 0.5 + LACE lace4px_simple, per-level Platt-calibrated detector s (prereg 5/5a)"},
                   "preds": preds}, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    sp = fp.replace("preds_", "scored_"); os.chdir(REPO)
    if not os.path.exists(sp):
        json.dump(SC.score_many([fp], os.path.join(PKG, "test_gt.json"), res=2048, max_dets=600, score_floor=0.0)[0], open(sp, "w"))
    c = json.load(open(sp))["canopy_neutral_crowd"]
    out = {"tag": "fpnC_s0_nms05_cal", "params": params, "test_informed": True,
           "mask": {k: c["mask"][k] for k in ("AP50", "AP75", "AP50_95", "n_det")}, "box": {k: c["box"][k] for k in ("AP50", "AP75", "AP50_95")}}
    json.dump(out, open(sfp, "w"), indent=1); print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["val", "test"]); a = ap.parse_args()
    val() if a.cmd == "val" else test()
