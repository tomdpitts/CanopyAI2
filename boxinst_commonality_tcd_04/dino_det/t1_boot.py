"""Paired tile-bootstrap CIs for Table 1 (tab:oamtcd439) and tab:sparse236: LACE (deployed: D1 + edge fix + R3 = \\hat s) minus
each baseline row, mask AP50 and AP50:95, canopy-neutral crowd arm (maxDets 600, each row's own protocol floor).
Pre-registration T1-boot, dino_det/README.md (2026-11-11). Extends boot_vs_bt.py to all rows and all 10 IoU thresholds.

Per-tile COCOeval statuses (scores, matches[T,D], ignores[T,D], #non-ignored GT) are computed once per row; each of 2,000 resamples
draws the same tiles (with replacement) for every row and recomputes COCO 101-point AP at every threshold. Captures test-set
sampling uncertainty only, not training-seed variance.

GATE (before any resampling): each row's full-sample AP50 and AP50:95 from this code must reproduce its scored_*.json
(canopy_neutral_crowd.mask) within 5e-4, else abort.

  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.t1_boot --split {test,sparse}
"""
import contextlib, io, json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); NR = os.path.join(PKG, "native_raster")
BT = os.path.join(PKG, "box_supervised_baselines", "tierb")
R = os.path.join(HERE, "results")
# name: (preds, scored (gate), floor)
ROWS = {
    "test": {
        "LACE": (f"{R}/preds_dino_swinl_s0_test_R3_ef_r2048.json", f"{R}/scored_dino_swinl_s0_test_R3_ef_r2048.json", 0.0),
        "Restor": (f"{NR}/preds_restor_rpn1000_r2048_treeonly.json", f"{NR}/scored_restor_rpn1000_r2048_tree.json", 0.05),
        "Detectree2": (f"{NR}/seamfix/preds_dt2_s0v2_r2048_seamfix.json", f"{NR}/seamfix/scored_dt2_s0v2_r2048_seamfix.json", 0.05),
        "Box2Mask_SwinL": (f"{NR}/seamfix/preds_box2mask_swinl_s0_r2048_seamfix.json", f"{NR}/seamfix/scored_box2mask_swinl_s0_r2048_seamfix.json", 0.05),
        "Box2Mask_R50": (f"{NR}/seamfix/preds_box2mask_r50_s0_r2048_seamfix.json", f"{NR}/seamfix/scored_box2mask_r50_s0_r2048_seamfix.json", 0.05),
        "BoxTeacher": (f"{BT}/results/20261101_192510/boxteacher/results/preds_boxteacher_sub_test_r2048.json",
                       f"{BT}/results/20261101_192510/boxteacher/results/scored_boxteacher_sub_test_r2048.json", 0.05)},
    "sparse": {
        "LACE": (f"{R}/preds_dino_swinl_s0_sparse_R3_ef_r2048.json", f"{R}/scored_dino_swinl_s0_sparse_R3_ef_r2048.json", 0.0),
        "Detectree2": (f"{NR}/seamfix/preds_dt2_sparse_s0v2_r2048_seamfix.json", f"{NR}/seamfix/scored_dt2_sparse_s0v2_r2048_seamfix.json", 0.05),
        "Box2Mask_SwinL": (f"{NR}/seamfix/preds_box2mask_swinl_sparse_s0_r2048_seamfix.json", f"{NR}/seamfix/scored_box2mask_swinl_sparse_s0_r2048_seamfix.json", 0.05),
        "BoxTeacher": (f"{BT}/results_sparse/preds_boxteacher_sub_sparse_r2048.json", f"{BT}/results_sparse/scored_boxteacher_sub_sparse_r2048.json", 0.05)}}
GT = {"test": os.path.join(PKG, "test_gt.json"), "sparse": os.path.join(os.path.dirname(PKG), "data", "tcd_sparse", "sparse_gt.json")}
N, SEED, GATE_TOL = 2000, 0, 5e-4


def per_tile(preds, gd, t2i, floor):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    dets = SL.build_dt(preds, t2i, 2048, floor, with_masks=True)
    cg = COCO(); cg.dataset = gd; cg.createIndex()
    with contextlib.redirect_stdout(io.StringIO()):
        cd = cg.loadRes([dict(d) for d in dets]); ev = COCOeval(cg, cd, "segm")
        ev.params.iouThrs = SL.IOU_THRS; ev.params.maxDets = [600]; ev.params.areaRng = [[0, 1e10]]; ev.params.areaRngLbl = ["all"]
        ev.evaluate()
    out = {}
    for e in ev.evalImgs:
        if e is None:
            continue
        out[e["image_id"]] = (np.asarray(e["dtScores"], np.float64), np.asarray(e["dtMatches"]) > 0,
                              np.asarray(e["dtIgnore"], bool), int((~np.asarray(e["gtIgnore"], bool)).sum()))
    return out


def ap_all(tiles, data, rec_thrs):
    """COCO accumulate() for one category/area/maxDet: returns AP per IoU threshold (T,)."""
    sc = np.concatenate([data[t][0] for t in tiles]); m = np.concatenate([data[t][1] for t in tiles], axis=1)
    ig = np.concatenate([data[t][2] for t in tiles], axis=1); npig = sum(data[t][3] for t in tiles)
    o = np.argsort(-sc, kind="mergesort"); m = m[:, o]; ig = ig[:, o]
    tp = np.cumsum(m & ~ig, axis=1, dtype=np.float64); fp = np.cumsum(~m & ~ig, axis=1, dtype=np.float64)
    T = m.shape[0]; ap = np.zeros(T)
    for t in range(T):
        keep = ~ig[t]; tpc = tp[t][keep]; fpc = fp[t][keep]
        if npig == 0 or len(tpc) == 0:
            continue
        rc = tpc / npig; pr = tpc / np.maximum(tpc + fpc, np.spacing(1)); pr = np.maximum.accumulate(pr[::-1])[::-1]
        idx = np.searchsorted(rc, rec_thrs, side="left"); ap[t] = np.where(idx < len(pr), pr[np.minimum(idx, len(pr) - 1)], 0.0).mean()
    return ap


def main(split):
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    out = os.path.join(R, f"t1_boot_{split}.json"); assert not os.path.exists(out), f"exists, one look only: {out}"
    g = json.load(open(GT[split])); tids = sorted(g); gd, t2i = SL.build_gt_crowd(g, 2048); ids = np.array([t2i[t] for t in tids])
    D, full, gate = {}, {}, {}
    for k, (fp, sfp, floor) in ROWS[split].items():
        t0 = time.time(); P = json.load(open(fp)); P = P.get("preds", P)
        missing = [t for t in tids if t not in P]
        D[k] = per_tile({t: P[t] for t in tids if t in P}, gd, t2i, floor); del P
        for i in ids:  # tiles with no detections still carry GT
            if i not in D[k]:
                D[k][i] = (np.zeros(0), np.zeros((10, 0), bool), np.zeros((10, 0), bool),
                           sum(1 for a in gd["annotations"] if a["image_id"] == i and not a["iscrowd"]))
        a = ap_all(ids, D[k], SL.REC_THRS); full[k] = {"AP50": float(a[0]), "AP50_95": float(a.mean())}
        ref = json.load(open(sfp))["canopy_neutral_crowd"]["mask"]
        gate[k] = {"ours": {m: round(v, 4) for m, v in full[k].items()}, "scored": {"AP50": ref["AP50"], "AP50_95": ref["AP50_95"]},
                   "missing_tiles": len(missing)}
        ok = abs(full[k]["AP50"] - ref["AP50"]) <= GATE_TOL and abs(full[k]["AP50_95"] - ref["AP50_95"]) <= GATE_TOL
        print(f"[{split}] {k}: AP50 {full[k]['AP50']:.4f} (scored {ref['AP50']}) AP50:95 {full[k]['AP50_95']:.4f} (scored {ref['AP50_95']}) "
              f"missing {len(missing)} gate {'PASS' if ok else 'FAIL'} [{time.time() - t0:.0f}s]", flush=True)
        if not ok:
            raise SystemExit(f"GATE FAIL {k}: {gate[k]}")
    rng = np.random.default_rng(SEED); others = [k for k in D if k != "LACE"]
    diffs = {k: {"AP50": [], "AP50_95": []} for k in others}; t0 = time.time()
    for b in range(N):
        s = ids[rng.integers(0, len(ids), len(ids))]; a = ap_all(s, D["LACE"], SL.REC_THRS)
        for k in others:
            c = ap_all(s, D[k], SL.REC_THRS); diffs[k]["AP50"].append(a[0] - c[0]); diffs[k]["AP50_95"].append(a.mean() - c.mean())
        if (b + 1) % 100 == 0:
            print(f"[{split}] resample {b + 1}/{N} [{time.time() - t0:.0f}s]", flush=True)
    rec = {"split": split, "n_tiles": int(len(ids)), "n_boot": N, "seed": SEED, "gate": gate,
           "LACE_minus": {k: {m: {"diff": round(full["LACE"][m] - full[k][m], 4),
                                  "ci95": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)],
                                  "p_le_0": round(float(np.mean(np.asarray(v) <= 0)), 4)} for m, v in d.items()} for k, d in diffs.items()},
           "note": "paired tile bootstrap, test-set sampling uncertainty only (not training-seed variance); LACE = D1 + edge fix + R3 (hat s)",
           "sec": round(time.time() - t0)}
    json.dump(rec, open(out, "w"), indent=1); print(json.dumps(rec["LACE_minus"], indent=1), flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--split", choices=["test", "sparse"], required=True)
    main(p.parse_args().split)
