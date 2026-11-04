"""Paired tile-bootstrap of test mask AP50 differences (439 tiles): D1 (DINO-Swin-L + LACE masker, s') vs BoxTeacher, C2 and
LACE s0. Matching is per image in COCOeval (segm, IoU 0.5, maxDets 600, crowd arm), so per-tile statuses are computed once and
AP is recomputed on each resample of tiles (with replacement, same draw for every system). 2,000 resamples, seed 0.
Captures test-set sampling uncertainty only, not training-seed variance.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.boot_vs_bt [--split sparse]
--split sparse (2026-11-03): same procedure on the 236 sparse tiles, D1 vs BoxTeacher (B2 frozen, inference only) and LACE s0.
"""
import contextlib, io, json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE)
BT = os.path.join(PKG, "box_supervised_baselines", "tierb", "results", "20261101_192510", "boxteacher", "results")
ROWS = {"D1_sprime": (os.path.join(HERE, "results", "preds_dino_swinl_s0_test_sprime_r2048.json"), 0.0),
        "boxteacher": (os.path.join(BT, "preds_boxteacher_sub_test_r2048.json"), 0.05),
        "C2_sprime": (os.path.join(PKG, "dethead", "results", "preds_fpnC2_s0_nms05_sprime_r2048.json"), 0.0),
        "lace_s0": (os.path.join(PKG, "native_raster", "hunt", "test", "preds_lace_s0_r2048_lace4px_simple.json"), 0.0)}
SP_ROWS = {"D1_sprime": (os.path.join(HERE, "results", "preds_dino_swinl_s0_sparse_sprime_r2048.json"), 0.0),
           "boxteacher": (os.path.join(PKG, "box_supervised_baselines", "tierb", "results_sparse", "preds_boxteacher_sub_sparse_r2048.json"), 0.05),
           "lace_s0": (os.path.join(PKG, "native_raster", "hunt", "test", "preds_lace_sparse_s0_r2048_lace4px_simple.json"), 0.0)}
N, SEED = 2000, 0


def per_tile(preds, gd, t2i, floor):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    dets = SL.build_dt(preds, t2i, 2048, floor, with_masks=True)
    cg = COCO(); cg.dataset = gd; cg.createIndex()
    with contextlib.redirect_stdout(io.StringIO()):
        cd = cg.loadRes([dict(d) for d in dets]); ev = COCOeval(cg, cd, "segm")
        ev.params.iouThrs = np.array([0.5]); ev.params.maxDets = [600]; ev.params.areaRng = [[0, 1e10]]; ev.params.areaRngLbl = ["all"]
        ev.evaluate()
    out = {}
    for e in ev.evalImgs:
        if e is None:
            continue
        keep = ~np.asarray(e["dtIgnore"][0], bool)
        out[e["image_id"]] = (np.asarray(e["dtScores"])[keep], (np.asarray(e["dtMatches"][0]) > 0)[keep],
                              int((~np.asarray(e["gtIgnore"], bool)).sum()))
    return out


def ap(tiles, data, rec_thrs):
    sc = np.concatenate([data[t][0] for t in tiles if t in data]); tp = np.concatenate([data[t][1] for t in tiles if t in data])
    npig = sum(data[t][2] for t in tiles if t in data)
    o = np.argsort(-sc, kind="mergesort"); tp = tp[o]
    tpc = np.cumsum(tp); fpc = np.cumsum(~tp); rc = tpc / max(npig, 1); pr = tpc / np.maximum(tpc + fpc, np.spacing(1))
    pr = np.maximum.accumulate(pr[::-1])[::-1]
    idx = np.searchsorted(rc, rec_thrs, side="left"); q = np.where(idx < len(pr), pr[np.minimum(idx, len(pr) - 1)], 0.0)
    return float(q.mean())


def main(split="test"):
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    out = os.path.join(HERE, "results", f"boot_{split}_mask_ap50.json"); assert not os.path.exists(out) or split == "test", out
    gt = os.path.join(PKG, "test_gt.json") if split == "test" else os.path.join(os.path.dirname(PKG), "data", "tcd_sparse", "sparse_gt.json")
    rows = ROWS if split == "test" else SP_ROWS
    g = json.load(open(gt)); tids = sorted(g); gd, t2i = SL.build_gt_crowd(g, 2048)
    ids = np.array([t2i[t] for t in tids])
    D = {}
    for k, (fp, floor) in rows.items():
        P = json.load(open(fp)); P = P.get("preds", P)
        D[k] = per_tile({t: P[t] for t in tids}, gd, t2i, floor)
        # images with no detections still carry GT: give them empty arrays
        for i in ids:
            if i not in D[k]:
                D[k][i] = (np.zeros(0), np.zeros(0, bool), sum(1 for a in gd["annotations"] if a["image_id"] == i and not a["iscrowd"]))
        print(f"[{k}] full AP50 {ap(ids, D[k], SL.REC_THRS):.4f}", flush=True)
    rng = np.random.default_rng(SEED); full = {k: ap(ids, D[k], SL.REC_THRS) for k in D}; diffs = {k: [] for k in D if k != "D1_sprime"}
    t0 = time.time()
    for b in range(N):
        s = ids[rng.integers(0, len(ids), len(ids))]
        a = ap(s, D["D1_sprime"], SL.REC_THRS)
        for k in diffs:
            diffs[k].append(a - ap(s, D[k], SL.REC_THRS))
    rec = {"split": split, "n_tiles": len(ids), "n_boot": N, "seed": SEED, "full_AP50": {k: round(v, 4) for k, v in full.items()},
           "D1_minus": {k: {"diff": round(full["D1_sprime"] - full[k], 4), "ci95": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)],
                            "p_le_0": round(float(np.mean(np.asarray(v) <= 0)), 4)} for k, v in diffs.items()},
           "note": "test-set sampling uncertainty only (tiles resampled), not training-seed variance", "sec": round(time.time() - t0)}
    json.dump(rec, open(out, "w"), indent=1); print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    import argparse
    ap_ = argparse.ArgumentParser(); ap_.add_argument("--split", choices=["test", "sparse"], default="test")
    main(ap_.parse_args().split)
