"""Split the D1 ranking gap into within-tile ORDER and cross-tile CALIBRATION (2026-11-03). Diagnostic only.

TP labels are read from oracle_rank.py's oracle_tp preds: score >= 2 means the det was Hungarian-assigned at mask IoU >= 0.5.
The dets and masks are the frozen s' preds.
  within_only: each tile keeps its own multiset of s' values, but they are re-dealt in oracle order (assigned TPs
               first by IoU, then the rest by s'). The within-tile order is perfect; the cross-tile calibration is s''s.
  calib_only : per tile, isotonic (PAV) fit of the TP label on s'. Ties are broken by s' (1e-6 * s'), so the within-tile
               order is exactly s''s, and the score equals the tile's oracle precision at that s'.
Crowd-ignored dets count as label 0 in the PAV fit (COCOeval ignores them at any score).
Frozen score_coco internals (2048, maxDets 600, floor 0, crowd arm).
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.oracle_split
"""
import json, os, time
import numpy as np

from boxinst_commonality_tcd_04 import score_coco as SC

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); REPO = os.path.dirname(PKG)
OUT = os.path.join(HERE, "results"); TAG = "dino_swinl_s0"


def pav(y):
    """Non-decreasing least-squares fit to y (y is already sorted by ascending s')."""
    v, w, n = [], [], []
    for yi in y:
        v.append(float(yi)); w.append(1.0); n.append(1)
        while len(v) > 1 and v[-2] > v[-1]:
            tw = w[-2] + w[-1]; v[-2] = (v[-2] * w[-2] + v[-1] * w[-1]) / tw; w[-2] = tw; n[-2] += n[-1]
            v.pop(); w.pop(); n.pop()
    return np.repeat(v, n)


def main():
    t0 = time.time()
    S = json.load(open(os.path.join(OUT, f"preds_{TAG}_test_sprime_r2048.json")))["preds"]
    O = json.load(open(os.path.join(OUT, f"preds_{TAG}_test_oracle_tp_r2048.json")))["preds"]
    within, calib = {}, {}
    for t, r in S.items():
        s = np.asarray(r["scores"], np.float64); o = np.asarray(O[t]["scores"], np.float64)
        assert len(s) == len(o)
        a = np.empty_like(s); a[np.argsort(-o, kind="mergesort")] = np.sort(s)[::-1]
        within[t] = {**r, "scores": a.tolist()}
        idx = np.argsort(s, kind="mergesort"); c = np.empty_like(s)
        c[idx] = pav((o[idx] >= 2.0).astype(np.float64)) if len(s) else c
        calib[t] = {**r, "scores": (c + 1e-6 * s).tolist()}
    gt = json.load(open(os.path.join(PKG, "test_gt.json"))); tids = sorted(gt)
    print("[split] rasterising GT at 2048 (x2) ...", flush=True)
    gt_fp, tid2img, tree_rles, canopy_union = SC.build_gt(gt, 2048, with_canopy=False)
    gt_cr, _, _, _ = SC.build_gt(gt, 2048, with_canopy=True)
    print(f"[split] GT done {time.time()-t0:.0f}s", flush=True)
    ctx = (gt_fp, gt_cr, tid2img, tree_rles, canopy_union, tids)
    os.chdir(REPO); rows = {}
    for name, preds in (("within_only", within), ("calib_only", calib)):
        fp = os.path.join(OUT, f"preds_{TAG}_test_{name}_r2048.json")
        json.dump({"meta": {"mask_res": 2048, "model": f"D1 frozen dets/masks, {name} scores (DIAGNOSTIC)"}, "preds": preds},
                  open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
        c = SC._score_one(fp, ctx, 2048, 600, 0.0)["canopy_neutral_crowd"]
        rows[name] = {"mask": {k: c["mask"][k] for k in ("AP50", "AP75", "AP50_95")}, "box": {k: c["box"][k] for k in ("AP50", "AP75")}}
        print(f"[split] {name}: {rows[name]} {time.time()-t0:.0f}s", flush=True)
    out = {"rows": rows, "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(out, open(os.path.join(OUT, f"oracle_split_{TAG}_test.json"), "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
