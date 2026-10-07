"""Val-only mask/box AP for any detector checkpoint: the adopt/stop harness for the retrain pilot.

The as-is arm of oracle_boxes_val.py, parametrised by --det (the original is not edited): 108
val tiles, local 4-phase L24 features, deployed masker (alpha 0.3, kappa 1.6), tau 0.40 at 2048
(0.25 at 512), detector-score ranking, score_coco canopy-neutral crowd arm. VAL ONLY — this
script cannot read test features.

GATE. Before scoring a new detector it re-runs the published seed-0 detector and requires
AP50 0.6045 and 40,058 dets at 2048 (cell 0.40 of val_knobs_tau_r2048_s0_local.json); the
gate result is cached in the output file, so it runs once per output.

    .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.det_val_eval \
        --det /Users/tompitts/dphil/feat_cache/det_v2_L24_s0.pt --name v2_L24_s0
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time

import numpy as np
import torch
from pycocotools import mask as maskUtils

from boxinst_commonality_tcd_04 import evaluate as E
from boxinst_commonality_tcd_04 import score_coco as S
from boxinst_commonality_tcd_04.detector import STRIDE8
from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.phase4_lib_tcd import Detector4Phase
from boxinst_commonality_tcd_04.native_raster.oracle_boxes_val import (
    DET_PATH, EM_PATH, FEAT_DIR, GATE_REF, HERE, KAPPA_SCALE, PRIOR_WEIGHT, TAU, VAL_GT)
from dapt.decode import decode


def run_arm(det_path, tiles, gt, ctx, masker, res, tau):
    ck = torch.load(det_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = Detector4Phase(cfg["in_dim"], width=cfg["width"], tower=cfg["tower"]).eval()
    model.load_state_dict(ck["state"])
    preds, n_det, t0 = {}, 0, time.time()
    for k, tid in enumerate(tiles):
        feat = np.load(os.path.join(FEAT_DIR, tid + ".npy")).astype(np.float32)
        with torch.no_grad():
            det = model(torch.from_numpy(feat)[None])
        bx, sc = decode(det, score_thr=0.05, stride=STRIDE8, topk=600)
        bx, sc = bx.numpy(), sc.numpy()
        n_det += len(bx)
        if len(bx) == 0:
            preds[tid] = {"boxes_2048": [], "scores": [], "canopy_ignore": [], "masks_rle": []}
            continue
        pm = E.pred_instance_masks(masker, masker.project(feat), feat.shape[-1], bx, res=res,
                                   scale=2048.0 / res, mask_thr=tau,
                                   prior_weight=PRIOR_WEIGHT, kappa_scale=KAPPA_SCALE)
        preds[tid] = {"boxes_2048": bx.tolist(), "scores": sc.tolist(),
                      "canopy_ignore": [False] * len(bx),
                      "masks_rle": [{"size": [res, res], "counts": maskUtils.encode(
                          np.asfortranarray(m.astype(np.uint8)))["counts"].decode("ascii")}
                          for m in pm]}
        if (k + 1) % 20 == 0 or k + 1 == len(tiles):
            print(f"  {os.path.basename(det_path)} {k+1}/{len(tiles)} {time.time()-t0:.0f}s",
                  flush=True)
    tmp = os.path.join(tempfile.gettempdir(), f"det_val_eval_{res}.json")
    with open(tmp, "w") as f:
        json.dump({"meta": {"mask_res": res}, "preds": preds}, f)
    c = S._score_one(tmp, ctx, res, S.MAX_DETS, 0.0)["canopy_neutral_crowd"]
    return {"det": det_path, "det_cfg": {k: v for k, v in cfg.items()},
            "n_det": n_det, "mask": c["mask"], "box": c["box"],
            "secs": round(time.time() - t0, 1)}, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--res", type=int, default=2048, choices=(512, 2048))
    a = ap.parse_args()
    res, tau = a.res, TAU[a.res]
    dst = os.path.join(HERE, f"det_val_eval_{a.name}_r{res}.json")
    preds_dst = os.path.join(HERE, f"preds_val_{a.name}_r{res}_tau{int(round(tau*100)):03d}.json")
    assert not os.path.exists(preds_dst), f"{preds_dst} exists -- refusing to overwrite"
    out = json.load(open(dst)) if os.path.exists(dst) else {
        "split": "val (108 tiles)", "res": res, "tau": tau, "prior_weight": PRIOR_WEIGHT,
        "kappa_scale": KAPPA_SCALE, "ranking": "detector score s",
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "numpy": np.__version__, "torch": torch.__version__}}
    assert out["res"] == res and "candidate" not in out, f"{dst} already complete"

    gt = json.load(open(VAL_GT))
    tiles = sorted(t for t in gt if os.path.exists(os.path.join(FEAT_DIR, t + ".npy")))
    assert len(tiles) == 108, len(tiles)
    gt = {t: gt[t] for t in tiles}
    gt_fp, tid2img, tree_rles, canopy_union = S.build_gt(gt, res, with_canopy=False)
    gt_cr, _, _, _ = S.build_gt(gt, res, with_canopy=True)
    ctx = (gt_fp, gt_cr, tid2img, tree_rles, canopy_union, tiles)
    masker = E.TCDMasker(EM_PATH)

    if "baseline" not in out:
        base, _ = run_arm(DET_PATH, tiles, gt, ctx, masker, res, tau)
        ref_fp, cell = GATE_REF[res]
        ref = json.load(open(os.path.join(HERE, ref_fp)))["cells"][cell]["AP50"]
        ok = abs(base["mask"]["AP50"] - ref) <= 1e-4 and (res != 2048 or base["n_det"] == 40058)
        out["baseline"] = base
        out["gate"] = {"ref_file": ref_fp, "cell": cell, "ref_AP50": ref,
                       "got_AP50": base["mask"]["AP50"], "n_det": base["n_det"], "pass": ok}
        json.dump(out, open(dst, "w"), indent=2)
        print(f"GATE baseline AP50 {base['mask']['AP50']} vs {ref} n_det {base['n_det']} -> "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            raise SystemExit("gate failed")
    cand, preds = run_arm(a.det, tiles, gt, ctx, masker, res, tau)
    out["candidate"] = cand
    out["delta"] = {g: {k: round(cand[g][k] - out["baseline"][g][k], 4)
                        for k in ("AP50", "AP75", "AP50_95")} for g in ("mask", "box")}
    with open(preds_dst, "w") as f:
        json.dump({"meta": {"mask_res": res, "det": a.det, "tau": tau}, "preds": preds}, f)
    json.dump(out, open(dst, "w"), indent=2)
    print(json.dumps({"baseline_mask": out["baseline"]["mask"], "candidate_mask": cand["mask"],
                      "delta": out["delta"], "n_det": cand["n_det"]}, indent=1),
          f"\n-> {dst}", flush=True)


if __name__ == "__main__":
    main()
