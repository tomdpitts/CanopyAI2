"""Local (M4 Max) tau sweep on the 108 val tiles — the Modal version, minus the Modal.

WHY LOCAL. The Modal container managed ~34-57 ms/det; this machine does 6.2 ms/det (both
measured, not assumed), so a 40k-detection cell costs ~4 min here against ~38 min there.
Ten cells: ~40 min vs ~6 h. It also removes the two failure modes that have already cost us
a run each -- container preemption, and `modal run`'s ephemeral app dying with the client.

WHY IT IS LEGITIMATE TO SELECT HERE AND REPORT ELSEWHERE. tau is a HYPERPARAMETER CHOICE,
not a reported number. The environment-parity rule (see the project's environment notes)
binds the set of figures that go in a table: those must all come from one environment. It
does not bind where a scalar was chosen. The 439-tile test numbers stay wherever they were
computed; only the value of tau travels.

Both rasters are swept so the movement of the optimum is measured, and the 512 arm doubles
as a cross-environment check: Modal already selected (0.3, 1.6, 0.25) as best-of-15 at 512
with AP50 0.6392, so this arm should land on tau=0.25 at ~that value. Small numeric drift is
expected (Accelerate vs OpenBLAS, different numpy/PIL); a different ARGMAX is not, and would
mean the local arm cannot stand in for the Modal one.

alpha/kappa are FROZEN at their 512-selected (0.3, 1.6). Mechanism, not budget: box_mask()
computes the posterior on the 256-cell grid, byte-identical at both rasters, and the bilinear
resize + `>= tau` cut happen AFTER it. tau is the only scalar the raster can miscalibrate.
This bounds the mechanism, not the outcome -- the alpha/kappa optimum could still shift
because the scoring GT changes. Not re-selected; disclose.

Checkpoints after every cell to a local JSON, and resumes from it.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.val_knobs_local --res 2048
    .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.val_knobs_local --res 512
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

from boxinst_commonality_tcd_04 import evaluate as E
from boxinst_commonality_tcd_04 import score_coco as S
from boxinst_commonality_tcd_04.detector import STRIDE8
from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.phase4_lib_tcd import (
    Detector4Phase)
from dapt.decode import decode

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
CACHE = "/Users/tompitts/dphil/feat_cache"
FEAT_DIR = f"{CACHE}/feat_4p_val"
DET_PATH = f"{CACHE}/det_phase4_L24_s0.pt"
EM_PATH = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "em_model_4p_fix.npz")
VAL_GT = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "val_gt.json")

DEPLOYED = (0.30, 1.60, 0.25)
TAU_GRID = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = a.res
    dst = a.out or os.path.join(HERE, f"val_knobs_tau_r{res}_s0_local.json")

    gt = json.load(open(VAL_GT))
    tiles = sorted(t for t in gt if os.path.exists(os.path.join(FEAT_DIR, t + ".npy")))
    if a.limit:
        tiles = tiles[:a.limit]
    gt = {t: gt[t] for t in tiles}
    assert tiles, f"no features in {FEAT_DIR} — run pull.sh first"
    print(f"[local] {len(tiles)} val tiles with features (of {len(json.load(open(VAL_GT)))})",
          flush=True)

    ck = torch.load(DET_PATH, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = Detector4Phase(cfg["in_dim"], width=cfg["width"], tower=cfg["tower"]).eval()
    model.load_state_dict(ck["state"])
    masker = E.TCDMasker(EM_PATH)

    t0 = time.time()
    boxes, scores, zns, gs = {}, {}, {}, {}
    for k, tid in enumerate(tiles):
        feat = np.load(os.path.join(FEAT_DIR, tid + ".npy")).astype(np.float32)
        with torch.no_grad():
            det = model(torch.from_numpy(feat)[None])
        bx, sc = decode(det, score_thr=0.05, stride=STRIDE8, topk=600)
        boxes[tid], scores[tid] = bx.numpy(), sc.numpy()
        zns[tid], gs[tid] = masker.project(feat), feat.shape[-1]
        if (k + 1) % 20 == 0 or k + 1 == len(tiles):
            print(f"  detector {k+1}/{len(tiles)}  {time.time()-t0:.0f}s", flush=True)
    n_det = int(sum(len(v) for v in boxes.values()))
    print(f"[local] detector done: {n_det} dets, {time.time()-t0:.0f}s "
          f"(Modal gave 40058 — a mismatch here means the environments diverge)", flush=True)

    out = {"n_val_tiles": len(tiles), "n_det": n_det, "res": res,
           "deployed_at_512": DEPLOYED, "alpha_kappa": "FROZEN at the 512 selection",
           "criterion": "canopy-neutral (crowd) mask AP50 on the 108 val tiles",
           "environment": {"where": "local", "platform": platform.platform(),
                           "python": platform.python_version(),
                           "numpy": np.__version__, "torch": torch.__version__},
           "cells": {}}
    if os.path.exists(dst):
        prev = json.load(open(dst))
        if prev.get("n_val_tiles") == len(tiles) and prev.get("res") == res:
            out["cells"] = prev.get("cells", {})
            print(f"[local] resuming: {len(out['cells'])} cells done", flush=True)

    t1 = time.time()
    gt_fp, tid2img, tree_rles, canopy_union = S.build_gt(gt, res, with_canopy=False)
    gt_cr, _, _, _ = S.build_gt(gt, res, with_canopy=True)
    ctx = (gt_fp, gt_cr, tid2img, tree_rles, canopy_union, tiles)
    print(f"[local] GT rasterised at {res} in {time.time()-t1:.0f}s", flush=True)

    from pycocotools import mask as maskUtils
    tmp = os.path.join(tempfile.gettempdir(), f"cell_{res}.json")
    pw0, ks0, _ = DEPLOYED
    for thr in TAU_GRID:
        key = f"{thr:.2f}"
        if key in out["cells"]:
            print(f"  tau={key} -> cached {out['cells'][key]['AP50']:.4f}", flush=True)
            continue
        tc = time.time()
        preds = {}
        for tid in tiles:
            bx = boxes[tid]
            if len(bx) == 0:
                preds[tid] = {"boxes_2048": [], "scores": [], "canopy_ignore": [],
                              "masks_rle": []}
                continue
            pm = E.pred_instance_masks(masker, zns[tid], gs[tid], bx, res=res,
                                       scale=2048.0 / res, mask_thr=thr,
                                       prior_weight=pw0, kappa_scale=ks0)
            preds[tid] = {
                "boxes_2048": bx.tolist(), "scores": scores[tid].tolist(),
                "canopy_ignore": [False] * len(bx),
                "masks_rle": [{"size": [res, res],
                               "counts": maskUtils.encode(np.asfortranarray(
                                   m.astype(np.uint8)))["counts"].decode("ascii")}
                              for m in pm]}
        with open(tmp, "w") as f:
            json.dump({"meta": {"mask_res": res}, "preds": preds}, f)
        # score_floor 0.0: the 0.05 decode floor already ran on the detector score above
        r = S._score_one(tmp, ctx, res, S.MAX_DETS, 0.0)
        m = r["canopy_neutral_crowd"]["mask"]
        out["cells"][key] = {"prior_weight": pw0, "kappa_scale": ks0, "mask_thr": thr,
                             "AP50": m["AP50"], "AP75": m["AP75"],
                             "AP50_95": m["AP50_95"], "secs": round(time.time() - tc, 1)}
        with open(dst, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  tau={key} -> AP50 {m['AP50']:.4f} AP75 {m['AP75']:.4f} "
              f"AP50:95 {m['AP50_95']:.4f}  ({time.time()-tc:.0f}s) [saved]", flush=True)

    best = max(out["cells"].values(), key=lambda c: c["AP50"])
    out["best"] = best
    out["best_is_grid_edge"] = best["mask_thr"] in (min(TAU_GRID), max(TAU_GRID))
    if out["best_is_grid_edge"]:
        print("[local] WARNING: optimum on a grid EDGE — grid does not bracket the peak",
              flush=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[local] res {res} BEST: {best}\n[local] wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
