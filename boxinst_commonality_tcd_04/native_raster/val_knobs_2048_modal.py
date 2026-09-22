"""Modal CPU app: re-select the three inference scalars on the 108 VAL tiles, at 2048.

WHY. alpha=0.3, kappa=1.6 and mask_thr=0.25 were all selected on these same 108 validation
images **at the 512 raster** (paper Section 3.2.5). Scoring at 2048 changes where the
posterior cut lands: `pred_instance_masks` bilinearly upsamples the 256-cell posterior to the
target resolution and thresholds AFTERWARDS, so an 8x ramp (2048) resolves the cell boundary
differently from a 2x ramp (512). Measured consequence on the 439 at the deployed knobs:
median mask area falls 1072 -> 966 tile-px^2 on a byte-identical detection set. Masks
under-cover at 2048, which is exactly the signature of a threshold calibrated for a coarser
raster.

Comparing a 2048-calibrated Restor against a 512-calibrated LACE would be the wrong
experiment, so the knobs are re-selected here under the SAME rule as originally used --
select on the 108 val tiles, report on the 439 test tiles, val mask AP50 as the criterion.
Both rasters are swept so the movement of the optimum is visible rather than asserted.

SELECTION RULE, FIXED BEFORE THE RUN (so this is not a search for a flattering number):
  sweep mask_thr at the deployed (alpha, kappa) = (0.3, 1.6); alpha/kappa FROZEN
  criterion: canopy-neutral (crowd) mask AP50 on the 108 val tiles
The 512 arm was run first and RE-SELECTED THE DEPLOYED (0.3, 1.6, 0.25) exactly, best of
all 15 cells -- which validates the harness and, incidentally, retro-justifies tau=0.25 on
a far wider grid than it was originally chosen on (the paper notes the validation grid only
ever spanned tau in {0.25, 0.30}). This app now sweeps tau at ONE raster, resumably.

NO GPU, NO RETRAINING. The detector runs on CPU over cached val features and its boxes are
computed ONCE and reused by every grid cell -- only the masker's rendering changes across
cells, so every cell scores an identical detection set.

Inputs (all already on tcd04-phase4-vol):
    feat_4p_train/{tile}.npy    the val tiles are a subset of the 900 training tiles
    out/det_phase4_L24_s0.pt    seed-0 detector
    out/em_model_4p_fix.npz     the deployed masker

Usage:
    modal run val_knobs_2048_modal.py::sweep --limit 6      # smoke
    modal run val_knobs_2048_modal.py::sweep
"""
import json
import os

import modal

APP_NAME = "tcd04-native-raster-valknobs"
VOL_NAME = "tcd04-phase4-vol"

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REPO = os.path.dirname(PKG)
PH4 = os.path.join(PKG, "modal_tcd_multiseed", "phase4")
STUBS = os.path.join(PH4, "stubs")

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME)

P = "/root/proj"
PKG_R = f"{P}/boxinst_commonality_tcd_04"
PH4_R = f"{PKG_R}/modal_tcd_multiseed/phase4"

FEAT_DIR = "/vol/feat_4p_train"
DET_PATH = "/vol/out/det_phase4_L24_s0.pt"
EM_PATH = "/vol/out/em_model_4p_fix.npz"
OUT_DIR = "/vol/out/native_raster"

DEPLOYED = (0.30, 1.60, 0.25)          # (prior_weight, kappa_scale, mask_thr) at 512
# Extended upward: at 2048 AP50 was still RISING at 0.35 when the first run died, so the
# original grid (built around the 512 optimum of 0.25) may not bracket the 2048 peak.
TAU_GRID = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.1", "torchvision==0.27.1", "numpy==2.2.6",
                 "transformers==4.57.1", "pillow", "contourpy", "pycocotools")
)
for rel in ("dapt/__init__.py", "dapt/backbone.py", "dapt/targets.py",
            "dapt/decode.py", "dapt/eval.py", "dapt/head.py"):
    image = image.add_local_file(os.path.join(REPO, rel), f"{P}/{rel}")
for rel in ("__init__.py", "detector.py", "train_detector_tiles.py", "evaluate.py",
            "em.py", "prepare_test.py", "cache_test.py", "cache_train_tiles.py",
            "score_coco.py", "test_gt.json"):
    image = image.add_local_file(os.path.join(PKG, rel), f"{PKG_R}/{rel}")
for rel in ("__init__.py",):
    image = image.add_local_file(os.path.join(PKG, "modal_tcd_multiseed", rel),
                                 f"{PKG_R}/modal_tcd_multiseed/{rel}")
for rel in ("__init__.py", "phase4_lib_tcd.py", "val_gt.json"):
    image = image.add_local_file(os.path.join(PH4, rel), f"{PH4_R}/{rel}")
STUB_FILES = {
    "boxinst": ("__init__.py", "cache_feats.py"),
    "boxinst_commonality": ("__init__.py", "em.py"),
    "boxinst_tcd": ("__init__.py", "build_canopy.py", "cache.py", "prepare.py"),
}
for _pkg, _files in STUB_FILES.items():
    for _f in _files:
        image = image.add_local_file(os.path.join(STUBS, _pkg, _f), f"{P}/{_pkg}/{_f}")


@app.function(image=image, volumes={"/vol": vol}, timeout=12 * 3600,
              cpu=4, memory=16384)
def sweep(res: int = 2048, limit: int = 0):
    import sys
    import tempfile
    import time

    import numpy as np
    import torch
    from pycocotools import mask as maskUtils

    sys.path.insert(0, P)
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04 import score_coco as S
    from boxinst_commonality_tcd_04.detector import STRIDE8
    from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.phase4_lib_tcd import (
        Detector4Phase)
    from dapt.decode import decode

    gt = json.load(open(f"{PH4_R}/val_gt.json"))
    tiles = sorted(t for t in gt if os.path.exists(os.path.join(FEAT_DIR, t + ".npy")))
    if limit:
        tiles = tiles[:limit]
    gt = {t: gt[t] for t in tiles}
    print(f"[valknobs] {len(tiles)} val tiles with features", flush=True)

    ck = torch.load(DET_PATH, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = Detector4Phase(cfg["in_dim"], width=cfg["width"], tower=cfg["tower"]).eval()
    model.load_state_dict(ck["state"])
    masker = E.TCDMasker(EM_PATH)

    # ---- one detector pass; boxes reused by EVERY grid cell -------------------------
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
    print(f"[valknobs] detector done: {n_det} dets, {time.time()-t0:.0f}s", flush=True)

    def cell(res, ctx, pw, ks, thr, tmp):
        preds = {}
        for tid in tiles:
            bx = boxes[tid]
            if len(bx) == 0:
                preds[tid] = {"boxes_2048": [], "scores": [], "canopy_ignore": [],
                              "masks_rle": []}
                continue
            pm = E.pred_instance_masks(masker, zns[tid], gs[tid], bx, res=res,
                                       scale=2048.0 / res, mask_thr=thr,
                                       prior_weight=pw, kappa_scale=ks)
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
        return {"prior_weight": pw, "kappa_scale": ks, "mask_thr": thr,
                "AP50": m["AP50"], "AP75": m["AP75"], "AP50_95": m["AP50_95"]}

    # ---- resumable, one raster, tau only -------------------------------------------
    # alpha/kappa are FROZEN at their 512-selected values. This is a mechanism argument,
    # not a budget one: box_mask() computes the posterior on the 256-cell grid, which is
    # byte-identical at both rasters, and the BILINEAR resize + `>= mask_thr` cut happen
    # AFTER it. tau is therefore the only scalar the scoring raster can miscalibrate.
    # Corroboration: the completed 512 arm's full 3x3 alpha/kappa sweep spanned
    # 0.6321-0.6392 (a 0.007 plateau) and re-selected the deployed (0.3, 1.6) exactly.
    # Limitation to disclose: this bounds the MECHANISM, not the outcome -- the alpha/kappa
    # optimum could still shift at 2048 because the scoring GT changes. Not re-selected.
    dst = f"{OUT_DIR}/val_knobs_tau_r{res}_s0.json"
    out = {"n_val_tiles": len(tiles), "n_det": n_det, "res": res,
           "deployed_at_512": DEPLOYED, "alpha_kappa": "FROZEN at the 512 selection",
           "criterion": "canopy-neutral (crowd) mask AP50 on the 108 val tiles",
           "cells": {}}
    if os.path.exists(dst):                       # resume after a preemption
        prev = json.load(open(dst))
        if prev.get("n_val_tiles") == len(tiles) and prev.get("res") == res:
            out["cells"] = prev.get("cells", {})
            print(f"[valknobs] resuming: {len(out['cells'])} cells already done",
                  flush=True)

    t1 = time.time()
    gt_fp, tid2img, tree_rles, canopy_union = S.build_gt(gt, res, with_canopy=False)
    gt_cr, _, _, _ = S.build_gt(gt, res, with_canopy=True)
    ctx = (gt_fp, gt_cr, tid2img, tree_rles, canopy_union, tiles)
    print(f"[valknobs] GT rasterised at {res} in {time.time()-t1:.0f}s", flush=True)

    tmp = os.path.join(tempfile.gettempdir(), "cell.json")
    pw0, ks0, _ = DEPLOYED
    os.makedirs(OUT_DIR, exist_ok=True)
    for thr in TAU_GRID:
        key = f"{thr:.2f}"
        if key in out["cells"]:
            print(f"  tau={key} -> cached {out['cells'][key]['AP50']:.4f}", flush=True)
            continue
        c = cell(res, ctx, pw0, ks0, thr, tmp)
        out["cells"][key] = c
        with open(dst, "w") as f:                 # checkpoint after EVERY cell
            json.dump(out, f, indent=2)
        vol.commit()
        print(f"  tau={key} -> AP50 {c['AP50']:.4f} AP75 {c['AP75']:.4f} "
              f"AP50:95 {c['AP50_95']:.4f}  [saved]", flush=True)

    best = max(out["cells"].values(), key=lambda c: c["AP50"])
    edge = best["mask_thr"] in (min(TAU_GRID), max(TAU_GRID))
    out["best"] = best
    out["best_is_grid_edge"] = edge
    if edge:
        print("[valknobs] WARNING: optimum sits on a grid EDGE -- grid does not bracket "
              "the peak, extend it before believing this value", flush=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    print(f"[valknobs] res {res} BEST: {best}\n[valknobs] wrote {dst}", flush=True)
    return out


@app.local_entrypoint()
def main(res: int = 2048, limit: int = 0):
    print(json.dumps(sweep.remote(res, limit), indent=2)[:4000])
