"""Oracle-localisation ceiling: how much would a better DETECTOR buy LACE, with the masker fixed?

QUESTION. At 2048, LACE's masker clears IoU 0.5 on 97.9% of crowns when handed GT boxes
(gtbox_bakeoff_r2048.json), yet loses 0.037 between box AP50 and mask AP50 on the 439 (Restor:
0.017), and its matched boxes are looser (mean box IoU ~0.73-0.75 vs Restor 0.80). This measures
the upper bound on fixing LOCALISATION alone.

DESIGN. 108 val tiles, seed-0 detector, deployed masker (alpha 0.3, kappa 1.6), local features.
Detections are decoded ONCE. Then each arm renders masks from a different box set and scores it:

  as_is          LACE's own boxes                                  (the reference)
  snap@T         every prediction matched to a GT crown at box IoU >= T (greedy by detector score,
                 each GT used once) has its box REPLACED by that GT box (polygon extent). Unmatched
                 predictions keep their own box. Detection SET, COUNT and SCORES are unchanged, so
                 recall and ranking are held fixed and only localisation improves.

Ranking is the DETECTOR score s in every arm, not the s*pbar*m product: the product's factors
come from the masker's posterior, which changes when the box changes, so ranking by it would
mix a ranking effect into a localisation effect. Report the ceiling as a DELTA vs as_is.

Default T values: 0.5 (only boxes that already count as detections) and 0.3 (also near-misses,
i.e. localisation failures that currently cost recall). The T=0.3 arm is the looser ceiling.

GATE. The as_is arm at res 2048, tau 0.40 is the same computation as cell "0.40" of
val_knobs_tau_r2048_s0_local.json (AP50 0.6045) and must reproduce it; at 512, tau 0.25, cell
"0.25" of val_knobs_tau_r512_s0_local.json (0.6392). A mismatch aborts before the oracle arms.

NOT a test-set number and not a claim: it bounds whether training a better detector is worth
GPU time. Checkpoints after every arm and resumes.

    .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.oracle_boxes_val            # 2048, tau 0.40
    .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.oracle_boxes_val --res 512  # tau 0.25
    ... --limit 5   # smoke (gate skipped)
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
from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.phase4_lib_tcd import Detector4Phase
from dapt.decode import decode

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
CACHE = "/Users/tompitts/dphil/feat_cache"
FEAT_DIR = f"{CACHE}/feat_4p_val"
DET_PATH = f"{CACHE}/det_phase4_L24_s0.pt"
EM_PATH = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "em_model_4p_fix.npz")
VAL_GT = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "val_gt.json")

PRIOR_WEIGHT, KAPPA_SCALE = 0.30, 1.60
TAU = {2048: 0.40, 512: 0.25}                       # val-selected per raster
GATE_REF = {2048: ("val_knobs_tau_r2048_s0_local.json", "0.40"),
            512: ("val_knobs_tau_r512_s0_local.json", "0.25")}


def gt_boxes(polys):
    if not polys:
        return np.zeros((0, 4), np.float64)
    return np.array([[*np.asarray(p, float).reshape(-1, 2).min(0),
                      *np.asarray(p, float).reshape(-1, 2).max(0)] for p in polys], np.float64)


def box_iou(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def snap(boxes, scores, gtb, thr):
    """Greedy by score, each GT used once: matched preds take the GT box. -> (boxes, n_snapped,
    mean box IoU of the snapped preds before snapping)."""
    out = boxes.astype(np.float64).copy()
    if not len(boxes) or not len(gtb):
        return out.astype(np.float32), 0, []
    iou = box_iou(out, gtb)
    taken = np.zeros(len(gtb), bool); before = []
    for i in np.argsort(-scores, kind="mergesort"):
        cand = np.where(taken, -1.0, iou[i])
        j = int(np.argmax(cand))
        if cand[j] >= thr:
            taken[j] = True; before.append(float(iou[i, j])); out[i] = gtb[j]
    return out.astype(np.float32), len(before), before


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=2048, choices=(512, 2048))
    ap.add_argument("--snap", default="0.5,0.3", help="box-IoU thresholds for the oracle arms")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res, tau = a.res, TAU[a.res]
    dst = a.out or os.path.join(HERE, f"oracle_boxes_val_r{res}{'_lim%d' % a.limit if a.limit else ''}.json")
    arms = ["as_is"] + [f"snap@{float(t):.2f}" for t in a.snap.split(",")]

    gt = json.load(open(VAL_GT))
    tiles = sorted(t for t in gt if os.path.exists(os.path.join(FEAT_DIR, t + ".npy")))
    if a.limit:
        tiles = tiles[:a.limit]
    gt = {t: gt[t] for t in tiles}
    assert len(tiles) == (a.limit or 108), f"expected 108 val tiles with features, got {len(tiles)}"

    out = {"question": "ceiling on mask AP from perfect LOCALISATION, masker + recall + ranking fixed",
           "res": res, "tau": tau, "prior_weight": PRIOR_WEIGHT, "kappa_scale": KAPPA_SCALE,
           "ranking": "detector score s in every arm", "n_val_tiles": len(tiles),
           "environment": {"platform": platform.platform(), "python": platform.python_version(),
                           "numpy": np.__version__, "torch": torch.__version__},
           "arms": {}}
    if os.path.exists(dst):
        prev = json.load(open(dst))
        assert (prev.get("res"), prev.get("tau"), prev.get("n_val_tiles")) == (res, tau, len(tiles)), \
            f"{dst} exists with a different config -- refusing to mix"
        out["arms"] = prev.get("arms", {})
        print(f"[oracle] resuming: {list(out['arms'])} done", flush=True)

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
    out["n_det"] = n_det
    print(f"[oracle] {n_det} dets (full val run gave 40058)", flush=True)
    if not a.limit:
        assert n_det == 40058, f"detector output drifted: {n_det} != 40058"

    gt_fp, tid2img, tree_rles, canopy_union = S.build_gt(gt, res, with_canopy=False)
    gt_cr, _, _, _ = S.build_gt(gt, res, with_canopy=True)
    ctx = (gt_fp, gt_cr, tid2img, tree_rles, canopy_union, tiles)
    gtb = {t: gt_boxes(gt[t]["trees"]) for t in tiles}

    from pycocotools import mask as maskUtils
    tmp = os.path.join(tempfile.gettempdir(), f"oracle_cell_{res}.json")
    for arm in arms:
        if arm in out["arms"]:
            print(f"  {arm}: cached AP50 {out['arms'][arm]['mask']['AP50']}", flush=True)
            continue
        tc = time.time(); preds = {}; n_snap = 0; before = []
        for tid in tiles:
            bx, sc = boxes[tid], scores[tid]
            if arm != "as_is":
                bx, ns, bf = snap(bx, sc, gtb[tid], float(arm.split("@")[1]))
                n_snap += ns; before += bf
            if len(bx) == 0:
                preds[tid] = {"boxes_2048": [], "scores": [], "canopy_ignore": [], "masks_rle": []}
                continue
            pm = E.pred_instance_masks(masker, zns[tid], gs[tid], bx, res=res, scale=2048.0 / res,
                                       mask_thr=tau, prior_weight=PRIOR_WEIGHT,
                                       kappa_scale=KAPPA_SCALE)
            preds[tid] = {"boxes_2048": np.asarray(bx).tolist(), "scores": sc.tolist(),
                          "canopy_ignore": [False] * len(bx),
                          "masks_rle": [{"size": [res, res], "counts": maskUtils.encode(
                              np.asfortranarray(m.astype(np.uint8)))["counts"].decode("ascii")}
                              for m in pm]}
        with open(tmp, "w") as f:
            json.dump({"meta": {"mask_res": res}, "preds": preds}, f)
        r = S._score_one(tmp, ctx, res, S.MAX_DETS, 0.0)     # 0.05 floor already ran at decode
        c = r["canopy_neutral_crowd"]
        out["arms"][arm] = {"mask": c["mask"], "box": c["box"], "n_snapped": n_snap,
                            "mean_box_iou_before_snap": round(float(np.mean(before)), 4) if before else None,
                            "secs": round(time.time() - tc, 1)}
        if arm == "as_is" and not a.limit:
            ref_fp, cell = GATE_REF[res]
            ref = json.load(open(os.path.join(HERE, ref_fp)))["cells"][cell]["AP50"]
            got = c["mask"]["AP50"]
            out["gate"] = {"ref_file": ref_fp, "cell": cell, "ref_AP50": ref, "got_AP50": got,
                           "pass": abs(got - ref) <= 1e-4}
            print(f"  GATE as_is AP50 {got} vs {ref_fp}[{cell}] {ref} -> "
                  f"{'PASS' if out['gate']['pass'] else 'FAIL'}", flush=True)
            if not out["gate"]["pass"]:
                json.dump(out, open(dst, "w"), indent=2)
                raise SystemExit("gate failed: the as_is arm does not reproduce the tau sweep")
        json.dump(out, open(dst, "w"), indent=2)
        print(f"  {arm}: mask AP50 {c['mask']['AP50']:.4f} AP75 {c['mask']['AP75']:.4f} "
              f"AP50:95 {c['mask']['AP50_95']:.4f} | box AP50 {c['box']['AP50']:.4f} | "
              f"snapped {n_snap} ({time.time()-tc:.0f}s) [saved]", flush=True)

    base = out["arms"]["as_is"]["mask"]
    out["delta_vs_as_is"] = {arm: {k: round(v["mask"][k] - base[k], 4) for k in ("AP50", "AP75", "AP50_95")}
                             for arm, v in out["arms"].items() if arm != "as_is"}
    json.dump(out, open(dst, "w"), indent=2)
    print(json.dumps(out["delta_vs_as_is"], indent=1), f"\n-> {dst}", flush=True)


if __name__ == "__main__":
    main()
