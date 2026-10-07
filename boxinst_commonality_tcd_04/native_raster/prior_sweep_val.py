"""Is the spatial prior too strong? alpha below the old grid edge, plus a no-prior arm. VAL ONLY.

The Aug-2026 val grid (BOX2MASK_LEVERS.md) tested alpha in {0.3, 0.4, 0.5, 0.7, 1.0} at 512 and
picked 0.3, the grid EDGE, with a monotone trend (lower alpha better at every kappa). Nothing
below 0.3 was tried, and alpha was never re-selected at 2048 (Phase B not run).

Hypothesis (Tom, 2026-10-07): the prior is learned on GT boxes, so it encodes a centred,
box-filling crown and is indexed by position INSIDE the predicted box; on a misplaced box it
pushes the mask away from the truth.

Arms, seed-0 published detector, 108 val tiles, 2048, local features:
  alpha in {0, 0.1, 0.2, 0.3} x kappa in {1.6, 2.0}          (the alpha-scaled fg-mass logit)
  noprior x kappa in {1.6, 2.0}: pi replaced by 0.5/K everywhere -> per-prototype spatial
      weights uniform AND fg-mass logit 0. Inference-time removal only; the EM fit (prototypes
      C) still had the prior. Approximates, not equals, a model fitted without one.
  each x tau in {0.30, 0.35, 0.40, 0.45, 0.50} x ranking {s, s*bimod*pfg_mean}.

GATE: (alpha 0.3, kappa 1.6, tau 0.40) must reproduce 0.6045 (s) and 0.6427 (product) and
40,058 dets (det_val_eval / det_val_product). The renderer below is E.pred_instance_masks'
body with the threshold applied to one resized probability map at several taus; the gate
proves equivalence at tau 0.40.

Stage 1 (pool over tiles) writes one pickle per tile to feat_cache/prior_sweep_val_r2048_tiles/ and
resumes from them. Stage 2 (pool over cells) scores with score_coco and checkpoints every cell
into prior_sweep_val_r2048.json. Refuses to overwrite a finished output.

    .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.prior_sweep_val --workers 12
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import pickle
import platform
import tempfile
import time

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"                                # one BLAS thread per worker

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TILE_DIR = "/Users/tompitts/dphil/feat_cache/prior_sweep_val_r2048_tiles"   # ~0.6 GB intermediates, outside git
OUT = os.path.join(HERE, "prior_sweep_val_r2048.json")
RES = 2048
ALPHAS = (0.0, 0.1, 0.2, 0.3)
KAPPAS = (1.6, 2.0)
TAUS = (0.30, 0.35, 0.40, 0.45, 0.50)
CONFIGS = [(f"a{a:g}_k{k:g}", a, k) for a in ALPHAS for k in KAPPAS] + \
          [(f"noprior_k{k:g}", None, k) for k in KAPPAS]
GATE = {"cfg": "a0.3_k1.6", "tau": 0.40, "s": 0.6045, "product": 0.6427, "n_det": 40058}

_W = {}


def _init_tile_worker():
    import torch
    torch.set_num_threads(1)
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.phase4_lib_tcd import Detector4Phase
    from boxinst_commonality_tcd_04.native_raster.oracle_boxes_val import DET_PATH, EM_PATH
    ck = torch.load(DET_PATH, map_location="cpu", weights_only=False)
    m = Detector4Phase(ck["cfg"]["in_dim"], width=ck["cfg"]["width"], tower=ck["cfg"]["tower"]).eval()
    m.load_state_dict(ck["state"])
    masker = E.TCDMasker(EM_PATH)
    flat = copy.copy(masker)
    flat.pi = np.full_like(masker.pi, 0.5 / masker.pi.shape[1])
    _W.update(model=m, masker=masker, flat=flat)


def render(masker, b, idx, r, g, taus):
    """E.pred_instance_masks' per-box body, thresholded at several taus."""
    from PIL import Image
    scale = 2048.0 / RES
    origin = getattr(masker, "origin", getattr(masker, "s", 2 * scale) / 2.0)
    s = getattr(masker, "s", 2 * scale)
    delta = int(round((origin - s / 2.0) / scale))
    grid = np.zeros(g * g, np.float32); grid[idx] = r
    prob = np.array(Image.fromarray(grid.reshape(g, g)).resize((RES, RES), Image.BILINEAR))
    if delta:
        shifted = np.zeros_like(prob)
        src = slice(max(0, -delta), RES - max(0, delta))
        dst = slice(max(0, delta), RES - max(0, -delta))
        shifted[dst, dst] = prob[src, src]
        prob = shifted
    x0, y0, x1, y1 = (np.asarray(b) / scale)
    box_m = np.zeros((RES, RES), bool)
    box_m[max(0, int(y0)):int(np.ceil(y1)), max(0, int(x0)):int(np.ceil(x1))] = True
    return [(prob >= t) & box_m for t in taus]


def tile_job(args):
    tid, feat_dir = args
    dst = os.path.join(TILE_DIR, tid + ".pkl")
    if os.path.exists(dst):
        return tid, 0.0
    import torch
    from pycocotools import mask as maskUtils
    from boxinst_commonality_tcd_04.detector import STRIDE8
    from dapt.decode import decode
    t0 = time.time()
    feat = np.load(os.path.join(feat_dir, tid + ".npy")).astype(np.float32)
    with torch.no_grad():
        det = _W["model"](torch.from_numpy(feat)[None])
    bx, sc = decode(det, score_thr=0.05, stride=STRIDE8, topk=600)
    bx, sc = bx.numpy(), sc.numpy()
    masker = _W["masker"]
    zn, g = masker.project(feat), feat.shape[-1]
    del feat
    out = {"boxes": bx.tolist(), "scores": sc.tolist(), "cfg": {}}
    for name, a, k in CONFIGS:
        mk = _W["flat"] if a is None else masker
        aw = 0.0 if a is None else a
        bim, pfm, rles = [], [], {t: [] for t in TAUS}
        for b in bx:
            idx, r = mk.box_mask(zn, g, b, prior_weight=aw, kappa_scale=k)
            bim.append(float(np.abs(2 * r - 1).mean())); pfm.append(float(r.mean()))
            for t, m in zip(TAUS, render(mk, b, idx, r, g, TAUS)):
                rles[t].append(maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode("ascii"))
        out["cfg"][name] = {"bimod": bim, "pfg_mean": pfm, "rles": rles}
    with open(dst + ".tmp", "wb") as f:
        pickle.dump(out, f)
    os.replace(dst + ".tmp", dst)
    return tid, time.time() - t0


_C = {}


def _init_score_worker(tids):
    from boxinst_commonality_tcd_04 import score_coco as S
    from boxinst_commonality_tcd_04.native_raster.oracle_boxes_val import VAL_GT
    gt = json.load(open(VAL_GT)); gt = {t: gt[t] for t in tids}
    gt_fp, tid2img, tree_rles, canopy_union = S.build_gt(gt, RES, with_canopy=False)
    gt_cr, _, _, _ = S.build_gt(gt, RES, with_canopy=True)
    _C.update(ctx=(gt_fp, gt_cr, tid2img, tree_rles, canopy_union, tids), tids=tids)


def score_job(cell):
    name, tau = cell
    from boxinst_commonality_tcd_04 import score_coco as S
    preds_s, preds_p = {}, {}
    n_det = 0
    for tid in _C["tids"]:
        T = pickle.load(open(os.path.join(TILE_DIR, tid + ".pkl"), "rb"))
        c = T["cfg"][name]
        s = np.asarray(T["scores"], np.float64)
        prod = (s * np.asarray(c["bimod"]) * np.asarray(c["pfg_mean"])).tolist() if len(s) else []
        n_det += len(s)
        base = {"boxes_2048": T["boxes"], "canopy_ignore": [False] * len(s),
                "masks_rle": [{"size": [RES, RES], "counts": x} for x in c["rles"][tau]]}
        preds_s[tid] = {**base, "scores": T["scores"]}
        preds_p[tid] = {**base, "scores": [round(float(v), 8) for v in prod]}
    res = {"n_det": n_det}
    for rank, P in (("s", preds_s), ("product", preds_p)):
        fd, tmp = tempfile.mkstemp(suffix=".json"); os.close(fd)
        with open(tmp, "w") as f:
            json.dump({"meta": {"mask_res": RES}, "preds": P}, f)
        c = S._score_one(tmp, _C["ctx"], RES, S.MAX_DETS, 0.0)["canopy_neutral_crowd"]
        os.remove(tmp)
        res[rank] = {"mask": c["mask"], "box": c["box"]}
    return name, tau, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    from boxinst_commonality_tcd_04.native_raster.oracle_boxes_val import FEAT_DIR, VAL_GT
    out = json.load(open(OUT)) if os.path.exists(OUT) else {
        "question": "spatial prior too strong? alpha below 0.3 and an inference-time no-prior arm",
        "split": "val (108 tiles)", "res": RES, "detector": "published seed 0",
        "configs": [c[0] for c in CONFIGS], "taus": list(TAUS),
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "numpy": np.__version__}, "cells": {}}
    assert "done" not in out, f"{OUT} complete -- refusing to overwrite"
    gt = json.load(open(VAL_GT))
    tids = sorted(t for t in gt if os.path.exists(os.path.join(FEAT_DIR, t + ".npy")))
    assert len(tids) == 108
    os.makedirs(TILE_DIR, exist_ok=True)
    ctx = mp.get_context("spawn")

    t0 = time.time(); todo = [t for t in tids if not os.path.exists(os.path.join(TILE_DIR, t + ".pkl"))]
    print(f"[stage1] {len(todo)}/108 tiles to do, {len(CONFIGS)} configs x {len(TAUS)} taus, "
          f"{a.workers} workers", flush=True)
    with ctx.Pool(a.workers, initializer=_init_tile_worker) as pool:
        for n, (tid, secs) in enumerate(pool.imap_unordered(tile_job, [(t, FEAT_DIR) for t in todo]), 1):
            if n % 6 == 0 or n == len(todo):
                print(f"  [stage1] {n}/{len(todo)} ({time.time()-t0:.0f}s, last {secs:.0f}s)", flush=True)

    cells = [(c[0], t) for c in CONFIGS for t in TAUS]
    gate_cell = (GATE["cfg"], GATE["tau"])
    cells.sort(key=lambda c: c != gate_cell)            # gate cell first
    todo = [c for c in cells if f"{c[0]}|{c[1]:.2f}" not in out["cells"]]
    print(f"[stage2] {len(todo)}/{len(cells)} cells to score", flush=True)
    t1 = time.time()
    with ctx.Pool(a.workers, initializer=_init_score_worker, initargs=(tids,)) as pool:
        for name, tau, res in pool.imap_unordered(score_job, todo):
            key = f"{name}|{tau:.2f}"
            out["cells"][key] = res
            if (name, tau) == gate_cell:
                ok = (abs(res["s"]["mask"]["AP50"] - GATE["s"]) <= 1e-4 and
                      abs(res["product"]["mask"]["AP50"] - GATE["product"]) <= 1e-4 and
                      res["n_det"] == GATE["n_det"])
                out["gate"] = {**GATE, "got_s": res["s"]["mask"]["AP50"],
                               "got_product": res["product"]["mask"]["AP50"],
                               "got_n_det": res["n_det"], "pass": ok}
                print(f"GATE {key}: s {res['s']['mask']['AP50']} product "
                      f"{res['product']['mask']['AP50']} dets {res['n_det']} -> "
                      f"{'PASS' if ok else 'FAIL'}", flush=True)
                if not ok:
                    json.dump(out, open(OUT, "w"), indent=2)
                    pool.terminate()
                    raise SystemExit("gate failed")
            json.dump(out, open(OUT, "w"), indent=2)
            print(f"  {key:22s} s {res['s']['mask']['AP50']:.4f}  product "
                  f"{res['product']['mask']['AP50']:.4f}  p.AP50:95 "
                  f"{res['product']['mask']['AP50_95']:.4f}  ({time.time()-t1:.0f}s)", flush=True)
    best = max(out["cells"].items(), key=lambda kv: kv[1]["product"]["mask"]["AP50"])
    out["best_by_product_AP50"] = {"cell": best[0], **best[1]["product"]["mask"]}
    out["done"] = True
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"BEST (product AP50): {best[0]} {best[1]['product']['mask']}", flush=True)


if __name__ == "__main__":
    main()
