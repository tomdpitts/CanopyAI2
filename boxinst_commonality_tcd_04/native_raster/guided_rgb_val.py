"""COLOUR-guided upsampling of LACE's posterior at 2048. VAL ONLY. Isolated experiment.

Copy of guided_render_val.py (left untouched) with the guide switched from luminance to the full
RGB image (He et al. 2010, colour-guide form: per-pixel 3x3 covariance solve), because crown/soil
and crown/crown edges are often hue rather than brightness differences. Arms: bilinear (gate),
grey r4 eps1e-4 (re-gates the previous best, 0.6488 product @ tau 0.40), colour r in {2, 4, 8} x
eps in {1e-4, 1e-3, 1e-2}. Original docstring follows.

Guided (edge-aware) upsampling of LACE's posterior at 2048.

WHY. LACE's whole 2048 deficit to Restor rpn1000 is box->mask conversion at fine raster: box AP is
level (0.667 vs 0.665), and the same detections lose 0.029 going 512 -> 2048 while Restor gains
0.023. LACE's mask is an 8 px posterior upsampled BILINEARLY, so the boundary ignores the image.
Here the bilinear map is passed through a guided filter (He et al. 2010) with the RGB tile as
guide, so probability steps land on visible crown edges; then threshold + box clip as before.

NOTHING EXISTING IS EDITED. Self-contained except read-only library imports. Detector, masker,
knobs (alpha 0.3, kappa 1.6), decode, box clip and ranking are the deployed ones. Rendering is
E.pred_instance_masks' per-box body, re-stated here, followed by the optional filter.

Arms: baseline (no filter) and guided filter with r in {4, 8} px, eps in {1e-4, 1e-3, 1e-2}
(guide = luminance in [0,1]), each at tau in {0.35, 0.40, 0.45}; scored with ranking s and
s*bimod*pfg_mean (the posterior is untouched by the filter, so the product factors are too).

GATE: baseline at tau 0.40 reproduces 0.6045 (s) / 0.6427 (product) / 40,058 dets.
RGB: data/tcd/train/{tile}.tif, verified per tile against phase4 manifest.json by pixel sha1
(or by image_id where the manifest has no sha1) before use.

Outputs (new paths only): native_raster/guided_render_val_r2048.json, per-tile intermediates
feat_cache/guided_render_val_r2048_tiles/, log logs/guided_rgb_val_r2048.log. Resumes.

    .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.guided_rgb_val --workers 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import platform
import tempfile
import time

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REPO = os.path.dirname(PKG)
RGB_DIR = os.path.join(REPO, "data", "tcd", "train")
MANIFEST = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "manifest.json")
TILE_DIR = "/Users/tompitts/dphil/feat_cache/guided_rgb_val_r2048_tiles"   # outside git
OUT = os.path.join(HERE, "guided_rgb_val_r2048.json")
RES = 2048
ALPHA, KAPPA = 0.30, 1.60
TAUS = (0.35, 0.40, 0.45)
FILTERS = [("bilinear", None, None), ("gf_r4_e0.0001", 4, 1e-4)] + \
          [(f"cgf_r{r}_e{e:g}", r, e) for r in (2, 4, 8) for e in (1e-4, 1e-3, 1e-2)]
GATE = {"arm": "bilinear", "tau": 0.40, "s": 0.6045, "product": 0.6427, "n_det": 40058}
GATE2 = {"arm": "gf_r4_e0.0001", "tau": 0.40, "product": 0.6488}   # guided_render_val best cell

_W = {}


def guided_filter(I, p, r, eps):
    """He et al. 2010, grey guide. I, p float32 (H,W); box filters of radius r (reflect)."""
    from scipy.ndimage import uniform_filter
    k = 2 * r + 1
    mean = lambda x: uniform_filter(x, size=k, mode="reflect")
    mI, mp = mean(I), mean(p)
    a = (mean(I * p) - mI * mp) / (mean(I * I) - mI * mI + eps)
    b = mp - a * mI
    return mean(a) * I + mean(b)


def color_guided_filter(I, p, r, eps):
    """He et al. 2010, colour guide. I (H,W,3) float32 in [0,1]; p (H,W). Closed-form 3x3 solve."""
    from scipy.ndimage import uniform_filter
    k = 2 * r + 1
    mean = lambda x: uniform_filter(x, size=k, mode="reflect")
    R, G, B = I[..., 0], I[..., 1], I[..., 2]
    mR, mG, mB, mp = mean(R), mean(G), mean(B), mean(p)
    cR, cG, cB = mean(R * p) - mR * mp, mean(G * p) - mG * mp, mean(B * p) - mB * mp
    rr = mean(R * R) - mR * mR + eps; gg = mean(G * G) - mG * mG + eps; bb = mean(B * B) - mB * mB + eps
    rg = mean(R * G) - mR * mG; rb = mean(R * B) - mR * mB; gb = mean(G * B) - mG * mB
    i_rr = gg * bb - gb * gb; i_rg = gb * rb - rg * bb; i_rb = rg * gb - gg * rb
    i_gg = rr * bb - rb * rb; i_gb = rb * rg - rr * gb; i_bb = rr * gg - rg * rg
    det = i_rr * rr + i_rg * rg + i_rb * rb
    aR = (i_rr * cR + i_rg * cG + i_rb * cB) / det
    aG = (i_rg * cR + i_gg * cG + i_gb * cB) / det
    aB = (i_rb * cR + i_gb * cG + i_bb * cB) / det
    b = mp - aR * mR - aG * mG - aB * mB
    return mean(aR) * R + mean(aG) * G + mean(aB) * B + mean(b)


def load_rgb_verified(tid, man):
    from PIL import Image
    arr = np.asarray(Image.open(os.path.join(RGB_DIR, tid + ".tif")).convert("RGB"))
    rec = man[tid]
    if "rgb_sha1" in rec:
        assert hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest() == rec["rgb_sha1"], tid
    else:
        meta = json.load(open(os.path.join(RGB_DIR, tid + "_meta.json")))
        assert meta["image_id"] == rec["image_id"], tid
    assert arr.shape == (2048, 2048, 3), (tid, arr.shape)
    return arr


def _init_tile_worker():
    import torch
    torch.set_num_threads(1)
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.phase4_lib_tcd import Detector4Phase
    from boxinst_commonality_tcd_04.native_raster.oracle_boxes_val import DET_PATH, EM_PATH
    ck = torch.load(DET_PATH, map_location="cpu", weights_only=False)
    m = Detector4Phase(ck["cfg"]["in_dim"], width=ck["cfg"]["width"], tower=ck["cfg"]["tower"]).eval()
    m.load_state_dict(ck["state"])
    _W.update(model=m, masker=E.TCDMasker(EM_PATH),
              man=json.load(open(MANIFEST))["feat_traintile"])


def tile_job(args):
    tid, feat_dir = args
    dst = os.path.join(TILE_DIR, tid + ".pkl")
    if os.path.exists(dst):
        return tid, 0.0
    import torch
    from PIL import Image
    from pycocotools import mask as maskUtils
    from boxinst_commonality_tcd_04.detector import STRIDE8
    from dapt.decode import decode
    t0 = time.time()
    rgb = load_rgb_verified(tid, _W["man"]).astype(np.float32) / 255.0
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    feat = np.load(os.path.join(feat_dir, tid + ".npy")).astype(np.float32)
    with torch.no_grad():
        det = _W["model"](torch.from_numpy(feat)[None])
    bx, sc = decode(det, score_thr=0.05, stride=STRIDE8, topk=600)
    bx, sc = bx.numpy(), sc.numpy()
    masker = _W["masker"]
    zn, g = masker.project(feat), feat.shape[-1]
    del feat
    scale = 2048.0 / RES
    origin = getattr(masker, "origin", getattr(masker, "s", 2 * scale) / 2.0)
    s_px = getattr(masker, "s", 2 * scale)
    delta = int(round((origin - s_px / 2.0) / scale))
    out = {"boxes": bx.tolist(), "scores": sc.tolist(), "bimod": [], "pfg_mean": [],
           "rles": {f[0]: {t: [] for t in TAUS} for f in FILTERS}}
    for b in bx:
        idx, r = masker.box_mask(zn, g, b, prior_weight=ALPHA, kappa_scale=KAPPA)
        out["bimod"].append(float(np.abs(2 * r - 1).mean())); out["pfg_mean"].append(float(r.mean()))
        # --- E.pred_instance_masks body (bilinear prob, origin shift, box clip) ---
        grid = np.zeros(g * g, np.float32); grid[idx] = r
        prob = np.array(Image.fromarray(grid.reshape(g, g)).resize((RES, RES), Image.BILINEAR))
        if delta:
            shifted = np.zeros_like(prob)
            src = slice(max(0, -delta), RES - max(0, delta))
            dst_ = slice(max(0, delta), RES - max(0, -delta))
            shifted[dst_, dst_] = prob[src, src]
            prob = shifted
        x0, y0, x1, y1 = (np.asarray(b) / scale)
        by0, by1 = max(0, int(y0)), int(np.ceil(y1))
        bx0, bx1 = max(0, int(x0)), int(np.ceil(x1))
        # filter window: the clip region plus 16 px context, inside the tile
        wy0, wy1 = max(0, by0 - 16), min(RES, by1 + 16)
        wx0, wx1 = max(0, bx0 - 16), min(RES, bx1 + 16)
        for name, rad, eps in FILTERS:
            if rad is None:
                q_win = prob[wy0:wy1, wx0:wx1]
            elif name.startswith("cgf"):
                q_win = color_guided_filter(rgb[wy0:wy1, wx0:wx1], prob[wy0:wy1, wx0:wx1], rad, eps)
            else:
                q_win = guided_filter(lum[wy0:wy1, wx0:wx1], prob[wy0:wy1, wx0:wx1], rad, eps)
            for t in TAUS:
                m = np.zeros((RES, RES), bool)
                m[wy0:wy1, wx0:wx1] = q_win >= t
                clip = np.zeros((RES, RES), bool); clip[by0:by1, bx0:bx1] = True
                out["rles"][name][t].append(maskUtils.encode(
                    np.asfortranarray((m & clip).astype(np.uint8)))["counts"].decode("ascii"))
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
    arm, tau = cell
    from boxinst_commonality_tcd_04 import score_coco as S
    Ps, Pp, n_det = {}, {}, 0
    for tid in _C["tids"]:
        T = pickle.load(open(os.path.join(TILE_DIR, tid + ".pkl"), "rb"))
        s = np.asarray(T["scores"], np.float64); n_det += len(s)
        prod = (s * np.asarray(T["bimod"]) * np.asarray(T["pfg_mean"])).tolist() if len(s) else []
        base = {"boxes_2048": T["boxes"], "canopy_ignore": [False] * len(s),
                "masks_rle": [{"size": [RES, RES], "counts": x} for x in T["rles"][arm][tau]]}
        Ps[tid] = {**base, "scores": T["scores"]}
        Pp[tid] = {**base, "scores": [round(float(v), 8) for v in prod]}
    res = {"n_det": n_det}
    for rank, P in (("s", Ps), ("product", Pp)):
        fd, tmp = tempfile.mkstemp(suffix=".json"); os.close(fd)
        with open(tmp, "w") as f:
            json.dump({"meta": {"mask_res": RES}, "preds": P}, f)
        c = S._score_one(tmp, _C["ctx"], RES, S.MAX_DETS, 0.0)["canopy_neutral_crowd"]
        os.remove(tmp)
        res[rank] = {"mask": c["mask"], "box": c["box"]}
    return arm, tau, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    from boxinst_commonality_tcd_04.native_raster.oracle_boxes_val import FEAT_DIR, VAL_GT
    out = json.load(open(OUT)) if os.path.exists(OUT) else {
        "question": "COLOUR-guided upsampling of the posterior vs grey guide vs bilinear",
        "split": "val (108 tiles)", "res": RES, "alpha": ALPHA, "kappa": KAPPA,
        "arms": [f[0] for f in FILTERS], "taus": list(TAUS), "guide": "gf_*: luminance; cgf_*: RGB, [0,1]",
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "numpy": np.__version__}, "cells": {}}
    assert "done" not in out, f"{OUT} complete -- refusing to overwrite"
    gt = json.load(open(VAL_GT))
    tids = sorted(t for t in gt if os.path.exists(os.path.join(FEAT_DIR, t + ".npy")))
    assert len(tids) == 108
    os.makedirs(TILE_DIR, exist_ok=True)
    ctx = mp.get_context("spawn")
    t0 = time.time()
    todo = [t for t in tids if not os.path.exists(os.path.join(TILE_DIR, t + ".pkl"))]
    print(f"[stage1] {len(todo)}/108 tiles, {len(FILTERS)} arms x {len(TAUS)} taus, "
          f"{a.workers} workers", flush=True)
    with ctx.Pool(a.workers, initializer=_init_tile_worker) as pool:
        for n, (tid, secs) in enumerate(pool.imap_unordered(tile_job, [(t, FEAT_DIR) for t in todo]), 1):
            if n % 6 == 0 or n == len(todo):
                print(f"  [stage1] {n}/{len(todo)} ({time.time()-t0:.0f}s, last {secs:.0f}s)", flush=True)
    cells = [(f[0], t) for f in FILTERS for t in TAUS]
    cells.sort(key=lambda c: c != (GATE["arm"], GATE["tau"]))
    todo = [c for c in cells if f"{c[0]}|{c[1]:.2f}" not in out["cells"]]
    print(f"[stage2] {len(todo)}/{len(cells)} cells", flush=True)
    with ctx.Pool(a.workers, initializer=_init_score_worker, initargs=(tids,)) as pool:
        for arm, tau, res in pool.imap_unordered(score_job, todo):
            key = f"{arm}|{tau:.2f}"
            out["cells"][key] = res
            if (arm, tau) == (GATE["arm"], GATE["tau"]):
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
                    json.dump(out, open(OUT, "w"), indent=2); pool.terminate()
                    raise SystemExit("gate failed")
            json.dump(out, open(OUT, "w"), indent=2)
            print(f"  {key:20s} product AP50 {res['product']['mask']['AP50']:.4f} "
                  f"AP50:95 {res['product']['mask']['AP50_95']:.4f} | s AP50 "
                  f"{res['s']['mask']['AP50']:.4f}", flush=True)
    g2 = out["cells"][f"{GATE2['arm']}|{GATE2['tau']:.2f}"]["product"]["mask"]["AP50"]
    out["gate2"] = {**GATE2, "got": g2, "pass": abs(g2 - GATE2["product"]) <= 1e-4}
    print(f"GATE2 {GATE2['arm']}|{GATE2['tau']:.2f}: {g2} vs {GATE2['product']} -> "
          f"{'PASS' if out['gate2']['pass'] else 'FAIL'}", flush=True)
    best = max(out["cells"].items(), key=lambda kv: kv[1]["product"]["mask"]["AP50"])
    out["best_by_product_AP50"] = {"cell": best[0], **best[1]["product"]["mask"]}
    out["done"] = True
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"BEST {best[0]} {best[1]['product']['mask']}", flush=True)


if __name__ == "__main__":
    main()
