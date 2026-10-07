"""Colour-guided upsampling: extended grid, split-half check, and the 512 protocol raster. VAL ONLY.

Follows guided_rgb_val.py (untouched), whose best cell (colour r8, eps 1e-4, tau 0.35: product
AP50 0.6610 vs 0.6427 bilinear) sat on three grid edges. This script:
  (1) extends the grid past those edges;
  (2) scores every cell on the full 108 val tiles AND on two fixed halves (sorted tile ids, even /
      odd positions), so a setting can be selected on one half and reported on the other;
  (3) runs the same at --res 512 (the paper's protocol raster): posterior resized to 512 as
      deployed, RGB guide area-downsampled to 512, radius in 512-px units.

Isolated: new file, new outputs, intermediates outside git; filter/render code re-stated from
guided_rgb_val.py. Filter context margin = max(16, r) px (16 for r <= 16, i.e. unchanged for
the re-gated r8 cell).

GATES. 2048: bilinear tau 0.40 -> 0.6045 (s) / 0.6427 (product) / 40,058 dets, and colour r8 eps
1e-4 tau 0.35 -> 0.6610 (product). 512: bilinear tau 0.25 -> 0.6392 (s; cell 0.25 of
val_knobs_tau_r512_s0_local.json).

    .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.guided_rgb_ext_val --res 2048
    .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.guided_rgb_ext_val --res 512
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
ALPHA, KAPPA = 0.30, 1.60

CFG = {
    2048: {"taus": (0.25, 0.30, 0.35, 0.40),
           "filters": [("bilinear", None, None)] +
                      [(f"cgf_r{r}_e{e:g}", r, e) for r in (8, 12, 16, 24) for e in (1e-5, 1e-4, 1e-3)],
           "gates": [("bilinear", 0.40, "s", 0.6045), ("bilinear", 0.40, "product", 0.6427),
                     ("cgf_r8_e0.0001", 0.35, "product", 0.6610)],
           "n_det": 40058},
    512: {"taus": (0.20, 0.25, 0.30, 0.35),
          "filters": [("bilinear", None, None)] +
                     [(f"cgf_r{r}_e{e:g}", r, e) for r in (1, 2, 3, 4, 6) for e in (1e-5, 1e-4, 1e-3)],
          "gates": [("bilinear", 0.25, "s", 0.6392)],
          "n_det": 40058},
}

_W = {}


def tile_dir(res):
    return f"/Users/tompitts/dphil/feat_cache/guided_rgb_ext_val_r{res}_tiles"   # outside git


def out_path(res):
    return os.path.join(HERE, f"guided_rgb_ext_val_r{res}.json")


def color_guided_filter(I, p, r, eps):
    """He et al. 2010, colour guide (as guided_rgb_val.py)."""
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


def _init_tile_worker(res):
    import torch
    torch.set_num_threads(1)
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.phase4_lib_tcd import Detector4Phase
    from boxinst_commonality_tcd_04.native_raster.oracle_boxes_val import DET_PATH, EM_PATH
    ck = torch.load(DET_PATH, map_location="cpu", weights_only=False)
    m = Detector4Phase(ck["cfg"]["in_dim"], width=ck["cfg"]["width"], tower=ck["cfg"]["tower"]).eval()
    m.load_state_dict(ck["state"])
    _W.update(model=m, masker=E.TCDMasker(EM_PATH), res=res,
              man=json.load(open(MANIFEST))["feat_traintile"])


def tile_job(args):
    tid, feat_dir = args
    res = _W["res"]; C = CFG[res]; taus = C["taus"]
    dst = os.path.join(tile_dir(res), tid + ".pkl")
    if os.path.exists(dst):
        return tid, 0.0
    import torch
    from PIL import Image
    from pycocotools import mask as maskUtils
    from boxinst_commonality_tcd_04.detector import STRIDE8
    from dapt.decode import decode
    t0 = time.time()
    rgb8 = load_rgb_verified(tid, _W["man"])
    if res != 2048:
        rgb8 = np.asarray(Image.fromarray(rgb8).resize((res, res), Image.BOX))
    rgb = rgb8.astype(np.float32) / 255.0
    feat = np.load(os.path.join(feat_dir, tid + ".npy")).astype(np.float32)
    with torch.no_grad():
        det = _W["model"](torch.from_numpy(feat)[None])
    bx, sc = decode(det, score_thr=0.05, stride=STRIDE8, topk=600)
    bx, sc = bx.numpy(), sc.numpy()
    masker = _W["masker"]
    zn, g = masker.project(feat), feat.shape[-1]
    del feat
    scale = 2048.0 / res
    origin = getattr(masker, "origin", getattr(masker, "s", 2 * scale) / 2.0)
    s_px = getattr(masker, "s", 2 * scale)
    delta = int(round((origin - s_px / 2.0) / scale))
    out = {"boxes": bx.tolist(), "scores": sc.tolist(), "bimod": [], "pfg_mean": [],
           "rles": {f[0]: {t: [] for t in taus} for f in C["filters"]}}
    for b in bx:
        idx, r = masker.box_mask(zn, g, b, prior_weight=ALPHA, kappa_scale=KAPPA)
        out["bimod"].append(float(np.abs(2 * r - 1).mean())); out["pfg_mean"].append(float(r.mean()))
        grid = np.zeros(g * g, np.float32); grid[idx] = r              # E.pred_instance_masks body
        prob = np.array(Image.fromarray(grid.reshape(g, g)).resize((res, res), Image.BILINEAR))
        if delta:
            shifted = np.zeros_like(prob)
            src = slice(max(0, -delta), res - max(0, delta))
            dst_ = slice(max(0, delta), res - max(0, -delta))
            shifted[dst_, dst_] = prob[src, src]
            prob = shifted
        x0, y0, x1, y1 = (np.asarray(b) / scale)
        by0, by1 = max(0, int(y0)), int(np.ceil(y1))
        bx0, bx1 = max(0, int(x0)), int(np.ceil(x1))
        clip = np.zeros((res, res), bool); clip[by0:by1, bx0:bx1] = True
        for name, rad, eps in C["filters"]:
            mg = 16 if rad is None else max(16, rad) if res == 2048 else max(4, rad)
            wy0, wy1 = max(0, by0 - mg), min(res, by1 + mg)
            wx0, wx1 = max(0, bx0 - mg), min(res, bx1 + mg)
            q = (prob[wy0:wy1, wx0:wx1] if rad is None else
                 color_guided_filter(rgb[wy0:wy1, wx0:wx1], prob[wy0:wy1, wx0:wx1], rad, eps))
            for t in taus:
                m = np.zeros((res, res), bool); m[wy0:wy1, wx0:wx1] = q >= t
                out["rles"][name][t].append(maskUtils.encode(
                    np.asfortranarray((m & clip).astype(np.uint8)))["counts"].decode("ascii"))
    with open(dst + ".tmp", "wb") as f:
        pickle.dump(out, f)
    os.replace(dst + ".tmp", dst)
    return tid, time.time() - t0


_C = {}


def _init_score_worker(tids, res):
    from boxinst_commonality_tcd_04 import score_coco as S
    from boxinst_commonality_tcd_04.native_raster.oracle_boxes_val import VAL_GT
    gt = json.load(open(VAL_GT))
    halves = {"all": tids, "A": tids[0::2], "B": tids[1::2]}
    ctxs = {}
    for h, ts in halves.items():
        sub = {t: gt[t] for t in ts}
        gt_fp, tid2img, tree_rles, canopy_union = S.build_gt(sub, res, with_canopy=False)
        gt_cr, _, _, _ = S.build_gt(sub, res, with_canopy=True)
        ctxs[h] = (gt_fp, gt_cr, tid2img, tree_rles, canopy_union, ts)
    _C.update(ctxs=ctxs, halves=halves, res=res)


def score_job(cell):
    arm, tau = cell
    from boxinst_commonality_tcd_04 import score_coco as S
    res = _C["res"]
    Ps, Pp, n_det = {}, {}, 0
    for tid in _C["halves"]["all"]:
        T = pickle.load(open(os.path.join(tile_dir(res), tid + ".pkl"), "rb"))
        s = np.asarray(T["scores"], np.float64); n_det += len(s)
        prod = (s * np.asarray(T["bimod"]) * np.asarray(T["pfg_mean"])).tolist() if len(s) else []
        base = {"boxes_2048": T["boxes"], "canopy_ignore": [False] * len(s),
                "masks_rle": [{"size": [res, res], "counts": x} for x in T["rles"][arm][tau]]}
        Ps[tid] = {**base, "scores": T["scores"]}
        Pp[tid] = {**base, "scores": [round(float(v), 8) for v in prod]}
    out = {"n_det": n_det}
    for h, ts in _C["halves"].items():
        out[h] = {}
        for rank, P in (("s", Ps), ("product", Pp)):
            fd, tmp = tempfile.mkstemp(suffix=".json"); os.close(fd)
            with open(tmp, "w") as f:
                json.dump({"meta": {"mask_res": res}, "preds": {t: P[t] for t in ts}}, f)
            c = S._score_one(tmp, _C["ctxs"][h], res, S.MAX_DETS, 0.0)["canopy_neutral_crowd"]
            os.remove(tmp)
            out[h][rank] = {"mask": c["mask"], "box": c["box"]}
    return arm, tau, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, required=True, choices=(512, 2048))
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    res = a.res; C = CFG[res]; OUT = out_path(res); TD = tile_dir(res)
    from boxinst_commonality_tcd_04.native_raster.oracle_boxes_val import FEAT_DIR, VAL_GT
    out = json.load(open(OUT)) if os.path.exists(OUT) else {
        "question": "colour-guided upsampling: extended grid + split-half + raster", "res": res,
        "split": "val (108 tiles); halves A = sorted ids [0::2], B = [1::2]",
        "alpha": ALPHA, "kappa": KAPPA, "arms": [f[0] for f in C["filters"]], "taus": list(C["taus"]),
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "numpy": np.__version__}, "cells": {}}
    assert "done" not in out, f"{OUT} complete -- refusing to overwrite"
    gt = json.load(open(VAL_GT))
    tids = sorted(t for t in gt if os.path.exists(os.path.join(FEAT_DIR, t + ".npy")))
    assert len(tids) == 108
    os.makedirs(TD, exist_ok=True)
    ctx = mp.get_context("spawn"); t0 = time.time()
    todo = [t for t in tids if not os.path.exists(os.path.join(TD, t + ".pkl"))]
    print(f"[stage1 r{res}] {len(todo)}/108 tiles, {len(C['filters'])} arms x {len(C['taus'])} taus",
          flush=True)
    with ctx.Pool(a.workers, initializer=_init_tile_worker, initargs=(res,)) as pool:
        for n, (tid, secs) in enumerate(pool.imap_unordered(tile_job, [(t, FEAT_DIR) for t in todo]), 1):
            if n % 12 == 0 or n == len(todo):
                print(f"  [stage1 r{res}] {n}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
    cells = [(f[0], t) for f in C["filters"] for t in C["taus"]]
    gate_cells = {(g[0], g[1]) for g in C["gates"]}
    cells.sort(key=lambda c: c not in gate_cells)
    todo = [c for c in cells if f"{c[0]}|{c[1]:.2f}" not in out["cells"]]
    print(f"[stage2 r{res}] {len(todo)}/{len(cells)} cells", flush=True)
    with ctx.Pool(a.workers, initializer=_init_score_worker, initargs=(tids, res)) as pool:
        for arm, tau, r in pool.imap_unordered(score_job, todo):
            key = f"{arm}|{tau:.2f}"
            out["cells"][key] = r
            for g_arm, g_tau, g_rank, g_ref in C["gates"]:
                if (arm, tau) == (g_arm, g_tau):
                    got = r["all"][g_rank]["mask"]["AP50"]
                    ok = abs(got - g_ref) <= 1e-4 and r["n_det"] == C["n_det"]
                    out.setdefault("gates", {})[f"{key}|{g_rank}"] = {"ref": g_ref, "got": got,
                                                                     "n_det": r["n_det"], "pass": ok}
                    print(f"GATE r{res} {key} {g_rank}: {got} vs {g_ref}, dets {r['n_det']} -> "
                          f"{'PASS' if ok else 'FAIL'}", flush=True)
                    if not ok:
                        json.dump(out, open(OUT, "w"), indent=2); pool.terminate()
                        raise SystemExit("gate failed")
            json.dump(out, open(OUT, "w"), indent=2)
    # split-half: select on one half by product AP50, report on the other
    sel = {}
    for pick, rep in (("A", "B"), ("B", "A")):
        k = max(out["cells"], key=lambda c: out["cells"][c][pick]["product"]["mask"]["AP50"])
        base = f"bilinear|{(0.40 if res == 2048 else 0.25):.2f}"
        sel[f"select_{pick}_report_{rep}"] = {
            "cell": k, "selected_on": out["cells"][k][pick]["product"]["mask"],
            "reported_on": out["cells"][k][rep]["product"]["mask"],
            "deployed_on_report_half": out["cells"][base][rep]["product"]["mask"],
            "gain_on_report_half_AP50": round(out["cells"][k][rep]["product"]["mask"]["AP50"]
                                              - out["cells"][base][rep]["product"]["mask"]["AP50"], 4)}
        print(f"SPLIT select {pick} -> {k}; on {rep}: gain AP50 "
              f"{sel[f'select_{pick}_report_{rep}']['gain_on_report_half_AP50']:+.4f}", flush=True)
    out["split_half"] = sel
    best = max(out["cells"], key=lambda c: out["cells"][c]["all"]["product"]["mask"]["AP50"])
    out["best_all_product_AP50"] = {"cell": best, **out["cells"][best]["all"]["product"]["mask"]}
    out["done"] = True
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"BEST r{res} {best} {out['best_all_product_AP50']}", flush=True)


if __name__ == "__main__":
    main()
