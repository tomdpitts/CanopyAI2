"""Modal CPU app: every LACE mask ablation re-run at native 2048 with the colour-guided render.

The paper is re-based on the native 2048 raster with the colour-guided render as LACE's standard
render (NATIVE_RASTER.md, DECISION 2026-10-07). Settings frozen as pre-registered: guide = tile
RGB, r 12 px, eps 1e-4, tau 0.35, alpha 0.3, kappa 1.6, box clip. Four functions, each writing
ONLY to a new directory and each with a gate against a published reference:

  sparse_render(seed)       sparse-236 slice. Boxes + product scores verbatim from
                            modal_sparse_tcd_multiseed/preds/ours_prod_s{seed}.json. Also emits the
                            bilinear tau 0.40 masks, which must equal the published
                            preds_lace_sparse_s{seed}_r2048_tau040.json (checked locally).
  cfg_render(seed, a, k)    tab:masker rows on the 439: masks with masker knobs (a, k) and the
                            product s * bimod * pfg_mean recomputed from THAT config's posterior
                            (s = detector score). Gate in container: at (0.3, 1.6) the recomputed
                            product equals the published product scores on the first tiles.
  gtbox_guided()            tab:gtbox LACE arm: GT boxes on the 439, knobs from the npz (0.3/1.6),
                            per-crown IoU in the gtbox_modal format so combine_gtbox can merge it
                            with the existing SAM 3 2048 arms. Gate in container: bilinear tau 0.25
                            reproduces results_439/gtbox_lace_r2048.json per-crown IoUs on the
                            first tiles.
  components_eval(npz, src) tab:components / tab:collapse: mean per-crown mask IoU on the 108 val
                            tiles, oracle (gt) or predicted boxes, for one masker variant. Computes
                            BOTH the published 512 bilinear tau 0.25 number (gate, compared locally
                            with phase4_collapse/results/eval/*.json) and the 2048 guided number in
                            one pass.

Per-tile checkpoints (JSONL / per-tile files) with resume; Modal retries on pre-emption; logs on
the volume. RGB verified per tile (sha1 or image_id) against the relevant manifest.

    python detach.py deploy guided_ablate_modal.py
    python detach.py spawn tcd04-guided-ablate sparse_render --seed 0
"""
import json
import os

import modal

APP_NAME = "tcd04-guided-ablate"
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REPO = os.path.dirname(PKG)
PH4 = os.path.join(PKG, "modal_tcd_multiseed", "phase4")
STUBS = os.path.join(PH4, "stubs")
SPARSE_PREDS = os.path.join(PKG, "modal_sparse_tcd_multiseed", "preds")

app = modal.App(APP_NAME)
p4vol = modal.Volume.from_name("tcd04-phase4-vol")
spvol = modal.Volume.from_name("tcd-sparse-vol")

P = "/root/proj"
PKG_R = f"{P}/boxinst_commonality_tcd_04"
IN = "/root/inputs"

RES = 2048
ALPHA, KAPPA = 0.30, 1.60
GF_R, GF_EPS, GF_TAU = 12, 1e-4, 0.35
MARGIN = 16
COMMIT_EVERY = 10
WORKERS = 7

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.1", "torchvision==0.27.1", "numpy==2.2.6",
                 "transformers==4.57.1", "pillow", "contourpy", "pycocotools", "scipy")
)
for rel in ("dapt/__init__.py", "dapt/backbone.py", "dapt/targets.py",
            "dapt/decode.py", "dapt/eval.py", "dapt/head.py"):
    image = image.add_local_file(os.path.join(REPO, rel), f"{P}/{rel}")
for rel in ("__init__.py", "detector.py", "train_detector_tiles.py", "evaluate.py",
            "em.py", "prepare_test.py", "cache_test.py", "cache_train_tiles.py", "test_gt.json"):
    image = image.add_local_file(os.path.join(PKG, rel), f"{PKG_R}/{rel}")
image = image.add_local_file(os.path.join(PKG, "modal_tcd_multiseed", "__init__.py"),
                             f"{PKG_R}/modal_tcd_multiseed/__init__.py")
for rel in ("__init__.py", "phase4_lib_tcd.py"):
    image = image.add_local_file(os.path.join(PH4, rel), f"{PKG_R}/modal_tcd_multiseed/phase4/{rel}")
for rel in ("__init__.py", "gtbox_lib.py"):
    image = image.add_local_file(os.path.join(PKG, "box_supervised_baselines", rel),
                                 f"{PKG_R}/box_supervised_baselines/{rel}")
for _pkg, _files in {"boxinst": ("__init__.py", "cache_feats.py"),
                     "boxinst_commonality": ("__init__.py", "em.py"),
                     "boxinst_tcd": ("__init__.py", "build_canopy.py", "cache.py",
                                     "prepare.py")}.items():
    for _f in _files:
        image = image.add_local_file(os.path.join(STUBS, _pkg, _f), f"{P}/{_pkg}/{_f}")
image = (image
         .add_local_file(os.path.join(PH4, "manifest.json"), f"{IN}/manifest.json")
         .add_local_file(os.path.join(PH4, "val_gt.json"), f"{IN}/val_gt.json")
         .add_local_file(os.path.join(REPO, "data", "tcd_sparse", "sparse_gt.json"), f"{IN}/sparse_gt.json")
         .add_local_file(os.path.join(REPO, "data", "tcd_sparse", "slice_manifest.json"),
                         f"{IN}/slice_manifest.json")
         .add_local_file(os.path.join(PKG, "results_439", "gtbox_lace_r2048.json"),
                         f"{IN}/gtbox_lace_r2048.json"))
for _s in (0, 1, 2):
    image = image.add_local_file(os.path.join(SPARSE_PREDS, f"ours_prod_s{_s}.json"),
                                 f"{IN}/sparse_prod_s{_s}.json")
    image = image.add_local_file(os.path.join(HERE, "guided_ablate_inputs", f"detscore_s{_s}.json"),
                                 f"{IN}/detscore_s{_s}.json")


# ------------------------------------------------------------------ shared helpers
def color_guided_filter(I, p, r, eps):
    """He et al. 2010, colour guide -- identical to lace_guided_modal.py / guided_rgb_ext_val.py."""
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


_G = {}


def _boot():
    import sys
    if P not in sys.path:
        sys.path.insert(0, P)
    from boxinst_commonality_tcd_04 import evaluate as E
    _G["E"] = E


def _rgb(path, tid, man):
    import hashlib
    import numpy as np
    from PIL import Image
    arr = np.asarray(Image.open(f"{path}/{tid}.tif").convert("RGB"))
    rec = man[tid]
    if "rgb_sha1" in rec:
        assert hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest() == rec["rgb_sha1"], tid
    else:
        assert json.load(open(f"{path}/{tid}_meta.json"))["image_id"] == rec["image_id"], tid
    assert arr.shape == (2048, 2048, 3)
    return arr.astype(np.float32) / 255.0


def _prob(masker, zn, g, b, alpha, kappa, res=RES):
    """evaluate.pred_instance_masks per-box body at `res` -> ((y0,y1,x0,x1) clip, prob, r)."""
    import numpy as np
    from PIL import Image
    scale = 2048.0 / res
    origin = getattr(masker, "origin", getattr(masker, "s", 2 * scale) / 2.0)
    s = getattr(masker, "s", 2 * scale)
    delta = int(round((origin - s / 2.0) / scale))
    idx, r = masker.box_mask(zn, g, b, prior_weight=alpha, kappa_scale=kappa)
    grid = np.zeros(g * g, np.float32); grid[idx] = r
    prob = np.array(Image.fromarray(grid.reshape(g, g)).resize((res, res), Image.BILINEAR))
    if delta:
        sh = np.zeros_like(prob)
        s_ = slice(max(0, -delta), res - max(0, delta)); d_ = slice(max(0, delta), res - max(0, -delta))
        sh[d_, d_] = prob[s_, s_]; prob = sh
    x0, y0, x1, y1 = np.asarray(b) / scale
    return (max(0, int(y0)), int(np.ceil(y1)), max(0, int(x0)), int(np.ceil(x1))), prob, r


def _guided(rgb, clip_yx, prob):
    import numpy as np
    by0, by1, bx0, bx1 = clip_yx
    wy0, wy1 = max(0, by0 - MARGIN), min(RES, by1 + MARGIN)
    wx0, wx1 = max(0, bx0 - MARGIN), min(RES, bx1 + MARGIN)
    q = color_guided_filter(rgb[wy0:wy1, wx0:wx1], prob[wy0:wy1, wx0:wx1], GF_R, GF_EPS)
    m = np.zeros((RES, RES), bool); m[wy0:wy1, wx0:wx1] = q >= GF_TAU
    c = np.zeros((RES, RES), bool); c[by0:by1, bx0:bx1] = True
    return m & c


def _clipmask(clip_yx, res=RES):
    import numpy as np
    by0, by1, bx0, bx1 = clip_yx
    c = np.zeros((res, res), bool); c[by0:by1, bx0:bx1] = True
    return c


def _rle(m):
    import numpy as np
    from pycocotools import mask as maskUtils
    return {"size": [int(m.shape[0]), int(m.shape[1])], "counts": maskUtils.encode(
        np.asfortranarray(m.astype(np.uint8)))["counts"].decode("ascii")}


def _jsonl_done(path):
    if not os.path.exists(path):
        return set(), 0
    done, good = set(), 0
    with open(path, "rb") as f:
        for raw in f:
            try:
                done.add(json.loads(raw.decode("utf-8"))["tile"]); good += len(raw)
            except Exception:
                break
    return done, good


def _truncate_to(path, nbytes):
    if os.path.exists(path) and os.path.getsize(path) != nbytes:
        with open(path, "r+b") as f:
            f.truncate(nbytes)


def _logger(path):
    import time
    f = open(path, "a")

    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True); f.write(line + "\n"); f.flush()
    return log


def _run_jsonl(tiles, worker, paths, vol, log):
    """Fork pool over tiles; the parent appends one JSONL line per tile per arm, commits often."""
    import multiprocessing as mp
    import time
    done = None
    for pth in paths.values():
        d, nb = _jsonl_done(pth); _truncate_to(pth, nb)
        done = d if done is None else (done & d)
    done = done or set()
    todo = [t for t in tiles if t not in done]
    log(f"  {len(tiles)} tiles, {len(done)} done, {len(todo)} to do, {WORKERS} workers")
    fhs = {a: open(p, "a") for a, p in paths.items()}
    t0, failed = time.time(), []
    try:
        with mp.get_context("fork").Pool(WORKERS) as pool:
            for k, (tid, out, err) in enumerate(pool.imap_unordered(worker, todo)):
                if out is None:
                    failed.append({"tile": tid, "error": err}); log(f"  !! FAILED {tid}: {err}")
                elif tid not in done:
                    for a in paths:
                        fhs[a].write(json.dumps(out[a]) + "\n")
                    done.add(tid)
                if (k + 1) % COMMIT_EVERY == 0 or k + 1 == len(todo):
                    for fh in fhs.values():
                        fh.flush(); os.fsync(fh.fileno())
                    vol.commit()
                    el = time.time() - t0
                    log(f"  {k+1}/{len(todo)} tiles {el:.0f}s eta {el/(k+1)*(len(todo)-k-1)/60:.0f} min")
    finally:
        for fh in fhs.values():
            fh.flush(); os.fsync(fh.fileno()); fh.close()
        vol.commit()
    return failed


def _safe_call(fn, tid):
    try:
        return tid, fn(tid), None
    except Exception as e:
        return tid, None, f"{type(e).__name__}: {e}"


# module-level wrappers: the fork pool pickles the callable, so it cannot be a closure
def _w_sparse(tid): return _safe_call(_sparse_tile, tid)
def _w_cfg(tid): return _safe_call(_cfg_tile, tid)
def _w_gtbox(tid): return _safe_call(_gtbox_tile, tid)
def _w_comp(tid): return _safe_call(_comp_tile, tid)


def box_iou_mat(a, b):
    """Verbatim from phase4_collapse/collapse_lib.py (the published components eval)."""
    import numpy as np
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix0 = np.maximum(a[:, 0][:, None], b[:, 0][None]); iy0 = np.maximum(a[:, 1][:, None], b[:, 1][None])
    ix1 = np.minimum(a[:, 2][:, None], b[:, 2][None]); iy1 = np.minimum(a[:, 3][:, None], b[:, 3][None])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None] - inter + 1e-9)


# ------------------------------------------------------------------ 1. sparse-236
def _sparse_tile(tid):
    import numpy as np
    E, masker, rec = _G["E"], _G["masker"], _G["src"][tid]
    bx = np.asarray(rec["boxes_2048"], np.float32).reshape(-1, 4)
    base = {"tile": tid, "boxes_2048": rec["boxes_2048"], "scores": rec["scores"]}
    out = {a: {**base, "canopy_ignore": [], "masks_rle": []} for a in ("cgf_r12_tau035", "bilinear_tau040")}
    if not len(bx):
        return out
    rgb = _rgb("/sp/rgb_sparse", tid, _G["man"])
    feat = np.load(f"/sp/feat_4p_sparse/{tid}.npy").astype(np.float32)
    g = feat.shape[-1]; zn = masker.project(feat); del feat
    can = np.array(E.raster(_G["gt"][tid]["canopy"], res=RES, scale=1.0))
    can = can.any(0) if len(can) else np.zeros((RES, RES), bool)
    for b in bx:
        clip, prob, _ = _prob(masker, zn, g, b, ALPHA, KAPPA)
        for a, m in (("cgf_r12_tau035", _guided(rgb, clip, prob)),
                     ("bilinear_tau040", (prob >= 0.40) & _clipmask(clip))):
            out[a]["masks_rle"].append(_rle(m))
            tot = int(m.sum()); out[a]["canopy_ignore"].append(bool(tot) and float((m & can).sum()) / tot > 0.5)
    return out


@app.function(image=image, volumes={"/vol": p4vol, "/sp": spvol}, timeout=12 * 3600, cpu=8,
              memory=49152, retries=modal.Retries(max_retries=3, initial_delay=30.0))
def sparse_render(seed: int = 0):
    _boot()
    E = _G["E"]
    _G.update(masker=E.TCDMasker("/vol/out/em_model_4p_fix.npz"),
              src=json.load(open(f"{IN}/sparse_prod_s{seed}.json"))["preds"],
              gt=json.load(open(f"{IN}/sparse_gt.json")),
              man=json.load(open(f"{IN}/slice_manifest.json"))["tiles"])
    out_dir = "/sp/out/native_raster/guided_r12"
    os.makedirs(out_dir, exist_ok=True)
    log = _logger(f"{out_dir}/run_sparse_s{seed}.log")
    tiles = sorted(_G["src"]); assert len(tiles) == 236, len(tiles)
    log(f"[sparse] seed {seed}: start")
    paths = {a: f"{out_dir}/preds_lace_sparse_s{seed}_r{RES}_{a}.jsonl"
             for a in ("cgf_r12_tau035", "bilinear_tau040")}
    failed = _run_jsonl(tiles, _w_sparse, paths, spvol, log)
    log(f"[sparse] DONE seed {seed} failed={failed}")
    return {"seed": seed, "failed": failed, "files": paths}


# ------------------------------------------------------------------ 2. masker configs (tab:masker)
def _cfg_tile(tid):
    import numpy as np
    masker, rec = _G["masker"], _G["det"][tid]
    a, k = _G["cfg"]
    bx = np.asarray(rec["boxes_2048"], np.float32).reshape(-1, 4)
    out = {"tile": tid, "boxes_2048": rec["boxes_2048"], "scores": [], "canopy_ignore": [],
           "masks_rle": []}
    if not len(bx):
        return {"cfg": out}
    E = _G["E"]
    rgb = _rgb("/vol/rgb_test", tid, _G["man"])
    feat = np.load(f"/vol/feat_4p_test/{tid}.npy").astype(np.float32)
    g = feat.shape[-1]; zn = masker.project(feat); del feat
    can = np.array(E.raster(_G["gt"][tid]["canopy"], res=RES, scale=1.0))
    can = can.any(0) if len(can) else np.zeros((RES, RES), bool)
    s = np.asarray(rec["scores"], np.float64)
    for i, b in enumerate(bx):
        clip, prob, r = _prob(masker, zn, g, b, a, k)
        out["scores"].append(round(float(s[i] * np.abs(2 * r - 1).mean() * r.mean()), 8))
        m = _guided(rgb, clip, prob)
        out["masks_rle"].append(_rle(m))
        tot = int(m.sum()); out["canopy_ignore"].append(bool(tot) and float((m & can).sum()) / tot > 0.5)
    return {"cfg": out}


@app.function(image=image, volumes={"/vol": p4vol}, timeout=12 * 3600, cpu=8, memory=49152,
              retries=modal.Retries(max_retries=3, initial_delay=30.0))
def cfg_render(seed: int = 0, alpha: float = 1.0, kappa: float = 1.0):
    import numpy as np
    _boot()
    E = _G["E"]
    _G.update(masker=E.TCDMasker("/vol/out/em_model_4p_fix.npz"),
              det=json.load(open(f"{IN}/detscore_s{seed}.json")),
              gt=json.load(open(f"{PKG_R}/test_gt.json")),
              man=json.load(open(f"{IN}/manifest.json"))["feat_test"], cfg=(alpha, kappa))
    tag = f"a{alpha:g}_k{kappa:g}"
    out_dir = "/vol/out/native_raster/guided_cfg"
    os.makedirs(out_dir, exist_ok=True)
    log = _logger(f"{out_dir}/run_s{seed}_{tag}.log")
    # GATE: at the deployed (0.3, 1.6) the recomputed product equals the published product
    pub = json.load(open(f"/vol/out/native_raster/src/lace_s{seed}.json"))["preds"]
    n_chk, worst = 0, 0.0
    for tid in sorted(_G["det"])[:3]:
        rec = _G["det"][tid]
        if not rec["scores"]:
            continue
        assert rec["boxes_2048"] == pub[tid]["boxes_2048"], f"box mismatch {tid}"
        feat = np.load(f"/vol/feat_4p_test/{tid}.npy").astype(np.float32)
        zn, g = _G["masker"].project(feat), feat.shape[-1]
        for i, b in enumerate(np.asarray(rec["boxes_2048"], np.float32).reshape(-1, 4)):
            _, _, r = _prob(_G["masker"], zn, g, b, ALPHA, KAPPA)
            p = rec["scores"][i] * np.abs(2 * r - 1).mean() * r.mean()
            worst = max(worst, abs(p - pub[tid]["scores"][i])); n_chk += 1
    assert worst < 5e-5, f"product gate failed: max |diff| {worst}"
    log(f"[cfg {tag}] seed {seed}: product gate PASSED on {n_chk} boxes (max |diff| {worst:.1e})")
    tiles = sorted(_G["det"]); assert len(tiles) == 439
    paths = {"cfg": f"{out_dir}/preds_lace_s{seed}_r{RES}_cgf_{tag}.jsonl"}
    failed = _run_jsonl(tiles, _w_cfg, paths, p4vol, log)
    log(f"[cfg {tag}] DONE seed {seed} failed={failed}")
    return {"seed": seed, "cfg": tag, "failed": failed, "files": paths}


# ------------------------------------------------------------------ 3. GT-box bake-off LACE arm
def _gtbox_tile(tid):
    import numpy as np
    from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as L
    masker, gt = _G["masker"], _G["gt"]
    polys = gt[tid]["trees"]
    boxes = L.gt_boxes_2048(polys)
    gt_rles, valid = L.gt_rles_and_valid(polys, res=RES)
    rgb = _rgb("/vol/rgb_test", tid, _G["man"])
    feat = np.load(f"/vol/feat_4p_test/{tid}.npy").astype(np.float32)
    g = feat.shape[-1]; zn = masker.project(feat); del feat
    p_rles, b_rles = [], []
    for b in boxes:
        clip, prob, _ = _prob(masker, zn, g, b, None, None)          # knobs from the npz
        p_rles.append(_rle(_guided(rgb, clip, prob)))
        b_rles.append(_rle((prob >= 0.25) & _clipmask(clip)))       # gate arm (published config)
    enc = lambda rs: [{"size": r["size"], "counts": r["counts"].encode()} for r in rs]
    pr, br = enc(p_rles), enc(b_rles)
    return {"g": {"tile": tid, "iou": np.round(L.per_crown_iou_rle(pr, gt_rles), 5).tolist(),
                  "valid": valid.tolist(), "double": L.double_assignment_rle(pr),
                  "masks_rle": p_rles},
            "b": {"tile": tid, "iou": np.round(L.per_crown_iou_rle(br, gt_rles), 5).tolist(),
                  "valid": valid.tolist()}}


@app.function(image=image, volumes={"/vol": p4vol}, timeout=12 * 3600, cpu=8, memory=49152,
              retries=modal.Retries(max_retries=3, initial_delay=30.0))
def gtbox_guided():
    _boot()
    E = _G["E"]
    gt = json.load(open(f"{PKG_R}/test_gt.json"))
    _G.update(masker=E.TCDMasker("/vol/out/em_model_4p_fix.npz"), gt=gt,
              man=json.load(open(f"{IN}/manifest.json"))["feat_test"])
    m = _G["masker"]
    assert abs(m.prior_weight - ALPHA) < 1e-9 and abs(m.kappa_scale - KAPPA) < 1e-9, \
        (m.prior_weight, m.kappa_scale)
    out_dir = "/vol/out/native_raster/gtbox_guided"
    os.makedirs(out_dir, exist_ok=True)
    log = _logger(f"{out_dir}/run.log")
    ref = json.load(open(f"{IN}/gtbox_lace_r2048.json"))["per_tile"]
    tiles = [t for t in sorted(gt) if gt[t]["trees"]]
    for tid in tiles[:2]:                                            # GATE before the run
        o = _gtbox_tile(tid)["b"]
        assert o["iou"] == ref[tid]["iou"] and o["valid"] == ref[tid]["valid"], f"gtbox gate {tid}"
    log(f"[gtbox] gate PASSED (bilinear tau 0.25 == gtbox_lace_r2048.json on {tiles[:2]}); "
        f"{len(tiles)} tiles")
    paths = {"g": f"{out_dir}/gtbox_lace_guided_r{RES}.jsonl", "b": f"{out_dir}/gtbox_lace_bilinear025_r{RES}.jsonl"}
    failed = _run_jsonl(tiles, _w_gtbox, paths, p4vol, log)
    log(f"[gtbox] DONE failed={failed}")
    return {"failed": failed, "files": paths}


# ------------------------------------------------------------------ 4. components / collapse
def _comp_tile(tid):
    import numpy as np
    import torch
    from dapt.decode import decode
    from boxinst_commonality_tcd_04.detector import STRIDE8
    E, masker, gt = _G["E"], _G["masker"], _G["gt"]
    polys = gt[tid]["trees"]
    gb = np.array([[*np.asarray(p, float).reshape(-1, 2).min(0), *np.asarray(p, float).reshape(-1, 2).max(0)]
                   for p in polys], np.float32).reshape(-1, 4)
    feat = np.load(f"/vol/feat_4p_train/{tid}.npy").astype(np.float32)
    g = feat.shape[-1]
    if _G["src"] == "gt":
        bx, sc = gb, np.ones(len(gb), np.float32)
    else:
        with torch.no_grad():
            det = _G["model"](torch.from_numpy(feat)[None])
        bx, sc = decode(det, score_thr=0.05, stride=STRIDE8, topk=600)
        bx, sc = bx.numpy(), sc.numpy()
    zn = masker.project(feat); del feat
    rgb = _rgb("/vol/rgb_val", tid, _G["man"])
    pairs = []
    if _G["src"] == "gt":
        pairs = [(i, i) for i in range(len(bx))]
    elif len(sc) and len(gb):
        bm = box_iou_mat(bx, gb)
        order, _, _, _, gidx = E.match_tile(bm, sc, np.zeros(len(sc), bool), 0.5)
        pairs = [(int(order[rk]), int(j)) for rk, j in enumerate(gidx) if j >= 0]
    need = sorted({i for i, _ in pairs})
    g512 = np.array(E.raster(polys)) if len(polys) else np.zeros((0, 512, 512), bool)
    iou512, iou2048 = [], []
    cache = {}
    for i in need:
        b = bx[i]
        clip5, prob5, _ = _prob(masker, zn, g, b, None, None, res=512)
        clip2, prob2, _ = _prob(masker, zn, g, b, None, None, res=RES)
        cache[i] = ((prob5 >= 0.25) & _clipmask(clip5, 512), _guided(rgb, clip2, prob2))
    for i, j in pairs:
        m5, m2 = cache[i]
        u = (m5 | g512[j]).sum(); iou512.append(float((m5 & g512[j]).sum() / u) if u else 0.0)
        gj = np.asarray(E.raster([polys[j]], res=RES, scale=1.0)[0])
        u = (m2 | gj).sum(); iou2048.append(float((m2 & gj).sum() / u) if u else 0.0)
    return {"c": {"tile": tid, "iou512_bilinear025": iou512, "iou2048_guided": iou2048}}


@app.function(image=image, volumes={"/vol": p4vol}, timeout=12 * 3600, cpu=8, memory=49152,
              retries=modal.Retries(max_retries=3, initial_delay=30.0))
def components_eval(npz: str = "collapse/k16_s0_b0.5_c1/em_k16_s0_b0.5_c1_fix.npz", src: str = "gt"):
    import numpy as np
    import torch
    _boot()
    E = _G["E"]
    assert src in ("gt", "pred")
    gt = json.load(open(f"{IN}/val_gt.json"))
    _G.update(masker=E.TCDMasker(f"/vol/{npz}"), gt=gt, src=src,
              man=json.load(open(f"{IN}/manifest.json"))["feat_traintile"])
    if src == "pred":
        from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.phase4_lib_tcd import Detector4Phase
        ck = torch.load("/vol/out/det_phase4_L24_s0.pt", map_location="cpu", weights_only=False)
        mdl = Detector4Phase(ck["cfg"]["in_dim"], width=ck["cfg"]["width"], tower=ck["cfg"]["tower"]).eval()
        mdl.load_state_dict(ck["state"]); _G["model"] = mdl
        torch.set_num_threads(1)
    tag = f"{os.path.basename(npz).replace('.npz', '')}_{src}"
    out_dir = "/vol/out/native_raster/components_guided"
    os.makedirs(out_dir, exist_ok=True)
    log = _logger(f"{out_dir}/run_{tag}.log")
    tiles = [t for t in sorted(gt) if gt[t]["trees"] and os.path.exists(f"/vol/feat_4p_train/{t}.npy")]
    log(f"[comp] {tag}: {len(tiles)} val tiles, knobs from npz ({_G['masker'].prior_weight}, "
        f"{_G['masker'].kappa_scale})")
    paths = {"c": f"{out_dir}/{tag}.jsonl"}
    failed = _run_jsonl(tiles, _w_comp, paths, p4vol, log)
    recs = [json.loads(l) for l in open(paths["c"])]
    a5 = np.concatenate([r["iou512_bilinear025"] for r in recs if r["iou512_bilinear025"]])
    a2 = np.concatenate([r["iou2048_guided"] for r in recs if r["iou2048_guided"]])
    summ = {"npz": npz, "src": src, "n_tiles": len(recs), "n_matched": int(len(a2)),
            "mean_crown_iou_512_bilinear025": round(float(a5.mean()), 4),
            "mean_crown_iou_2048_guided": round(float(a2.mean()), 4), "failed": failed}
    json.dump(summ, open(f"{out_dir}/{tag}_summary.json", "w"), indent=2)
    p4vol.commit()
    log(f"[comp] DONE {json.dumps(summ)}")
    return summ
