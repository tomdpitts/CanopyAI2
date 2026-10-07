"""Modal CPU app: LACE test masks at native 2048 with the COLOUR-GUIDED render (pre-registered).

Pre-registration: NATIVE_RASTER.md "PRE-REGISTRATION — LACE + colour-guided render" (2026-10-07).
Built from lace_multitau_modal.py (untouched). Per seed, from the SAME frozen detections (boxes +
scores verbatim from out/native_raster/src/lace_s{seed}.json, i.e. the published product-ranked
row), it writes two JSONL files:
  * cgf_r12_tau035 : bilinear posterior -> colour-guided filter (guide = tile RGB, r 12 px,
                     eps 1e-4) -> tau 0.35 -> box clip            (the pre-registered row)
  * bilinear_tau040: the deployed render at tau 0.40               (GATE: must reproduce the
                     published scored_lace_s{seed}_r2048_tau040.json when scored locally)
ROBUSTNESS: per-tile JSONL checkpoints committed every COMMIT_EVERY tiles; resume on restart
(a tile is done only if present in BOTH files; truncated tails dropped); new output dir only;
in-container gates before any work: (1) the bilinear path equals evaluate.pred_instance_masks
on the first tile; (2) the guided filter reproduces p when the guide contains p, and equals the
grey filter for a guide with constant extra channels; every RGB tile is verified against
manifest.json (pixel sha1 where present, else image_id from the tile's meta json).
Tiles are processed by a fork pool inside the container; the parent alone writes the files.

    python detach.py deploy lace_guided_modal.py
    python detach.py spawn tcd04-lace-guided render --seed 0 --note "..."
"""
import json
import os

import modal

APP_NAME = "tcd04-lace-guided"
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

RES = 2048
PRIOR_WEIGHT, KAPPA_SCALE = 0.30, 1.60
GF_R, GF_EPS, GF_TAU = 12, 1e-4, 0.35
GATE_TAU = 0.40
MARGIN = 16
COMMIT_EVERY = 10
WORKERS = 7

FEAT_DIR = "/vol/feat_4p_test"
RGB_DIR = "/vol/rgb_test"
EM_PATH = "/vol/out/em_model_4p_fix.npz"
OUT_DIR = "/vol/out/native_raster/guided_r12"          # NEW directory
SRC_DIR = "/vol/out/native_raster/src"
ARMS = {"cgf_r12_tau035": "cgf", "bilinear_tau040": "bilinear"}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.1", "torchvision==0.27.1", "numpy==2.2.6",
                 "transformers==4.57.1", "pillow", "contourpy", "pycocotools", "scipy")
)
for rel in ("dapt/__init__.py", "dapt/backbone.py", "dapt/targets.py",
            "dapt/decode.py", "dapt/eval.py", "dapt/head.py"):
    image = image.add_local_file(os.path.join(REPO, rel), f"{P}/{rel}")
for rel in ("__init__.py", "detector.py", "train_detector_tiles.py", "evaluate.py",
            "em.py", "prepare_test.py", "cache_test.py", "cache_train_tiles.py",
            "test_gt.json"):
    image = image.add_local_file(os.path.join(PKG, rel), f"{PKG_R}/{rel}")
image = image.add_local_file(os.path.join(PH4, "manifest.json"), f"{P}/manifest.json")
for _pkg, _files in {"boxinst": ("__init__.py", "cache_feats.py"),
                     "boxinst_commonality": ("__init__.py", "em.py"),
                     "boxinst_tcd": ("__init__.py", "build_canopy.py", "cache.py",
                                     "prepare.py")}.items():
    for _f in _files:
        image = image.add_local_file(os.path.join(STUBS, _pkg, _f), f"{P}/{_pkg}/{_f}")


def color_guided_filter(I, p, r, eps):
    """He et al. 2010, colour guide -- identical to native_raster/guided_rgb_ext_val.py."""
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
        return True
    return False


_G = {}


def _setup(seed):
    import sys
    sys.path.insert(0, P)
    from boxinst_commonality_tcd_04 import evaluate as E
    _G["E"] = E
    _G["masker"] = E.TCDMasker(EM_PATH)
    _G["preds"] = json.load(open(f"{SRC_DIR}/lace_s{seed}.json"))["preds"]
    _G["gt"] = json.load(open(f"{PKG_R}/test_gt.json"))
    _G["man"] = json.load(open(f"{P}/manifest.json"))["feat_test"]
    m = _G["masker"]
    _G["delta"] = int(round((getattr(m, "origin", m.s / 2.0) - m.s / 2.0) / (2048.0 / RES)))


def _rgb(tid):
    import hashlib
    import numpy as np
    from PIL import Image
    arr = np.asarray(Image.open(f"{RGB_DIR}/{tid}.tif").convert("RGB"))
    rec = _G["man"][tid]
    if "rgb_sha1" in rec:
        assert hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest() == rec["rgb_sha1"], \
            f"RGB sha1 mismatch {tid}"
    else:
        meta = json.load(open(f"{RGB_DIR}/{tid}_meta.json"))
        assert meta["image_id"] == rec["image_id"], f"RGB image_id mismatch {tid}"
    assert arr.shape == (2048, 2048, 3)
    return arr.astype(np.float32) / 255.0


def _box_prob(zn, g, b):
    """evaluate.pred_instance_masks per-box body -> (clip, prob)."""
    import numpy as np
    from PIL import Image
    m, delta = _G["masker"], _G["delta"]
    idx, r = m.box_mask(zn, g, b, prior_weight=PRIOR_WEIGHT, kappa_scale=KAPPA_SCALE)
    grid = np.zeros(g * g, np.float32); grid[idx] = r
    prob = np.array(Image.fromarray(grid.reshape(g, g)).resize((RES, RES), Image.BILINEAR))
    if delta:
        sh = np.zeros_like(prob)
        s_ = slice(max(0, -delta), RES - max(0, delta)); d_ = slice(max(0, delta), RES - max(0, -delta))
        sh[d_, d_] = prob[s_, s_]; prob = sh
    x0, y0, x1, y1 = np.asarray(b) / (2048.0 / RES)
    return (max(0, int(y0)), int(np.ceil(y1)), max(0, int(x0)), int(np.ceil(x1))), prob


def _tile(tid):
    import numpy as np
    from pycocotools import mask as maskUtils
    E = _G["E"]; rec = _G["preds"][tid]
    bx = np.asarray(rec["boxes_2048"], np.float32).reshape(-1, 4)
    out = {a: {"tile": tid, "boxes_2048": rec["boxes_2048"], "scores": rec["scores"],
               "canopy_ignore": [], "masks_rle": []} for a in ARMS}
    if len(bx) == 0:
        return tid, out, 0
    rgb = _rgb(tid)
    feat = np.load(os.path.join(FEAT_DIR, tid + ".npy")).astype(np.float32)
    g = feat.shape[-1]; zn = _G["masker"].project(feat); del feat
    can = np.array(E.raster(_G["gt"][tid]["canopy"], res=RES, scale=2048.0 / RES))
    can = can.any(0) if len(can) else np.zeros((RES, RES), bool)
    for b in bx:
        (by0, by1, bx0, bx1), prob = _box_prob(zn, g, b)
        clip = np.zeros((RES, RES), bool); clip[by0:by1, bx0:bx1] = True
        wy0, wy1 = max(0, by0 - MARGIN), min(RES, by1 + MARGIN)
        wx0, wx1 = max(0, bx0 - MARGIN), min(RES, bx1 + MARGIN)
        q = color_guided_filter(rgb[wy0:wy1, wx0:wx1], prob[wy0:wy1, wx0:wx1], GF_R, GF_EPS)
        mg = np.zeros((RES, RES), bool); mg[wy0:wy1, wx0:wx1] = q >= GF_TAU
        masks = {"cgf_r12_tau035": mg & clip, "bilinear_tau040": (prob >= GATE_TAU) & clip}
        for a, m in masks.items():
            out[a]["masks_rle"].append({"size": [RES, RES], "counts": maskUtils.encode(
                np.asfortranarray(m.astype(np.uint8)))["counts"].decode("ascii")})
            tot = int(m.sum())
            out[a]["canopy_ignore"].append(bool(tot) and float((m & can).sum()) / tot > 0.5)
    return tid, out, len(bx)


@app.function(image=image, volumes={"/vol": vol}, timeout=12 * 3600, cpu=8, memory=49152,
              retries=modal.Retries(max_retries=3, initial_delay=30.0))
def render(seed: int = 0, limit: int = 0):
    import multiprocessing as mp
    import time

    import numpy as np
    _setup(seed)
    E, masker = _G["E"], _G["masker"]
    out_dir = OUT_DIR if not limit else f"{OUT_DIR}_smoke_lim{limit}"   # smoke never mixes in
    os.makedirs(out_dir, exist_ok=True)
    logf = open(f"{out_dir}/run_s{seed}.log", "a")

    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    tiles = sorted(_G["preds"])
    assert len(tiles) == 439, len(tiles)
    if limit:
        tiles = tiles[:limit]
    n_rgb = len([f for f in os.listdir(RGB_DIR) if f.endswith(".tif")])
    assert n_rgb >= 439, f"rgb_test incomplete: {n_rgb}"

    # ---- gates ----
    rng = np.random.default_rng(0); p = rng.random((60, 70)).astype(np.float32)
    I = np.stack([p, rng.random((60, 70)), rng.random((60, 70))], 2).astype(np.float32)
    e1 = float(np.abs(color_guided_filter(I, p, 4, 1e-6) - p).max())
    assert e1 < 1e-4, f"filter self-guide gate failed: {e1}"
    gate_tile = next(t for t in tiles if len(_G["preds"][t]["boxes_2048"]))
    gf = np.load(os.path.join(FEAT_DIR, gate_tile + ".npy")).astype(np.float32)
    gzn, gg = masker.project(gf), gf.shape[-1]; del gf
    gbx = np.asarray(_G["preds"][gate_tile]["boxes_2048"], np.float32).reshape(-1, 4)
    ref = E.pred_instance_masks(masker, gzn, gg, gbx, res=RES, scale=2048.0 / RES,
                                mask_thr=GATE_TAU, prior_weight=PRIOR_WEIGHT,
                                kappa_scale=KAPPA_SCALE)
    mine = []
    for b in gbx:
        (y0, y1, x0, x1), prob = _box_prob(gzn, gg, b)
        c = np.zeros((RES, RES), bool); c[y0:y1, x0:x1] = True
        mine.append((prob >= GATE_TAU) & c)
    assert np.array_equal(ref, np.array(mine)), "bilinear path != evaluate.pred_instance_masks"
    _rgb(gate_tile)
    log(f"[guided] seed {seed}: gates PASSED (filter self-guide err {e1:.1e}; bilinear == "
        f"pred_instance_masks on {gate_tile}, {len(gbx)} boxes, delta {_G['delta']}; RGB verified)")
    del gzn

    paths = {a: f"{out_dir}/preds_lace_s{seed}_r{RES}_{a}.jsonl" for a in ARMS}
    done = None
    for a, pth in paths.items():
        d, nb = _jsonl_done(pth)
        if _truncate_to(pth, nb):
            log(f"[guided] truncated partial tail of {os.path.basename(pth)}")
        done = d if done is None else (done & d)
    done = done or set()
    todo = [t for t in tiles if t not in done]
    log(f"[guided] seed {seed}: {len(tiles)} tiles, {len(done)} done, {len(todo)} to do, "
        f"{WORKERS} workers")

    fhs = {a: open(pth, "a") for a, pth in paths.items()}
    t0, n_det, failed = time.time(), 0, []
    try:
        with mp.get_context("fork").Pool(WORKERS) as pool:
            for k, res in enumerate(pool.imap_unordered(_safe_tile, todo)):
                tid, out, nd = res
                if out is None:
                    failed.append({"tile": tid, "error": nd}); log(f"  !! FAILED {tid}: {nd}")
                else:
                    if tid in done:
                        continue
                    for a in ARMS:
                        fhs[a].write(json.dumps(out[a]) + "\n")
                    n_det += nd; done.add(tid)
                if (k + 1) % COMMIT_EVERY == 0 or k + 1 == len(todo):
                    for fh in fhs.values():
                        fh.flush(); os.fsync(fh.fileno())
                    vol.commit()
                    el = time.time() - t0
                    log(f"  {k+1}/{len(todo)} tiles {n_det} dets {el:.0f}s "
                        f"eta {el/(k+1)*(len(todo)-k-1)/60:.0f} min [committed]")
    finally:
        for fh in fhs.values():
            fh.flush(); os.fsync(fh.fileno()); fh.close()
        vol.commit()
    summary = {"seed": seed, "res": RES, "arms": list(ARMS), "gf": {"r": GF_R, "eps": GF_EPS,
               "tau": GF_TAU}, "gate_tau": GATE_TAU, "n_tiles": len(tiles), "n_det_this_call": n_det,
               "failed": failed, "secs": round(time.time() - t0, 1),
               "files": paths, "preregistration": "NATIVE_RASTER.md 2026-10-07"}
    json.dump(summary, open(f"{out_dir}/summary_s{seed}.json", "w"), indent=2)
    vol.commit()
    log(f"[guided] DONE seed {seed}: {json.dumps({k: v for k, v in summary.items() if k != 'files'})}")
    return summary


def _safe_tile(tid):
    try:
        return _tile(tid)
    except Exception as e:
        return tid, None, f"{type(e).__name__}: {e}"
