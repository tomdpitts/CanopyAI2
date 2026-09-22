"""Modal CPU app: regenerate LACE's 439-tile masks at the NATIVE 2048 raster.

WHY THIS IS A RE-RUN AND NOT AN UPSAMPLE. `evaluate.pred_instance_masks` bilinearly
resizes the 256-cell posterior grid DIRECTLY to `res` and thresholds afterwards, so the
2048 mask is not the 512 mask upsampled -- the interpolation happens at the target
resolution and the thresholded boundary genuinely differs. Nearest-upsampling the stored
512 RLE would fabricate a mask the model never emitted. Hence this pass.

NO GPU, NO RETRAINING, NO DETECTOR PASS. The detections are read verbatim from the
published preds files (`boxes_2048`, `scores`, and the product-rerank ordering already
baked into `scores`), so the detection SET and its RANKING are byte-identical to the
published rows and the only thing that moves is the mask raster. That is what makes the
512-vs-2048 comparison paired.

Inputs, all already on `tcd04-phase4-vol` (verified present 2026-09-03):
    feat_4p_test/{tile}.npy        4-phase L24 features, (1024, 256, 256) float32
    out/em_model_4p_fix.npz        the fitted masker (NOT refitted here)
and the published detections from the repo:
    confidence/preds_knobbed_s{0,1,2}_product.json

Knobs are the published inference scalars (Section 3.2.4 of the paper): prior_weight
0.30, kappa_scale 1.60 (= kappa 16 from a fit-time 10), mask_thr 0.25. They are passed
EXPLICITLY rather than left to the masker's stored defaults so this file records the
deployed configuration rather than inheriting it silently.

PERSISTENCE RULE (the reason this campaign exists at all). Masks are written as RLE at
2048 -- the finest resolution the model can express. Every coarser raster, score floor,
ranking key and canopy rule is then a free CPU re-score through `score_coco.py`. Never
write only the downsample again.

Usage:
    modal run lace_masks_modal.py::regen --seed 0
    modal run lace_masks_modal.py                 # all three seeds
"""
import json
import os

import modal

APP_NAME = "tcd04-native-raster-lace"
VOL_NAME = "tcd04-phase4-vol"

HERE = os.path.dirname(os.path.abspath(__file__))      # .../native_raster
PKG = os.path.dirname(HERE)                            # boxinst_commonality_tcd_04
REPO = os.path.dirname(PKG)
PH4 = os.path.join(PKG, "modal_tcd_multiseed", "phase4")
STUBS = os.path.join(PH4, "stubs")

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME)

P = "/root/proj"
PKG_R = f"{P}/boxinst_commonality_tcd_04"

RES = 2048                      # native imagery resolution -- the whole point
FEAT_DIR = "/vol/feat_4p_test"
EM_PATH = "/vol/out/em_model_4p_fix.npz"
OUT_DIR = "/vol/out/native_raster"
SRC_DIR = "/vol/out/native_raster/src"   # staged copies of the published 512 detections

# published inference scalars -- see modal_tcd_multiseed/phase4/HYPERPARAMS.md
PRIOR_WEIGHT = 0.30
KAPPA_SCALE = 1.60
MASK_THR = 0.25

# Image mirrors phase4_modal.py's file set. `evaluate.py` imports torch (via dapt.backbone)
# and `em.py` imports the boxinst_* sibling packages, so the same stubs are required even
# though nothing here touches a GPU or a backbone.
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
            "test_gt.json"):
    image = image.add_local_file(os.path.join(PKG, rel), f"{PKG_R}/{rel}")
STUB_FILES = {
    "boxinst": ("__init__.py", "cache_feats.py"),
    "boxinst_commonality": ("__init__.py", "em.py"),
    "boxinst_tcd": ("__init__.py", "build_canopy.py", "cache.py", "prepare.py"),
}
for _pkg, _files in STUB_FILES.items():
    for _f in _files:
        image = image.add_local_file(os.path.join(STUBS, _pkg, _f), f"{P}/{_pkg}/{_f}")


@app.function(image=image, volumes={"/vol": vol}, timeout=6 * 3600,
              cpu=8, memory=32768)
def regen(seed: int, limit: int = 0):
    """Re-render seed `seed`'s masks at 2048 from the SAME boxes. Returns a summary dict;
    the preds file itself lands on the volume (and is fetched to the repo afterwards).

    The published 512 detections are read from the volume, not passed as an argument --
    they are ~25 MB per seed, well past Modal's argument size limit. Stage them first:
        modal volume put tcd04-phase4-vol \\
            confidence/preds_knobbed_s0_product.json out/native_raster/src/lace_s0.json
    """
    import sys
    import time

    import numpy as np
    from pycocotools import mask as maskUtils

    sys.path.insert(0, P)
    # same import surface phase4_lib_tcd.eval_selfmask uses: E re-exports TCDMasker
    from boxinst_commonality_tcd_04 import evaluate as E

    src = f"{SRC_DIR}/lace_s{seed}.json"
    assert os.path.exists(src), (
        f"{src} not staged. Run:\n  modal volume put {VOL_NAME} "
        f"boxinst_commonality_tcd_04/confidence/preds_knobbed_s{seed}_product.json "
        f"out/native_raster/src/lace_s{seed}.json")
    published = json.load(open(src))
    meta_in, preds_in = published["meta"], published["preds"]
    assert meta_in["mask_res"] == 512, f"expected a 512 source, got {meta_in['mask_res']}"
    assert meta_in["tile_px"] == 2048

    masker = E.TCDMasker(EM_PATH)
    gt = json.load(open(f"{PKG_R}/test_gt.json"))
    tiles = sorted(preds_in)
    if limit:
        tiles = tiles[:limit]
    print(f"[lace2048] seed {seed}: {len(tiles)} tiles, masker s={masker.s}", flush=True)

    out, n_det, t0 = {}, 0, time.time()
    for k, tid in enumerate(tiles):
        rec = preds_in[tid]
        bx = np.asarray(rec["boxes_2048"], np.float32).reshape(-1, 4)
        if len(bx) == 0:
            out[tid] = {"boxes_2048": [], "scores": [], "canopy_ignore": [],
                        "masks_rle": []}
            continue
        feat = np.load(os.path.join(FEAT_DIR, tid + ".npy")).astype(np.float32)
        g = feat.shape[-1]
        zn = masker.project(feat)
        pm = E.pred_instance_masks(masker, zn, g, bx, res=RES, scale=2048.0 / RES,
                                   mask_thr=MASK_THR, prior_weight=PRIOR_WEIGHT,
                                   kappa_scale=KAPPA_SCALE)
        # canopy_ignore recomputed AT 2048 -- the 512 flags were computed on a coarser
        # raster and a >50%-in-canopy test is raster dependent. Only the legacy scoring
        # arm reads this; the frozen protocol arm uses iscrowd GT inside COCOeval.
        can = np.array(E.raster(gt[tid]["canopy"], res=RES, scale=2048.0 / RES))
        can = can.any(0) if len(can) else np.zeros((RES, RES), bool)
        ci = [bool(m.sum()) and float((m & can).sum()) / float(m.sum()) > 0.5 for m in pm]
        out[tid] = {
            "boxes_2048": rec["boxes_2048"],
            "scores": rec["scores"],
            "canopy_ignore": ci,
            "masks_rle": [{"size": [RES, RES],
                           "counts": maskUtils.encode(
                               np.asfortranarray(m.astype(np.uint8)))["counts"].decode("ascii")}
                          for m in pm]}
        n_det += len(pm)
        if (k + 1) % 25 == 0 or k + 1 == len(tiles):
            print(f"  {k+1}/{len(tiles)} tiles, {n_det} dets, "
                  f"{time.time()-t0:.0f}s", flush=True)

    meta = dict(meta_in)
    meta.update({"mask_res": RES, "scale_box_to_mask": 1.0,
                 "regenerated_at_native_raster": True,
                 "source_preds_meta_mask_res": 512,
                 "prior_weight": PRIOR_WEIGHT, "kappa_scale": KAPPA_SCALE,
                 "mask_thr": MASK_THR,
                 "provenance": "boxes + scores copied verbatim from the published 512 "
                               "row; ONLY the mask raster differs. Masks re-rendered by "
                               "evaluate.pred_instance_masks at res=2048 (NOT upsampled "
                               "from 512 -- the bilinear posterior resize happens at the "
                               "target resolution, so an upsample would be a different "
                               "mask than the model emits)."})
    os.makedirs(OUT_DIR, exist_ok=True)
    dst = f"{OUT_DIR}/preds_lace_s{seed}_r2048.json"
    with open(dst, "w") as f:
        json.dump({"meta": meta, "preds": out}, f)
    vol.commit()
    size_mb = os.path.getsize(dst) / 1e6
    print(f"[lace2048] wrote {dst} ({size_mb:.0f} MB, {n_det} dets)", flush=True)
    return {"seed": seed, "n_tiles": len(tiles), "n_det": n_det,
            "path": dst, "size_mb": round(size_mb, 1)}


@app.local_entrypoint()
def main(seeds: str = "0,1,2", limit: int = 0):
    """Regenerate each seed's masks at 2048. Stage the sources first (see regen docstring)."""
    reports = [regen.remote(int(s), limit) for s in seeds.split(",")]
    print(json.dumps(reports, indent=2))
