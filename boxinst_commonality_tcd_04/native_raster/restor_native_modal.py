"""Restor's released Mask R-CNN on the OAM-TCD 439, persisting masks at the NATIVE 2048 raster.

This is `restor_baseline/restor_modal.py::predict` with ONE substantive change: the (2048,2048)
mask detectron2 returns is RLE-encoded as-is instead of being downsampled to 512 and thrown away
(restor_modal.py:193-197 -- the "third occurrence" in NATIVE_RASTER.md). The 512 downsample is
ALSO written, from the same forward pass, with the identical PIL BILINEAR>=128 call, so the
published `preds_restor_rpn1000.json` is a byte-level gate on this run.

Inference is otherwise IDENTICAL to the published rpn1000 arm: whole 2048 tiles, no resize,
SCORE_THRESH 0.05, DETECTIONS_PER_IMAGE 600, RPN PRE/POST_NMS_TOPK_TEST 1000, every class
emitted with `pred_classes` so tree-only vs pooled stays a scoring decision.

Persistence (the campaign rule): per-tile JSON files on the volume, committed every 10 tiles,
resumed on restart; a run.log on the volume for mid-flight progress; nothing written outside
OUT_DIR, which did not exist before this run (verified 2026-09-22).

    modal run --detach restor_native_modal.py::predict  [--rpn-topk 512|1000|2000]
    modal run          restor_native_modal.py::assemble [--rpn-topk ...]   # CPU; re-runnable
rpn_topk defaults to 1000 (the arm run 2026-09-22); 512 = Restor's released cap, 2000 = the
near-ceiling arm of tab:rpn. Each arm gets its own OUT_DIR.
"""
import json
import os

import modal

APP = "tcd04-restor-native"
OUT_VOL = "tcd04-baselines-vol"
HFDATA_VOL = "canopyai-deepforest-data"

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)

HF_REPO = "restor/tcd-mask-rcnn-r50"
NATIVE = 2048
LOW = 512
RPN_TOPK = 1000
TREE_CLASS = 1                       # verified empirically in the published run (meta.tree_class)

app = modal.App(APP)
vol = modal.Volume.from_name(OUT_VOL)
hfvol = modal.Volume.from_name(HFDATA_VOL)
hf_secret = modal.Secret.from_name("huggingface")

# Image definition copied verbatim from restor_baseline/restor_modal.py so the build is cached
# and the runtime is the same one that produced the published arms.
image = (
    modal.Image.from_registry("pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel")
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
    .apt_install("git", "g++", "ninja-build", "libgl1", "libglib2.0-0")
    .env({"TORCH_CUDA_ARCH_LIST": "8.0", "FORCE_CUDA": "1"})
    .pip_install("numpy<2", "opencv-python-headless", "pycocotools", "pillow",
                 "datasets==4.0.0", "huggingface_hub")
    .run_commands("pip install --no-build-isolation "
                  "'git+https://github.com/facebookresearch/detectron2.git'")
    .env({"HF_HOME": "/hfdata/hf_cache", "HF_HUB_OFFLINE": "0",
          "HF_DATASETS_OFFLINE": "1"})
    .add_local_file(os.path.join(PKG, "modal_tcd_multiseed", "phase4", "manifest.json"),
                    "/root/manifest.json")
    .add_local_file(os.path.join(PKG, "test_gt.json"), "/root/test_gt.json")
)

def _dirs(rpn_topk):
    """One UNIQUE output dir per RPN arm; the rpn1000 arm is the one recorded 2026-09-22."""
    out_dir = f"/vol/out/native_raster/restor_rpn{rpn_topk}_r{NATIVE}"
    return out_dir, os.path.join(out_dir, "tiles"), os.path.join(out_dir, "run.log")


def _log(msg, log):
    print(msg, flush=True)
    with open(log, "a") as fh:
        fh.write(msg + "\n")


def _env_versions():
    import numpy, PIL, pycocotools, torch
    try:
        import detectron2; d2 = detectron2.__version__
    except Exception:
        d2 = "?"
    return {"torch": torch.__version__, "numpy": numpy.__version__,
            "pillow": PIL.__version__, "pycocotools": getattr(pycocotools, "__version__", "?"),
            "detectron2": d2, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


@app.function(gpu="A100", image=image, volumes={"/vol": vol, "/hfdata": hfvol},
              timeout=4 * 3600, cpu=8, memory=32768, secrets=[hf_secret])
def predict(limit: int = 0, rpn_topk: int = RPN_TOPK):
    import numpy as np
    import torch
    from PIL import Image
    from datasets import load_dataset
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.logger import setup_logger
    from huggingface_hub import hf_hub_download
    from pycocotools import mask as maskUtils

    assert torch.cuda.is_available(), "no CUDA"
    setup_logger()
    OUT_DIR, TILES, LOG = _dirs(rpn_topk)
    os.makedirs(TILES, exist_ok=True)
    _log(f"[restor-native] start  OUT_DIR={OUT_DIR}  rpn_topk={rpn_topk}  "
         f"env={json.dumps(_env_versions())}", LOG)

    ckpt = hf_hub_download(HF_REPO, "model.pth")
    cfg_fp = hf_hub_download(HF_REPO, "config.yaml")
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(cfg_fp)
    cfg.MODEL.WEIGHTS = ckpt
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.TEST.DETECTIONS_PER_IMAGE = 600
    cfg.INPUT.MIN_SIZE_TEST = 0
    cfg.INPUT.MAX_SIZE_TEST = 2048
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = rpn_topk
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = rpn_topk
    _log(f"[restor-native] score_thr={cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST} "
         f"dets={cfg.TEST.DETECTIONS_PER_IMAGE} num_classes={cfg.MODEL.ROI_HEADS.NUM_CLASSES} "
         f"min_size_test={cfg.INPUT.MIN_SIZE_TEST} max={cfg.INPUT.MAX_SIZE_TEST} "
         f"fmt={cfg.INPUT.FORMAT} rpn_pre={cfg.MODEL.RPN.PRE_NMS_TOPK_TEST} "
         f"rpn_post={cfg.MODEL.RPN.POST_NMS_TOPK_TEST}", LOG)
    predictor = DefaultPredictor(cfg)

    gt = json.load(open("/root/test_gt.json"))
    manifest = json.load(open("/root/manifest.json"))["feat_test"]
    ds = load_dataset("restor/tcd", split="test")
    idx = {int(i): k for k, i in enumerate(ds["image_id"])}
    tids = [t for t in sorted(gt) if t in manifest]
    if limit:
        tids = tids[:limit]

    done = {f[:-5] for f in os.listdir(TILES) if f.endswith(".json")}
    todo = [t for t in tids if t not in done]
    _log(f"[restor-native] {len(tids)} tiles, {len(done)} already on volume, {len(todo)} to do", LOG)

    n_tree = n_can = 0
    for k, tid in enumerate(todo):
        iid = int(manifest[tid]["image_id"])
        rgb = np.asarray(ds[idx[iid]]["image"].convert("RGB"))
        inst = predictor(rgb[:, :, ::-1])["instances"].to("cpu")
        cls = inst.pred_classes.numpy()
        boxes = inst.pred_boxes.tensor.numpy()
        scores = inst.scores.numpy()
        rle_n, rle_l = [], []
        for i in range(len(cls)):
            m = inst.pred_masks[i].numpy()                        # (2048,2048) bool, NATIVE
            assert m.shape == (NATIVE, NATIVE), m.shape
            r = maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
            rle_n.append({"size": [NATIVE, NATIVE], "counts": r["counts"].decode("ascii")})
            small = np.asarray(Image.fromarray(m.astype(np.uint8) * 255)
                               .resize((LOW, LOW), Image.BILINEAR)) >= 128   # == published path
            r = maskUtils.encode(np.asfortranarray(small.astype(np.uint8)))
            rle_l.append({"size": [LOW, LOW], "counts": r["counts"].decode("ascii")})
        rec = {"boxes_2048": np.round(boxes, 2).tolist(),
               "scores": np.round(scores, 4).tolist(),
               "pred_classes": cls.astype(int).tolist(),
               "masks_rle_2048": rle_n, "masks_rle_512": rle_l}
        tmp = os.path.join(TILES, tid + ".json.tmp")
        json.dump(rec, open(tmp, "w"))
        os.replace(tmp, os.path.join(TILES, tid + ".json"))      # atomic: no half-written tiles
        n_tree += int((cls == TREE_CLASS).sum()); n_can += int((cls != TREE_CLASS).sum())
        if (k + 1) % 10 == 0 or k + 1 == len(todo):
            vol.commit()
            _log(f"  {k+1}/{len(todo)} tiles this run ({len(done)+k+1}/{len(tids)} total)  "
                 f"tree={n_tree} canopy={n_can}", LOG)
    vol.commit()
    _log("[restor-native] predict done", LOG)
    return {"tiles_total": len(tids), "done_this_run": len(todo)}


@app.function(image=image, volumes={"/vol": vol}, timeout=3600, cpu=2, memory=16384)
def assemble(rpn_topk: int = RPN_TOPK):
    """CPU-only: tile files -> two preds files in the standard schema. Re-runnable; refuses to
    overwrite an existing output."""
    OUT_DIR, TILES, LOG = _dirs(rpn_topk)
    gt = json.load(open("/root/test_gt.json"))
    manifest = json.load(open("/root/manifest.json"))["feat_test"]
    tids = [t for t in sorted(gt) if t in manifest]
    missing = [t for t in tids if not os.path.exists(os.path.join(TILES, t + ".json"))]
    assert not missing, f"{len(missing)} tiles missing, e.g. {missing[:3]}"
    base = {"model": "Restor Mask R-CNN R50-FPN (restor/tcd-mask-rcnn-r50), whole 2048 tiles, "
                     "released checkpoint", "hf_repo": HF_REPO, "tree_class": TREE_CLASS,
            "score_thr": 0.05, "dets_per_image": 600, "rpn_topk_test": rpn_topk,
            "rpn_topk_overridden": True, "emits_all_classes": True, "n_tiles": len(tids),
            "source_tiles": TILES}
    for key, res, name in (("masks_rle_2048", NATIVE, f"preds_restor_rpn{rpn_topk}_r{NATIVE}.json"),
                           ("masks_rle_512", LOW, f"preds_restor_rpn{rpn_topk}_r{LOW}_incontainer.json")):
        fp = os.path.join(OUT_DIR, name)
        if os.path.exists(fp):                       # never overwrite; re-runnable after a drop
            _log(f"[restor-native] {fp} exists -- skipping", LOG); continue
        preds, nt, nc = {}, 0, 0
        for t in tids:
            r = json.load(open(os.path.join(TILES, t + ".json")))
            preds[t] = {"boxes_2048": r["boxes_2048"], "scores": r["scores"],
                        "masks_rle": r[key], "pred_classes": r["pred_classes"]}
            nt += sum(1 for c in r["pred_classes"] if c == TREE_CLASS)
            nc += sum(1 for c in r["pred_classes"] if c != TREE_CLASS)
        meta = dict(base, mask_res=res, n_crowns=nt, n_canopy=nc)
        json.dump({"meta": meta, "preds": preds}, open(fp, "w"))
        _log(f"[restor-native] wrote {fp}: {len(preds)} tiles, {nt} tree, {nc} canopy", LOG)
    vol.commit()
    return {"out_dir": OUT_DIR}
