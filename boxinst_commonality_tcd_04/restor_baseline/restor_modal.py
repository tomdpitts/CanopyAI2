"""Modal A100 app: Restor's released Mask R-CNN (`restor/tcd-mask-rcnn-r50`) on the OAM-TCD
439 test tiles, dumped into OUR preds schema so score_coco.py grades it under the frozen
Table-1 protocol (PROTOCOL_439.md).

WHY THIS EXISTS
    Table 1 currently cites Restor's published 0.432, measured under their own protocol
    (no canopy-ignore, their scorer). The abstract additionally claims 0.586, which has no
    provenance anywhere in this repo. This is the first time their released checkpoint is
    actually run by us.

PROTOCOL ALIGNMENT -- their test config is already whole-tile, so geometry is untouched:
    MIN_SIZE_TEST 0 / MAX_SIZE_TEST 2048   their config -> no resize, whole 2048^2 tile.
                                           Identical geometry to LACE; nothing is tiled.
    SCORE_THRESH_TEST 0.2 -> 0.05          their default truncates the PR tail; 0.05 is the
                                           frozen floor for every row.
    DETECTIONS_PER_IMAGE 512 -> 600        the frozen maxDets.
    NUM_CLASSES 2                          OAM-TCD COCO has category_id 1 = canopy,
                                           2 = tree (prepare_test.py:8-9). Detectron2 maps
                                           sorted ids to contiguous, so class 0 = canopy,
                                           class 1 = tree. `--smoke` VERIFIES this from the
                                           instance-count/area split rather than trusting it;
                                           canopy is few-and-huge, trees many-and-small.
                                           Only the tree class is emitted -- their canopy
                                           predictions are discarded, exactly as every other
                                           row has no canopy channel.

Masks come out of detectron2 at full 2048^2 and are downsampled to 512^2 (PIL BILINEAR,
threshold >=128) -- the same reduction detectree2_baseline/stitch.py applies, so all rows
land on the identical scoring raster.

Run:
    modal run restor_modal.py::predict --smoke 1     # 8 tiles, prints the class split
    modal run restor_modal.py::predict               # full 439
"""
import json
import os

import modal

APP = "tcd04-restor-baseline"
OUT_VOL = "tcd04-baselines-vol"                 # write: our-schema preds
HFDATA_VOL = "canopyai-deepforest-data"         # read: HF restor/tcd + hub cache

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REPO = os.path.dirname(PKG)

HF_REPO = "restor/tcd-mask-rcnn-r50"
RES = 512                                        # scoring raster, matches every other row

app = modal.App(APP)
vol = modal.Volume.from_name(OUT_VOL, create_if_missing=True)
hfvol = modal.Volume.from_name(HFDATA_VOL)
hf_secret = modal.Secret.from_name("huggingface")

# Same proven recipe as detectree2_baseline/detectree2_modal.py: modern CUDA torch base and
# build detectron2 from source against it (no prebuilt wheel exists for py>=3.10).
# TORCH_CUDA_ARCH_LIST 8.0 = A100 sm_80, so this image must run on A100.
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

OUT = "/vol/out"


@app.function(gpu="A100", image=image, volumes={"/vol": vol, "/hfdata": hfvol},
              timeout=4 * 3600, cpu=8, memory=32768, secrets=[hf_secret])
def predict(smoke: int = 0, tree_class: int = -1, rpn_topk: int = 0, tag: str = ""):
    """Run the released checkpoint over the 439 and write preds in our schema.

    tree_class: -1 = auto (pick the class with the SMALLER median instance area; canopy
    regions are far larger than individual crowns). `--smoke` prints the split so the
    choice is auditable rather than assumed."""
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
    setup_logger()          # DefaultPredictor does NOT do this, so without it the
    #                         DetectionCheckpointer's "Loading ..." / "Skip loading
    #                         parameter ... shape mismatch" lines are silently swallowed
    #                         and a failed weight load is indistinguishable from a quiet one.
    os.makedirs(OUT, exist_ok=True)

    ckpt = hf_hub_download(HF_REPO, "model.pth")
    cfg_fp = hf_hub_download(HF_REPO, "config.yaml")
    cfg = get_cfg()
    cfg.set_new_allowed(True)                    # their dump may carry keys this build lacks
    cfg.merge_from_file(cfg_fp)
    cfg.MODEL.WEIGHTS = ckpt
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05     # was 0.2 -- frozen floor
    cfg.TEST.DETECTIONS_PER_IMAGE = 600              # was 512 -- frozen maxDets
    cfg.INPUT.MIN_SIZE_TEST = 0                      # no resize (their own setting)
    cfg.INPUT.MAX_SIZE_TEST = 2048
    if rpn_topk:
        # THIRD truncation knob. Their config sets RPN PRE/POST_NMS_TOPK_TEST = 512
        # (detectron2's FPN default is 1000), which caps proposals before the ROI heads ever
        # run and left the released config emitting only ~98 dets/tile against LACE's 364 and
        # DetecTree2's 348. These are test-time inference knobs of exactly the same kind as
        # SCORE_THRESH_TEST, which we already raise, and the densest test tile holds 450 GT
        # instances -- so 512 proposals is a binding cap, not a neutral default. Reported as a
        # SEPARATE row, never silently merged into the released-config number.
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = rpn_topk
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = rpn_topk
    print(f"[restor] score_thr={cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST} "
          f"dets={cfg.TEST.DETECTIONS_PER_IMAGE} num_classes={cfg.MODEL.ROI_HEADS.NUM_CLASSES} "
          f"min_size_test={cfg.INPUT.MIN_SIZE_TEST} fmt={cfg.INPUT.FORMAT} "
          f"rpn_topk={cfg.MODEL.RPN.POST_NMS_TOPK_TEST}", flush=True)
    predictor = DefaultPredictor(cfg)

    gt = json.load(open("/root/test_gt.json"))
    manifest = json.load(open("/root/manifest.json"))["feat_test"]
    ds = load_dataset("restor/tcd", split="test")
    idx = {int(i): k for k, i in enumerate(ds["image_id"])}
    tids = [t for t in sorted(gt) if t in manifest]
    if smoke:
        tids = tids[:max(8, int(smoke))]
    print(f"[restor] {len(tids)} tiles", flush=True)

    # --- pass 1 on a few tiles: identify the tree class empirically -------------------
    def _infer(tid):
        iid = int(manifest[tid]["image_id"])
        rgb = np.asarray(ds[idx[iid]]["image"].convert("RGB"))
        out = predictor(rgb[:, :, ::-1])                      # detectron2 wants BGR
        return out["instances"].to("cpu")

    if tree_class < 0:
        stats, n_empty = {}, 0
        for tid in tids[:8]:
            inst = _infer(tid)
            cls = inst.pred_classes.numpy()
            n_empty += int(len(cls) == 0)
            # sum over (H,W) rather than reshape(len(cls),-1): the reshape cannot infer -1
            # from a size-0 array, so a tile with no detections crashed it.
            areas = inst.pred_masks.numpy().sum(axis=(1, 2)) if len(cls) else np.zeros(0)
            print(f"[restor]   {tid}: {len(cls)} instances"
                  f"{'' if len(cls) else '  <-- EMPTY'}", flush=True)
            for c in np.unique(cls):
                stats.setdefault(int(c), []).extend(areas[cls == c].tolist())
        if not stats:
            raise RuntimeError(
                f"All {len(tids[:8])} probe tiles produced ZERO detections at "
                f"score_thr={cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}. That is not a plausible "
                f"result for a trained crown model -- suspect the checkpoint did not load "
                f"(check the setup_logger output above for 'Skip loading parameter' / shape "
                f"mismatch), or that INPUT.FORMAT/normalisation is wrong. Do NOT proceed to "
                f"the full run until this is understood.")
        for c, a in sorted(stats.items()):
            print(f"[restor]   class {c}: n={len(a)} median_area={np.median(a):.0f}px "
                  f"mean={np.mean(a):.0f}px", flush=True)
        if n_empty:
            print(f"[restor]   ({n_empty}/8 probe tiles had no detections)", flush=True)
        tree_class = min(stats, key=lambda c: float(np.median(stats[c])))
        print(f"[restor] -> tree_class={tree_class} (smaller median area; "
              f"canopy regions are far larger than crowns)", flush=True)

    # --- full pass -------------------------------------------------------------------
    preds, n_kept, n_drop = {}, 0, 0
    for k, tid in enumerate(tids):
        inst = _infer(tid)
        cls = inst.pred_classes.numpy()
        # Emit EVERY instance and record its class, rather than dropping the canopy class
        # here. Dropping it at generation time silently hands Restor canopy-ignore in the
        # canopy=FP arm too: it labels canopy regions "canopy" and escapes the penalty,
        # while LACE/DetecTree2 have no canopy class, predict "tree" there, and eat the FP.
        # Scoring decides what to keep; `pred_classes` makes that choice explicit and
        # reversible without another GPU run.
        sel = np.arange(len(cls))
        n_kept += int((cls == tree_class).sum()); n_drop += int((cls != tree_class).sum())
        boxes = inst.pred_boxes.tensor.numpy()[sel]           # xyxy @2048
        scores = inst.scores.numpy()[sel]
        rles = []
        for i in sel:
            m = inst.pred_masks[i].numpy()                    # (2048,2048) bool
            small = np.asarray(Image.fromarray(m.astype(np.uint8) * 255)
                               .resize((RES, RES), Image.BILINEAR)) >= 128
            r = maskUtils.encode(np.asfortranarray(small.astype(np.uint8)))
            rles.append({"size": [RES, RES], "counts": r["counts"].decode("ascii")})
        preds[tid] = {"boxes_2048": np.round(boxes, 2).tolist(),
                      "scores": np.round(scores, 4).tolist(),
                      "masks_rle": rles,
                      "pred_classes": cls[sel].astype(int).tolist()}
        if (k + 1) % 25 == 0 or k + 1 == len(tids):
            print(f"  {k+1}/{len(tids)} tiles, {n_kept} crowns", flush=True)

    res = {"meta": {"model": "Restor Mask R-CNN R50-FPN (restor/tcd-mask-rcnn-r50), "
                             "whole 2048 tiles, released checkpoint",
                    "hf_repo": HF_REPO, "mask_res": RES, "tree_class": int(tree_class),
                    "score_thr": 0.05, "dets_per_image": 600,
                    "rpn_topk_test": int(cfg.MODEL.RPN.POST_NMS_TOPK_TEST),
                    "rpn_topk_overridden": bool(rpn_topk),
                    "n_tiles": len(preds), "n_crowns": n_kept,
                    "n_canopy": n_drop,
                    "emits_all_classes": True},
           "preds": preds}
    suffix = tag or ("_smoke" if smoke else "")
    fp = os.path.join(OUT, f"preds_restor{suffix}.json")
    json.dump(res, open(fp, "w"))
    vol.commit()
    print(f"[restor] saved {fp}: {len(preds)} tiles, {n_kept} tree crowns, "
          f"{n_drop} canopy predictions discarded", flush=True)
    return {"preds": fp, "n_tiles": len(preds), "n_crowns": n_kept}
