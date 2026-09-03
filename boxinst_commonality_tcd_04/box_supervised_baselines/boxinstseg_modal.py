"""Box-supervised baselines on our 792 box-only OAM-TCD tiles — Modal app.

Trains Box2Mask (TPAMI 2024) and BoxInst (CVPR 2021) via BoxInstSeg
(https://github.com/LiWentomng/BoxInstSeg), the Box2Mask authors' toolbox, which carries
BoxInst / DiscoBox / BoxLevelSet / Box2Mask in one mmdet-2.25 codebase. Every function takes
`--method`; each method keeps its OWN published config and is changed only where our data
forces it (one class, our paths, and a schedule rescaled to our dataset size).

WHAT MAKES THIS COMPARABLE TO LACE
  * The SAME PNG crops DetecTree2 trained on, read from `tcd-detectree2-vol` at
    /vol/data/{train,val}/images. Nothing is re-tiled or re-encoded, so "same training data" is
    a filesystem fact rather than two builders agreeing.
  * Supervision reduced to boxes by `make_box_only_coco.py`, which replaces each polygon with
    its own bounding rectangle and ASSERTS that nothing else survives.
  * Canopy stays `iscrowd=1` (ignore, never background), as everywhere else in this project.
  * Model selection on val **box** AP50 -- segm AP against a box-only COCO would be scoring
    rectangles against rectangles and would mean nothing.
  * Scoring is the frozen `score_coco.py` gate, after stitching 1024 subtiles back to 2048.

BUILD RISK. mmcv-full 1.x needs an exact torch+CUDA pair and must come from OpenMMLab's
prebuilt wheel index -- compiling it from source on Modal is slow and fragile. `imports_ok` is
a cheap gate that proves the whole stack loads and the BoxInst config builds BEFORE any A100
hour is spent on a run that would die at import.

Run (see PLAN.md section 8 for the full order):
    modal run boxinstseg_modal.py::imports_ok                    # cheap gate, run this first
    modal run boxinstseg_modal.py::dryrun     --method box2mask  # config/data/model, no GPU-hours
    modal run boxinstseg_modal.py::fetch_ckpt --method box2mask  # their COCO-box checkpoint
    modal run boxinstseg_modal.py::train      --method box2mask
    modal run boxinstseg_modal.py::predict    --method box2mask
    modal run boxinstseg_modal.py::stitch     --method box2mask
"""
import os

import modal

APP = "tcd04-boxinstseg"
DT2_VOL = "tcd-detectree2-vol"        # the built 1024@50% crops + coco.json
OUT_VOL = "tcd04-baselines-vol"       # checkpoints + predictions

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)

# torch 1.11 / cu113 has mmcv-full 1.5.x wheels for cp37-cp310, and cu113 covers the A100
# (sm_80). The fork asserts mmcv-full in [1.3.17, 1.6.0], so 1.5.3.
#
# NVCC IS REQUIRED, despite the top-level setup.py declaring `ext_modules=[]`. That is
# misleading: BoxInstSeg keeps its compiled ops in their own sub-packages with their own
# setup.py. BOTH are needed even to import mmdet, because heads on the package init chain
# import them at module scope:
#   pairwise    -> condinst_head  (BoxInst)
#   tree_filter -> box_solov2_head (BoxLevelSet), box2mask_head (Box2Mask)
# tree_filter includes THC headers that PyTorch deleted in 1.11; `patch_boxinstseg.py` strips
# them, which is safe because they are vestigial (the only THC symbol used is `atomicAdd`, a
# CUDA built-in). That keeps ALL FOUR methods available rather than only BoxInst and DiscoBox.
PY_VER, TORCH, TV, CUDA, MMCV = "3.10", "1.11.0", "0.12.0", "cu113", "1.5.3"
MMCV_INDEX = f"https://download.openmmlab.com/mmcv/dist/{CUDA}/torch{TORCH}/index.html"
TORCH_INDEX = f"https://download.pytorch.org/whl/{CUDA}"
BOXINSTSEG = "https://github.com/LiWentomng/BoxInstSeg.git"

app = modal.App(APP)
dt2vol = modal.Volume.from_name(DT2_VOL)
vol = modal.Volume.from_name(OUT_VOL, create_if_missing=True)

image = (
    # CUDA 11.3 devel for nvcc, matching the cu113 torch/mmcv wheels exactly. FORCE_CUDA is
    # needed because no GPU is attached at build time, and the arch list is narrowed to the
    # A100 (sm_80) so the compile stays short.
    modal.Image.from_registry("nvidia/cuda:11.3.1-devel-ubuntu20.04", add_python=PY_VER)
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC",
          "TORCH_CUDA_ARCH_LIST": "8.0", "FORCE_CUDA": "1",
          "CUDA_HOME": "/usr/local/cuda",
          # tree_filter's setup.py demands LD_LIBRARY_PATH be set explicitly.
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
          # The base image leaves CXX pointing at clang++, which torch's cpp_extension probes
          # for and then fails on ("which clang++" -> exit 1). nvcc also expects a GCC host
          # compiler, so pin both explicitly to what build-essential provides.
          "CC": "gcc", "CXX": "g++"})
    .apt_install("git", "build-essential", "libgl1", "libglib2.0-0")
    .pip_install(f"torch=={TORCH}+{CUDA}", f"torchvision=={TV}+{CUDA}",
                 extra_index_url=TORCH_INDEX)
    # Everything pinned to the numpy-1.x era: torch 1.11's C extension is compiled against the
    # numpy 1.x ABI, so any numpy>=2 makes it fail with "_ARRAY_API not found" -- a WARNING at
    # import that only becomes an error deep in training. skimage is an UNDECLARED dependency of
    # condinst_head (BoxInst's LAB colour space). setuptools is held below 60 so the ops'
    # `setup.py build_ext` still works.
    .pip_install("setuptools==59.5.0", "wheel")
    .pip_install("numpy==1.23.5", "yapf==0.32.0", "pycocotools", "scipy==1.10.1",
                 "matplotlib==3.7.3", "terminaltables", "six",
                 "opencv-python-headless==4.8.1.78", "scikit-image==0.19.3")
    # find_links (-f), not an index: OpenMMLab serve a flat HTML links page. If the exact
    # (mmcv, torch, cu, cpXX) wheel is missing, pip SILENTLY falls back to a source build --
    # hence the version assertions in imports_ok.
    .pip_install(f"mmcv-full=={MMCV}", find_links=MMCV_INDEX)
    .add_local_file(os.path.join(HERE, "patch_boxinstseg.py"), "/root/patch_boxinstseg.py",
                    copy=True)
    .run_commands(
        f"git clone {BOXINSTSEG} /opt/BoxInstSeg",
        "cd /opt/BoxInstSeg && pip install -v --no-build-isolation -e .",
        # build_ext --inplace (not their `build develop`) so pairwise_ext.so lands beside
        # pairwise.py and resolves as mmdet.ops.pairwise.pairwise_ext.
        "python /root/patch_boxinstseg.py /opt/BoxInstSeg",
        "cd /opt/BoxInstSeg/mmdet/ops/pairwise && python setup.py build_ext --inplace",
    )
    # tree_filter's setup.py gates on `torch.cuda.is_available()` and ignores FORCE_CUDA, so
    # unlike pairwise it cannot compile on a CPU builder. Give this one layer a GPU.
    .run_commands(
        "cd /opt/BoxInstSeg/mmdet/ops/tree_filter && python setup.py build develop",
        gpu="A100",
    )
    .run_commands(
        # BoxInstSeg's runtime.txt lists a bare `numpy`, which can pull numpy 2 back in on top
        # of everything above. Re-pin LAST so the resolved version is the one torch needs.
        "pip install --no-deps --force-reinstall numpy==1.23.5",
    )
    # Box2Mask's COCO checkpoint is only published as a Google Drive link (their README model
    # zoo), which needs gdown's confirm-token handling. It pulls no numpy, so it is safe after
    # the re-pin above.
    .pip_install("gdown==5.2.0")
    .add_local_file(os.path.join(HERE, "make_box_only_coco.py"),
                    "/root/make_box_only_coco.py")
    .add_local_file(os.path.join(PKG, "detectree2_baseline", "stitch.py"), "/root/stitch.py")
    .add_local_file(os.path.join(HERE, "inject_boxes.py"), "/root/inject_boxes.py")
    .add_local_file(os.path.join(PKG, "test_gt.json"), "/root/test_gt.json")
)

DATA = "/dt2/data"
OUT = "/vol/boxinst"


@app.function(image=image, gpu="A100", timeout=1800)
def imports_ok():
    """Cheap gate: prove the stack loads and a BoxInst model actually builds.

    Deliberately builds the DETECTOR, not just the imports: an mmcv/mmdet version mismatch
    typically imports fine and then fails in the registry, which is exactly the failure that
    would otherwise surface an hour into training."""
    import glob

    import mmcv
    import torch
    info = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "cuda": torch.version.cuda, "mmcv": mmcv.__version__}
    assert torch.cuda.is_available(), "no CUDA"

    # torch 1.11 is compiled against the numpy 1.x ABI. A numpy 2 in the image degrades to a
    # warning at import and an error much later, so make it fail HERE instead.
    import numpy as np
    info["numpy"] = np.__version__
    assert np.__version__.startswith("1."), f"numpy {np.__version__} breaks torch {TORCH}'s ABI"
    assert torch.from_numpy(np.zeros(3, np.float32)).sum().item() == 0.0, \
        "torch<->numpy interop broken (ABI mismatch)"

    # The classic failure is not an import error: mmcv-full loads, and its compiled ops then
    # fail at call time against a mismatched torch ABI or an unsupported arch. Call one on the
    # GPU so that failure surfaces here rather than an hour into training.
    from mmcv.ops import nms
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.], [50., 50., 60., 60.]],
                         device="cuda")
    scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
    kept = nms(boxes, scores, 0.5)[1]
    assert len(kept) == 2, f"mmcv nms returned {len(kept)} boxes, expected 2"
    info["mmcv_cuda_op"] = f"nms OK, kept {len(kept)}/3"
    # The compiled BoxInst op, built from mmdet/ops/pairwise. condinst_head imports it at
    # module scope, so a missing .so takes down the whole mmdet import.
    from mmdet.ops.pairwise import pairwise_nlog
    import tree_filter_cuda                       # noqa: F401
    info["compiled_ops"] = "pairwise_ext + tree_filter_cuda importable"

    import mmdet
    from mmdet.models import build_detector
    info["mmdet"] = mmdet.__version__
    info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    # All four methods must register now that tree_filter builds -- confirm, because a silent
    # import failure would leave a method missing rather than erroring.
    from mmdet.models.builder import DETECTORS
    info["registered"] = [k for k in ("CondInst", "DiscoBoxSOLOv2", "BoxLevelSet", "Box2Mask")
                          if k in DETECTORS.module_dict]
    cfgs = sorted(glob.glob("/opt/BoxInstSeg/configs/boxinst/*.py"))
    info["boxinst_configs"] = [os.path.basename(c) for c in cfgs]
    target = [c for c in cfgs if "r50" in c and "coco" in c and "1x" in c]
    assert target, f"no BoxInst R-50 1x config among {info['boxinst_configs']}"
    cfg = mmcv.Config.fromfile(target[0])
    info["config_used"] = os.path.basename(target[0])
    model = build_detector(cfg.model)
    info["model_class"] = type(model).__name__
    info["n_params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    print(info, flush=True)
    return info


@app.function(image=image, volumes={"/dt2": dt2vol}, timeout=1800, cpu=4, memory=16384)
def prep_data():
    """Derive the box-only COCO beside DetecTree2's own, on its volume. CPU, no GPU."""
    import json
    import sys
    sys.path.insert(0, "/root")
    import make_box_only_coco as B

    out = {}
    for split in ("train", "val"):
        src = os.path.join(DATA, split, "coco.json")
        assert os.path.exists(src), f"{src} missing -- build the DetecTree2 datasets first"
        d = B.convert(json.load(open(src)))
        stats = B.verify(d)                       # asserts no mask information survived
        dst = os.path.join(DATA, split, "coco_boxonly.json")
        json.dump(d, open(dst, "w"))
        out[split] = stats
        print(f"[box-only] {split}: {stats} -> {dst}", flush=True)
    dt2vol.commit()
    return out


# ------------------------------------------------------------------ config
METHODS = ("boxinst", "box2mask")

REPORTS = f"{OUT}/reports"
MODEL_DESC = {
    "boxinst": "BoxInst R-50 FPN (BoxInstSeg, CVPR 2021)",
    "box2mask": "Box2Mask-T R-50 (BoxInstSeg, TPAMI 2024), COCO-box-pretrained",
}

# BoxInstSeg's own COCO model-zoo checkpoints (Google Drive ids from their README table).
# Fine-tuning from these is NOT mask leakage: their COCO training used box annotations only --
# that is the method -- so these weights have never seen a mask. See PLAN.md section 5.
# `cls_key` is the classifier whose 80-class weights must be dropped; `n_cls_ckpt` is the row
# count we assert to prove we downloaded the real file rather than a Drive error page.
COCO_CKPT = {
    "box2mask": dict(gid="1KFJabXdGodgcO-GerNpJkgyZX9whnNYe",
                     name="box2mask_r50_lsj_8x2_50e_coco.pth",
                     cls_key="panoptic_head.cls_embed", n_cls_ckpt=81),
    "box2mask_swin-l": dict(gid="1EKa4cna_A0ec-HFL8jli_FCqXSZ3bWhV",
                            name="box2mask_swin-l_lsj_8x1_50e_coco.pth",
                            cls_key="panoptic_head.cls_embed", n_cls_ckpt=81),
}
# Backbone variants of the same method. The key into COCO_CKPT is method+suffix.
BACKBONES = {
    "box2mask": {
        "r50": dict(cfg="box2mask_r50_lsj_8x2_50e_coco.py", micro_batch=2, coco_ap=35.9),
        "swin-l": dict(cfg="box2mask_swin-l-p4-w12-384-lsj_8x1_50e_coco.py",
                       micro_batch=1, coco_ap=42.5),
    },
    "boxinst": {"r50": dict(cfg=None, micro_batch=4, coco_ap=33.2)},
}


def _ckpt_key(method, backbone):
    return method if backbone in (None, "r50") else f"{method}_{backbone}"
CKPT_DIR = f"{OUT}/ckpt"


def _report(name, payload):
    """Persist a function's result to the volume AND print it.

    Long runs must be launched with `modal run --detach`, or a local network blip stops the
    app mid-training (it has already happened twice). Detached runs cannot hand a return value
    back to the client, so the artifact on the volume is the only record."""
    import json
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, f"{name}.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    vol.commit()
    print(json.dumps(payload, indent=2, default=str), flush=True)
    return payload


def _stripped_ckpt(method, backbone="r50"):
    spec = COCO_CKPT[_ckpt_key(method, backbone)]
    return os.path.join(CKPT_DIR, spec["name"].replace(".pth", "_nocls.pth"))


def _attach_data(cfg, train_pipeline, test_pipeline, samples_per_gpu, workers_per_gpu=8):
    """The two splits, identical for every method. See PLAN.md section 4.

    TRAIN on canopy-blacked crops with canopy annotations removed, because the BoxInst family
    cannot be given an ignore label (see prep_data_blackout).

    VALIDATE on the ORIGINAL crops with canopy retained as `iscrowd=1`. This is deliberate and
    is NOT the training condition: selection must happen under the condition the model is
    actually tested in -- real imagery, canopy neither rewarded nor punished. pycocotools
    ignores crowd matches, so val box AP50 here is canopy-neutral in exactly the way
    score_coco.py is on the 439. Selecting on blacked val images instead would optimise for a
    distribution that only exists during training; selecting against a canopy-free val GT
    would actively favour checkpoints that suppress canopy, which is the bias we just spent
    the blackout removing.
    """
    def _ds(split, pipeline, ann, imgs, filter_empty_gt=True):
        return dict(type="CocoDataset", classes=("tree",),
                    ann_file=f"{DATA}/{split}/{ann}",
                    img_prefix=f"{DATA}/{split}/{imgs}/", pipeline=pipeline,
                    filter_empty_gt=filter_empty_gt)

    val = _ds("val", test_pipeline, "coco_boxonly.json", "images")
    cfg.data = dict(samples_per_gpu=samples_per_gpu,
                    workers_per_gpu=workers_per_gpu,
                    # filter_empty_gt=False: removing the canopy annotations left 618
                    # subtiles with no annotations, which mmdet would otherwise drop. That
                    # emptiness is an artefact of OUR blackout, not of the data -- those
                    # subtiles carried canopy annotations until we stripped them, and
                    # DetecTree2 trained on all of them. Measured, they are bimodal: 292 are
                    # under 50% blacked (real ground/road with no trees -- exactly the hard
                    # negatives that hold precision up) and 152 are over 90% blacked (trivial,
                    # 2.5% of the set, no GT so they contribute only background loss). Dropping
                    # them would let our intervention silently shrink the baseline's data.
                    train=_ds("train", train_pipeline, "coco_boxonly_black.json",
                              "images_canopyblack", filter_empty_gt=False),
                    val=val, test=val)
    return cfg


def _cfg_boxinst(work_dir, samples_per_gpu, epochs, seed, score_thr, n_train_imgs):
    """BoxInst's own published 1x config, adapted only where our data forces it.

    Deliberately NOT re-tuned. The four changes are: one class instead of eighty; our dataset
    paths; the LR scaled by the linear rule for a smaller batch; and the multiscale range
    rebased onto our native 1024 tile. Everything else -- 12 epochs, decay at 8/11, warmup 500,
    SGD momentum/weight-decay, the losses -- is theirs untouched.
    """
    import mmcv
    cfg = mmcv.Config.fromfile(
        "/opt/BoxInstSeg/configs/boxinst/boxinst_r50_fpn_1x_coco.py")

    cfg.model.bbox_head.num_classes = 1

    # Their multiscale is short-side 640-800 against a 1333 cap, i.e. 0.8-1.0 of the max. Our
    # subtiles are 1024 native, so we keep the SAME relative range rather than their absolute
    # pixel sizes -- feeding 1024 crops to an 800-short-side pipeline would resample the arm
    # off OAM-TCD's 0.1 m/px and shrink exactly the small crowns that are hardest here.
    scales = [(1024, s) for s in (1024, 983, 942, 901, 860, 819)]
    nrm = cfg.img_norm_cfg          # top-level key; do not index into train_pipeline
    train_pipeline = [
        dict(type="LoadImageFromFile"),
        # with_mask=False is BoxInst's own setting: the head synthesises its projection
        # target from the BOX. Combined with our box-only COCO, supervision is box-only twice
        # over -- the polygons are absent from the file AND never requested.
        dict(type="LoadAnnotations", with_bbox=True, with_mask=False),
        dict(type="Resize", img_scale=scales, multiscale_mode="value", keep_ratio=True),
        dict(type="RandomFlip", flip_ratio=0.5),
        dict(type="Normalize", **nrm),
        dict(type="Pad", size_divisor=32),
        dict(type="DefaultFormatBundle"),
        dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
    ]
    test_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="MultiScaleFlipAug", img_scale=(1024, 1024), flip=False,
             transforms=[
                 dict(type="Resize", keep_ratio=True),
                 dict(type="RandomFlip"),
                 dict(type="Normalize", **nrm),
                 dict(type="Pad", size_divisor=32),
                 dict(type="ImageToTensor", keys=["img"]),
                 dict(type="Collect", keys=["img"]),
             ]),
    ]
    _attach_data(cfg, train_pipeline, test_pipeline, samples_per_gpu)

    # Linear scaling rule: their 0.005 is for 8 GPUs x 2 imgs = 16. Hardware forces this, not us.
    cfg.optimizer.lr = 0.005 * samples_per_gpu / 16.0

    # `pairwise_warmup` ramps in BoxInst's pairwise (mask) loss over an ABSOLUTE number of
    # iterations -- 10,000 of COCO 1x's ~88,700, i.e. the first 11.3% of training. Left alone on
    # our ~20k-iteration run it would span HALF the schedule and starve the mask branch. Scale it
    # to preserve their fraction, the same principle as rebasing the multiscale range.
    iters = -(-n_train_imgs // samples_per_gpu) * epochs
    cfg.model.mask_head.pairwise_warmup = max(1, round(iters * 10000 / 88715))
    cfg.runner.max_epochs = epochs
    cfg.lr_config.step = [round(epochs * 8 / 12), round(epochs * 11 / 12)]

    cfg.checkpoint_config = dict(interval=1, max_keep_ckpts=2)
    cfg.model.test_cfg.score_thr = score_thr
    cfg._sched = dict(runner="epoch", epochs=epochs, iters_per_epoch=iters // max(epochs, 1),
                      total_iters=iters, effective_batch=samples_per_gpu,
                      optimizer="SGD", lr=cfg.optimizer.lr, decay_at=cfg.lr_config.step,
                      pairwise_warmup=cfg.model.mask_head.pairwise_warmup,
                      load_from=None)
    return cfg


def _cfg_box2mask(work_dir, samples_per_gpu, epochs, seed, score_thr, n_train_imgs,
                  load_from=None, fp16=False, evals_per_epoch=1, backbone="r50"):
    """Box2Mask's own published 50e LSJ config, adapted only where our data forces it.

    Differences from the BoxInst arm, all of them theirs rather than ours:

    * The base is `coco_panoptic.py`, but their own config already `_delete_`s the whole `data`
      block and uses `CocoDataset`, so only the class counts need changing.
    * Supervision is box-only THREE times over: our COCO carries rectangles, `LoadAnnotations`
      is called with `with_mask=False`, and `GenerateBoxMask` then paints `gt_masks` from
      `gt_bboxes` alone. No polygon can reach the loss even in principle.
    * Their input size is already (1024, 1024) -- our native subtile -- so unlike BoxInst there
      is no multiscale range to rebase. The large-scale jitter (0.1-2.0) is theirs, untouched.
    * `IterBasedRunner`, so the schedule does NOT transfer from their 368,750 iterations the way
      BoxInst's epochs did; it is rescaled below.
    """
    import mmcv
    cfg = mmcv.Config.fromfile(
        "/opt/BoxInstSeg/configs/box2mask/" + BACKBONES["box2mask"][backbone]["cfg"])

    if backbone != "r50":
        # THEIR SWIN CONFIGS CARRY TWO DEFECTS. Both are fixed here, deliberately and visibly.
        #
        # 1. `custom_keys` (the per-layer AdamW decay/LR multipliers) is BUILT FROM `depths` in
        #    the Swin-T file. Swin-L redefines depths [2,2,6,2] -> [2,2,18,2] AFTER that dict was
        #    computed, and mmcv config inheritance is a dict merge, not a re-execution, so the
        #    inherited dict still describes Swin-T. Twelve of stage 2's blocks would silently
        #    miss the norm no-decay rule their recipe intends. Rebuild it for the real depths.
        # 2. The file is named `8x1` (8 GPUs x 1 image = effective batch 8) but inherits
        #    `samples_per_gpu=4` and `max_iters=184376` from Swin-T, which is 50 COCO epochs only
        #    at effective batch 32. The published config is therefore self-inconsistent and there
        #    is no "their recipe" to follow for the batch/LR pair. WE CHOOSE effective batch 16
        #    and lr 1e-4 -- identical to our own R-50 arm -- so that the backbone is the ONLY
        #    difference between our two Box2Mask rows. State this as our choice, never theirs.
        depths = cfg.model.backbone.depths
        embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
        norm_multi = dict(decay_mult=0.0, lr_mult=0.1)
        ck = {"backbone": dict(lr_mult=0.1, decay_mult=1.0),
              "backbone.patch_embed.norm": norm_multi,
              "backbone.norm": norm_multi,
              "absolute_pos_embed": norm_multi,
              "relative_position_bias_table": norm_multi,
              "query_embed": embed_multi, "query_feat": embed_multi,
              "level_embed": embed_multi}
        ck.update({f"backbone.stages.{si}.blocks.{bi}.norm": norm_multi
                   for si, nb in enumerate(depths) for bi in range(nb)})
        ck.update({f"backbone.stages.{si}.downsample.norm": norm_multi
                   for si in range(len(depths) - 1)})
        cfg.optimizer.paramwise_cfg = dict(custom_keys=ck, norm_decay_mult=0.0)
        cfg.optimizer.lr = 0.0001
        # Their Swin init downloads ImageNet-22k weights; `load_from` (their COCO box-supervised
        # checkpoint) supersedes it entirely, so drop the download rather than pay for it.
        cfg.model.backbone.init_cfg = None

    # One thing class, no stuff. `class_weight` is per-class plus a trailing no-object weight,
    # so it must be rebuilt rather than sliced.
    cfg.num_thing_classes = cfg.num_classes = 1
    cfg.num_stuff_classes = 0
    for head in (cfg.model.panoptic_head, cfg.model.panoptic_fusion_head):
        head.num_things_classes = 1
        head.num_stuff_classes = 0
    cfg.model.panoptic_head.loss_cls.class_weight = [1.0] * 1 + [0.1]

    # Their pipelines, verbatim apart from the test scale: (1333, 800) is COCO's, and our
    # subtiles are 1024 native, so resizing them would resample the arm off OAM-TCD's 0.1 m/px.
    train_pipeline = cfg.train_pipeline
    test_pipeline = cfg.test_pipeline
    test_pipeline[1]["img_scale"] = (1024, 1024)
    _attach_data(cfg, train_pipeline, test_pipeline, samples_per_gpu)

    # SCHEDULE. Their 368,750 iters x 16 imgs / 118,287 COCO images = 50 epochs. Reused as-is on
    # 6,692 subtiles that would be ~880 epochs, so it is rescaled to `epochs` passes over OUR
    # data, with the two LR decays at the same fractions of the run that COCO used
    # (327,778/368,750 = 88.89% and 355,092/368,750 = 96.30%).
    #
    # Rather than rescale the LR for a batch our single GPU can hold, gradient accumulation
    # restores their EXACT effective batch of 16 (8 GPUs x 2), so `lr=1e-4`, the AdamW betas,
    # the 0.05 weight decay, the 0.1 backbone lr_mult and the 0.01 grad clip are all theirs
    # untouched. An "iteration" below is therefore one micro-batch of `samples_per_gpu`, and
    # the optimiser steps every `cumulative_iters` of them.
    accum = max(1, round(16 / samples_per_gpu))
    ipe = -(-n_train_imgs // samples_per_gpu)
    max_iters = ipe * epochs
    cfg.max_iters = max_iters
    cfg.runner = dict(type="IterBasedRunner", max_iters=max_iters)
    cfg.lr_config.step = [round(max_iters * 327778 / 368750),
                          round(max_iters * 355092 / 368750)]
    # fp16 IS RISKY HERE, hence the flag and the probe. mmcv's `auto_fp16` on
    # `BaseDetector.forward` casts `img` to half AND wraps the whole forward in
    # `autocast(enabled=True)`, but Box2Mask's level-set loss goes through `tree_filter`, whose
    # CUDA kernels are hard-coded `float*` / `.data<float>()`. Half tensors reaching them throw.
    # Never enable this for a real run without `--max-iters` proving it first.
    cfg.optimizer_config = dict(
        type="GradientCumulativeFp16OptimizerHook" if fp16
        else "GradientCumulativeOptimizerHook",
        cumulative_iters=accum, grad_clip=dict(max_norm=0.01, norm_type=2))
    if fp16:
        cfg.optimizer_config["loss_scale"] = "dynamic"

    # Evaluate and checkpoint once per epoch, matching the BoxInst arm's selection cadence.
    # Their `dynamic_intervals` densifies evaluation near the end of a 50-epoch COCO run and
    # has nothing to say about a 12-epoch fine-tune; dropped so both arms select identically.
    eval_every = max(1, ipe // max(1, evals_per_epoch))
    cfg.workflow = [("train", eval_every)]
    cfg.checkpoint_config = dict(by_epoch=False, interval=eval_every, save_last=True,
                                 max_keep_ckpts=2)
    cfg._eval_every = eval_every
    cfg.pop("dynamic_intervals", None)
    cfg.log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook", by_epoch=False)])

    if load_from:
        cfg.load_from = load_from
    cfg._sched = dict(runner="iter", epochs=epochs, iters_per_epoch=ipe,
                      backbone=backbone, eval_every=eval_every, fp16=fp16,
                      total_iters=max_iters, micro_batch=samples_per_gpu,
                      cumulative_iters=accum, effective_batch=samples_per_gpu * accum,
                      optimizer="AdamW", lr=cfg.optimizer.lr, decay_at=cfg.lr_config.step,
                      num_queries=cfg.model.panoptic_head.num_queries,
                      max_per_image=cfg.model.test_cfg.max_per_image,
                      load_from=cfg.get("load_from"))
    return cfg


def _build_cfg(method, work_dir, samples_per_gpu, epochs, seed, score_thr, n_train_imgs,
               load_from=None, fp16=False, evals_per_epoch=1, backbone="r50"):
    """Dispatch to the method's own published config, then apply what every arm shares."""
    assert method in METHODS, f"{method} not in {METHODS}"
    if method == "boxinst":
        assert not load_from and not fp16, "BoxInst arm has no COCO checkpoint and no fp16 path"
        cfg = _cfg_boxinst(work_dir, samples_per_gpu, epochs, seed, score_thr, n_train_imgs)
    else:
        cfg = _cfg_box2mask(work_dir, samples_per_gpu, epochs, seed, score_thr, n_train_imgs,
                            load_from=load_from, fp16=fp16,
                            evals_per_epoch=evals_per_epoch, backbone=backbone)

    # bbox ONLY. segm AP against the box-only COCO would score rectangles against rectangles
    # and mean nothing; `save_best` makes the selection criterion val box AP50, matching how
    # LACE's detector and the DetecTree2 row are selected. `by_epoch` is set for us by
    # train_detector from the runner type, so it is deliberately absent here.
    cfg.evaluation = dict(interval=cfg._sched["eval_every"] if cfg._sched["runner"] == "iter"
                          else 1,
                          metric=["bbox"], save_best="bbox_mAP_50", rule="greater")
    cfg.log_config.setdefault("interval", 50)
    cfg.work_dir = work_dir
    cfg.seed = seed
    cfg.gpu_ids = [0]
    cfg.device = "cuda"
    cfg._sched["method"] = method
    return cfg


@app.function(image=image, volumes={"/vol": vol}, timeout=3600, cpu=4, memory=16384)
def fetch_ckpt(method: str = "box2mask", backbone: str = "r50", force: bool = False):
    """Download the method's COCO checkpoint and strip its 80-class classifier.

    WHY THIS IS NOT MASK LEAKAGE (PLAN.md §5). Box2Mask's COCO training used box annotations
    only -- that IS the method -- so these weights have never seen a mask. Initialising from
    them removes the data-starvation objection (34M+ parameters from ImageNet init on 792 tiles
    is not a fair test of the method) and matches how the DetecTree2 row was produced. Restor
    and DetecTree2 both start from COCO *instance-seg* weights and do not have this property.

    The classifier is dropped rather than left to mmcv's shape-mismatch warning, so that the
    one deliberately discarded tensor is named, counted and asserted instead of silently
    skipped in a log line nobody reads."""
    import json

    import gdown
    import torch

    spec = COCO_CKPT[_ckpt_key(method, backbone)]
    os.makedirs(CKPT_DIR, exist_ok=True)
    raw = os.path.join(CKPT_DIR, spec["name"])
    out = _stripped_ckpt(method, backbone)
    if os.path.exists(out) and not force:
        print(f"[fetch_ckpt] {out} exists", flush=True)
        return {"ckpt": out, "cached": True}

    if not os.path.exists(raw) or force:
        # Drive throttles this to well under 1 MB/s, so a 529 MB pull takes ~15 min and a
        # dropped client kills it. resume=True makes gdown use a DETERMINISTIC `<output>.part`
        # (without it the partial gets a random suffix and cannot be resumed), so a re-run
        # continues instead of restarting. Stale randomly-named partials are swept first.
        for f in os.listdir(CKPT_DIR):
            if f.startswith(spec["name"]) and f.endswith(".part") \
                    and f != spec["name"] + ".part":
                print(f"[fetch_ckpt] removing stale partial {f}", flush=True)
                os.remove(os.path.join(CKPT_DIR, f))
        gdown.download(id=spec["gid"], output=raw, quiet=False, resume=True)
    size_mb = round(os.path.getsize(raw) / 1e6, 1)

    # A Drive quota page is HTML and would land here as a few-KB "checkpoint"; torch.load then
    # fails, and the class-count assertion below catches anything that somehow loads.
    ck = torch.load(raw, map_location="cpu")
    sd = ck.get("state_dict", ck)
    key = spec["cls_key"]
    dropped = {k: list(v.shape) for k, v in sd.items() if k.startswith(key + ".")}
    assert dropped, f"{key}.* not in checkpoint -- wrong file? keys: {list(sd)[:5]}"
    assert dropped[key + ".weight"][0] == spec["n_cls_ckpt"], \
        f"{key}.weight is {dropped[key + '.weight']}, expected {spec['n_cls_ckpt']} rows"
    for k in list(dropped):
        sd.pop(k)
    torch.save({"state_dict": sd, "meta": {"source": spec["name"], "dropped": dropped}}, out)
    os.remove(raw)
    vol.commit()
    return _report(f"fetch_ckpt_{_ckpt_key(method, backbone)}",
                   {"ckpt": out, "downloaded_mb": size_mb, "tensors_kept": len(sd),
                    "dropped": dropped})


def _train_impl(method: str = "box2mask", epochs: int = 10, samples_per_gpu: int = 0,
                seed: int = 0, score_thr: float = 0.05, max_iters: int = 0,
                max_hours: float = 0.0, fp16: bool = False, patience: int = 0,
                min_epochs: int = 0, evals_per_epoch: int = 1, smoke: bool = False,
                backbone: str = "r50", resume: bool = False):
    """Train one box-supervised baseline on the 792 box-only tiles. Selection: val box AP50.

    `max_iters` > 0 truncates the run WITHOUT changing the schedule -- the LR decays and the
    total the config was built for stay where a full run would put them. That makes it a timing
    probe, not a short recipe, so the minutes/iteration it reports project the real run."""
    import json
    import time

    import torch
    from mmdet.apis import set_random_seed, train_detector
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    assert torch.cuda.is_available(), "no CUDA"

    samples_per_gpu = samples_per_gpu or BACKBONES[method][backbone]["micro_batch"]
    tag = f"{method}_{backbone}" if backbone != "r50" else method
    work_dir = f"{OUT}/{tag}_s{seed}"
    os.makedirs(work_dir, exist_ok=True)
    set_random_seed(seed, deterministic=False)

    load_from = None
    if _ckpt_key(method, backbone) in COCO_CKPT:
        load_from = _stripped_ckpt(method, backbone)
        if not os.path.exists(load_from):
            # A timing probe measures the architecture, not the weights, so it may run before
            # the download lands. A real run may not: the COCO-box init is part of the recipe.
            assert max_iters, \
                f"{load_from} missing -- run `fetch_ckpt --method {method}` first (PLAN.md §5)"
            print(f"[train] PROBE without {load_from} (timing is weight-independent)",
                  flush=True)
            load_from = None

    # Two passes: build the dataset once to learn its size, then build the final config with
    # the schedule-dependent knobs (pairwise_warmup / max_iters) scaled to it.
    kw = dict(load_from=load_from, fp16=fp16, evals_per_epoch=evals_per_epoch,
              backbone=backbone)
    probe = _build_cfg(method, work_dir, samples_per_gpu, epochs, seed, score_thr, 1, **kw)
    n_train = len(build_dataset(probe.data.train))
    cfg = _build_cfg(method, work_dir, samples_per_gpu, epochs, seed, score_thr, n_train, **kw)
    datasets = [build_dataset(cfg.data.train)]
    sched = json.loads(json.dumps(cfg._sched, default=str))
    sched["n_train"] = n_train
    print(f"[train] {method} · {n_train} train subtiles · "
          f"{json.dumps(sched, default=str)}", flush=True)

    # HARD WALL-CLOCK GUARD. The A100 budget for this phase is fixed, so the run must stop
    # even if the schedule has not, and it must stop at a point where a checkpoint was just
    # selected. The hook therefore does not kill mid-epoch: it moves `max_iters` to the NEXT
    # evaluation boundary, so EvalHook fires, `save_best` writes, and the run ends there. A run
    # that hits this is a TRUNCATED run and must be reported as one -- `sched["truncated"]`.
    from mmcv.runner import HOOKS, Hook

    @HOOKS.register_module(force=True)
    class WallClockGuard(Hook):
        def __init__(self, max_hours, boundary):
            self.limit, self.boundary, self.fired, self.t0 = max_hours * 3600, boundary, False, None

        def before_run(self, runner):
            self.t0 = time.time()

        def after_train_iter(self, runner):
            if self.fired or time.time() - self.t0 < self.limit:
                return
            stop = -(-(runner.iter + 1) // self.boundary) * self.boundary
            runner._max_iters = min(runner._max_iters, stop)
            self.fired = True
            runner.logger.warning(
                f"[WallClockGuard] {max_hours} h elapsed at iter {runner.iter}; "
                f"stopping at {runner._max_iters} (next eval boundary)")

    # EARLY STOPPING, by the same RULE the other rows we trained used, rather than by an epoch
    # count we picked: LACE early-stops on validation box AP50 (minimum 12 epochs, patience 2)
    # and the DetecTree2 row used patience 6 on val segm AP50. Reading
    # `runner.meta['hook_msgs']['best_score']` is deliberate -- EvalHook writes it ONLY on an
    # improvement, which is exactly the signal patience needs, and unlike `log_buffer.output` it
    # is not cleared by whichever logger hook happens to run first.
    @HOOKS.register_module(force=True)
    class APEarlyStop(Hook):
        def __init__(self, interval, patience, min_iters):
            self.interval, self.patience, self.min_iters = interval, patience, min_iters
            self.best, self.counter, self.history = None, 0, []

        def after_train_iter(self, runner):
            if (runner.iter + 1) % self.interval:
                return
            score = runner.meta.get("hook_msgs", {}).get("best_score")
            self.history.append(score)
            if self.best is None or (score is not None and score > self.best):
                self.best, self.counter = score, 0
            else:
                self.counter += 1
            runner.logger.info(
                f"[APEarlyStop] eval {len(self.history)} · best val bbox_mAP_50 {self.best} · "
                f"{self.counter}/{self.patience} non-improving")
            if self.counter >= self.patience and runner.iter + 1 >= self.min_iters:
                runner.logger.warning(
                    f"[APEarlyStop] stopping at iter {runner.iter + 1}: no val AP50 gain in "
                    f"{self.patience} evaluations")
                runner._max_iters = runner.iter + 1

    if patience and not max_iters:
        min_iters = (min_epochs or 0) * cfg._sched["iters_per_epoch"]
        cfg.custom_hooks = list(cfg.get("custom_hooks", [])) + [
            dict(type="APEarlyStop", interval=cfg._sched["eval_every"], patience=patience,
                 min_iters=min_iters, priority="VERY_LOW")]
        sched["early_stop"] = {"patience_evals": patience, "min_epochs": min_epochs,
                               "eval_every_iters": cfg._sched["eval_every"]}

    if max_hours and not max_iters:
        cfg.custom_hooks = list(cfg.get("custom_hooks", [])) + [
            dict(type="WallClockGuard", max_hours=max_hours,
                 boundary=cfg._sched["eval_every"], priority="VERY_LOW")]
        sched["max_hours"] = max_hours

    if max_iters:
        # Probe: cap the wall clock only. Everything the optimiser sees is unchanged.
        #
        # `--smoke` additionally forces ONE validation at the cap, so the paths that would
        # otherwise first execute 40 minutes into a real run -- EvalHook over Box2Mask's
        # instance results, `save_best` writing best_bbox_mAP_50_iter_*.pth, and APEarlyStop
        # reading it back -- are proven for ~$0.3 instead of costing a restart.
        if cfg._sched["runner"] == "iter":
            cfg.runner.max_iters = min(cfg.runner.max_iters, max_iters)
        else:
            cfg.runner = dict(type="IterBasedRunner", max_iters=max_iters)
        cfg.workflow = [("train", max_iters)]
        cfg.checkpoint_config = dict(by_epoch=False, interval=max_iters, max_keep_ckpts=1)
        cfg.log_config = dict(interval=25, hooks=[dict(type="TextLoggerHook", by_epoch=False)])
        work_dir = cfg.work_dir = f"{OUT}/probe_{tag}"
        os.makedirs(work_dir, exist_ok=True)
        if smoke:
            cfg.evaluation["interval"] = max_iters
            cfg.custom_hooks = [h for h in cfg.get("custom_hooks", [])
                                if h.get("type") != "WallClockGuard"]
            cfg.custom_hooks = cfg.custom_hooks + [
                dict(type="APEarlyStop", interval=max_iters, patience=1, min_iters=0,
                     priority="VERY_LOW")]
            print(f"[train] SMOKE: {max_iters} iterations then ONE full validation",
                  flush=True)
        else:
            cfg.evaluation = None
            print(f"[train] PROBE: {max_iters} iterations only, no validation", flush=True)

    # RESUME, not restart. `auto_resume` picks up `latest.pth` from the work dir and restores
    # weights, AdamW moment state and the iteration counter. Because the LR schedule is
    # iteration-based and derived from `max_iters`, the run lands back on exactly the right point
    # of it -- so continuing after a WallClockGuard stop is lossless, and in particular can still
    # reach the two decays at 88.9% and 96.3% of the schedule that a truncated run never sees.
    # The COCO key check below is skipped when resuming: `load_from` is irrelevant then, the
    # checkpoint being restored is our own.
    if resume:
        cfg.auto_resume = True
        cfg.load_from = None
        load_from = None
        sched["resumed_from"] = "latest.pth in work_dir"
        print(f"[train] RESUMING from latest.pth in {work_dir}", flush=True)

    model = build_detector(cfg.model)

    # FAIL FAST on the COCO init. `runner.load_checkpoint` loads with strict=False, so a
    # checkpoint that matched almost nothing would train silently from scratch and only show
    # up hours later as a bad AP. Compare the key sets here, on CPU, before any GPU time: the
    # only tolerated absence is the classifier we deliberately dropped in fetch_ckpt.
    if load_from:
        ck = torch.load(load_from, map_location="cpu")
        cksd = ck.get("state_dict", ck)
        msd = model.state_dict()
        shared = [k for k in cksd if k in msd]
        mismatched = [k for k in shared if tuple(cksd[k].shape) != tuple(msd[k].shape)]
        missing = [k for k in msd if k not in cksd]
        unexpected = [k for k in cksd if k not in msd]
        cls_key = COCO_CKPT[_ckpt_key(method, backbone)]["cls_key"]
        stray = [k for k in missing if not k.startswith(cls_key + ".")]
        assert not mismatched, f"shape mismatch on {len(mismatched)}: {mismatched[:5]}"
        assert not unexpected, f"{len(unexpected)} checkpoint keys not in model: {unexpected[:5]}"
        assert not stray, f"{len(stray)} model weights NOT initialised from COCO: {stray[:5]}"
        sched["coco_init"] = {"loaded": len(shared), "reinit_from_scratch": missing,
                              "source": os.path.basename(load_from)}
        print(f"[train] COCO init: {len(shared)}/{len(msd)} tensors loaded, "
              f"{len(missing)} reinitialised ({missing})", flush=True)

    model.init_weights()
    model.CLASSES = datasets[0].CLASSES
    cfg.dump(os.path.join(work_dir, "config.py"))

    t0 = time.time()
    train_detector(model, datasets, cfg, distributed=False, validate=bool(smoke or not max_iters),
                   timestamp=time.strftime("%Y%m%d_%H%M%S"), meta=dict(seed=seed))
    mins = (time.time() - t0) / 60
    best = [f for f in os.listdir(work_dir) if f.startswith("best_")]
    vol.commit()
    out = {"method": method, "backbone": backbone, "work_dir": work_dir, "best": best, "train_min": round(mins, 1),
           "sched": sched, "usd_at_2.10_per_h": round(mins / 60 * 2.10, 2)}
    if max_iters:
        per = mins / max_iters
        out["probe"] = {"iters_run": max_iters, "min_per_iter": round(per, 5),
                        "projected_full_h": round(per * sched["total_iters"] / 60, 2),
                        "projected_usd_at_2.10_per_h":
                            round(per * sched["total_iters"] / 60 * 2.10, 2)}
    return _report(f"train_{tag}_s{seed}" + ("_probe" if max_iters else ""), out)


@app.function(gpu="A100", image=image, volumes={"/dt2": dt2vol, "/vol": vol},
              timeout=12 * 3600, cpu=16, memory=65536)
def train(method: str = "box2mask", epochs: int = 10, samples_per_gpu: int = 0,
                seed: int = 0, score_thr: float = 0.05, max_iters: int = 0,
                max_hours: float = 0.0, fp16: bool = False, patience: int = 0,
                min_epochs: int = 0, evals_per_epoch: int = 1, smoke: bool = False,
                backbone: str = "r50", resume: bool = False):
    """R-50-scale training on a 40GB A100."""
    return _train_impl(method=method, epochs=epochs, samples_per_gpu=samples_per_gpu, seed=seed,
                       score_thr=score_thr, max_iters=max_iters, max_hours=max_hours,
                       fp16=fp16, patience=patience, min_epochs=min_epochs,
                       evals_per_epoch=evals_per_epoch, smoke=smoke, backbone=backbone,
                       resume=resume)


@app.function(gpu="A100-80GB", image=image, volumes={"/dt2": dt2vol, "/vol": vol},
              timeout=12 * 3600, cpu=16, memory=131072)
def train80(method: str = "box2mask", epochs: int = 10, samples_per_gpu: int = 0,
                seed: int = 0, score_thr: float = 0.05, max_iters: int = 0,
                max_hours: float = 0.0, fp16: bool = False, patience: int = 0,
                min_epochs: int = 0, evals_per_epoch: int = 1, smoke: bool = False,
                backbone: str = "r50", resume: bool = False):
    """Same body on an 80GB A100. Swin-L at 1024^2 does not fit in 40GB, and Modal fixes the GPU
    at decoration time, so the size has to be a separate function rather than an argument.

    The signature is spelled out rather than **kw because Modal builds its CLI flags by
    introspecting the function; **kw exposes none of them."""
    return _train_impl(method=method, epochs=epochs, samples_per_gpu=samples_per_gpu, seed=seed,
                       score_thr=score_thr, max_iters=max_iters, max_hours=max_hours,
                       fp16=fp16, patience=patience, min_epochs=min_epochs,
                       evals_per_epoch=evals_per_epoch, smoke=smoke, backbone=backbone,
                       resume=resume)


@app.function(gpu="A100", image=image, volumes={"/dt2": dt2vol, "/vol": vol},
              timeout=8 * 3600, cpu=8, memory=65536)
def predict(method: str = "box2mask", seed: int = 0, score_thr: float = 0.05,
            split: str = "test", backbone: str = "r50"):
    """Infer on every test subtile and write one Prediction_<tid>_<oy>_<ox>.json per subtile.

    RAW per-subtile output is what gets persisted -- masks at the subtile's own 1024 -- so
    stitching, dedup and scoring are all CPU re-runs afterwards and never need the GPU again.
    Runs on ALL 3,951 subtiles (439 x 9): the training set drops empty subtiles, but dropping
    any at inference would mean false positives there are never counted, silently inflating
    precision (see detectree2_baseline/build_coco.py)."""
    import glob
    import json as _json
    import time

    import numpy as np
    import torch
    from mmdet.apis import inference_detector, init_detector
    from pycocotools import mask as maskUtils
    assert torch.cuda.is_available(), "no CUDA"

    tag = f"{method}_{backbone}" if backbone != "r50" else method
    work_dir = f"{OUT}/{tag}_s{seed}"
    cand = sorted(glob.glob(os.path.join(work_dir, "best_*.pth")))
    assert cand, f"no best_*.pth in {work_dir} -- run train first"

    # SELECT BY RECORDED SCORE, NOT BY FILENAME. A resumed run can leave MORE THAN ONE
    # `best_*.pth`: mmcv's EvalHook restores `best_score` from the checkpoint's
    # `meta['hook_msgs']`, and when that does not survive the round trip the resumed run starts
    # from -inf and writes a fresh "best" that may be WORSE than the pre-resume one. It also
    # does not delete the older file, because it has no record of it. Taking `sorted(...)[-1]`
    # would then pick the latest iteration rather than the best model -- silently scoring the
    # wrong checkpoint. Observed on the Swin-L arm: iter 23422 = 0.596 (true best) alongside
    # iter 30114 = 0.590 written after the resume.
    scored = []
    for c in cand:
        try:
            m = torch.load(c, map_location="cpu").get("meta", {})
            scored.append((m.get("hook_msgs", {}).get("best_score"), c))
        except Exception as e:                       # never let selection crash the run
            print(f"[predict] WARN could not read meta from {c}: {e}", flush=True)
            scored.append((None, c))
    have = [(v, c) for v, c in scored if v is not None]
    if have:
        ckpt = max(have)[1]
    else:
        ckpt = cand[-1]
        print("[predict] WARN no best_score in any checkpoint meta; falling back to newest",
              flush=True)
    for v, c in scored:
        mark = " <- SELECTED" if c == ckpt else ""
        print(f"[predict]   candidate {os.path.basename(c)} best_score={v}{mark}", flush=True)
    model = init_detector(os.path.join(work_dir, "config.py"), ckpt, device="cuda")
    print(f"[predict] {ckpt}", flush=True)

    img_dir = f"{DATA}/{split}/images"
    files = sorted(os.listdir(img_dir))
    out_dir = f"{OUT}/pred_raw_{tag}_s{seed}_{split}"
    os.makedirs(out_dir, exist_ok=True)
    expected = 3951 if split == "test" else None
    if expected:
        assert len(files) == expected, \
            f"{len(files)} subtiles, expected {expected} (439x9) -- coverage must be complete"

    # Box2Mask emits at most `num_queries` instances per subtile -- an ARCHITECTURAL cap baked
    # into the checkpoint's learned query embeddings, not a configurable budget like Restor's
    # RPN top-k, so it cannot be raised without discarding the COCO initialisation. Count how
    # often it actually binds above the protocol score floor, so the row can be read honestly.
    cap = getattr(getattr(model, "panoptic_head", None), "num_queries", 0)
    t0, tot, saturated = time.time(), 0, 0
    for k, fn in enumerate(files):
        res = inference_detector(model, os.path.join(img_dir, fn))
        bbox_res, segm_res = res if isinstance(res, tuple) else (res, None)
        bboxes = bbox_res[0]                      # single class
        segms = segm_res[0] if segm_res is not None else []
        insts = []
        for b, m in zip(bboxes, segms):
            s = float(b[4])
            if s < score_thr:
                continue
            if not isinstance(m, dict):           # bitmap -> RLE
                m = maskUtils.encode(np.asfortranarray(np.asarray(m).astype(np.uint8)))
            counts = m["counts"]
            insts.append({"score": round(s, 5),
                          "segmentation": {"size": [int(x) for x in m["size"]],
                                           "counts": counts.decode("ascii")
                                           if isinstance(counts, bytes) else counts}})
        saturated += int(cap and len(insts) >= cap)
        tid, oy, ox = fn[:-4].rsplit("_", 2)      # tile ids contain underscores
        _json.dump(insts, open(os.path.join(out_dir, f"Prediction_{tid}_{oy}_{ox}.json"), "w"))
        tot += len(insts)
        if (k + 1) % 250 == 0 or k + 1 == len(files):
            print(f"  [predict] {k+1}/{len(files)} subtiles · {tot} instances", flush=True)
    vol.commit()
    mins = (time.time() - t0) / 60
    print(f"[predict] {len(files)} subtiles · {tot} instances in {mins:.1f} min -> {out_dir}",
          flush=True)
    return _report(f"predict_{tag}_s{seed}_{split}",
                   {"out_dir": out_dir, "n_subtiles": len(files), "n_inst": tot,
                    "ckpt_candidates": {os.path.basename(c): v for v, c in scored},
                    "predict_min": round(mins, 1), "ckpt": os.path.basename(ckpt),
                    "query_cap": cap, "subtiles_at_cap": saturated,
                    "frac_subtiles_at_cap": round(saturated / max(len(files), 1), 4)})


@app.function(image=image, volumes={"/dt2": dt2vol}, timeout=3600, cpu=4, memory=16384)
def _blackout_shard(arg):
    """One shard of the blackout. Resumable: an already-written, READABLE output is skipped, so
    a killed run is resumed rather than repeated. The readability check matters -- a run killed
    mid-write can leave a truncated PNG, which a bare exists() test would happily keep."""
    import json

    import numpy as np
    from PIL import Image
    from pycocotools import mask as maskUtils
    split, shard, n_shard = arg
    src_dir, dst_dir = f"{DATA}/{split}/images", f"{DATA}/{split}/images_canopyblack"
    os.makedirs(dst_dir, exist_ok=True)
    coco = json.load(open(f"{DATA}/{split}/coco.json"))
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    imgs = [im for i, im in enumerate(sorted(coco["images"], key=lambda x: x["id"]))
            if i % n_shard == shard]

    n_black = n_tot = n_crown_px = n_crown_lost = n_skip = 0
    for k, im in enumerate(imgs):
        out_fp = os.path.join(dst_dir, im["file_name"])
        if os.path.exists(out_fp):
            try:
                Image.open(out_fp).verify()
                n_skip += 1
                continue
            except Exception:
                pass                      # truncated from a killed run -- redo it
        anns = by_img.get(im["id"], [])
        h, w = im["height"], im["width"]

        def _union(sel):
            ms = [maskUtils.decode(
                    {"size": a["segmentation"]["size"],
                     "counts": a["segmentation"]["counts"].encode("ascii")
                     if isinstance(a["segmentation"]["counts"], str)
                     else a["segmentation"]["counts"]}).astype(bool)
                  for a in anns if sel(a)]
            return np.logical_or.reduce(ms) if ms else np.zeros((h, w), bool)

        canopy = _union(lambda a: a.get("iscrowd", 0) == 1)
        crown = _union(lambda a: a.get("iscrowd", 0) == 0)
        black = canopy & ~crown
        assert not (black & crown).any(), f"{im['file_name']}: blackout hits a crown"
        arr = np.asarray(Image.open(os.path.join(src_dir, im["file_name"]))
                         .convert("RGB")).copy()
        arr[black] = 0
        Image.fromarray(arr).save(out_fp, compress_level=1)   # speed over size
        n_black += int(black.sum()); n_tot += h * w
        n_crown_px += int(crown.sum()); n_crown_lost += int((crown & black).sum())
        if (k + 1) % 200 == 0:
            print(f"  [{split} shard {shard}] {k+1}/{len(imgs)}", flush=True)
    return {"split": split, "n": len(imgs), "skipped": n_skip, "black_px": n_black,
            "tot_px": n_tot, "crown_px": n_crown_px, "crown_lost": n_crown_lost}


@app.function(image=image, volumes={"/dt2": dt2vol}, timeout=3600, cpu=4, memory=16384)
def prep_data_blackout(split_list: str = "train,val", n_shard: int = 16):
    """Canopy-blackout training crops, so canopy is not trained as BACKGROUND.

    WHY THIS IS NEEDED. LACE treats canopy as an ignore label and detectron2 honours `iscrowd`
    for the DetecTree2 row, but BoxInst can do neither: `CondInstBoxHead.loss` accepts
    `gt_bboxes_ignore` and never uses it, and the train pipeline's `Collect` does not even pass
    it. Left alone, canopy -- 19% of annotations, ~25% of tile area, and full of REAL but
    undelineated crowns -- becomes an ordinary negative, teaching the model to suppress
    detections exactly where the scorer later makes them free. That would make the row measure
    our data handling rather than the method.

    The remedy is the one the tree-crown literature already uses on this dataset: CanopyRS /
    SelvaBox delete canopy annotations and black out those pixels. Each of the three arms then
    gets canopy-neutral TRAINING by whatever mechanism its framework allows -- ignore label
    (LACE), `iscrowd` (DetecTree2 / detectron2), blackout (BoxInst).

    ONE DEPARTURE FROM THEIR RECIPE, AND IT IS NECESSARY. Canopy and crown polygons are NOT
    disjoint in OAM-TCD: 6.36% of crown area falls inside a canopy polygon, and a naive blackout
    erases 3.4% of crowns outright and damages 9.5% of them. So the blackout mask is
    `canopy AND NOT crown` -- every labelled crown pixel survives, asserted below. A crown
    sitting inside canopy is left as an island of real imagery in a blacked field, which is
    exactly what it is.

    TEST IMAGES ARE NEVER TOUCHED. The 439-tile test set is the frozen protocol and must be
    byte-identical across rows; canopy is handled there by the scorer's `iscrowd` rule instead.
    """
    import json
    import sys
    sys.path.insert(0, "/root")
    import make_box_only_coco as B

    splits = [x.strip() for x in split_list.split(",") if x.strip()]
    jobs = [(sp, i, n_shard) for sp in splits for i in range(n_shard)]
    agg = {}
    for r in _blackout_shard.map(jobs):
        a = agg.setdefault(r["split"], {k: 0 for k in
                                        ("n", "skipped", "black_px", "tot_px",
                                         "crown_px", "crown_lost")})
        for k in a:
            a[k] += r[k]

    report = {}
    for split in splits:
        coco = json.load(open(f"{DATA}/{split}/coco.json"))
        keep = [a for a in coco["annotations"] if a.get("iscrowd", 0) == 0]
        d = B.convert({"images": coco["images"], "categories": coco["categories"],
                       "annotations": keep})
        stats = B.verify(d)
        d["info"]["canopy"] = ("annotations removed; their pixels blacked out in "
                               "images_canopyblack/, except where a labelled crown overlaps")
        json.dump(d, open(f"{DATA}/{split}/coco_boxonly_black.json", "w"))
        assert stats["n_ignore_canopy"] == 0, "canopy annotation survived the crown-only filter"
        a = agg[split]
        assert a["crown_lost"] == 0, f"{split}: blackout destroyed {a['crown_lost']} crown px"
        report[split] = {**stats, "subtiles": a["n"], "reused": a["skipped"],
                         "pct_pixels_blacked": round(100 * a["black_px"] / a["tot_px"], 2)}
        print(f"[blackout] {split}: {report[split]}", flush=True)
    dt2vol.commit()
    return report


@app.function(image=image, volumes={"/dt2": dt2vol, "/vol": vol}, timeout=1800,
              cpu=8, memory=32768)
def dryrun(method: str = "box2mask", epochs: int = 12, samples_per_gpu: int = 0,
           seed: int = 0, backbone: str = "r50"):
    """Build the config, BOTH datasets and the model against the real paths -- no training.

    Same principle as `imports_ok`: a wrong ann_file, a class-name mismatch or an empty split
    costs pennies here and an A100-hour if it surfaces inside train_detector. It also pulls one
    real training sample THROUGH the pipeline, which is the only way to prove that the tensor
    reaching the loss is what we think it is."""
    import json

    import numpy as np
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector

    samples_per_gpu = samples_per_gpu or BACKBONES[method][backbone]["micro_batch"]
    kw = dict(backbone=backbone)
    probe = _build_cfg(method, "/tmp/dry", samples_per_gpu, epochs, seed, 0.05, 1, **kw)
    n_train = len(build_dataset(probe.data.train))
    cfg = _build_cfg(method, "/tmp/dry", samples_per_gpu, epochs, seed, 0.05, n_train, **kw)
    tr, va = build_dataset(cfg.data.train), build_dataset(cfg.data.val)

    n_tr = [len(tr.get_ann_info(i)["bboxes"]) for i in range(len(tr))]
    n_va = [len(va.get_ann_info(i)["bboxes"]) for i in range(len(va))]
    ign_va = sum(len(va.get_ann_info(i)["bboxes_ignore"]) for i in range(len(va)))
    ign_tr = sum(len(tr.get_ann_info(i)["bboxes_ignore"]) for i in range(len(tr)))

    info = {
        "method": method, "backbone": backbone,
        "train": {"imgs": len(tr), "boxes": int(np.sum(n_tr)), "ignore": ign_tr,
                  "max_boxes_per_subtile": int(np.max(n_tr)),
                  "empty_subtiles": int(np.sum(np.asarray(n_tr) == 0)),
                  "src": cfg.data.train["img_prefix"]},
        "val": {"imgs": len(va), "boxes": int(np.sum(n_va)), "ignore": ign_va,
                "max_boxes_per_subtile": int(np.max(n_va)),
                "src": cfg.data.val["img_prefix"]},
        "classes": tr.CLASSES,
        # json round-trip: mmcv ConfigDict does not unpickle locally, where there is no mmcv
        "schedule": json.loads(json.dumps(cfg._sched, default=str)),
        "selection": json.loads(json.dumps(cfg.evaluation, default=str)),
    }
    # The whole point of the blackout: canopy must be gone from TRAIN and present as ignore in
    # VAL. If these flip, the arm silently measures the wrong thing.
    assert ign_tr == 0, f"train still carries {ign_tr} ignore (canopy) boxes"
    assert ign_va > 0, "val lost its canopy ignore boxes -- selection would punish canopy dets"
    assert tr.CLASSES == ("tree",), tr.CLASSES

    # Pull ONE real sample through the real pipeline. `make_box_only_coco.verify()` proves the
    # FILE is box-only; this proves the TENSOR is. For Box2Mask that matters more than for
    # BoxInst, because Box2Mask does have a `gt_masks` key -- it is just painted from the boxes
    # by `GenerateBoxMask`, so every mask must be exactly its own bounding rectangle.
    sample = tr[int(np.argmax(n_tr))]
    info["sample_keys"] = sorted(sample.keys())
    if "gt_masks" in sample:
        masks = sample["gt_masks"].data.masks
        boxes = sample["gt_bboxes"].data.numpy()
        bad, checked = 0, 0
        for m, b in zip(masks, boxes):
            ys, xs = np.nonzero(m)
            if not len(xs):
                continue
            checked += 1
            # GenerateBoxMask paints round(x0)..round(x1) inclusive, so allow a 1 px slack.
            if (abs(xs.min() - b[0]) > 2.0 or abs(xs.max() - b[2]) > 2.0
                    or abs(ys.min() - b[1]) > 2.0 or abs(ys.max() - b[3]) > 2.0
                    or m.sum() != (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)):
                bad += 1
        assert checked and not bad, \
            f"{bad}/{checked} gt_masks are not their own box -- supervision is NOT box-only"
        info["box_only_tensor_check"] = {"masks_checked": checked, "non_rectangular": bad}

    model = build_detector(cfg.model)
    info["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)

    # Box2Mask emits at most `num_queries` instances per subtile, a hard ceiling BoxInst does
    # not have. If a subtile carries more crowns than that, the method cannot express them and
    # the shortfall must be reported, not discovered later in the AP.
    if cfg._sched.get("num_queries"):
        q = cfg._sched["num_queries"]
        info["query_ceiling"] = {
            "num_queries": q, "max_per_image": cfg._sched["max_per_image"],
            "train_subtiles_over_ceiling": int(np.sum(np.asarray(n_tr) > q)),
            "val_subtiles_over_ceiling": int(np.sum(np.asarray(n_va) > q)),
            "train_boxes_beyond_ceiling":
                int(np.sum(np.clip(np.asarray(n_tr) - q, 0, None)))}
    print(json.dumps(info, indent=2, default=str), flush=True)
    return info


@app.function(image=image, volumes={"/dt2": dt2vol}, timeout=1800, cpu=4, memory=16384)
def inspect_empty_subtiles(split: str = "train"):
    """How much real imagery survives in the subtiles the blackout emptied?

    Removing canopy annotations leaves some subtiles with no annotations at all, which mmdet's
    `filter_empty_gt=True` then drops. Whether that is a loss or a mercy depends on how much of
    those subtiles is still real pixels rather than blacked canopy -- measured here from the
    COCO masks alone, no images needed."""
    import json

    import numpy as np
    from pycocotools import mask as maskUtils

    coco = json.load(open(f"{DATA}/{split}/coco.json"))
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    fracs = []
    for im in coco["images"]:
        anns = by_img.get(im["id"], [])
        if any(a.get("iscrowd", 0) == 0 for a in anns):
            continue                                   # has a crown -> survives filtering
        h, w = im["height"], im["width"]
        ms = [maskUtils.decode({"size": a["segmentation"]["size"],
                                "counts": a["segmentation"]["counts"].encode("ascii")
                                if isinstance(a["segmentation"]["counts"], str)
                                else a["segmentation"]["counts"]}).astype(bool)
              for a in anns if a.get("iscrowd", 0) == 1]
        u = np.logical_or.reduce(ms) if ms else np.zeros((h, w), bool)
        fracs.append(float(u.sum()) / (h * w))

    f = np.asarray(fracs)
    out = {"split": split, "n_emptied_subtiles": int(len(f)),
           "blacked_fraction": {"mean": round(float(f.mean()), 4),
                                "median": round(float(np.median(f)), 4),
                                "p10": round(float(np.percentile(f, 10)), 4),
                                "p90": round(float(np.percentile(f, 90)), 4)},
           "n_over_90pct_black": int((f > 0.9).sum()),
           "n_under_50pct_black": int((f < 0.5).sum())}
    print(json.dumps(out, indent=2), flush=True)
    return out


# cpu=1: `stitch_all` is a SERIAL for-loop over tiles and the work inside it (pycocotools RLE
# decode, PIL rasterisation on 512x512 masks) is single-threaded, so requesting 8 cores billed
# ~7 idle ones for the whole run -- measured 6.0 s/tile for Swin-L, ~44 min, i.e. ~8x the cost
# it needed. Memory likewise: the loop holds one tile's RLEs at a time, not all 439.
# FURTHER WIN NOT TAKEN: tiles are independent, so sharding via .map() (the pattern
# `_blackout_shard` already uses, n_shard=16) would cut wall clock ~16x at the same core-hours.
# Not implemented because the serial path is proven and we have no further Box2Mask arm to
# stitch; do it if this is ever run at scale again.
@app.function(image=image, volumes={"/vol": vol}, timeout=4 * 3600, cpu=1, memory=8192)
def stitch(method: str = "box2mask", seed: int = 0, split: str = "test",
           min_score: float = 0.05, backbone: str = "r50"):
    """Stitch the 9 subtile predictions per tile back to 2048 and emit our preds schema.

    `min_score` DEFAULTS TO THE PROTOCOL FLOOR (0.05), NOT stitch.py's 0.1. That default was
    chosen for DetecTree2, whose own recipe floors at 0.2; applying it here would impose a
    stricter floor on BoxInst than `score_coco.py` applies to every other row and would
    understate it. The floor must be the protocol's, once, at scoring.

    Also reports detections/tile and the canopy-landing rate, because the blackout creates a
    specific risk: BoxInst trains with canopy blacked but is TESTED on real imagery, so like
    SelvaBox it will fire freely in canopy. maxDets=600 is applied BEFORE canopy-ignore, so if
    most detections land in canopy the budget is spent on predictions the scorer then discards.
    PROTOCOL_439.md measured exactly this for SelvaBox (it cost 0.008 AP50); this is the number
    that says whether a maxDets sensitivity is needed here too."""
    import json
    import sys

    import numpy as np
    sys.path.insert(0, "/root")
    import stitch as ST
    from pycocotools import mask as maskUtils

    tag = f"{method}_{backbone}" if backbone != "r50" else method
    pred_dir = f"{OUT}/pred_raw_{tag}_s{seed}_{split}"
    assert os.path.isdir(pred_dir), f"{pred_dir} missing -- run predict first"
    gt = json.load(open("/root/test_gt.json"))
    tids = sorted(gt)
    out = ST.stitch_all(pred_dir, tids, min_score=min_score)
    out["meta"]["model"] = (f"{MODEL_DESC[method]}, box-only, canopy-blacked training, "
                            "1024@50% subtiles, clean_crowns dedup")
    out["meta"]["min_score"] = min_score

    # canopy-landing diagnostics
    n_det, n_can, per_tile = 0, 0, []
    for t in tids:
        rec = out["preds"].get(t)
        if not rec or not rec["masks_rle"]:
            per_tile.append(0)
            continue
        rles = [{"size": r["size"], "counts": r["counts"].encode("ascii")}
                for r in rec["masks_rle"]]
        can = ST.raster_canopy(gt[t]) if hasattr(ST, "raster_canopy") else None
        if can is None:
            from PIL import Image, ImageDraw
            img = Image.new("L", (512, 512), 0)
            for poly in gt[t]["canopy"]:
                pts = (np.asarray(poly, np.float32).reshape(-1, 2) / 4.0)
                ImageDraw.Draw(img).polygon([tuple(v) for v in pts], fill=1)
            can = np.asarray(img, bool)
        cr = maskUtils.encode(np.asfortranarray(can.astype(np.uint8)))
        for r in rles:
            a = float(maskUtils.area(r))
            inter = float(maskUtils.area(maskUtils.merge([r, cr], intersect=1)))
            n_can += int(a > 0 and inter / a > 0.5)
        n_det += len(rles)
        per_tile.append(len(rles))

    pt = np.asarray(per_tile)
    out["meta"]["diagnostics"] = {
        "dets_total": int(n_det), "dets_per_tile_mean": round(float(pt.mean()), 1),
        "dets_per_tile_max": int(pt.max()),
        "tiles_over_maxdets_600": int((pt > 600).sum()),
        "canopy_landing_rate": round(n_can / max(n_det, 1), 4)}
    fp = f"{OUT}/preds_{tag}_s{seed}_{split}.json"
    json.dump(out, open(fp, "w"))
    vol.commit()
    print(f"-> {fp}", flush=True)
    return _report(f"stitch_{tag}_s{seed}_{split}",
                   dict(out["meta"]["diagnostics"], preds=fp, **out["meta"]["dedup"]))


@app.function(image=image, gpu="A100", volumes={"/dt2": dt2vol, "/vol": vol},
              timeout=3600, cpu=8, memory=32768)
def injection_gate(seed: int = 0, n_tiles: int = 12, trained: bool = False):
    """THE GATE for putting BoxInst in the box-to-mask module table.

    Injects the model's own detections and checks the resulting masks reproduce its native
    ones. Runs on an UNTRAINED model by default: the round trip tests the code path, not the
    weights, so this tells us whether the module row is reachable BEFORE paying for training.
    Pass --trained to re-check with the real checkpoint afterwards."""
    import glob
    import json
    import sys

    import numpy as np
    import torch
    sys.path.insert(0, "/root")
    import inject_boxes as IB
    from mmdet.apis import init_detector
    from mmdet.datasets import build_dataset
    from mmdet.datasets.pipelines import Compose

    if trained:
        wd = f"{OUT}/boxinst_s{seed}"
        ck = sorted(glob.glob(os.path.join(wd, "best_*.pth")))
        assert ck, f"no checkpoint in {wd}"
        model = init_detector(os.path.join(wd, "config.py"), ck[-1], device="cuda")
        tag = f"trained:{os.path.basename(ck[-1])}"
    else:
        from mmdet.models import build_detector
        cfg = _build_cfg("boxinst", "/tmp/gate", 4, 12, seed, 0.05, 1)
        # An untrained head suppresses nothing, so the default 2000 dets/img makes the mask
        # head's interpolate allocate ~8 GiB. Irrelevant to the assigner check; cap it.
        cfg.model.test_cfg.max_per_img = 100
        cfg.model.test_cfg.nms_pre = 500
        model = build_detector(cfg.model).cuda().eval()
        model.CLASSES = ("tree",)
        model.cfg = cfg
        tag = "UNTRAINED (ImageNet backbone only)"
    print(f"[gate] model: {tag}", flush=True)

    cfg = model.cfg
    pipe = Compose(cfg.data.val["pipeline"])
    # GT boxes per val image, for the assignment diagnostic
    vc = json.load(open(f"{DATA}/val/coco_boxonly.json"))
    fn2id = {im["file_name"]: im["id"] for im in vc["images"]}
    gt_boxes = {}
    for a in vc["annotations"]:
        if a.get("iscrowd", 0):
            continue
        x, y, w, h = a["bbox"]
        gt_boxes.setdefault(a["image_id"], []).append([x, y, x + w, y + h])
    gt_boxes = {f: gt_boxes.get(i, []) for f, i in fn2id.items()}

    files = sorted(os.listdir(f"{DATA}/val/images"))[:n_tiles]
    rows, diag, gt_assigned = [], [], []
    assigned_sizes, unassigned_sizes = [], []
    for fn in files:
        d = pipe(dict(img_info=dict(filename=fn), img_prefix=f"{DATA}/val/images",
                      img_fields=[], bbox_fields=[], mask_fields=[], seg_fields=[]))
        img = d["img"][0].unsqueeze(0).cuda()
        metas = [d["img_metas"][0].data]
        # An untrained head is near-chance, so drop the floor to find any detections at all.
        r = IB.round_trip(model, img, metas, score_thr=0.0 if not trained else 0.05)
        if r:
            rows.append(r)
        # The assigner is a pure function of box geometry, so this needs no inference:
        # feed real GT boxes and record which sizes it accepts. That is what answers "why not
        # 100%?" -- and unlike the untrained detector's own boxes, GT geometry is valid.
        gtb = gt_boxes.get(fn)
        if gtb is not None and len(gtb):
            t = torch.as_tensor(gtb, dtype=torch.float32).cuda()
            feats = model.extract_feat(img)
            cs, _, _, _ = model.bbox_head.forward(feats, model.mask_head.param_conv)
            got = IB._assign(model.bbox_head, [c.size()[-2:] for c in cs], t,
                             t.device, t.dtype)
            nk = 0 if got is None else len(got[3])
            gt_assigned.append((nk, len(gtb)))
            # WHICH boxes fail? Record the size of every box, flagged by whether the assigner
            # gave it a location, so "why not 100%?" is answered with the distribution rather
            # than a guess.
            kept_set = set() if got is None else set(got[3].cpu().tolist())
            wh = (t[:, 2] - t[:, 0]).cpu().numpy(), (t[:, 3] - t[:, 1]).cpu().numpy()
            side = np.sqrt(np.maximum(wh[0], 1e-6) * np.maximum(wh[1], 1e-6))
            for i, sd in enumerate(side):
                (assigned_sizes if i in kept_set else unassigned_sizes).append(float(sd))
    n = sum(r["n"] for r in rows)
    med = float(np.median([r["median_iou"] for r in rows if r["median_iou"] is not None])) \
        if rows else 0.0
    frac = float(np.mean([r["frac_ge_0.95"] for r in rows if r.get("frac_ge_0.95") is not None])) \
        if rows else 0.0
    una = sum(r["unassigned"] for r in rows)
    d = np.asarray(diag) if diag else np.zeros((0, 3))
    ga = np.asarray(gt_assigned) if gt_assigned else np.zeros((0, 2))
    out = {"model": tag, "tiles": len(rows), "instances": n,
           "gt_box_assignment_rate": round(float(ga[:, 0].sum() / max(ga[:, 1].sum(), 1)), 4)
           if len(ga) else None,
           "assigned_box_side_px": {
               "n": len(assigned_sizes),
               "p1": round(float(np.percentile(assigned_sizes, 1)), 2),
               "median": round(float(np.median(assigned_sizes)), 2)}
           if assigned_sizes else None,
           "UNassigned_box_side_px": {
               "n": len(unassigned_sizes),
               "min": round(float(np.min(unassigned_sizes)), 2),
               "median": round(float(np.median(unassigned_sizes)), 2),
               "max": round(float(np.max(unassigned_sizes)), 2)}
           if unassigned_sizes else None,
           "median_iou_native_vs_injected": round(med, 4),
           "frac_ge_0.95": round(frac, 4), "unassigned_boxes": una,
           "VERDICT": "PASS -- injection reproduces native masks; module row is reachable"
                      if med >= 0.95 else
                      "INCONCLUSIVE -- the assigner works on real GT boxes but the untrained "
                      "detector emits degenerate boxes, so the round trip cannot run yet; "
                      "re-run with --trained"
                      if (len(ga) and ga[:, 0].sum() / max(ga[:, 1].sum(), 1) > 0.8) else
                      "FAIL -- injected masks differ from native; BoxInst belongs in the "
                      "system and strata tables only"}
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.function(image=image, volumes={"/dt2": dt2vol}, timeout=3600, cpu=8, memory=32768)
def assigner_coverage(split: str = "val"):
    """For what fraction of ALL GT boxes is the injection path even DEFINED?

    The assigner is a pure function of box geometry and feature-map sizes -- no forward pass,
    no weights, no GPU -- so this runs over the whole split instead of the 12-subtile sample
    `injection_gate` uses. Coverage is a NECESSARY condition for putting BoxInst in the
    box-to-mask table (a box with no responsible location has no mask under this method); it is
    not sufficient, which is what the round trip tests."""
    import json
    import sys

    import numpy as np
    import torch
    sys.path.insert(0, "/root")
    import inject_boxes as IB
    from mmdet.models import build_detector

    cfg = _build_cfg("boxinst", "/tmp/cov", 4, 12, 0, 0.05, 1)
    head = build_detector(cfg.model).bbox_head.eval()

    coco = json.load(open(f"{DATA}/{split}/coco_boxonly.json"))
    dims = {im["id"]: (im["height"], im["width"]) for im in coco["images"]}
    boxes = {}
    for a in coco["annotations"]:
        if a.get("iscrowd", 0):
            continue
        x, y, w, h = a["bbox"]
        boxes.setdefault(a["image_id"], []).append([x, y, x + w, y + h])

    ok_sz, bad_sz = [], []
    for iid, bl in boxes.items():
        h, w = dims[iid]
        fms = [(int(np.ceil(h / s)), int(np.ceil(w / s))) for s in head.strides]
        t = torch.as_tensor(bl, dtype=torch.float32)
        got = IB._assign(head, fms, t, t.device, t.dtype)
        kept = set() if got is None else set(got[3].tolist())
        side = np.sqrt(np.maximum(t[:, 2] - t[:, 0], 1e-6)
                       * np.maximum(t[:, 3] - t[:, 1], 1e-6)).numpy()
        for i, sd in enumerate(side):
            (ok_sz if i in kept else bad_sz).append(float(sd))

    ok, bad = np.asarray(ok_sz), np.asarray(bad_sz)
    n = len(ok) + len(bad)
    out = {"split": split, "n_boxes": int(n), "assigned": int(len(ok)),
           "unassigned": int(len(bad)),
           "coverage": round(float(len(ok) / max(n, 1)), 5),
           "assigned_side_px": {"p1": round(float(np.percentile(ok, 1)), 2),
                                "median": round(float(np.median(ok)), 2)} if len(ok) else None,
           "unassigned_side_px": {"min": round(float(bad.min()), 2),
                                  "median": round(float(np.median(bad)), 2),
                                  "max": round(float(bad.max()), 2)} if len(bad) else None,
           "finest_stride": int(head.strides[0])}
    if len(bad):
        for thr in (8, 10, 12, 16):
            out[f"unassigned_frac_of_boxes_under_{thr}px"] = round(
                float((bad < thr).sum() / max((ok < thr).sum() + (bad < thr).sum(), 1)), 4)
    print(json.dumps(out, indent=2), flush=True)
    return out
