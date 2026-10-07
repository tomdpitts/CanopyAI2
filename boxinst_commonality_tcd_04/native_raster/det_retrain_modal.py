"""Modal A100: detector retrain pilot (det_retrain_v2), TRAIN ONLY.

Recipe and decisions: NATIVE_RASTER.md "DETECTOR RETRAIN". This function never reads test
features and never calls phase4_lib_tcd.eval_4p (which scores the 439 test tiles): the
adopt/stop decision is taken on val only, before any test number exists for the new detector.

Reads the existing 4-phase L24 cache on tcd04-phase4-vol (/vol/feat_4p_train: 792 train + 108
val tiles). Writes ONLY under /vol/out/det_retrain/{tag}/: det_{tag}.pt (best on val box
AP50:95), state_{tag}.pt (full training state, every epoch), run.log. Resumes from the state
file on restart; Modal retries the call on preemption.

    python detach.py deploy det_retrain_modal.py
    python detach.py spawn tcd04-det-retrain train --seed 0 --note "pilot v2 s0"
    modal volume get tcd04-phase4-vol out/det_retrain/v2_L24_s0/run.log -   # progress
"""
import os

import modal

APP_NAME = "tcd04-det-retrain"
VOL_NAME = "tcd04-phase4-vol"

HERE = os.path.dirname(os.path.abspath(__file__))     # .../native_raster
PKG = os.path.dirname(HERE)                            # boxinst_commonality_tcd_04
REPO = os.path.dirname(PKG)
PH4 = os.path.join(PKG, "modal_tcd_multiseed", "phase4")

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME)

P = "/root/proj"
PKG_R = f"{P}/boxinst_commonality_tcd_04"
image = (                                              # pins identical to phase4_modal.py
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.1", "torchvision==0.27.1", "numpy==2.2.6",
                 "transformers==4.57.1", "datasets==4.0.0", "pillow", "contourpy",
                 "pycocotools")
)
for rel in ("dapt/__init__.py", "dapt/backbone.py", "dapt/targets.py",
            "dapt/decode.py", "dapt/eval.py", "dapt/head.py"):
    image = image.add_local_file(os.path.join(REPO, rel), f"{P}/{rel}")
for rel in ("__init__.py", "detector.py", "train_detector_tiles.py", "evaluate.py",
            "cache_train_tiles.py", "cache_test.py", "prepare_test.py",
            "train_tiles_gt.json"):
    image = image.add_local_file(os.path.join(PKG, rel), f"{PKG_R}/{rel}")
image = image.add_local_file(os.path.join(PKG, "modal_tcd_multiseed", "__init__.py"),
                             f"{PKG_R}/modal_tcd_multiseed/__init__.py")
for rel in ("__init__.py", "phase4_lib_tcd.py"):
    image = image.add_local_file(os.path.join(PH4, rel),
                                 f"{PKG_R}/modal_tcd_multiseed/phase4/{rel}")
image = image.add_local_file(os.path.join(HERE, "det_retrain_v2.py"),
                             f"{PKG_R}/native_raster/det_retrain_v2.py")
STUBS = os.path.join(PH4, "stubs")                     # imports evaluate.py pulls in
for pkg, files in {"boxinst": ("__init__.py", "cache_feats.py"),
                   "boxinst_commonality": ("__init__.py", "em.py"),
                   "boxinst_tcd": ("__init__.py", "build_canopy.py", "cache.py",
                                   "prepare.py")}.items():
    for f in files:
        image = image.add_local_file(os.path.join(STUBS, pkg, f), f"{P}/{pkg}/{f}")
for rel in ("em.py",):
    image = image.add_local_file(os.path.join(PKG, rel), f"{PKG_R}/{rel}")

FEAT_4P_TRAIN = "/vol/feat_4p_train"


@app.function(gpu="A100", image=image, volumes={"/vol": vol}, timeout=8 * 3600,
              cpu=8, memory=65536, retries=modal.Retries(max_retries=3, initial_delay=30.0))
def train(seed: int = 0, tag: str = "", epochs: int = 60, w_size: float = 0.5,
          w_giou: float = 2.0, es_metric: str = "mAP50_95", min_epochs: int = 20,
          es_patience: int = 3, es_min_delta: float = 0.002):
    import argparse
    import json
    import sys
    import time

    import torch
    if P not in sys.path:
        sys.path.insert(0, P)
    assert torch.cuda.is_available(), "no CUDA"
    import boxinst_commonality_tcd_04.train_detector_tiles as T
    from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.phase4_lib_tcd import \
        Detector4Phase
    from boxinst_commonality_tcd_04.native_raster.det_retrain_v2 import train_v2

    tag = tag or f"v2_L24_s{seed}"
    out = f"/vol/out/det_retrain/{tag}"
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "run.log"), "a")

    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    def persist(_path):
        logf.flush(); vol.commit()

    n = len(os.listdir(FEAT_4P_TRAIN))
    assert n == 900, f"feature cache incomplete: {n}/900"
    T.Detector8 = Detector4Phase                      # exactly as phase4_lib_tcd.train_4p
    T.cache_dir = lambda arm: FEAT_4P_TRAIN
    args = argparse.Namespace(
        tag=tag, epochs=epochs, seed=seed, lr=1e-3, wd=1e-4, bs=3, width=256, tower=3,
        eval_every=5, arm="web", device="cuda", canvas=2048, gt_path=None,
        min_epochs=min_epochs, es_patience=es_patience, es_min_delta=es_min_delta,
        w_size=w_size, w_giou=w_giou, es_metric=es_metric)
    log(f"[det_retrain] gpu={torch.cuda.get_device_name(0)} args={vars(args)}")
    t0 = time.time()
    best = train_v2(args, out, persist=persist, log=log)
    mins = (time.time() - t0) / 60
    summ = {"tag": tag, "best_epoch": best["epoch"], "val": best["rep"],
            "es_metric": es_metric, "minutes_this_call": round(mins, 1),
            "est_cost_usd_this_call": round(mins * 0.035, 2)}
    json.dump(summ, open(os.path.join(out, f"summary_{tag}.json"), "w"), indent=2)
    log(f"[det_retrain] DONE {json.dumps(summ)}")
    persist(None)
    return summ
