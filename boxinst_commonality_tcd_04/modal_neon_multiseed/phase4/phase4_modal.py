"""Modal H100 app for the 4-phase ("center-registration") NEON seed-0 experiment.

ISOLATED: reuses the existing `neon-multiseed-vol` Volume for INPUTS ONLY (RGB tiles +
GT jsons already uploaded by the native pipeline) and writes every output under
`/vol/phase4/**`. Deleting this folder + `/vol/phase4` is a complete, risk-free tidy-up;
nothing in the native pipeline or its defaults is touched.

Stages (H100):
    modal run phase4/phase4_modal.py::extract_4p      # 4-phase feats -> /vol/phase4/feat_*
    modal run phase4/phase4_modal.py::train_eval_4p   # seed-0 train + eval -> preds json
Score post-hoc, locally, apples-to-apples vs native seed-0:  python phase4/phase4_score.py
"""
import json
import os

import modal

APP = "neon-phase4"
VOL_NAME = "neon-multiseed-vol"          # SAME volume; we only add a /vol/phase4 subtree
GPU = "H100"

HERE = os.path.dirname(os.path.abspath(__file__))          # .../modal_neon_multiseed/phase4
NEON = os.path.dirname(HERE)                               # .../modal_neon_multiseed
PKG = os.path.dirname(NEON)                                # boxinst_commonality_tcd_04
REPO = os.path.dirname(PKG)
P = "/root/proj"
PKG_R = f"{P}/boxinst_commonality_tcd_04"
NEON_R = f"{PKG_R}/modal_neon_multiseed"
PH4_R = f"{NEON_R}/phase4"

app = modal.App(APP)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.1", "torchvision==0.27.1", "numpy==2.2.6",
                 "transformers==4.57.1", "pillow")
    .env({"HF_HOME": "/vol/hf_cache", "HF_HUB_OFFLINE": "0"})
)
for rel in ("dapt/__init__.py", "dapt/backbone.py", "dapt/targets.py",
            "dapt/decode.py", "dapt/eval.py", "dapt/head.py"):
    image = image.add_local_file(os.path.join(REPO, rel), f"{P}/{rel}")
for rel in ("__init__.py", "detector.py"):
    image = image.add_local_file(os.path.join(PKG, rel), f"{PKG_R}/{rel}")
for rel in ("__init__.py", "neon_features.py", "neon_train_lib.py"):
    image = image.add_local_file(os.path.join(NEON, rel), f"{NEON_R}/{rel}")
for rel in ("__init__.py", "phase4_features.py", "phase4_lib.py"):
    image = image.add_local_file(os.path.join(HERE, rel), f"{PH4_R}/{rel}")

VOL = "/vol"
TRAIN_RGB = f"{VOL}/train_patches"          # *.png (native inputs, read-only)
EVAL_RGB = f"{VOL}/eval_rgb"                # *.tif (native inputs, read-only)
PH4 = f"{VOL}/phase4"                       # everything we write lives here
FEAT_TRAIN = f"{PH4}/feat_train"
FEAT_EVAL = f"{PH4}/feat_eval"
OUT = f"{PH4}/out"
PARITY_TILE = "2018_SJER_3_252000_4104000_image_628"


def _setup_path():
    import sys
    for p in (P, PKG_R):
        if p not in sys.path:
            sys.path.insert(0, p)


@app.function(gpu=GPU, image=image, volumes={VOL: vol}, timeout=6 * 3600,
              cpu=8, memory=32768, secrets=[hf_secret])
def extract_4p():
    """4-phase real-8px features for the 2063 train patches + 194 eval tiles.
    Runs the abort-fast registration self-test on 1 real tile FIRST (seconds); a bad
    interleave dies before the expensive loop. Idempotent per-tile; resumable."""
    import glob
    import time
    import torch
    _setup_path()
    assert torch.cuda.is_available(), "no CUDA"
    from boxinst_commonality_tcd_04.modal_neon_multiseed import neon_features as nf
    from boxinst_commonality_tcd_04.modal_neon_multiseed.phase4 import phase4_features as p4
    net = nf.build_net(device="cuda")     # layers=(21,22,23,24) trap-guarded in build_net
    print(f"[extract_4p] gpu={torch.cuda.get_device_name(0)} out_dim={net.out_dim} "
          f"layers={net.layers}", flush=True)
    # ---- abort-fast registration guard (real DINO) BEFORE the full extract ----
    ptile = os.path.join(EVAL_RGB, PARITY_TILE + ".tif")
    if os.path.exists(ptile):
        p4.registration_self_test(net, ptile)
    else:
        print(f"[extract_4p] WARN parity tile missing ({ptile}); skipping reg-test",
              flush=True)
    os.makedirs(PH4, exist_ok=True)
    t0 = time.time()
    train_imgs = sorted(glob.glob(os.path.join(TRAIN_RGB, "*.png")))
    eval_imgs = sorted(glob.glob(os.path.join(EVAL_RGB, "*.tif")))
    print(f"[extract_4p] train patches={len(train_imgs)} eval tiles={len(eval_imgs)}",
          flush=True)
    assert train_imgs and eval_imgs, (
        f"empty input dirs: {TRAIN_RGB} / {EVAL_RGB} (Volume not populated?)")
    p4.extract_dir_4phase(net, train_imgs, FEAT_TRAIN); vol.commit()
    p4.extract_dir_4phase(net, eval_imgs, FEAT_EVAL); vol.commit()
    n_tr = len(glob.glob(os.path.join(FEAT_TRAIN, "*.npy")))
    n_ev = len(glob.glob(os.path.join(FEAT_EVAL, "*.npy")))
    dt = (time.time() - t0) / 60
    print(f"[extract_4p] done feat_train={n_tr} feat_eval={n_ev} in {dt:.1f} min",
          flush=True)
    return {"feat_train": n_tr, "feat_eval": n_ev, "extract_min": round(dt, 1)}


@app.function(gpu=GPU, image=image, volumes={VOL: vol}, timeout=4 * 3600,
              cpu=8, memory=131072, secrets=[hf_secret])   # 128GB: ~69GB 4x train cache
def train_eval_4p(seed: int = 0, epochs: int = 40):
    """Train ONE seed on the 4-phase features (native recipe) and write eval predictions
    for the NEON scorer. Byte-identical recipe to native seed-0 except the features/head.
    Reports a live budget estimate from wall-time (H100 ~= $0.082/min, the README basis)."""
    import time
    import torch
    _setup_path()
    assert torch.cuda.is_available(), "no CUDA"
    from boxinst_commonality_tcd_04.modal_neon_multiseed.phase4 import phase4_lib as L
    os.makedirs(OUT, exist_ok=True)
    tag = f"phase4_s{seed}"
    t0 = time.time()
    best = L.train_seed(FEAT_TRAIN, f"{VOL}/train_patches_gt.json", OUT, tag=tag,
                        seed=seed, epochs=epochs, device="cuda", commit=vol.commit)
    train_min = (time.time() - t0) / 60
    ckpt = os.path.join(OUT, f"det_{tag}.pt")
    preds_fp = os.path.join(OUT, f"preds_{tag}.json")
    t1 = time.time()
    L.predict_boxes_4p(ckpt, FEAT_EVAL, f"{VOL}/neon_gt.json", EVAL_RGB, preds_fp,
                       device="cuda")
    eval_min = (time.time() - t1) / 60
    res = {"tag": tag, "preds": preds_fp, "train_min": round(train_min, 1),
           "eval_min": round(eval_min, 1), "best_val_boxAP50": round(best["mAP50"], 4),
           "best_epoch": best["epoch"],
           "est_cost_usd": round((train_min + eval_min) * 0.082, 2)}
    json.dump(res, open(os.path.join(OUT, f"results_{tag}.json"), "w"), indent=2)
    vol.commit()
    print(f"[train_eval_4p] {tag}: train {train_min:.1f}min eval {eval_min:.1f}min "
          f"valAP50={best['mAP50']:.3f} est_cost≈${res['est_cost_usd']} "
          f"(score post-hoc: df_scorer)", flush=True)
    return res


@app.function(gpu=GPU, image=image, volumes={VOL: vol}, timeout=6 * 3600,
              cpu=8, memory=131072, secrets=[hf_secret])   # 128GB: ~82GB train+eval cache
def multiseed_4p(seeds: str = "1,2,3,4", epochs: int = 40):
    """AMORTIZED 4-phase band: ONE container, preload train+eval once, train+eval each
    seed reusing in-RAM caches. Per-seed idempotent skip + resumable. Reports container-
    lifetime wall + est cost (H100 ~= $0.082/min). Seed 0 already ran (train_eval_4p)."""
    import time
    import torch
    _setup_path()
    assert torch.cuda.is_available(), "no CUDA"
    from boxinst_commonality_tcd_04.modal_neon_multiseed.phase4 import phase4_lib as L
    os.makedirs(OUT, exist_ok=True)
    sl = [int(s) for s in str(seeds).split(",") if s.strip() != ""]
    print(f"[multiseed_4p] gpu={torch.cuda.get_device_name(0)} seeds={sl}", flush=True)
    t0 = time.time()
    results = L.run_multiseed(FEAT_TRAIN, f"{VOL}/train_patches_gt.json", FEAT_EVAL,
                              f"{VOL}/neon_gt.json", EVAL_RGB, OUT, sl, epochs=epochs,
                              device="cuda", commit=vol.commit)
    mins = (time.time() - t0) / 60
    vol.commit()
    print(f"[multiseed_4p] {len(sl)} seeds CONTAINER {mins:.1f}min est≈${mins*0.082:.2f}",
          flush=True)
    return {"container_min": round(mins, 1), "est_cost_usd": round(mins * 0.082, 2),
            "seeds": {s: results[s].get("best_val_boxAP50") for s in results}}


@app.local_entrypoint()
def main(seed: int = 0, epochs: int = 40, skip_extract: bool = False):
    if not skip_extract:
        print("== extract_4p ==")
        print(json.dumps(extract_4p.remote(), indent=2))
    print("== train_eval_4p ==")
    print(json.dumps(train_eval_4p.remote(seed=seed, epochs=epochs), indent=2))
