"""Modal A100 gate for block-1024: does det_t8's exact recipe hold box mAP50?

Validates PICKUP.md's decision point — train Detector8 on the last-block 1024-dim
slice of the DINOv3-web features with det_t8's EXACT hyperparameters, so the only
change vs the 0.555 single-scale box-mAP50 baseline is feature width. Within ~0.02
(covers GPU/MPS non-determinism) = block-1024 validated -> the ~121 GB full-4k
local caching is green-lit.

Feature source: instead of uploading 45 GB of pre-sliced local features, we
extract on the A100 from the raw tiles ALREADY on Modal — the HF restor/tcd
dataset on the canopyai-deepforest-data volume. Tiles are joined to the exact
900+439 baseline set by IMAGE_ID (the stable key in each local meta.json AND in
the HF rows), never by filename. verify() proves the join (all ids present,
width/height + ITC-count match, and a pixel-sha1 sample proves HF image == local
tile) before any GPU spend; extract() then runs cache_test.tile_feature VERBATIM
(2x2 of 1024 windows), keeps the last-block slice [3072:4096], and writes it to
the scratch volume; train_eval() reads that volume and runs the repo trainer +
box-only eval unmodified.

Stages (run in order):
    modal run app.py::verify
    modal run app.py::extract
    modal run app.py::main          # train_eval + verdict
Or the whole chain:
    modal run app.py
"""
import json
import os

import modal

APP_NAME = "tcd04-feat-ablation"
VOL_NAME = "tcd04-block1024-vol"           # scratch: extracted feats + outputs
HFDATA_VOL = "canopyai-deepforest-data"    # read-only: HF restor/tcd + hub cache

HERE = os.path.dirname(os.path.abspath(__file__))
FA = os.path.dirname(HERE)                 # feat_ablation/
PKG = os.path.dirname(FA)                  # boxinst_commonality_tcd_04/
REPO = os.path.dirname(PKG)
STUBS = os.path.join(HERE, "stubs")

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
hfvol = modal.Volume.from_name(HFDATA_VOL)

P = "/root/proj"
PKG_R = f"{P}/boxinst_commonality_tcd_04"
HF_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # pinned to the local venv so device is the only uncontrolled delta
        "torch==2.12.1", "torchvision==0.27.1", "numpy==2.2.6",
        # datasets 4.x: the cached restor/tcd arrow uses the 'List' feature type,
        # unknown to datasets 3.x (ValueError: Feature type 'List' not found)
        "transformers==4.57.1", "datasets==4.0.0", "pillow", "contourpy",
        "pycocotools",
    )
    .env({"HF_HOME": "/hfdata/hf_cache",
          "HF_HUB_OFFLINE": "0",           # model weights download (gated, via secret)
          "HF_DATASETS_OFFLINE": "1"})     # dataset is cached on the volume
)
# real modules, verbatim (backbone IS real here — extraction needs DINOv3)
for rel in ("dapt/__init__.py", "dapt/backbone.py", "dapt/targets.py",
            "dapt/decode.py", "dapt/eval.py", "dapt/head.py"):
    image = image.add_local_file(os.path.join(REPO, rel), f"{P}/{rel}")
for rel in ("__init__.py", "detector.py", "train_detector_tiles.py",
            "evaluate.py", "em.py", "prepare_test.py", "cache_test.py",
            "cache_train_tiles.py", "test_gt.json", "train_tiles_gt.json"):
    image = image.add_local_file(os.path.join(PKG, rel), f"{PKG_R}/{rel}")
for rel in ("__init__.py", "variants.py", "train_variant.py", "eval_box.py"):
    image = image.add_local_file(os.path.join(FA, rel),
                                 f"{PKG_R}/feat_ablation/{rel}")
image = image.add_local_file(os.path.join(HERE, "manifest.json"),
                             f"{P}/manifest.json")
# stubs only for the EM / extraction-lib imports that eval_box never executes.
# Files listed EXPLICITLY (no os.listdir): Modal re-imports this module inside the
# container, where the local stubs/ dir does not exist.
STUB_FILES = {
    "boxinst": ("__init__.py", "cache_feats.py"),
    "boxinst_commonality": ("__init__.py", "em.py"),
    "boxinst_tcd": ("__init__.py", "build_canopy.py", "cache.py", "prepare.py"),
}
for pkg, files in STUB_FILES.items():
    for f in files:
        image = image.add_local_file(os.path.join(STUBS, pkg, f),
                                     f"{P}/{pkg}/{f}")

hf_secret = modal.Secret.from_name("huggingface")
TMP = "/tmp/fa"
BLOCK_LO = 3072
HF_SPLIT = {"feat_traintile": "train", "feat_test": "test"}


def _load_hf():
    """image_id -> (split, hf.Dataset, row_index) for restor/tcd train+test.

    Reads only the image_id column to index (no image decode) so the map builds
    in seconds. Images are decoded lazily per-row later.
    """
    from datasets import load_dataset
    idx = {}
    ds_by_split = {}
    for split in ("train", "test"):
        ds = load_dataset("restor/tcd", split=split)
        ds_by_split[split] = ds
        for i, iid in enumerate(ds["image_id"]):
            idx[int(iid)] = (split, i)
    return idx, ds_by_split


@app.function(image=image, volumes={"/hfdata": hfvol}, timeout=1800,
              cpu=4, memory=32768, secrets=[hf_secret])
def verify():
    """Prove the image_id join before spending GPU: every baseline tile's
    image_id is in HF, width/height + ITC box-count agree, and a pixel-sha1
    sample confirms HF image bytes == local tile bytes."""
    import hashlib

    import numpy as np

    tok = [k for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
                       "HF_API_TOKEN")
           if os.environ.get(k)]
    warn = "" if tok else " (NONE — extract will fail to fetch gated DINOv3)"
    print(f"[verify] HF token env vars present: {tok}{warn}", flush=True)

    man = json.load(open(f"{P}/manifest.json"))
    idx, ds_by_split = _load_hf()
    print(f"[verify] HF train={len(ds_by_split['train'])} "
          f"test={len(ds_by_split['test'])}", flush=True)

    def cat2(ds_row):
        return sum(a["category_id"] == 2
                   for a in json.loads(ds_row["coco_annotations"]))

    miss, wh_bad, box_bad, hash_ok, hash_bad = [], [], [], 0, []
    for split, tiles in man.items():
        want_split = HF_SPLIT[split]
        for tid, rec in tiles.items():
            iid = rec["image_id"]
            if iid not in idx:
                miss.append((split, tid, iid)); continue
            hsplit, i = idx[iid]
            if hsplit != want_split:
                miss.append((split, tid, iid, f"in {hsplit} not {want_split}"))
                continue
            row = ds_by_split[hsplit][i]
            if row["width"] != rec["width"] or row["height"] != rec["height"]:
                wh_bad.append((tid, iid, row["width"], row["height"]))
            if cat2(row) != rec["n_cat2"]:
                box_bad.append((tid, iid, cat2(row), rec["n_cat2"]))
            if "rgb_sha1" in rec:
                arr = np.asarray(row["image"].convert("RGB"))
                h = hashlib.sha1(arr.tobytes()).hexdigest()
                if h == rec["rgb_sha1"] and list(arr.shape) == rec["rgb_shape"]:
                    hash_ok += 1
                else:
                    hash_bad.append((tid, iid, list(arr.shape), rec["rgb_shape"]))
    print(f"[verify] missing={len(miss)} wh_mismatch={len(wh_bad)} "
          f"box_count_mismatch={len(box_bad)} "
          f"pixel_sha1 ok={hash_ok} bad={len(hash_bad)}", flush=True)
    for label, lst in (("MISSING", miss), ("WH", wh_bad),
                       ("BOXES", box_bad), ("HASH", hash_bad)):
        for x in lst[:10]:
            print(f"  {label}: {x}", flush=True)
    ok = not (miss or wh_bad or box_bad or hash_bad) and hash_ok > 0
    print(f"[verify] JOIN {'OK' if ok else 'FAILED'}", flush=True)
    assert ok, "image_id join verification failed — do NOT extract"
    return {"train": len(man["feat_traintile"]), "test": len(man["feat_test"]),
            "pixel_hash_checked": hash_ok}


@app.function(gpu="A100", image=image, volumes={"/vol": vol, "/hfdata": hfvol},
              timeout=6 * 3600, cpu=8, memory=49152, secrets=[hf_secret])
def extract():
    """HF tile (by image_id) -> cache_test.tile_feature (2x2 1024 windows) ->
    block-1024 slice -> /vol/feat_<split>/<tid>.npy. Skips already-written tiles
    and commits periodically, so a timeout/kill resumes cleanly."""
    import sys

    import numpy as np
    import torch

    assert torch.cuda.is_available(), "no CUDA — fix image before burning GPU"
    sys.path.insert(0, P)
    from dapt.backbone import FrozenDinoV3Features
    from boxinst_commonality_tcd_04.cache_test import tile_feature

    man = json.load(open(f"{P}/manifest.json"))
    idx, ds_by_split = _load_hf()
    # layers MUST be boxinst.cache_feats.LAYERS (last 4 blocks), hardcoded since
    # that module isn't shipped: a bare call falls back to dapt.backbone's
    # (3,6,9,12) default — same out_dim, silently different features. This bug
    # cost a full multiseed extraction+training run before it was caught.
    net = FrozenDinoV3Features("web", layers=(21, 22, 23, 24), device="cuda")
    print(f"[extract] gpu={torch.cuda.get_device_name(0)} "
          f"out_dim={net.out_dim} layers={net.layers}", flush=True)
    assert net.out_dim == 4096, f"expected 4096-dim web features, got {net.out_dim}"
    assert net.layers == (21, 22, 23, 24), net.layers

    import time
    for split, tiles in man.items():
        outdir = f"/vol/{split}"
        os.makedirs(outdir, exist_ok=True)
        todo = [t for t in sorted(tiles)
                if not os.path.exists(os.path.join(outdir, t + ".npy"))]
        print(f"[extract:{split}] {len(todo)}/{len(tiles)} to extract", flush=True)
        t0 = time.time()
        for k, tid in enumerate(todo):
            hsplit, i = idx[tiles[tid]["image_id"]]
            img = ds_by_split[hsplit][i]["image"].convert("RGB")
            feat = tile_feature(net, img, net.device)     # (4096,128,128) fp16
            assert feat.shape[0] == 4096, f"{tid}: {feat.shape}"
            sl = np.ascontiguousarray(feat[BLOCK_LO:])    # (1024,128,128)
            np.save(os.path.join(outdir, tid + ".npy"), sl)
            if (k + 1) % 25 == 0 or k + 1 == len(todo):
                dt = time.time() - t0
                eta = (len(todo) - k - 1) * dt / (k + 1) / 60
                print(f"  {k+1}/{len(todo)}  {dt/ (k+1):.1f}s/tile  "
                      f"ETA {eta:.0f} min", flush=True)
                vol.commit()
        vol.commit()
    n_tr = len(os.listdir("/vol/feat_traintile"))
    n_te = len(os.listdir("/vol/feat_test"))
    print(f"[extract] done: traintile={n_tr} test={n_te}", flush=True)
    return {"traintile": n_tr, "test": n_te}


@app.function(gpu="A100", image=image, volumes={"/vol": vol}, timeout=6 * 3600,
              cpu=8, memory=49152)
def train_eval(epochs: int = 40, bs: int = 3, eval_every: int = 5,
               seed: int = 0, lr: float = 1e-3, wd: float = 1e-4,
               width: int = 256, tower: int = 3):
    import shutil
    import sys
    import types

    import numpy as np
    import torch

    assert torch.cuda.is_available(), "no CUDA — fix image before burning GPU"
    print(f"[preflight] gpu={torch.cuda.get_device_name(0)} "
          f"torch={torch.__version__}", flush=True)
    n_tr = len(os.listdir("/vol/feat_traintile"))
    n_te = len(os.listdir("/vol/feat_test"))
    assert n_tr == 900 and n_te == 439, \
        f"volume incomplete: traintile={n_tr}/900 test={n_te}/439 — run extract"

    sys.path.insert(0, P)
    # volume files are ALREADY the 3072:4096 slice -> plain load, no re-slicing
    import boxinst_commonality_tcd_04.feat_ablation.variants as V
    V.feat_dir = lambda variant, split: f"/vol/{split}"
    V.load_feat = lambda variant, split, tid: np.load(f"/vol/{split}/{tid}.npy")

    # imported AFTER the patch so their `from variants import ...` bind to it
    import boxinst_commonality_tcd_04.feat_ablation.train_variant as tv
    import boxinst_commonality_tcd_04.train_detector_tiles as tdt
    tv.patch("block1024")
    tdt.ART = os.path.join(TMP, "artifacts")      # mounted tree is not writable

    args = types.SimpleNamespace(
        variant="block1024", tag="fa_block1024", arm="block1024",
        epochs=epochs, seed=seed, lr=lr, wd=wd, bs=bs, width=width,
        tower=tower, eval_every=eval_every, device=None)
    tdt.train(args)

    import boxinst_commonality_tcd_04.feat_ablation.eval_box as eb
    eb.HERE = TMP
    ckpt = os.path.join(TMP, "artifacts", "det_fa_block1024.pt")
    eb.run(types.SimpleNamespace(
        variant="block1024", ckpt=ckpt, tag=None, limit=None,
        topk=600, eval_score_thr=0.05, device=None))

    os.makedirs("/vol/out", exist_ok=True)
    shutil.copy(ckpt, "/vol/out/")
    res_fp = os.path.join(TMP, "results", "eval_box_block1024.json")
    shutil.copy(res_fp, "/vol/out/")
    vol.commit()
    return json.load(open(res_fp))


@app.local_entrypoint()
def main(epochs: int = 40, bs: int = 3, eval_every: int = 5, seed: int = 0,
         skip_verify: bool = False, skip_extract: bool = False):
    if not skip_verify:
        print("== verify ==")
        print(json.dumps(verify.remote(), indent=2))
    if not skip_extract:
        print("== extract ==")
        print(json.dumps(extract.remote(), indent=2))
    print("== train + eval ==")
    res = train_eval.remote(epochs=epochs, bs=bs, eval_every=eval_every,
                            seed=seed)
    print(json.dumps(res, indent=2))
    print(f"\nblock-1024 box mAP50 = {res['box_mAP50']:.4f} "
          f"vs full-4096 baseline 0.555 (gate: within ~0.02)")
