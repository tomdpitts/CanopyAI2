"""Modal app: fully-supervised DetecTree2 (Mask R-CNN R101-FPN) baseline on OAM-TCD, for a
perfect apples-to-apples vs our weakly-supervised 0.615 mask mAP50 — SAME 792 train / 108
val / 439 test, SAME evaluate.py scorer + canopy-ignore.

Pipeline (checkpointed so we can pause before the training spend):
  build_data   (CPU)  : HF restor/tcd images -> 1024@50% subtiles + COCO (crowns=full-res
                        masks, canopy=iscrowd ignore), 792/108/439 -> volume.  no GPU.
  fetch_weights(CPU)  : DetecTree2 released 250312_flexi.pth (Zenodo) -> volume.
  imports_ok   (GPU)  : cheap image-build gate: import torch/detectron2/detectree2, load cfg.
  train        (A100) : detectree2 setup_cfg (R101-FPN, their recipe) + released weights +
                        MyTrainer (AP50 early-stop) on trees_train, model-select on trees_val.
  predict      (A100) : DefaultPredictor over the 439 test subtiles -> per-subtile RLE+score
                        -> stitch.py (clean_crowns dedup) -> our preds schema (--save_preds).
Score runs LOCALLY afterwards: detectree2_baseline/score_detectree2.py (no detectron2 dep).
"""
import json
import os

import modal

APP = "tcd-detectree2"
VOL_NAME = "tcd-detectree2-vol"
HFDATA_VOL = "canopyai-deepforest-data"          # read HF restor/tcd images (reused)

HERE = os.path.dirname(os.path.abspath(__file__))          # .../detectree2_baseline
PKG = os.path.dirname(HERE)                                # boxinst_commonality_tcd_04
REPO = os.path.dirname(PKG)

app = modal.App(APP)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
hfvol = modal.Volume.from_name(HFDATA_VOL)
hf_secret = modal.Secret.from_name("huggingface")

P = "/root/proj"
PKG_R = f"{P}/boxinst_commonality_tcd_04"
DT2_R = f"{PKG_R}/detectree2_baseline"

# --- light CPU image for the data build (no detectron2) ---
data_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "datasets==4.0.0", "pillow", "pycocotools")
    .env({"HF_HOME": "/hfdata/hf_cache", "HF_DATASETS_OFFLINE": "1"})
    .add_local_file(os.path.join(HERE, "build_coco.py"), f"{DT2_R}/build_coco.py")
    .add_local_file(os.path.join(HERE, "stitch.py"), f"{DT2_R}/stitch.py")
    .add_local_file(os.path.join(PKG, "train_tiles_gt.json"),
                    f"{PKG_R}/train_tiles_gt.json")
    .add_local_file(os.path.join(PKG, "test_gt.json"), f"{PKG_R}/test_gt.json")
    .add_local_file(os.path.join(PKG, "modal_tcd_multiseed", "phase4", "manifest.json"),
                    f"{DT2_R}/manifest.json")
    .add_local_file(os.path.join(PKG, "modal_sparse_tcd_multiseed", "manifest_sparse.json"),
                    f"{DT2_R}/manifest_sparse.json")
    .add_local_file(os.path.join(REPO, "data/tcd_sparse/sparse_gt.json"),
                    f"{DT2_R}/sparse_gt.json")          # stitch_cpu resolves tids from this
)

# --- heavy GPU image: Modal's in-container runtime needs Python>=3.10, which rules out the
# torch1.10/detectron2-0.6 prebuilt-wheel combo (py<=3.9). So use a modern py3.10 CUDA base
# (torch 2.1 + cu121, ubuntu-jammy) and BUILD detectron2 from source against it. We reproduce
# DetecTree2's recipe via detectron2 (dt2_recipe.py) driving its released 250312_flexi.pth
# weights; the detectree2 package itself is not needed (its geo/GDAL layer is bypassed for
# our fixed-pixel RGB tiles). detectron2's config/trainer/hook APIs are stable 0.6->main. ---
dt2_image = (
    modal.Image.from_registry("pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel")
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})   # avoid tzdata apt prompt
    .apt_install("git", "g++", "ninja-build", "libgl1", "libglib2.0-0")
    .env({"TORCH_CUDA_ARCH_LIST": "8.0", "FORCE_CUDA": "1"})    # A100 sm_80; build cuda ops
    .pip_install("numpy<2", "opencv-python-headless", "pycocotools")
    .run_commands("pip install --no-build-isolation "
                  "'git+https://github.com/facebookresearch/detectron2.git'")
    .add_local_file(os.path.join(HERE, "build_coco.py"), f"{DT2_R}/build_coco.py")
    .add_local_file(os.path.join(HERE, "stitch.py"), f"{DT2_R}/stitch.py")
    .add_local_file(os.path.join(HERE, "dt2_recipe.py"), f"{DT2_R}/dt2_recipe.py")
    .add_local_file(os.path.join(PKG, "test_gt.json"), f"{PKG_R}/test_gt.json")
    .add_local_file(os.path.join(REPO, "data/tcd_sparse/sparse_gt.json"),
                    f"{DT2_R}/sparse_gt.json")
)

VOL = "/vol"
DATA = f"{VOL}/data"                              # data/{train,val,test}/{images,coco.json}
WEIGHTS = f"{VOL}/weights"
OUT = f"{VOL}/out"
MANIFEST = f"{DT2_R}/manifest.json"
TRAIN_GT = f"{PKG_R}/train_tiles_gt.json"
FLEXI = "250312_flexi.pth"
FLEXI_URL = "https://zenodo.org/records/15014353/files/250312_flexi.pth?download=1"


def _setup_path():
    import sys
    if P not in sys.path:
        sys.path.insert(0, P)


def _load_hf():
    from datasets import load_dataset
    idx, ds = {}, {}
    for split in ("train", "test"):
        d = load_dataset("restor/tcd", split=split)
        ds[split] = d
        for i, iid in enumerate(d["image_id"]):
            idx[int(iid)] = (split, i)
    return idx, ds


N_SHARDS = 10                                    # parallel data-build containers (cost cap)


SPARSE_MANIFEST = f"{DT2_R}/manifest_sparse.json"


def _paths(dataset, seed):
    """Per-dataset image dir / GT / raw-pred dir / preds file.

    dataset='tcd' reproduces the ORIGINAL hardcoded paths exactly, so the settled 439 run
    (out/preds_dt2_s0.json, 0.545) is untouched and re-runnable. Any other dataset gets its
    own subtree — nothing can overwrite /vol/data/test/images."""
    if dataset == "tcd":
        return (f"{DATA}/test/images", f"{PKG_R}/test_gt.json",
                f"{OUT}/pred_raw_s{seed}", f"{OUT}/preds_dt2_s{seed}.json")
    return (f"{DATA}/{dataset}/images", f"{DT2_R}/{dataset}_gt.json",
            f"{OUT}/pred_raw_{dataset}_s{seed}", f"{OUT}/preds_dt2_{dataset}_s{seed}.json")


def _sparse_items():
    """[(hf_split, image_id, tid)] for the 236-tile sparse slice.

    The slice draws from BOTH HF splits (220 from `train`, 16 from `test`), so the split is
    resolved per-tile from the image_id index rather than assumed."""
    man = json.load(open(SPARSE_MANIFEST))["feat_test"]
    return [("sparse", rec["image_id"], tid) for tid, rec in sorted(man.items())]


def _cohort_items():
    """[(split, hf_split, image_id, tid)] for the whole 792/108/439 cohort."""
    man = json.load(open(MANIFEST))
    tgt = json.load(open(TRAIN_GT))
    items = []
    for tid, rec in man["feat_traintile"].items():
        items.append((tgt[tid]["partition"], "train", rec["image_id"], tid))
    for tid, rec in man["feat_test"].items():
        items.append(("test", "test", rec["image_id"], tid))
    return items


@app.function(image=data_image, volumes={"/vol": vol, "/hfdata": hfvol}, timeout=3600,
              cpu=2, memory=16384, secrets=[hf_secret], max_containers=10)
def build_shard(shard):
    """Build one shard of tiles: write subtile PNGs to the volume, return per-tile COCO
    records (local ids). Idempotent per PNG (skip if exists)."""
    import numpy as np
    from PIL import Image
    import sys
    sys.path.insert(0, DT2_R)
    import build_coco as B
    idx, ds = _load_hf()
    out = {"train": [], "val": [], "test": []}
    for (split, hfs, image_id, tid) in shard:
        os.makedirs(f"{DATA}/{split}/images", exist_ok=True)
        row = ds[hfs][idx[image_id][1]]
        img = np.asarray(row["image"].convert("RGB"))
        ca = json.loads(row["coco_annotations"])
        im_t, an_t, crops = B.build_tile(tid, img, ca, 0, 0)      # tile-local ids
        for fn, arr in crops:
            # always (over)write: a prior timed-out run can leave truncated PNGs that a
            # skip-if-exists would keep, and a truncated tile crashes training.
            Image.fromarray(arr).save(os.path.join(f"{DATA}/{split}/images", fn))
        out[split].append({"images": im_t, "anns": an_t})
    vol.commit()
    return out


@app.function(image=data_image, volumes={"/vol": vol}, timeout=3600, cpu=4, memory=32768)
def build_data():
    """Parallel 1024@50% subtile COCO build for train(792)/val(108)/test(439): fan tiles out
    over N_SHARDS containers via .map, then assemble per-split coco.json (renumber ids)."""
    import sys
    sys.path.insert(0, DT2_R)
    import build_coco as B
    if os.path.exists(f"{DATA}/test/coco.json"):
        print("[build_data] exists -> skip", flush=True)
        return {"skipped": True}
    items = _cohort_items()
    shards = [items[i::N_SHARDS] for i in range(N_SHARDS)]
    n = {s: sum(1 for it in items if it[0] == s) for s in ("train", "val", "test")}
    print(f"[build_data] {len(items)} tiles ({n}) over {N_SHARDS} shards", flush=True)
    results = list(build_shard.map(shards))          # parallel fan-out
    out = {}
    for split in ("train", "val", "test"):
        images, anns = [], []
        iid = aid = 0
        for r in results:
            for tile in r[split]:
                remap = {}
                for im in tile["images"]:
                    remap[im["id"]] = iid
                    images.append({**im, "id": iid}); iid += 1
                for an in tile["anns"]:
                    anns.append({**an, "id": aid, "image_id": remap[an["image_id"]]}); aid += 1
        os.makedirs(f"{DATA}/{split}", exist_ok=True)     # shard-created in other containers
        json.dump(B.coco_dict(images, anns), open(f"{DATA}/{split}/coco.json", "w"))
        out[split] = {"subtiles": len(images),
                      "crowns": sum(a["iscrowd"] == 0 for a in anns),
                      "canopy_ignore": sum(a["iscrowd"] == 1 for a in anns)}
        print(f"[build_data] {split}: {out[split]}", flush=True)
    vol.commit()
    return out


@app.function(image=data_image, volumes={"/vol": vol, "/hfdata": hfvol}, timeout=3600,
              cpu=2, memory=16384, secrets=[hf_secret], max_containers=10)
def build_shard_sparse(shard):
    """Write 1024@50% subtile PNGs for one shard, into {DATA}/{subdir}/images.

    Prediction needs images only -- no COCO records. keep_empty=True: ALL 9 subtiles, so the
    model is charged for false positives in label-free regions (see build_coco.build_tile)."""
    import numpy as np
    from PIL import Image
    import sys
    sys.path.insert(0, DT2_R)
    import build_coco as B
    idx, ds = _load_hf()
    n = 0
    for (subdir, image_id, tid) in shard:
        os.makedirs(f"{DATA}/{subdir}/images", exist_ok=True)
        hfs, row_i = idx[int(image_id)]                  # split resolved from the image_id
        row = ds[hfs][row_i]
        img = np.asarray(row["image"].convert("RGB"))
        ca = json.loads(row["coco_annotations"])
        # keep_empty=True: predict on ALL 9 subtiles. Skipping label-free subtiles would
        # never charge DetecTree2 for false positives there, while our method (whole-tile,
        # no subtiling) IS charged — an unfair asymmetry in an open-canopy comparison.
        _, _, crops = B.build_tile(tid, img, ca, 0, 0, keep_empty=True)
        for fn, arr in crops:
            Image.fromarray(arr).save(os.path.join(f"{DATA}/{subdir}/images", fn))
        n += len(crops)
    vol.commit()
    return n


@app.function(image=data_image, volumes={"/vol": vol}, timeout=3600, cpu=4, memory=16384)
def build_data_sparse():
    """Fan the 236 sparse tiles out over N_SHARDS containers -> 2124 subtile PNGs.

    Writes to /vol/data/sparse/ -- /vol/data/test/ (the 439) is never touched."""
    items = _sparse_items()
    want = len(items) * 9
    imgdir = f"{DATA}/sparse/images"
    vol.reload()                      # shards commit from OTHER containers; refresh this view
    have = len(os.listdir(imgdir)) if os.path.isdir(imgdir) else 0
    if have == want:
        print(f"[build_data_sparse] {have} subtiles already on volume -> skip", flush=True)
        return {"tiles": len(items), "subtiles": have, "skipped": True}
    shards = [items[i::N_SHARDS] for i in range(N_SHARDS)]
    print(f"[build_data_sparse] {len(items)} tiles over {N_SHARDS} shards "
          f"({have}/{want} present)", flush=True)
    n = sum(build_shard_sparse.map(shards))
    vol.reload()
    have = len(os.listdir(imgdir))
    print(f"[build_data_sparse] wrote {n} subtiles; on volume: {have}", flush=True)
    assert have == len(items) * 9, f"expected {len(items) * 9} subtiles, found {have}"
    return {"tiles": len(items), "subtiles": have}


@app.function(image=data_image, volumes={"/vol": vol}, timeout=3600, cpu=4, memory=16384)
def build_test_fullcov():
    """Re-emit the 439 TEST subtiles with keep_empty=True -> full 9/9 coverage (3951).

    The original build dropped subtiles carrying no crown and no canopy, so DetecTree2 was
    never inferred on those regions and never charged for false positives there (3189/3951 =
    80.7% coverage), while our whole-tile method always is. This makes the 439 baseline
    directly comparable to the sparse run, which is already full-coverage.

    Additive: the existing 3189 PNGs are rewritten identically and 762 are added. The
    published out/preds_dt2_s0.json is NOT touched -- stitch to a new name."""
    man = json.load(open(MANIFEST))["feat_test"]
    items = [("test", rec["image_id"], tid) for tid, rec in sorted(man.items())]
    want = len(items) * 9
    vol.reload()
    have = len(os.listdir(f"{DATA}/test/images"))
    print(f"[build_test_fullcov] {len(items)} tiles, {have}/{want} subtiles present", flush=True)
    if have == want:
        print("[build_test_fullcov] already full coverage -> skip", flush=True)
        return {"tiles": len(items), "subtiles": have, "skipped": True}
    shards = [items[i::N_SHARDS] for i in range(N_SHARDS)]
    n = sum(build_shard_sparse.map(shards))
    vol.reload()
    have = len(os.listdir(f"{DATA}/test/images"))
    print(f"[build_test_fullcov] wrote {n}; on volume: {have}", flush=True)
    assert have == want, f"expected {want}, found {have}"
    return {"tiles": len(items), "subtiles": have}


@app.function(image=data_image, volumes={"/vol": vol}, timeout=3600, cpu=8)
def validate_data():
    """Cheap CPU gate before the A100: parse each split's coco.json and fully load EVERY
    referenced PNG (catches truncated/corrupt tiles). Fails loudly if anything is wrong, so a
    bad file can never waste a GPU training run again."""
    from PIL import Image
    bad, missing, total = [], [], 0
    summary = {}
    for split in ("train", "val", "test"):
        coco = json.load(open(f"{DATA}/{split}/coco.json"))
        imgs = coco["images"]
        summary[split] = {"images": len(imgs), "anns": len(coco["annotations"])}
        for im in imgs:
            p = f"{DATA}/{split}/images/{im['file_name']}"
            total += 1
            if not os.path.exists(p):
                missing.append(p); continue
            try:
                Image.open(p).load()               # full decode -> raises on truncation
            except Exception as e:
                bad.append((p, str(e)))
        print(f"[validate] {split}: {summary[split]} checked {len(imgs)} pngs", flush=True)
    print(f"[validate] total={total} missing={len(missing)} corrupt={len(bad)}", flush=True)
    if missing or bad:
        print(f"  missing[:5]={missing[:5]}\n  corrupt[:5]={bad[:5]}", flush=True)
    assert not missing and not bad, f"DATA INVALID: {len(missing)} missing, {len(bad)} corrupt"
    print("[validate] ALL CLEAN — safe to train", flush=True)
    return {"total": total, "ok": True, **summary}


@app.function(image=data_image, volumes={"/vol": vol}, timeout=6 * 3600, cpu=8,
              memory=32768)
def stitch_cpu(seed: int = 0, iou_thr: float = 0.7, cont_thr: float = 0.85,
               min_score: float = 0.1, dataset: str = "tcd", out_name: str = ""):
    """Single-container CPU stitch of the committed raw per-subtile predictions -> our preds
    schema (avoids paying A100 rates for the mask-decode/dedup CPU work, and avoids the flaky
    .map-from-within-a-function path). min_score drops junk low-conf preds before decode so the
    stitch is tractable. stitch.stitch_all logs 'stitched N/439'. -> preds_dt2_s{seed}.json."""
    import sys
    sys.path.insert(0, DT2_R)
    import stitch
    _, gt_path, raw, fp = _paths(dataset, seed)
    if out_name:                       # never clobber a published preds file
        fp = f"{OUT}/{out_name}"
    tids = sorted(json.load(open(gt_path)).keys())
    res = stitch.stitch_all(raw, tids, iou_thr=iou_thr,
                            cont_thr=cont_thr, min_score=min_score)
    res["meta"]["seed"] = seed
    res["meta"]["dataset"] = dataset
    res["meta"]["min_score"] = min_score
    json.dump(res, open(fp, "w"))
    vol.commit()
    n = sum(len(p["scores"]) for p in res["preds"].values())
    print(f"[stitch_cpu] saved {fp}: {len(res['preds'])} tiles, {n} crowns", flush=True)
    return {"preds": fp, "n_tiles": len(res["preds"]), "n_crowns": n}


@app.function(image=data_image, volumes={"/vol": vol}, timeout=1800)
def fetch_weights():
    """Download DetecTree2's released 250312_flexi.pth (general model) to the volume."""
    import urllib.request
    os.makedirs(WEIGHTS, exist_ok=True)
    dst = f"{WEIGHTS}/{FLEXI}"
    if os.path.exists(dst):
        print(f"[fetch_weights] exists -> {dst}", flush=True)
        return {"skipped": True, "path": dst}
    print(f"[fetch_weights] downloading {FLEXI_URL}", flush=True)
    urllib.request.urlretrieve(FLEXI_URL, dst)
    vol.commit()
    print(f"[fetch_weights] saved {dst} ({os.path.getsize(dst)/1e6:.0f} MB)", flush=True)
    return {"path": dst, "mb": round(os.path.getsize(dst) / 1e6)}


@app.function(image=dt2_image, gpu="A100", volumes={"/vol": vol}, timeout=1800)
def imports_ok():
    """Cheap image-build gate: confirm torch+CUDA + detectron2 import and DetecTree2's config
    (dt2_recipe.setup_cfg) loads with the R101-FPN zoo config — before spending on training."""
    import sys
    sys.path.insert(0, DT2_R)
    import torch
    import detectron2
    import dt2_recipe
    info = {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "detectron2": detectron2.__version__}
    cfg = dt2_recipe.setup_cfg(update_model=None, max_iter=1)
    info["base_lr"] = cfg.SOLVER.BASE_LR
    info["freeze_at"] = cfg.MODEL.BACKBONE.FREEZE_AT
    info["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
    info["base_model"] = dt2_recipe.BASE_MODEL
    print(json.dumps(info, indent=2), flush=True)
    return info


def _register(name, split):
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog, DatasetCatalog
    if name in DatasetCatalog.list():
        return
    register_coco_instances(name, {}, f"{DATA}/{split}/coco.json", f"{DATA}/{split}/images")
    MetadataCatalog.get(name).thing_classes = ["tree"]


@app.function(image=dt2_image, gpu="A100", volumes={"/vol": vol}, timeout=6 * 3600)
def train(seed: int = 0, max_iter: int = 4000, eval_period: int = 500, patience: int = 6,
          dets: int = 500, ims_per_batch: int = 4, val_eval_n: int = 256):
    """Fine-tune DetecTree2 (R101-FPN + released 250312_flexi.pth) on the 6,692 train subtiles,
    model-select by segm AP50 on a val-eval SUBSET (early-stop). Full training regime; only the
    in-training eval is trimmed (subset + eval_period 500) for speed — final report is on all 439
    test tiles. PREEMPTION-SAFE: checkpoints + AP history are committed to the volume every
    eval_period iters and resume=True continues from the last committed checkpoint on a restart.
    Saves best -> out/model_best_s{seed}.pth."""
    import sys
    sys.path.insert(0, DT2_R)
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True     # a stray truncated tile can't crash the A100 run
    import torch
    from detectron2.utils import comm  # noqa
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.engine.train_loop import HookBase
    import dt2_recipe
    assert torch.cuda.is_available(), "no CUDA"
    assert os.path.exists(f"{DATA}/train/coco.json"), "run build_data first"
    assert os.path.exists(f"{WEIGHTS}/{FLEXI}"), "run fetch_weights first"
    _register("trees_train", "train")
    # val-eval SUBSET: a strided ~val_eval_n-subtile slice of the 896 val subtiles — a fast,
    # representative AP50 signal for model selection (each full-val eval is ~10min; this ~3min).
    # Model SELECTION only; the reported comparison is on the full 439 test tiles via our scorer.
    vf = json.load(open(f"{DATA}/val/coco.json"))
    step = max(1, len(vf["images"]) // val_eval_n)
    vimgs = vf["images"][::step][:val_eval_n]
    keep = {im["id"] for im in vimgs}
    os.makedirs(f"{DATA}/val_eval", exist_ok=True)
    json.dump({"images": vimgs,
               "annotations": [a for a in vf["annotations"] if a["image_id"] in keep],
               "categories": vf["categories"]}, open(f"{DATA}/val_eval/coco.json", "w"))
    if "trees_val" not in DatasetCatalog.list():
        register_coco_instances("trees_val", {}, f"{DATA}/val_eval/coco.json",
                                f"{DATA}/val/images")
        MetadataCatalog.get("trees_val").thing_classes = ["tree"]
    outdir = f"{OUT}/train_s{seed}"
    cfg = dt2_recipe.setup_cfg(trains=("trees_train",), tests=("trees_val",),
                              update_model=f"{WEIGHTS}/{FLEXI}", max_iter=max_iter,
                              eval_period=eval_period, out_dir=outdir,
                              ims_per_batch=ims_per_batch, workers=8)
    cfg.INPUT.MASK_FORMAT = "bitmask"                     # our RLE crown masks
    cfg.SEED = seed
    cfg.TEST.DETECTIONS_PER_IMAGE = dets                 # dense TCD tiles (up to ~150/subtile)
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 3000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1500
    cfg.SOLVER.CHECKPOINT_PERIOD = eval_period           # periodic resume points
    gpu = torch.cuda.get_device_name(0)                  # record exact A100 variant (40 vs 80GB)
    print(f"[train] seed={seed} gpu={gpu} base_lr={cfg.SOLVER.BASE_LR} "
          f"freeze={cfg.MODEL.BACKBONE.FREEZE_AT} max_iter={max_iter} eval_every={eval_period} "
          f"patience={patience} dets={dets} val_eval={len(vimgs)}/{len(vf['images'])} subtiles",
          flush=True)
    trainer = dt2_recipe.MyTrainer(cfg, patience)

    class _CommitHook(HookBase):
        """Commit the volume every eval_period iters so checkpoints/ap_history are durable for a
        preemption-resume (uncommitted volume writes are lost when a container is preempted)."""
        def __init__(self, period):
            self._p = period
        def after_step(self):
            n = self.trainer.iter + 1
            if n % self._p == 0 or n == self.trainer.max_iter:
                vol.commit()
                print(f"[commit] volume committed @ iter {n}", flush=True)
    trainer.register_hooks([_CommitHook(eval_period)])   # runs last -> after checkpoint writes
    trainer.resume_or_load(resume=True)                  # continue from last commit if preempted
    trainer.train()
    best = f"{OUT}/model_best_s{seed}.pth"
    DetectionCheckpointer(trainer.model, save_dir=OUT).save(f"model_best_s{seed}")
    info = {"seed": seed, "gpu": gpu, "best_ap50": max(trainer.APs) if trainer.APs else None,
            "n_evals": len(trainer.APs), "max_iter": max_iter, "eval_period": eval_period,
            "patience": patience, "dets": dets, "base_lr": cfg.SOLVER.BASE_LR,
            "weights": FLEXI, "model": best}
    json.dump(info, open(f"{OUT}/train_info_s{seed}.json", "w"), indent=2)   # durable run record
    vol.commit()
    print(f"[train] done -> {best}  gpu={gpu}  "
          f"best_AP50={max(trainer.APs) if trainer.APs else None}", flush=True)
    return info


@app.function(image=dt2_image, gpu="A100", volumes={"/vol": vol}, timeout=4 * 3600)
def predict(seed: int = 0, score_thr: float = 0.05, dets: int = 500,
            iou_thr: float = 0.7, cont_thr: float = 0.85, dataset: str = "tcd",
            stitch_inline: bool = True):
    """Predict crowns with the fine-tuned model, stitch to whole tiles (clean_crowns dedup),
    and SAVE predictions in our schema.

    dataset='tcd' (default) = the original 439 test run, byte-identical paths.
    dataset='sparse' = the 236-tile unseen open-canopy slice (data/tcd_sparse). The model is
    UNCHANGED — same model_best_s{seed}.pth fine-tuned on the 792/108, which is disjoint from
    the sparse slice by construction, so this is a genuine zero-shot transfer measurement.

    stitch_inline=False stops after committing the raw per-subtile predictions, so the
    mask-decode/dedup can run via stitch_cpu at CPU rates instead of A100 rates (that CPU
    work dominated the 439 run). Default True keeps the original one-shot behaviour."""
    import sys
    sys.path.insert(0, DT2_R)
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    import cv2
    import torch
    from detectron2.engine import DefaultPredictor
    from detectron2.evaluation.coco_evaluation import instances_to_coco_json
    import dt2_recipe
    import stitch
    assert torch.cuda.is_available(), "no CUDA"
    model = f"{OUT}/model_best_s{seed}.pth"
    assert os.path.exists(model), f"{model} missing — run train --seed {seed}"
    cfg = dt2_recipe.setup_cfg(update_model=model, out_dir=f"{OUT}/pred_s{seed}")
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thr
    cfg.TEST.DETECTIONS_PER_IMAGE = dets
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1500
    predictor = DefaultPredictor(cfg)
    gpu = torch.cuda.get_device_name(0)                     # record exact A100 variant

    imgdir, gt_path, raw, fp = _paths(dataset, seed)
    os.makedirs(raw, exist_ok=True)
    assert os.path.isdir(imgdir), f"{imgdir} missing — run build_data{'_sparse' if dataset != 'tcd' else ''}"
    files = sorted(f for f in os.listdir(imgdir) if f.endswith(".png"))
    print(f"[predict] dataset={dataset}: {len(files)} subtiles on gpu={gpu}, "
          f"score_thr={score_thr} dets={dets}", flush=True)
    for k, fn in enumerate(files):
        stem = fn[:-4]
        outp = os.path.join(raw, f"Prediction_{stem}.json")
        if os.path.exists(outp):
            continue
        img = cv2.imread(os.path.join(imgdir, fn))          # BGR, matches cfg.INPUT.FORMAT
        inst = predictor(img)["instances"].to("cpu")
        json.dump(instances_to_coco_json(inst, stem), open(outp, "w"))
        if (k + 1) % 300 == 0 or k + 1 == len(files):
            # commit as we go: an ephemeral app dies with the local client, and an
            # end-of-loop-only commit throws away everything since the last flush.
            vol.commit()
            print(f"  predicted {k+1}/{len(files)}", flush=True)
    vol.commit()

    if not stitch_inline:
        print(f"[predict] raw predictions committed to {raw}; "
              f"run stitch_cpu --dataset {dataset} to finish (CPU rates)", flush=True)
        return {"seed": seed, "dataset": dataset, "gpu": gpu, "raw": raw,
                "n_subtiles": len(files), "stitched": False}

    tids = sorted(json.load(open(gt_path)).keys())
    res = stitch.stitch_all(raw, tids, iou_thr=iou_thr, cont_thr=cont_thr)
    res["meta"]["seed"] = seed
    res["meta"]["score_thr"] = score_thr
    res["meta"]["gpu"] = gpu
    res["meta"]["dataset"] = dataset
    json.dump(res, open(fp, "w"))
    vol.commit()
    n = sum(len(p["scores"]) for p in res["preds"].values())
    print(f"[predict] saved {fp} (gpu={gpu}): {len(res['preds'])} tiles, {n} crowns total",
          flush=True)
    return {"seed": seed, "dataset": dataset, "gpu": gpu, "preds": fp,
            "n_tiles": len(res["preds"]), "n_crowns": n}


@app.local_entrypoint()
def main():
    print("Run stages explicitly, e.g.:")
    print("  modal run detectree2_modal.py::build_data")
    print("  modal run detectree2_modal.py::fetch_weights")
    print("  modal run detectree2_modal.py::imports_ok")
