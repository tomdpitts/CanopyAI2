"""GT-box bake-off — Modal app. One prompt set, three box->mask modules.

Every arm is handed the SAME ground-truth crown boxes (`gtbox_lib.gt_boxes_2048`) and returns
one mask per box. Mask i is scored against crown i: no matching, no ranking, no score floor,
no detection confound. See `PLAN.md` for why this replaces the masker_lab bake-off.

Each arm writes ONLY per-crown IoUs (plus the GT-derived valid flags and the per-tile
double-assignment) to `{OUT}/gtbox_<arm>.json`. Masks are not persisted: at 25,705 crowns they
would be ~10x the size of the numbers anyone wants, and every downstream question -- by size,
by stratum, by tile -- is answerable from the per-crown IoU array. `combine_gtbox.py` merges
the arms locally.

Run (each ~40-60 min on A100; confirm the budget before launching all three):
    modal run gtbox_modal.py::eval_lace     --limit 20     # smoke
    modal run gtbox_modal.py::eval_lace
    modal run gtbox_modal.py::eval_sam3_base
    modal run gtbox_modal.py::eval_sam3_ft
"""
import json
import os

import modal

APP = "tcd04-gtbox-bakeoff"
PHASE4_VOL = "tcd04-phase4-vol"          # read: 4-phase test feats + the shipped EM npz
BAKE_VOL = "tcd04-baselines-vol"         # write: gtbox_*.json
HFDATA_VOL = "canopyai-deepforest-data"  # read: HF restor/tcd + hub cache

HERE = os.path.dirname(os.path.abspath(__file__))          # .../box_supervised_baselines
PKG = os.path.dirname(HERE)                                # boxinst_commonality_tcd_04
REPO = os.path.dirname(PKG)
MTS = os.path.join(PKG, "modal_tcd_multiseed")
PH4 = os.path.join(MTS, "phase4")
SAMD = os.path.join(MTS, "phase4_sam")

P = "/root/proj"
PKG_R = f"{P}/boxinst_commonality_tcd_04"
BSB_R = f"{PKG_R}/box_supervised_baselines"
PH4_R = f"{PKG_R}/modal_tcd_multiseed/phase4"

SAM3_COMMIT = "46957e47805eaa273f4aa7bbbd25a88bca9108ce"
SAM_BASE = "facebook/sam3"
SAM_FT_REPO = "CanopyRS/sam3-multi-selvabox-selvamask-FT"
SAM_FT_FILE = "sam3_selvamask_ft_model_best.pt"
SAM_FT_TILE = 1777        # the resolution CanopyRS fine-tuned and deploys at -- the setting
                          # most favourable to their model, so the arm cannot be called starved

# The SHIPPED masker: em_model_4p_fix.npz at mask_thr 0.25, knobs read from the npz itself
# (alpha=0.3, kappa x1.6). Confirmed against phase4/preds/knobbed_s0.json's own meta.
EM_NPZ = "/p4/out/em_model_4p_fix.npz"
FEAT_4P_TEST = "/p4/feat_4p_test"
MASK_THR = 0.25

OUT = "/vol/out"

app = modal.App(APP)
p4vol = modal.Volume.from_name(PHASE4_VOL)
vol = modal.Volume.from_name(BAKE_VOL, create_if_missing=True)
hfvol = modal.Volume.from_name(HFDATA_VOL)
hf_secret = modal.Secret.from_name("huggingface")


STUBS = os.path.join(PH4, "stubs")
# `evaluate.py` reaches into dapt (backbone/decode) and `em.py` into three sibling packages
# that do not exist on Modal, so the same file set phase4_modal.py mounts is required here --
# transformers included, since dapt.backbone imports it at module scope.
STUB_FILES = {"boxinst": ("__init__.py", "cache_feats.py"),
              "boxinst_commonality": ("__init__.py", "em.py"),
              "boxinst_tcd": ("__init__.py", "build_canopy.py", "cache.py", "prepare.py")}


def _with_core(img):
    """Add the scoring core + GT to any arm's image, so all three score identically."""
    for rel in ("dapt/__init__.py", "dapt/backbone.py", "dapt/targets.py",
                "dapt/decode.py", "dapt/eval.py", "dapt/head.py"):
        img = img.add_local_file(os.path.join(REPO, rel), f"{P}/{rel}")
    for rel in ("__init__.py", "detector.py", "train_detector_tiles.py", "evaluate.py",
                "em.py", "prepare_test.py", "cache_test.py", "cache_train_tiles.py",
                "test_gt.json", "train_tiles_gt.json"):
        img = img.add_local_file(os.path.join(PKG, rel), f"{PKG_R}/{rel}")
    img = img.add_local_file(os.path.join(PKG, "vault", "em_model.npz"),
                             f"{PKG_R}/vault/em_model.npz")
    for pkg, files in STUB_FILES.items():
        for f in files:
            img = img.add_local_file(os.path.join(STUBS, pkg, f), f"{P}/{pkg}/{f}")
    img = img.add_local_file(os.path.join(HERE, "gtbox_lib.py"), f"{BSB_R}/gtbox_lib.py")
    img = img.add_local_file(os.path.join(HERE, "__init__.py"), f"{BSB_R}/__init__.py")
    return img


base_pip = ("numpy==2.2.6", "pillow", "pycocotools", "contourpy",
            "transformers==4.57.1")

lace_image = _with_core(
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.1", "torchvision==0.27.1", *base_pip))

sam3_image = _with_core(
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.12.1", "torchvision==0.27.1", "numpy==1.26.4", "pillow",
                 "datasets==4.0.0", "einops", "pycocotools", "open_clip_torch", "psutil", "contourpy",
                 "transformers==4.57.1")
    .pip_install(f"git+https://github.com/facebookresearch/sam3.git@{SAM3_COMMIT}")
    .env({"HF_HOME": "/hfdata/hf_cache", "HF_HUB_OFFLINE": "0",
          "HF_DATASETS_OFFLINE": "1"})
    .add_local_file(os.path.join(SAMD, "sam3_eval_tcd.py"), "/root/sam3_eval_tcd.py")
    .add_local_file(os.path.join(PH4, "manifest.json"), "/root/manifest.json"))

samft_image = _with_core(
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel")
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("numpy<2", "pillow", "pycocotools", "datasets==4.0.0",
                 "huggingface_hub", "timm", "transformers>=5.12", "contourpy")
    .env({"HF_HOME": "/hfdata/hf_cache", "HF_HUB_OFFLINE": "0",
          "HF_DATASETS_OFFLINE": "1"})
    .add_local_file(os.path.join(PH4, "manifest.json"), "/root/manifest.json"))


def _setup():
    import sys
    if P not in sys.path:
        sys.path.insert(0, P)


def _prompts(limit=0, require=None):
    """-> (tids, gt, {tid: (boxes_2048, valid)}). Identical for every arm.

    Deliberately returns NO masks. An earlier version pre-rasterised every tile's GT up front,
    which is 6.7 GB at 512 and ~108 GB at 2048 -- an instant OOM. GT masks are built one tile at
    a time inside each arm instead, so peak memory is a single tile regardless of resolution.

    `require(tid) -> bool` drops tiles an arm cannot reach (e.g. no cached feature file), so the
    arm is never silently credited with tiles it skipped."""
    _setup()
    from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as L
    gt = json.load(open(f"{PKG_R}/test_gt.json"))
    tids = [t for t in sorted(gt) if gt[t]["trees"]]
    if require:
        tids = [t for t in tids if require(t)]
    if limit:
        tids = tids[:limit]
    pr = {}
    for t in tids:
        polys = gt[t]["trees"]
        _, v = L.gt_masks_and_valid(polys, res=L.RES)      # cheap: valid is pinned to 512
        pr[t] = (L.gt_boxes_2048(polys), v)
    n = sum(len(v[1]) for v in pr.values())
    nv = sum(int(v[1].sum()) for v in pr.values())
    print(f"[gtbox] {len(tids)} tiles · {n} crowns · {nv} valid prompts", flush=True)
    return tids, gt, pr


def _score_tile(L, np, pm, gt_rles, valid):
    """One tile's per-crown IoUs + double-assignment, plus the masks for persistence.

    The predictions are encoded to RLE ONCE and that encoding serves both scoring and
    persistence: at 2048 the RLE intersection runs in C rather than over 4.2M-element boolean
    arrays, which is what keeps the run affordable. Verified bit-identical to the dense path.
    Masks are persisted so resolution, ranking and canopy questions are all free re-scores --
    two experiments here have already needed a full A100 pass because masks were discarded."""
    p_rles = L.to_rle(pm)
    return {"iou": np.round(L.per_crown_iou_rle(p_rles, gt_rles), 5).tolist(),
            "valid": valid.tolist(), "double": L.double_assignment_rle(p_rles),
            "masks_rle": [{"size": list(r["size"]), "counts": r["counts"].decode("ascii")}
                          for r in p_rles]}


def _finish(arm, per_tile, meta, res_px):
    """Persist per-crown IoUs (and the masks themselves) and print the arm's headline."""
    import numpy as np
    _setup()
    from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as L
    ious = np.concatenate([np.asarray(v["iou"])[np.asarray(v["valid"], bool)]
                           for v in per_tile.values() if len(v["iou"])])
    dbl = [v["double"] for v in per_tile.values() if v["double"] is not None]
    summ = L.summarise(ious, {"mean_double_assignment": round(float(np.mean(dbl)), 4)
                              if dbl else None})
    masks = {t: per_tile[t].pop("masks_rle") for t in per_tile}
    res = {"arm": arm, "n_tiles": len(per_tile), "res": res_px, **meta,
           "summary": summ, "per_tile": per_tile}
    os.makedirs(OUT, exist_ok=True)
    tag = f"{arm}_r{res_px}"
    json.dump(res, open(os.path.join(OUT, f"gtbox_{tag}.json"), "w"))
    mp = os.path.join(OUT, f"gtbox_masks_{tag}.json")
    json.dump({"arm": arm, "res": res_px, "masks_rle": masks}, open(mp, "w"))
    vol.commit()
    print(f"\n[gtbox] {arm} @{res_px}: mean IoU {summ['mean_iou']} · {summ['rates']} · "
          f"double {summ['mean_double_assignment']} · n={summ['n']}\n"
          f"        -> gtbox_{tag}.json  + masks {mp}", flush=True)
    return {k: v for k, v in res.items() if k != "per_tile"}


# ------------------------------------------------------------------ arm 1: LACE
@app.function(gpu="A100", image=lace_image, volumes={"/p4": p4vol, "/vol": vol},
              timeout=6 * 3600, cpu=8, memory=65536)
def eval_lace(limit: int = 0, res: int = 512):
    """The shipped commonality-EM masker, prompted with GT boxes instead of detections."""
    import numpy as np
    import torch
    _setup()
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as L
    assert os.path.exists(EM_NPZ), f"{EM_NPZ} missing"

    tids, gt, pr = _prompts(limit, require=lambda t: os.path.exists(
        os.path.join(FEAT_4P_TEST, t + ".npy")))
    masker = E.TCDMasker(EM_NPZ)
    print(f"[gtbox/lace] masker s={masker.s} mask_thr={MASK_THR} res={res} "
          f"(knobs from the npz)", flush=True)

    per_tile = {}
    for k, t in enumerate(tids):
        boxes, valid = pr[t]
        gt_rles, _ = L.gt_rles_and_valid(gt[t]["trees"], res=res)
        feat = np.load(os.path.join(FEAT_4P_TEST, t + ".npy")).astype(np.float32)
        g = feat.shape[-1]
        zn = masker.project(feat)
        # prior_weight/kappa_scale left None -> the npz's own stored calibration, which is what
        # ships. Do NOT pass the predicted-box knobs explicitly here.
        pm = E.pred_instance_masks(masker, zn, g, boxes, res=res, scale=2048.0 / res,
                                   mask_thr=MASK_THR)
        per_tile[t] = _score_tile(L, np, pm, gt_rles, valid)
        del pm, gt_rles, feat, zn
        if (k + 1) % 50 == 0 or k + 1 == len(tids):
            print(f"  [lace] {k+1}/{len(tids)} tiles", flush=True)
    del torch
    return _finish("lace", per_tile, {"em": os.path.basename(EM_NPZ),
                                      "mask_thr": MASK_THR, "knobs": "from npz"}, res)


# ------------------------------------------------------------ arm 2: SAM 3 base
@app.function(gpu="A100", image=sam3_image, volumes={"/vol": vol, "/hfdata": hfvol},
              timeout=6 * 3600, cpu=8, memory=65536, secrets=[hf_secret])
def eval_sam3_base(limit: int = 0, chunk: int = 64, res: int = 512):
    """Zero-shot SAM 3, box-prompted at the native 2048 tile (no downscale handicap)."""
    import sys
    import time

    import numpy as np
    import torch
    sys.path.insert(0, "/root")
    import sam3_eval_tcd as EV
    from datasets import load_dataset
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    _setup()
    from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as L
    assert torch.cuda.is_available(), "no CUDA"

    manifest = json.load(open("/root/manifest.json"))["feat_test"]
    tids, gt, pr = _prompts(limit, require=lambda t: t in manifest)
    ds = load_dataset("restor/tcd", split="test")
    idx = {int(i): n for n, i in enumerate(ds["image_id"])}

    t0 = time.time()
    model = build_sam3_image_model(enable_inst_interactivity=True)
    EV.model_g, EV.processor_g = model, Sam3Processor(model)
    print(f"[gtbox/sam3] model ready in {(time.time()-t0)/60:.1f}min · res={res}", flush=True)

    per_tile = {}
    for k, t in enumerate(tids):
        boxes, valid = pr[t]
        gt_rles, _ = L.gt_rles_and_valid(gt[t]["trees"], res=res)
        rgb = ds[idx[int(manifest[t]["image_id"])]]["image"].convert("RGB")
        pm, _sc = EV.sam_masks_for_tile(EV.model_g, EV.processor_g, rgb, boxes, chunk, res=res)
        per_tile[t] = _score_tile(L, np, pm, gt_rles, valid)
        del pm, gt_rles
        if (k + 1) % 25 == 0 or k + 1 == len(tids):
            print(f"  [sam3] {k+1}/{len(tids)} tiles", flush=True)
    return _finish("sam3_base", per_tile,
                   {"ckpt": SAM_BASE, "commit": SAM3_COMMIT, "prompt_res": 2048}, res)


# -------------------------------------------------------------- arm 3: SAM 3 FT
@app.function(gpu="A100", image=samft_image, volumes={"/vol": vol, "/hfdata": hfvol},
              timeout=6 * 3600, cpu=8, memory=65536, secrets=[hf_secret])
def eval_sam3_ft(limit: int = 0, box_batch: int = 64, sam_tile: int = SAM_FT_TILE,
                 res: int = 512):
    """CanopyRS's SelvaMask-fine-tuned SAM 3, at the tile size they fine-tuned at.

    NOTE the two resolutions are independent here. `sam_tile` (1777) is what the MODEL sees --
    their wrapper's default and the size they fine-tuned at, so it stays fixed. `res` is what
    the mask is SCORED on. At res=2048 the model's 1777 output is upsampled, so this arm is
    interpolated rather than native at that scoring resolution; that is a property of their
    deployment choice, not a handicap we impose."""
    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    _setup()
    from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as L
    assert torch.cuda.is_available(), "no CUDA"

    proc = Sam3TrackerProcessor.from_pretrained(SAM_BASE)
    sam = Sam3TrackerModel.from_pretrained(SAM_BASE).to("cuda").eval()
    sd = torch.load(hf_hub_download(SAM_FT_REPO, SAM_FT_FILE), map_location="cpu")
    sd = sd.get("model_state_dict", sd)
    exp, got = set(sam.state_dict()), set(sd)
    # CanopyRS's own loader calls load_state_dict(strict=False) then prints success
    # unconditionally -- it would report a clean load having matched nothing. Assert instead.
    if len(exp & got) != len(exp) or (got - exp):
        raise RuntimeError(f"FT SAM 3 mismatch: matched {len(exp & got)}/{len(exp)}, "
                           f"{len(got - exp)} unexpected. Refusing to run -- strict=False "
                           f"would silently leave the model un-fine-tuned.")
    sam.load_state_dict(sd, strict=False)
    print(f"[gtbox/sam3ft] {len(exp)}/{len(exp)} tensors matched · model tile {sam_tile} "
          f"· scored at {res}", flush=True)

    manifest = json.load(open("/root/manifest.json"))["feat_test"]
    tids, gt, pr = _prompts(limit, require=lambda t: t in manifest)
    ds = load_dataset("restor/tcd", split="test")
    idx = {int(i): n for n, i in enumerate(ds["image_id"])}
    f = sam_tile / 2048.0

    per_tile = {}
    for k, t in enumerate(tids):
        boxes, valid = pr[t]
        gt_rles, _ = L.gt_rles_and_valid(gt[t]["trees"], res=res)
        rgb = ds[idx[int(manifest[t]["image_id"])]]["image"].convert("RGB")
        img = rgb.resize((sam_tile, sam_tile), Image.BILINEAR) if rgb.size[0] != sam_tile \
            else rgb
        inputs = proc(images=img, input_boxes=[(np.asarray(boxes) * f).tolist()],
                      return_tensors="pt")
        with torch.no_grad():                      # encode the tile once, reuse per batch
            emb = sam.get_image_embeddings(inputs["pixel_values"].to("cuda"))
        allb, osz = inputs["input_boxes"], inputs["original_sizes"]
        masks = []
        for i0 in range(0, len(boxes), box_batch):
            with torch.no_grad():
                so = sam(input_boxes=allb[:, i0:i0 + box_batch].to("cuda"),
                         image_embeddings=emb, multimask_output=False)
            m = proc.post_process_masks(so.pred_masks.cpu(), osz)[0]
            for mm in m.squeeze(1).numpy().astype(np.uint8):
                masks.append(np.asarray(Image.fromarray(mm * 255).resize(
                    (res, res), Image.BILINEAR)) >= 128)
        pm = np.asarray(masks) if masks else np.zeros((0, res, res), bool)
        per_tile[t] = _score_tile(L, np, pm, gt_rles, valid)
        del pm, gt_rles, masks
        if (k + 1) % 25 == 0 or k + 1 == len(tids):
            print(f"  [sam3ft] {k+1}/{len(tids)} tiles", flush=True)
    return _finish("sam3_ft", per_tile,
                   {"ckpt": SAM_FT_REPO, "model_tile": sam_tile}, res)
