"""Modal A100 app: SelvaBox (DINO-Swin-L, SelvaMask-FT) -> SAM 3 (SelvaMask-FT) on the
OAM-TCD 439, dumped into OUR preds schema for score_coco.py. See PROTOCOL_439.md.

WHY NOT JUST INSTALL CanopyRS
    Their install needs GDAL 3.6.2 via mamba plus a detrex git submodule, and their runtime
    threads everything through `geodataset` rasters. We work on fixed-pixel RGB tiles, so the
    geo layer is dead weight. This drives detrex/detectron2 and HF transformers directly --
    exactly the bypass detectree2_baseline/dt2_recipe.py already uses for DetecTree2 ("the
    detectree2 package itself is not needed (its geo/GDAL layer is bypassed)").

WHAT IS REPRODUCED FROM THEIR RECIPE (train_detrex.py:408, augmentation.py:542, sam3.py)
    detector    projects/dino/configs/dino-swin/dino_swin_large_384_5scale_36ep.py,
                num_classes=1, weights = dino-swin-l-384-multi-NQOS-selvamask-FT.
    resize      DeterministicResizeWithinRange(1024, 1777) at test time. Both tilings we run
                (1024 and 1777) already sit inside that range, so nothing is resampled.
    segmenter   transformers Sam3TrackerModel + Sam3TrackerProcessor from facebook/sam3, then
                their FT state dict. VERIFIED locally: 685/685 tensors match exactly.
                Their own loader uses strict=False and prints success unconditionally, so
                `_load_ft_sam3` asserts the matched count instead of trusting the message.
    SAM input   their wrapper resizes each tile to target_tile_size (default 1777) before SAM.
                Kept -- so "native GSD" describes the DETECTOR, not the segmenter.

PIPELINE ORDER matches theirs (detector -> aggregator -> segmenter): detect per subtile,
map boxes to 2048-tile coords, IoU-NMS 0.7 across the whole tile, THEN prompt SAM once per
2048 tile with the survivors. Segmenting after aggregation avoids re-segmenting the ~4x
duplicate boxes that overlapping subtiles produce, and avoids stitching masks at all.

NO EDGE-BAND CULL. Their aggregator's edge_band_buffer_percentage=0.05 is absent from the
path that produced their published tile-level AP (that is computed on the model component's
raw COCO, pre-aggregator), and on a 2048 tile it would delete every crown touching a 51 px
frame -- 16.0% of GT crowns, capping recall at 0.84. See PROTOCOL_439.md.

Run:
    modal run selvabox_modal.py::build_check              # detrex + both checkpoints, no data
    modal run selvabox_modal.py::predict --limit 4        # smoke
    modal run selvabox_modal.py::predict                  # full 439 (benchmark tiling)
    modal run selvabox_modal.py::predict --tile-px 1777 --ground-res 0.07 --tag _fast
"""
import json
import os

import modal

APP = "tcd04-selvabox-baseline"
OUT_VOL = "tcd04-baselines-vol"
HFDATA_VOL = "canopyai-deepforest-data"

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)

DET_REPO = "CanopyRS/dino-swin-l-384-multi-NQOS-selvamask-FT"
DET_FILE = "model_best.pth"
SAM_REPO = "CanopyRS/sam3-multi-selvabox-selvamask-FT"
SAM_FILE = "sam3_selvamask_ft_model_best.pt"
SAM_BASE = "facebook/sam3"
DETREX_CFG = "dino/configs/dino-swin/dino_swin_large_384_5scale_36ep.py"

RES = 512                    # scoring raster, identical to every other row
SAM_TILE = 1777              # their target_tile_size default
NMS_IOU = 0.7                # their aggregator's nms_threshold for detection boxes
SCORE_FLOOR = 0.05           # frozen protocol floor (NOT their 0.4/0.5 deployment floors)
# Keep up to CanopyRS's OWN budget (their OamTcdDataset tile_level_eval_maxDets=400 per
# 1024^2 subtile = 1600 per 2048^2-equivalent area, 2.7x our 600) so the protocol cap stays
# in the SCORER, not baked into the artifact. Scoring at 600 vs 1600 is then a re-score,
# not a re-run. This matters far more for SelvaBox than for other rows: maxDets is applied
# BEFORE canopy-ignore, and ~83% of its detections fall in canopy, so a tight cap keeps
# mostly-ignored detections and starves the arm of scoreable ones.

app = modal.App(APP)
vol = modal.Volume.from_name(OUT_VOL, create_if_missing=True)
hfvol = modal.Volume.from_name(HFDATA_VOL)
hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel")
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
    .apt_install("git", "g++", "ninja-build", "libgl1", "libglib2.0-0")
    .env({"TORCH_CUDA_ARCH_LIST": "8.0", "FORCE_CUDA": "1"})     # A100 sm_80
    .pip_install("numpy<2", "opencv-python-headless", "pycocotools", "pillow",
                 "datasets==4.0.0", "huggingface_hub", "timm",
                 "transformers>=5.12", "scipy")
    .run_commands(
        "git clone --recursive https://github.com/IDEA-Research/detrex.git /opt/detrex",
        "pip install --no-build-isolation -e /opt/detrex/detectron2",
        "pip install --no-build-isolation -e /opt/detrex",
    )
    .env({"HF_HOME": "/hfdata/hf_cache", "HF_HUB_OFFLINE": "0",
          "HF_DATASETS_OFFLINE": "1", "DETREX_ROOT": "/opt/detrex"})
    .add_local_file(os.path.join(PKG, "modal_tcd_multiseed", "phase4", "manifest.json"),
                    "/root/manifest.json")
    .add_local_file(os.path.join(PKG, "test_gt.json"), "/root/test_gt.json")
)

OUT = "/vol/out"


# ---------------------------------------------------------------- model builders
def _build_detector():
    """detrex DINO-Swin-L with their num_classes=1 head + the SelvaMask-FT weights."""
    import sys
    import torch
    sys.path.insert(0, "/opt/detrex")
    from detectron2.config import LazyConfig, instantiate
    from detrex.checkpoint import DetectionCheckpointer
    from huggingface_hub import hf_hub_download

    # detrex checkpoints embed omegaconf objects, which torch>=2.6 rejects under its new
    # weights_only=True default ("Unsupported global: omegaconf.listconfig.ListConfig").
    # CanopyRS patches torch.load the same way for the same reason
    # (detectron2_infer.py:7-16, "Patch torch.load for PyTorch 2.6+"). Scoped to this call:
    # the file is their own published checkpoint, already verified by hash on download.
    _real_load = torch.load

    def _load_compat(f, *a, **kw):
        kw.setdefault("weights_only", False)
        return _real_load(f, *a, **kw)

    ckpt = hf_hub_download(DET_REPO, DET_FILE)
    cfg = LazyConfig.load(os.path.join("/opt/detrex/projects", DETREX_CFG))
    if hasattr(cfg.model, "num_classes"):
        cfg.model.num_classes = 1                 # their detector.yaml: single 'tree' class
    model = instantiate(cfg.model).to("cuda").eval()
    torch.load = _load_compat
    try:
        DetectionCheckpointer(model).load(ckpt)
    finally:
        torch.load = _real_load          # never leave the global patched for SAM 3's load
    return model, cfg


def _load_ft_sam3():
    """Sam3TrackerModel + their FT weights, with the key overlap ASSERTED.

    CanopyRS (sam3.py:93) calls load_state_dict(strict=False) then prints
    '✓ Fine-tuned weights loaded successfully!' unconditionally -- it reports success even
    if zero keys matched. Verified offline that this checkpoint is 685/685 exact against
    Sam3TrackerModel, so anything less than a full match here is a real regression."""
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor

    proc = Sam3TrackerProcessor.from_pretrained(SAM_BASE)
    model = Sam3TrackerModel.from_pretrained(SAM_BASE).to("cuda").eval()
    sd = torch.load(hf_hub_download(SAM_REPO, SAM_FILE), map_location="cpu")
    sd = sd.get("model_state_dict", sd)
    exp = set(model.state_dict().keys())
    got = set(sd.keys())
    matched = len(exp & got)
    print(f"[selvabox] SAM3 FT keys: matched {matched}/{len(exp)}  "
          f"unexpected {len(got - exp)}  missing {len(exp - got)}", flush=True)
    if matched != len(exp) or (got - exp):
        raise RuntimeError(
            f"FT SAM 3 state dict does not fully match Sam3TrackerModel "
            f"(matched {matched}/{len(exp)}, {len(got - exp)} unexpected). Refusing to run: "
            f"strict=False would silently leave the model un-fine-tuned and every number "
            f"below would describe base SAM 3, not theirs.")
    model.load_state_dict(sd, strict=False)
    return model, proc


@app.function(gpu="A100", image=image, volumes={"/hfdata": hfvol}, timeout=3600,
              cpu=8, memory=32768, secrets=[hf_secret])
def build_check():
    """Cheapest possible failure: build both models and load both checkpoints, no data."""
    import torch
    det, _ = _build_detector()
    n = sum(p.numel() for p in det.parameters())
    print(f"[selvabox] detector built: {n/1e6:.0f}M params", flush=True)
    sam, _ = _load_ft_sam3()
    m = sum(p.numel() for p in sam.parameters())
    print(f"[selvabox] SAM3 FT built: {m/1e6:.0f}M params", flush=True)
    print("[selvabox] BUILD OK", flush=True)
    return {"detector_params_m": round(n / 1e6), "sam3_params_m": round(m / 1e6)}


def _nms(boxes, scores, thr):
    """Plain IoU-NMS -- CanopyRS aggregator's nms_algorithm='iou', nms_threshold 0.7."""
    import numpy as np
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = rest[inter / (a[i] + a[rest] - inter + 1e-9) <= thr]
    return np.asarray(keep, int)


@app.function(gpu="A100", image=image, volumes={"/vol": vol, "/hfdata": hfvol},
              timeout=6 * 3600, cpu=8, memory=65536, secrets=[hf_secret])
def predict(limit: int = 0, tile_px: int = 1024, overlap: float = 0.5,
            ground_res: float = 0.10, tag: str = "", box_batch: int = 64,
            keep_dets: int = 1600):
    """SelvaBox detector over subtiles -> aggregate -> SAM 3 FT masks -> our preds schema.

    Defaults are THEIR OAM-TCD benchmark tiling (1024 @ 0.5 overlap, native 0.10 m/px), which
    is the config that produced their published OAM-TCD numbers. `--ground-res 0.07
    --tile-px 1777` reproduces their `fast` deployment preset instead (same 9 subtiles/tile).
    """
    import json as J
    import numpy as np
    import torch
    from PIL import Image
    from datasets import load_dataset
    from pycocotools import mask as maskUtils

    KEEP_DETS = keep_dets
    det, cfg = _build_detector()
    sam, proc = _load_ft_sam3()
    pm = getattr(det, "pixel_mean", None)
    print(f"[selvabox] detector pixel_mean={None if pm is None else pm.flatten().tolist()} "
          f"input_format={getattr(det, 'input_format', '?')}", flush=True)

    gt = J.load(open("/root/test_gt.json"))
    manifest = J.load(open("/root/manifest.json"))["feat_test"]
    ds = load_dataset("restor/tcd", split="test")
    idx = {int(i): k for k, i in enumerate(ds["image_id"])}
    tids = [t for t in sorted(gt) if t in manifest]
    if limit:
        tids = tids[:limit]

    scale = 0.10 / ground_res                    # 2048 px @0.1 m/px -> this many px
    full = int(round(2048 * scale))
    step = int(round(tile_px * (1 - overlap)))
    origins = sorted(set(min(i * step, full - tile_px)
                         for i in range(int(np.ceil((full - tile_px) / step)) + 1)))
    print(f"[selvabox] {len(tids)} tiles · resample {full}px @{ground_res} m/px · "
          f"{tile_px}px @{overlap} overlap · {len(origins)**2} subtiles/tile", flush=True)

    # RESUMABLE: per-tile results are checkpointed to the volume every CKPT_EVERY tiles, so a
    # crash costs only the tiles since the last checkpoint rather than the whole run. A previous
    # partial is loaded and its tiles skipped. (This app already died once at ~1500 bytes of log
    # on an unguarded empty-detection tile, losing everything.)
    CKPT_EVERY = 20
    os.makedirs(OUT, exist_ok=True)
    part_fp = os.path.join(OUT, f"partial_selvabox{tag or '_run'}.json")
    out = {}
    if os.path.exists(part_fp):
        try:
            out = J.load(open(part_fp))
            print(f"[selvabox] resuming: {len(out)} tiles already done", flush=True)
        except Exception as e:
            print(f"[selvabox] partial unreadable ({e}); starting fresh", flush=True)
    todo = [t for t in tids if t not in out]
    tot = sum(len(v["masks_rle"]) for v in out.values())
    print(f"[selvabox] {len(todo)} tiles to process ({len(out)} skipped)", flush=True)

    def _ckpt():
        J.dump(out, open(part_fp, "w"))
        vol.commit()

    for n, tid in enumerate(todo):
        rgb = ds[idx[int(manifest[tid]["image_id"])]]["image"].convert("RGB")
        if full != 2048:
            rgb = rgb.resize((full, full), Image.BILINEAR)
        arr = np.asarray(rgb)
        B, S = [], []
        for oy in origins:
            for ox in origins:
                sub = arr[oy:oy + tile_px, ox:ox + tile_px]
                inp = {"image": torch.as_tensor(sub.transpose(2, 0, 1).copy()),
                       "height": tile_px, "width": tile_px}
                with torch.no_grad():
                    o = det([inp])[0]["instances"].to("cpu")
                if not len(o):
                    continue
                b = o.pred_boxes.tensor.numpy() + np.array([ox, oy, ox, oy], np.float32)
                B.append(b / scale)                       # -> 2048-tile coords
                S.append(o.scores.numpy())
        if not B:
            out[tid] = {"boxes_2048": [], "scores": [], "masks_rle": []}
            continue
        B = np.concatenate(B); S = np.concatenate(S)
        k = S >= SCORE_FLOOR
        B, S = B[k], S[k]
        k = _nms(B, S, NMS_IOU)                            # aggregate BEFORE segmenting
        B, S = B[k], S[k]
        if len(S) > KEEP_DETS:
            k = np.argsort(-S)[:KEEP_DETS]
            k.sort()
            B, S = B[k], S[k]
        if len(B) == 0:
            # Everything was filtered out by the score floor or NMS. The earlier guard only
            # covers "no subtile produced anything"; this covers "produced only sub-floor
            # detections". Sam3TrackerProcessor needs 3-level nested boxes and an empty
            # array yields [[]] (2 levels), which it rejects outright.
            out[tid] = {"boxes_2048": [], "scores": [], "masks_rle": []}
            continue

        # SAM 3 on the whole 2048 tile at their target_tile_size, prompted with the survivors
        sam_img = rgb.resize((SAM_TILE, SAM_TILE), Image.BILINEAR) if full != SAM_TILE \
            else rgb
        f = SAM_TILE / 2048.0
        inputs = proc(images=sam_img, input_boxes=[(B * f).tolist()],
                      return_tensors="pt")
        # Encode the 1777^2 image ONCE and reuse across box batches. Calling the processor
        # per batch re-ran the vision encoder every time (~19 redundant encodings/tile).
        with torch.no_grad():
            emb = sam.get_image_embeddings(inputs["pixel_values"].to("cuda"))
        all_boxes = inputs["input_boxes"]
        osz = inputs["original_sizes"]
        rles = []
        for i0 in range(0, len(B), box_batch):
            with torch.no_grad():
                so = sam(input_boxes=all_boxes[:, i0:i0 + box_batch].to("cuda"),
                         image_embeddings=emb, multimask_output=False)
            m = proc.post_process_masks(so.pred_masks.cpu(), osz)[0]
            m = m.squeeze(1).numpy().astype(np.uint8)      # (n, SAM_TILE, SAM_TILE)
            for mm in m:
                small = np.asarray(Image.fromarray(mm * 255).resize((RES, RES),
                                                                    Image.BILINEAR)) >= 128
                r = maskUtils.encode(np.asfortranarray(small.astype(np.uint8)))
                rles.append({"size": [RES, RES], "counts": r["counts"].decode("ascii")})
        out[tid] = {"boxes_2048": np.round(B, 2).tolist(),
                    "scores": np.round(S, 4).tolist(), "masks_rle": rles}
        tot += len(rles)
        if (n + 1) % 5 == 0 or n + 1 == len(todo):
            print(f"  {n+1}/{len(todo)} tiles ({len(out)}/{len(tids)} total), "
                  f"{tot} crowns", flush=True)
        if (n + 1) % CKPT_EVERY == 0:
            _ckpt()
            print(f"  [ckpt] {len(out)} tiles saved", flush=True)

    res = {"meta": {"model": f"SelvaBox DINO-Swin-L (NQOS-selvamask-FT) -> SAM 3 "
                             f"(selvamask-FT), {tile_px}px @{overlap} ovl @{ground_res} m/px, "
                             f"IoU-NMS {NMS_IOU}, no edge-band cull",
                    "det_repo": DET_REPO, "sam_repo": SAM_REPO, "mask_res": RES,
                    "sam_tile": SAM_TILE, "tile_px": tile_px, "overlap": overlap,
                    "ground_res": ground_res, "score_floor": SCORE_FLOOR, "keep_dets": keep_dets,
                    "maxdets_note": "cap left to the scorer; CanopyRS own budget = 1600/2048^2-equiv",
                    "n_tiles": len(out), "n_crowns": tot},
            "preds": out}
    _ckpt()
    fp = os.path.join(OUT, f"preds_selvabox{tag or ('_smoke' if limit else '')}.json")
    J.dump(res, open(fp, "w"))
    vol.commit()
    print(f"[selvabox] saved {fp}: {len(out)} tiles, {tot} crowns", flush=True)
    return {"preds": fp, "n_tiles": len(out), "n_crowns": tot}
