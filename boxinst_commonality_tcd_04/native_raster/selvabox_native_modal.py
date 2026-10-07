"""SelvaBox (DINO-Swin-L, SelvaMask-FT) -> SAM 3 (SelvaMask-FT) on the OAM-TCD 439, persisting
masks at every resolution the pipeline touches, so the row never needs a GPU again.

Derived from `selvabox_baseline/selvabox_modal.py::predict` (the published `_bench` arm:
1024 px subtiles @0.5 overlap, native 0.10 m/px, IoU-NMS 0.7, floor 0.05, keep_dets 1600).
Inference is IDENTICAL; only persistence changes:

  masks_rle_1777   SAM 3's own output raster (their target_tile_size) -- the true native
  masks_rle_2048   1777 -> 2048 PIL BILINEAR >= 128 (a 1.15x UPSAMPLE; disclose)
  masks_rle_512    1777 -> 512  PIL BILINEAR >= 128, the exact published path -> the gate
  raw_dets         per-subtile detector boxes+scores BEFORE floor / NMS / keep_dets
  sam_scores       SAM 3's own per-mask IoU prediction

Robustness: one JSON per tile under OUT_DIR/tiles (atomic rename), vol.commit() every
CKPT_EVERY tiles, resume by listing the dir, run.log on the volume. OUT_DIR is NEW (verified
absent 2026-09-23). Launch via detach.py deploy + spawn -- NEVER with an attached client, which
this harness kills and Modal then cancels the call.

    python detach.py deploy selvabox_native_modal.py
    python detach.py spawn tcd04-selvabox-native predict
    modal run selvabox_native_modal.py::assemble          # CPU
"""
import json
import os

import modal

APP = "tcd04-selvabox-native"
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

NATIVE = 2048
LOW = 512
SAM_TILE = 1777
NMS_IOU = 0.7
SCORE_FLOOR = 0.05
TILE_PX, OVERLAP, GROUND_RES, KEEP_DETS, BOX_BATCH = 1024, 0.5, 0.10, 1600, 64   # == _bench
CKPT_EVERY = 5

app = modal.App(APP)
vol = modal.Volume.from_name(OUT_VOL)
hfvol = modal.Volume.from_name(HFDATA_VOL)
hf_secret = modal.Secret.from_name("huggingface")

# image: verbatim from selvabox_baseline/selvabox_modal.py so the build is cached
image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel")
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
    .apt_install("git", "g++", "ninja-build", "libgl1", "libglib2.0-0")
    .env({"TORCH_CUDA_ARCH_LIST": "8.0", "FORCE_CUDA": "1"})
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

OUT_DIR = f"/vol/out/native_raster/selvabox_bench_r{NATIVE}"
TILES = os.path.join(OUT_DIR, "tiles")
LOG = os.path.join(OUT_DIR, "run.log")


def _log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as fh:
        fh.write(msg + "\n")


def _env_versions():
    import numpy, PIL, torch, transformers
    return {"torch": torch.__version__, "numpy": numpy.__version__, "pillow": PIL.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


# ---- model builders: verbatim from selvabox_modal.py ------------------------------------
def _build_detector():
    import sys
    import torch
    sys.path.insert(0, "/opt/detrex")
    from detectron2.config import LazyConfig, instantiate
    from detrex.checkpoint import DetectionCheckpointer
    from huggingface_hub import hf_hub_download
    _real_load = torch.load

    def _load_compat(f, *a, **kw):
        kw.setdefault("weights_only", False)
        return _real_load(f, *a, **kw)

    ckpt = hf_hub_download(DET_REPO, DET_FILE)
    cfg = LazyConfig.load(os.path.join("/opt/detrex/projects", DETREX_CFG))
    if hasattr(cfg.model, "num_classes"):
        cfg.model.num_classes = 1
    model = instantiate(cfg.model).to("cuda").eval()
    torch.load = _load_compat
    try:
        DetectionCheckpointer(model).load(ckpt)
    finally:
        torch.load = _real_load
    return model, cfg


def _load_ft_sam3():
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    proc = Sam3TrackerProcessor.from_pretrained(SAM_BASE)
    model = Sam3TrackerModel.from_pretrained(SAM_BASE).to("cuda").eval()
    sd = torch.load(hf_hub_download(SAM_REPO, SAM_FILE), map_location="cpu")
    sd = sd.get("model_state_dict", sd)
    exp = set(model.state_dict().keys()); got = set(sd.keys())
    matched = len(exp & got)
    if matched != len(exp) or (got - exp):
        raise RuntimeError(f"FT SAM 3 state dict mismatch: matched {matched}/{len(exp)}, "
                           f"{len(got - exp)} unexpected")
    model.load_state_dict(sd, strict=False)
    return model, proc


def _nms(boxes, scores, thr):
    import numpy as np
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = rest[inter / (a[i] + a[rest] - inter + 1e-9) <= thr]
    return np.asarray(keep, int)
# -------------------------------------------------------------------------------------------


@app.function(gpu="A100", image=image, volumes={"/vol": vol, "/hfdata": hfvol},
              timeout=8 * 3600, cpu=8, memory=65536, secrets=[hf_secret])
def predict(limit: int = 0):
    import numpy as np
    import torch
    from PIL import Image
    from datasets import load_dataset
    from pycocotools import mask as maskUtils

    os.makedirs(TILES, exist_ok=True)
    _log(f"[selvabox-native] start OUT_DIR={OUT_DIR} env={json.dumps(_env_versions())} "
         f"tile_px={TILE_PX} overlap={OVERLAP} ground_res={GROUND_RES} keep_dets={KEEP_DETS} "
         f"nms={NMS_IOU} floor={SCORE_FLOOR} sam_tile={SAM_TILE}")
    det, cfg = _build_detector()
    sam, proc = _load_ft_sam3()

    gt = json.load(open("/root/test_gt.json"))
    manifest = json.load(open("/root/manifest.json"))["feat_test"]
    ds = load_dataset("restor/tcd", split="test")
    idx = {int(i): k for k, i in enumerate(ds["image_id"])}
    tids = [t for t in sorted(gt) if t in manifest]
    if limit:
        tids = tids[:limit]

    scale = 0.10 / GROUND_RES
    full = int(round(2048 * scale))
    step = int(round(TILE_PX * (1 - OVERLAP)))
    origins = sorted(set(min(i * step, full - TILE_PX)
                         for i in range(int(np.ceil((full - TILE_PX) / step)) + 1)))
    done = {f[:-5] for f in os.listdir(TILES) if f.endswith(".json")}
    todo = [t for t in tids if t not in done]
    _log(f"[selvabox-native] {len(tids)} tiles, {len(done)} on volume, {len(todo)} to do; "
         f"{len(origins)**2} subtiles/tile")

    def enc(m_uint8, res):
        small = m_uint8 if m_uint8.shape[0] == res else \
            np.asarray(Image.fromarray(m_uint8 * 255).resize((res, res), Image.BILINEAR)) >= 128
        r = maskUtils.encode(np.asfortranarray(np.asarray(small, np.uint8)))
        return {"size": [res, res], "counts": r["counts"].decode("ascii")}

    tot = 0
    for n, tid in enumerate(todo):
        rgb = ds[idx[int(manifest[tid]["image_id"])]]["image"].convert("RGB")
        if full != 2048:
            rgb = rgb.resize((full, full), Image.BILINEAR)
        arr = np.asarray(rgb)
        B, S, raw = [], [], []
        for oy in origins:
            for ox in origins:
                sub = arr[oy:oy + TILE_PX, ox:ox + TILE_PX]
                inp = {"image": torch.as_tensor(sub.transpose(2, 0, 1).copy()),
                       "height": TILE_PX, "width": TILE_PX}
                with torch.no_grad():
                    o = det([inp])[0]["instances"].to("cpu")
                if not len(o):
                    continue
                b = (o.pred_boxes.tensor.numpy() + np.array([ox, oy, ox, oy], np.float32)) / scale
                s = o.scores.numpy()
                raw.append({"oy": int(oy), "ox": int(ox), "boxes_2048": np.round(b, 2).tolist(),
                            "scores": np.round(s, 4).tolist()})
                B.append(b); S.append(s)
        rec = {"raw_dets": raw, "boxes_2048": [], "scores": [], "sam_scores": [],
               "masks_rle_1777": [], "masks_rle_2048": [], "masks_rle_512": []}
        if B:
            B = np.concatenate(B); S = np.concatenate(S)
            k = S >= SCORE_FLOOR; B, S = B[k], S[k]
            k = _nms(B, S, NMS_IOU); B, S = B[k], S[k]
            if len(S) > KEEP_DETS:
                k = np.argsort(-S)[:KEEP_DETS]; k.sort(); B, S = B[k], S[k]
        if len(B) if B is not None and not isinstance(B, list) else False:
            sam_img = rgb.resize((SAM_TILE, SAM_TILE), Image.BILINEAR) if full != SAM_TILE else rgb
            f = SAM_TILE / 2048.0
            inputs = proc(images=sam_img, input_boxes=[(B * f).tolist()], return_tensors="pt")
            with torch.no_grad():
                emb = sam.get_image_embeddings(inputs["pixel_values"].to("cuda"))
            all_boxes = inputs["input_boxes"]; osz = inputs["original_sizes"]
            r1777, r2048, r512, ssc = [], [], [], []
            for i0 in range(0, len(B), BOX_BATCH):
                with torch.no_grad():
                    so = sam(input_boxes=all_boxes[:, i0:i0 + BOX_BATCH].to("cuda"),
                             image_embeddings=emb, multimask_output=False)
                m = proc.post_process_masks(so.pred_masks.cpu(), osz)[0]
                m = m.squeeze(1).numpy().astype(np.uint8)          # (n, 1777, 1777)
                iou = getattr(so, "iou_scores", None)
                if iou is not None:
                    ssc.extend(np.asarray(iou.cpu()).reshape(len(m), -1)[:, 0].round(4).tolist())
                for mm in m:
                    assert mm.shape == (SAM_TILE, SAM_TILE), mm.shape
                    r1777.append(enc(mm, SAM_TILE))
                    r2048.append(enc(mm, NATIVE))
                    r512.append(enc(mm, LOW))                     # == published path
            rec.update({"boxes_2048": np.round(B, 2).tolist(), "scores": np.round(S, 4).tolist(),
                        "sam_scores": ssc, "masks_rle_1777": r1777, "masks_rle_2048": r2048,
                        "masks_rle_512": r512})
            tot += len(r512)
        tmp = os.path.join(TILES, tid + ".json.tmp")
        json.dump(rec, open(tmp, "w"))
        os.replace(tmp, os.path.join(TILES, tid + ".json"))
        if (n + 1) % CKPT_EVERY == 0 or n + 1 == len(todo):
            vol.commit()
            _log(f"  {n+1}/{len(todo)} this run ({len(done)+n+1}/{len(tids)} total)  crowns={tot}")
    vol.commit()
    _log("[selvabox-native] predict done")
    return {"tiles_total": len(tids), "done_this_run": len(todo)}


@app.function(image=image, volumes={"/vol": vol}, timeout=3600, cpu=2, memory=32768)
def assemble():
    gt = json.load(open("/root/test_gt.json"))
    manifest = json.load(open("/root/manifest.json"))["feat_test"]
    tids = [t for t in sorted(gt) if t in manifest]
    missing = [t for t in tids if not os.path.exists(os.path.join(TILES, t + ".json"))]
    assert not missing, f"{len(missing)} tiles missing, e.g. {missing[:3]}"
    base = {"model": f"SelvaBox DINO-Swin-L (NQOS-selvamask-FT) -> SAM 3 (selvamask-FT), "
                     f"{TILE_PX}px @{OVERLAP} ovl @{GROUND_RES} m/px, IoU-NMS {NMS_IOU}, "
                     f"no edge-band cull", "det_repo": DET_REPO, "sam_repo": SAM_REPO,
            "sam_tile": SAM_TILE, "tile_px": TILE_PX, "overlap": OVERLAP,
            "ground_res": GROUND_RES, "score_floor": SCORE_FLOOR, "keep_dets": KEEP_DETS,
            "maxdets_note": "cap left to the scorer; CanopyRS own budget = 1600/2048^2-equiv",
            "n_tiles": len(tids), "source_tiles": TILES}
    for key, res, name in (("masks_rle_2048", NATIVE, f"preds_selvabox_bench_r{NATIVE}.json"),
                           ("masks_rle_1777", SAM_TILE, f"preds_selvabox_bench_r{SAM_TILE}_native.json"),
                           ("masks_rle_512", LOW, f"preds_selvabox_bench_r{LOW}_incontainer.json")):
        fp = os.path.join(OUT_DIR, name)
        assert not os.path.exists(fp), f"REFUSING to overwrite {fp}"
        preds, tot = {}, 0
        for t in tids:
            r = json.load(open(os.path.join(TILES, t + ".json")))
            preds[t] = {"boxes_2048": r["boxes_2048"], "scores": r["scores"],
                        "sam_scores": r["sam_scores"], "masks_rle": r[key]}
            tot += len(r["scores"])
        json.dump({"meta": dict(base, mask_res=res, n_crowns=tot), "preds": preds}, open(fp, "w"))
        _log(f"[selvabox-native] wrote {fp}: {len(preds)} tiles, {tot} crowns")
    vol.commit()
    return {"out_dir": OUT_DIR}
