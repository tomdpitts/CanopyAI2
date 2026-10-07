"""SAM 3 box-prompted on LACE's seed-0 boxes (the tab:masker ablation row), persisting masks at
the NATIVE 2048 raster and the 512 area-majority downsample from the same forward pass.

Derived from modal_tcd_multiseed/phase4_sam/sam3_modal.py. Same image, same SAM 3 commit, same
source boxes (/vol/out/preds_selfmask_fix_thr025_phase4_L24_s0/preds.json), same
`sam3_eval_tcd.sam_masks_for_tile` call -- with res=2048, which that function already supports
("keep SAM's own resolution"). The 512 arm is the identical area-majority rule the published
run used (reshape(res,4,res,4).mean >= 0.5), so it is a byte-level gate on the masks IF a
512 preds file exists; the published run predates the persistence block, so the gate here is
the legacy-scorer AP in sam3_results_439.json reproduced by score_coco on the 512 arm to the
tolerance those two scorers agree at (cocoeval_parity.json: AP50 identical to 4 dp).

Per-tile JSON checkpoints under OUT_DIR/tiles, vol.commit() every CKPT_EVERY tiles, resume,
run.log on the volume. OUT_DIR is NEW. Launch ONLY via detach.py deploy + spawn.

    python detach.py deploy sam3_native_modal.py
    python detach.py spawn tcd04-sam3-native predict
    modal run sam3_native_modal.py::assemble
"""
import json
import os

import modal

APP_NAME = "tcd04-sam3-native"
PHASE4_VOL = "tcd04-phase4-vol"
HFDATA_VOL = "canopyai-deepforest-data"

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
PH4SAM = os.path.join(PKG, "modal_tcd_multiseed", "phase4_sam")
PHASE4 = os.path.join(PKG, "modal_tcd_multiseed", "phase4")

SAM3_COMMIT = "46957e47805eaa273f4aa7bbbd25a88bca9108ce"      # == sam3_modal.py
NATIVE, LOW = 2048, 512
CKPT_EVERY = 10

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(PHASE4_VOL)
hfvol = modal.Volume.from_name(HFDATA_VOL)
hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.12.1", "torchvision==0.27.1", "numpy==1.26.4", "pillow",
                 "datasets==4.0.0", "einops", "pycocotools", "open_clip_torch", "psutil")
    .pip_install(f"git+https://github.com/facebookresearch/sam3.git@{SAM3_COMMIT}")
    .env({"HF_HOME": "/hfdata/hf_cache", "HF_HUB_OFFLINE": "0", "HF_DATASETS_OFFLINE": "1"})
    .add_local_file(os.path.join(PH4SAM, "sam3_eval_tcd.py"), "/root/sam3_eval_tcd.py")
    .add_local_file(os.path.join(PHASE4, "manifest.json"), "/root/manifest.json")
    .add_local_file(os.path.join(PKG, "test_gt.json"), "/root/test_gt.json")
)

PREDS = "/vol/out/preds_selfmask_fix_thr025_phase4_L24_s0/preds.json"
OUT_DIR = f"/vol/out/native_raster/sam3_s0_r{NATIVE}"
TILES = os.path.join(OUT_DIR, "tiles")
LOG = os.path.join(OUT_DIR, "run.log")


def _log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as fh:
        fh.write(msg + "\n")


@app.function(gpu="A100", image=image, volumes={"/vol": vol, "/hfdata": hfvol},
              timeout=6 * 3600, cpu=8, memory=65536, secrets=[hf_secret])
def predict(limit: int = 0, chunk: int = 64):
    import sys
    import numpy as np
    import torch
    from datasets import load_dataset
    from pycocotools import mask as maskUtils
    sys.path.insert(0, "/root")
    import sam3_eval_tcd as EV
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    assert torch.cuda.is_available()
    os.makedirs(TILES, exist_ok=True)
    _log(f"[sam3-native] start OUT_DIR={OUT_DIR} torch={torch.__version__} "
         f"numpy={np.__version__} sam3={SAM3_COMMIT} gpu={torch.cuda.get_device_name(0)}")
    preds_all = json.load(open(PREDS))
    preds, meta = preds_all["preds"], preds_all["meta"]
    gt = json.load(open("/root/test_gt.json"))
    manifest = json.load(open("/root/manifest.json"))["feat_test"]
    tids = [t for t in sorted(preds) if t in gt and t in manifest]
    if limit:
        tids = tids[:limit]
    done = {f[:-5] for f in os.listdir(TILES) if f.endswith(".json")}
    todo = [t for t in tids if t not in done]
    _log(f"[sam3-native] {len(tids)} tiles, {len(done)} on volume, {len(todo)} to do; "
         f"op_thr={meta.get('op_thr')} src={PREDS}")

    model = build_sam3_image_model(enable_inst_interactivity=True)
    processor = Sam3Processor(model)
    ds = load_dataset("restor/tcd", split="test")
    idx = {int(i): k for k, i in enumerate(ds["image_id"])}

    def enc(m, res):
        r = maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
        return {"size": [res, res], "counts": r["counts"].decode("ascii")}

    tot = 0
    for k, tid in enumerate(todo):
        boxes = np.asarray(preds[tid]["boxes_2048"], np.float32).reshape(-1, 4)
        img = ds[idx[int(manifest[tid]["image_id"])]]["image"].convert("RGB")
        masks, sam_sc = EV.sam_masks_for_tile(model, processor, img, boxes, chunk, res=NATIVE)
        r2048, r512 = [], []
        for m in masks:
            assert m.shape == (NATIVE, NATIVE), m.shape
            r2048.append(enc(m, NATIVE))
            f = NATIVE // LOW
            r512.append(enc(m.reshape(LOW, f, LOW, f).mean((1, 3)) >= 0.5, LOW))  # == published rule
        rec = {"boxes_2048": boxes.tolist(), "scores": [float(s) for s in preds[tid]["scores"]],
               "sam_scores": np.asarray(sam_sc, np.float32).round(4).tolist(),
               "masks_rle_2048": r2048, "masks_rle_512": r512}
        tmp = os.path.join(TILES, tid + ".json.tmp")
        json.dump(rec, open(tmp, "w")); os.replace(tmp, os.path.join(TILES, tid + ".json"))
        tot += len(r2048)
        if (k + 1) % CKPT_EVERY == 0 or k + 1 == len(todo):
            vol.commit()
            _log(f"  {k+1}/{len(todo)} this run ({len(done)+k+1}/{len(tids)} total) masks={tot}")
    vol.commit()
    _log("[sam3-native] predict done")
    return {"tiles_total": len(tids), "done_this_run": len(todo)}


@app.function(image=image, volumes={"/vol": vol}, timeout=3600, cpu=2, memory=32768)
def assemble():
    preds_all = json.load(open(PREDS))
    meta = preds_all["meta"]
    gt = json.load(open("/root/test_gt.json"))
    manifest = json.load(open("/root/manifest.json"))["feat_test"]
    tids = [t for t in sorted(preds_all["preds"]) if t in gt and t in manifest]
    missing = [t for t in tids if not os.path.exists(os.path.join(TILES, t + ".json"))]
    assert not missing, f"{len(missing)} missing e.g. {missing[:3]}"
    base = {"model": "SAM 3 image (box-prompt), seed-0 LACE boxes", "sam3_commit": SAM3_COMMIT,
            "src_preds": PREDS, "op_thr": meta.get("op_thr"), "tile_px": 2048,
            "n_tiles": len(tids), "source_tiles": TILES,
            "notes": "scores = LACE detector confidence; sam_scores = SAM's own per-mask score"}
    for key, res, name in (("masks_rle_2048", NATIVE, f"preds_sam3_s0_r{NATIVE}.json"),
                           ("masks_rle_512", LOW, f"preds_sam3_s0_r{LOW}_incontainer.json")):
        fp = os.path.join(OUT_DIR, name)
        assert not os.path.exists(fp), f"REFUSING to overwrite {fp}"
        preds, tot = {}, 0
        for t in tids:
            r = json.load(open(os.path.join(TILES, t + ".json")))
            preds[t] = {"boxes_2048": r["boxes_2048"], "scores": r["scores"],
                        "sam_scores": r["sam_scores"], "masks_rle": r[key]}
            tot += len(r["scores"])
        json.dump({"meta": dict(base, mask_res=res, scale_box_to_mask=2048 / res, n_masks=tot),
                   "preds": preds}, open(fp, "w"))
        _log(f"[sam3-native] wrote {fp}: {len(preds)} tiles, {tot} masks")
    vol.commit()
    return {"out_dir": OUT_DIR}
