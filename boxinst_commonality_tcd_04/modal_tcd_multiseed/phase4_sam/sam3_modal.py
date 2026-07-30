"""Modal A100 app: SAM 3 box-prompted instance seg on OAM-TCD 439 — ablation vs our
commonality masker, on the SAME seed-0 detector boxes.

Apples-to-apples: reads the saved seed-0 β=0.5-fixed predictions (boxes_2048 + detector
scores) from the phase4 Volume, prompts SAM 3 with those exact boxes, and scores the SAM
masks with the byte-identical COCO-101pt mask-AP core (sam3_eval_tcd.py, copied verbatim
from evaluate.py). RGB tiles come from HF restor/tcd via the same manifest image_id join
phase4 uses. Nothing about the detector or the boxes changes — only the box->mask module.

Run:
    modal run sam3_modal.py::eval_sam3            # full 439
    modal run sam3_modal.py::eval_sam3 --limit 20 # quick smoke test

Prereq: the HF token in the `huggingface` secret must have accepted the license for
`facebook/sam3` (gated). Compare target: our seed-0 mask mAP50 = 0.620 (β=0.5-fixed) /
0.579 (β=0-fixed) on the identical boxes.
"""
import json
import os

import modal

APP_NAME = "tcd04-sam3-ablation"
PHASE4_VOL = "tcd04-phase4-vol"               # read: seed-0 preds (boxes+scores)
HFDATA_VOL = "canopyai-deepforest-data"       # read: HF restor/tcd + hub cache

HERE = os.path.dirname(os.path.abspath(__file__))         # .../phase4_sam
MTS = os.path.dirname(HERE)                               # .../modal_tcd_multiseed
PKG = os.path.dirname(MTS)                                # boxinst_commonality_tcd_04
PHASE4 = os.path.join(MTS, "phase4")

SAM3_COMMIT = "46957e47805eaa273f4aa7bbbd25a88bca9108ce"  # pin the cloned _sam3_repo

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(PHASE4_VOL)
hfvol = modal.Volume.from_name(HFDATA_VOL)
hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.12.1", "torchvision==0.27.1", "numpy==1.26.4", "pillow",
                 "datasets==4.0.0",
                 # runtime imports sam3's predictor pulls in but does NOT declare in its deps:
                 "einops", "pycocotools", "open_clip_torch", "psutil")
    .pip_install(f"git+https://github.com/facebookresearch/sam3.git@{SAM3_COMMIT}")
    .env({"HF_HOME": "/hfdata/hf_cache", "HF_HUB_OFFLINE": "0",
          "HF_DATASETS_OFFLINE": "1"})
    .add_local_file(os.path.join(HERE, "sam3_eval_tcd.py"), "/root/sam3_eval_tcd.py")
    .add_local_file(os.path.join(PHASE4, "manifest.json"), "/root/manifest.json")
    .add_local_file(os.path.join(PKG, "test_gt.json"), "/root/test_gt.json")
)

VOL = "/vol"
PREDS = f"{VOL}/out/preds_selfmask_fix_thr025_phase4_L24_s0/preds.json"  # seed-0 boxes
OUT = f"{VOL}/out"
HF_SPLIT = {"feat_traintile": "train", "feat_test": "test"}


def _load_hf_test():
    """HF restor/tcd test split + image_id -> row index (verbatim pattern from phase4)."""
    from datasets import load_dataset
    ds = load_dataset("restor/tcd", split="test")
    idx = {int(iid): i for i, iid in enumerate(ds["image_id"])}
    return ds, idx


@app.function(gpu="A100", image=image, volumes={"/vol": vol, "/hfdata": hfvol},
              timeout=4 * 3600, cpu=8, memory=65536, secrets=[hf_secret])
def eval_sam3(limit: int = 0, chunk: int = 64):
    """Box-prompt SAM 3 with the seed-0 detections and score vs our masker."""
    import sys
    import time

    import torch
    sys.path.insert(0, "/root")
    import sam3_eval_tcd as EV
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    assert torch.cuda.is_available(), "no CUDA"
    print(f"[sam3] parity: {EV._parity_check()}", flush=True)

    preds_all = json.load(open(PREDS))
    preds, meta = preds_all["preds"], preds_all["meta"]
    op_thr = float(meta["op_thr"])
    gt = json.load(open("/root/test_gt.json"))
    manifest = json.load(open("/root/manifest.json"))["feat_test"]

    tids = [t for t in sorted(preds) if t in gt and t in manifest]
    if limit:
        tids = tids[:limit]
    preds = {t: preds[t] for t in tids}
    print(f"[sam3] {len(tids)} tiles · op_thr={op_thr} · "
          f"{sum(len(preds[t]['boxes_2048']) for t in tids)} boxes", flush=True)

    print("[sam3] building SAM 3 image model (downloads facebook/sam3 on first run)...",
          flush=True)
    t0 = time.time()
    model = build_sam3_image_model(enable_inst_interactivity=True)  # SAM1 box task
    processor = Sam3Processor(model)
    EV.model_g, EV.processor_g = model, processor
    print(f"[sam3] model ready in {(time.time()-t0)/60:.1f}min", flush=True)

    ds, idx = _load_hf_test()

    def get_rgb(tid):
        iid = int(manifest[tid]["image_id"])
        return ds[idx[iid]]["image"].convert("RGB")

    t1 = time.time()
    res = EV.evaluate_sam(preds, gt, get_rgb, op_thr, chunk=chunk)
    res["sam3_commit"] = SAM3_COMMIT
    res["wall_min"] = round((time.time() - t1) / 60, 1)
    res["compare"] = {"ours_beta0.5_fixed_seed0": 0.620, "ours_beta0_fixed_seed0": 0.579,
                      "note": "same boxes; mask mAP50, det-score-ranked"}

    out_fp = os.path.join(OUT, f"sam3_results{'_lim'+str(limit) if limit else ''}.json")
    json.dump(res, open(out_fp, "w"), indent=2)
    vol.commit()
    u = res["uncropped"]["det_score_ranked"]; c = res["box_clipped"]["det_score_ranked"]
    print(json.dumps(res, indent=2), flush=True)
    print(f"\n[sam3] === HEADLINE (uncropped, det-score-ranked) ===", flush=True)
    print(f"  SAM3   mask mAP50={u['mask_mAP50']}  mAP50-95={u['mask_mAP50_95']}  "
          f"(box-clipped {c['mask_mAP50']}/{c['mask_mAP50_95']})", flush=True)
    print(f"  OURS   β0.5-fix 0.620 / β0-fix 0.579  (same seed-0 boxes) -> {out_fp}",
          flush=True)
    return res
