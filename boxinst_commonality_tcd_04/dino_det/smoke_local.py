"""LOCAL timing smoke for DINO-Swin-L (detrex `dino_swin_large_384_5scale_36ep`, COCO-pretrained) on OAM-TCD 1024 crops,
Apple MPS. Purpose: a MEASURED s/iter and memory to decide local vs Modal training. No number from it is a result.

- toolchain: ~/dphil/dino_local/{venv,detrex,dino_swin_large_384_5scale_36ep.pth} (install_local.sh), isolated from .venv
- data: N_CROPS 1024 crops (top-left quadrant) of the first train tiles (train_tiles_gt.json partition 'train'), local tifs,
  GT boxes clipped to the crop (kept if >= 50% of the box area survives); no canopy handling (timing only)
- model: num_classes 1 (COCO class head not loaded: shape mismatch), device mps, AdamW lr 1e-4, backbone x0.1 (detrex default)
- measures: forward+backward s/iter at batch 1 and 2 (after 3 warm-up iters), inference s/crop, peak MPS memory, and
  whether the deformable-attention op ran on its PyTorch fallback
  /Users/tompitts/dphil/dino_local/venv/bin/python boxinst_commonality_tcd_04/dino_det/smoke_local.py
"""
import json, os, sys, time

import numpy as np
import torch

W = "/Users/tompitts/dphil/dino_local"
REPO = "/Users/tompitts/dphil/CanopyAI2"; PKG = os.path.join(REPO, "boxinst_commonality_tcd_04")
CFG = "projects/dino/configs/dino-swin/dino_swin_large_384_5scale_36ep.py"
CKPT = os.path.join(W, "dino_swin_large_384_5scale_36ep.pth")
OUT = os.path.join(PKG, "dino_det", "smoke_local.json")
N_CROPS, WARM, ITERS = 12, 3, 10


def crops():
    from PIL import Image
    gt = json.load(open(os.path.join(PKG, "train_tiles_gt.json")))
    tids = sorted(t for t, v in gt.items() if v["partition"] == "train")[:N_CROPS]
    out = []
    for t in tids:
        a = np.asarray(Image.open(os.path.join(REPO, "data", "tcd", "train", t + ".tif")).convert("RGB"))[:1024, :1024]
        b = np.asarray(gt[t]["boxes"], np.float32).reshape(-1, 4)
        c = b.copy(); c[:, [0, 2]] = c[:, [0, 2]].clip(0, 1024); c[:, [1, 3]] = c[:, [1, 3]].clip(0, 1024)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]); ca = (c[:, 2] - c[:, 0]).clip(0) * (c[:, 3] - c[:, 1]).clip(0)
        out.append((t, a, c[ca >= 0.5 * np.maximum(area, 1e-6)]))
    return out


def sample(img, boxes, dev):
    from detectron2.structures import Boxes, Instances
    inst = Instances((1024, 1024)); inst.gt_boxes = Boxes(torch.as_tensor(boxes, dtype=torch.float32)); inst.gt_classes = torch.zeros(len(boxes), dtype=torch.int64)
    return {"image": torch.as_tensor(img).permute(2, 0, 1).contiguous(), "instances": inst.to(dev), "height": 1024, "width": 1024}


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def redirect_cuda(dev):
    """detrex DINO hard-codes .cuda() / .to("cuda") in its denoising code (dino.py, dn_criterion.py). Without CUDA, map
    both to `dev` here instead of editing detrex."""
    if torch.cuda.is_available():
        return False
    to0 = torch.Tensor.to
    def to(self, *a, **k):
        a = tuple(dev if (isinstance(x, str) and x.startswith("cuda")) else x for x in a)
        if isinstance(k.get("device"), str) and k["device"].startswith("cuda"):
            k["device"] = dev
        return to0(self, *a, **k)
    torch.Tensor.to = to
    torch.Tensor.cuda = lambda self, *a, **k: to0(self, dev)
    return True


def main():
    sys.path.insert(0, os.path.join(W, "detrex"))
    from detectron2.config import LazyConfig, instantiate
    from detrex.checkpoint import DetectionCheckpointer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rec = {"torch": torch.__version__, "device": dev, "cfg": CFG, "cuda_redirected": redirect_cuda(dev)}
    cfg = LazyConfig.load(os.path.join(W, "detrex", CFG))
    cfg.model.num_classes = 1
    if hasattr(cfg.model, "device"):
        cfg.model.device = dev
    model = instantiate(cfg.model).to(dev)
    real = torch.load
    torch.load = lambda f, *a, **k: real(f, *a, **{**k, "weights_only": False})
    try:
        DetectionCheckpointer(model).load(CKPT)
    finally:
        torch.load = real
    rec["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
    import detrex.layers.multi_scale_deform_attn as MSDA
    rec["msda_cuda_ext_loaded"] = bool(getattr(MSDA, "_C", None)) if hasattr(MSDA, "_C") else "unknown"
    data = crops(); rec["n_crops"] = len(data); rec["mean_gt_per_crop"] = round(float(np.mean([len(d[2]) for d in data])), 1)
    bb = [p for n, p in model.named_parameters() if "backbone" in n]; rest = [p for n, p in model.named_parameters() if "backbone" not in n]
    opt = torch.optim.AdamW([{"params": bb, "lr": 1e-5}, {"params": rest, "lr": 1e-4}], weight_decay=1e-4)
    model.train()
    for bs in (1, 2):
        times = []
        for it in range(WARM + ITERS):
            batch = [sample(*data[(it * bs + j) % len(data)][1:], dev) for j in range(bs)]
            sync(); t0 = time.time()
            losses = model(batch); loss = sum(losses.values())
            opt.zero_grad(); loss.backward(); opt.step(); sync()
            if it >= WARM:
                times.append(time.time() - t0)
            print(f"  bs{bs} it{it} {time.time()-t0:.2f}s loss {loss.item():.3f}", flush=True)
        rec[f"train_s_per_iter_bs{bs}"] = round(float(np.median(times)), 3)
        rec[f"train_s_per_crop_bs{bs}"] = round(float(np.median(times)) / bs, 3)
    model.eval(); times = []
    with torch.no_grad():
        for it in range(6):
            s = sample(*data[it][1:], dev); s.pop("instances")
            sync(); t0 = time.time(); model([s]); sync()
            if it >= 1:
                times.append(time.time() - t0)
    rec["infer_s_per_crop"] = round(float(np.median(times)), 3)
    if dev == "mps":
        rec["mps_driver_alloc_GB"] = round(torch.mps.driver_allocated_memory() / 2**30, 1)
    spc = min(rec["train_s_per_crop_bs1"], rec["train_s_per_crop_bs2"])
    crops_per_epoch = 792 * 4
    rec["projection_hours"] = {f"{e}ep": round(spc * crops_per_epoch * e / 3600, 1) for e in (12, 24, 36)}
    rec["date"] = time.strftime("%Y-%m-%d %H:%M")
    json.dump(rec, open(OUT, "w"), indent=1); print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()
