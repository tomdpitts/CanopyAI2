"""Modal app for pre-registration D1 (dino_det/README.md): DINO-Swin-L (detrex, COCO init) on OAM-TCD boxes.

Volume tcd04-tierb-vol (Tier B's staged HF pixels at /vol/rgb); everything written under /vol/runs/dino_swinl_s0[_smoke]/.
Functions (spawn detached via native_raster/detach.py):
  smoke()            1 x A100-80GB: ~20 iters at batch 2, mini-eval on 1 val tile, checkpoint, then a resume check
  train()            8 x A100-80GB: train_dino.py under the budget ledger; resumable (retries + last.pth)
  predict(split)     1 x A100-80GB: model_best.pth, 3x3 1024 subtiles, BoxTeacher's seam-fixed stitch on boxes -> preds_<split>.json
Spend ledger: /vol/runs/dino_swinl_s0/spend.jsonl (billed seconds x list rate, CPU + memory included, + startup).
"""
import json
import os
import subprocess
import threading
import time

import modal

APP = "tcd04-dino-swinl"
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
TIERB = os.path.join(PKG, "box_supervised_baselines", "tierb")
DETREX_SHA, D2_SHA = "e244e6c", "70f4543"
CKPT_URL = "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_swin_large_384_5scale_36ep.pth"
CKPT_SHA256 = "55dbb4776319223f968d"          # prefix of the local sha256 (dino_local/*.sha256), full value asserted below

app = modal.App(APP)
vol = modal.Volume.from_name("tcd04-tierb-vol")
RUN = "/vol/runs/dino_swinl_s0"
CAP_USD, RESERVE_USD = 75.0, 4.0
# Modal list prices (USD/s): A100-80GB 0.000694, CPU core 0.0000131, memory GiB 0.00000222 (CPU+memory on top of GPU)
def usd_h(gpus, cores, mem_gib):
    return (gpus * 0.000694 + cores * 0.0000131 + mem_gib * 0.00000222) * 3600
SPEC = {"smoke": (1, 8, 64), "smoke2": (2, 12, 96), "train": (8, 40, 192), "predict": (1, 8, 64)}
STARTUP_S = 240

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel")
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
    .apt_install("git", "g++", "ninja-build", "libgl1", "libglib2.0-0", "curl")
    .env({"TORCH_CUDA_ARCH_LIST": "8.0", "FORCE_CUDA": "1"})
    .pip_install("numpy<2", "opencv-python-headless", "pycocotools", "pillow", "scipy", "timm", "setuptools<81")
    .run_commands(
        "git clone --recursive https://github.com/IDEA-Research/detrex.git /opt/detrex",
        f"cd /opt/detrex && git checkout {DETREX_SHA} && git submodule update --init --recursive",
        f"test $(git -C /opt/detrex/detectron2 rev-parse --short HEAD) = {D2_SHA}",
        "pip install --no-build-isolation -e /opt/detrex/detectron2",
        "pip install --no-build-isolation -e /opt/detrex",
        f"mkdir -p /root/ckpt && curl -sL -o /root/ckpt/dino_swin_large_384_5scale_36ep.pth {CKPT_URL}",
        f"sha256sum /root/ckpt/dino_swin_large_384_5scale_36ep.pth | grep -q '^{CKPT_SHA256}'",
    )
    .env({"DETREX_ROOT": "/opt/detrex", "PYTHONPATH": "/root:/root/proj:/opt/detrex"})
    .add_local_dir(TIERB, "/root/tierb", copy=False,
                   ignore=["tierb_data/**", "gates/**", "**/__pycache__/**", "results/**", "calls.jsonl", "mal/**"])
    .add_local_dir(os.path.join(TIERB, "tierb_data", "canopy"), "/root/tierb_data/canopy", copy=False)
    .add_local_file(os.path.join(TIERB, "tierb_data", "val_sub_gt_1024.json"), "/root/tierb_data/val_sub_gt_1024.json", copy=False)
    .add_local_file(os.path.join(PKG, "train_tiles_gt.json"), "/root/proj/train_tiles_gt.json", copy=False)
    .add_local_file(os.path.join(HERE, "train_dino.py"), "/root/proj/dino_det/train_dino.py", copy=False)
)


def _spend(run, kind, seconds):
    g, c, m = SPEC[kind]
    usd = (seconds + STARTUP_S) / 3600 * usd_h(g, c, m)
    with open(f"{run}/spend.jsonl", "a") as f:
        f.write(json.dumps({"kind": kind, "seconds": round(seconds), "startup_s": STARTUP_S, "usd": round(usd, 3),
                            "time": time.strftime("%F %T")}) + "\n")
    return usd


def _spent(run=None):
    """Total over every D1 ledger (smoke + train + predict), so the cap covers the whole experiment."""
    tot = 0.0
    for r in (RUN, RUN + "_smoke", RUN + "_smoke2gpu"):
        p = f"{r}/spend.jsonl"
        if os.path.exists(p):
            tot += sum(json.loads(l)["usd"] for l in open(p))
    return tot


def _committer(stop):
    while not stop.wait(60):
        try:
            vol.commit()
        except Exception as e:                                    # never kill the run over a commit
            print(f"[commit] {e}", flush=True)


def _run_train(run, gpus, kind, extra):
    os.makedirs(run, exist_ok=True)
    spent0 = _spent(run)
    g, c, m = SPEC[kind]
    bfile = f"{run}/budget.json"
    json.dump({"cap_usd": CAP_USD, "reserve_usd": RESERVE_USD, "spent_usd_before": round(spent0, 3),
               "usd_per_hour": usd_h(g, c, m), "t0": time.time()}, open(bfile, "w"))
    if spent0 > CAP_USD - RESERVE_USD:
        raise RuntimeError(f"budget exhausted before start: ${spent0:.2f}")
    stop = threading.Event(); th = threading.Thread(target=_committer, args=(stop,), daemon=True); th.start()
    t0 = time.time()
    try:
        cmd = ["python", "-u", "/root/proj/dino_det/train_dino.py", "--run", run, "--gpus", str(gpus), "--budget", bfile] + extra
        with open(f"{run}/stdout.log", "a") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    finally:
        stop.set(); _spend(run, kind, time.time() - t0); vol.commit()
    if rc != 0:
        raise RuntimeError(f"train_dino.py exit {rc}; see {run}/stdout.log")
    return {"run": run, "spent_usd_total": round(_spent(run), 2), "evals": open(f"{run}/evals.jsonl").read().splitlines()[-3:]
            if os.path.exists(f"{run}/evals.jsonl") else []}


@app.function(image=image, gpu="A100-80GB", cpu=SPEC["smoke"][1], memory=SPEC["smoke"][2] * 1024, volumes={"/vol": vol},
              timeout=3600)
def smoke():
    run = RUN + "_smoke"
    a = _run_train(run, 1, "smoke", ["--smoke", "--smoke_iters", "20"])
    # resume check: a second launch must load last.pth and stop (1 epoch done = max for smoke)
    b = _run_train(run, 1, "smoke", ["--smoke", "--smoke_iters", "2"])
    log = open(f"{run}/train.log").read()
    return {"first": a, "second": b, "resumed": "[resume] from epoch 1" in log}


@app.function(image=image, gpu="A100-80GB:2", cpu=SPEC["smoke2"][1], memory=SPEC["smoke2"][2] * 1024, volumes={"/vol": vol},
              timeout=3600)
def smoke2():
    """2-GPU DDP smoke: exercises the spawned-process path (DataLoader workers, gather, broadcast) that 1 GPU skips."""
    return _run_train(RUN + "_smoke2gpu", 2, "smoke2", ["--smoke", "--smoke_iters", "10"])


@app.function(image=image, gpu="A100-80GB:8", cpu=SPEC["train"][1], memory=SPEC["train"][2] * 1024, volumes={"/vol": vol},
              timeout=12 * 3600, retries=modal.Retries(max_retries=2, initial_delay=30.0))
def train():
    return _run_train(RUN, 8, "train", [])


def _edge_cut(b, oy, ox, edge=10, sub=1024, tile=2048):
    x0, y0, x1, y1 = b
    return ((ox > 0 and x0 < edge) or (ox + sub < tile and x1 > sub - edge) or
            (oy > 0 and y0 < edge) or (oy + sub < tile and y1 > sub - edge))


def stitch(raw, floor=0.05, iou_thr=0.7, cont_thr=0.85, keep=600):
    """BoxTeacher's seam-fixed stitch (finalize_lib: edge 10 px at interior subtile edges, dedup IoU 0.7 /
    containment 0.85, floor 0.05) applied to boxes. raw: [(oy, ox, boxes_window, scores)]."""
    import numpy as np
    B, S = [], []
    for oy, ox, bx, sc in raw:
        for b, s in zip(bx, sc):
            if s < floor or _edge_cut(b, oy, ox):
                continue
            B.append([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]); S.append(float(s))
    if not B:
        return [], []
    B, S = np.asarray(B, float), np.asarray(S, float); order = np.argsort(-S, kind="stable"); kept = []
    area = (B[:, 2] - B[:, 0]).clip(0) * (B[:, 3] - B[:, 1]).clip(0)
    for i in order:
        if kept:
            K = np.asarray(kept)
            ix = (np.minimum(B[i, 2], B[K, 2]) - np.maximum(B[i, 0], B[K, 0])).clip(0)
            iy = (np.minimum(B[i, 3], B[K, 3]) - np.maximum(B[i, 1], B[K, 1])).clip(0)
            inter = ix * iy
            if ((inter / np.maximum(area[i] + area[K] - inter, 1e-9)) > iou_thr).any() or (inter / max(area[i], 1e-9) > cont_thr).any():
                continue
        kept.append(i)
        if len(kept) == keep:
            break
    return B[kept].tolist(), S[kept].tolist()


@app.function(image=image, gpu="A100-80GB", cpu=SPEC["predict"][1], memory=SPEC["predict"][2] * 1024, volumes={"/vol": vol},
              timeout=4 * 3600, retries=modal.Retries(max_retries=2, initial_delay=30.0))
def predict(split: str = "val"):
    import sys
    import numpy as np
    import torch
    sys.path.insert(0, "/opt/detrex"); sys.path.insert(0, "/root"); sys.path.insert(0, "/root/proj/dino_det")
    from tierb import tierb_common as TC
    import train_dino as T
    out = f"{RUN}/preds_{split}.json"
    if os.path.exists(out):
        return {"exists": out}
    t0 = time.time()
    tids = TC.manifest()["ids"][split]
    model, _ = T.build_model("cuda"); ck = torch.load(f"{RUN}/model_best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    rawf = f"{RUN}/raw_{split}.jsonl"; done = {}
    if os.path.exists(rawf):
        for l in open(rawf):
            r = json.loads(l); done[r["tid"]] = r
    origins = [(oy, ox) for oy in (0, 512, 1024) for ox in (0, 512, 1024)]
    for k, t in enumerate(tids):
        if t in done:
            continue
        rgb = TC.load_rgb(t); win = []
        for oy, ox in origins:
            b, s = T.window_infer(model, rgb, oy, ox)
            win.append({"oy": oy, "ox": ox, "boxes": np.round(b, 3).tolist(), "scores": np.round(s, 6).tolist()})
        rec = {"tid": t, "windows": win}
        with open(rawf, "a") as f:
            f.write(json.dumps(rec) + "\n")
        done[t] = rec
        if (k + 1) % 25 == 0:
            vol.commit(); print(f"  {split} {k+1}/{len(tids)} {time.time()-t0:.0f}s", flush=True)
    preds = {}
    for t in tids:
        b, s = stitch([(w["oy"], w["ox"], w["boxes"], w["scores"]) for w in done[t]["windows"]])
        preds[t] = {"boxes_2048": b, "scores": s}
    json.dump({"meta": {"model": "DINO-Swin-L COCO-init, trained on OAM-TCD boxes (prereg D1)", "best_epoch": ck["epoch"],
                        "val_box": ck["val_box"], "stitch": "edge 10 px interior, dedup IoU 0.7 / cont 0.85, floor 0.05, top 600"},
               "preds": preds}, open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
    _spend(RUN, "predict", time.time() - t0); vol.commit()
    return {"out": out, "n_tiles": len(preds), "n_det": sum(len(v["scores"]) for v in preds.values()), "spent_usd_total": round(_spent(RUN), 2)}
