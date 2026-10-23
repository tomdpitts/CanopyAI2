"""1-GPU gate for B3 (MAL). Every assertion must pass before a training GPU-hour is spent.

G0 environment pins; MAE weights cached (MAL downloads them at every model build otherwise)
G1 released COCO labeler loads into MAL: every student/teacher tensor, shapes equal
G2 training json = LACE's 792-tile boxes; MAL's own filter at 920 keeps exactly 31,341 boxes
G3 MAL's train transform on 40 boxes, edge boxes included (the wrap-around crop path)
G4 a real PL fit (1 GPU, DDP, fp16) of 6 steps: finite losses
G5 OUR tile-batched inference == MAL's own `main.py --box_inputs --label_dump_path`, BITWISE, on
   one val tile's GT boxes (released labeler)
G6 our uint8 crop -> GPU ToTensor/Normalize == MAL's test transform tensor, bitwise
G7 val IoU on 2 val tiles with the released labeler (recorded, not thresholded); timing
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

from tierb import tierb_common as TC
from tierb.mal import mal_run as MR

PAR = "/tmp/mal_parity"


def gate(a):
    import pytorch_lightning as pl
    import timm
    import mmcv
    import fairscale
    from models.mal import MAL
    from datasets.coco import BoxLabelCOCO
    from datasets.data_aug import data_aug_pipelines
    rep = {"method": "mal", "time": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}}
    ck = rep["checks"]
    t00 = time.time()

    # G0
    th = os.environ.get("TORCH_HOME", "")
    cached = [os.path.join(dp, f) for dp, _, fs in os.walk(th) for f in fs if f.endswith(".pth")]
    assert cached, f"no MAE weights cached under TORCH_HOME={th}"
    ck["G0_env"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                    "pl": pl.__version__, "timm": timm.__version__, "mmcv": mmcv.__version__,
                    "fairscale": fairscale.__version__, "numpy": np.__version__,
                    "mae_cache": [os.path.basename(c) for c in cached]}
    assert np.__version__.startswith("1.") and int(np.__version__.split(".")[1]) < 24
    assert torch.__version__.startswith("1.") and pl.__version__.startswith("1.8")

    # G1
    MR.patch_datapaths()
    args = MR.mal_args(["--dataset_type", "coco"], 1)
    model = MAL(args=args, num_iter_per_epoch=1)
    ck["G1_labeler"] = MR.load_labeler(model)
    assert ck["G1_labeler"]["ckpt_only"] == [], ck["G1_labeler"]["ckpt_only"]   # CLI loads strict
    model.cuda().eval()

    # G2
    tj = json.load(open(MR.TRAIN_JSON))
    man = TC.manifest()
    assert len(tj["images"]) == 792 and sorted(i["file_name"][:-4] for i in tj["images"]) == man["ids"]["train"]
    assert len(tj["annotations"]) == 54191
    ds = BoxLabelCOCO(MR.TRAIN_JSON, TC.RGB_DIR, min_obj_size=MR.MIN_OBJ_SIZE, max_obj_size=1e10,
                      transform=data_aug_pipelines["train"](MR.mal_args(["--dataset_type", "coco"], 1)),
                      args=args)
    assert len(ds) == 31341, len(ds)
    ck["G2_train_json"] = {"images": 792, "annotations": 54191, "kept_by_mal_filter": len(ds),
                           "min_obj_size": MR.MIN_OBJ_SIZE}

    # G3: edge boxes exercise MAL's wrap-around crop
    anns = ds.coco.dataset["annotations"]
    edge = [i for i, an in enumerate(anns) if an["bbox"][0] < 30 or an["bbox"][1] < 30
            or an["bbox"][0] + an["bbox"][2] > 2018 or an["bbox"][1] + an["bbox"][3] > 2018][:20]
    idxs = edge + list(np.random.default_rng(0).choice(len(ds), 20, replace=False))
    for i in idxs:
        d = ds[int(i)]
        assert tuple(d["image"].shape) == (3, 512, 512) and tuple(d["timage"].shape) == (3, 512, 512)
        assert tuple(d["mask"].shape) == (1, 512, 512)
    ck["G3_train_items"] = {"n": len(idxs), "edge": len(edge)}

    # G5 + G6: parity against MAL's own CLI on one val tile's GT boxes
    os.makedirs(PAR, exist_ok=True)
    boxes, gts = MR.val_gt_boxes()
    # a busy tile (<=60 boxes) whose boxes ALL lie inside it: MAL's own CLI gets no clipping, so the
    # comparison is like for like (135 of 7,041 val boxes do reach past the frame)
    inside = [t for t in boxes if all(MR.clip_xywh(b) == [float(v) for v in b] for b in boxes[t])]
    tid = max(inside, key=lambda t: (len(boxes[t]) <= 60, len(boxes[t])))
    xywh = boxes[tid]
    coco = {"images": [{"id": 1, "file_name": f"{tid}.png", "height": 2048, "width": 2048}],
            "categories": [{"id": 1, "name": "tree"}],
            "annotations": [{"id": i + 1, "image_id": 1, "category_id": 1, "bbox": b, "iscrowd": 0,
                             "area": b[2] * b[3]} for i, b in enumerate(xywh)]}
    json.dump(coco, open(f"{PAR}/coco.json", "w"))
    json.dump([dict(an, score=1.0) for an in coco["annotations"]], open(f"{PAR}/boxes.json", "w"))
    out = f"{PAR}/dump.json"
    cmd = [sys.executable, "/root/tierb/mal/mal_cli_launch.py",
           json.dumps({"coco": f"{PAR}/coco.json", "img_dir": TC.RGB_DIR}),
           "--resume", MR.LABELER, "--label_dump_path", out, "--not_eval_mask",
           "--box_inputs", f"{PAR}/boxes.json", "--dataset_type", "coco", "--gpus", "0",
           "--strategy", "ddp", "--batch_size", "4"]
    env = {k: v for k, v in os.environ.items() if k not in (
        "LOCAL_RANK", "RANK", "WORLD_SIZE", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT",
        "LOCAL_WORLD_SIZE", "GROUP_RANK")}
    p = subprocess.run(cmd, cwd=MR.MAL_ROOT, capture_output=True, text=True, env=env, timeout=1800)
    open(f"{PAR}/cli.log", "w").write(p.stdout + "\n" + p.stderr)
    assert os.path.exists(out + ".result"), f"MAL CLI produced no labels (rc {p.returncode}); see {PAR}/cli.log"
    theirs = {r["id"]: r["segmentation"] for r in json.load(open(out + ".result"))}
    ours = MR.run_boxes(model, args, {tid: xywh}, workers=2)[tid]
    from pycocotools import mask as mu
    n_eq = 0
    for i, (clip, binm, _) in enumerate(ours):
        a_ = mu.decode({"size": theirs[i + 1]["size"], "counts": theirs[i + 1]["counts"].encode()}).astype(bool)
        b_ = mu.decode({"size": [2048, 2048], "counts": MR.window_rle(binm, clip)["counts"].encode()}).astype(bool)
        assert np.array_equal(a_, b_), f"box {i}: {int((a_ != b_).sum())} px differ"
        n_eq += 1
    ck["G5_cli_parity"] = {"tile": tid, "boxes": n_eq, "bitwise_equal": True, "cli_rc": p.returncode}

    from PIL import Image
    from datasets.data_aug import data_aug_pipelines as DAP
    tf = DAP["test"](args)
    img = Image.open(TC.rgb_path(tid)).convert("RGB")
    item = MR.TileBoxes([(tid, list(range(8)), xywh[:8])], args)[0]
    mine = MR.to_input(item["img_u8"]).cpu()
    for j, (bx, by, bw, bh) in enumerate(xywh[:8]):
        canvas = np.zeros((2048, 2048))
        x0, y0, x1, y1 = int(bx), int(by), int(bx + bw), int(by + bh)
        canvas[y0:y1 + 1, x0:x1 + 1] = 1
        d = tf({"image": img, "boxmask": canvas, "height": 2048, "width": 2048,
                "bbox": np.array([bx, by, bx + bw, by + bh], dtype=np.float32)})
        assert torch.equal(d["image"], mine[j]), f"input tensor {j} differs"
        assert torch.equal(d["boxmask"], item["boxmask"][j][None]), f"boxmask {j} differs"
    ck["G6_input_parity"] = {"boxes": 8, "bitwise_equal": True}

    # G4: a real PL fit, 6 steps (AFTER the CLI parity run: PL's in-process DDP leaves
    # MASTER_PORT/LOCAL_RANK in os.environ and its process group bound)
    sub = {"images": tj["images"], "categories": tj["categories"],
           "annotations": [an for an in tj["annotations"] if an["bbox"][2] * an["bbox"][3] > 4000][:64]}
    os.makedirs(PAR, exist_ok=True)
    json.dump(sub, open(f"{PAR}/fit_sub.json", "w"))
    MR.patch_datapaths(train_json=f"{PAR}/fit_sub.json")
    from datasets.pl_data_module import WSISDataModule
    fargs = MR.mal_args(["--dataset_type", "coco", "--batch_size", "4", "--strategy", "ddp"], 1)
    dm = WSISDataModule(num_workers=2, load_train=True, load_val=False, args=fargs)
    fm = MAL(args=fargs, num_iter_per_epoch=len(dm.train_dataloader()))
    MR.load_labeler(fm)
    seen = []

    class Grab(pl.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            seen.append({k: float(v) for k, v in trainer.callback_metrics.items()})
    tr = pl.Trainer(gpus=[0], strategy="ddp", accelerator="gpu", precision=16, max_steps=6,
                    limit_val_batches=0, num_sanity_val_steps=0, enable_progress_bar=False,
                    logger=False, enable_checkpointing=False, callbacks=[Grab()])
    tr.fit(fm, train_dataloaders=dm.train_dataloader())
    assert seen and all(np.isfinite(v) for s in seen for v in s.values()), seen
    ck["G4_fit"] = {"steps": len(seen), "last": seen[-1]}
    MR.patch_datapaths()
    del fm, tr

    # G7
    t0 = time.time()
    tids = sorted(boxes)[:2]
    ious = MR.val_iou(model, args, tids, boxes, gts, workers=2)
    nb = sum(len(boxes[t]) for t in tids)
    ck["G7_val_released_labeler"] = {"tiles": tids, "n_valid": len(ious),
                                     "mean_iou": float(np.mean(ious)) if ious else None,
                                     "sec_per_box": round((time.time() - t0) / max(nb, 1), 4),
                                     "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    rep["pass"] = True
    rep["seconds"] = round(time.time() - t00, 1)
    TC.report("mal", "gate", rep)
    TC.mark_done("mal", "gate", {"pass": True})
    print(json.dumps(rep, indent=1, default=str), flush=True)
    print("GATE PASS", flush=True)
