"""B3: MAL phase-1 labeler (ViT-MAE-B/16, NVlabs/mask-auto-labeler @0ab9be4) inside the mal image.

    python /root/tierb/mal/mal_run.py gate                 # 1 GPU
    python /root/tierb/mal/mal_run.py train [--smoke]      # 8 GPUs (PL DDP re-launches this file)
    python /root/tierb/mal/mal_run.py predict --sets ...   # 8 GPUs, one process per GPU

Run with cwd=/opt/mal; MAL's packages (`models`, `datasets`, `utils`) are imported top-level.

TRAINING. MAL's own CLI defaults (= the released COCO labeler `epoch=6-arch=vit-mae-base-16-...`),
fine-tuned from that checkpoint on LACE's 792 training tiles' crown boxes (box-only; MAL never reads
a mask in phase 1). Changed only where our data forces it, each by a rule (TIERB.md section 5):
  * --min_obj_size 0 (Tom 2026-10-27; was 920): MAL trains on EVERY crown box LACE trains on (MAL's
    own w > 2 and h > 2 px rule still applies: 54,184 of 54,191). The released 2048 px^2 (and the
    COCO-quantile-matched 920, gates/coco_min_obj_size.json) would drop 61% / 42% of our crowns;
  * --loss_mil_step / --loss_crf_step: the released 1000 / 4000 steps are 6.44% / 25.77% of a COCO
    epoch-0 at 8 GPUs x 4 (15,522 steps); kept as the same fractions of OUR epoch-0;
  * --strategy ddp instead of ddp_sharded: same arithmetic (OSS only shards optimizer state).
  * learning rate: MAL's own rule lr = 1.5e-6 x n_gpus x batch (8 x 4 -> 4.8e-5), 10 epochs, 1 warm-up
    epoch then cosine to 0.2 lr -- all released.
Checkpoints at every epoch end only (MAL's LR schedule restarts its per-epoch step counter each
epoch, so a mid-epoch resume would replay part of the schedule); a dropped container loses at most
one epoch and resumes from the last completed one.

SELECTION. After every epoch the STUDENT (MAL's default for inference) masks the GT boxes of the
108 val tiles at the released threshold 0.1; selection = best mean per-crown IoU over valid crowns
(gtbox_lib's rule). The selected epoch's weights are saved by this file, with the score, before
anything else can overwrite them.

INFERENCE. MAL's test path, exactly: per box, RandomCropV2(random=False, margin 0.6 x 0.5 = 0.3 a
side, wrap-around crop, 512 resize) -> student -> sigmoid x box mask -> bilinear to the expanded box
-> paste (clipped) -> per-size-bucket threshold (all 0.1). Implemented as a tile-batched loop and
gated BITWISE against MAL's own `main.py --box_inputs --label_dump_path` on real boxes (`gate`).
Saved per tile: binary masks as RLE at 2048 AND the soft (pre-threshold) pasted crops as uint8, so
any re-threshold is a CPU job.

BOX CLIPPING (protocol, disclosed): MAL assumes in-image boxes; a box with x0 or y0 < 0 makes its
box-mask slice `[y0:y1+1, x0:x1+1]` count from the far end and come out EMPTY, so the mask is empty.
1-3% of LACE's boxes reach past the tile frame. Every box is therefore clipped to the tile before
MAL sees it (boxes already inside are passed through untouched); a box that clips to nothing gets
an empty mask without a forward pass. Scoring keeps the ORIGINAL boxes and scores.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

MAL_ROOT = "/opt/mal"
sys.path.insert(0, MAL_ROOT)
sys.path.insert(0, "/root")

import numpy as np                                                          # noqa: E402
import torch                                                                # noqa: E402
import torch.nn.functional as F                                             # noqa: E402

from tierb import tierb_common as TC                                       # noqa: E402
from tierb import gtbox_pure as GP                                         # noqa: E402

RUN = "mal"
LABELER = f"{TC.VOL}/weights/mal_vit-mae-base-16_coco_epoch6.ckpt"
TRAIN_JSON = f"{TC.DATA}/mal_train_boxes.json"
BOXSETS = f"{TC.DATA}/boxsets"             # prep_local.py, shipped in the image
STOP_FILE = "/tmp/tierb_STOP"
MIN_OBJ_SIZE = 0                            # train on every crown (TIERB.md section 00)
COCO_EPOCH0_STEPS = 15522                  # ceil(496,706 / 4) // 8 : MAL's own per-rank count
RELEASED_MIL_STEP, RELEASED_CRF_STEP = 1000, 4000
N_GPUS = 8
BATCH = 4


def run_dir(smoke=False):
    return TC.run_dir(RUN + ("_smoke" if smoke else ""))


# ------------------------------------------------------------------ MAL args (their CLI)
def mal_args(extra, n_gpus):
    """MAL's own parse_args() on an explicit argv, then main.py's post-processing verbatim."""
    import main as M
    saved = sys.argv
    sys.argv = ["main.py"] + list(extra)
    try:
        args = M.parse_args()
    finally:
        sys.argv = saved
    args.optim_betas = list(map(float, args.optim_betas.split(",")))
    args.margin_rate = list(map(float, args.margin_rate.split(",")))
    args.test_margin_rate = list(map(float, args.test_margin_rate.split(",")))
    args.mask_thres = list(map(float, args.mask_thres.split(",")))
    if len(args.mask_thres) == 1:
        args.mask_thres = [args.mask_thres[0] for _ in range(3)]
    args.frozen_stages = list(map(int, args.frozen_stages.split(",")))
    if len(args.frozen_stages) == 1:
        args.frozen_stages = [0, args.frozen_stages[0]]
    if len(args.margin_rate) == 1:
        args.margin_rate = args.margin_rate[0]
    args.gpus = list(range(n_gpus))
    args.lr = args.lr * len(args.gpus) * args.batch_size
    args.min_lr = args.lr * args.min_lr_rate
    return args


def patch_datapaths(train_json=TRAIN_JSON, val_json=None, val_img_dir=None):
    from datasets import pl_data_module as PDM
    c = PDM.datapath_configs["coco"]
    c["training_config"].update(train_ann_path=train_json, train_img_data_dir=TC.RGB_DIR,
                                val_ann_path=val_json or train_json,
                                val_img_data_dir=val_img_dir or TC.RGB_DIR)
    c["generating_pseudo_label_config"].update(val_ann_path=val_json or train_json,
                                               val_img_data_dir=val_img_dir or TC.RGB_DIR,
                                               train_ann_path=train_json,
                                               train_img_data_dir=TC.RGB_DIR)


def train_extra(steps_per_epoch, smoke=False):
    mil = max(1, round(RELEASED_MIL_STEP * steps_per_epoch / COCO_EPOCH0_STEPS))
    crf = max(1, round(RELEASED_CRF_STEP * steps_per_epoch / COCO_EPOCH0_STEPS))
    ex = ["--dataset_type", "coco", "--min_obj_size", str(MIN_OBJ_SIZE),
          "--batch_size", str(BATCH), "--num_workers_per_gpu", "1", "--strategy", "ddp",
          "--loss_mil_step", str(mil), "--loss_crf_step", str(crf), "--val_interval", "1000",
          "--seed", "0"]
    ex += ["--max_epochs", "2" if smoke else "20"]      # cap; plateau.py decides the actual stop
    return ex, {"loss_mil_step": mil, "loss_crf_step": crf, "steps_per_epoch": steps_per_epoch}


def load_labeler(model, path=LABELER):
    """Released COCO labeler -> MAL model. Every student/teacher tensor must load with its shape."""
    ck = torch.load(path, map_location="cpu")
    sd = ck.get("state_dict", ck)
    msd = model.state_dict()
    net = [k for k in msd if k.startswith(("student.", "teacher."))]
    missing = [k for k in net if k not in sd]
    mism = [k for k in net if k in sd and tuple(sd[k].shape) != tuple(msd[k].shape)]
    assert not missing and not mism, f"labeler mismatch: missing {missing[:5]} shape {mism[:5]}"
    res = model.load_state_dict({k: v for k, v in sd.items() if k in msd}, strict=False)
    return {"loaded": len(net), "model_keys": len(msd), "ckpt_keys": len(sd),
            "not_in_ckpt_non_network": sorted(set(msd) - set(sd)),
            "ckpt_only": sorted(set(sd) - set(msd))[:20], "epoch": ck.get("epoch"),
            "missing_after_load": list(res.missing_keys)[:20]}


# ------------------------------------------------------------------ inference (MAL's test path)
class TileBoxes(torch.utils.data.Dataset):
    """Item = (tile, chunk of <=256 boxes). Runs MAL's own RandomCropV2(random=False) per box on
    CPU workers; returns uint8 crops so the float conversion happens once on GPU (gated equal)."""

    def __init__(self, items, args):
        from datasets.data_aug import RandomCropV2
        self.items = items
        self.crop = RandomCropV2(args.crop_size, margin_rate=args.test_margin_rate, random=False,
                                 crop_fields=["image", "boxmask", "mask"])
        self._tid, self._img, self._canvas = None, None, None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        tid, k0, xywh = self.items[i]           # k0 = original indices of these boxes
        if tid != self._tid:
            self._img = Image.open(TC.rgb_path(tid)).convert("RGB")
            self._canvas = np.zeros((self._img.size[1], self._img.size[0]), np.float64)
            self._tid = tid
        w, h = self._img.size
        crops, bms, bbs, exts = [], [], [], []
        for (bx, by, bw, bh) in xywh:
            x0, y0, x1, y1 = int(bx), int(by), int(bx + bw), int(by + bh)   # InstSegVOC*
            self._canvas[y0:y1 + 1, x0:x1 + 1] = 1
            d = {"image": self._img, "boxmask": self._canvas, "height": h, "width": w,
                 "bbox": np.array([bx, by, bx + bw, by + bh], dtype=np.float32)}
            d = self.crop(d)
            self._canvas[y0:y1 + 1, x0:x1 + 1] = 0
            crops.append(np.asarray(d["image"], np.uint8))
            bms.append(np.asarray(d["boxmask"]))
            bbs.append(d["bbox"])
            exts.append(np.asarray(d["ext_boxes"], np.int64))
        return {"tid": tid, "k0": torch.as_tensor(k0, dtype=torch.long), "img_u8": torch.from_numpy(np.stack(crops)),
                "boxmask": torch.from_numpy(np.stack(bms)).float(),
                "bbox": torch.from_numpy(np.stack(bbs)), "ext": torch.from_numpy(np.stack(exts)),
                "h": h, "w": w}


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


_LUT = {}


def to_input(img_u8):
    """torchvision ToTensor + Normalize, on GPU, bitwise (gated). ToTensor divides on CPU by true
    division; CUDA `x.div(255)` with a CPU scalar multiplies by fl(1/255) instead and differs in the
    last bit for 126 of 256 values, so the CPU-computed table is gathered instead. Normalize's
    (x - mean) / std is tensor-tensor true division on both devices."""
    dev = torch.device("cuda")
    if "lut" not in _LUT:
        _LUT["lut"] = torch.arange(256, dtype=torch.float32).div(255).to(dev)
    x = _LUT["lut"][img_u8.to(dev, non_blocking=True).long()].permute(0, 3, 1, 2).contiguous()
    m = torch.tensor(MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(dev)
    s = torch.tensor(STD, dtype=torch.float32).view(1, 3, 1, 1).to(dev)
    return (x - m) / s


@torch.no_grad()
def masks_for_batch(model, args, batch):
    """MAL.validation_step's mask path (student, no flip, no mix), in MAL's own forward batch size
    (args.batch_size = 4), returning per box: (clip [cy0, cx0, cy1, cx1], soft float64 window,
    binary bool window)."""
    out = []
    n = len(batch["bbox"])
    for s in range(0, n, args.batch_size):
        sub = {k: (v[s:s + args.batch_size] if k in ("img_u8", "boxmask", "bbox", "ext", "k0") else v)
               for k, v in batch.items()}
        out += _masks_minibatch(model, args, sub)
    return out


def _masks_minibatch(model, args, batch):
    imgs = to_input(batch["img_u8"])
    boxmasks = batch["boxmask"].cuda()[:, None]
    boxes = batch["bbox"].cuda()
    H, W = imgs.shape[-2:]
    with torch.cuda.amp.autocast():
        segs = model.student(imgs, boxmasks, boxes)["seg"]
        segs = F.interpolate(segs, (H, W), align_corners=False, mode="bilinear")
        segs = segs.sigmoid()
        segs = segs * boxmasks
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        thres_list = [0, 32 ** 2, 96 ** 2, 1e5 ** 2]
        out = []
        h, w = int(batch["h"]), int(batch["w"])
        for seg, ext, area in zip(segs, batch["ext"].tolist(), areas.tolist()):
            ext_h, ext_w = ext[3] - ext[1], ext[2] - ext[0]
            roi = F.interpolate(seg[None, ...], (ext_h, ext_w), mode="bilinear",
                                align_corners=False)[0][0]
            cy0, cy1 = max(ext[1], 0), min(ext[3], h)
            cx0, cx1 = max(ext[0], 0), min(ext[2], w)
            win = roi[max(0 - ext[1], 0):ext_h + min(0, h - ext[3]),
                      max(0 - ext[0], 0):ext_w + min(0, w - ext[2])].cpu().numpy().astype(np.float64)
            binm = None
            for idx, (lth, hth) in enumerate(zip(thres_list[:-1], thres_list[1:])):
                if lth < area <= hth:
                    binm = win > args.mask_thres[idx]
            if binm is None:                    # released: un-bucketed boxes stay float, then uint8
                binm = win.astype(np.uint8).astype(bool)
            out.append(((cy0, cx0, cy1, cx1), win, binm))
    return out


def window_rle(win_bool, clip, h=2048, w=2048):
    from pycocotools import mask as mu
    canvas = np.zeros((h, w), np.uint8)
    cy0, cx0, cy1, cx1 = clip
    canvas[cy0:cy1, cx0:cx1] = win_bool
    r = mu.encode(np.asfortranarray(canvas))
    return {"size": r["size"], "counts": r["counts"].decode("ascii")}


def clip_xywh(b, W=2048, H=2048):
    """Clip to the tile; boxes already inside are returned UNCHANGED (same floats). None if empty."""
    x, y, w, h = (float(v) for v in b)
    if x >= 0 and y >= 0 and x + w <= W and y + h <= H:
        return [x, y, w, h]
    x0, y0, x1, y1 = max(x, 0.0), max(y, 0.0), min(x + w, float(W)), min(y + h, float(H))
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


def make_items(tile_boxes, chunk=256):
    """tile_boxes: {tid: [[x,y,w,h], ...]} -> (dataset items (tid, [orig indices], clipped xywh),
    {tid: [orig indices of boxes that clip to nothing]})."""
    items, empty = [], {}
    for tid in sorted(tile_boxes):
        idx, cl = [], []
        for k, b in enumerate(tile_boxes[tid]):
            c = clip_xywh(b)
            if c is None:
                empty.setdefault(tid, []).append(k)
            else:
                idx.append(k); cl.append(c)
        for k0 in range(0, len(cl), chunk):
            items.append((tid, idx[k0:k0 + chunk], cl[k0:k0 + chunk]))
    return items, empty


def run_boxes(model, args, tile_boxes, workers=8, want_soft=False, on_tile=None):
    """Mask every box. Returns {tid: [(clip, binary, soft?) per box in input order]} unless
    `on_tile(tid, results)` is given (streaming: called once per finished tile)."""
    items, empty = make_items(tile_boxes)
    acc, out = {}, {}
    need = {t: len(b) for t, b in tile_boxes.items()}
    for tid, ks in empty.items():                 # clipped to nothing: empty mask, no forward pass
        lst = acc.setdefault(tid, {})
        for k in ks:
            lst[k] = ((0, 0, 0, 0), np.zeros((0, 0), bool), np.zeros((0, 0)) if want_soft else None)
    for tid in list(acc):
        if len(acc[tid]) == need[tid]:
            ordered = [acc[tid][k] for k in range(need[tid])]
            del acc[tid]
            on_tile(tid, ordered) if on_tile else out.__setitem__(tid, ordered)
    dl = torch.utils.data.DataLoader(TileBoxes(items, args), batch_size=None, shuffle=False,
                                     num_workers=workers, persistent_workers=False) if items else []
    for batch in dl:
        res = masks_for_batch(model, args, batch)
        tid = batch["tid"]
        lst = acc.setdefault(tid, {})
        for k, (clip, win, binm) in zip(batch["k0"].tolist(), res):
            lst[k] = (clip, binm, win if want_soft else None)
        if len(lst) == need[tid]:
            ordered = [lst[k] for k in range(need[tid])]
            del acc[tid]
            if on_tile:
                on_tile(tid, ordered)
            else:
                out[tid] = ordered
    for t, n in need.items():                  # tiles with no boxes still produce a record
        if n == 0:
            if on_tile:
                on_tile(t, [])
            else:
                out[t] = []
    assert not acc, f"incomplete tiles {list(acc)[:3]}"
    return out


# ------------------------------------------------------------------ val selection
def val_gt_boxes():
    """{tid: xywh list} + {tid: (gt RLEs @2048, valid)} for the 108 val tiles."""
    vg = json.load(open("/root/proj/boxinst_commonality_tcd_04/modal_tcd_multiseed/phase4/val_gt.json"))
    boxes, gts = {}, {}
    for t in TC.manifest()["ids"]["val"]:
        b = GP.gt_boxes_2048(vg[t]["trees"]).astype(np.float64)
        boxes[t] = [[x0, y0, x1 - x0, y1 - y0] for x0, y0, x1, y1 in b.tolist()]
        gts[t] = GP.gt_rles_and_valid(vg[t]["trees"], res=2048)
    return boxes, gts


def val_iou(model, args, tids, boxes, gts, workers=6):
    sub = {t: boxes[t] for t in tids}
    res = run_boxes(model, args, sub, workers=workers)
    ious = []
    for t in tids:
        g, valid = gts[t]
        pred = [window_rle(b, c) for (c, b, _) in res[t]]
        from pycocotools import mask as mu
        pr = [{"size": p["size"], "counts": p["counts"].encode()} for p in pred]
        iou = GP.per_crown_iou_rle(pr, g)
        ious += iou[valid].tolist()
    return ious


# ------------------------------------------------------------------ callbacks (training)
def _callbacks(run, args, smoke):
    import pytorch_lightning as pl
    import torch.distributed as dist

    class ValSelect(pl.Callback):
        """Epoch end: save this epoch's weights (rank 0), then val IoU on all ranks, then record.
        Keyed by epoch with last-write-wins, so a retried epoch re-saves AND re-scores together."""

        def __init__(self):
            self.boxes = self.gts = None

        def on_train_epoch_end(self, trainer, pl_module):
            e = trainer.current_epoch
            rank, world = trainer.global_rank, trainer.world_size
            sel = os.path.join(run, "sel", f"epoch{e}.pt")
            if rank == 0:
                os.makedirs(os.path.dirname(sel), exist_ok=True)
                torch.save({"state_dict": pl_module.state_dict(), "epoch": e,
                            "global_step": trainer.global_step}, sel + ".tmp")
                os.replace(sel + ".tmp", sel)
            if self.boxes is None:
                self.boxes, self.gts = val_gt_boxes()
            tids = sorted(self.boxes)[rank::world]
            t0 = time.time()
            was = pl_module.training
            pl_module.eval()
            ious = val_iou(pl_module, args, tids, self.boxes, self.gts, workers=4)
            pl_module.train(was)
            allv = [None] * world
            if world > 1:
                dist.all_gather_object(allv, ious)
            else:
                allv = [ious]
            if rank == 0:
                v = np.concatenate([np.asarray(a, np.float64) for a in allv])
                rec = {"epoch": e, "global_step": trainer.global_step, "val_mean_iou": float(v.mean()),
                       "val_n_crowns": int(len(v)), "ge50": float((v >= .5).mean()),
                       "ge75": float((v >= .75).mean()), "ckpt": sel,
                       "sec": round(time.time() - t0, 1), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                st = _PLATEAU.record(e + 1, rec["val_mean_iou"])          # 1-based epochs
                rec["plateau"] = {k: st[k] for k in ("best", "best_epoch", "bad", "decays", "lr_factor", "stop", "reason")}
                rows = [r for r in TC.read_jsonl(os.path.join(run, "evals.jsonl")) if r["epoch"] != e]
                rows.append(rec)
                tmp = os.path.join(run, "evals.jsonl.tmp")
                with open(tmp, "w") as f:
                    f.write("".join(json.dumps(r) + "\n" for r in sorted(rows, key=lambda r: r["epoch"])))
                os.replace(tmp, os.path.join(run, "evals.jsonl"))
                best = max(rows, key=lambda r: (r["val_mean_iou"], -r["epoch"]))   # ties -> earlier
                TC.atomic_json(os.path.join(run, "best.json"), best, indent=1)
                print(f"[val] epoch {e} mean IoU {rec['val_mean_iou']:.4f} n={rec['val_n_crowns']} "
                      f"best epoch {best['epoch']} ({rec['sec']}s)", flush=True)
            if world > 1:
                dist.barrier()
            st = _PLATEAU.load()                    # every rank reads what rank 0 wrote
            if st["stop"]:
                print(f"[plateau] stop: {st['reason']}", flush=True)
                trainer.should_stop = True

    every = 2 if smoke else 20

    class Progress(pl.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.global_rank == 0 and batch_idx % every == 0:
                rec = {"epoch": trainer.current_epoch, "batch": batch_idx,
                       "global_step": trainer.global_step, "max_epochs": trainer.max_epochs,
                       "t": time.time(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "losses": {k: float(v) for k, v in trainer.callback_metrics.items()
                                  if k.startswith("train/")}}
                TC.atomic_json(os.path.join(run, "progress.json"), rec)
                TC.append_jsonl(os.path.join(run, "progress_log.jsonl"), rec)

    return [ValSelect(), Progress()]


def last_good_ckpt(ckdir):
    """Newest loadable `epoch=N.ckpt` (PL 1.8 names them exactly so). `last.ckpt` is a copy written
    AFTER the epoch file, so it is never newer and is ignored. Called once, by the Modal parent,
    which passes the choice to every DDP rank via --resume-ckpt (deterministic, one load)."""
    import re
    if not os.path.isdir(ckdir):
        return None
    eps = sorted((f for f in os.listdir(ckdir) if re.fullmatch(r"epoch=\d+\.ckpt", f)),
                 key=lambda f: int(re.match(r"epoch=(\d+)", f).group(1)), reverse=True)
    for f in eps:
        c = os.path.join(ckdir, f)
        try:
            torch.load(c, map_location="cpu")
            return c
        except Exception as e:                      # torn by a kill mid-write
            print(f"[resume] {c} unreadable ({type(e).__name__}); trying older", flush=True)
    return None


_PLATEAU = None


def _plateau_lr(self, optimizer, local_step):
    """Replaces MAL.set_lr_per_iteration (released: 1-epoch linear warm-up, then cosine to 0.2 lr):
    the same warm-up, then base lr x the plateau factor (plateau.py), as for B1/B2. Disclosed."""
    args = self.args
    epoch = 1. * local_step / self._num_iter_per_epoch + self.current_epoch
    lr = args.lr * (epoch / args.warmup_epochs) if epoch < args.warmup_epochs else args.lr
    lr *= _PLATEAU.lr_factor
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_group["lr_scale"] if "lr_scale" in param_group else lr
    return lr


def train(a):
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from models.mal import MAL
    from datasets.pl_data_module import WSISDataModule
    run = run_dir(a.smoke)
    os.makedirs(run, exist_ok=True)
    patch_datapaths()
    n_train = len(json.load(open(TRAIN_JSON))["annotations"])
    ng = a.num_gpus                       # 8 for the real run; the smoke uses fewer, same per-GPU batch
    # steps per epoch exactly as MAL computes them: len(loader) // n_gpus, after its own filter
    probe_args = mal_args(train_extra(1, a.smoke)[0], ng)
    dm = WSISDataModule(num_workers=ng * probe_args.num_workers_per_gpu, load_train=True,
                        load_val=False, args=probe_args)
    n_kept = len(dm.train_dataloader().dataset)
    spe = len(dm.train_dataloader()) // ng
    extra, warm = train_extra(spe, a.smoke)
    args = mal_args(extra, ng)
    seed_everything(args.seed)
    dm = WSISDataModule(num_workers=ng * args.num_workers_per_gpu, load_train=True,
                        load_val=False, args=args)
    train_dl = dm.train_dataloader()
    global _PLATEAU
    from tierb import plateau as PLT
    _PLATEAU = PLT.Plateau(os.path.join(run, "plateau.json"), **(PLT.SMOKE if a.smoke else {}))
    MAL.set_lr_per_iteration = _plateau_lr
    if _PLATEAU.state["stop"] and int(os.environ.get("LOCAL_RANK", "0")) == 0 and \
            os.path.exists(os.path.join(run, "best.json")):         # stop decided before a dropout
        TC.mark_done(RUN + ("_smoke" if a.smoke else ""), "train",
                     {"plateau": _PLATEAU.state, "best": json.load(open(os.path.join(run, "best.json")))})
        return
    model = MAL(args=args, num_iter_per_epoch=len(train_dl))
    ckdir = os.path.join(run, "train", "ckpt")
    resume = a.resume_ckpt or None
    info = {"n_train_boxes": n_train, "n_kept_after_filter": n_kept, "min_obj_size": MIN_OBJ_SIZE,
            "steps_per_epoch_per_rank": spe, "warmups": warm, "lr": args.lr, "min_lr": args.min_lr,
            "max_epochs": args.max_epochs, "batch_per_gpu": args.batch_size, "n_gpus": ng,
            "resume_from": resume}
    if resume is None:
        info["labeler"] = load_labeler(model)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        info["pixels_verified"] = [TC.verify_split("train"), TC.verify_split("val")]   # every launch
        TC.report(RUN + ("_smoke" if a.smoke else ""), f"train_launch_{time.strftime('%Y%m%d_%H%M%S')}", info)
        print(json.dumps(info, indent=1, default=str), flush=True)
    ckcb = ModelCheckpoint(dirpath=ckdir, filename="{epoch}", save_top_k=-1,
                           every_n_epochs=1, save_last=True)
    kw = dict(gpus=args.gpus, num_nodes=1, strategy=args.strategy, accelerator="gpu",
              max_epochs=args.max_epochs, precision=16, callbacks=[ckcb] + _callbacks(run, args, a.smoke),
              logger=pl.loggers.TensorBoardLogger(os.path.join(run, "tb")),
              limit_val_batches=0, num_sanity_val_steps=0, enable_progress_bar=False)
    if a.smoke:
        kw["limit_train_batches"] = a.smoke_batches
    trainer = Trainer(**kw)
    trainer.fit(model, train_dataloaders=train_dl, ckpt_path=resume)
    if trainer.global_rank == 0:
        TC.mark_done(RUN + ("_smoke" if a.smoke else ""), "train",
                     {"plateau": _PLATEAU.load(), "best": json.load(open(os.path.join(run, "best.json"))), **info})


# ------------------------------------------------------------------ predict
def boxsets():
    return {os.path.splitext(f)[0]: os.path.join(BOXSETS, f) for f in sorted(os.listdir(BOXSETS))
            if f.endswith(".json")}


def predict_worker(local_rank, a):
    torch.cuda.set_device(local_rank)
    from models.mal import MAL
    run = run_dir(a.smoke)
    best = json.load(open(os.path.join(run, "best.json")))
    args = mal_args(["--dataset_type", "coco"], a.num_gpus)
    model = MAL(args=args, num_iter_per_epoch=1)
    ck = torch.load(best["ckpt"], map_location="cpu")
    assert ck["epoch"] == best["epoch"]
    model.load_state_dict(ck["state_dict"], strict=True)
    model.cuda().eval()
    rec = TC.stage_record()
    sets = boxsets()
    for name in a.sets.split(","):
        bs = json.load(open(sets[name]))
        d = os.path.join(run, "predict", name)
        os.makedirs(d, exist_ok=True)
        tids = sorted(bs["tiles"])[:a.limit_tiles or None][local_rank::a.num_gpus]   # smoke: first N
        todo = {t: bs["tiles"][t]["xywh"] for t in tids if not os.path.exists(os.path.join(d, f"{t}.json"))}
        for t in todo:
            TC.load_rgb(t, verify=rec["tiles"][t]["rgb_sha256"])

        def on_tile(tid, res):
            rles = [window_rle(b, c) for (c, b, _) in res]
            soft = [np.clip(np.round(w * 255), 0, 255).astype(np.uint8) for (_, _, w) in res]
            clips = np.asarray([c for (c, _, _) in res], np.int64).reshape(-1, 4)
            flat = np.concatenate([s.ravel() for s in soft]) if soft else np.zeros(0, np.uint8)
            np.savez_compressed(os.path.join(d, f"{tid}.soft.npz.tmp.npz"), clips=clips, flat=flat,
                                shapes=np.asarray([s.shape for s in soft], np.int64).reshape(-1, 2))
            os.replace(os.path.join(d, f"{tid}.soft.npz.tmp.npz"), os.path.join(d, f"{tid}.soft.npz"))
            src = bs["tiles"][tid]
            TC.atomic_json(os.path.join(d, f"{tid}.json"),
                           {"tid": tid, "set": name, "xywh": src["xywh"],
                            "boxes_2048": src["xyxy"], "scores": src.get("scores"),
                            "masks_rle": rles, "mask_thres": args.mask_thres,
                            "clips": clips.tolist(), "epoch": best["epoch"]})
        t0 = time.time()
        run_boxes(model, args, todo, workers=a.workers, want_soft=True, on_tile=on_tile)
        print(f"[predict] rank {local_rank} {name}: {len(todo)} tiles {time.time()-t0:.0f}s", flush=True)
    open(os.path.join(run, "predict", f".rank{local_rank}.done"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))


def predict(a):
    import torch.multiprocessing as mp
    mp.spawn(predict_worker, args=(a,), nprocs=a.num_gpus, join=True)
    run = run_dir(a.smoke)
    TC.mark_done(RUN + ("_smoke" if a.smoke else ""), "predict",
                 {"sets": a.sets.split(","), "best": json.load(open(os.path.join(run, "best.json")))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "train", "predict"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-batches", type=int, default=30)
    ap.add_argument("--sets", default="")
    ap.add_argument("--num-gpus", type=int, default=N_GPUS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume-ckpt", default="")
    ap.add_argument("--limit-tiles", type=int, default=0)
    a, _ = ap.parse_known_args()            # PL's DDP re-launch appends nothing we need
    os.chdir(MAL_ROOT)
    if a.cmd == "gate":
        from tierb.mal.mal_gate import gate
        gate(a)
    elif a.cmd == "train":
        train(a)
    else:
        predict(a)


if __name__ == "__main__":
    main()
