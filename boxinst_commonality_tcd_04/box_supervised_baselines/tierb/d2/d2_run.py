"""B1 (BoxInst R-101 3x) / B2 (BoxTeacher Swin-B 3x) inside the d2 image.

    python -m tierb.d2.d2_run gate    --method {boxinst,boxteacher}            # 1 GPU, ~10 min
    python -m tierb.d2.d2_run train   --method M --num-gpus 8 [--smoke ...]    # resumable
    python -m tierb.d2.d2_run predict --method M --num-gpus 8                  # resumable

Launched as a SUBPROCESS by tierb_modal.py, which owns the volume commits, the heartbeat lock and
the STOP flag (it touches STOP_FILE, which the StopHook below polls). Everything written to the
volume is written atomically; every stage resumes from what is already there.

TRAINING (crops.py has the arithmetic): released optimiser and LR at batch 16 = 8 GPUs x 2, the
same per-GPU batch as the release (the mask branch uses plain BN, so this matters); 36 epochs of
198 iterations; decays at 5,544 / 6,600; every scheduled decay happens; NO early stopping and no
wall-clock stop. [SUPERSEDED by TIERB.md section 0: convergence-driven plateau schedule (plateau.py),
min 4 / max 20 epochs, two x0.1 drops; the 36-epoch text above is the original plan.] Initialised from the released COCO box-supervised checkpoint; only the 80-way
classifier is re-initialised (asserted). Iteration-based loss warm-ups restart at our scaled
lengths (the checkpoint's `_iter` buffers are reset on a fresh start, never on resume).

SELECTION: after every epoch, val box AP50 on the 108 whole val tiles at 2048 (score_lite = the
frozen scorer's crowd arm, parity-gated), maxDets 600, floor 0.05. The best is kept as
model_best.pth with best.json, both persisted, so a resume can never re-select (the Box2Mask
resume trap, PLAN.md section 5d).

INFERENCE (both geometries, val-selected primary; TIERB.md section 4): native 0.1 m/px; 'whole' =
the 2048 tile in one pass (pre-NMS top-k 4000/level = the released 1000 per ~1 MP, area-scaled);
'sub' = 3x3 1024 subtiles at stride 512 (released 1000/level), stitched later by the seam-fixed
stitcher. Both: post-NMS cap 600 (protocol), FCOS candidate threshold on cls*ctr = PRE_THRESH below
(BoxInst 0.0025, BoxTeacher 6.25e-6) so every detection that can REPORT >= 0.05 is produced, NMS 0.6 (released), mask
threshold 0.5 (released).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import signal
import sys
import time

import numpy as np
import torch

from tierb import crops as C
from tierb import score_lite as SL
from tierb import tierb_common as TC

BT_ROOT = "/opt/BoxTeacher"
CFG_FILE = {"boxinst": f"{BT_ROOT}/AdelaiDet/configs/BoxInst/MS_R_101_3x.yaml",
            "boxteacher": f"{BT_ROOT}/projects/BoxTeacher/configs/coco/boxteacher_swin-base_3x.yaml"}
WEIGHTS = {"boxinst": f"{TC.VOL}/weights/BoxInst_MS_R_101_3x.pth",
           "boxteacher": f"{TC.VOL}/weights/boxteacher_swin_base_aug_3x.pth"}
TRAIN_GT = "/root/proj/boxinst_commonality_tcd_04/train_tiles_gt.json"
VAL_GT_CROWD = f"{TC.DATA}/val_gt_crowd_2048.json"
STOP_FILE = "/tmp/tierb_STOP"
STOPPED_FILE = "/tmp/tierb_STOPPED"

# Inference-only overrides of FCOSOutputs test attributes. The candidate threshold acts on cls*ctr
# (THRESH_WITH_CTR), and the REPORTED score is what the protocol floors at 0.05:
#   BoxInst    reports sqrt(cls*ctr)                        -> threshold 0.05^2      = 2.5e-3
#   BoxTeacher reports sqrt(sqrt(cls*ctr) * mask_score), mask_score <= 1
#                                                          -> threshold 0.05^2^2    = 6.25e-6
# so for both, every detection that could reach a reported 0.05 is produced, and the scorer's own
# 0.05 floor decides (released 0.05 on cls*ctr would floor BoxInst at 0.22 and BoxTeacher at 0.16).
PRE_THRESH = {"boxinst": 0.0025, "boxteacher": 6.25e-6}
CAPS = {
    "whole": {"pre_nms_topk_test": 4000, "post_nms_topk_test": 600},
    "sub": {"pre_nms_topk_test": 1000, "post_nms_topk_test": 600},
}
RELEASED_TEST = {"pre_nms_thresh_test": 0.05, "pre_nms_topk_test": 1000, "post_nms_topk_test": 100}
# BoxTeacher's TEACHER post-NMS cap is removed (Tom 2026-10-27, TIERB.md section 00). The released
# teacher runs in train mode (TEACHER_EVAL False) and keeps at most POST_NMS_TOPK_TRAIN = 100
# detections per crop for pseudo-masks; 2.7% of our 1024 crops hold more crowns (4.5% of training
# boxes would get no pseudo-mask). FCOSOutputs applies the cap only when post_nms_topk > 0, so 0 = no
# cap. Teacher only; the student, the released pre-NMS top-k and thresholds are unchanged.
TEACHER_UNCAPPED = {"post_nms_topk_train": 0, "post_nms_topk_test": 0}


def uncap_teacher(model, method):
    if method != "boxteacher":
        return
    fo = unwrap(model).teacher.proposal_generator.fcos_outputs
    for k, v in TEACHER_UNCAPPED.items():
        setattr(fo, k, v)


class TierBStopped(Exception):
    pass


class EarlyStopped(Exception):
    """plateau.py decided to stop (plateau after the last decay, or the max-epoch cap)."""


_PLATEAU = None          # plateau.Plateau, set in train_worker before the trainer is built


def run_dir(method, smoke=False):
    return TC.run_dir(method + ("_smoke" if smoke else ""))


# ------------------------------------------------------------------ config / model
def resume_seed(out_dir):
    """0 on a fresh start; otherwise the iteration the resume starts at, so the sampler and worker
    RNGs (seeded from cfg.SEED by default_setup) do not replay epoch 0's order after a resume."""
    tag = os.path.join(out_dir, "last_checkpoint")
    if not os.path.exists(tag):
        return 0
    name = open(tag).read().strip()
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) + 1 if digits else 0


def build_cfg(method, out_dir, sched, num_gpus, seed=0, smoke=False):
    from adet.config import get_cfg
    cfg = get_cfg()
    if method == "boxteacher":
        sys.path.insert(0, f"{BT_ROOT}/projects/BoxTeacher")
        from boxteacher import add_box_teacher_config
        add_box_teacher_config(cfg)
    cfg.merge_from_file(CFG_FILE[method])
    # released schedule landmarks, as fractions of our 36-epoch run (crops.schedule)
    assert tuple(cfg.INPUT.MIN_SIZE_TRAIN) == C.RELEASED_MIN_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN
    assert cfg.INPUT.MIN_SIZE_TEST == C.RELEASED_MIN_SIZE_TEST
    assert cfg.SOLVER.IMS_PER_BATCH == C.IMS_PER_BATCH
    assert tuple(cfg.SOLVER.STEPS) == C.RELEASED_3X["steps"] and cfg.SOLVER.MAX_ITER == C.RELEASED_3X["max_iter"]
    assert cfg.SOLVER.WARMUP_ITERS == C.RELEASED_3X["warmup"]
    assert cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS == C.RELEASED_3X["pairwise_warmup"]
    cfg.MODEL.WEIGHTS = WEIGHTS[method]
    cfg.MODEL.FCOS.NUM_CLASSES = 1
    cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED = 0        # COCO-specific (bottom-bar artefact)
    cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS = sched["pairwise_warmup_iters"]
    if method == "boxteacher":
        assert cfg.MODEL.BOX_TEACHER.WARMUP_ITERS == C.RELEASED_3X["boxteacher_mask_warmup"]
        cfg.MODEL.BOX_TEACHER.WARMUP_ITERS = sched["boxteacher_mask_warmup_iters"]
    cfg.DATASETS.TRAIN = ("tierb_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False    # empty crops kept (P3)
    cfg.DATALOADER.NUM_WORKERS = 3                     # 8 ranks x 3 on 24 cores (budget)
    cfg.SOLVER.MAX_ITER = sched["max_iter"]
    cfg.SOLVER.STEPS = tuple(sched["steps"])
    cfg.SOLVER.WARMUP_ITERS = sched["warmup_iters"]
    cfg.SOLVER.CHECKPOINT_PERIOD = sched["iters_per_epoch"]
    cfg.TEST.EVAL_PERIOD = 0
    cfg.SEED = seed
    cfg.OUTPUT_DIR = out_dir
    if smoke:            # smoke runs on fewer GPUs at the RELEASED per-GPU batch (2): same per-GPU
        cfg.SOLVER.IMS_PER_BATCH = 2 * num_gpus     # work and memory as the real 8 x 2 run
    assert cfg.SOLVER.IMS_PER_BATCH % num_gpus == 0
    assert smoke or cfg.SOLVER.IMS_PER_BATCH // num_gpus == 2, "real runs: 2 images per GPU (BN)"
    cfg.freeze()
    return cfg


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def infer_net(model, method):
    """The CondInst instance whose outputs are reported (BoxTeacher: teacher, as released)."""
    m = unwrap(model)
    if method == "boxteacher":
        return m.teacher if m.use_teacher_inference else m.student
    return m


@contextlib.contextmanager
def caps(model, method, geometry):
    """Inference-only test attributes on EVERY FCOSOutputs of the model; restored afterwards so
    training (BoxTeacher's teacher reads these during training) always sees the released values."""
    m = unwrap(model)
    nets = [m.teacher, m.student] if method == "boxteacher" else [m]
    want = dict(CAPS[geometry], pre_nms_thresh_test=PRE_THRESH[method])
    saved = [{k: getattr(n.proposal_generator.fcos_outputs, k) for k in want} for n in nets]
    for n in nets:
        for k, v in want.items():
            setattr(n.proposal_generator.fcos_outputs, k, v)
    try:
        yield
    finally:
        for n, sv in zip(nets, saved):
            for k, v in sv.items():
                setattr(n.proposal_generator.fcos_outputs, k, v)


def check_released_test_attrs(model, method):
    """Released test attributes everywhere, except the teacher's removed post-NMS cap (TEACHER_UNCAPPED)."""
    m = unwrap(model)
    nets = [m.teacher, m.student] if method == "boxteacher" else [m]
    for n in nets:
        want = dict(RELEASED_TEST, **(TEACHER_UNCAPPED if method == "boxteacher" and n is m.teacher else {}))
        for k, v in want.items():
            assert getattr(n.proposal_generator.fcos_outputs, k) == v, (k, v)


def ckpt_key_check(model, path, method):
    """The COCO checkpoint must load everything except the 80-way classifier."""
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("model", sd)
    msd = unwrap(model).state_dict()
    missing = sorted(k for k in msd if k not in sd)
    unexpected = sorted(k for k in sd if k not in msd)
    mism = sorted(k for k in msd if k in sd and tuple(sd[k].shape) != tuple(msd[k].shape))
    cls = {"proposal_generator.fcos_head.cls_logits.weight",
           "proposal_generator.fcos_head.cls_logits.bias"}
    want = cls if method == "boxinst" else {f"{p}.{k}" for p in ("student", "teacher") for k in cls}
    assert set(mism) == want, f"shape mismatches {mism} != {sorted(want)}"
    assert not missing, f"model keys absent from checkpoint: {missing[:10]}"
    iters = {k: float(v.flatten()[0]) for k, v in sd.items() if k.endswith("mask_head._iter")}
    return {"n_ckpt": len(sd), "n_model": len(msd), "loaded": len(msd) - len(mism),
            "reinitialised": mism, "unexpected": unexpected, "ckpt_iter_buffers": iters}


def reset_loss_warmups(model, method):
    """Fresh start only: the released `_iter` buffers saturate every loss warm-up; restart them so
    each runs its released fraction of OUR schedule (crops.schedule)."""
    m = unwrap(model)
    heads = [m.student.mask_head, m.teacher.mask_head] if method == "boxteacher" else [m.mask_head]
    before = [float(h._iter.item()) for h in heads]
    for h in heads:
        h._iter.zero_()
    return before


# ------------------------------------------------------------------ inference
def _rle(masks_bool_cpu):
    from pycocotools import mask as mu
    if len(masks_bool_cpu) == 0:
        return []
    arr = np.asfortranarray(masks_bool_cpu.numpy().transpose(1, 2, 0).astype(np.uint8))
    out = mu.encode(arr)
    return [{"size": r["size"], "counts": r["counts"].decode("ascii")} for r in out]


@torch.no_grad()
def run_window(model, method, rgb, oy=0, ox=0, size=None, masks=True):
    """One detectron2 forward on the tile (size=None) or a size x size window. Returns raw outputs
    in WINDOW coordinates: boxes (N,4) float64 list, scores, optional fcos_scores, mask RLEs."""
    from tierb.d2.d2_data import inference_input
    inp = inference_input(rgb, oy, ox, size)
    inst = unwrap(model)([inp])[0]["instances"]
    boxes = inst.pred_boxes.tensor.detach().cpu().double().tolist()
    scores = inst.scores.detach().cpu().double().tolist()
    rec = {"boxes": boxes, "scores": scores}
    if inst.has("tierb_fcos_scores"):
        rec["fcos_scores"] = inst.tierb_fcos_scores.detach().cpu().double().tolist()
    if masks:
        rles = []
        for c in range(0, len(inst), 64):
            rles += _rle(inst.pred_masks[c:c + 64].detach().cpu().bool())
        rec["masks_rle"] = rles
    return rec


def predict_tile(model, method, rgb, geometry, masks=True):
    if geometry == "whole":
        with caps(model, method, "whole"):
            r = run_window(model, method, rgb, masks=masks)
        out = {"boxes_2048": r["boxes"], "scores": r["scores"]}
        if masks:
            out["masks_rle"] = r["masks_rle"]
        if "fcos_scores" in r:
            out["fcos_scores"] = r["fcos_scores"]
        return out
    subs = {}
    with caps(model, method, "sub"):
        for (oy, ox) in C.subtile_origins():
            r = run_window(model, method, rgb, oy, ox, C.CROP, masks=masks)
            subs[(oy, ox)] = r
    return subs


# ------------------------------------------------------------------ hooks
def _hooks():
    from detectron2.engine import HookBase
    from detectron2.utils import comm

    class StopHook(HookBase):
        """Polls STOP_FILE (rank 0, every `every` iterations, broadcast) -> checkpoint -> stop."""

        def __init__(self, every=10):
            self.every = every

        def after_step(self):
            if (self.trainer.iter + 1) % self.every:
                return
            flag = torch.tensor([int(comm.is_main_process() and os.path.exists(STOP_FILE))],
                                device="cuda")
            if comm.get_world_size() > 1:
                torch.distributed.broadcast(flag, 0)
            if int(flag.item()):
                if comm.is_main_process():
                    self.trainer.checkpointer.save(f"model_{self.trainer.iter:07d}")
                    open(STOPPED_FILE, "w").write(str(self.trainer.iter))
                comm.synchronize()
                raise TierBStopped(self.trainer.iter)

    class ProgressHook(HookBase):
        def __init__(self, path, ipe, every=20):
            self.path, self.ipe, self.every, self.t0 = path, ipe, every, None

        def before_train(self):
            self.t0, self.i0 = time.time(), self.trainer.iter

        def after_step(self):
            it = self.trainer.iter + 1
            if not comm.is_main_process() or (it % self.every and it != self.trainer.max_iter):
                return
            el = time.time() - self.t0
            rate = el / max(it - self.i0, 1)
            TC.atomic_json(self.path, {"iter": it, "max_iter": self.trainer.max_iter,
                                       "epoch": round(it / self.ipe, 3), "sec_per_iter": rate,
                                       "eta_s": rate * (self.trainer.max_iter - it),
                                       "time": time.strftime("%Y-%m-%d %H:%M:%S")})

    class EvalSelectHook(HookBase):
        """Per-epoch val box AP50 on 108 whole tiles; keeps model_best.pth + best.json."""

        def __init__(self, method, run, period, val_tids, val_hash):
            self.method, self.run, self.period = method, run, period
            self.val_tids, self.val_hash = val_tids, val_hash
            self.evals = os.path.join(run, "evals.jsonl")
            self.best_json = os.path.join(run, "best.json")
            self.gt = self.t2i = None

        def before_train(self):
            # model_best.pth is authoritative; a kill between it and best.json is repaired here
            if comm.is_main_process():
                mb = os.path.join(self.run, "model_best.pth")
                if os.path.exists(mb):
                    ck = torch.load(mb, map_location="cpu")
                    cur = json.load(open(self.best_json)) if os.path.exists(self.best_json) else None
                    if cur is None or cur["val_box"] != ck["val_box"] or cur["iter"] != ck["iteration"] + 1:
                        TC.atomic_json(self.best_json, {"iter": ck["iteration"] + 1, "val_box": ck["val_box"],
                                                        "epoch": round((ck["iteration"] + 1) / self.period, 3),
                                                        "reconciled_from": "model_best.pth"}, indent=1)
            comm.synchronize()
            # A container lost DURING an epoch-end eval resumes from that epoch's checkpoint, i.e.
            # at start_iter == k*period, so the after_step that would evaluate it never re-runs.
            # Evaluate it here (the loaded weights ARE that checkpoint) before training continues.
            si = self.trainer.start_iter
            if si > 0 and (si % self.period == 0 or si == self.trainer.max_iter):
                self.evaluate(si)
                self._check_stop(si)                # a stop decided before the container died

        def after_step(self):
            nxt = self.trainer.iter + 1
            if nxt % self.period == 0 or nxt == self.trainer.max_iter:
                self.evaluate(nxt)

        def evaluate(self, nxt):
            if nxt in {r["iter"] for r in TC.read_jsonl(self.evals)}:
                return                                  # evaluated before a resume: never twice
            t0 = time.time()
            model = unwrap(self.trainer.model)
            # DDP runs with broadcast_buffers=False (detectron2), so the mask branch's BN running
            # stats differ per rank; rank 0's model is the one saved and predicted. Evaluate THAT
            # model on every rank, then restore each rank's own buffers (released training).
            stash = None
            if comm.get_world_size() > 1:
                stash = [b.detach().clone() for b in model.buffers()]
                for b in model.buffers():
                    torch.distributed.broadcast(b, 0)
            model.eval()
            rank, world = comm.get_rank(), comm.get_world_size()
            preds = {}
            for t in self.val_tids[rank::world]:
                rgb = TC.load_rgb(t)
                preds[t] = predict_tile(model, self.method, rgb, "whole", masks=False)
            model.train()
            if stash is not None:
                with torch.no_grad():
                    for b, v in zip(model.buffers(), stash):
                        b.copy_(v)
            check_released_test_attrs(model, self.method)
            allp = comm.gather(preds, dst=0)
            if comm.is_main_process():
                merged = {k: v for d in allp for k, v in d.items()}
                assert len(merged) == len(self.val_tids)
                if self.gt is None:
                    self.gt = SL.gt_from_json(json.load(open(VAL_GT_CROWD)))
                    self.t2i = {im["file_name"]: im["id"] for im in self.gt["images"]}
                    assert set(self.t2i) == set(self.val_tids)
                box = SL.score(merged, self.gt, self.t2i, masks=False)["box"]
                rec = {"iter": nxt, "epoch": round(nxt / self.period, 3), "val_box": box,
                       "sec": round(time.time() - t0, 1), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                # order matters for a kill at any instant: weights -> best.json -> evals.jsonl, so an
                # eval is only ever recorded once its best-candidate has been persisted
                best = json.load(open(self.best_json)) if os.path.exists(self.best_json) else None
                if best is None or box["AP50"] > best["val_box"]["AP50"]:   # ties -> earlier
                    p = os.path.join(self.run, "model_best.pth")
                    torch.save({"model": model.state_dict(), "iteration": nxt - 1,
                                "val_box": box}, p + ".tmp")
                    os.replace(p + ".tmp", p)
                    TC.atomic_json(self.best_json, rec, indent=1)
                    logging.getLogger("tierb").info(f"[eval] NEW BEST iter {nxt} val box AP50 {box['AP50']}")
                st = _PLATEAU.record(int(round(nxt / self.period)), box["AP50"])
                rec["plateau"] = {k: st[k] for k in ("best", "best_epoch", "bad", "decays", "lr_factor", "stop", "reason")}
                TC.append_jsonl(self.evals, rec)
                logging.getLogger("tierb").info(f"[eval] iter {nxt} val box {box} ({rec['sec']}s) "
                                                f"plateau {rec['plateau']}")
            comm.synchronize()
            self._check_stop(nxt)

        def _check_stop(self, it):
            st = _PLATEAU.load()                    # every rank reads what rank 0 wrote
            if st["stop"] and it < self.trainer.max_iter:
                raise EarlyStopped(f"{st['reason']} at iter {it}")

    return StopHook, ProgressHook, EvalSelectHook


def _atomic_checkpointer():
    from detectron2.checkpoint import DetectionCheckpointer

    class AtomicCheckpointer(DetectionCheckpointer):
        """fvcore Checkpointer.save, but tmp + os.replace for the file AND the last_checkpoint tag,
        so a container killed mid-write never leaves a torn checkpoint behind the tag."""

        def save(self, name, **kwargs):
            if not self.save_dir or not self.save_to_disk:
                return
            data = {"model": self.model.state_dict()}
            for key, obj in self.checkpointables.items():
                data[key] = obj.state_dict()
            data.update(kwargs)
            basename = f"{name}.pth"
            save_file = os.path.join(self.save_dir, basename)
            self.logger.info("Saving checkpoint to {}".format(save_file))
            with open(save_file + ".tmp", "wb") as f:
                torch.save(data, f)
                f.flush(); os.fsync(f.fileno())
            os.replace(save_file + ".tmp", save_file)
            self.tag_last_checkpoint(basename)

        def tag_last_checkpoint(self, last_filename_basename):
            p = os.path.join(self.save_dir, "last_checkpoint")
            with open(p + ".tmp", "w") as f:
                f.write(last_filename_basename)
                f.flush(); os.fsync(f.fileno())
            os.replace(p + ".tmp", p)

    return AtomicCheckpointer


def repair_last_checkpoint(out_dir):
    """If the tagged checkpoint does not load, point the tag at the newest one that does."""
    tag = os.path.join(out_dir, "last_checkpoint")
    if not os.path.exists(tag):
        return None
    cands = sorted((f for f in os.listdir(out_dir) if f.startswith("model_") and f.endswith(".pth")
                    and f != "model_best.pth"), reverse=True)
    cur = open(tag).read().strip()
    order = [cur] + [c for c in cands if c != cur]
    for c in order:
        try:
            torch.load(os.path.join(out_dir, c), map_location="cpu")
            if c != cur:
                with open(tag + ".tmp", "w") as f:
                    f.write(c)
                os.replace(tag + ".tmp", tag)
            return c
        except Exception as e:                                       # torn / truncated file
            print(f"[resume] {c} unreadable ({type(e).__name__}); trying older", flush=True)
    os.remove(tag)
    return None


# ------------------------------------------------------------------ trainer
def make_trainer_cls(method):
    from detectron2.data import build_detection_train_loader
    from detectron2.engine import DefaultTrainer, hooks
    from tierb.d2.d2_data import TierBMapper
    AtomicCheckpointer = _atomic_checkpointer()

    class TierBTrainer(DefaultTrainer):
        def __init__(self, cfg, extra_hooks):
            self._extra_hooks = extra_hooks
            super().__init__(cfg)
            self.checkpointer = AtomicCheckpointer(self.model, cfg.OUTPUT_DIR,
                                                   trainer=__import__("weakref").proxy(self))
            for h in self._hooks:                    # PeriodicCheckpointer must use the atomic one
                if isinstance(h, hooks.PeriodicCheckpointer):
                    h.checkpointer = self.checkpointer
                    h.max_to_keep = 3                # + model_best.pth (run dir) + model_final

        @classmethod
        def build_train_loader(cls, cfg):
            mapper = TierBMapper(cfg.INPUT.FORMAT,
                                 aug_type=cfg.INPUT.AUG_TYPE if method == "boxteacher" else "none",
                                 aug_extra=cfg.INPUT.AUG_EXTRA if method == "boxteacher" else True)
            return build_detection_train_loader(cfg, mapper=mapper)

        @classmethod
        def build_lr_scheduler(cls, cfg, optimizer):
            """Released linear warm-up, then base LR x the plateau factor (plateau.py)."""
            from fvcore.common.param_scheduler import ParamScheduler
            from detectron2.solver.lr_scheduler import LRMultiplier, WarmupParamScheduler

            class PlateauParam(ParamScheduler):
                def __call__(self, where):
                    return _PLATEAU.lr_factor
            sched = WarmupParamScheduler(PlateauParam(), cfg.SOLVER.WARMUP_FACTOR,
                                         cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER,
                                         cfg.SOLVER.WARMUP_METHOD)
            return LRMultiplier(optimizer, multiplier=sched, max_iter=cfg.SOLVER.MAX_ITER)

        @classmethod
        def build_optimizer(cls, cfg, model):
            if method == "boxteacher":
                return bt_build_optimizer(cfg, model)
            return super().build_optimizer(cfg, model)       # AdelaiDet: detectron2 default

        def build_hooks(self):
            ret = [h for h in super().build_hooks() if not isinstance(h, hooks.EvalHook)]
            if method == "boxteacher":
                from boxteacher import BoxTeacherEMAHook
                ret.append(BoxTeacherEMAHook(momentum=self.cfg.MODEL.BOX_TEACHER.MOMENTUM))
            # order: [timer, LR, (checkpointer), writer, EMA] + eval/select, progress, stop
            writer = [h for h in ret if isinstance(h, hooks.PeriodicWriter)]
            others = [h for h in ret if not isinstance(h, hooks.PeriodicWriter)]
            return others + self._extra_hooks + writer

    return TierBTrainer


def bt_build_optimizer(cfg, model):
    """projects/BoxTeacher/train_net.py Trainer.build_optimizer @6692d92, verbatim logic."""
    import itertools
    from detectron2.solver.build import maybe_add_gradient_clipping
    params, memo = [], set()
    for key, value in model.named_parameters(recurse=True):
        if not value.requires_grad:
            continue
        if value in memo:
            continue
        memo.add(value)
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "backbone" in key:
            lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    def maybe_add_full_model_gradient_clipping(optim):
        clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        enable = (cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                  and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model" and clip_norm_val > 0.0)

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)
        return FullModelGradientClippingOptimizer if enable else optim

    t = cfg.SOLVER.OPTIMIZER
    if t == "SGD":
        opt = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    elif t == "ADAMW":
        opt = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(params, cfg.SOLVER.BASE_LR)
    else:
        raise NotImplementedError(t)
    if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
        opt = maybe_add_gradient_clipping(cfg, opt)
    return opt


# ------------------------------------------------------------------ train
def _sched(args):
    s = C.schedule()
    if args.smoke:
        ipe = args.smoke_period
        s = dict(s, iters_per_epoch=ipe, max_iter=args.smoke_iters, smoke=True)
    return s


def _tf32():
    """torch 1.12 turned TF32 matmul OFF by default; torch <= 1.11 (BoxTeacher was trained on 1.9.1)
    had it ON on Ampere. Restoring it = the released numerics on A100, and ~1.5x faster Swin."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def train_worker(args):
    _tf32()
    from detectron2.engine import default_setup
    from detectron2.utils import comm
    from tierb.d2 import d2_patches as P
    from tierb.d2.d2_data import register
    if args.method == "boxteacher":
        sys.path.insert(0, f"{BT_ROOT}/projects/BoxTeacher")
    patches = P.apply(args.method)
    run = run_dir(args.method, args.smoke)
    out = os.path.join(run, "train")
    sched = _sched(args)
    if comm.is_main_process():
        repair_last_checkpoint(out)
    comm.synchronize()
    cfg = build_cfg(args.method, out, sched, args.num_gpus, seed=resume_seed(out), smoke=args.smoke)
    default_setup(cfg, argparse.Namespace(config_file="", eval_only=False, resume=True, opts=[]))
    from detectron2.utils.logger import setup_logger
    setup_logger(out, distributed_rank=comm.get_rank(), name="tierb")
    register(TRAIN_GT)
    man = TC.manifest()
    val_tids = man["ids"]["val"]
    if comm.is_main_process():
        ver = [TC.verify_split("train"), TC.verify_split("val")]     # every pixel, every launch
        TC.report(args.method + ("_smoke" if args.smoke else ""),
                  f"train_pixels_{time.strftime('%Y%m%d_%H%M%S')}", {"verified": ver})
    comm.synchronize()

    global _PLATEAU
    from tierb import plateau as PL
    _PLATEAU = PL.Plateau(os.path.join(run, "plateau.json"), **(PL.SMOKE if args.smoke else {}))
    StopHook, ProgressHook, EvalSelectHook = _hooks()
    extra = [EvalSelectHook(args.method, run, sched["iters_per_epoch"], val_tids, None),
             ProgressHook(os.path.join(run, "progress.json"), sched["iters_per_epoch"],
                          every=5 if args.smoke else 20),
             StopHook()]
    Trainer = make_trainer_cls(args.method)
    trainer = Trainer(cfg, extra)
    fresh = not trainer.checkpointer.has_checkpoint()
    if fresh and comm.is_main_process():
        keys = ckpt_key_check(trainer.model, cfg.MODEL.WEIGHTS, args.method)
        TC.report(args.method + ("_smoke" if args.smoke else ""), "train_init",
                  {"ckpt": keys, "sched": sched, "patches": patches, "num_gpus": args.num_gpus})
    trainer.resume_or_load(resume=True)
    if fresh:
        before = reset_loss_warmups(trainer.model, args.method)
        logging.getLogger("tierb").info(f"[init] fresh start; loss warm-up _iter {before} -> 0")
    else:
        logging.getLogger("tierb").info(f"[init] RESUMED at iter {trainer.start_iter}")
    uncap_teacher(trainer.model, args.method)                          # every launch, fresh or resumed
    check_released_test_attrs(trainer.model, args.method)
    try:
        trainer.train()
    except TierBStopped as e:
        logging.getLogger("tierb").info(f"[stop] stopped cleanly after iter {e}")
        return
    except EarlyStopped as e:
        logging.getLogger("tierb").info(f"[plateau] {e}")
    if comm.is_main_process():
        TC.mark_done(args.method + ("_smoke" if args.smoke else ""), "train",
                     {"max_iter": sched["max_iter"], "plateau": _PLATEAU.load(),
                      "best": json.load(open(os.path.join(run, "best.json")))})


# ------------------------------------------------------------------ predict
def predict_worker(args):
    _tf32()
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model
    from detectron2.utils import comm
    from tierb.d2 import d2_patches as P
    if args.method == "boxteacher":
        sys.path.insert(0, f"{BT_ROOT}/projects/BoxTeacher")
    P.apply(args.method)
    run = run_dir(args.method, args.smoke)
    best = json.load(open(os.path.join(run, "best.json")))
    cfg = build_cfg(args.method, os.path.join(run, "train"), _sched(args), args.num_gpus, smoke=args.smoke)
    model = build_model(cfg)
    ck = torch.load(os.path.join(run, "model_best.pth"), map_location="cpu")   # authoritative
    if not (ck["iteration"] == best["iter"] - 1 and ck["val_box"] == best["val_box"]):
        print(f"[predict] best.json disagrees with model_best.pth; using model_best.pth "
              f"(iter {ck['iteration'] + 1}, val box {ck['val_box']})", flush=True)
        best = {"iter": ck["iteration"] + 1, "val_box": ck["val_box"], "reconciled_from": "model_best.pth"}
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    rank, world = comm.get_rank(), comm.get_world_size()
    man, rec = TC.manifest(), TC.stage_record()
    splits = args.splits.split(",")
    for split in splits:
        tids = man["ids"][split][:args.limit_tiles or None]       # smoke: first N tiles only
        for geom in ("whole", "sub"):
            d = os.path.join(run, "predict", split, geom)
            os.makedirs(d, exist_ok=True)
            n = 0
            t0 = time.time()
            mine = tids[rank::world]
            for k, t in enumerate(mine):
                done = os.path.join(d, f"{t}.json" if geom == "whole" else f"{t}.done")
                if os.path.exists(done):
                    continue
                rgb = TC.load_rgb(t, verify=rec["tiles"][t]["rgb_sha256"])
                out = predict_tile(model, args.method, rgb, geom, masks=True)
                if geom == "whole":
                    TC.atomic_json(done, dict(out, tid=t, geometry="whole"))
                else:
                    for (oy, ox), r in out.items():
                        lst = [{"score": s, "bbox": b, "segmentation": m}
                               | ({"fcos_score": r["fcos_scores"][i]} if "fcos_scores" in r else {})
                               for i, (s, b, m) in enumerate(zip(r["scores"], r["boxes"], r["masks_rle"]))]
                        TC.atomic_json(os.path.join(d, f"Prediction_{t}_{oy}_{ox}.json"), lst)
                    TC.atomic_json(done, {"tid": t, "n": {f"{oy}_{ox}": len(r["scores"])
                                                         for (oy, ox), r in out.items()}})
                n += 1
                if rank == 0 and (k + 1) % 5 == 0:
                    el = time.time() - t0
                    print(f"[predict] {split}/{geom} rank0 {k+1}/{len(mine)} {el:.0f}s", flush=True)
            comm.synchronize()
            if rank == 0:
                print(f"[predict] {split}/{geom} complete ({len(tids)} tiles)", flush=True)
    if comm.is_main_process():
        TC.mark_done(args.method + ("_smoke" if args.smoke else ""), "predict",
                     {"best": best, "splits": splits})


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "train", "predict"])
    ap.add_argument("--method", required=True, choices=["boxinst", "boxteacher"])
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-iters", type=int, default=80)
    ap.add_argument("--smoke-period", type=int, default=40)
    ap.add_argument("--splits", default="val,test")      # sparse-236: optional, later, from model_best
    ap.add_argument("--limit-tiles", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "gate":
        from tierb.d2.d2_gate import gate
        gate(args)
        return
    from datetime import timedelta
    from detectron2.engine import launch
    fn = train_worker if args.cmd == "train" else predict_worker
    # NCCL_ASYNC_ERROR_HANDLING=1 (image env) turns a hung collective into a crash after `timeout`
    # (-> retry -> resume) instead of an indefinite hang. Training collectives are seconds apart;
    # predict's per-geometry barrier can wait for the slowest rank's whole shard.
    to = timedelta(minutes=30) if args.cmd == "train" else timedelta(hours=6)
    launch(fn, args.num_gpus, num_machines=1, machine_rank=0, dist_url="auto", args=(args,), timeout=to)


if __name__ == "__main__":
    main()
