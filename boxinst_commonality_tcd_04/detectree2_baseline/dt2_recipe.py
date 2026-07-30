"""Faithful reproduction of DetecTree2's training recipe using Detectron2 directly.

DetecTree2 2.1.2 has no Python-3.8 wheel (the torch1.10/detectron2-0.6 base's Python), so we
reproduce its recipe rather than import the package. This is a line-for-line port of
detectree2.models.train's `setup_cfg`, `MyTrainer` (augmentations + AP50-early-stop loop) and
`LossEvalHook` — same config values, same augmentations, same best-checkpoint-by-AP50 model
selection — driving the SAME released weights (250312_flexi.pth). Only the geospatial I/O
layer (FlexibleDatasetMapper/rasterio) is dropped, since we feed fixed-pixel RGB PNG tiles via
standard COCO registration. So the model, config, weights and selection are DetecTree2's; the
data plumbing is ours.
"""
import json
import os

import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (DatasetMapper, build_detection_train_loader,
                             build_detection_test_loader)
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer
from detectron2.engine import hooks as d2hooks
from detectron2.engine.train_loop import HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.events import EventStorage

BASE_MODEL = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"


def setup_cfg(trains=("trees_train",), tests=("trees_val",), update_model=None,
              workers=2, ims_per_batch=2, gamma=0.1, backbone_freeze=3, warm_iter=120,
              momentum=0.9, batch_size_per_im=1024, base_lr=0.0003389, weight_decay=0.001,
              max_iter=1000, eval_period=100, out_dir="./out", resize="fixed", num_classes=1):
    """== detectree2.models.train.setup_cfg (verbatim values)."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_MODEL))
    cfg.DATASETS.TRAIN = trains
    cfg.DATASETS.TEST = tests
    cfg.DATALOADER.NUM_WORKERS = workers
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.GAMMA = gamma
    cfg.MODEL.BACKBONE.FREEZE_AT = backbone_freeze
    cfg.SOLVER.WARMUP_ITERS = warm_iter
    cfg.SOLVER.MOMENTUM = momentum
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = batch_size_per_im
    cfg.SOLVER.WEIGHT_DECAY = weight_decay
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.TEST.EVAL_PERIOD = eval_period
    cfg.RESIZE = resize
    cfg.INPUT.MIN_SIZE_TRAIN = 1000
    cfg.OUTPUT_DIR = out_dir
    os.makedirs(out_dir, exist_ok=True)
    cfg.MODEL.WEIGHTS = update_model if update_model else \
        model_zoo.get_checkpoint_url(BASE_MODEL)
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    return cfg


def _augs(cfg):
    """== MyTrainer.build_train_loader augmentations (resize='fixed')."""
    a = [T.RandomRotation(angle=[0, 360], expand=False),
         T.RandomFlip(prob=0.5, horizontal=True, vertical=False)]
    a += [T.RandomBrightness(0.7, 1.5), T.RandomLighting(0.7),
          T.RandomContrast(0.6, 1.3), T.RandomSaturation(0.8, 1.4)]
    a.append(T.ResizeShortestEdge([1000, 1000], 1333))
    return a


class APEarlyStopHook(HookBase):
    """== detectree2 LossEvalHook's model-selection core: every eval_period run val, track
    segm AP50, save the best checkpoint, early-stop after `patience` non-improving evals.
    (Val-loss logging is dropped; it does not affect selection.)

    Resume-safe: the AP history / best / patience-counter are persisted to `hist_path` after
    every eval and restored in before_train, so a preemption-restart (resume=True) continues
    model-selection instead of forgetting all pre-preemption evals."""

    def __init__(self, eval_period, patience, hist_path):
        self._period = eval_period
        self.patience = patience
        self.hist_path = hist_path
        self.counter = 0
        self.max_ap = 0.0

    def before_train(self):
        if os.path.exists(self.hist_path):
            d = json.load(open(self.hist_path))
            self.trainer.APs = list(d["APs"])
            self.max_ap = d["max_ap"]
            self.counter = d["counter"]
            print(f"[resume] AP history restored: {len(self.trainer.APs)} evals, "
                  f"best={self.max_ap:.3f}, counter={self.counter}", flush=True)

    def _eval(self):
        res = self.trainer.test(self.trainer.cfg, self.trainer.model)
        ap = float(res["segm"]["AP50"])
        if not np.isfinite(ap):
            ap = 0.0
        self.trainer.APs.append(ap)
        self.trainer.storage.put_scalar("val_AP50", ap)
        print(f"[eval] iter {self.trainer.iter+1} segm AP50={ap:.3f} "
              f"(best {self.max_ap:.3f})", flush=True)
        if ap > self.max_ap:
            self.max_ap = ap
            self.counter = 0
            self.trainer.checkpointer.save(f"model_{len(self.trainer.APs)}")
        else:
            self.counter += 1
        json.dump({"APs": self.trainer.APs, "max_ap": self.max_ap,
                   "counter": self.counter}, open(self.hist_path, "w"))

    def after_step(self):
        nxt = self.trainer.iter + 1
        if nxt == self.trainer.max_iter or (self._period > 0 and nxt % self._period == 0):
            self._eval()
            if self.counter >= self.patience:
                self.trainer.early_stop = True
                print(f"[early-stop] no AP50 gain in {self.patience} evals "
                      f"(best {self.max_ap:.3f})", flush=True)

    def after_train(self):
        if not self.trainer.APs:
            return
        idx = self.trainer.APs.index(max(self.trainer.APs)) + 1
        p = os.path.join(self.trainer.cfg.OUTPUT_DIR, f"model_{idx}.pth")
        if os.path.exists(p):
            self.trainer.checkpointer.load(p)
        print(f"[best] loaded {p} (AP50={max(self.trainer.APs):.3f})", flush=True)


class MyTrainer(DefaultTrainer):
    """== detectree2.models.train.MyTrainer: custom augs, COCOEvaluator, AP50 early-stop."""

    def __init__(self, cfg, patience):
        self.patience = patience
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=_augs(cfg),
                               use_instance_mask=cfg.MODEL.MASK_ON,
                               instance_mask_format=cfg.INPUT.MASK_FORMAT,
                               recompute_boxes=False)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, output_dir=output_folder or cfg.OUTPUT_DIR)

    def build_hooks(self):
        # drop detectron2's default EvalHook (our APEarlyStopHook does the eval) to avoid
        # double val inference; keep the rest (writer, checkpointer, lr scheduler).
        hooks = [h for h in super().build_hooks() if not isinstance(h, d2hooks.EvalHook)]
        hooks.insert(-1, APEarlyStopHook(self.cfg.TEST.EVAL_PERIOD, self.patience,
                                         os.path.join(self.cfg.OUTPUT_DIR, "ap_history.json")))
        return hooks

    def train(self):
        """== MyTrainer.train: DefaultTrainer loop that breaks on early_stop, then loads best."""
        start_iter, max_iter = self.start_iter, self.max_iter
        self.iter = self.start_iter = start_iter
        self.early_stop = False
        self.APs = []
        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                    if self.early_stop:
                        break
                self.iter += 1
            finally:
                self.after_train()
