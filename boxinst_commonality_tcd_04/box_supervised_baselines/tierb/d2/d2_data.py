"""Training data for B1/B2: LACE's exact supervision, LACE's exact pixels.

  * boxes  = `train_tiles_gt.json` `boxes` (the list LACE's detector trained on, xyxy @2048);
  * canopy = `train_tiles_gt.json` `canopy` polygons, rasterised by LACE's own `canopy_px`
             (prep_local.py -> `/root/tierb_data/canopy/<tid>.png`, 0/1, sha256 in DATA_MANIFEST);
  * images = the staged HF `restor/tcd` pixels LACE's features were extracted from, hash-checked.
No crown polygon is ever read here: the dataset dicts carry rectangles only, and `gate` asserts the
mapped instances carry no mask field.

One dataset dict per (tile, quadrant): 792 x 4 = 3,168. The mapper resizes the tile by a released
scale s, takes that quadrant (crops.quadrant_offset), flips after the crop, and maps boxes and
canopy through the SAME transform list (canopy: nearest-neighbour, detectron2's segmentation path).
BoxTeacher additionally gets its released student colour augmentation (`strongv2`, AUG_EXTRA), copied
from boxteacher/dataset_mapper.py.
"""
from __future__ import annotations

import copy
import json
import random

import numpy as np
import torch
from PIL import Image

from tierb import crops as C
from tierb import tierb_common as TC

TRAIN_NAME = "tierb_train"


def train_dicts(train_gt_path, man=None):
    from detectron2.structures import BoxMode
    man = man or TC.manifest()
    gt = json.load(open(train_gt_path))
    tids = man["ids"]["train"]
    assert len(tids) == C.N_TRAIN_TILES and all(gt[t]["partition"] == "train" for t in tids)
    out = []
    for t in tids:
        annos = [{"bbox": [float(v) for v in b], "bbox_mode": BoxMode.XYXY_ABS,
                  "category_id": 0, "iscrowd": 0} for b in gt[t]["boxes"]]
        for q in range(C.CROPS_PER_TILE):
            out.append({"file_name": TC.rgb_path(t), "canopy_file": f"{TC.DATA}/canopy/{t}.png",
                        "height": C.TILE, "width": C.TILE, "image_id": f"{t}_q{q}",
                        "tierb_tid": t, "tierb_quadrant": q, "annotations": annos})
    assert len(out) == C.N_TRAIN_TILES * C.CROPS_PER_TILE
    return out


def register(train_gt_path):
    from detectron2.data import DatasetCatalog, MetadataCatalog
    if TRAIN_NAME not in DatasetCatalog.list():
        DatasetCatalog.register(TRAIN_NAME, lambda: train_dicts(train_gt_path))
        MetadataCatalog.get(TRAIN_NAME).set(thing_classes=["tree"])


class TierBMapper:
    """Training mapper. `aug_type`: 'none' (BoxInst) or 'strongv2' (BoxTeacher, + aug_extra)."""

    def __init__(self, image_format="BGR", aug_type="none", aug_extra=True, scales=C.SCALES,
                 hflip=True):
        self.image_format, self.aug_type, self.aug_extra = image_format, aug_type, aug_extra
        self.scales, self.hflip = tuple(scales), hflip

    def transforms_for(self, quadrant, s):
        from detectron2.data import transforms as T
        L = C.scaled_size(s)
        oy, ox = C.quadrant_offset(L, quadrant)
        augs = [T.Resize((L, L)), T.CropTransform(ox, oy, C.CROP, C.CROP)]
        if self.hflip:
            augs.append(T.RandomFlip(prob=0.5, horizontal=True, vertical=False))
        return augs, L

    def __call__(self, dataset_dict):
        from detectron2.data import detection_utils as utils
        from detectron2.data import transforms as T
        dd = copy.deepcopy(dataset_dict)
        image = utils.read_image(dd["file_name"], format=self.image_format)
        utils.check_image_size(dd, image)
        canopy = np.asarray(Image.open(dd["canopy_file"]), dtype=np.uint8)
        assert canopy.shape == image.shape[:2], (canopy.shape, image.shape)

        s = random.choice(self.scales)
        augs, L = self.transforms_for(dd["tierb_quadrant"], s)
        boxes = np.asarray([a["bbox"] for a in dd["annotations"]], dtype=np.float64).reshape(-1, 4)
        aug_input = T.AugInput(image, boxes=boxes, sem_seg=canopy)
        transforms = T.AugmentationList(augs)(aug_input)
        image, canopy = aug_input.image, aug_input.sem_seg
        image_shape = image.shape[:2]
        assert image_shape == (C.CROP, C.CROP), image_shape

        dd["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dd["canopy"] = torch.as_tensor(np.ascontiguousarray(canopy.astype(np.uint8)))
        dd["tierb_scale"] = L / C.TILE
        dd["height"], dd["width"] = image_shape

        if self.aug_type != "none":                 # boxteacher/dataset_mapper.py, verbatim ops
            from torchvision.transforms import transforms as tvt
            if self.aug_type == "strongv2":
                color_aug = tvt.Compose(
                    [tvt.RandomApply([tvt.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                     tvt.RandomGrayscale(0.2),
                     tvt.RandomApply([tvt.GaussianBlur(3)], p=0.5)])
            elif self.aug_type == "strong":
                color_aug = tvt.Compose(
                    [tvt.RandomApply([tvt.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                     tvt.RandomGrayscale(0.2)])
            else:
                raise ValueError(self.aug_type)
            # BGR -> RGB -> (aug) -> BGR
            aug_image_ = np.asarray(color_aug(Image.fromarray(image[:, :, ::-1])))[:, :, ::-1]
            key = "aug_image" if self.aug_extra else "image"
            dd[key] = torch.as_tensor(np.ascontiguousarray(aug_image_.transpose(2, 0, 1)))

        annos = [utils.transform_instance_annotations(obj, transforms, image_shape)
                 for obj in dd.pop("annotations") if obj.get("iscrowd", 0) == 0]
        instances = utils.annotations_to_instances(annos, image_shape)
        dd["instances"] = utils.filter_empty_instances(instances)
        dd.pop("canopy_file", None)
        return dd


def inference_input(rgb, oy=0, ox=0, size=None):
    """(H,W,3) RGB uint8 -> detectron2 input dict (BGR CHW), optionally a (size x size) window."""
    a = rgb if size is None else rgb[oy:oy + size, ox:ox + size]
    h, w = a.shape[:2]
    bgr = np.ascontiguousarray(a[:, :, ::-1].transpose(2, 0, 1))
    return {"image": torch.as_tensor(bgr), "height": h, "width": w}
