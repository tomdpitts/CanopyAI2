"""Verify saved masks + export them as a COCO instance-segmentation file.

Each saved binary tile mask -> connected components (4-connectivity, matching the
Divide gaps) -> one instance per component (RLE segmentation). Prints per-tile
instance counts and flags any empty/missing tile.
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
EDIT = os.path.join(HERE, "pencil_tool", "edited")
TILE_DIR = os.path.join(HERE, "tiles", "WON")
OUT = os.path.join(HERE, "exports", "won_try_instances_coco.json")
MIN_AREA = 20   # drop specks from erasing/smoothing

expected = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(TILE_DIR, "*.png")))
saved = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(EDIT, "*.png")))
missing = [s for s in expected if s not in saved]

images, anns, aid = [], [], 1
per_tile = {}
for i, stem in enumerate(saved):
    m = np.array(Image.open(os.path.join(EDIT, stem + ".png")).convert("L")) > 127
    H, W = m.shape
    images.append({"id": i, "file_name": stem + ".png", "width": W, "height": H})
    lab, n = ndi.label(m)     # 4-connectivity (default cross structure)
    cnt = 0
    for k in range(1, n + 1):
        comp = (lab == k)
        if comp.sum() < MIN_AREA:
            continue
        rle = mask_util.encode(np.asfortranarray(comp[:, :, None].astype(np.uint8)))[0]
        rle["counts"] = rle["counts"].decode("ascii")
        anns.append({"id": aid, "image_id": i, "category_id": 1, "segmentation": rle,
                     "bbox": mask_util.toBbox(rle).tolist(), "area": float(mask_util.area(rle)),
                     "iscrowd": 0})
        aid += 1; cnt += 1
    per_tile[stem] = cnt

coco = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "tree"}]}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(coco, open(OUT, "w"))

print(f"expected tiles : {len(expected)}")
print(f"saved tiles    : {len(saved)}")
print(f"MISSING        : {missing if missing else 'none'}")
print(f"empty tiles    : {[s for s,c in per_tile.items() if c==0] or 'none'}")
print(f"total crowns   : {len(anns)}")
print("per tile       : " + ", ".join(f"{s.replace('won_tile_','t')}={c}" for s, c in per_tile.items()))
print(f"COCO written   : {OUT}")
sys.exit(1 if (missing or any(c == 0 for c in per_tile.values())) else 0)
