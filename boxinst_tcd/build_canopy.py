"""Build complete per-crop canopy masks (polygon AND RLE) — the ITC ignore region.

Canopy (category_id==1) is stored in TWO COCO formats: polygons and RLE. The polygon
loader silently dropped RLE, losing ~15% of canopy in our crops — and the RLE ones are
the LARGE closed-canopy regions (median ~765px) we most need to ignore. This rasterizes
both into one binary mask per crop, so canopy ignore / IoP is complete.

Operates on the EXISTING crops in split.json (no re-cropping, features/boxes untouched).
Writes boxinst_tcd/canopy_masks/<name>.png (binary). load_canopy_mask() is the single
source of truth used by the loss ignore-mask and the detection metric.

Usage:
    .venv/bin/python -m boxinst_tcd.build_canopy
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as maskutils

from boxinst_tcd.prepare import OUT, TCD_TRAIN, WIN

CANOPY_DIR = os.path.join(OUT, "canopy_masks")


def load_canopy_mask(tile_path, res=512):
    """(res,res) bool canopy mask for a crop; False everywhere if none."""
    f = os.path.join(CANOPY_DIR, os.path.splitext(os.path.basename(tile_path))[0] + ".png")
    if not os.path.exists(f):
        return np.zeros((res, res), bool)
    m = np.asarray(Image.open(f), bool)
    if m.shape != (res, res):
        m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((res, res)), bool)
    return m


def build():
    os.makedirs(CANOPY_DIR, exist_ok=True)
    split = json.load(open(os.path.join(OUT, "split.json")))
    meta_cache = {}
    n_poly = n_rle = 0
    for i, (p, t) in enumerate(split["tiles"].items()):
        src = t["source"]
        if src not in meta_cache:
            meta_cache[src] = json.load(open(os.path.join(TCD_TRAIN, src + "_meta.json")))
        meta = meta_cache[src]
        name = t["tile_id"]
        ox = int(name.split("_x")[1].split("_y")[0]); oy = int(name.split("_y")[1])
        full = np.zeros((meta["height"], meta["width"]), bool)   # full-tile canopy
        for a in json.loads(meta["coco_annotations"]):
            if a["category_id"] != 1:
                continue
            cx, cy, cw, ch = a["bbox"]
            if cx + cw <= ox or cx >= ox + WIN or cy + ch <= oy or cy >= oy + WIN:
                continue                                          # doesn't touch crop
            seg = a["segmentation"]
            if isinstance(seg, dict):                             # RLE
                rle = seg
                if isinstance(rle.get("counts"), list):           # uncompressed -> compress
                    rle = maskutils.frPyObjects(rle, meta["height"], meta["width"])
                full |= maskutils.decode(rle).astype(bool)
                n_rle += 1
            else:                                                 # polygon(s)
                img = Image.new("L", (meta["width"], meta["height"]), 0)
                d = ImageDraw.Draw(img)
                for ring in seg:
                    d.polygon([tuple(v) for v in np.array(ring).reshape(-1, 2)], fill=1)
                full |= np.asarray(img, bool)
                n_poly += 1
        crop = full[oy:oy + WIN, ox:ox + WIN]
        Image.fromarray(crop.astype(np.uint8) * 255).save(
            os.path.join(CANOPY_DIR, name + ".png"))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(split['tiles'])}")
    print(f"built {len(split['tiles'])} canopy masks (polygon anns={n_poly}, RLE anns={n_rle})")


if __name__ == "__main__":
    build()
