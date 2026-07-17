"""Step 4b — build the NEON training patch set from the hand-annotated RGB tiles.

GENUINELY-ANNOTATED ONLY (no train/eval pollution):
  - Uses the 21 hand-annotated training-tile XMLs, but DROPS tiles whose annotations
    cover < COVER_MIN of the image (only a sub-region labelled) — those would mint
    false negatives. At COVER_MIN=0.85 this drops exactly 2019_SJER_4 (41.8%) and
    2019_TOOL (0.0%); every kept tile is ~85-100% annotated.
  - Patches are tiled ONLY within each tile's annotation bounding box, so we never
    crop into an unlabelled margin. Empty patches inside that box are REAL negatives
    (open ground), kept up to EMPTY_RATIO x positives to teach background without
    swamping the positives.
  - 400 px patches @ native 0.1 m/px = 40x40 m, identical geometry to the 194 eval
    tiles (400 px). Edge patches are flushed to the bbox edge (no partial overhang).

RGB-ONLY: reads only the downloaded RGB tiles + annotation XMLs. No LiDAR/CHM/HSI.

Val split is TILE-LEVEL (whole tiles held out) so no patch from a training tile leaks
into val — val is only the early-stopping / best-on-val signal; the REPORTED metric is
the fully-separate 194-tile NEON benchmark.

Usage: .venv/bin/python -m boxinst_commonality_tcd_04.mps_neon_multiseed.prepare_neon_train
"""
import glob
import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.abspath(os.path.dirname(__file__))
ANN_DIR = os.path.join(HERE, "NeonTreeEvaluation", "annotations")
IMG_DIRS = [os.path.join(HERE, "zenodo_dl"),                       # individual RGB
            os.path.join(HERE, "zenodo_dl", "training", "RGB")]    # from training.zip
OUT_IMG = os.path.join(HERE, "train_patches")
OUT_GT = os.path.join(HERE, "train_patches_gt.json")

PATCH = 400
STRIDE = 400                 # non-overlapping
COVER_MIN = 0.85             # drop tiles annotated over < this fraction of the image
EMPTY_RATIO = 0.3            # keep empties up to this x positives (per tile)
MIN_BOX = 4                  # drop boxes whose clipped side < this (px)
KEEP_FRAC = 0.5              # keep a clipped box only if >= this of its area survives
VAL_FRAC = 0.12              # target fraction of boxes held out for the val signal
VAL_MAX_TILE_FRAC = 0.25     # never put a tile holding > this of all boxes in val


def choose_val_tiles(tile_boxes):
    """Tile-level val split (no patch leakage): hold out whole tiles totalling ~VAL_FRAC
    of boxes, adding smallest-first and skipping any single tile that alone exceeds
    VAL_MAX_TILE_FRAC (keeps the big NIWO/TEAK tiles in train). Deterministic; adapts
    to whichever tiles are present."""
    total = sum(tile_boxes.values()) or 1
    val, acc = set(), 0
    for t, n in sorted(tile_boxes.items(), key=lambda kv: (kv[1], kv[0])):
        if n / total > VAL_MAX_TILE_FRAC:
            continue
        if acc / total >= VAL_FRAC:
            break
        val.add(t); acc += n
    return val


def find_image(fn):
    for d in IMG_DIRS:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    return None


def load_ann(xml_path):
    r = ET.parse(xml_path).getroot()
    fn = r.findtext("filename")
    sz = r.find("size")
    W, H = int(float(sz.find("width").text)), int(float(sz.find("height").text))
    b = np.array([[float(o.find("bndbox").find(t).text)
                   for t in ("xmin", "ymin", "xmax", "ymax")]
                  for o in r.findall("object")], np.float64).reshape(-1, 4)
    return fn, W, H, b


def patch_grid(x0, y0, x1, y1):
    """Non-overlapping PATCH-sized starts tiling [x0,x1) x [y0,y1); last one flush."""
    def starts(lo, hi):
        s = list(range(int(lo), int(hi) - PATCH + 1, STRIDE))
        if not s or s[-1] + PATCH < hi:
            s.append(max(int(lo), int(hi) - PATCH))
        return sorted(set(s))
    return [(px, py) for py in starts(y0, y1) for px in starts(x0, x1)]


def main():
    os.makedirs(OUT_IMG, exist_ok=True)
    xmls = [p for p in glob.glob(os.path.join(ANN_DIR, "*.xml"))
            if re.search(r"_image(_crop\d*)?$",
                         os.path.splitext(os.path.basename(p))[0])]
    # PASS 1: find usable tiles (annotated >= COVER_MIN AND imagery present) + box counts
    usable, stats, n_missing = {}, [], 0
    for xp in sorted(xmls):
        fn, W, H, boxes = load_ann(xp)
        if len(boxes) == 0:
            continue
        tile = os.path.splitext(os.path.basename(xp))[0]
        ax0, ay0 = boxes[:, 0].min(), boxes[:, 1].min()
        ax1, ay1 = boxes[:, 2].max(), boxes[:, 3].max()
        cover = ((ax1 - ax0) * (ay1 - ay0)) / (W * H)
        if cover < COVER_MIN:
            stats.append((tile, "DROP(cover=%.2f)" % cover, 0, 0))
            continue
        img_path = find_image(fn)
        if img_path is None:
            n_missing += 1
            stats.append((tile, "MISSING_IMG", 0, 0))
            continue
        usable[tile] = (img_path, boxes, (ax0, ay0, ax1, ay1))
    val_tiles = choose_val_tiles({t: len(v[1]) for t, v in usable.items()})
    print(f"[val] held-out tiles: {sorted(val_tiles)}")

    # PASS 2: crop patches from the annotation bbox of each usable tile
    gt = {}
    for tile in sorted(usable):
        img_path, boxes, (ax0, ay0, ax1, ay1) = usable[tile]
        img = Image.open(img_path).convert("RGB")
        part = "val" if tile in val_tiles else "train"
        pos, empt = [], []
        for (px, py) in patch_grid(ax0, ay0, ax1, ay1):
            # boxes with centre inside the patch, clipped to patch, kept if enough survives
            cx = (boxes[:, 0] + boxes[:, 2]) / 2
            cy = (boxes[:, 1] + boxes[:, 3]) / 2
            sel = (cx >= px) & (cx < px + PATCH) & (cy >= py) & (cy < py + PATCH)
            pb = boxes[sel].copy()
            keep = []
            for bx in pb:
                x0 = max(bx[0], px); y0 = max(bx[1], py)
                x1 = min(bx[2], px + PATCH); y1 = min(bx[3], py + PATCH)
                if x1 - x0 < MIN_BOX or y1 - y0 < MIN_BOX:
                    continue
                orig = (bx[2] - bx[0]) * (bx[3] - bx[1])
                if orig <= 0 or (x1 - x0) * (y1 - y0) / orig < KEEP_FRAC:
                    continue
                keep.append([x0 - px, y0 - py, x1 - px, y1 - py])
            rec = (px, py, keep)
            (pos if keep else empt).append(rec)
        # cap empties
        n_keep_empt = min(len(empt), int(round(EMPTY_RATIO * len(pos))))
        empt = empt[:: max(1, len(empt) // n_keep_empt)][:n_keep_empt] if n_keep_empt else []
        recs = pos + empt
        nbox = 0
        for i, (px, py, keep) in enumerate(recs):
            crop = img.crop((px, py, px + PATCH, py + PATCH))
            if crop.size != (PATCH, PATCH):            # flush-edge safety pad
                pad = Image.new("RGB", (PATCH, PATCH))
                pad.paste(crop, (0, 0)); crop = pad
            pid = f"{tile}__p{px}_{py}"
            crop.save(os.path.join(OUT_IMG, pid + ".png"))
            gt[pid] = {"boxes": keep, "partition": part, "src_tile": tile}
            nbox += len(keep)
        stats.append((tile, part, len(recs), nbox))
    json.dump(gt, open(OUT_GT, "w"))

    print(f"{'tile':44s} {'part':6s} {'patches':7s} {'boxes'}")
    for t, p, npat, nb in stats:
        print(f"{t:44s} {p:6s} {npat:7d} {nb}")
    tr = [k for k, v in gt.items() if v["partition"] == "train"]
    va = [k for k, v in gt.items() if v["partition"] == "val"]
    nb_tr = sum(len(gt[k]["boxes"]) for k in tr)
    nb_va = sum(len(gt[k]["boxes"]) for k in va)
    n_empty = sum(1 for v in gt.values() if not v["boxes"])
    print(f"\nTOTAL patches {len(gt)} (train {len(tr)} / val {len(va)}), "
          f"empty {n_empty}")
    print(f"TOTAL boxes {nb_tr + nb_va} (train {nb_tr} / val {nb_va})")
    if n_missing:
        print(f"[warn] {n_missing} tiles missing imagery (download incomplete?)")
    print(f"wrote {OUT_GT} + patches -> {OUT_IMG}/")


if __name__ == "__main__":
    main()
