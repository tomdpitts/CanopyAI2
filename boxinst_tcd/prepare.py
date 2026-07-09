"""Sample OAM-TCD INDIVIDUAL-TREE crops (category_id==2) at NATIVE resolution,
representative across many biomes, with canopy (category_id==1) kept as an IGNORE
region (never a negative).

CATEGORY SEMANTICS (verified 2026-07-06 by rendering, see 06_annotation_check/):
  category 2 = individual tree crowns  -> POSITIVES (small compact crowns, p50~41px)
  category 1 = canopy / closed-canopy   -> IGNORE   (large regions, p95~1075px)
An earlier build used cat 1 by mistake (trained on canopy); this is the fix.

Canopy handling (per ITC intent): a detection landing in canopy is VALID, just not
individually labelled, so canopy must not be a negative. We store canopy polygons per
crop; the target encoder turns them into an ignore mask that removes those cells from
the detection NEGATIVE loss (positives inside canopy are still supervised).

Representative split: every biome with >= MIN_BIOME_SRC eligible source tiles is
included and split BY SOURCE TILE into train/val/test, so all biomes appear in every
partition and no crop from a val/test source leaks into train.

Products (strict separation): boxes.json (tree xyxy, TRAIN input) | gt_polys.json (tree
polygons, EVAL only) | canopy.json (canopy polygons, ignore-mask only).

Usage:
    .venv/bin/python -m boxinst_tcd.prepare
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TCD_TRAIN = os.path.join(REPO, "data/tcd/train")
OUT = os.path.join(REPO, "boxinst_tcd")
TILES = os.path.join(OUT, "tiles")
NATIVE, RES = 2048, 512
SEED = 0
WIN, STEP = 512, 320                 # native crop window / slide
TREE_MIN, TREE_MAX = 20, 180         # individual-tree box side (native px)
MIN_TREES, MAX_TREES = 4, 28         # trees fully inside a window
MIN_BIOME_SRC = 15                   # a biome needs this many source tiles to be in
MAX_PER_SOURCE = 4
CAP_TRAIN, CAP_VAL, CAP_TEST = 720, 160, 160


def trees_of(meta):
    return [a for a in json.loads(meta["coco_annotations"])
            if a["category_id"] == 2 and a.get("iscrowd", 0) == 0]


def canopy_of(meta):
    return [a for a in json.loads(meta["coco_annotations"])
            if a["category_id"] == 1]


def _compact(a):
    w, h = a["bbox"][2], a["bbox"][3]
    return TREE_MIN <= max(w, h) <= TREE_MAX and min(w, h) >= 8


def windows(meta, trees):
    for oy in range(0, NATIVE - WIN + 1, STEP):
        for ox in range(0, NATIVE - WIN + 1, STEP):
            inside = [a for a in trees
                      if a["bbox"][0] >= ox and a["bbox"][1] >= oy and
                      a["bbox"][0] + a["bbox"][2] <= ox + WIN and
                      a["bbox"][1] + a["bbox"][3] <= oy + WIN]
            if MIN_TREES <= len(inside) <= MAX_TREES:
                yield ox, oy, inside


def clip_poly(seg, ox, oy):
    if isinstance(seg, dict) or not seg:
        return []
    ring = max(seg, key=len)
    pts = np.array(ring, np.float32).reshape(-1, 2) - [ox, oy]
    return [round(float(v), 2) for v in np.clip(pts, 0, WIN - 1).reshape(-1)]


def main():
    os.makedirs(TILES, exist_ok=True)
    metas = sorted(f for f in os.listdir(TCD_TRAIN) if f.endswith("_meta.json"))
    by_biome = defaultdict(list)
    for mf in metas:
        meta = json.load(open(os.path.join(TCD_TRAIN, mf)))
        if meta["width"] != NATIVE:
            continue
        trees = [a for a in trees_of(meta) if _compact(a)]
        wins = list(windows(meta, trees))
        if wins:
            bn = meta.get("biome_name") or "unknown"
            by_biome[bn].append((mf, meta, wins))
    biomes = sorted(b for b, v in by_biome.items()
                    if b != "unknown" and len(v) >= MIN_BIOME_SRC)
    rng = np.random.default_rng(SEED)
    # per-biome source-tile split
    src_part = {}
    for b in biomes:
        items = sorted(by_biome[b], key=lambda t: t[0])
        rng.shuffle(items)
        n = len(items); n_te = max(1, round(n * 0.15)); n_va = max(1, round(n * 0.15))
        for i, (mf, _, _) in enumerate(items):
            src_part[mf] = "test" if i < n_te else "val" if i < n_te + n_va else "train"
    n_src = len(src_part)
    print(f"{len(biomes)} biomes (>= {MIN_BIOME_SRC} src each), {n_src} source tiles "
          f"({Counter(src_part.values())})")

    crops = {"train": [], "val": [], "test": []}
    for b in biomes:
        for (mf, meta, wins) in by_biome[b]:
            part = src_part[mf]
            ws = list(wins); rng.shuffle(ws)
            for (ox, oy, inside) in ws[:MAX_PER_SOURCE]:
                crops[part].append((mf, meta, ox, oy, inside, b))
    for k in crops:
        rng.shuffle(crops[k])
    caps = {"train": CAP_TRAIN, "val": CAP_VAL, "test": CAP_TEST}
    for k in crops:
        crops[k] = crops[k][:caps[k]]

    split = {"tiles": {}, "res": RES, "native_window": WIN, "seed": SEED,
             "class": "individual_tree(category_id=2)", "canopy": "category_id=1 (ignore)",
             "n_biomes": len(biomes), "split": "multi-biome, by-source-tile, native crops"}
    boxes_out, polys_out, canopy_out = {}, {}, {}
    ntree = ncanopy = 0
    for part in ("train", "val", "test"):
        for (mf, meta, ox, oy, inside, biome) in crops[part]:
            tid = mf.replace("_meta.json", "")
            crop = Image.open(os.path.join(TCD_TRAIN, tid + ".tif")).convert("RGB")
            crop = crop.crop((ox, oy, ox + WIN, oy + WIN))
            name = f"{tid}_x{ox}_y{oy}"
            dst = os.path.join(TILES, name + ".png")
            crop.save(dst)
            bxs, pls = [], []
            for a in inside:
                x, y, w, h = a["bbox"]
                bxs.append([round(x - ox, 2), round(y - oy, 2),
                            round(x + w - ox, 2), round(y + h - oy, 2)])
                pls.append(clip_poly(a["segmentation"], ox, oy))
            # canopy polygons intersecting the window (ignore region)
            cps = []
            for a in canopy_of(meta):
                cx, cy, cw, ch = a["bbox"]
                if cx + cw <= ox or cx >= ox + WIN or cy + ch <= oy or cy >= oy + WIN:
                    continue
                poly = clip_poly(a["segmentation"], ox, oy)
                if poly:
                    cps.append(poly)
            split["tiles"][dst] = {"partition": part, "n_boxes": len(bxs),
                                   "tile_id": name, "source": tid, "biome": biome,
                                   "n_canopy": len(cps)}
            boxes_out[dst] = bxs; polys_out[dst] = pls; canopy_out[dst] = cps
            ntree += len(bxs); ncanopy += len(cps)

    json.dump(split, open(os.path.join(OUT, "split.json"), "w"), indent=1)
    json.dump(boxes_out, open(os.path.join(OUT, "boxes.json"), "w"))
    json.dump(polys_out, open(os.path.join(OUT, "gt_polys.json"), "w"))
    json.dump(canopy_out, open(os.path.join(OUT, "canopy.json"), "w"))
    pc = Counter(t["partition"] for t in split["tiles"].values())
    print(f"wrote {len(split['tiles'])} crops ({dict(pc)}), {ntree} tree boxes, "
          f"{ncanopy} canopy ignore-polys, {len(biomes)} biomes")


if __name__ == "__main__":
    main()
