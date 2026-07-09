"""Cohort loader for the shadow-prior efficiency study (133 azimuth scenes).

Each tile -> square-padded -> resized to 512 (azimuth-preserving: padding keeps
aspect ratio, so the annotated shadow direction stays valid). Boxes are scaled to
512 space. Unit of analysis is the scene (= base tile); 3 acquisitions WON/BRU/NEON.

Caches rgb512 (uint8) and the 32x32 crown-occupancy target per scene so feature
extraction and shadow recompute never re-read/re-resize the source tiles.
"""
from __future__ import annotations

import csv
import os
import re
import sys
from dataclasses import dataclass

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from shadow_prior.geometry import vector_to_azimuth  # noqa: E402

CSV_FILES = [os.path.join(ROOT, "data/finetune/phase22X_train.csv"),
             os.path.join(ROOT, "data/finetune/phase22X_val_fixed.csv")]
CACHE = os.path.join(os.path.dirname(__file__), "cache")
IMG = 512
PATCH = 16
GRID = IMG // PATCH  # 32


def base_scene(p):
    return re.sub(r"_rot\d+$", "", os.path.splitext(os.path.basename(p))[0])


@dataclass
class Rec:
    scene: str
    acq: str
    path: str
    azimuth: float
    boxes: np.ndarray   # (N,4) xyxy in 512 space


def load_cohort():
    rows = []
    for fn in CSV_FILES:
        if not os.path.exists(fn):
            continue
        rows += list(csv.DictReader(open(fn)))
    by_img = {}
    for r in rows:
        if r["shadow_angle"].strip() == "":
            continue
        p = r["image_path"]
        d = by_img.setdefault(p, {"acq": r["domain"], "boxes": [],
                                  "sx": float(r["shadow_x"]), "sy": float(r["shadow_y"])})
        if all(r[k].strip() for k in ("xmin", "ymin", "xmax", "ymax")):
            x0, y0, x1, y1 = (float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
            if x1 > x0 and y1 > y0:
                d["boxes"].append([x0, y0, x1, y1])
    recs = []
    for p, d in by_img.items():
        if not d["boxes"]:
            continue
        full = p if os.path.isabs(p) else os.path.join(ROOT, p)
        if not os.path.exists(full):
            continue
        w, h = Image.open(full).size
        side = max(w, h)
        s = IMG / side                       # square-pad (no distortion) then resize
        boxes = np.array(d["boxes"], np.float32) * s
        recs.append(Rec(base_scene(p), d["acq"], full,
                        vector_to_azimuth(d["sx"], d["sy"]), boxes))
    recs.sort(key=lambda r: r.scene)
    return recs


def load_rgb512(rec):
    im = Image.open(rec.path).convert("RGB")
    w, h = im.size
    side = max(w, h)
    sq = Image.new("RGB", (side, side))
    sq.paste(im, (0, 0))                      # pad bottom/right; box origin preserved
    return np.asarray(sq.resize((IMG, IMG), Image.BILINEAR), dtype=np.uint8)


def occupancy_target(boxes, grid=GRID, patch=PATCH):
    """(grid,grid) float {0,1}: patch is positive if any crown box overlaps its cell."""
    t = np.zeros((grid, grid), np.float32)
    for x0, y0, x1, y1 in boxes:
        c0, c1 = int(x0 // patch), int(np.ceil(x1 / patch))
        r0, r1 = int(y0 // patch), int(np.ceil(y1 / patch))
        t[max(r0, 0):min(r1, grid), max(c0, 0):min(c1, grid)] = 1.0
    return t


def box_centers(boxes):
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)


# ---- splits (scene unit, acquisition-stratified) -------------------------- #
def confirm_explore_split(recs, frac_confirm=0.30, seed=20260630):
    """Lock `frac_confirm` of scenes (stratified by acquisition) as CONFIRMATION."""
    rng = np.random.default_rng(seed)
    confirm, explore = [], []
    acqs = sorted(set(r.acq for r in recs))
    for a in acqs:
        idx = [i for i, r in enumerate(recs) if r.acq == a]
        rng.shuffle(idx)
        k = int(round(frac_confirm * len(idx)))
        cset = set(idx[:k])
        for i in idx:
            (confirm if i in cset else explore).append(recs[i])
    return explore, confirm


def subsample_scenes(recs, n, seed):
    """Pick n scenes stratified across acquisitions (proportional), seeded."""
    rng = np.random.default_rng(seed)
    acqs = sorted(set(r.acq for r in recs))
    by = {a: [r for r in recs if r.acq == a] for a in acqs}
    for a in by:
        rng.shuffle(by[a])
    picked, i = [], 0
    quotas = {a: max(1, round(n * len(by[a]) / len(recs))) for a in acqs}
    for a in acqs:
        picked += by[a][:quotas[a]]
    rng.shuffle(picked)
    return picked[:n]


if __name__ == "__main__":
    recs = load_cohort()
    import collections
    print(f"scenes={len(recs)} acq={dict(collections.Counter(r.acq for r in recs))}")
    ex, cf = confirm_explore_split(recs)
    print(f"explore={len(ex)} confirm={len(cf)} "
          f"(confirm acq={dict(collections.Counter(r.acq for r in cf))})")
    r0 = recs[0]
    rgb = load_rgb512(r0); occ = occupancy_target(r0.boxes)
    print(f"rec0 scene={r0.scene} acq={r0.acq} boxes={len(r0.boxes)} az={r0.azimuth:.3f} "
          f"rgb={rgb.shape} occ_fg={occ.mean():.3f}")
