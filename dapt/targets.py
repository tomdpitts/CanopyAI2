"""FROZEN CenterNet target encoder: boxes -> (heatmap, offset, size, reg_mask).

>>> This config is HEAD-SIDE and MUST be byte-identical across every arm
>>> (web / sat / dapt / shuffled-dapt). Do NOT tune it per-arm — that reintroduces
>>> the exact confound the experiment controls for. Freeze once; reuse unchanged.

Design (see dapt/PLAN.md §4 and the target-encoder rationale):
- Grid resolution = the 32x32 patch grid (stride 16). Localization inside a 16 px
  cell is recovered by the OFFSET branch, not by any upsample.
- Gaussian radius is CenterNet's IoU radius computed in GRID CELLS, then clamped to
  >= RADIUS_MIN_CELLS (>= 1 feature cell) so small crowns don't collapse to a
  one-pixel target. Radius is a logged KNOB: smaller Gaussians separate touching
  peaks better but hurt small-crown detectability. Run `python -m dapt.targets check`
  before training to see the realized radius distribution.
- Offset target in [0,1) spans the FULL 16 px cell (regressed unconstrained).
- Size target is NATURAL-LOG pixels (log w, log h), so the ~4x crown-size range
  doesn't let big crowns starve small ones in the L1 gradient.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TargetConfig:
    grid: int = 32              # 512 / 16
    stride: int = 16            # px per cell (= DINOv3 patch size)
    min_overlap: float = 0.7    # CenterNet gaussian_radius IoU target
    radius_min_cells: float = 1.0   # clamp: never below one feature cell


CFG = TargetConfig()            # the one frozen instance every arm imports


def gaussian_radius(det_h_cells: float, det_w_cells: float,
                    min_overlap: float = CFG.min_overlap) -> float:
    """CenterNet IoU-based radius (in the same units as the inputs = grid cells)."""
    h, w = det_h_cells, det_w_cells
    a1, b1, c1 = 1.0, (h + w), w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - math.sqrt(max(b1 * b1 - 4 * a1 * c1, 0.0))) / (2 * a1)
    a2, b2, c2 = 4.0, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 - math.sqrt(max(b2 * b2 - 4 * a2 * c2, 0.0))) / (2 * a2)
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(max(b3 * b3 - 4 * a3 * c3, 0.0))) / (2 * a3)
    return min(r1, r2, r3)


def _draw_gaussian(hm: np.ndarray, cx: int, cy: int, radius: int):
    """Splat an unnormalized 2D Gaussian (peak=1) onto hm via elementwise max."""
    sigma = (2 * radius + 1) / 6.0
    ax = np.arange(-radius, radius + 1)
    g = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma * sigma))
    G = hm.shape[0]
    x0, x1 = max(0, cx - radius), min(G, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(G, cy + radius + 1)
    gx0, gy0 = x0 - (cx - radius), y0 - (cy - radius)
    sub = hm[y0:y1, x0:x1]
    np.maximum(sub, g[gy0:gy0 + (y1 - y0), gx0:gx0 + (x1 - x0)], out=sub)


def encode(boxes: np.ndarray, cfg: TargetConfig = CFG):
    """boxes:(N,4) xyxy in tile px -> dict of numpy target maps + per-box radius log.

    Returns:
        heatmap (1,G,G), offset (2,G,G), size (2,G,G) log-px, reg_mask (G,G) bool,
        radii  list[float] realized (post-clamp) radius per box (for diagnostics),
        cell_collisions int  boxes whose centre cell already held a centre.
    """
    G, s = cfg.grid, cfg.stride
    hm = np.zeros((1, G, G), np.float32)
    offset = np.zeros((2, G, G), np.float32)
    size = np.zeros((2, G, G), np.float32)
    mask = np.zeros((G, G), bool)
    radii, collisions = [], 0

    for xmin, ymin, xmax, ymax in boxes:
        w = max(xmax - xmin, 1.0)
        h = max(ymax - ymin, 1.0)
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        gx = min(int(cx // s), G - 1)
        gy = min(int(cy // s), G - 1)
        r = max(gaussian_radius(h / s, w / s, cfg.min_overlap), cfg.radius_min_cells)
        radii.append(r)
        _draw_gaussian(hm[0], gx, gy, int(round(r)))
        if mask[gy, gx]:
            collisions += 1                       # touching pair merged into one cell
        offset[:, gy, gx] = (cx / s - gx, cy / s - gy)   # in [0,1), spans full cell
        size[:, gy, gx] = (math.log(w), math.log(h))     # natural-log pixels
        mask[gy, gx] = True

    return {"heatmap": hm, "offset": offset, "size": size, "reg_mask": mask,
            "radii": radii, "cell_collisions": collisions}


def _check():
    """Pre-training diagnostic: realized radius distribution over the whole cohort."""
    import json
    import os
    from dapt.data.cohort import load_boxes, REPO

    split = json.load(open(os.path.join(REPO, "dapt/data/split.json")))
    boxes_by_tile, _ = load_boxes(split["csv"])

    all_radii, all_wh, total_boxes, total_collisions = [], [], 0, 0
    clamped = 0
    for path in split["tiles"]:
        bx = boxes_by_tile.get(path)
        if bx is None:
            continue
        enc = encode(bx)
        for (xmin, ymin, xmax, ymax), r in zip(bx, enc["radii"]):
            raw = gaussian_radius((ymax - ymin) / CFG.stride,
                                  (xmax - xmin) / CFG.stride)
            if raw < CFG.radius_min_cells:
                clamped += 1
            all_wh.append((xmax - xmin, ymax - ymin))
        all_radii.extend(enc["radii"])
        total_boxes += len(bx)
        total_collisions += enc["cell_collisions"]

    r = np.array(all_radii)
    wh = np.array(all_wh)
    diam = wh.mean(axis=1)          # mean side in px per box
    print(f"config: grid={CFG.grid} stride={CFG.stride}px "
          f"min_overlap={CFG.min_overlap} radius_min_cells={CFG.radius_min_cells}")
    print(f"boxes: {total_boxes}   tiles: {len(split['tiles'])}")
    print(f"crown side (px):  min={wh.min():.0f}  p5={np.percentile(diam,5):.0f}  "
          f"median={np.median(diam):.0f}  p95={np.percentile(diam,95):.0f}  "
          f"max={wh.max():.0f}")
    print(f"radius (cells):   min={r.min():.2f}  median={np.median(r):.2f}  "
          f"max={r.max():.2f}   (1 cell = {CFG.stride}px)")
    print(f"clamped to min:   {clamped}/{total_boxes} boxes "
          f"({100*clamped/total_boxes:.0f}%) had raw radius < {CFG.radius_min_cells}")
    print(f"cell collisions:  {total_collisions}/{total_boxes} boxes "
          f"({100*total_collisions/total_boxes:.0f}%) share a centre cell "
          f"(touching -> merged peak; tighter radius helps separation)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        _check()
    else:
        print("usage: python -m dapt.targets check")
