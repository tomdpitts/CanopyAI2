"""Stratification helpers: tile-level joins and instance-level crown-overlap flags.

TILE LEVEL joins the 439 test tiles to `modal_sparse_tcd_multiseed/tile_index.json`, which
carries `canopy_frac` (canopy-closure proxy), `n_crowns` (density), `biome` and
`scene_clean` per tile. Verified join: all 439 test tids resolve, every record is
split=test / seen=False, no nulls, and sum(n_crowns) == 25705 == the GT tree count.

INSTANCE LEVEL ports the per-TP record from
`mps_tcd_multiseed_4phase/masker_lab/failure_analysis.py` -- crucially `touching`, i.e.
"this GT box overlaps another GT box" -- which that script computes but persists only as
group means. The original is not modified; `gate` checks this port against its recorded
means before any new number is reported.
"""
import json
import os

import numpy as np

from boxinst_commonality_tcd_04 import evaluate as E
from . import io

TILE_INDEX = os.path.join(io.SPARSE, "tile_index.json")

# Fixed, interpretable edges beat quartiles for a paper table: the reader can map a bin to
# a landscape. Reported alongside n per bin so an underpopulated bin is visible.
CLOSURE_EDGES = [0.0, 0.10, 0.25, 0.50, 1.01]
DENSITY_EDGES = [0, 10, 30, 80, 10**9]


def tile_meta(tids=None):
    """{tid: record} for the requested tids (default: all)."""
    m = {t["tid"]: t for t in json.load(open(TILE_INDEX))["tiles"]}
    return m if tids is None else {t: m[t] for t in tids}


def bin_of(v, edges):
    """Index of the half-open bin [edges[i], edges[i+1]) containing v."""
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return i
    return len(edges) - 2


def bin_label(edges, i, fmt="{:g}"):
    lo, hi = edges[i], edges[i + 1]
    return f"[{fmt.format(lo)}, {fmt.format(hi)})" if hi < 10**8 else f">={fmt.format(lo)}"


def gt_boxes(polys):
    """Tight box per GT polygon, 2048 tile coords. Ported from failure_analysis.py:38."""
    if not polys:
        return np.zeros((0, 4), np.float32)
    return np.array([[*np.asarray(p, float).reshape(-1, 2).min(0),
                      *np.asarray(p, float).reshape(-1, 2).max(0)] for p in polys], np.float32)


def box_iou_mat(a, b):
    """Ported from failure_analysis.py:41; checked against brute force by
    scripts/t04_strata_instance.py --selftest."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix0 = np.maximum(a[:, 0][:, None], b[:, 0][None]); iy0 = np.maximum(a[:, 1][:, None], b[:, 1][None])
    ix1 = np.minimum(a[:, 2][:, None], b[:, 2][None]); iy1 = np.minimum(a[:, 3][:, None], b[:, 3][None])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None] - inter + 1e-9)


def touching_flags(gb):
    """GT box overlaps another GT box. failure_analysis.py:106."""
    if len(gb) == 0:
        return np.zeros(0, bool)
    return (box_iou_mat(gb, gb) > 0).sum(1) > 1


def collect(preds_path, gt, progress=100):
    """One streaming pass -> per-tile IoU / scores / ignore, keyed by tid.

    Kept as matrices rather than reduced, so any subset AP can be pooled afterwards
    without re-decoding 160k masks per stratum.
    """
    out = {}
    for t in io.per_tile(preds_path, gt, progress=progress):
        gb = gt_boxes(gt[t["tid"]]["trees"])
        out[t["tid"]] = {"iou": E.mask_iou(t["pm"], t["gm"]),
                         "box_iou": box_iou_mat(t["boxes"], gb),
                         "scores": t["scores"], "ignore": t["ignore"],
                         "n_gt": len(t["gm"]), "n_pred": len(t["scores"])}
    return out


def pooled_ap(per, tids, iou_thr=0.5, key="iou"):
    """AP pooled over a SUBSET of tiles -- the same cut convention as
    modal_sparse_tcd_multiseed subsets_sparse_band.json. n_gt is the subset's own GT count,
    so bins are internally normalised and directly comparable."""
    tids = [t for t in tids if t in per]
    if not tids:
        return float("nan"), 0
    n_gt = sum(per[t]["n_gt"] for t in tids)
    ap = E._greedy_ap([per[t][key] for t in tids],
                      [per[t]["scores"] for t in tids],
                      [per[t]["ignore"] for t in tids], n_gt, iou_thr)
    return ap, n_gt


def pooled_ap_5095(per, tids):
    v = [pooled_ap(per, tids, t)[0] for t in E.IOU_50_95]
    return float(np.nanmean(v))
