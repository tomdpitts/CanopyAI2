"""Seam-aware stitch = `stitch_res` + the released pre-dedup interior-edge filter.

Why (NATIVE_RASTER.md "OPEN DEFECT", 2026-10-13): `stitch_res.dedup` is greedy by score, so a
crown truncated at its own subtile's edge can outscore the full crown from the overlapping
subtile; the fragment is kept and the full crown is ALSO kept (it only half-overlaps the
fragment). Deleting fragments after dedup is the wrong counterfactual, so the filter acts
BEFORE dedup, on each subtile prediction.

Released rule (detectree2 `models/outputs.py`, github.com/PatBall1/detectree2 @39da1ba):
`stitch_crowns(folder, shift=1)` keeps a crown only if its polygon lies `within`
`box_make(minx - buffer + shift, ..., minx + width + buffer - shift)`, i.e. the buffered tile
(= our 1024 subtile) inset by `shift` = 1 m = 10 px at 0.1 m/px. Polygons come from
`cv2.findContours` vertices projected with `rasterio.transform.xy` (pixel centres), so a mask
spanning columns c0..c1 is `within` the inset box iff c0 >= 10 and c1 + 1 <= 1024 - 10.
In pixel-edge bbox terms (x1 exclusive, as `stitch_res._bbox`): drop iff x0 < 10 or x1 > 1014.

Applied at INTERIOR subtile edges only. The outer 2048 frame is kept: the GT contains crowns
cut by the frame, and detectree2's own `tile_data(full_coverage=True)` places edge tiles so
the buffered box extends past the raster, i.e. frame crowns are never culled there either.

`edge_px=None` is the identity: the load path is then exactly `stitch_res.load_tile_crowns`
(gated byte-for-byte against the current 2048 files; see run_stitch_seam.py). Dedup,
serialisation and rounding are `stitch_res`'s, imported, never edited.

Box2Mask has no released tiling; it uses this same rule (same crops, grid and stitcher).
"""
from __future__ import annotations

import json
import os

from boxinst_commonality_tcd_04.native_raster import stitch_res as ST

TILE = 2048
DT2_SHIFT_PX = 10          # detectree2 stitch_crowns default shift = 1 m at 0.1 m/px


def interior_edge_cut(bb, oy, ox, edge_px, sub=ST.SUB, tile=TILE):
    """bb = (x0, y0, x1, y1) of the mask in its own subtile's pixel-edge coords (x1, y1
    exclusive). True iff the mask comes within `edge_px` of an edge of that subtile that is
    interior to the whole tile."""
    x0, y0, x1, y1 = bb
    if ox > 0 and x0 < edge_px:
        return True
    if ox + sub < tile and x1 > sub - edge_px:
        return True
    if oy > 0 and y0 < edge_px:
        return True
    if oy + sub < tile and y1 > sub - edge_px:
        return True
    return False


def load_tile_crowns(pred_dir, tid, res, min_score=0.1, edge_px=None):
    """`stitch_res.load_tile_crowns` with the optional pre-dedup filter.
    Returns (crowns, n_edge_dropped)."""
    crowns, n_drop = [], 0
    for (oy, ox) in ST.subtile_origins():
        fp = os.path.join(pred_dir, f"Prediction_{tid}_{oy}_{ox}.json")
        if not os.path.exists(fp):
            continue
        for inst in json.load(open(fp)):
            if float(inst["score"]) < min_score:
                continue
            m_sub = ST._decode_rle(inst["segmentation"])
            if edge_px is not None:
                bl = ST._bbox(m_sub)
                if bl is not None and interior_edge_cut(bl, oy, ox, edge_px):
                    n_drop += 1
                    continue
            m = ST._to_canvas(m_sub, oy, ox, res)
            bb = ST._bbox(m)
            if bb is not None:
                crowns.append((m, float(inst["score"]), bb))
    return crowns, n_drop
