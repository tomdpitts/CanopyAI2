"""Extract GT for the official OAM-TCD 439-tile TEST benchmark (scored natively).

The 439 test tiles are 2048x2048. We do NOT cut crop files: features are cached
as one stitched 128x128 grid per tile (cache_test.py, 2x2 overlapping 1024
windows), and detection + masks run on that whole-tile grid — so there is no
crop-seam NMS. This module only reads the annotations:

  individual-tree crowns (category_id==2) -> GT polygons  (EVAL ONLY, never train)
  canopy (category_id==1)                 -> ignore polys  (a predicted crown in
      canopy is valid-but-unlabelled: COCO ignore, never a false positive)

Product: test_gt.json  {source_tile: {trees:[poly@2048], canopy:[poly@2048], W, H}}

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.prepare_test
"""
import json
import os

import contourpy
import numpy as np
import pycocotools.mask as M

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TCD_TEST = os.path.join(REPO, "data/tcd/test")
OUT = os.path.join(REPO, "boxinst_commonality_tcd_04")
RED = 4                              # mask block-reduce before contouring (=eval raster scale)


def anns_of(meta, cat):
    return [a for a in json.loads(meta["coco_annotations"])
            if a["category_id"] == cat]


def _rle_to_rings(seg, H, W):
    """RLE dict -> list of flat [x,y,...] polygon rings @ full (2048) coords.

    Handles compressed (counts bytes/str) and uncompressed (counts list) RLE.
    The mask is block-reduced by RED before tracing: canopy RLEs cover most of a
    tile, and eval rasters at 2048/RED anyway, so this keeps rings compact and
    eval-resolution-faithful. The reduced mask is ZERO-PADDED by one cell before
    contouring so regions that touch the tile border still close into fillable
    loops (else their 0.5 contour is an open curve that PIL fills to a sliver —
    the tile-188 canopy is 88% coverage but traced to 10% without the pad). Every
    contour is kept (RLE masks can be multi-component), so a prediction anywhere
    in the ignore region is covered.
    """
    rle = M.frPyObjects(seg, H, W) if isinstance(seg.get("counts"), list) else seg
    m = M.decode(rle)
    if m.ndim == 3:
        m = m.any(2)
    m = m.astype(bool)
    if not m.any():
        return []
    h, w = (H // RED) * RED, (W // RED) * RED
    red = m[:h, :w].reshape(h // RED, RED, w // RED, RED).any((1, 3))
    pad = np.pad(red, 1).astype(np.float32)               # close border-touching
    lines = contourpy.contour_generator(z=pad).lines(0.5)
    rings = []
    for ln in lines:
        if len(ln) < 3:
            continue
        xy = (ln - 1) * RED                               # de-pad, reduced -> 2048
        rings.append([round(float(v), 1) for v in xy.reshape(-1)])
    return rings


def seg_rings(seg, H, W):
    """COCO segmentation -> list of flat [x,y,...] rings (polygon OR RLE).

    dict -> RLE decode (all contours); list -> the largest polygon ring (prior
    behaviour, unchanged). Empty list if degenerate. FIXES the earlier bug where
    RLE dicts were silently dropped (467 anns / 210 tiles: 371 canopy + 96 ITC).
    """
    if isinstance(seg, dict):
        return _rle_to_rings(seg, H, W)
    if not seg:
        return []
    r = max(seg, key=len)
    return [[round(float(v), 2) for v in np.asarray(r, np.float32).reshape(-1)]]


def main():
    metas = sorted(f for f in os.listdir(TCD_TEST) if f.endswith("_meta.json"))
    gt = {}
    n_trees = n_canopy = n_empty = n_rle = 0
    for k, mf in enumerate(metas):
        meta = json.load(open(os.path.join(TCD_TEST, mf)))
        tid = mf.replace("_meta.json", "")
        H, W = meta["height"], meta["width"]
        trees, canopy = [], []
        for a in anns_of(meta, 2):                 # ITC = one GT instance per ann
            n_rle += isinstance(a["segmentation"], dict)
            rs = [r for r in seg_rings(a["segmentation"], H, W) if len(r) >= 6]
            if rs:
                trees.append(max(rs, key=len))     # largest contour = the crown
        for a in anns_of(meta, 1):                 # canopy = union ignore (all parts)
            n_rle += isinstance(a["segmentation"], dict)
            canopy += [r for r in seg_rings(a["segmentation"], H, W) if len(r) >= 6]
        gt[tid] = {"trees": trees, "canopy": canopy, "W": W, "H": H}
        n_trees += len(trees); n_canopy += len(canopy)
        n_empty += (not trees)
        if (k + 1) % 100 == 0 or k + 1 == len(metas):
            print(f"  {k+1}/{len(metas)}", flush=True)
    json.dump(gt, open(os.path.join(OUT, "test_gt.json"), "w"))
    print(f"439-test: {len(metas)} tiles ({n_empty} with no ITC), "
          f"{n_trees} tree rings, {n_canopy} canopy rings "
          f"({n_rle} RLE anns decoded) -> test_gt.json")


if __name__ == "__main__":
    main()
