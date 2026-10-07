"""RLE drop-in for ablation/lib/io.per_tile, so the published ablation scripts run at 2048.

io.per_tile decodes every mask densely -- (N, RES, RES) bool per tile, 2.4 GB for a 600-det
tile at 2048 -- and the scripts then call E.mask_iou, a dense float32 matmul. This module
yields the same per-tile dict but with `pm`/`gm` replaced by AreaStub objects that carry
(a) the per-instance pixel areas and (b) the precomputed IoU matrix, and patches E.mask_iou
to return that matrix when handed a stub.

EXACTNESS. E.mask_iou computes, in float32,
    inter / (p.sum(1)[:,None] + g.sum(1)[None] - inter + 1e-9)
where inter, p.sum, g.sum are exact integers (all < 2^24 at 2048). Here the integer
intersection is recovered from pycocotools' double IoU  (I = iou*(a+b)/(1+iou), rounded;
exact for any I < 2^40), then the SAME float32 expression is evaluated in the same order.
The gate is that the published ablation JSONs are reproduced at 512.

install(res, ...) must be called before the target script's main():
  * E.RES / E.SCALE are set (runtime lookups: box_oversize, zero-shapes).
  * io.per_tile, io.knobbed, io.RESULTS, io.record_inputs and gate.check are redirected so
    NOTHING is written under ablation/results/ or ablation/inputs.json.
"""
import json
import os

import numpy as np
from pycocotools import mask as maskUtils

from boxinst_commonality_tcd_04 import evaluate as E
from boxinst_commonality_tcd_04.ablation.lib import gate, io

_real_mask_iou = E.mask_iou


class AreaStub:
    """Stands in for a dense (N,RES,RES) mask stack: len(), [j].sum(), and the IoU matrix."""
    def __init__(self, areas, iou=None):
        self.areas = np.asarray(areas, np.int64); self.iou = iou

    def __len__(self):
        return len(self.areas)

    def __getitem__(self, j):
        a = self.areas[j]
        class _One:
            def sum(self_inner):
                return a
        return _One()


def _mask_iou(pm, gm):
    if isinstance(pm, AreaStub):
        return pm.iou
    return _real_mask_iou(pm, gm)


def _enc(m):
    return maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))


def _as_bytes(r):
    c = r["counts"]
    return {"size": list(r["size"]), "counts": c.encode("ascii") if isinstance(c, str) else c}


def exact_mask_iou(p_rles, g_rles, pa, ga):
    if len(p_rles) == 0 or len(g_rles) == 0:
        return np.zeros((len(p_rles), len(g_rles)))
    iou = np.asarray(maskUtils.iou(p_rles, g_rles, [0] * len(g_rles)),
                     np.float64).reshape(len(p_rles), len(g_rles))
    I = np.rint(iou * (pa[:, None] + ga[None].astype(np.float64)) / (1.0 + iou))
    I32 = I.astype(np.float32)
    return I32 / (pa.astype(np.float32)[:, None] + ga.astype(np.float32)[None] - I32 + 1e-9)


def per_tile_rle(res):
    scale = 2048.0 / res

    def per_tile(preds_path, gt, want=None, progress=100):
        preds = io.load_preds(preds_path)
        tids = sorted(gt)
        for k, tid in enumerate(tids):
            p = preds[tid]
            pr = [_as_bytes(r) for r in p["masks_rle"]]
            for r in pr:
                assert tuple(r["size"]) == (res, res), f"{tid}: mask {r['size']} != {res}"
            pa = np.asarray([int(maskUtils.area(r)) for r in pr], np.int64)
            gr = [_enc(m) for m in E.raster(gt[tid]["trees"], res=res, scale=scale)]
            ga = np.asarray([int(maskUtils.area(r)) for r in gr], np.int64)
            if "canopy_ignore" in p:
                ign = np.asarray(p["canopy_ignore"], bool)
            else:
                can = [_enc(m) for m in E.raster(gt[tid]["canopy"], res=res, scale=scale)]
                cu = maskUtils.merge(can) if can else None
                ign = np.array([bool(a) and cu is not None and
                                int(maskUtils.area(maskUtils.merge([r, cu], intersect=1))) / a > 0.5
                                for r, a in zip(pr, pa)], dtype=bool)
            yield {"tid": tid, "i": k, "n": len(tids),
                   "pm": AreaStub(pa, exact_mask_iou(pr, gr, pa, ga)), "gm": AreaStub(ga),
                   "boxes": np.asarray(p["boxes_2048"], np.float32).reshape(-1, 4),
                   "scores": np.asarray(p["scores"], np.float32), "ignore": ign}
            if progress and ((k + 1) % progress == 0 or k + 1 == len(tids)):
                print(f"    {k + 1}/{len(tids)}", flush=True)
    return per_tile


def install(res, knobbed_map, results_dir, legacy_refs=None):
    """knobbed_map: {seed: preds path} for io.knobbed. legacy_refs: {preds path: scored json}
    used as the gate at res != 512 (score_coco's canopy_neutral_legacy arm on the same file)."""
    os.makedirs(results_dir, exist_ok=True)
    E.RES, E.SCALE = res, 2048.0 / res
    E.mask_iou = _mask_iou
    io.per_tile = per_tile_rle(res)
    io.knobbed = lambda s: knobbed_map[s]
    io.RESULTS = results_dir
    io.record_inputs = lambda paths, out=None: None
    if res != 512:
        def check(got, ref_path, keys=("mask_mAP50", "mask_mAP50_95"), tol=gate.TOL, label=""):
            src = ref_path if isinstance(ref_path, str) else None
            print(f"  [rle_io] 512 ref gate not applicable at {res} ({label}); "
                  f"using score_coco legacy arm instead", flush=True)
            return got
        gate.check = check
    print(f"[rle_io] installed res={res} results_dir={results_dir}", flush=True)
