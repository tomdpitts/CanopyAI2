"""Dependency-light copy of the frozen scorer's `canopy_neutral_crowd` arm, for use INSIDE the
training containers (torch-1.x / detectron2 images that cannot import `evaluate.py`).

Used for two things only, both on VAL:
  1. checkpoint selection during training (val box AP50; B1/B2 on the 896 val 1024 subtiles
     since 2026-10-29, via window_dets), and
  2. scoring the stitched val predictions for the record (was: the val choice of geometry,
     dropped 2026-10-29).
Every TEST number, and the LACE stop-gate, comes from the frozen `score_coco.py` itself
(`tierb_modal.py::score`), never from this file.

It is the frozen code path, copied rather than re-derived: `raster` is `evaluate.raster`
verbatim (PIL polygon fill at res, scale 2048/res); GT and detections are built as in
`score_coco.build_gt` / `build_dt`; COCOeval runs with the same iouThrs, recThrs, maxDets and
areaRng and reads AP from eval['precision'] the same way. `parity()` gates it against stored
frozen-scorer outputs (`python -m ...tierb.score_lite --parity`), and must print PASS before
any Modal run.
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_THRS = np.linspace(0.0, 1.00, 101)
MAX_DETS = 600
SCORE_FLOOR = 0.05
CAT = [{"id": 1, "name": "tree"}]


def raster(polys, res=2048, scale=1.0):
    """`evaluate.raster`, verbatim."""
    out = []
    for poly in polys:
        if not poly or len(poly) < 6:
            out.append(np.zeros((res, res), bool)); continue
        img = Image.new("L", (res, res), 0)
        pts = (np.asarray(poly, np.float32).reshape(-1, 2) / scale)
        ImageDraw.Draw(img).polygon([tuple(v) for v in pts], fill=1)
        out.append(np.asarray(img, bool))
    return out


def _as_bytes(rle):
    c = rle["counts"]
    return {"size": list(rle["size"]), "counts": c.encode("ascii") if isinstance(c, str) else c}


def _polys_to_rle(polys, res):
    if not polys:
        return []
    return [maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
            for m in raster(polys, res=res, scale=2048.0 / res)]


def build_gt_crowd(gt, res=2048):
    """`score_coco.build_gt(gt, res, with_canopy=True)` -> (coco_dict, tid2img)."""
    images, anns = [], []
    tids = sorted(gt)
    tid2img = {t: i + 1 for i, t in enumerate(tids)}
    aid = 1
    for t in tids:
        images.append({"id": tid2img[t], "file_name": t, "width": res, "height": res})
        for r in _polys_to_rle(gt[t]["trees"], res):
            anns.append({"id": aid, "image_id": tid2img[t], "category_id": 1,
                         "segmentation": r, "iscrowd": 0, "area": float(maskUtils.area(r)),
                         "bbox": [float(v) for v in maskUtils.toBbox(r)]})
            aid += 1
        for r in _polys_to_rle(gt[t]["canopy"], res):
            anns.append({"id": aid, "image_id": tid2img[t], "category_id": 1,
                         "segmentation": r, "iscrowd": 1, "area": float(maskUtils.area(r)),
                         "bbox": [float(v) for v in maskUtils.toBbox(r)]})
            aid += 1
    return {"images": images, "annotations": anns, "categories": CAT}, tid2img


def gt_to_json(gt_dict):
    """RLE bytes -> str so a built GT can be cached on the volume (rasterising 108 val tiles
    at 2048 takes minutes; every eval reuses one build)."""
    out = dict(gt_dict)
    out["annotations"] = [dict(a, segmentation={"size": a["segmentation"]["size"],
                                                "counts": a["segmentation"]["counts"].decode()
                                                if isinstance(a["segmentation"]["counts"], bytes)
                                                else a["segmentation"]["counts"]})
                          for a in gt_dict["annotations"]]
    return out


def gt_from_json(d):
    d = dict(d)
    d["annotations"] = [dict(a, segmentation=_as_bytes(a["segmentation"]))
                        for a in d["annotations"]]
    return d


def build_dt(preds, tid2img, res=2048, score_floor=SCORE_FLOOR, with_masks=True):
    """`score_coco.build_dt` (same order, same floor). with_masks=False builds bbox-only
    detections for the fast box-AP selection signal (identical bbox AP: bbox eval never reads
    the segmentation, and areaRng is 'all')."""
    dets = []
    scale = 2048.0 / res
    for t, rec in preds.items():
        if t not in tid2img:
            continue
        sc = np.asarray(rec["scores"], np.float32)
        bx = np.asarray(rec["boxes_2048"], np.float32).reshape(-1, 4) / scale
        keep = np.flatnonzero(sc >= score_floor)
        rles = None
        if with_masks:
            rles = [_as_bytes(rec["masks_rle"][i]) for i in keep]
            for r in rles:
                if tuple(r["size"]) != (res, res):
                    raise ValueError(f"{t}: mask {r['size']} != res {res}")
        for j, i in enumerate(keep):
            x0, y0, x1, y1 = bx[i]
            d = {"image_id": tid2img[t], "category_id": 1, "score": float(sc[i]),
                 "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]}
            if with_masks:
                d["segmentation"] = rles[j]
                d["area"] = float(maskUtils.area(rles[j]))
            dets.append(d)
    return dets


def window_dets(image_id, boxes_xyxy, scores, score_floor=SCORE_FLOOR):
    """One 1024 window's raw boxes (window coords, xyxy) -> bbox-only COCO dets for the B1/B2
    selection signal on the 896 val subtiles (tierb_data/val_sub_gt_1024.json; amendment
    2026-10-29). Same floor and xywh conversion as build_dt."""
    return [{"image_id": image_id, "category_id": 1, "score": float(s),
             "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]}
            for (x0, y0, x1, y1), s in zip(boxes_xyxy, scores) if s >= score_floor]


def cocoeval(gt_dict, dets, iou_type, max_dets=MAX_DETS):
    """`score_coco._cocoeval`, verbatim."""
    cg = COCO(); cg.dataset = gt_dict; cg.createIndex()
    if not dets:
        return {"AP50": 0.0, "AP50_95": 0.0, "AP75": 0.0, "n_det": 0}
    cd = cg.loadRes([dict(d) for d in dets])
    ev = COCOeval(cg, cd, iouType=iou_type)
    ev.params.iouThrs = IOU_THRS
    ev.params.recThrs = REC_THRS
    ev.params.maxDets = [max_dets]
    ev.params.areaRng = [[0.0, 1e10]]
    ev.params.areaRngLbl = ["all"]
    ev.evaluate(); ev.accumulate()
    p = ev.eval["precision"]

    def _ap(ti):
        s = p[ti, :, :, 0, 0] if ti is not None else p[:, :, :, 0, 0]
        s = s[s > -1]
        return float(s.mean()) if s.size else float("nan")
    i50 = int(np.argmin(np.abs(IOU_THRS - 0.50)))
    i75 = int(np.argmin(np.abs(IOU_THRS - 0.75)))
    return {"AP50": round(_ap(i50), 4), "AP75": round(_ap(i75), 4),
            "AP50_95": round(_ap(None), 4), "n_det": len(dets)}


def score(preds, gt_dict, tid2img, masks=True, max_dets=MAX_DETS, floor=SCORE_FLOOR):
    """-> {"box": {...}, "mask": {...} or None}; the frozen crowd arm's numbers."""
    out = {"box": cocoeval(gt_dict, build_dt(preds, tid2img, 2048, floor, with_masks=False),
                           "bbox", max_dets), "mask": None}
    if masks:
        out["mask"] = cocoeval(gt_dict, build_dt(preds, tid2img, 2048, floor), "segm", max_dets)
    return out


# ---------------------------------------------------------------- parity gate
PARITY = [  # (preds, frozen scored file, floor) -- all at res 2048, maxDets 600
    ("native_raster/preds_lace_s0_r2048_cgf_r12_tau035_guidedrun.json",
     "native_raster/scored_lace_s0_r2048_cgf_r12_tau035_guidedrun.json", 0.0),
    ("native_raster/seamfix/preds_box2mask_swinl_s0_r2048_seamfix.json",
     "native_raster/seamfix/scored_box2mask_swinl_s0_r2048_seamfix.json", 0.05),
]


def parity(pkg, out_path):
    """Score the PARITY rows with this file and compare to the frozen scorer's stored crowd-arm
    numbers. Exact equality required (both round to 4 dp)."""
    import os
    import time
    gt = json.load(open(os.path.join(pkg, "test_gt.json")))
    t0 = time.time()
    gd, t2i = build_gt_crowd(gt, 2048)
    print(f"[parity] GT built {time.time()-t0:.0f}s", flush=True)
    rows, ok = [], True
    for p, s, floor in PARITY:
        pp, sp = os.path.join(pkg, p), os.path.join(pkg, s)
        ref = json.load(open(sp))
        assert ref["protocol"]["res"] == 2048 and ref["protocol"]["max_dets"] == 600
        assert abs(ref["protocol"]["score_floor"] - floor) < 1e-9, (s, ref["protocol"])
        P = json.load(open(pp))
        r = score(P.get("preds", P), gd, t2i, masks=True, floor=floor)
        cr = ref["canopy_neutral_crowd"]
        row = {"preds": p, "ref": s}
        for arm in ("box", "mask"):
            for k in ("AP50", "AP75", "AP50_95", "n_det"):
                a, b = r[arm][k], cr[arm][k]
                row[f"{arm}_{k}"] = [a, b]
                ok &= (a == b)
        rows.append(row)
        print(f"[parity] {os.path.basename(p)}: "
              + " ".join(f"{k}={v[0]}/{v[1]}" for k, v in row.items()
                         if isinstance(v, list)), flush=True)
    rec = {"gate": "score_lite == score_coco (crowd arm, res 2048, maxDets 600)",
           "pass": bool(ok), "rows": rows, "date": time.strftime("%Y-%m-%d %H:%M")}
    assert not os.path.exists(out_path), f"REFUSING to overwrite {out_path}"
    json.dump(rec, open(out_path, "w"), indent=1)
    print(f"PARITY {'PASS' if ok else 'FAIL'} -> {out_path}", flush=True)
    return ok


if __name__ == "__main__":
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", action="store_true")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    if a.parity:
        parity(os.path.dirname(os.path.dirname(here)),
               os.path.join(here, "gates", "score_lite_parity.json"))
