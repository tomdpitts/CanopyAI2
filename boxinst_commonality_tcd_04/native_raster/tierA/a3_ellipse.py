"""Tier A3 (PLAN.md §13, approved 2026-10-14): ellipse-in-box lower bound. For each box the mask is the
axis-aligned ellipse inscribed in it, rasterised at 2048 (pixel centre (x+.5, y+.5) inside
((x-cx)/a)^2 + ((y-cy)/b)^2 <= 1), ranked by the detector score. Two box sources:
  lace_s{0,1,2}  LACE detector boxes + s (guided_ablate_inputs/detscore_s*.json)
  selvabox       seam-fixed SelvaBox boxes + detector score
Writes tierA/preds_a3_ellipse_<src>_r2048.json (refuses to overwrite).

  PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.tierA.a3_ellipse
"""
import json, os
import numpy as np
from pycocotools import mask as mu

HERE = os.path.dirname(os.path.abspath(__file__))
NR = os.path.dirname(HERE)


def ellipse_rle(b):
    x0, y0, x1, y1 = [float(v) for v in b]
    cx, cy, a, bb = (x0 + x1) / 2, (y0 + y1) / 2, max((x1 - x0) / 2, 1e-6), max((y1 - y0) / 2, 1e-6)
    X0, X1 = max(0, int(np.floor(x0))), min(2048, int(np.ceil(x1)))
    Y0, Y1 = max(0, int(np.floor(y0))), min(2048, int(np.ceil(y1)))
    m = np.zeros((2048, 2048), np.uint8)
    if X1 > X0 and Y1 > Y0:
        xs = np.arange(X0, X1) + 0.5; ys = np.arange(Y0, Y1) + 0.5
        m[Y0:Y1, X0:X1] = (((xs[None, :] - cx) / a) ** 2 + ((ys[:, None] - cy) / bb) ** 2 <= 1).astype(np.uint8)
    e = mu.encode(np.asfortranarray(m))
    return {"size": [2048, 2048], "counts": e["counts"].decode("ascii")}


def build(src, boxes_scores):
    fp = os.path.join(HERE, f"preds_a3_ellipse_{src}_r2048.json"); assert not os.path.exists(fp), fp
    preds = {t: {"boxes_2048": r["boxes_2048"], "scores": r["scores"],
                 "masks_rle": [ellipse_rle(b) for b in r["boxes_2048"]]} for t, r in boxes_scores.items()}
    json.dump({"meta": {"model": f"ellipse-in-box on {src} boxes, ranked by detector score", "mask_res": 2048},
               "preds": preds}, open(fp, "w"))
    print("->", fp, sum(len(v["scores"]) for v in preds.values()), flush=True)


def main():
    for s in (0, 1, 2):
        build(f"lace_s{s}", json.load(open(os.path.join(NR, "guided_ablate_inputs", f"detscore_s{s}.json"))))
    sel = json.load(open(os.path.join(NR, "seamfix", "preds_selvabox_seamfix_on_r2048.json")))["preds"]
    build("selvabox", {t: {"boxes_2048": r["boxes_2048"], "scores": r["scores"]} for t, r in sel.items()})


if __name__ == "__main__":
    main()
