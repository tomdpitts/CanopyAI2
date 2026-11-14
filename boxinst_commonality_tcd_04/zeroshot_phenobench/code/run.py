"""PhenoBench zero-shot run, CPU arms on identical GT boxes (all plant instances incl. partial; scoring subsets later).

Arms (results/<arm>/<stem>.json: masks_rle native, iou unrounded, valid, double):
  lace_tt     per-image test-time fit (lace_core.fit_image, frozen settings) + deployed lace4px_simple render. MAIN.
  lace_tt_a0  SAME fitted model, render with alpha = 0 (spatial-prior logit off; rho_hat still selects prototypes).
              Sensitivity arm for the pi/4 mass pin (PhenoBench GT fill 0.40 vs the pinned 0.785).
  lace_tree   negative control: the deployed tree-fitted masker as-is.
  box, ellipse geometric baselines (lace_core.box_clip / ellipse).
  exg         colour baseline: Excess Green (Woebbecke 1995) on chromatic coords, ExG = 2g - r - b with r = R/(R+G+B)
              etc.; one Otsu threshold per IMAGE (256-bin histogram over the image's ExG range); mask = ExG > t, clipped
              to the box. No tuning.
Diagnostics in results/lace_tt/fit/<stem>.json: fit report + rho foreground-mass maps (sum_k pi, per size band, 8x8)
and per-box fill for lace_tt / lace_tt_a0.

  cd zeroshot_phenobench/code && PYTHONPATH=/Users/tompitts/dphil/CanopyAI2 ../../../.venv/bin/python -u run.py
"""
import json, os, time

import numpy as np
from pycocotools import mask as maskUtils

from common import RES, ROOT, RUN, stems, load_rgbf, gt_image, enc, rle_json
import _verbatim as V
import lace_core as LC

ARMS = ("lace_tt", "lace_tt_a0", "lace_tree", "box", "ellipse", "exg")


def otsu(x, bins=256):
    h, e = np.histogram(x, bins=bins); c = (e[:-1] + e[1:]) / 2; p = h / h.sum()
    w0 = np.cumsum(p); w1 = 1 - w0; m0 = np.cumsum(p * c); mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mt * w0 - m0) ** 2 / (w0 * w1)
    return float(c[np.nanargmax(sb)])


def exg_map(rgbf):
    s = rgbf.sum(2) + 1e-6; r, g, b = (rgbf[..., i] / s for i in range(3))
    return 2 * g - r - b


def render_alpha(m, zn, gh, gw, boxes, rgbf, H, W, alpha):
    """lace_core.render_boxes with the apply-time prior weight overridden (restored afterwards)."""
    old = LC.APPLY["prior_weight"]; LC.APPLY["prior_weight"] = alpha
    try:
        return LC.render_boxes(m, zn, gh, gw, boxes, rgbf, H, W)
    finally:
        LC.APPLY["prior_weight"] = old


def setup():
    assert V.verify() and LC.check_frozen()
    for a in ARMS:
        os.makedirs(os.path.join(RES, a), exist_ok=True)
    for d in ("gt", "lace_tt/fit", "lace_tt/models"):
        os.makedirs(os.path.join(RES, d), exist_ok=True)
    return LC.Masker4(np.load(LC.DEPLOYED_NPZ))


def todo_arms(t):
    return [a for a in ARMS if not os.path.exists(os.path.join(RES, a, t + ".json"))]


def process(t, F4, tree, label=""):
    """All CPU arms for one image. F4: (256, 256, 1024) float32 raw cells. Per-arm atomic checkpoint."""
    todo = todo_arms(t)
    if not todo:
        return None
    boxes, gt_rles, valid, cls, ids = gt_image(t); H, W = 1024, 1024
    gfp = os.path.join(RES, "gt", t + ".json")
    if not os.path.exists(gfp):
        json.dump({"ids": ids, "cls": cls, "boxes": boxes.tolist(), "valid": valid.tolist(),
                   "gt_area": [int(maskUtils.area(r)) for r in gt_rles]}, open(gfp, "w"))
    rgbf = load_rgbf(t); gh, gw = F4.shape[:2]
    line = [f"{label} {t} n={len(boxes)}"]; fitted = None
    if len(boxes) == 0:                                   # no plants: nothing to score; record so it is not retried
        for a in todo:
            json.dump({"arm": a, "image": t, "masks_rle": [], "iou": [], "valid": [], "double": 0, "empty": True}, open(os.path.join(RES, a, t + ".json"), "w"))
        return "  " + line[0] + " (no instances)"
    for a in todo:
        t0 = time.time(); extra = {}
        if a in ("lace_tt", "lace_tt_a0"):
            if isinstance(fitted, str):                       # fit already failed for this image: recorded below
                continue
            if fitted is None:
                mp = os.path.join(RES, "lace_tt", "models", t + ".npz")
                if os.path.exists(mp):
                    model = dict(np.load(mp)); rep = json.load(open(os.path.join(RES, "lace_tt", "fit", t + ".json")))
                else:
                    try:
                        model, rep = LC.fit_image(F4[0::2, 0::2], boxes, H, W)
                    except RuntimeError as err:                   # too few cells to fit (e.g. only tiny plants)
                        # Not silently dropped: both test-time arms score EMPTY masks (IoU 0), flagged fit_failed.
                        for b in ("lace_tt", "lace_tt_a0"):
                            pr = [enc(np.zeros((H, W), bool)) for _ in boxes]; iou = V.per_crown_iou_rle(pr, gt_rles)
                            json.dump({"arm": b, "image": t, "masks_rle": [rle_json(r) for r in pr], "iou": iou.tolist(),
                                       "valid": valid.tolist(), "double": 0, "fit_failed": str(err)}, open(os.path.join(RES, b, t + ".json"), "w"))
                        line.append(f"lace_tt FIT FAILED ({err}) -> empty masks"); fitted = "failed"
                        continue
                    np.savez(mp, **model)
                    pm = model["pi"].sum(1)                                   # (3 size bands, 8, 8) fg mass
                    rep["rho_fg_mass"] = {"per_band_mean": [round(float(x), 4) for x in pm.mean((1, 2))],
                                          "maps": np.round(pm, 3).tolist()}
                    json.dump(rep, open(os.path.join(RES, "lace_tt", "fit", t + ".json"), "w"), indent=1)
                m = LC.Masker4(model); fitted = (m, m.project(F4), rep)
            m, zn, rep = fitted
            masks, pbar = render_alpha(m, zn, gh, gw, boxes, rgbf, H, W, 0.3 if a == "lace_tt" else 0.0)
            extra = {"pbar": pbar, "k_effective": rep["k_effective"], "fit_s": rep["fit_s"], "alpha": 0.3 if a == "lace_tt" else 0.0}
        elif a == "lace_tree":
            masks, pbar = LC.render_boxes(tree, tree.project(F4), gh, gw, boxes, rgbf, H, W); extra = {"pbar": pbar}
        elif a == "box":
            masks = [LC.box_clip(b, H, W) for b in boxes]
        elif a == "ellipse":
            masks = [LC.ellipse(b, H, W) for b in boxes]
        else:
            e = exg_map(rgbf); th = otsu(e); g = e > th
            masks = [g & LC.box_clip(b, H, W) for b in boxes]; extra = {"otsu_t": th, "green_frac": round(float(g.mean()), 4)}
        pr = [enc(x) for x in masks]; iou = V.per_crown_iou_rle(pr, gt_rles)
        rec = {"arm": a, "image": t, "masks_rle": [rle_json(r) for r in pr], "iou": iou.tolist(), "valid": valid.tolist(),
               "double": V.double_assignment_rle(pr), "sec": round(time.time() - t0, 3), **extra}
        fp = os.path.join(RES, a, t + ".json"); json.dump(rec, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
        line.append(f"{a} {iou.mean():.3f}")
    return "  " + " | ".join(line)


def main():
    """Features read from features/<RUN>/<stem>.npy (sample60 run)."""
    tree = setup(); S = stems(); T0 = time.time()
    print(f"[pb run {RUN}] {len(S)} images, arms {ARMS}", flush=True)
    for k, t in enumerate(S):
        if not todo_arms(t):
            continue
        F4 = np.load(os.path.join(ROOT, "features", RUN, t + ".npy")).astype(np.float32)
        print(process(t, F4, tree, f"{k+1}/{len(S)}") + f"  [{time.time()-T0:.0f}s]", flush=True)
    print(f"[pb run {RUN}] done", flush=True)


if __name__ == "__main__":
    main()
