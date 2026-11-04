"""Contact sheet (2026-11-04, Tom): the 30 test pairs where SAM 3 most outperforms LACE in mask IoU to GT, same D1 box.
Pairing as mask_diag.py (det -> GT tree with the highest box IoU >= 0.5). Overlays: GT white, LACE red, SAM 3 cyan, D1 box yellow.
-> results/fig_sam_beats_lace_top30_test.jpg (+ .json with the 30 rows)
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.fig_sam_vs_lace [--reverse]  (LACE beats SAM -> results/fig_lace_beats_sam_top30_test.jpg)
"""
import argparse, json, os
import numpy as np

from boxinst_commonality_tcd_04.dino_det import rerank as R
from boxinst_commonality_tcd_04.dino_det.mask_diag import SAMD

TIF = os.path.join(R.REPO, "data", "tcd", "test", "{}.tif")


def _tile(a):
    from pycocotools import mask as M
    t, L, S, boxes, gts = a
    if not gts or not boxes:
        return []
    b = np.asarray(boxes, float).reshape(-1, 4); gb = np.asarray([M.toBbox(g) for g in gts], float); gb[:, 2:] += gb[:, :2]
    bi = R._box_iou(b, gb); j = bi.argmax(1); keep = np.flatnonzero(bi[np.arange(len(b)), j] >= 0.5)
    if not len(keep):
        return []
    il = np.asarray(M.iou([L[i] for i in keep], gts, [0] * len(gts))).reshape(len(keep), len(gts))
    is_ = np.asarray(M.iou([S[i] for i in keep], gts, [0] * len(gts))).reshape(len(keep), len(gts))
    return [(t, int(i), int(j[i]), float(il[k, j[i]]), float(is_[k, j[i]]), float(M.area(gts[j[i]]))) for k, i in enumerate(keep)]


def outline(m):
    from scipy import ndimage as ndi
    return m & ~ndi.binary_erosion(m)


def main():
    import multiprocessing as mp
    from PIL import Image, ImageDraw
    from pycocotools import mask as M
    from scipy import ndimage as ndi
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    a = argparse.ArgumentParser(); a.add_argument("--n", type=int, default=30); a.add_argument("--reverse", action="store_true", help="LACE beats SAM"); a = a.parse_args()
    G = json.load(open(R.TEST_GT)); St = json.load(open(os.path.join(R.OUT, f"preds_{R.TAG}_test_sprime_r2048.json")))["preds"]; tids = sorted(St)
    gd, t2i = SL.build_gt_crowd(G, 2048); by = {i: [] for i in t2i.values()}
    for an in gd["annotations"]:
        if not an["iscrowd"]:
            by[an["image_id"]].append(an["segmentation"])
    Sm = {t: json.load(open(os.path.join(SAMD, t + ".json"))) for t in tids}
    jobs = [(t, [SL._as_bytes(r) for r in St[t]["masks_rle"]], [SL._as_bytes(r) for r in Sm[t]["masks_rle"]], St[t]["boxes_2048"], by[t2i[t]]) for t in tids]
    with mp.get_context("spawn").Pool(14) as pool:
        rows = sum(pool.map(_tile, jobs, chunksize=4), [])
    sgn = -1 if a.reverse else 1; rows.sort(key=lambda r: -sgn * (r[4] - r[3])); top = rows[:a.n]
    print(f"[fig] {len(rows)} pairs; top {a.n} SAM-LACE IoU diff {top[0][4]-top[0][3]:.3f} .. {top[-1][4]-top[-1][3]:.3f}", flush=True)
    C, cell, lab = 6, 300, 34; sheet = Image.new("RGB", (C * cell, ((a.n + C - 1) // C) * (cell + lab)), (20, 20, 20)); d = ImageDraw.Draw(sheet)
    out = []
    for k, (t, i, j, il, is_, ga) in enumerate(top):
        img = np.asarray(Image.open(TIF.format(t)).convert("RGB")); g = M.decode(by[t2i[t]][j]).astype(bool)
        lm = M.decode(SL._as_bytes(St[t]["masks_rle"][i])).astype(bool); sm = M.decode(SL._as_bytes(Sm[t]["masks_rle"][i])).astype(bool)
        x0, y0, x1, y1 = St[t]["boxes_2048"][i]; U = g | lm | sm; ys, xs = np.nonzero(U)
        cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2; h = max(ys.max() - ys.min(), xs.max() - xs.min(), y1 - y0, x1 - x0) * 0.8 + 16
        Y0, Y1, X0, X1 = int(max(0, cy - h)), int(min(2048, cy + h)), int(max(0, cx - h)), int(min(2048, cx + h))
        crop = img[Y0:Y1, X0:X1].copy(); sc = cell / max(crop.shape[:2])
        big = lambda m: np.asarray(Image.fromarray(m[Y0:Y1, X0:X1].astype(np.uint8) * 255).resize((int(crop.shape[1] * sc), int(crop.shape[0] * sc)), Image.NEAREST)) > 0
        rgb = np.asarray(Image.fromarray(crop).resize((int(crop.shape[1] * sc), int(crop.shape[0] * sc)), Image.BILINEAR)).copy()
        for m, col in ((g, (255, 255, 255)), (lm, (255, 40, 40)), (sm, (0, 230, 255))):
            o = ndi.binary_dilation(outline(big(m)), iterations=1); rgb[o] = col
        tile = Image.fromarray(rgb); td = ImageDraw.Draw(tile)
        td.rectangle([(x0 - X0) * sc, (y0 - Y0) * sc, (x1 - X0) * sc, (y1 - Y0) * sc], outline=(255, 220, 0), width=1)
        r_, c_ = divmod(k, C); sheet.paste(tile, (c_ * cell, r_ * (cell + lab)))
        dg = np.sqrt(ga * 4 / np.pi)
        d.text((c_ * cell + 4, r_ * (cell + lab) + cell + 2), f"#{k+1} {t[-8:]} d{dg:.0f}px", fill=(220, 220, 220))
        d.text((c_ * cell + 4, r_ * (cell + lab) + cell + 16), f"LACE {il:.2f}  SAM {is_:.2f}", fill=(220, 220, 220))
        out.append({"rank": k + 1, "tile": t, "det": i, "gt": j, "iou_lace": round(il, 3), "iou_sam": round(is_, 3), "gt_diam_px": round(dg, 1)})
    fp = os.path.join(R.OUT, f"fig_{'lace_beats_sam' if a.reverse else 'sam_beats_lace'}_top{a.n}_test.jpg"); sheet.save(fp, quality=92)
    json.dump(out, open(fp.replace(".jpg", ".json"), "w"), indent=1); print(f"[fig] -> {fp}", flush=True)


if __name__ == "__main__":
    main()
