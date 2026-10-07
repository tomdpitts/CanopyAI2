"""Qualitative mask figures: GT vs model predictions, native 2048 raster.

Two sets, selected with --set:
  tcd439  OAM-TCD 439-tile holdout: GT | LACE | Restor | Detectree2 | Box2Mask Swin-L (1 tile per row)
  sparse  sparse-canopy 236-tile holdout: GT | LACE | Detectree2 (2 tiles per row)

Every panel is a CROP_PX window of the 2048 tile (auto-chosen: most GT centroids). Fills are
per-instance colours shared between a prediction and the GT crown it matches (mask IoU >= 0.5,
COCO-style greedy score-ordered matching); white outline = match, red = false positive, dashed
yellow = GT crown the model missed. Predictions sitting >50 % in unlabelled canopy are ignored by
the scorer and are NOT drawn; tiles were chosen with <0.5 % canopy so this is immaterial. Each
model is shown above its own best-F1 score threshold over the whole set (display only; the AP
tables use the full ranked list). Thresholds are cached in qualitative_thresholds_<set>.json.

Usage: .venv/bin/python boxinst_commonality_tcd_04/ablation/results/figs/make_qualitative.py --set tcd439
env overrides: TILES=180,122,...  CROP_PX=768  SUFFIX=_x
"""
import argparse, json, os
import numpy as np
from PIL import Image
from pycocotools import mask as mu
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPO = os.path.abspath(os.path.join(ROOT, ".."))
NR = os.path.join(ROOT, "native_raster")
OUT = os.path.dirname(os.path.abspath(__file__))
RES = 2048
IOU_T = 0.5
OUT_PX = 900

SETS = {
    "tcd439": dict(
        gt=os.path.join(ROOT, "test_gt.json"), imgdir=os.path.join(REPO, "data/tcd/test"), prefix="tcd_val_tile_",
        models=[("LACE (ours)", "preds_lace_s0_r2048_tau040.json"),
                ("Restor Mask R-CNN", "preds_restor_rpn1000_r2048_treeonly.json"),
                ("Detectree2", "preds_dt2_s0v2_r2048.json"),
                ("Box2Mask Swin-L", "preds_box2mask_swinl_s0_r2048.json")],
        tiles="24,146,431,258", per_row=1, crop=768),
    "sparse": dict(
        gt=os.path.join(REPO, "data/tcd_sparse/sparse_gt.json"), imgdir=os.path.join(REPO, "data/tcd_sparse/test"), prefix="tcd_tile_",
        models=[("LACE (ours)", "preds_lace_sparse_s0_r2048_tau040.json"),
                ("Detectree2", "preds_dt2_sparse_s0v2_r2048.json")],
        tiles="426,2009,3569,413,1762,3969", per_row=2, crop=1024),
}

PAL = np.array([
    [66, 133, 244], [219, 68, 55], [244, 180, 0], [15, 157, 88], [171, 71, 188],
    [0, 172, 193], [255, 112, 67], [156, 204, 101], [92, 107, 192], [236, 64, 122],
    [255, 202, 40], [38, 166, 154], [141, 110, 99], [124, 179, 66], [255, 143, 0],
], dtype=np.float32)
WHITE = np.array([255, 255, 255], np.float32)
RED = np.array([255, 40, 40], np.float32)
YEL = np.array([255, 235, 59], np.float32)


def polys_to_rles(polys):
    return [mu.merge(mu.frPyObjects([p], RES, RES)) for p in polys if len(p) >= 6]


def union_rle(rles):
    return mu.merge(rles, intersect=False) if rles else None


def match_tile(gt_rles, canopy, pred_rles, scores):
    """Greedy score-ordered matching. Returns (order, status per pred in that order, matched GT idx)."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    status, gidx, taken = [], [], set()
    iou = mu.iou([pred_rles[i] for i in order], gt_rles, [0] * len(gt_rles)) if (len(order) and gt_rles) \
        else np.zeros((len(order), len(gt_rles)))
    cfrac = mu.iou([pred_rles[i] for i in order], [canopy], [1])[:, 0] if (len(order) and canopy is not None) \
        else np.zeros(len(order))
    for r in range(len(order)):
        best, bj = IOU_T, -1
        for j in range(len(gt_rles)):
            if j not in taken and iou[r, j] >= best:
                best, bj = iou[r, j], j
        if bj >= 0:
            taken.add(bj); status.append("tp"); gidx.append(bj)
        elif cfrac[r] > 0.5:
            status.append("ign"); gidx.append(-1)
        else:
            status.append("fp"); gidx.append(-1)
    return order, status, gidx


def best_f1_threshold(gt, preds, grid=np.round(np.arange(0.05, 0.96, 0.01), 2)):
    tp = np.zeros(len(grid)); fp = np.zeros(len(grid)); n_gt = 0
    for t, g in gt.items():
        gt_rles = polys_to_rles(g["trees"]); n_gt += len(gt_rles)
        p = preds[t]
        if not p["masks_rle"]:
            continue
        canopy = union_rle(polys_to_rles(g["canopy"]))
        order, status, _ = match_tile(gt_rles, canopy, p["masks_rle"], p["scores"])
        s = np.asarray(p["scores"])[order]
        st = np.asarray(status)
        for k, th in enumerate(grid):
            keep = s >= th
            tp[k] += int((keep & (st == "tp")).sum()); fp[k] += int((keep & (st == "fp")).sum())
    fn = n_gt - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    k = int(np.argmax(f1))
    return float(grid[k]), float(f1[k]), dict(tp=int(tp[k]), fp=int(fp[k]), fn=int(fn[k]))


def outline(m, px=3):
    return m & ~ndi.binary_erosion(m, iterations=px, border_value=0)


def dashed(m, period=28, duty=18):
    yy, xx = np.nonzero(m)
    keep = ((xx + yy) % period) < duty
    o = np.zeros_like(m); o[yy[keep], xx[keep]] = True
    return o


def render(img, masks, fills, edges, dashes=(), alpha=0.42):
    out = img.copy()
    for m, c in zip(masks, fills):
        out[m] = out[m] * (1 - alpha) + c * alpha
    for m, c in edges:
        out[outline(m)] = c
    for m, c in dashes:
        out[dashed(outline(m, 5))] = c
    return out


def auto_crop(gt_masks, size, stride=64):
    if size >= RES or not gt_masks:
        return (0, 0)
    cy = np.array([ndi.center_of_mass(m) for m in gt_masks])
    best, bxy = -1, (0, 0)
    for y0 in range(0, RES - size + 1, stride):
        for x0 in range(0, RES - size + 1, stride):
            n = int(((cy[:, 0] >= y0) & (cy[:, 0] < y0 + size) & (cy[:, 1] >= x0) & (cy[:, 1] < x0 + size)).sum())
            if n > best:
                best, bxy = n, (x0, y0)
    return bxy


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="tcd439", choices=SETS); a = ap.parse_args()
    cfg = SETS[a.set]
    tiles = [cfg["prefix"] + x for x in os.environ.get("TILES", cfg["tiles"]).split(",")]
    crop_px = int(os.environ.get("CROP_PX", cfg["crop"]))
    suffix = os.environ.get("SUFFIX", "")
    per_row = cfg["per_row"]
    models = cfg["models"]

    gt = json.load(open(cfg["gt"]))
    preds = {lab: json.load(open(os.path.join(NR, f)))["preds"] for lab, f in models}

    thr_path = os.path.join(OUT, f"qualitative_thresholds_{a.set}.json")
    thr = json.load(open(thr_path)) if os.path.exists(thr_path) else {}
    for lab, _ in models:
        if lab not in thr:
            th, f1, c = best_f1_threshold(gt, preds[lab])
            thr[lab] = dict(threshold=th, f1=f1, **c)
            print(f"{lab:20s} best-F1 thr={th:.2f} F1={f1:.4f} {c}", flush=True)
            json.dump(thr, open(thr_path, "w"), indent=1)

    npan = 1 + len(models)
    ncol = npan * per_row
    nrow = (len(tiles) + per_row - 1) // per_row
    fig_w = 18.4 / 2.54
    pw = fig_w / ncol
    gap = 0.012
    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, pw * nrow + 0.35), squeeze=False,
                             gridspec_kw=dict(wspace=gap, hspace=gap * 1.2))
    crops, counts = {}, {}
    for ti, t in enumerate(tiles):
        r, c0 = divmod(ti, per_row); c0 *= npan
        img = np.asarray(Image.open(os.path.join(cfg["imgdir"], f"{t}.tif")).convert("RGB"), np.float32)
        g = gt[t]
        gt_rles = polys_to_rles(g["trees"])
        gt_masks = [mu.decode(x).astype(bool) for x in gt_rles]
        canopy = union_rle(polys_to_rles(g["canopy"]))
        x0, y0 = crops.setdefault(t, auto_crop(gt_masks, crop_px))
        inwin = lambda m: bool(m[y0:y0 + crop_px, x0:x0 + crop_px].any())

        def show(ax, arr, title, label):
            im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            if crop_px < RES:
                im = im.crop((x0, y0, x0 + crop_px, y0 + crop_px))
            im = im.resize((OUT_PX, OUT_PX), Image.LANCZOS)
            ax.imshow(im, interpolation="none"); ax.set_axis_off()
            if r == 0:
                ax.set_title(title, fontsize=7.5, pad=2.5)
            ax.text(0.015, 0.985, label, transform=ax.transAxes, fontsize=5.6, va="top", ha="left", color="white",
                    bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.65))

        gt_fill = [PAL[i % len(PAL)] for i in range(len(gt_masks))]
        gt_img = render(img, gt_masks, gt_fill, [(m, WHITE) for m in gt_masks])
        tname = t.replace(cfg["prefix"], "tile ")
        show(axes[r, c0], gt_img, "Ground truth", f"{tname}   {sum(inwin(m) for m in gt_masks)} crowns")

        for ci, (lab, _) in enumerate(models):
            p = preds[lab][t]
            th = thr[lab]["threshold"]
            order, status, gidx = match_tile(gt_rles, canopy, p["masks_rle"], p["scores"])
            s = np.asarray(p["scores"])[order] if len(order) else np.zeros(0)
            masks, fills, edges, matched = [], [], [], set()
            for i, st, gj, kp in zip(order, status, gidx, s >= th):
                if not kp or st == "ign":
                    continue
                m = mu.decode(p["masks_rle"][i]).astype(bool)
                if st == "tp":
                    masks.append(m); fills.append(PAL[gj % len(PAL)]); edges.append((m, WHITE)); matched.add(gj)
                else:
                    masks.append(m); fills.append(RED); edges.append((m, RED))
            fn_masks = [gt_masks[j] for j in range(len(gt_masks)) if j not in matched]
            arr = render(img, masks, fills, edges, dashes=[(m, YEL) for m in fn_masks])
            tp = sum(1 for m, f in zip(masks, fills) if inwin(m) and f is not RED)
            fp = sum(1 for m, f in zip(masks, fills) if inwin(m) and f is RED)
            fn = sum(1 for m in fn_masks if inwin(m))
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            counts[f"{t}|{lab}"] = dict(f1=round(f1, 4), tp=tp, fp=fp, fn=fn)
            show(axes[r, c0 + 1 + ci], arr, lab, f"F1 {f1:.2f}  TP {tp}  FP {fp}  FN {fn}")
        print("rendered", t, flush=True)

    handles = [Line2D([0], [0], color="white", lw=1.4, label="matched (IoU $\\geq$ 0.5)"),
               Line2D([0], [0], color="#ff2828", lw=1.4, label="false positive"),
               Line2D([0], [0], color="#ffeb3b", lw=1.4, ls=(0, (3, 2)), label="missed crown")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.5, frameon=True, facecolor="#3a3a3a",
               edgecolor="none", labelcolor="white", fancybox=True, bbox_to_anchor=(0.5, -0.003),
               handlelength=2.2, columnspacing=1.6)
    plt.subplots_adjust(left=0.003, right=0.997, top=0.955, bottom=0.05)
    stem = os.path.join(OUT, f"qualitative_{a.set}{suffix}")
    fig.savefig(stem + ".png", dpi=400)
    im = Image.open(stem + ".png").convert("RGB")
    im.save(stem + ".jpg", quality=92, subsampling=0, dpi=(400, 400))
    im.save(stem + ".pdf", "PDF", resolution=400.0, quality=92)
    os.remove(stem + ".png")
    json.dump({"set": a.set, "tiles": tiles, "thresholds": thr, "crop_px": crop_px, "crop_xy": crops,
               "counts": counts}, open(stem + "_meta.json", "w"), indent=1)
    print("wrote", stem + ".pdf")


if __name__ == "__main__":
    main()
