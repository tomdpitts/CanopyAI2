"""Qualitative box figure on NeonTreeEvaluation (194 image-annotated 400x400 tiles, 0.1 m/px).

Panels per tile: GT boxes | LACE 4-phase (seed 0) | DeepForest 2.1.0. Matching is greedy in score
order at box IoU >= 0.4 (the benchmark's threshold); white box = match, red = false positive,
dashed yellow = missed GT box. Each model is drawn above its best-F1 score threshold, where F1 is
of precision and recall macro-averaged over the 194 tiles (the benchmark convention and the
operating point used in the paper's NEON table). Thresholds cached in qualitative_thresholds_neon.json.

Usage: .venv/bin/python boxinst_commonality_tcd_04/ablation/results/figs/make_qualitative_neon.py
env: TILES=comma-separated tile keys, SUFFIX=
"""
import json, os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NEON = os.path.join(ROOT, "modal_neon_multiseed")
OUT = os.path.dirname(os.path.abspath(__file__))
IOU_T = 0.4
UP = 3  # draw at 3x (1200 px) for crisp lines

MODELS = [("LACE (ours)", os.path.join(NEON, "phase4/dl/preds_phase4_s0.json")),
          ("DeepForest 2.1.0", os.path.join(NEON, "preds_deepforest.json"))]
DEFAULT_TILES = "SJER_057_2018,TEAK_057_2018,NIWO_012_2018,OSBS_003_2019,MLBS_069_2018,DSNY_025_2018"
PER_ROW = 2


def box_iou(a, b):
    a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def match(gt, boxes, scores):
    order = np.argsort(-np.asarray(scores), kind="stable")
    iou = box_iou([boxes[i] for i in order], gt) if (len(order) and len(gt)) else np.zeros((len(order), len(gt)))
    status, gidx, taken = [], [], set()
    for r in range(len(order)):
        best, bj = IOU_T, -1
        for j in range(len(gt)):
            if j not in taken and iou[r, j] >= best:
                best, bj = iou[r, j], j
        if bj >= 0:
            taken.add(bj); status.append("tp"); gidx.append(bj)
        else:
            status.append("fp"); gidx.append(-1)
    return order, status, gidx


def best_f1_macro(gt, preds, grid=np.round(np.arange(0.05, 0.96, 0.01), 2)):
    P = np.zeros(len(grid)); R = np.zeros(len(grid)); n = 0
    for t, g in gt.items():
        if not g:
            continue
        p = preds.get(t, {"boxes": [], "scores": []})
        order, status, _ = match(g, p["boxes"], p["scores"])
        s = np.asarray(p["scores"])[order] if len(order) else np.zeros(0)
        st = np.asarray(status)
        n += 1
        for k, th in enumerate(grid):
            keep = s >= th
            tp = int((keep & (st == "tp")).sum()); npred = int(keep.sum())
            P[k] += tp / npred if npred else 0.0
            R[k] += tp / len(g)
    P /= n; R /= n
    f1 = 2 * P * R / np.maximum(P + R, 1e-9)
    k = int(np.argmax(f1))
    return float(grid[k]), float(f1[k]), dict(P=float(P[k]), R=float(R[k]))


def draw_boxes(im, boxes, colour, width=2, dash=0):
    d = ImageDraw.Draw(im)
    for b in boxes:
        x1, y1, x2, y2 = [v * UP for v in b]
        if not dash:
            d.rectangle([x1, y1, x2, y2], outline=colour, width=width)
        else:
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
            for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
                L = max(abs(bx - ax), abs(by - ay)); nseg = max(int(L // dash), 1)
                for k in range(0, nseg, 2):
                    fa, fb = k / nseg, min((k + 1) / nseg, 1)
                    d.line([(ax + (bx - ax) * fa, ay + (by - ay) * fa), (ax + (bx - ax) * fb, ay + (by - ay) * fb)],
                           fill=colour, width=width)
    return im


def main():
    gt = json.load(open(os.path.join(NEON, "neon_gt.json")))
    preds = {lab: json.load(open(f)) for lab, f in MODELS}
    tiles = os.environ.get("TILES", DEFAULT_TILES).split(",")
    suffix = os.environ.get("SUFFIX", "")

    thr_path = os.path.join(OUT, "qualitative_thresholds_neon.json")
    thr = json.load(open(thr_path)) if os.path.exists(thr_path) else {}
    for lab, _ in MODELS:
        if lab not in thr:
            th, f1, pr = best_f1_macro(gt, preds[lab])
            thr[lab] = dict(threshold=th, f1=f1, **pr)
            print(f"{lab:18s} best-F1 thr={th:.2f} F1={f1:.4f} {pr}", flush=True)
            json.dump(thr, open(thr_path, "w"), indent=1)

    npan = 1 + len(MODELS); ncol = npan * PER_ROW; nrow = (len(tiles) + PER_ROW - 1) // PER_ROW
    fig_w = 18.4 / 2.54; pw = fig_w / ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, pw * nrow + 0.35), squeeze=False,
                             gridspec_kw=dict(wspace=0.012, hspace=0.015))
    counts = {}
    WHITE, RED, YEL = (255, 255, 255), (255, 40, 40), (255, 235, 59)
    for ti, t in enumerate(tiles):
        r, c0 = divmod(ti, PER_ROW); c0 *= npan
        base = Image.open(os.path.join(NEON, "NeonTreeEvaluation/evaluation/RGB", f"{t}.tif")).convert("RGB")
        base = base.resize((base.width * UP, base.height * UP), Image.LANCZOS)
        g = gt[t]
        site = t.split("_")[1] if t.startswith("20") else t.split("_")[0]

        def show(ax, im, title, label):
            ax.imshow(im, interpolation="lanczos"); ax.set_axis_off()
            if r == 0:
                ax.set_title(title, fontsize=7.5, pad=2.5)
            ax.text(0.015, 0.985, label, transform=ax.transAxes, fontsize=5.6, va="top", ha="left", color="white",
                    bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.65))

        show(axes[r, c0], draw_boxes(base.copy(), g, WHITE, 4), "Ground truth", f"{site}   {len(g)} crowns")
        for ci, (lab, _) in enumerate(MODELS):
            p = preds[lab].get(t, {"boxes": [], "scores": []})
            order, status, gidx = match(g, p["boxes"], p["scores"])
            s = np.asarray(p["scores"])[order] if len(order) else np.zeros(0)
            th = thr[lab]["threshold"]
            tp_b = [p["boxes"][i] for i, st, kp in zip(order, status, s >= th) if kp and st == "tp"]
            fp_b = [p["boxes"][i] for i, st, kp in zip(order, status, s >= th) if kp and st == "fp"]
            matched = {gj for gj, st, kp in zip(gidx, status, s >= th) if kp and st == "tp"}
            fn_b = [g[j] for j in range(len(g)) if j not in matched]
            im = draw_boxes(base.copy(), fn_b, YEL, 4, dash=16)
            im = draw_boxes(im, fp_b, RED, 4)
            im = draw_boxes(im, tp_b, WHITE, 4)
            f1 = 2 * len(tp_b) / max(2 * len(tp_b) + len(fp_b) + len(fn_b), 1)
            counts[f"{t}|{lab}"] = dict(f1=round(f1, 4), tp=len(tp_b), fp=len(fp_b), fn=len(fn_b))
            show(axes[r, c0 + 1 + ci], im, lab, f"F1 {f1:.2f}  TP {len(tp_b)}  FP {len(fp_b)}  FN {len(fn_b)}")
        print("rendered", t, flush=True)

    handles = [Line2D([0], [0], color="white", lw=1.4, label="matched (IoU $\\geq$ 0.4)"),
               Line2D([0], [0], color="#ff2828", lw=1.4, label="false positive"),
               Line2D([0], [0], color="#ffeb3b", lw=1.4, ls=(0, (3, 2)), label="missed crown")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.5, frameon=True, facecolor="#3a3a3a",
               edgecolor="none", labelcolor="white", fancybox=True, bbox_to_anchor=(0.5, -0.003),
               handlelength=2.2, columnspacing=1.6)
    plt.subplots_adjust(left=0.003, right=0.997, top=0.955, bottom=0.05)
    stem = os.path.join(OUT, f"qualitative_neon{suffix}")
    fig.savefig(stem + ".png", dpi=400)
    im = Image.open(stem + ".png").convert("RGB")
    im.save(stem + ".jpg", quality=92, subsampling=0, dpi=(400, 400))
    im.save(stem + ".pdf", "PDF", resolution=400.0, quality=92)
    os.remove(stem + ".png")
    json.dump({"tiles": tiles, "thresholds": thr, "counts": counts}, open(stem + "_meta.json", "w"), indent=1)
    print("wrote", stem + ".pdf")


if __name__ == "__main__":
    main()
