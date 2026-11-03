"""VAL: the 10 highest-ranked mislocalised false positives of C2 (NMS 0.5, s' ranking = deployed), cropped. Mislocalised =
mask FP in the crowd-arm segm eval at IoU 0.5 whose best mask IoU with a non-crowd GT is in [0.1, 0.5). One per tile.
Panel: RGB crop; GT crowns green (the best-IoU GT thick), predicted mask red fill + outline, predicted box yellow.
Cause: a = box off (box IoU with that GT < 0.5); b = box ok, mask wrong (box IoU >= 0.5); c = merge (box covers >= 50%
of 2+ GT boxes). -> results/figs/mislocalised_val_C2.png (+ .json with the cases).
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.viz_misloc
"""
import contextlib, io, json, os
import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); HERE = os.path.join(PKG, "dethead"); REPO = os.path.dirname(PKG)
VAL_GT = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "val_gt.json")
MASKS = "/Users/tompitts/dphil/feat_cache/dethead_fpnC2_s0_val_nms05_mask.json"
OUT = os.path.join(HERE, "results", "figs"); N = 10; PAD = 40


def biou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(x1 - x0, 0) * max(y1 - y0, 0); return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from PIL import Image
    from pycocotools import mask as MU
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    vg = json.load(open(VAL_GT)); tids = sorted(vg); gd, t2i = SL.build_gt_crowd(vg, 2048); i2t = {v: k for k, v in t2i.items()}
    V = json.load(open(MASKS))
    preds = {t: {"boxes_2048": V[t]["boxes_2048"], "scores": V[t]["s_prime"], "masks_rle": V[t]["masks_rle"]} for t in tids}
    dets = SL.build_dt(preds, t2i, 2048, 0.0, with_masks=True)
    src = [(t, i) for t in preds for i in range(len(preds[t]["scores"]))]
    cg = COCO(); cg.dataset = gd; cg.createIndex()
    with contextlib.redirect_stdout(io.StringIO()):
        cd = cg.loadRes([dict(d) for d in dets]); ev = COCOeval(cg, cd, "segm")
        ev.params.iouThrs = np.array([0.5]); ev.params.maxDets = [600]; ev.params.areaRng = [[0, 1e10]]; ev.params.areaRngLbl = ["all"]; ev.evaluate()
    cands = []
    for e in ev.evalImgs:
        if e is None:
            continue
        I = np.asarray(ev.ious[(e["image_id"], 1)]); G = ev._gts[e["image_id"], 1]
        for k, did in enumerate(e["dtIds"]):
            if e["dtIgnore"][0][k] or e["dtMatches"][0][k] > 0 or not I.size:
                continue
            row = I[k].copy(); row[[g["iscrowd"] == 1 for g in G]] = -1; j = int(row.argmax())
            if 0.1 <= row[j] < 0.5:
                cands.append((e["dtScores"][k], did, G[j]["id"], float(row[j])))
    cands.sort(key=lambda c: -c[0]); picked, seen = [], set()
    for c in cands:
        t, i = src[c[1] - 1]
        if t not in seen:
            picked.append(c); seen.add(t)
        if len(picked) == N:
            break
    gtb = {a["id"]: a["bbox"] for a in gd["annotations"]}
    fig, axs = plt.subplots(2, 5, figsize=(20, 8.6)); cases = []
    for ax, (score, did, gid, miou) in zip(axs.ravel(), picked):
        t, i = src[did - 1]; b = preds[t]["boxes_2048"][i]; gb = gtb[gid]; gxy = [gb[0], gb[1], gb[0] + gb[2], gb[1] + gb[3]]
        bi = biou(b, gxy)
        trees = [np.asarray(p, float).reshape(-1, 2) for p in vg[t]["trees"] if len(p) >= 6]
        tb = [[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in trees]
        cov = [max(min(b[2], q[2]) - max(b[0], q[0]), 0) * max(min(b[3], q[3]) - max(b[1], q[1]), 0) / max((q[2] - q[0]) * (q[3] - q[1]), 1e-9) for q in tb]
        cause = "c merge" if sum(c >= 0.5 for c in cov) >= 2 else ("a box off" if bi < 0.5 else "b mask wrong")
        x0 = int(max(min(b[0], gxy[0]) - PAD, 0)); y0 = int(max(min(b[1], gxy[1]) - PAD, 0))
        x1 = int(min(max(b[2], gxy[2]) + PAD, 2048)); y1 = int(min(max(b[3], gxy[3]) + PAD, 2048))
        rgb = np.asarray(Image.open(os.path.join(REPO, "data", "tcd", "train", t + ".tif")).convert("RGB"))[y0:y1, x0:x1]
        m = MU.decode(SL._as_bytes(preds[t]["masks_rle"][i]))[y0:y1, x0:x1].astype(bool)
        ax.imshow(rgb); ov = np.zeros((*m.shape, 4)); ov[m] = (1, 0, 0, 0.35); ax.imshow(ov); ax.contour(m, levels=[0.5], colors="red", linewidths=1.2)
        for p, q in zip(trees, tb):
            if q[2] < x0 or q[0] > x1 or q[3] < y0 or q[1] > y1:
                continue
            thick = abs(q[0] - gxy[0]) < 1.5 and abs(q[1] - gxy[1]) < 1.5 and abs(q[2] - gxy[2]) < 1.5 and abs(q[3] - gxy[3]) < 1.5
            pp = np.vstack([p, p[:1]]); ax.plot(pp[:, 0] - x0, pp[:, 1] - y0, color="lime", lw=2.4 if thick else 0.9)
        ax.add_patch(plt.Rectangle((b[0] - x0, b[1] - y0), b[2] - b[0], b[3] - b[1], fill=False, ec="yellow", lw=1.5, ls="--"))
        ax.set_xlim(0, x1 - x0); ax.set_ylim(y1 - y0, 0)
        ax.set_title(f"{t}  s'={score:.2f}\nmask IoU {miou:.2f} | box IoU {bi:.2f} | {cause}", fontsize=9); ax.axis("off")
        cases.append({"tile": t, "det_index": i, "s_prime": round(score, 4), "mask_iou": round(miou, 3), "box_iou": round(bi, 3), "cause": cause,
                      "box": [round(v, 1) for v in b], "gt_box": [round(v, 1) for v in gxy]})
    fig.suptitle("C2 val (NMS 0.5, s' ranking): 10 highest-ranked mislocalised FPs (mask IoU 0.1-0.5), one per tile.  "
                 "green = GT crowns (thick = nearest), red = predicted mask, yellow = predicted box", fontsize=11)
    fig.tight_layout(); os.makedirs(OUT, exist_ok=True); fp = os.path.join(OUT, "mislocalised_val_C2.png"); fig.savefig(fp, dpi=110)
    n_all = len(cands)
    json.dump({"n_mislocalised_val": n_all, "cases": cases}, open(fp.replace(".png", ".json"), "w"), indent=1)
    print(fp, n_all, [c["cause"] for c in cases])


if __name__ == "__main__":
    main()
