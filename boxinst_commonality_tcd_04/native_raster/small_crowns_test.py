"""Per-crown-size recall on the 439 test tiles at the native 2048 raster: LACE vs Restor.

For every GT crown (size = sqrt of its polygon-extent box area, tile px at 0.1 m/px):
  box outcome   one-to-one greedy-by-score matching on BOX IoU -> TP (>=0.5) / near (0.3-0.5) / FN
  mask outcome  one-to-one greedy-by-score matching on MASK IoU at 2048 -> recalled at >=0.5, >=0.75,
                and the matched mask IoU
Matching uses ALL detections (no maxDets cut, no canopy filter): this is a recall/quality
breakdown, not an AP, so precision is reported separately as predicted-box counts by size.

    .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.small_crowns_test
"""
import json, os
import numpy as np
from pycocotools import mask as maskUtils
from boxinst_commonality_tcd_04 import evaluate as E

NR = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(NR)
ROWS = {"LACE s0 (product, tau 0.40)": f"{NR}/preds_lace_s0_r2048_tau040.json",
        "Restor rpn1000 tree-only": f"{NR}/preds_restor_rpn1000_r2048_treeonly.json",
        "Restor rpn2000 tree-only": f"{NR}/preds_restor_rpn2000_r2048_treeonly.json"}
BINS = [(0, 20), (20, 30), (30, 60), (60, 1e9)]
OUT = f"{NR}/small_crowns_test_r2048.json"


def gtb(p):
    return np.array([[*np.asarray(q, float).reshape(-1, 2).min(0), *np.asarray(q, float).reshape(-1, 2).max(0)]
                     for q in p]) if p else np.zeros((0, 4))


def biou(a, b):
    if not len(a) or not len(b): return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    i = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return i / np.maximum(((a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]))[:, None] + ((b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1]))[None] - i, 1e-9)


def greedy(M, s, thr):
    """one-to-one, greedy by score -> best matched value per GT (or -1)."""
    out = np.full(M.shape[1], -1.0); taken = np.zeros(M.shape[1], bool)
    for i in np.argsort(-s, kind="mergesort"):
        c = np.where(taken, -1, M[i]); j = int(c.argmax()) if len(c) else 0
        if len(c) and c[j] >= thr: taken[j] = True; out[j] = c[j]
    return out


def main():
    assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
    gt = json.load(open(f"{PKG}/test_gt.json")); tids = sorted(gt)
    G = {}
    for k, t in enumerate(tids):
        g = gtb(gt[t]["trees"])
        G[t] = (g, [maskUtils.encode(np.asfortranarray(m.astype(np.uint8))) for m in E.raster(gt[t]["trees"], res=2048, scale=1.0)])
        if (k + 1) % 100 == 0: print(f"  GT {k+1}/439", flush=True)
    size = np.concatenate([np.sqrt((G[t][0][:, 2]-G[t][0][:, 0])*(G[t][0][:, 3]-G[t][0][:, 1])) for t in tids if len(G[t][0])])
    res = {"n_gt": int(len(size)), "bins": [f"{lo}-{hi if hi < 1e9 else 'inf'}px" for lo, hi in BINS],
           "n_gt_per_bin": [int(((size >= lo) & (size < hi)).sum()) for lo, hi in BINS], "rows": {}}
    for lab, fp in ROWS.items():
        P = json.load(open(fp))["preds"]; bx, mk, psz = [], [], []
        for t in tids:
            g, gr = G[t]
            if not len(g): continue
            r = P.get(t) or {"boxes_2048": [], "scores": [], "masks_rle": []}
            p = np.asarray(r["boxes_2048"], float).reshape(-1, 4); s = np.asarray(r["scores"], float)
            psz += list(np.sqrt(np.clip((p[:, 2]-p[:, 0])*(p[:, 3]-p[:, 1]), 0, None)))
            if not len(p):
                bx.append(np.full(len(g), -1.0)); mk.append(np.full(len(g), -1.0)); continue
            pr = [{"size": x["size"], "counts": x["counts"].encode()} for x in r["masks_rle"]]
            assert tuple(pr[0]["size"]) == (2048, 2048)
            bx.append(greedy(biou(p, g), s, 0.3))
            mk.append(greedy(np.asarray(maskUtils.iou(pr, gr, [0]*len(gr))).reshape(len(pr), len(gr)), s, 0.5))
        bx, mk, psz = np.concatenate(bx), np.concatenate(mk), np.array(psz)
        row = []
        for lo, hi in BINS:
            sel = (size >= lo) & (size < hi)
            row.append({"box_TP": round(float((bx[sel] >= 0.5).mean()), 4), "box_near": round(float(((bx[sel] >= 0.3) & (bx[sel] < 0.5)).mean()), 4),
                        "box_FN": round(float((bx[sel] < 0.3).mean()), 4),
                        "mask_recall_50": round(float((mk[sel] >= 0.5).mean()), 4), "mask_recall_75": round(float((mk[sel] >= 0.75).mean()), 4),
                        "mean_mask_iou_matched": round(float(mk[sel][mk[sel] >= 0.5].mean()), 4) if (mk[sel] >= 0.5).any() else None,
                        "n_pred_boxes_in_size_range": int(((psz >= lo) & (psz < hi)).sum())})
        res["rows"][lab] = row
        print(lab, flush=True)
        for (lo, hi), b in zip(BINS, row): print(f"   {lo:>2}-{hi if hi < 1e9 else 'inf':>3}px  {b}", flush=True)
    json.dump(res, open(OUT, "w"), indent=2); print("->", OUT)


if __name__ == "__main__":
    main()
