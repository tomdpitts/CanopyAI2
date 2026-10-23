"""Compare two mask sets on the same VAL detections (same boxes, LACE product order, legacy canopy rule, mask IoU 0.5).
A spec is DIR:KEY -- per-tile JSON in feat_cache/DIR with masks under masks_rle[KEY] (or masks_rle if KEY is '-').
Reports TP gains/losses of B vs A by GT size, and for each: IoU under A and B vs the crown, recall/precision of A and B,
emptiness, area/GT, box IoU, and whether B's mask extends beyond the box clip.

  PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.hunt.diag_pair A_SPEC B_SPEC OUT.json
"""
import json, os, sys
import numpy as np
from pycocotools import mask as mu
from boxinst_commonality_tcd_04 import evaluate as E, score_coco as SCC
from boxinst_commonality_tcd_04.native_raster.hunt import fastap as F
from boxinst_commonality_tcd_04.native_raster.guided_ablate_modal import box_iou_mat
from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as G

EDGES = (622.0, 2456.3)


def load(spec, t):
    d, k = spec.split(":")
    r = json.load(open(f"{F.FC}/{d}/{t}.json"))
    return r, (r["masks_rle"] if k == "-" else r["masks_rle"][k])


def outside_frac(rle, b):
    m = mu.decode(rle).astype(bool); tot = m.sum()
    if not tot:
        return 0.0
    y0, y1, x0, x1 = max(0, int(b[1])), int(np.ceil(b[3])), max(0, int(b[0])), int(np.ceil(b[2]))
    return float(1 - m[y0:y1, x0:x1].sum() / tot)


def main():
    sa, sb, out = sys.argv[1], sys.argv[2], sys.argv[3]
    assert not os.path.exists(out), out
    gtc = F.gt_cache("val"); gt = json.load(open(F.VAL_GT)); tids = sorted(gtc)
    rows, SA, SB, SC = [], [], [], []
    for t in tids:
        ra, ma = load(sa, t); rb, mb = load(sb, t)
        assert ra["boxes_2048"] == rb["boxes_2048"]
        sc = np.asarray(ra["scores_product"], np.float64); bx = np.asarray(ra["boxes_2048"]).reshape(-1, 4)
        A, B = F.enc(ma), F.enc(mb)
        iA, gA = F.tile_stats(A, gtc[t]); iB, gB = F.tile_stats(B, gtc[t]); SA.append((iA, gA)); SB.append((iB, gB)); SC.append(sc)
        if not len(sc):
            continue
        tp = {}
        for nm, iou, ig in (("A", iA, gA), ("B", iB, gB)):
            o, _, tpk, _, gid = E.match_tile(iou, sc, ig, 0.5)
            f = np.zeros(len(sc), bool); g_ = np.full(len(sc), -1); f[o] = tpk; g_[o] = gid; tp[nm] = (f, g_)
        polys = gt[t]["trees"]; gb = G.gt_boxes_2048(polys).astype(np.float64)
        biou = box_iou_mat(bx, gb); aA = SCC._areas(A); aB = SCC._areas(B); aG = SCC._areas(gtc[t]["trees"])
        for i in np.flatnonzero(tp["A"][0] ^ tp["B"][0]):
            gain = bool(tp["B"][0][i]); j = tp["B"][1][i] if gain else tp["A"][1][i]
            def pr(iou, a):
                inter = iou * (a + aG[j]) / (1 + iou) if a else 0.0
                return inter / aG[j], (inter / a if a else 0.0)
            rA, pA = pr(iA[i, j], aA[i]); rB, pB = pr(iB[i, j], aB[i])
            rows.append(dict(tile=t, det=int(i), gain=gain, score=float(sc[i]), areaG=int(aG[j]), iouA=float(iA[i, j]), iouB=float(iB[i, j]),
                             recA=rA, precA=pA, recB=rB, precB=pB, emptyA=bool(aA[i] == 0), emptyB=bool(aB[i] == 0),
                             areaA_G=float(aA[i] / aG[j]), areaB_G=float(aB[i] / aG[j]), box_iou=float(biou[i, j]),
                             outB=outside_frac(B[i], bx[i]), outA=outside_frac(A[i], bx[i])))
    n_gt = {t: len(gtc[t]["trees"]) for t in tids}
    summ = {"A": sa, "B": sb, "AP50": {"A": F.ap_halves(tids, SA, SC, n_gt), "B": F.ap_halves(tids, SB, SC, n_gt)},
            "AP75": {"A": F.ap_halves(tids, SA, SC, n_gt, .75), "B": F.ap_halves(tids, SB, SC, n_gt, .75)}, "by_size": {}}
    for st, lo, hi in (("small", 0, EDGES[0]), ("mid", EDGES[0], EDGES[1]), ("large", EDGES[1], 1e12)):
        row = {}
        for kind in ("gain", "loss"):
            x = [r for r in rows if lo <= r["areaG"] < hi and r["gain"] == (kind == "gain")]
            f = lambda k: round(float(np.median([r[k] for r in x])), 3) if x else None
            row[kind] = {"n": len(x), **{k: f(k) for k in ("iouA", "iouB", "recA", "precA", "recB", "precB", "areaA_G", "areaB_G", "box_iou", "outA", "outB", "score")},
                         "emptyA": round(float(np.mean([r["emptyA"] for r in x])), 3) if x else None}
        summ["by_size"][st] = row
    json.dump({"summary": summ, "rows": rows}, open(out, "w"))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
