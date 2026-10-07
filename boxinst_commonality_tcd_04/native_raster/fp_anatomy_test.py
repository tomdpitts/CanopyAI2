"""Why does LACE lose AP50 to Restor at 2048 despite higher recall? Precision-recall anatomy.

439 test tiles, native 2048 raster, mask IoU 0.5, one-to-one greedy-by-score matching per tile
(COCO rule), canopy-neutral by the project's legacy rule (an UNMATCHED prediction with >50% of
its area in canopy is dropped, not counted FP) -- equal to score_coco's crowd arm at AP50 to
within 1e-4. Pooled over tiles in global score order; no maxDets cut.

Every FP is classified by its best mask IoU with ANY GT crown on its tile:
  dup    >= 0.5 with a GT already taken by a higher-scored prediction (duplicate)
  loc    0.1 - 0.5 with some GT (overlaps a real crown, mask/box not good enough)
  bg     < 0.1 with every GT (fires on nothing labelled)
Reports precision at fixed recalls and the FP anatomy inside the top-k needed to reach them.

    .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.fp_anatomy_test
"""
import json, os
import numpy as np
from pycocotools import mask as maskUtils
from boxinst_commonality_tcd_04 import evaluate as E

NR = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(NR)
ROWS = {"LACE s0 (product)": f"{NR}/preds_lace_s0_r2048_tau040.json",
        "Restor rpn1000 tree": f"{NR}/preds_restor_rpn1000_r2048_treeonly.json",
        "Restor rpn2000 tree": f"{NR}/preds_restor_rpn2000_r2048_treeonly.json"}
OUT = f"{NR}/fp_anatomy_test_r2048.json"
RECALLS = (0.5, 0.6, 0.7)


def main():
    assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
    gt = json.load(open(f"{PKG}/test_gt.json")); tids = sorted(gt)
    G = {}
    for k, t in enumerate(tids):
        enc = lambda ms: [maskUtils.encode(np.asfortranarray(m.astype(np.uint8))) for m in ms]
        tr = enc(E.raster(gt[t]["trees"], res=2048, scale=1.0)); cn = enc(E.raster(gt[t]["canopy"], res=2048, scale=1.0))
        G[t] = (tr, maskUtils.merge(cn) if cn else None)
        if (k + 1) % 100 == 0: print(f"  GT {k+1}/439", flush=True)
    n_gt = sum(len(G[t][0]) for t in tids)
    out = {"n_gt": n_gt, "rows": {}}
    for lab, fp in ROWS.items():
        P = json.load(open(fp))["preds"]; S, KIND, SZ = [], [], []
        for t in tids:
            r = P.get(t)
            if not r or not r["scores"]: continue
            pr = [{"size": x["size"], "counts": x["counts"].encode()} for x in r["masks_rle"]]
            s = np.asarray(r["scores"], float); tr, cu = G[t]
            M = np.asarray(maskUtils.iou(pr, tr, [0]*len(tr))).reshape(len(pr), len(tr)) if tr else np.zeros((len(pr), 0))
            area = np.array([maskUtils.area(x) for x in pr], float)
            inc = (np.array([maskUtils.area(maskUtils.merge([x, cu], intersect=1)) for x in pr], float) / np.maximum(area, 1)
                   if cu is not None else np.zeros(len(pr)))
            bx = np.asarray(r["boxes_2048"], float).reshape(-1, 4); sz = np.sqrt(np.clip((bx[:, 2]-bx[:, 0])*(bx[:, 3]-bx[:, 1]), 0, None))
            taken = np.zeros(M.shape[1], bool)
            for i in np.argsort(-s, kind="mergesort"):
                row = M[i] if M.shape[1] else np.zeros(0)
                c = np.where(taken, -1, row); j = int(c.argmax()) if len(c) else -1
                if j >= 0 and c[j] >= 0.5:
                    taken[j] = True; k_ = "tp"
                elif inc[i] > 0.5:
                    continue                                # canopy-neutral: dropped
                elif len(row) and row.max() >= 0.5:
                    k_ = "dup"
                elif len(row) and row.max() >= 0.1:
                    k_ = "loc"
                else:
                    k_ = "bg"
                S.append(s[i]); KIND.append(k_); SZ.append(sz[i])
        S, KIND, SZ = np.array(S), np.array(KIND), np.array(SZ)
        o = np.argsort(-S, kind="mergesort"); KIND, SZ = KIND[o], SZ[o]
        tp = np.cumsum(KIND == "tp"); rec = tp / n_gt; prec = tp / np.arange(1, len(tp) + 1)
        ap = float(np.mean([prec[rec >= r].max() if (rec >= r).any() else 0 for r in np.linspace(0, 1, 101)]))
        row = {"ap50_check": round(ap, 4), "max_recall": round(float(rec[-1]), 4), "n_kept": int(len(KIND)), "at_recall": {}}
        for R in RECALLS:
            if rec[-1] < R: row["at_recall"][R] = None; continue
            k = int(np.argmax(rec >= R)) + 1; top = KIND[:k]; nfp = int((top != "tp").sum())
            row["at_recall"][R] = {"precision": round(float(prec[k-1]), 4), "n_dets": k, "n_fp": nfp,
                                   **{f"fp_{c}": round(float((top == c).sum() / max(nfp, 1)), 3) for c in ("dup", "loc", "bg")},
                                   "fp_small_lt20px": round(float(((top != "tp") & (SZ[:k] < 20)).sum() / max(nfp, 1)), 3)}
        out["rows"][lab] = row; print(lab, json.dumps(row), flush=True)
    json.dump(out, open(OUT, "w"), indent=2); print("->", OUT)


if __name__ == "__main__":
    main()
