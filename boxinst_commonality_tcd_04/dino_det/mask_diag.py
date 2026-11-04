"""Diagnostic (2026-11-04, Tom: "are we sure boundary precision is even the issue? are SAM 3 masks smaller/larger?"): LACE vs SAM 3
masks on the SAME frozen D1 boxes, test 439 (SAM masks cached in feat_cache/sam3_d1_test). No selection uses this.

Each detection is paired to the GT tree with the highest BOX IoU (>= 0.5; method-neutral, and the same pair for both maskers).
Per pair and per masker: mask IoU, area ratio pred/GT, over-segmentation (pred outside GT)/GT, under-segmentation (GT missed)/GT,
centroid offset (px), boundary roughness (perimeter^2 / (4 pi area); 1 = disc), and outright failure (mask IoU < 0.5).
Binned by GT equivalent diameter. -> results/mask_diag_lace_vs_sam3_test.json
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.mask_diag
"""
import json, os, time
import numpy as np

from boxinst_commonality_tcd_04.dino_det import rerank as R

SAMD = os.path.join(R.FC_, "sam3_d1_test"); BINS = ((0, 24), (24, 48), (48, 96), (96, 1e9))


def _perim(m):
    from scipy import ndimage as ndi
    return float((m & ~ndi.binary_erosion(m, ndi.generate_binary_structure(2, 1))).sum())


def _tile(a):
    from pycocotools import mask as M
    t, L, S, boxes, gts = a; out = []
    if not gts or not boxes:
        return out
    b = np.asarray(boxes, float).reshape(-1, 4)
    gm = [M.decode(g).astype(bool) for g in gts]; gb = np.asarray([M.toBbox(g) for g in gts], float); gb[:, 2:] += gb[:, :2]
    bi = R._box_iou(b, gb); j = bi.argmax(1)
    for i in np.flatnonzero(bi[np.arange(len(b)), j] >= 0.5):
        g = gm[j[i]]; ys, xs = np.nonzero(g); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        lm, sm = M.decode(L[i]).astype(bool), M.decode(S[i]).astype(bool)
        # crop to the union bbox of GT and both masks, padded
        U = g | lm | sm; uy, ux = np.nonzero(U); p = 3
        cy0, cy1, cx0, cx1 = max(0, uy.min() - p), min(2048, uy.max() + p + 1), max(0, ux.min() - p), min(2048, ux.max() + p + 1)
        gc, lc, sc = g[cy0:cy1, cx0:cx1], lm[cy0:cy1, cx0:cx1], sm[cy0:cy1, cx0:cx1]; ga = gc.sum(); gcen = np.argwhere(gc).mean(0)
        row = {"gt_d": float(np.sqrt(ga * 4 / np.pi)), "gt_round": _perim(gc) ** 2 / (4 * np.pi * ga)}
        for k, m in (("lace", lc), ("sam", sc)):
            a_ = m.sum(); inter = (m & gc).sum()
            row[k] = {"iou": float(inter / max(1, (m | gc).sum())), "ratio": float(a_ / ga), "over": float((m & ~gc).sum() / ga),
                      "under": float((gc & ~m).sum() / ga), "cen": float(np.hypot(*(np.argwhere(m).mean(0) - gcen))) if a_ else np.nan,
                      "round": _perim(m) ** 2 / (4 * np.pi * a_) if a_ else np.nan}
        row["ls_iou"] = float((lc & sc).sum() / max(1, (lc | sc).sum())); out.append(row)
    return out


def main():
    import multiprocessing as mp
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    G = json.load(open(R.TEST_GT)); St = json.load(open(os.path.join(R.OUT, f"preds_{R.TAG}_test_sprime_r2048.json")))["preds"]; tids = sorted(St)
    gd, t2i = SL.build_gt_crowd(G, 2048); by = {i: [] for i in t2i.values()}
    for a in gd["annotations"]:
        if not a["iscrowd"]:
            by[a["image_id"]].append(a["segmentation"])
    jobs = []
    for t in tids:
        S = json.load(open(os.path.join(SAMD, t + ".json"))); assert S["boxes_2048"] == St[t]["boxes_2048"], t
        jobs.append((t, [SL._as_bytes(r) for r in St[t]["masks_rle"]], [SL._as_bytes(r) for r in S["masks_rle"]], St[t]["boxes_2048"], by[t2i[t]]))
    t0 = time.time()
    with mp.get_context("spawn").Pool(14) as pool:
        rows = sum(pool.map(_tile, jobs, chunksize=4), [])
    print(f"[diag] {len(rows)} pairs {time.time()-t0:.0f}s", flush=True)
    d = np.asarray([r["gt_d"] for r in rows]); rec = {"n_pairs": len(rows), "bins": {}}
    for lo, hi in BINS + ((0, 1e9),):
        k = (d >= lo) & (d < hi); R_ = [r for r, kk in zip(rows, k) if kk]; b = {"n": int(k.sum())}
        for m in ("lace", "sam"):
            v = lambda f: np.asarray([r[m][f] for r in R_], float)
            b[m] = {"iou_med": round(float(np.median(v("iou"))), 3), "iou_mean": round(float(np.mean(v("iou"))), 3),
                    "ratio_med": round(float(np.median(v("ratio"))), 3), "over_mean": round(float(np.mean(v("over"))), 3),
                    "under_mean": round(float(np.mean(v("under"))), 3), "cen_med_px": round(float(np.nanmedian(v("cen"))), 2),
                    "round_med": round(float(np.nanmedian(v("round"))), 3), "fail_iou_lt_05": round(float((v("iou") < 0.5).mean()), 3),
                    "frac_iou_ge_075": round(float((v("iou") >= 0.75).mean()), 3)}
        li, si = np.asarray([r["lace"]["iou"] for r in R_]), np.asarray([r["sam"]["iou"] for r in R_])
        b["gt_round_med"] = round(float(np.median([r["gt_round"] for r in R_])), 3)
        b["sam_better_frac"] = round(float((si > li).mean()), 3); b["sam_minus_lace_iou_mean"] = round(float((si - li).mean()), 4)
        b["lace_ok_sam_fail"] = round(float(((li >= 0.5) & (si < 0.5)).mean()), 3); b["sam_ok_lace_fail"] = round(float(((si >= 0.5) & (li < 0.5)).mean()), 3)
        b["lace_sam_iou_med"] = round(float(np.median([r["ls_iou"] for r in R_])), 3)
        rec["bins"][f"{lo}-{int(min(hi, 9999))}px"] = b; print(f"[diag] {lo}-{hi}: {json.dumps(b)}", flush=True)
    json.dump(rec, open(os.path.join(R.OUT, "mask_diag_lace_vs_sam3_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
