"""DRAFT re-pick of Figure 6 (fig:qualgrid), ranking on what the panels show (2026-11-26, Tom). Not yet in paper.tex.

Why: v1-v3 ranked tiles by the LACE - Restor AP50 margin over the WHOLE 2048 px tile, but each panel shows a 512 px crop
at a single display threshold. Area and metric both differ from what is drawn, so a row's percentile label need not match
its panels (v3 median row, tile 333: tile margin +0.047, yet in-crop LACE F1 0.85 < Restor 0.89).

Rule v4 (fixed before any v4 pick was viewed): candidates are v1's (OAM-TCD 439 holdout tiles with >= 10 GT crowns,
tile 122 excluded) with v1's auto-crop (512 px window holding most GT centroids). POOL = candidates whose auto-crop is
OPEN (OAM-TCD canopy class < 2 % of the crop) and holds >= 10 GT crowns. For each pool tile, in-crop F1 of LACE and of
Restor is computed exactly as make_qualitative.py draws and counts it (each model above its whole-set best-F1 display
threshold, qualitative_thresholds_tcd439_grid3.json; greedy score-ordered matching at mask IoU >= 0.5 over the whole tile;
canopy-ignored predictions dropped; a mask or GT crown counts if any pixel lies in the crop). CROP MARGIN = F1_LACE -
F1_Restor. Percentiles are taken over the whole pool. Rows q25 / q50 / q75: the tile with crop margin closest to that
percentile among pool tiles with LACE in-crop F1 >= 0.8 (display rule), no tile used twice. Failure row: the pool tile
(no F1 filter) closest to the 5th percentile. Ties broken by more in-crop GT crowns, then tile name.

Usage: .venv/bin/python boxinst_commonality_tcd_04/ablation/results/figs/select_qual_grid_v4.py
Writes qualitative_grid_select_tcd439_v4.json next to this script; progress to stdout.
"""
import json, os, sys
import numpy as np
from pycocotools import mask as mu

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from select_qual_tile import rles, MODELS  # noqa: E402
from select_qual_grid import MIN_CROWNS, EXCLUDE, GT  # noqa: E402
import make_qualitative as MQ  # noqa: E402

CROP, MAX_CROP_CANOPY, MIN_CROP_CROWNS, MIN_F1 = 512, 0.02, 10, 0.8
QS, FAIL_Q = (0.25, 0.50, 0.75), 0.05
THR_FILE = os.path.join(HERE, "qualitative_thresholds_tcd439_grid3.json")
THR = {"LACE": json.load(open(THR_FILE))["LACE (ours)"]["threshold"], "Restor": json.load(open(THR_FILE))["Restor"]["threshold"]}


def crop_counts(g_rles, gt_masks, canopy, p, thr, x0, y0):
    """In-crop TP/FP/FN/F1 exactly as make_qualitative.py counts them."""
    inwin = lambda m: bool(m[y0:y0 + CROP, x0:x0 + CROP].any())
    order, status, gidx = MQ.match_tile(g_rles, canopy, p["masks_rle"], p["scores"])
    s = np.asarray(p["scores"])[order] if len(order) else np.zeros(0)
    tp = fp = 0; matched = set()
    for i, st, gj, kp in zip(order, status, gidx, s >= thr):
        if not kp or st == "ign":
            continue
        if st == "tp":
            matched.add(gj)
        if inwin(mu.decode(p["masks_rle"][i])):
            tp += st == "tp"; fp += st != "tp"
    fn = sum(inwin(gt_masks[j]) for j in range(len(gt_masks)) if j not in matched)
    return dict(f1=round(2 * tp / max(2 * tp + fp + fn, 1), 4), tp=int(tp), fp=int(fp), fn=int(fn))


def pick(rows, target, used):
    c = [r for r in rows if r["tile"] not in used]
    return sorted(c, key=lambda r: (round(abs(r["crop_margin"] - target), 6), -r["crop_crowns"], r["tile"]))


def main():
    gt = json.load(open(GT))
    preds = {k: json.load(open(v))["preds"] for k, v in MODELS.items()}
    for k in preds:  # pycocotools wants bytes counts
        for t in preds[k]:
            for r in preds[k][t]["masks_rle"]:
                if isinstance(r["counts"], str):
                    r["counts"] = r["counts"].encode()
    pool, n_cand = [], 0
    for n, (t, g) in enumerate(gt.items()):
        tr = rles(g["trees"])
        if t in EXCLUDE or len(tr) < MIN_CROWNS:
            continue
        n_cand += 1
        gt_masks = [mu.decode(x).astype(bool) for x in tr]
        x0, y0 = MQ.auto_crop(gt_masks, CROP)
        cn = rles(g["canopy"]); canopy = mu.merge(cn, intersect=False) if cn else None
        ccf = float(mu.decode(canopy)[y0:y0 + CROP, x0:x0 + CROP].mean()) if canopy is not None else 0.0
        ncr = sum(bool(m[y0:y0 + CROP, x0:x0 + CROP].any()) for m in gt_masks)
        if ccf >= MAX_CROP_CANOPY or ncr < MIN_CROP_CROWNS:
            continue
        c = {k: crop_counts(tr, gt_masks, canopy, preds[k][t], THR[k], x0, y0) for k in MODELS}
        pool.append(dict(tile=t, crop_xy=[int(x0), int(y0)], crop_canopy_frac=round(ccf, 4), crop_crowns=int(ncr),
                         lace=c["LACE"], restor=c["Restor"], crop_margin=round(c["LACE"]["f1"] - c["Restor"]["f1"], 4)))
        if (n + 1) % 50 == 0:
            print(f"scanned {n + 1}/{len(gt)} tiles, pool {len(pool)}", flush=True)
    m = np.array([r["crop_margin"] for r in pool])
    el = [r for r in pool if r["lace"]["f1"] >= MIN_F1]
    print(f"candidates {n_cand}, pool (open crop, >= {MIN_CROP_CROWNS} crowns) {len(pool)}, LACE F1 >= {MIN_F1}: {len(el)}; "
          f"LACE ahead in-crop on {float((m > 0).mean()):.0%}, tied {float((m == 0).mean()):.0%}", flush=True)
    out = dict(rule=__doc__.strip().split("\n\n")[2], thresholds=THR, n_candidates=n_cand, n_pool=len(pool), n_eligible=len(el),
               frac_lace_ahead=round(float((m > 0).mean()), 4), frac_tied=round(float((m == 0).mean()), 4), rows={})
    used = set()
    for q in QS:
        tq = float(np.quantile(m, q)); ranked = pick(el, tq, used); used.add(ranked[0]["tile"])
        out["rows"][f"q{int(q * 100)}"] = dict(quantile=q, margin_at_q=round(tq, 4), chosen=ranked[0], next3=ranked[1:4])
    tq = float(np.quantile(m, FAIL_Q)); ranked = pick(pool, tq, used)
    out["failure"] = dict(quantile=FAIL_Q, margin_at_q=round(tq, 4), chosen=ranked[0], next3=ranked[1:4])
    for k, v in list(out["rows"].items()) + [("failure", out["failure"])]:
        r = v["chosen"]
        print(f"{k:8s} q margin {v['margin_at_q']:+.4f} -> {r['tile']} crop margin {r['crop_margin']:+.4f} "
              f"LACE F1 {r['lace']['f1']} Restor F1 {r['restor']['f1']} crowns {r['crop_crowns']}", flush=True)
    out["pool"] = pool
    json.dump(out, open(os.path.join(HERE, "qualitative_grid_select_tcd439_v4.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
