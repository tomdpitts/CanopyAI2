"""Rank tiles by LACE's per-tile F1 margin over the best baseline, at the display thresholds.

Writes qualitative_tile_ranking_<set>.json: every eligible tile (canopy area < 0.5 %, >= MIN_GT crowns)
with each model's whole-tile F1 at its cached best-F1 threshold, sorted by LACE margin.
Usage: .venv/bin/python select_tiles.py --set tcd439|sparse|neon
"""
import argparse, json, os, sys
import numpy as np
from pycocotools import mask as mu

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import make_qualitative as Q
import make_qualitative_neon as N

MIN_GT = 10
CANOPY_MAX = float(os.environ.get("CANOPY_MAX", "0.005"))


def f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def rank_masks(setname):
    cfg = Q.SETS[setname]
    gt = json.load(open(cfg["gt"]))
    preds = {lab: json.load(open(os.path.join(Q.NR, f)))["preds"] for lab, f in cfg["models"]}
    thr = json.load(open(os.path.join(HERE, f"qualitative_thresholds_{setname}.json")))
    rows = []
    for t, g in gt.items():
        if len(g["trees"]) < MIN_GT:
            continue
        can = Q.union_rle(Q.polys_to_rles(g["canopy"]))
        cfrac = float(mu.area(can)) / Q.RES ** 2 if can is not None else 0.0
        if cfrac > CANOPY_MAX:
            continue
        gt_rles = Q.polys_to_rles(g["trees"])
        row = dict(tile=t, n_gt=len(gt_rles), canopy=round(cfrac, 4))
        for lab, _ in cfg["models"]:
            p = preds[lab][t]
            order, status, _ = Q.match_tile(gt_rles, can, p["masks_rle"], p["scores"])
            s = np.asarray(p["scores"])[order] if len(order) else np.zeros(0)
            st = np.asarray(status); keep = s >= thr[lab]["threshold"]
            tp = int((keep & (st == "tp")).sum()); fp = int((keep & (st == "fp")).sum())
            row[lab] = round(f1(tp, fp, len(gt_rles) - tp), 4)
        base = max(v for k, v in row.items() if k not in ("tile", "n_gt", "canopy", "LACE (ours)"))
        row["margin"] = round(row["LACE (ours)"] - base, 4)
        rows.append(row); print(row, flush=True)
    return rows


def rank_neon():
    gt = json.load(open(os.path.join(N.NEON, "neon_gt.json")))
    preds = {lab: json.load(open(f)) for lab, f in N.MODELS}
    thr = json.load(open(os.path.join(HERE, "qualitative_thresholds_neon.json")))
    rows = []
    for t, g in gt.items():
        if len(g) < MIN_GT:
            continue
        row = dict(tile=t, n_gt=len(g))
        for lab, _ in N.MODELS:
            p = preds[lab].get(t, {"boxes": [], "scores": []})
            order, status, _ = N.match(g, p["boxes"], p["scores"])
            s = np.asarray(p["scores"])[order] if len(order) else np.zeros(0)
            st = np.asarray(status); keep = s >= thr[lab]["threshold"]
            tp = int((keep & (st == "tp")).sum()); fp = int((keep & (st == "fp")).sum())
            row[lab] = round(f1(tp, fp, len(g) - tp), 4)
        row["margin"] = round(row["LACE (ours)"] - row["DeepForest 2.1.0"], 4)
        rows.append(row)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True); a = ap.parse_args()
    rows = rank_neon() if a.set == "neon" else rank_masks(a.set)
    rows.sort(key=lambda r: -r["margin"])
    json.dump(rows, open(os.path.join(HERE, f"qualitative_tile_ranking_{a.set}.json"), "w"), indent=1)
    print(f"\n{a.set}: {len(rows)} eligible tiles; LACE best in {sum(r['margin'] > 0 for r in rows)}")
    for r in rows[:15]:
        print(r)
