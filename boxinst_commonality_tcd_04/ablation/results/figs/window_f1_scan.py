"""Screen tiles for the qualitative figures by each model's F1 INSIDE the displayed crop window.

Uses make_qualitative's own pieces (thresholds, match_tile, auto_crop) so the screened F1 is the
panel F1, except that a mask counts as "in the window" when its RLE bounding box overlaps the
window (the figure tests for any pixel; the final choice is confirmed by rendering).

Eligible: canopy < CANOPY_MAX of the tile, >= MIN_WIN GT crowns in the window.
Writes window_f1_scan_<set>.json sorted by LACE window F1.

    .venv/bin/python boxinst_commonality_tcd_04/ablation/results/figs/window_f1_scan.py --set sparse
"""
import argparse, json, os, sys
import numpy as np
from pycocotools import mask as mu

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import make_qualitative as Q

CANOPY_MAX = float(os.environ.get("CANOPY_MAX", "0.005"))
MIN_WIN = 8


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True); a = ap.parse_args()
    cfg = Q.SETS[a.set]; S = cfg["crop"]
    gt = json.load(open(cfg["gt"]))
    preds = {lab: json.load(open(os.path.join(Q.NR, f)))["preds"] for lab, f in cfg["models"]}
    thr = json.load(open(os.path.join(HERE, f"qualitative_thresholds_{a.set}.json")))
    biome = {}
    if a.set == "sparse":
        man = json.load(open(os.path.join(Q.REPO, "data/tcd_sparse/slice_manifest.json")))["tiles"]
        biome = {t: (v["biome"], v["biome_name"]) for t, v in man.items()}
    rows = []
    for t, g in gt.items():
        gt_rles = Q.polys_to_rles(g["trees"])
        if len(gt_rles) < MIN_WIN:
            continue
        can = Q.union_rle(Q.polys_to_rles(g["canopy"]))
        cfrac = float(mu.area(can)) / Q.RES ** 2 if can is not None else 0.0
        if cfrac > CANOPY_MAX:
            continue
        gt_masks = [mu.decode(r).astype(bool) for r in gt_rles]
        x0, y0 = Q.auto_crop(gt_masks, S)
        inwin = lambda r: (lambda b: b[0] < x0 + S and b[0] + b[2] > x0 and b[1] < y0 + S and b[1] + b[3] > y0)(mu.toBbox(r))
        n_win = sum(inwin(r) for r in gt_rles)
        if n_win < MIN_WIN:
            continue
        row = dict(tile=t, n_win=int(n_win), canopy=round(cfrac, 4), crop=[int(x0), int(y0)])
        if biome:
            row["biome"] = biome[t]
        for lab, _ in cfg["models"]:
            p = preds[lab][t]
            if not p["masks_rle"]:
                row[lab] = 0.0; continue
            rles = [{"size": r["size"], "counts": r["counts"].encode() if isinstance(r["counts"], str) else r["counts"]}
                    for r in p["masks_rle"]]
            order, status, gidx = Q.match_tile(gt_rles, can, rles, p["scores"])
            s = np.asarray(p["scores"])[order]
            matched = set(); tp = fp = 0
            for i, st, gj, kp in zip(order, status, gidx, s >= thr[lab]["threshold"]):
                if not kp or st == "ign":
                    continue
                if st == "tp":
                    matched.add(gj); tp += inwin(rles[i])
                else:
                    fp += inwin(rles[i])
            fn = sum(1 for j, r in enumerate(gt_rles) if j not in matched and inwin(r))
            row[lab] = round(2 * tp / max(2 * tp + fp + fn, 1), 4)
        rows.append(row)
    rows.sort(key=lambda r: -r["LACE (ours)"])
    out = os.path.join(HERE, f"window_f1_scan_{a.set}.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"{a.set}: {len(rows)} eligible; LACE window F1 >= 0.8 on {sum(r['LACE (ours)'] >= 0.8 for r in rows)}")
    for r in rows[:25]:
        print(r)


if __name__ == "__main__":
    main()
