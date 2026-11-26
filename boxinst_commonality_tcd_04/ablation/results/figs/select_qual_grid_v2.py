"""Re-pick of the main-text qualitative comparison grid (2026-11-24, Tom: "re-pick"). Supersedes select_qual_grid.py's picks.

Why: v1 (select_qual_grid.py) chose each band's median-margin tile and then auto-cropped it; three of its four crops showed
LACE at in-crop F1 0.13-0.59, which breaks Tom's display rule of 2026-10-08 ("let's not show images with F1 < 0.8 for LACE
anywhere") and says little about mask shape because most crowns in the window are missed by every model.

Rule v2 (fixed before any new tile was viewed): candidates, canopy bands, per-tile LACE - Restor AP50 margin and the band
median are exactly v1's (read from v1's code, all 291 candidates). A tile is DISPLAY-ELIGIBLE if its auto-crop (v1's:
512 px window holding most GT centroids) contains >= MIN_CROP_CROWNS GT crowns and LACE's in-crop F1 there, at LACE's
whole-set best-F1 display threshold, is >= MIN_F1. In each band the chosen tile is the display-eligible tile whose margin
is closest to that band's median margin (median over ALL band candidates, eligible or not, so the margin shown stays the
typical one). The failure row is v1's, unchanged and exempt (5th-percentile margin tile), and is labelled as a failure.
The eligibility filter conditions on LACE's own in-crop F1; the caption must say so.

Usage: .venv/bin/python boxinst_commonality_tcd_04/ablation/results/figs/select_qual_grid_v2.py
Writes qualitative_grid_select_tcd439_v2.json next to this script; progress to stdout.
"""
import json, os, sys
import numpy as np
from pycocotools import mask as mu

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from select_qual_tile import ap50, rles, ROOT, MODELS  # noqa: E402
from select_qual_grid import MIN_CROWNS, FAIL_Q, EXCLUDE, BANDS, GT, TI  # noqa: E402
import make_qualitative as MQ  # noqa: E402

MIN_F1, MIN_CROP_CROWNS, CROP = 0.8, 10, 512
THR = json.load(open(os.path.join(HERE, "qualitative_thresholds_tcd439_grid.json")))["LACE (ours)"]["threshold"]


def in_window(rle, x0, y0):
    bx, by, bw, bh = mu.toBbox(rle)
    if bx >= x0 + CROP or bx + bw <= x0 or by >= y0 + CROP or by + bh <= y0:
        return False
    return bool(mu.decode(rle)[y0:y0 + CROP, x0:x0 + CROP].any())


def crop_f1(g_rles, canopy, p):
    """LACE in-crop F1 exactly as make_qualitative.py draws it (kept preds >= THR, canopy-ignored preds dropped)."""
    gt_masks = [mu.decode(x).astype(bool) for x in g_rles]
    x0, y0 = MQ.auto_crop(gt_masks, CROP)
    n_in = sum(bool(m[y0:y0 + CROP, x0:x0 + CROP].any()) for m in gt_masks)
    order, status, gidx = MQ.match_tile(g_rles, canopy, p["masks_rle"], p["scores"])
    s = np.asarray(p["scores"])[order] if len(order) else np.zeros(0)
    tp = fp = 0; matched = set()
    for i, st, gj, kp in zip(order, status, gidx, s >= THR):
        if not kp or st == "ign":
            continue
        if st == "tp":
            matched.add(gj)
        if in_window(p["masks_rle"][i], x0, y0):
            tp += st == "tp"; fp += st != "tp"
    fn = sum(bool(gt_masks[j][y0:y0 + CROP, x0:x0 + CROP].any()) for j in range(len(gt_masks)) if j not in matched)
    return dict(crop_xy=[int(x0), int(y0)], crop_crowns=int(n_in), crop_f1=round(2 * tp / max(2 * tp + fp + fn, 1), 4),
                tp=int(tp), fp=int(fp), fn=int(fn))


def main():
    gt = json.load(open(GT)); ti = {x["tid"]: x for x in json.load(open(TI))["tiles"] if x["split"] == "test"}
    assert set(ti) == set(gt), "tile_index / test_gt mismatch"
    preds = {k: json.load(open(v))["preds"] for k, v in MODELS.items()}
    rows = []
    for n, (t, g) in enumerate(gt.items()):
        tr = rles(g["trees"])
        if t in EXCLUDE or len(tr) < MIN_CROWNS:
            continue
        cn = rles(g["canopy"]); canopy = mu.merge(cn, intersect=False) if cn else None
        a = {k: ap50(tr, canopy, preds[k][t]["masks_rle"], preds[k][t]["scores"]) for k in MODELS}
        r = dict(tile=t, biome=ti[t]["biome_name"], canopy_frac=ti[t]["canopy_frac"], crowns=len(tr),
                 ap50_lace=round(a["LACE"], 4), ap50_restor=round(a["Restor"], 4), margin=round(a["LACE"] - a["Restor"], 4))
        r.update(crop_f1(tr, canopy, preds["LACE"][t]))
        r["eligible"] = bool(r["crop_f1"] >= MIN_F1 and r["crop_crowns"] >= MIN_CROP_CROWNS)
        rows.append(r)
        if len(rows) % 20 == 0:
            print(f"scanned {len(rows)} candidates ({n + 1}/{len(gt)} tiles)", flush=True)
    out = dict(rule=__doc__.strip().split("\n\n")[2], lace_display_threshold=THR, min_f1=MIN_F1,
               min_crop_crowns=MIN_CROP_CROWNS, n_candidates=len(rows), n_eligible=sum(r["eligible"] for r in rows), bands={})
    for name, lo, hi in BANDS:
        c = [r for r in rows if lo <= r["canopy_frac"] < hi]; m = np.array([r["margin"] for r in c]); med = float(np.median(m))
        ranked = sorted(c, key=lambda r: abs(r["margin"] - med))
        el = [r for r in ranked if r["eligible"]]
        out["bands"][name] = dict(edges=[lo, hi], n=len(c), n_eligible=len(el), median_margin=round(med, 4),
                                  frac_lace_ahead=round(float((m > 0).mean()), 4),
                                  frac_lace_ahead_eligible=round(float(np.mean([r["margin"] > 0 for r in el])), 4) if el else None,
                                  chosen=el[0] if el else None, rank_among_all=ranked.index(el[0]) + 1 if el else None,
                                  next3=el[1:4], v1_pick=ranked[0])
        print(f"{name:13s} n={len(c):3d} eligible={len(el):3d} median margin {med:+.4f} -> "
              f"{el[0] if el else None} (rank {ranked.index(el[0]) + 1 if el else None} of {len(c)})", flush=True)
    m = np.array([r["margin"] for r in rows]); q = float(np.quantile(m, FAIL_Q))
    f = sorted(rows, key=lambda r: abs(r["margin"] - q))
    out["failure"] = dict(quantile=FAIL_Q, margin_at_q=round(q, 4), chosen=f[0], note="unchanged from v1; exempt from eligibility")
    print(f"failure q={FAIL_Q} margin {q:+.4f} -> {f[0]}", flush=True)
    out["all_rows"] = rows
    json.dump(out, open(os.path.join(HERE, "qualitative_grid_select_tcd439_v2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
