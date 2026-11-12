"""Rule-based tile choice for the D1 qualitative strip (Figure 1): the representative dryland tile.

Candidates: OAM-TCD 439 holdout tiles (unseen by LACE, Restor and BoxTeacher; the sparse-236 slice is NOT used because 220
of its tiles are in the OAM-TCD train split Restor was trained on) whose image is >= MIN_SOIL bare soil, with >= MIN_CROWNS
GT crowns and <= MAX_CANOPY canopy-class area (so scorer-ignored regions are immaterial). Bare-soil fraction is measured
from the RGB image alone (512 px resize, nodata excluded): a pixel is soil if warm-toned (R - B >= 15), not green
(excess green 2G - R - B < 10) and not dark (mean > 80); this rejects vegetation, shadow and grey roads/roofs. Tile median
is 0.05. For each candidate, per-tile mask AP50 (COCO 101-point interpolation, greedy score-ordered matching at
IoU >= 0.5, predictions > 50 % in canopy ignored) is computed for LACE and Restor. The chosen tile is the candidate whose
LACE - Restor AP50 margin is closest to the candidate median.

History (2026-11-12): v1 biomes 7-10, 12, 13 -> tile 291 (urban temperate grassland); v2 biome 7 only -> tile 262 (did not
look like savannah); v3 labelled cover <= 10 % -> tile 158 (9 crowns in a crop field), most low-cover tiles urban. v4
(this) defines open ground from the image. MIN_SOIL = 0.4 was set after viewing tiles; the sensitivity block in the JSON
shows the pick at 0.3 and 0.5. The 512 px crop of tile 122 used in the figure, (128, 640), was chosen by eye within the tile;
it is less favourable to LACE than the auto-crop (crop F1 margin vs Restor +0.06 vs +0.13 at (0, 832)).

Usage: .venv/bin/python boxinst_commonality_tcd_04/ablation/results/figs/select_qual_tile.py
Writes qualitative_tile_select_tcd439_d1_soil.json next to this script.
"""
import json, os
import numpy as np
from PIL import Image
from pycocotools import mask as mu

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RES = 2048
MIN_SOIL, MIN_CROWNS, MAX_CANOPY = 0.4, 10, 0.05
SENSITIVITY = (0.3, 0.4, 0.5)
IMGDIR = os.path.join(ROOT, "..", "data", "tcd", "test")
GT = os.path.join(ROOT, "test_gt.json")
MODELS = {
    "LACE": os.path.join(ROOT, "dino_det/results/preds_dino_swinl_s0_test_R3_ef_r2048.json"),
    "Restor": os.path.join(ROOT, "native_raster/preds_restor_rpn1000_r2048_treeonly.json"),
}


def rles(polys):
    return [mu.merge(mu.frPyObjects([p], RES, RES)) for p in polys if len(p) >= 6]


def soil_fraction(path):
    a = np.asarray(Image.open(path).convert("RGB").resize((512, 512)), float)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    valid = a.sum(-1) > 30
    soil = valid & (r - b >= 15) & (2 * g - r - b < 10) & (a.mean(-1) > 80)
    return float(soil.sum() / max(valid.sum(), 1))


def ap50(gt_r, canopy, pr, sc):
    if not gt_r:
        return float("nan")
    if not pr:
        return 0.0
    order = np.argsort(-np.asarray(sc), kind="stable")
    iou = mu.iou([pr[i] for i in order], gt_r, [0] * len(gt_r))
    cf = mu.iou([pr[i] for i in order], [canopy], [1])[:, 0] if canopy is not None else np.zeros(len(order))
    taken, tp = set(), []
    for r in range(len(order)):
        best, bj = 0.5, -1
        for j in range(len(gt_r)):
            if j not in taken and iou[r, j] >= best:
                best, bj = iou[r, j], j
        if bj >= 0:
            taken.add(bj); tp.append(1)
        elif cf[r] > 0.5:
            continue
        else:
            tp.append(0)
    tp = np.asarray(tp, float)
    ctp = np.cumsum(tp); cfp = np.cumsum(1 - tp)
    rec = ctp / len(gt_r); prec = ctp / np.maximum(ctp + cfp, 1e-9)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, rs, side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))


def pick(rows, min_soil):
    c = [r for r in rows if r["soil"] >= min_soil]
    m = np.array([r["margin"] for r in c])
    med = float(np.median(m))
    c = sorted(c, key=lambda r: abs(r["margin"] - med))
    return dict(min_soil=min_soil, n=len(c), median_margin=round(med, 4), frac_lace_ahead=round(float((m > 0).mean()), 4),
                chosen=c[0]["tile"], ranked=c)


def main():
    gt = json.load(open(GT))
    preds = {k: json.load(open(v))["preds"] for k, v in MODELS.items()}
    rows = []
    for t, g in gt.items():
        sf = soil_fraction(os.path.join(IMGDIR, f"{t}.tif"))
        if sf < min(SENSITIVITY):
            continue
        tr, cn = rles(g["trees"]), rles(g["canopy"])
        canopy = mu.merge(cn, intersect=False) if cn else None
        can = mu.area(canopy) / RES ** 2 if canopy is not None else 0.0
        if can > MAX_CANOPY or len(tr) < MIN_CROWNS:
            continue
        biome = json.load(open(os.path.join(IMGDIR, f"{t}_meta.json"))).get("biome")
        a = {k: ap50(tr, canopy, preds[k][t]["masks_rle"], preds[k][t]["scores"]) for k in MODELS}
        rows.append(dict(tile=t, biome=biome, soil=round(sf, 4), crowns=len(tr), canopy=round(float(can), 5),
                         ap50_lace=round(a["LACE"], 4), ap50_restor=round(a["Restor"], 4),
                         margin=round(a["LACE"] - a["Restor"], 4)))
    main_pick = pick(rows, MIN_SOIL)
    sens = {str(s): {k: v for k, v in pick(rows, s).items() if k != "ranked"} for s in SENSITIVITY}
    out = dict(rule=__doc__.strip().split("\n\n")[1], **main_pick, sensitivity=sens, models=MODELS)
    json.dump(out, open(os.path.join(HERE, "qualitative_tile_select_tcd439_d1_soil.json"), "w"), indent=1)
    print(f"soil>={MIN_SOIL}: {main_pick['n']} candidates; median margin {main_pick['median_margin']:+.4f}; "
          f"LACE ahead on {main_pick['frac_lace_ahead']:.0%}; chosen {main_pick['chosen']}")
    for s, v in sens.items():
        print("  sensitivity", s, v)
    for r in main_pick["ranked"]:
        print(" ", r)


if __name__ == "__main__":
    main()
