"""T0.9 -- Do the dryland shape proxies (fill/corner/centre) track AP?

WHY THIS EXISTS
    Whitening is ablated only on the dryland development set, where no mask GT exists, so
    the ablation is reported in shape PROXIES (boxinst_commonality/README.md:107-119):
    removing whitening moves `corner` 0.37 -> 0.44 against a within-box-contrast failure
    mode at 0.61. Whether that is a meaningful degradation depends entirely on whether
    `corner` tracks segmentation accuracy at all. Nothing in the repo establishes that.

    This script establishes it where BOTH quantities are available: OAM-TCD, where several
    masker arms have saved predictions AND published AP.

METHOD
    Compute fill/corner/centre (formulas ported from boxinst/eval_masks.py:61-71) over the
    same 439 tiles for each arm, and relate them to that arm's published mask AP50.

CAVEATS, both structural and to be quoted with any use of the result:
    - The dryland proxies were GT-BOX prompted; these arms are PREDICTED-box. The
      calibration is directional, not a transfer function.
    - The AP range spanned here is narrow (0.579 - 0.630) and the arms differ in the
      masker only. This can show that corner MOVES WITH AP across masker changes; it
      cannot show corner predicts AP across architectures.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.ablation.scripts.t09_proxy_calibration
"""
import argparse
import json
import os

import numpy as np

from boxinst_commonality_tcd_04 import evaluate as E
from ..lib import io

# (label, preds file, published mask AP50, published mAP50-95, provenance of the AP)
ARMS = [
    ("beta=0 (fill)", os.path.join(io.PHASE4, "preds_b0_fix_thr025_439.json"),
     0.5790, 0.2003, "results_b0_fix_thr025_439.json"),
    ("beta=0.5 vanilla", os.path.join(io.PHASE4, "preds_b05_fix_thr025_439.json"),
     0.6203, 0.2436, "results_b05_fix_thr025_439.json"),
    ("beta=0.5 knobbed s0", io.knobbed(0), 0.6250, 0.2574, "preds/ref_knobbed_s0.json"),
    ("beta=0.5 knobbed s1", io.knobbed(1), 0.6358, 0.2567, "preds/ref_knobbed_s1.json"),
    ("beta=0.5 knobbed s2", io.knobbed(2), 0.6294, 0.2564, "preds/ref_knobbed_s2.json"),
]


def proxies(preds_path, gt, min_score=0.0, progress=200):
    """fill / corner / centre per predicted instance, averaged.

    Ported from boxinst/eval_masks.py:61-71. Normalised (u,v) inside each predicted box:
      fill    mask area inside box / box area        (a box-filler -> ~1.0)
      corner  mean mask occupancy in the four corner zones (outer 25% of u AND v)
      centre  mean mask occupancy in the central 50% x 50%
    A crown is a blob: high centre, low corner, moderate fill.
    """
    acc = {"fill": [], "corner": [], "centre": []}
    n = 0
    for t in io.per_tile(preds_path, gt, progress=progress):
        for i in range(len(t["scores"])):
            if t["scores"][i] < min_score:
                continue
            x0, y0, x1, y1 = t["boxes"][i] / (2048.0 / E.RES)
            xi0, yi0 = int(np.floor(x0)), int(np.floor(y0))
            xi1, yi1 = int(np.ceil(x1)), int(np.ceil(y1))
            xi0, yi0 = max(0, xi0), max(0, yi0)
            xi1, yi1 = min(E.RES, xi1), min(E.RES, yi1)
            if xi1 - xi0 < 2 or yi1 - yi0 < 2:
                continue
            sub = t["pm"][i][yi0:yi1, xi0:xi1].astype(np.float32)
            h, w = sub.shape
            u = (np.arange(w) + 0.5) / w
            v = (np.arange(h) + 0.5) / h
            U, V = np.meshgrid(u, v)
            corner = ((U < .25) | (U > .75)) & ((V < .25) | (V > .75))
            centre = (U > .25) & (U < .75) & (V > .25) & (V < .75)
            acc["fill"].append(float(sub.mean()))
            acc["corner"].append(float(sub[corner].mean()) if corner.any() else 0.0)
            acc["centre"].append(float(sub[centre].mean()) if centre.any() else 0.0)
            n += 1
    return {k: round(float(np.mean(v)), 4) for k, v in acc.items()}, n


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out", default=os.path.join(io.RESULTS, "proxy_calibration.json"))
    a = ap_.parse_args()
    gt = io.load_gt()

    rows, used = [], [io.GT]
    for lab, path, ap50, ap5095, prov in ARMS:
        if not os.path.exists(path):
            print(f"  SKIP {lab}: {path} missing", flush=True)
            continue
        print(f"[{lab}]", flush=True)
        px, n = proxies(path, gt)
        rows.append({"arm": lab, "preds": os.path.relpath(path, io.REPO),
                     "n_instances": n, "mask_mAP50": ap50, "mask_mAP50_95": ap5095,
                     "ap_provenance": prov, **px})
        used.append(path)

    corr = {}
    if len(rows) >= 3:
        for m in ("fill", "corner", "centre"):
            corr[m] = {
                "pearson_vs_AP50": round(float(np.corrcoef([r[m] for r in rows],
                                                           [r["mask_mAP50"] for r in rows])[0, 1]), 3),
                "spearman_vs_AP50": round(spearman([r[m] for r in rows],
                                                   [r["mask_mAP50"] for r in rows]), 3),
                "spearman_vs_AP50_95": round(spearman([r[m] for r in rows],
                                                      [r["mask_mAP50_95"] for r in rows]), 3),
            }

    ap_vals = [r["mask_mAP50"] for r in rows]
    out = {"label": "Shape proxies vs published AP, 439 OAM-TCD, masker arms on one detector",
           "purpose": ("establish whether corner/fill/centre track segmentation accuracy, so "
                       "the dryland whitening ablation (reported only in these proxies) can "
                       "be read directionally"),
           "caveats": [
               "dryland proxies were GT-box prompted; these arms are predicted-box",
               f"AP range spanned is narrow ({min(ap_vals):.3f}-{max(ap_vals):.3f})",
               "arms differ in the MASKER only (same detector family), so this speaks to "
               "masker changes, not to cross-architecture comparison",
               "3 of the 5 arms are seeds of one configuration, so they cluster in AP",
           ],
           "dryland_reference": {"whiten": {"fill": 0.72, "corner": 0.37, "centre": 0.91},
                                 "no_whiten": {"fill": 0.75, "corner": 0.44, "centre": 0.91},
                                 "no_within_box_contrast": {"fill": 0.89, "corner": 0.61,
                                                            "centre": 0.86},
                                 "source": "boxinst_commonality/README.md:107-119 (seed 0, 16px, "
                                           "359 GT-box crowns, 33 tiles)"},
           "correlations": corr, "arms": rows}
    json.dump(out, open(a.out, "w"), indent=2)
    io.record_inputs(used)

    print(f"\n=== shape proxies vs AP ({len(rows)} arms, 439 tiles) ===")
    print(f"{'arm':<24}{'n_inst':>9}{'AP50':>8}{'fill':>8}{'corner':>8}{'centre':>8}")
    for r in rows:
        print(f"{r['arm']:<24}{r['n_instances']:>9}{r['mask_mAP50']:>8.4f}"
              f"{r['fill']:>8.4f}{r['corner']:>8.4f}{r['centre']:>8.4f}")
    if corr:
        print(f"\n{'proxy':<10}{'pearson AP50':>14}{'spearman AP50':>15}{'spearman 50-95':>16}")
        for m, c in corr.items():
            print(f"{m:<10}{c['pearson_vs_AP50']:>14.3f}{c['spearman_vs_AP50']:>15.3f}"
                  f"{c['spearman_vs_AP50_95']:>16.3f}")
    print(f"\n-> {os.path.relpath(a.out, io.REPO)}")


if __name__ == "__main__":
    main()
