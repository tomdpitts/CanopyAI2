"""T0.1 -- Boundary-F1 of the DEPLOYED masker on the 439, vs DetecTree2 and a filled box.

WHY THIS SUPERSEDES phase4_research/figures/boundary_f1.json
    That file's "ours" arm is `crown_mask` -- the RGB-guided-filter research variant
    phase4/README.md records as FAILED. Its BF@3 of 0.286 is the boundary score of an
    abandoned method, not of LACE, and it is scored on 13 tiles of size-filtered crowns
    with GT boxes at 2048 px. It must not be quoted as LACE's boundary quality.
    This script measures the shipped configuration: all 439 tiles, PREDICTED boxes, the
    512 scoring raster, the same knobbed preds that produce the published 0.630 band.

WHAT IS MEASURED
    BF on MATCHED true positives only (greedy, score-ordered, IoU>=0.5 -- the identical
    rule as evaluate._greedy_ap). Conditioning on a successful detection is deliberate: it
    isolates mask-boundary quality from detection recall, which mAP50 conflates. A method
    cannot win here by detecting less.

    Three arms on the IDENTICAL matched pairs, so the comparison is paired:
      ours     the EM self-mask
      boxfill  the predicted box, filled -- the trivial baseline the method must beat
      dt2      DetecTree2, mask-supervised, its own detections and its own matched set

    Tolerances are on the 512 raster; the 2048-tile equivalent is 4x and is reported too.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.ablation.scripts.t01_boundary
"""
import argparse
import json
import os

import numpy as np

from boxinst_commonality_tcd_04 import evaluate as E
from ..lib import boundary as B
from ..lib import gate, io

TOLS = (1, 2, 3, 5)


def run_arm(preds_path, gt, tols, label, progress=100):
    """One pass: AP (for the gate) + paired BF for ours/boxfill on matched TPs."""
    acc = {"ours": {t: [] for t in tols}, "boxfill": {t: [] for t in tols}}
    mI, S, IGN, n_gt, n_tp = [], [], [], 0, 0
    for tl in io.per_tile(preds_path, gt, progress=progress):
        iou = E.mask_iou(tl["pm"], tl["gm"])
        mI.append(iou); S.append(tl["scores"]); IGN.append(tl["ignore"])
        n_gt += len(tl["gm"])
        for pi, gj in B.match_tps(iou, tl["scores"], tl["ignore"], 0.5):
            gm = tl["gm"][gj]
            for t, v in B.bf_multi(tl["pm"][pi], gm, tols).items():
                acc["ours"][t].append(v)
            for t, v in B.bf_multi(B.box_fill_mask(tl["boxes"][pi]), gm, tols).items():
                acc["boxfill"][t].append(v)
            n_tp += 1
    ap = gate.score_ap(mI, S, IGN, n_gt)
    out = {"label": label, "preds": os.path.relpath(preds_path, io.REPO),
           "n_matched_tp": n_tp, "n_gt": n_gt, "ap": ap,
           "bf": {m: {f"tol{t}": round(float(np.mean(v)), 4) for t, v in d.items()}
                  for m, d in acc.items()}}
    return out, acc


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--seeds", default="0,1,2")
    ap_.add_argument("--tols", default=",".join(map(str, TOLS)))
    ap_.add_argument("--out", default=os.path.join(io.RESULTS, "boundary_band.json"))
    a = ap_.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    tols = tuple(int(t) for t in a.tols.split(","))

    gt = io.load_gt()
    arms, used = {}, [io.GT]

    for s in seeds:
        print(f"[ours s{s}]", flush=True)
        r, _ = run_arm(io.knobbed(s), gt, tols, f"ours knobbed s{s}")
        gate.check(r["ap"], io.ref_knobbed(s), label=f"ours s{s}")
        arms[f"ours_s{s}"] = r
        used += [io.knobbed(s), io.ref_knobbed(s)]

    dt2p = os.path.join(io.DT2, "preds_dt2_s0v2.json")
    dt2r = os.path.join(io.DT2, "results_dt2_s0v2.json")
    if os.path.exists(dt2p):
        print("[detectree2 s0]", flush=True)
        r, _ = run_arm(dt2p, gt, tols, "DetecTree2 s0 (mask-supervised)")
        gate.check(r["ap"], dt2r, label="dt2 s0")
        arms["dt2_s0"] = r
        used += [dt2p, dt2r]

    # band over the ours seeds; boxfill banded too since its matched set moves with the detector
    okeys = [k for k in arms if k.startswith("ours_s")]
    band = {m: {f"tol{t}": io.band([arms[k]["bf"][m][f"tol{t}"] for k in okeys])
                for t in tols} for m in ("ours", "boxfill")}

    out = {
        "label": "Boundary-F1 on matched TPs, 439 OAM-TCD tiles, deployed knobbed masker",
        "supersedes": {
            "file": "phase4_research/figures/boundary_f1.json",
            "why": ("its 'ours' arm is crown_mask (the RGB-guided-filter variant recorded "
                    "as FAILED in phase4/README.md), on 13 tiles of size-filtered crowns "
                    "with GT boxes at 2048px. Not the deployed EM masker; do not quote its "
                    "0.286 as LACE's boundary quality."),
        },
        "protocol": {
            "matching": "greedy, score-ordered, mask IoU>=0.5, identical to evaluate._greedy_ap",
            "conditioning": ("matched TPs only -- isolates boundary quality from detection "
                             "recall; a method cannot win by detecting less"),
            "raster": f"{E.RES}px (tile is 2048px, so tolerance t here = {int(2048/E.RES)}t tile px)",
            "tolerances_512px": list(tols),
            "tolerances_2048px_equiv": [int(2048 / E.RES) * t for t in tols],
            "paired": "ours and boxfill share the identical matched pairs; dt2 has its own",
            "discriminative_tolerance": 'BF saturates above 1px on the 512 raster (>=0.90 at 2px, >=0.99 at 5px), because a matched TP at IoU>=0.5 is already well aligned at that granularity. BF@1px (= 4 tile px) is the informative column; the wider tolerances are reported for completeness, not for ranking. A finer comparison would need GT rasterised at the full 2048.',
            "dt2_population_caveat": 'DetecTree2 is scored on ITS OWN matched set (19,217 TPs vs our ~20,750), because its detector differs. ours-vs-boxfill is a paired comparison on identical pairs and is exact; ours-vs-dt2 compares means over different instance populations and should be read as approximate.',
        },
        "band_mean_std": band,
        "per_arm": arms,
    }
    json.dump(out, open(a.out, "w"), indent=2)
    io.record_inputs(used)

    print(f"\n=== Boundary-F1, matched TPs @IoU0.5, {E.RES}px raster ===")
    hdr = "".join(f"{'BF@'+str(t)+'px':>10}" for t in tols)
    print(f"{'arm':<26}{'n_TP':>8}{hdr}")
    for k, r in arms.items():
        for m in (("ours", "boxfill") if k.startswith("ours") else ("ours",)):
            nm = f"{k} [{m}]" if k.startswith("ours") else k
            print(f"{nm:<26}{r['n_matched_tp']:>8}"
                  + "".join(f"{r['bf'][m][f'tol{t}']:>10.4f}" for t in tols))
    print(f"\n{'ours band (mean+-sd)':<26}{'':>8}"
          + "".join(f"{band['ours'][f'tol{t}'][0]:>10.4f}" for t in tols))
    print(f"{'boxfill band':<26}{'':>8}"
          + "".join(f"{band['boxfill'][f'tol{t}'][0]:>10.4f}" for t in tols))
    print(f"\n-> {os.path.relpath(a.out, io.REPO)}")


if __name__ == "__main__":
    main()
