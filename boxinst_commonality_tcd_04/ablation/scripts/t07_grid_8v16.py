"""T0.7 -- Is the 8px-vs-16px feature grid a clean ablation? (Answer: no.)

The 4-offset interlaced 8px grid is a headline contribution, and the apparent delta is the
largest in the repo: the 16px 5-seed band is mask mAP50 0.4917 while the 8px 5-seed vanilla
band is 0.615. Quoting that as a feature-grid ablation would be wrong.

This script diffs what actually changed between the two cohorts and reports the delta with
every confound enumerated, so the paper can either (a) not claim it as an ablation, or
(b) claim it only with the confounds stated. It deliberately does NOT emit a clean
"stride ablation" row, because no such experiment exists.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.ablation.scripts.t07_grid_8v16
"""
import argparse
import glob
import json
import os

import numpy as np

from ..lib import io

# 8px 4-phase vanilla band (alpha=1, kappa=1, beta=0.5, fixed grid), from phase4/README.md.
# Kept as literals with provenance because the per-seed JSONs for seeds 3,4 live on the
# Modal volume, not in the repo.
EIGHT_PX = {"per_seed": {0: 0.6203, 1: 0.6253, 2: 0.6241, 3: 0.6045, 4: 0.6011},
            "source": "modal_tcd_multiseed/phase4/README.md, 5-seed VANILLA band table",
            "config": {"grid_px": 8, "offsets": "4-phase interlaced", "detector": "phase4_L24",
                       "masker": "em_model_4p_fix.npz", "eval": "single-scale",
                       "compute": "Modal A100", "transformers": "4.57.1"}}

CONFOUNDS = [
    ("feature grid", "16 px, single offset", "8 px, 4-offset interlaced",
     "the intended variable"),
    ("detector recipe", "det_t8 (width 256, tower 3)", "phase4_L24",
     "different architecture/width, not just a different input grid"),
    ("masker", "em_model.npz (16px vault fit)", "em_model_4p_fix.npz (8px 4-phase fit)",
     "the EM was refit; masker and detector changed together"),
    ("eval protocol", "multiscale (native + 0.5x downscale arm)", "single-scale",
     "the 16px arm gets an extra detection pass the 8px arm does not"),
    ("compute / stack", "local MPS, transformers 5.12.1", "Modal A100, transformers 4.57.1",
     "local and Modal DINOv3 features agree only at cos 0.86 and must not be bit-compared"),
]


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out", default=os.path.join(io.RESULTS, "grid_8v16.json"))
    a = ap_.parse_args()

    sixteen, used = {}, []
    for f in sorted(glob.glob(os.path.join(io.MPS, "eval_s*.json"))):
        d = json.load(open(f))
        s = int(os.path.basename(f).split("_s")[1].split(".")[0])
        sixteen[s] = {"mask_mAP50": d["mask_mAP50"], "mask_mAP50_95": d["mask_mAP50_95"],
                      "box_mAP50": d["box_mAP50"], "det": d.get("det"), "em": d.get("em")}
        used.append(f)

    b16 = io.band([v["mask_mAP50"] for v in sixteen.values()])
    b8 = io.band(list(EIGHT_PX["per_seed"].values()))

    out = {
        "label": "8px 4-offset vs 16px feature grid -- CONFOUNDED, not an ablation",
        "verdict": (
            "These two cohorts differ in at least five ways at once (see `confounds`). The "
            "+0.123 mask AP50 difference cannot be attributed to the feature grid. The repo "
            "contains no single-variable stride experiment on OAM-TCD. Do not present this "
            "as a feature-grid ablation; either drop the claim or state every confound."),
        "nearest_clean_comparison": (
            "boxinst_commonality/README.md:107-119 varies 8px vs 16px under one protocol on "
            "the dryland development set, but reports shape proxies (fill/corner/centre), not "
            "AP, because dryland has no mask GT: 8px 0.70/0.27/0.96 vs 16px 0.72/0.37/0.91. "
            "Same limitation as the whitening ablation -- see results/proxy_calibration.json."),
        "confounds": [{"axis": a_, "arm_16px": b, "arm_8px": c, "why_it_matters": d}
                      for a_, b, c, d in CONFOUNDS],
        "bands_mask_mAP50": {
            "16px_5seed": {"mean_sd": b16, "per_seed": {k: v["mask_mAP50"]
                                                        for k, v in sixteen.items()},
                           "source": "mps_multiseed/eval_s{0..4}.json"},
            "8px_5seed_vanilla": {"mean_sd": b8, "per_seed": EIGHT_PX["per_seed"],
                                  "source": EIGHT_PX["source"]},
            "difference": round(b8[0] - b16[0], 4),
        },
        "config_8px": EIGHT_PX["config"],
        "config_16px_observed": {"det": sorted({v["det"] for v in sixteen.values()}),
                                 "em": sorted({v["em"] for v in sixteen.values()})},
    }
    json.dump(out, open(a.out, "w"), indent=2)
    io.record_inputs(used)

    print("=== 8px 4-offset vs 16px -- CONFOUNDED ===")
    print(f"  16px 5-seed mask AP50   {b16[0]:.4f} +- {b16[1]:.4f}   (mps_multiseed)")
    print(f"  8px  5-seed mask AP50   {b8[0]:.4f} +- {b8[1]:.4f}   (phase4 vanilla)")
    print(f"  difference              {b8[0]-b16[0]:+.4f}   <- NOT attributable to the grid")
    print(f"\n  {'axis':<18}{'16px arm':<36}{'8px arm'}")
    for ax, l, r, _ in CONFOUNDS:
        print(f"  {ax:<18}{l:<36}{r}")
    print(f"\n-> {os.path.relpath(a.out, io.REPO)}")


if __name__ == "__main__":
    main()
