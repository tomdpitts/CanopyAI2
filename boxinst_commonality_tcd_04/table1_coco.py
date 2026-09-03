"""Build the Table-1 rows we can score WITHOUT re-inference, under the frozen COCOeval
protocol (score_coco.py). Writes a fresh results file + markdown table; touches nothing that
the paper currently cites.

Rows here are only the ones whose predictions already exist at 512^2:
    LACE (ours), knobbed seeds 0/1/2  -- phase4/preds/knobbed_s{0,1,2}.json
    DetecTree2, seed 0                -- detectree2_baseline/preds_dt2_s0v2.json
Restor Mask R-CNN and SelvaBox -> SAM 3 still need inference; see PROTOCOL_439.md.

Canopy-neutral is reported from the COCO `iscrowd` arm (the frozen choice). The legacy
fixed-50% arm is carried alongside so the two can be compared per row rather than assumed
equal -- they agree exactly at AP50 and diverge slightly above it.

    .venv/bin/python -m boxinst_commonality_tcd_04.table1_coco --res 512
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from boxinst_commonality_tcd_04 import score_coco as SC

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = os.path.join(HERE, "modal_tcd_multiseed", "phase4", "preds")
ROWS = [
    ("LACE (ours) s0", os.path.join(P4, "knobbed_s0.json")),
    ("LACE (ours) s1", os.path.join(P4, "knobbed_s1.json")),
    ("LACE (ours) s2", os.path.join(P4, "knobbed_s2.json")),
    ("DetecTree2 s0", os.path.join(HERE, "detectree2_baseline",
                                   "preds_dt2_s0v2.json")),
]


def _cells(r):
    """The five Table-1 metrics, canopy-neutral taken from the crowd arm."""
    cn, fp = r["canopy_neutral_crowd"], r["canopy_fp"]
    return {"cn_ap50": cn["mask"]["AP50"], "cn_ap5095": cn["mask"]["AP50_95"],
            "fp_ap50": fp["mask"]["AP50"], "fp_ap5095": fp["mask"]["AP50_95"],
            "box_ap50": cn["box"]["AP50"],
            "legacy_ap50": r["canopy_neutral_legacy"]["mask"]["AP50"],
            "legacy_ap5095": r["canopy_neutral_legacy"]["mask"]["AP50_95"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--gt", default=os.path.join(HERE, "test_gt.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "results_table1_coco.json"))
    a = ap.parse_args()

    labels = [n for n, _ in ROWS]
    results = SC.score_many([p for _, p in ROWS], a.gt, res=a.res)
    cells = {n: _cells(r) for n, r in zip(labels, results)}

    lace = [cells[n] for n in labels if n.startswith("LACE")]
    band = {k: (float(np.mean([c[k] for c in lace])),
                float(np.std([c[k] for c in lace], ddof=1))) for k in lace[0]}

    hdr = (f"| Method | CN AP50 | CN AP50:95 | FP AP50 | FP AP50:95 | Box AP50 |")
    sep = "|---|---|---|---|---|---|"
    lines = [f"### Table 1 rows, frozen COCOeval protocol (res {a.res}, maxDets "
             f"{SC.MAX_DETS}, floor {SC.SCORE_FLOOR})", "",
             "Canopy-neutral = COCO `iscrowd`. Legacy fixed-50% arm shown below for comparison.",
             "", hdr, sep]
    for n in labels:
        c = cells[n]
        lines.append(f"| {n} | {c['cn_ap50']:.4f} | {c['cn_ap5095']:.4f} | "
                     f"{c['fp_ap50']:.4f} | {c['fp_ap5095']:.4f} | {c['box_ap50']:.4f} |")
    lines.append(f"| **LACE 3-seed mean ± sd** | "
                 f"**{band['cn_ap50'][0]:.4f} ± {band['cn_ap50'][1]:.4f}** | "
                 f"**{band['cn_ap5095'][0]:.4f} ± {band['cn_ap5095'][1]:.4f}** | "
                 f"**{band['fp_ap50'][0]:.4f} ± {band['fp_ap50'][1]:.4f}** | "
                 f"**{band['fp_ap5095'][0]:.4f} ± {band['fp_ap5095'][1]:.4f}** | "
                 f"**{band['box_ap50'][0]:.4f} ± {band['box_ap50'][1]:.4f}** |")

    lines += ["", "**crowd vs legacy canopy-ignore** (same predictions, same matcher, only "
              "the ignore rule differs):", "",
              "| Method | AP50 crowd | AP50 legacy | Δ | AP50:95 crowd | AP50:95 legacy | Δ |",
              "|---|---|---|---|---|---|---|"]
    for n in labels:
        c = cells[n]
        lines.append(f"| {n} | {c['cn_ap50']:.4f} | {c['legacy_ap50']:.4f} | "
                     f"{c['cn_ap50'] - c['legacy_ap50']:+.4f} | {c['cn_ap5095']:.4f} | "
                     f"{c['legacy_ap5095']:.4f} | {c['cn_ap5095'] - c['legacy_ap5095']:+.4f} |")

    table = "\n".join(lines)
    print("\n" + table, flush=True)
    slim = [{k: v for k, v in r.items() if k != "areas"} for r in results]
    json.dump({"protocol": results[0]["protocol"], "rows": dict(zip(labels, slim)),
               "lace_3seed_mean_sd_ddof1": band, "markdown": table},
              open(a.out, "w"), indent=2)
    print(f"\n-> {os.path.relpath(a.out)}", flush=True)


if __name__ == "__main__":
    main()
