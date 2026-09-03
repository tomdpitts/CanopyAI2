"""Consolidate every row scored under the frozen protocol into one table.

Reads the per-row score_coco outputs in this directory and emits table_439.json +
table_439.md. Add a row by adding a (label, file, arm-override) entry to ROWS.

    .venv/bin/python -m boxinst_commonality_tcd_04.results_439.build_table
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)

# (label, filename-or-None, note). None => pulled from results_table1_coco.json instead.
ROWS = [
    ("LACE (ours) s0",                      None, "detector heatmap score"),
    ("LACE (ours) s1",                      None, "detector heatmap score"),
    ("LACE (ours) s2",                      None, "detector heatmap score"),
    ("LACE (ours) s0 + posterior product",   "rerank_product_scored.json",
     "score x bimod x pfg_mean -- the masker's own posterior multiplied in. Zero fitted "
     "parameters, no supervision (confidence/README.md)"),
    ("LACE s1 + posterior product",          "rerank_product_s1.json",
     "posterior product, seed 1"),
    ("LACE s2 + posterior product",          "rerank_product_s2.json",
     "posterior product, seed 2"),
    ("LACE (ours) s0 + EM rerank (superseded)", "rerank_val_scored.json",
     "8-feature logistic reranker fit on 108 held-out val tiles (box-IoU target). Kept as "
     "the ablation's comparison row: equal at AP50, WORSE at AP75/AP50:95, 9 fitted numbers"),
    ("Restor MRCNN, released (rpn 512)",    "restor_coco512.json",
     "their shipped config; RPN proposal cap binds hard (98 dets/tile)"),
    ("Restor MRCNN, default (rpn 1000)",    "restor_rpn1000_tree.json",
     "detectron2 FPN default -- THE published operating point"),
    ("Restor MRCNN, 2x default (rpn 2000)", "restor_rpn2000.json",
     "above framework default; sensitivity only"),
    ("Restor, default, pooled classes",     "restor_rpn1000_pooled.json",
     "canopy predictions counted as instances -- the fair canopy=FP comparison"),
    ("Restor, released, pooled classes",    "restor_allcls.json",
     "as above at their shipped rpn 512"),
    ("DetecTree2 s0",                       "dt2_s0v2_coco512.json",
     "Retrained 2026-09-03 to detectree2's OWN AP50 early-stopping rule: it stopped itself at "
     "iter 14000 (two evals past the best) and selected iter 10000 = 6.0 epochs, well inside a "
     "16000 budget. Selection on all 896 val subtiles. INPUT.MIN_SIZE_TEST set to 1000 to match "
     "detectree2's own MIN_SIZE_TRAIN, instead of detectron2's inherited COCO default of 800, "
     "which silently downscaled our 1024 subtiles at test but not at train. Stitched at the "
     "protocol floor 0.05, not 0.1. The SUPERSEDED row (max_iter 4000 = 2.39 epochs and still "
     "improving; 256/896 val; scale 800; floor 0.1) scored 0.5344 / 0.1435 / 0.2240 / 0.5310 on "
     "152,690 dets -- all four defects understated it. See detectree2_baseline/RETRAIN_V2.md"),
    ("Box2Mask Swin-L (box-supervised)",    "box2mask_swinl_s0_coco512.json",
     "THE box-supervised row to quote: Box2Mask's STRONGEST released backbone (42.5 COCO mask "
     "AP). Same crops, grid, stitcher and scorer as DetecTree2; initialised from their COCO "
     "box-supervised checkpoint so the row is box-supervised end to end. Batch/LR (effective 16, "
     "lr 1e-4) are OUR choice matching the R-50 arm, because their published Swin-L config is "
     "self-inconsistent. Peaked at epoch 3.5 (val box AP50 0.596) with three later evals below "
     "the peak. +0.153 mask AP50 over R-50. Wins AP50 but loses at strict IoU (AP75 0.088 vs "
     "DetecTree2 0.144): finds crowns well, delineates them coarsely. num_queries=100 binds on "
     "1.16% of test subtiles (PLAN.md section 5d)"),
    ("Box2Mask R-50 (box-supervised)",      "box2mask_s0_coco512.json",
     "the only box-supervised competitor in the table -- LACE's own supervision class. Their "
     "published LSJ recipe, fine-tuned from their COCO box-supervised checkpoint (box "
     "annotations only, so no mask leakage) on the same 792/108 crops as DetecTree2, same "
     "1024@0.5 grid and stitcher. Trained under a 9-epoch budget with early stopping on val "
     "box AP50; it stopped itself at epoch 6 and selected epoch 4 (val AP50 0.479), with four "
     "later evaluations all below the peak while train loss kept falling -- so this is its "
     "validation optimum, not a truncation. Its boxes match DetecTree2's quality on the crowns "
     "it finds (mean box IoU 0.7365); the AP gap is recall (14,293 crowns matched vs "
     "DetecTree2's 19,603) plus the weakest masks at every matched box-IoU stratum "
     "(box_iou_strata.json). "
     "Capped at 100 instances per 1024 subtile by num_queries; that ceiling exceeds the GT crown "
     "count on all but 1.16% of test subtiles, so it is a footnote not a confound "
     "(box_supervised_baselines/PLAN.md sections 5, 5b and 10)"),
    ("SelvaBox -> SAM 3 (both FT)",         "selvabox_bench_600.json",
     "their OAM-TCD benchmark tiling (1024 @0.5, native GSD); maxDets 600 like every other "
     "row. READ THE CANOPY-NEUTRAL COLUMNS ONLY -- see note below"),
    ("SelvaBox -> SAM 3, maxDets 1600",     "selvabox_bench_1600.json",
     "their own budget (tile_level_eval_maxDets 400 per 1024^2 = 1600 per 2048^2-equiv). "
     "Sensitivity: shows what our 600 cap costs them"),
]

# SelvaBox is the one row whose canopy=FP figure is NOT comparable. CanopyRS trains with
# canopy annotations deleted and those pixels blacked out, so the model detects individual
# crowns inside OAM-TCD's canopy GROUP polygons -- correct behaviour under its own labelling
# convention, but unlabelled here. Measured on a smoke tile: 83% of its predictions centre
# inside canopy covering 25% of the tile, while it still recovered 6/7 labelled crowns at
# IoU>=0.5. The canopy=FP arm would therefore report the annotation mismatch, not the model.
# This is the mirror of the Restor tree-only correction (there, discarding its canopy class
# handed it free canopy-ignore in the arm designed to penalise exactly that).
FP_NOT_COMPARABLE = {"SelvaBox -> SAM 3 (both FT)", "SelvaBox -> SAM 3, maxDets 1600"}

# Mean mask IoU comes from box_iou_strata.py, on the COMMON SUBSET -- the crowns EVERY listed row
# matched. It is mask quality CONDITIONAL ON DETECTION (box IoU >= 0.5) and carries no recall
# information, unlike AP; a row can lead here and trail badly on AP. Only rows whose predictions
# we still hold can have a value, and the subset is defined by the row set, so these numbers move
# if a row is added or removed.
STRATA_KEY = {
    "LACE (ours) s0 + posterior product": "LACE s0 (product)",
    "LACE s1 + posterior product":        "LACE s1 (product)",
    "LACE s2 + posterior product":        "LACE s2 (product)",
    "Restor MRCNN, default (rpn 1000)":   "Restor MRCNN rpn1000",
    "DetecTree2 s0":                      "DetecTree2 s0",
    "SelvaBox -> SAM 3 (both FT)":        "SelvaBox -> SAM 3",
    "Box2Mask R-50 (box-supervised)":     "Box2Mask R-50",
    "Box2Mask Swin-L (box-supervised)":   "Box2Mask Swin-L",
}


def cells(r):
    cn, fp = r["canopy_neutral_crowd"], r["canopy_fp"]
    return dict(cn50=cn["mask"]["AP50"], cn75=cn["mask"]["AP75"],
                cn5095=cn["mask"]["AP50_95"], fp50=fp["mask"]["AP50"],
                fp5095=fp["mask"]["AP50_95"], box50=cn["box"]["AP50"],
                n_det=fp["mask"]["n_det"])


def main():
    base = json.load(open(os.path.join(PKG, "results_table1_coco.json")))
    out, notes = {}, {}
    for label, fn, note in ROWS:
        if fn is None:
            src = base["rows"].get(label)
            if src is None:
                continue
            out[label] = cells(src)
        else:
            fp = os.path.join(HERE, fn)
            if not os.path.exists(fp):
                print(f"  (missing {fn} -- skipping {label})")
                continue
            out[label] = cells(json.load(open(fp)))
        notes[label] = note

    lace = [out[f"LACE (ours) s{i}"] for i in range(3) if f"LACE (ours) s{i}" in out]
    if len(lace) > 1:
        out["LACE 3-seed mean"] = {k: float(np.mean([c[k] for c in lace])) for k in lace[0]}
        out["LACE 3-seed sd(ddof1)"] = {k: float(np.std([c[k] for c in lace], ddof=1))
                                        for k in lace[0]}
        notes["LACE 3-seed mean"] = "heatmap score alone"
    prod = [out[k] for k in ("LACE (ours) s0 + posterior product",
                             "LACE s1 + posterior product",
                             "LACE s2 + posterior product") if k in out]
    if len(prod) > 1:
        out["LACE + product 3-seed mean"] = {k: float(np.mean([c[k] for c in prod]))
                                             for k in prod[0]}
        out["LACE + product 3-seed sd(ddof1)"] = {k: float(np.std([c[k] for c in prod], ddof=1))
                                                  for k in prod[0]}
        notes["LACE + product 3-seed mean"] = ("THE HEADLINE. Positive on every seed and every "
                                               "metric; sd on CN AP50 drops 0.0054 -> 0.0012")

    strata_fp = os.path.join(HERE, "box_iou_strata.json")
    strata, n_common = {}, None
    if os.path.exists(strata_fp):
        sd = json.load(open(strata_fp))
        n_common = sd.get("n_common")
        strata = {lbl: sd["common_subset"][k]["mean_mask_iou"]
                  for lbl, k in STRATA_KEY.items() if k in sd.get("common_subset", {})}

    hdr = ["Method", "CN AP50", "CN AP75", "CN AP50:95", "FP AP50", "FP AP50:95",
           "Box AP50", "mask IoU", "dets"]
    lines = ["# OAM-TCD 439 — frozen COCOeval protocol", "",
             "pycocotools COCOeval · 439 whole 2048² tiles · masks at 512² · maxDets 600 · "
             "score floor 0.05 · iouThrs linspace(0.5,0.95,10) · canopy as `iscrowd`.",
             "See ../PROTOCOL_439.md.", "",
             (f"**mask IoU** = mean per-crown mask IoU on the {n_common:,} crowns EVERY row "
              "below matched, at matched detection quality (box IoU >= 0.5), from "
              "`box_iou_strata.json`. It is mask quality GIVEN a detection and contains no "
              "recall term, so it must NOT be read as a system metric, and it REORDERS the "
              "table: both mask-supervised models beat LACE's masker on it -- Restor 0.7655 "
              "and DetecTree2 0.7595 against LACE's 0.7430 +/- 0.0013 -- while DetecTree2 "
              "trails LACE by 0.066 on CN AP50 and Restor by 0.037. LACE's system lead "
              "therefore rests on detection and ranking, NOT on mask quality; the one place "
              "the masker leads outright is on GT prompt boxes (tab:gtbox), a different "
              "question. Box2Mask R-50 sits 0.052 below SelvaBox here against a 0.182 CN AP50 "
              "gap. `--` = predictions no longer held, so not computable."
              if n_common else ""), "",
             "| " + " | ".join(hdr) + " |",
             "|" + "---|" * len(hdr)]
    for label in out:
        c = out[label]
        if "sd" in label:
            lines.append(f"| _{label}_ | " + " | ".join(
                f"±{c[k]:.4f}" for k in ("cn50", "cn75", "cn5095", "fp50", "fp5095", "box50"))
                + " | | |")
        else:
            vals = []
            for k in ("cn50", "cn75", "cn5095", "fp50", "fp5095", "box50"):
                v = f"{c[k]:.4f}"
                if label in FP_NOT_COMPARABLE and k.startswith("fp"):
                    v = f"({v})*"        # bracketed: not comparable, see note
                vals.append(v)
            miou = strata.get(label)
            vals.append(f"{miou:.4f}" if miou is not None else "--")
            lines.append(f"| {label} | " + " | ".join(vals) + f" | {c['n_det']:,} |")
    if strata:
        seeds = [strata[k] for k in ("LACE (ours) s0 + posterior product",
                                     "LACE s1 + posterior product",
                                     "LACE s2 + posterior product") if k in strata]
        if len(seeds) > 1:
            import statistics as _st
            lines += ["", f"LACE mask IoU over 3 seeds: {_st.mean(seeds):.4f} "
                          f"± {_st.stdev(seeds):.4f} (ddof=1)."]
    lines += ["", "## Notes", ""] + [f"- **{k}** — {v}" for k, v in notes.items()]
    if any(l in out for l in FP_NOT_COMPARABLE):
        lines += ["",
                  "\\* **SelvaBox canopy=FP figures are bracketed as NOT comparable.** CanopyRS "
                  "trains with canopy annotations deleted and those pixels blacked out, so the "
                  "model detects individual crowns inside OAM-TCD's canopy *group* polygons — "
                  "correct under its own convention, unlabelled under this one. Measured: 83% of "
                  "its predictions centre inside canopy covering 25% of the tile, while it still "
                  "recovered 6/7 labelled crowns at IoU≥0.5. The FP arm would report the "
                  "annotation mismatch, not the model. Read the canopy-neutral columns."]

    md = "\n".join(lines)
    open(os.path.join(HERE, "table_439.md"), "w").write(md + "\n")
    json.dump({"rows": out, "notes": notes}, open(os.path.join(HERE, "table_439.json"), "w"),
              indent=2)
    print(md)


if __name__ == "__main__":
    main()
