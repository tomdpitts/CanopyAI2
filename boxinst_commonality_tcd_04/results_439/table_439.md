# OAM-TCD 439 — frozen COCOeval protocol

pycocotools COCOeval · 439 whole 2048² tiles · masks at 512² · maxDets 600 · score floor 0.05 · iouThrs linspace(0.5,0.95,10) · canopy as `iscrowd`.
See ../PROTOCOL_439.md.

**mask IoU** = mean per-crown mask IoU on the 10,020 crowns EVERY row below matched, at matched detection quality (box IoU >= 0.5), from `box_iou_strata.json`. It is mask quality GIVEN a detection and contains no recall term, so it must NOT be read as a system metric, and it REORDERS the table: both mask-supervised models beat LACE's masker on it -- Restor 0.7655 and DetecTree2 0.7595 against LACE's 0.7430 +/- 0.0013 -- while DetecTree2 trails LACE by 0.066 on CN AP50 and Restor by 0.037. LACE's system lead therefore rests on detection and ranking, NOT on mask quality; the one place the masker leads outright is on GT prompt boxes (tab:gtbox), a different question. Box2Mask R-50 sits 0.052 below SelvaBox here against a 0.182 CN AP50 gap. `--` = predictions no longer held, so not computable.

| Method | CN AP50 | CN AP75 | CN AP50:95 | FP AP50 | FP AP50:95 | Box AP50 | mask IoU | dets |
|---|---|---|---|---|---|---|---|---|
| LACE (ours) s0 | 0.6250 | 0.1433 | 0.2578 | 0.4867 | 0.2049 | 0.6211 | -- | 159,687 |
| LACE (ours) s1 | 0.6358 | 0.1376 | 0.2575 | 0.4838 | 0.2005 | 0.5835 | -- | 162,017 |
| LACE (ours) s2 | 0.6293 | 0.1417 | 0.2567 | 0.4846 | 0.2021 | 0.6230 | -- | 177,598 |
| LACE (ours) s0 + posterior product | 0.6615 | 0.1718 | 0.2821 | 0.5040 | 0.2212 | 0.6535 | 0.7445 | 159,687 |
| LACE s1 + posterior product | 0.6639 | 0.1566 | 0.2752 | 0.4923 | 0.2102 | 0.6119 | 0.7424 | 162,017 |
| LACE s2 + posterior product | 0.6622 | 0.1662 | 0.2798 | 0.4966 | 0.2165 | 0.6533 | 0.7420 | 177,598 |
| LACE (ours) s0 + EM rerank (superseded) | 0.6616 | 0.1648 | 0.2795 | 0.5073 | 0.2202 | 0.6525 | -- | 159,687 |
| Restor MRCNN, released (rpn 512) | 0.5706 | 0.1806 | 0.2575 | 0.5068 | 0.2314 | 0.5868 | -- | 42,987 |
| Restor MRCNN, default (rpn 1000) | 0.6255 | 0.1872 | 0.2766 | 0.5465 | 0.2454 | 0.6536 | 0.7655 | 56,438 |
| Restor MRCNN, 2x default (rpn 2000) | 0.6605 | 0.1923 | 0.2887 | 0.5699 | 0.2534 | 0.6934 | -- | 71,606 |
| Restor, default, pooled classes | 0.6137 | 0.1893 | 0.2738 | 0.5142 | 0.2339 | 0.6392 | -- | 79,201 |
| Restor, released, pooled classes | 0.5620 | 0.1811 | 0.2555 | 0.4789 | 0.2212 | 0.5825 | -- | 61,182 |
| DetecTree2 s0 | 0.5969 | 0.1748 | 0.2578 | 0.4863 | 0.2147 | 0.5913 | 0.7595 | 196,927 |
| Box2Mask Swin-L (box-supervised) | 0.5396 | 0.0884 | 0.1980 | 0.2801 | 0.1079 | 0.5821 | 0.6943 | 138,820 |
| Box2Mask R-50 (box-supervised) | 0.3865 | 0.0509 | 0.1299 | 0.1945 | 0.0699 | 0.4707 | 0.6436 | 117,027 |
| SelvaBox -> SAM 3 (both FT) | 0.5687 | 0.0746 | 0.2028 | (0.2563)* | (0.0919)* | 0.7297 | 0.6960 | 491,847 |
| SelvaBox -> SAM 3, maxDets 1600 | 0.5767 | 0.0754 | 0.2050 | (0.2586)* | (0.0925)* | 0.7480 | -- | 491,847 |
| LACE 3-seed mean | 0.6300 | 0.1409 | 0.2573 | 0.4850 | 0.2025 | 0.6092 | -- | 166,434.0 |
| _LACE 3-seed sd(ddof1)_ | ±0.0054 | ±0.0029 | ±0.0006 | ±0.0015 | ±0.0022 | ±0.0223 | | |
| LACE + product 3-seed mean | 0.6625 | 0.1649 | 0.2790 | 0.4976 | 0.2160 | 0.6396 | -- | 166,434.0 |
| _LACE + product 3-seed sd(ddof1)_ | ±0.0012 | ±0.0077 | ±0.0035 | ±0.0059 | ±0.0055 | ±0.0240 | | |

LACE mask IoU over 3 seeds: 0.7430 ± 0.0013 (ddof=1).

## Notes

- **LACE (ours) s0** — detector heatmap score
- **LACE (ours) s1** — detector heatmap score
- **LACE (ours) s2** — detector heatmap score
- **LACE (ours) s0 + posterior product** — score x bimod x pfg_mean -- the masker's own posterior multiplied in. Zero fitted parameters, no supervision (confidence/README.md)
- **LACE s1 + posterior product** — posterior product, seed 1
- **LACE s2 + posterior product** — posterior product, seed 2
- **LACE (ours) s0 + EM rerank (superseded)** — 8-feature logistic reranker fit on 108 held-out val tiles (box-IoU target). Kept as the ablation's comparison row: equal at AP50, WORSE at AP75/AP50:95, 9 fitted numbers
- **Restor MRCNN, released (rpn 512)** — their shipped config; RPN proposal cap binds hard (98 dets/tile)
- **Restor MRCNN, default (rpn 1000)** — detectron2 FPN default -- THE published operating point
- **Restor MRCNN, 2x default (rpn 2000)** — above framework default; sensitivity only
- **Restor, default, pooled classes** — canopy predictions counted as instances -- the fair canopy=FP comparison
- **Restor, released, pooled classes** — as above at their shipped rpn 512
- **DetecTree2 s0** — Retrained 2026-09-03 to detectree2's OWN AP50 early-stopping rule: it stopped itself at iter 14000 (two evals past the best) and selected iter 10000 = 6.0 epochs, well inside a 16000 budget. Selection on all 896 val subtiles. INPUT.MIN_SIZE_TEST set to 1000 to match detectree2's own MIN_SIZE_TRAIN, instead of detectron2's inherited COCO default of 800, which silently downscaled our 1024 subtiles at test but not at train. Stitched at the protocol floor 0.05, not 0.1. The SUPERSEDED row (max_iter 4000 = 2.39 epochs and still improving; 256/896 val; scale 800; floor 0.1) scored 0.5344 / 0.1435 / 0.2240 / 0.5310 on 152,690 dets -- all four defects understated it. See detectree2_baseline/RETRAIN_V2.md
- **Box2Mask Swin-L (box-supervised)** — THE box-supervised row to quote: Box2Mask's STRONGEST released backbone (42.5 COCO mask AP). Same crops, grid, stitcher and scorer as DetecTree2; initialised from their COCO box-supervised checkpoint so the row is box-supervised end to end. Batch/LR (effective 16, lr 1e-4) are OUR choice matching the R-50 arm, because their published Swin-L config is self-inconsistent. Peaked at epoch 3.5 (val box AP50 0.596) with three later evals below the peak. +0.153 mask AP50 over R-50. Wins AP50 but loses at strict IoU (AP75 0.088 vs DetecTree2 0.144): finds crowns well, delineates them coarsely. num_queries=100 binds on 1.16% of test subtiles (PLAN.md section 5d)
- **Box2Mask R-50 (box-supervised)** — the only box-supervised competitor in the table -- LACE's own supervision class. Their published LSJ recipe, fine-tuned from their COCO box-supervised checkpoint (box annotations only, so no mask leakage) on the same 792/108 crops as DetecTree2, same 1024@0.5 grid and stitcher. Trained under a 9-epoch budget with early stopping on val box AP50; it stopped itself at epoch 6 and selected epoch 4 (val AP50 0.479), with four later evaluations all below the peak while train loss kept falling -- so this is its validation optimum, not a truncation. Its boxes match DetecTree2's quality on the crowns it finds (mean box IoU 0.7365); the AP gap is recall (14,293 crowns matched vs DetecTree2's 19,603) plus the weakest masks at every matched box-IoU stratum (box_iou_strata.json). Capped at 100 instances per 1024 subtile by num_queries; that ceiling exceeds the GT crown count on all but 1.16% of test subtiles, so it is a footnote not a confound (box_supervised_baselines/PLAN.md sections 5, 5b and 10)
- **SelvaBox -> SAM 3 (both FT)** — their OAM-TCD benchmark tiling (1024 @0.5, native GSD); maxDets 600 like every other row. READ THE CANOPY-NEUTRAL COLUMNS ONLY -- see note below
- **SelvaBox -> SAM 3, maxDets 1600** — their own budget (tile_level_eval_maxDets 400 per 1024^2 = 1600 per 2048^2-equiv). Sensitivity: shows what our 600 cap costs them
- **LACE 3-seed mean** — heatmap score alone
- **LACE + product 3-seed mean** — THE HEADLINE. Positive on every seed and every metric; sd on CN AP50 drops 0.0054 -> 0.0012

\* **SelvaBox canopy=FP figures are bracketed as NOT comparable.** CanopyRS trains with canopy annotations deleted and those pixels blacked out, so the model detects individual crowns inside OAM-TCD's canopy *group* polygons — correct under its own convention, unlabelled under this one. Measured: 83% of its predictions centre inside canopy covering 25% of the tile, while it still recovered 6/7 labelled crowns at IoU≥0.5. The FP arm would report the annotation mismatch, not the model. Read the canopy-neutral columns.
