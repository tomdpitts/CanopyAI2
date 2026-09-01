# OAM-TCD 439 — frozen COCOeval protocol

pycocotools COCOeval · 439 whole 2048² tiles · masks at 512² · maxDets 600 · score floor 0.05 · iouThrs linspace(0.5,0.95,10) · canopy as `iscrowd`.
See ../PROTOCOL_439.md.

| Method | CN AP50 | CN AP75 | CN AP50:95 | FP AP50 | FP AP50:95 | Box AP50 | dets |
|---|---|---|---|---|---|---|---|
| LACE (ours) s0 | 0.6250 | 0.1433 | 0.2578 | 0.4867 | 0.2049 | 0.6211 | 159,687 |
| LACE (ours) s1 | 0.6358 | 0.1376 | 0.2575 | 0.4838 | 0.2005 | 0.5835 | 162,017 |
| LACE (ours) s2 | 0.6293 | 0.1417 | 0.2567 | 0.4846 | 0.2021 | 0.6230 | 177,598 |
| LACE (ours) s0 + posterior product | 0.6615 | 0.1718 | 0.2821 | 0.5040 | 0.2212 | 0.6535 | 159,687 |
| LACE s1 + posterior product | 0.6639 | 0.1566 | 0.2752 | 0.4923 | 0.2102 | 0.6119 | 162,017 |
| LACE s2 + posterior product | 0.6622 | 0.1662 | 0.2798 | 0.4966 | 0.2165 | 0.6533 | 177,598 |
| LACE (ours) s0 + EM rerank (superseded) | 0.6616 | 0.1648 | 0.2795 | 0.5073 | 0.2202 | 0.6525 | 159,687 |
| Restor MRCNN, released (rpn 512) | 0.5706 | 0.1806 | 0.2575 | 0.5068 | 0.2314 | 0.5868 | 42,987 |
| Restor MRCNN, default (rpn 1000) | 0.6255 | 0.1872 | 0.2766 | 0.5465 | 0.2454 | 0.6536 | 56,438 |
| Restor MRCNN, 2x default (rpn 2000) | 0.6605 | 0.1923 | 0.2887 | 0.5699 | 0.2534 | 0.6934 | 71,606 |
| Restor, default, pooled classes | 0.6137 | 0.1893 | 0.2738 | 0.5142 | 0.2339 | 0.6392 | 79,201 |
| Restor, released, pooled classes | 0.5620 | 0.1811 | 0.2555 | 0.4789 | 0.2212 | 0.5825 | 61,182 |
| DetecTree2 s0 | 0.5344 | 0.1435 | 0.2240 | 0.4214 | 0.1810 | 0.5310 | 152,690 |
| SelvaBox -> SAM 3 (both FT) | 0.5687 | 0.0746 | 0.2028 | (0.2563)* | (0.0919)* | 0.7297 | 491,847 |
| SelvaBox -> SAM 3, maxDets 1600 | 0.5767 | 0.0754 | 0.2050 | (0.2586)* | (0.0925)* | 0.7480 | 491,847 |
| LACE 3-seed mean | 0.6300 | 0.1409 | 0.2573 | 0.4850 | 0.2025 | 0.6092 | 166,434.0 |
| _LACE 3-seed sd(ddof1)_ | ±0.0054 | ±0.0029 | ±0.0006 | ±0.0015 | ±0.0022 | ±0.0223 | |
| LACE + product 3-seed mean | 0.6625 | 0.1649 | 0.2790 | 0.4976 | 0.2160 | 0.6396 | 166,434.0 |
| _LACE + product 3-seed sd(ddof1)_ | ±0.0012 | ±0.0077 | ±0.0035 | ±0.0059 | ±0.0055 | ±0.0240 | |

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
- **DetecTree2 s0** — floored at 0.1 by its stitcher
- **SelvaBox -> SAM 3 (both FT)** — their OAM-TCD benchmark tiling (1024 @0.5, native GSD); maxDets 600 like every other row. READ THE CANOPY-NEUTRAL COLUMNS ONLY -- see note below
- **SelvaBox -> SAM 3, maxDets 1600** — their own budget (tile_level_eval_maxDets 400 per 1024^2 = 1600 per 2048^2-equiv). Sensitivity: shows what our 600 cap costs them
- **LACE 3-seed mean** — heatmap score alone
- **LACE + product 3-seed mean** — THE HEADLINE. Positive on every seed and every metric; sd on CN AP50 drops 0.0054 -> 0.0012

\* **SelvaBox canopy=FP figures are bracketed as NOT comparable.** CanopyRS trains with canopy annotations deleted and those pixels blacked out, so the model detects individual crowns inside OAM-TCD's canopy *group* polygons — correct under its own convention, unlabelled under this one. Measured: 83% of its predictions centre inside canopy covering 25% of the tile, while it still recovered 6/7 labelled crowns at IoU≥0.5. The FP arm would report the annotation mismatch, not the model. Read the canopy-neutral columns.
