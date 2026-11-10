# tab:strata-maskiou (CVPR Table S9) — D1 re-base, 2026-11-04

Script: `native_raster/ablation_r2048/box_iou_strata_d1.py` (wraps `box_supervised_baselines/box_iou_strata.py`, unmodified). Log: `native_raster/logs/strata_d1_2026-11-04.log`. Local CPU, ~15 min.

Supersedes `box_iou_strata_lace4px_simple_r2048.json` (10,833 common crowns) for the paper table.

Changes vs the previous table:
- SelvaBox -> SAM 3 dropped (removed from paper results, Tom 2026-11-04).
- BoxTeacher Swin-B added (Tier B B2 test preds).
- LACE = D1 + edge fix + R3 (`dino_det/results/preds_dino_swinl_s0_test_R3_ef_r2048.json`), single seed; replaces CenterNet-variant s0-s2.
- Restor = tree-only file (the one Table 1 scores); old strata used the both-class file.
- Every row filtered as Table 1: score >= 0.05, top 600 per tile (old strata uncapped).

Common subset: 11,755 crowns (matched by all 6 rows).

| Row | matched | box IoU | mask IoU | [.5,.6) | [.6,.7) | [.7,.8) | [.8,.9) | [.9,1] |
|---|---|---|---|---|---|---|---|---|
| Restor | 18532 | 0.7925 | 0.7852 | 0.6098 | 0.6908 | 0.7628 | 0.8247 | 0.8727 |
| Detectree2 | 20483 | 0.7817 | 0.7745 | 0.6000 | 0.6796 | 0.7541 | 0.8258 | 0.8812 |
| Box2Mask Swin-L | 17604 | 0.7817 | 0.7153 | 0.5517 | 0.6360 | 0.6994 | 0.7566 | 0.8156 |
| Box2Mask R-50 | 14780 | 0.7699 | 0.6621 | 0.5154 | 0.5840 | 0.6444 | 0.7139 | 0.7895 |
| BoxTeacher | 21130 | 0.7983 | 0.7508 | 0.5681 | 0.6531 | 0.7257 | 0.7862 | 0.8399 |
| LACE D1+ef+R3 | 20727 | 0.8074 | 0.7648 | 0.5913 | 0.6674 | 0.7354 | 0.7928 | 0.8400 |

("matched" = all matches per row; other columns = common subset.)

Finding change: the old table had LACE leading the lowest stratum (0.620 vs Restor 0.611). With D1 it no longer does; the mask-supervised rows lead every stratum. LACE leads all box-supervised rows in every stratum (ties BoxTeacher at [0.9,1]) and has the highest mean box IoU.
