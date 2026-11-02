# btmask: LACE's masker on BoxTeacher's detections (local only; Tom 2026-11-01)

Question: with the boxes held fixed, is LACE's masker better than BoxTeacher's own mask head? And what does BoxTeacher's detector + LACE's masker score as a system?

## Pre-registration (written 2026-11-01, BEFORE any number from this experiment)

**Boxes.** BoxTeacher B2's stitched detections, at the identical detection set and scores as its Table-1-protocol row (`tierb/results/20261101_192510/boxteacher/results/preds_boxteacher_sub_{val,test}_r2048.json`). Box = the stitched crown's mask-derived box, as in its scored row. Per tile, the top 600 by BoxTeacher's score are kept: maxDets 600 means no lower-ranked detection can affect any metric at that ranking.

**Arms (test, 439 tiles, frozen `score_coco`, 2048², maxDets 600, canopy crowd):**
- **A.** BoxTeacher masks, BoxTeacher score. This is the existing row: 0.6642 / 0.2231 / 0.3049. Re-scored here on the top-600 file as a gate, and it must reproduce exactly.
- **B0.** LACE `lace4px_simple` (FROZEN_simple_2026-10-26: `post4`, `c_peak` 0.33, r 12, eps 1e-4, alpha 0.3, kappa 1.6, 4 px DINOv3 features) on the same boxes, ranked by BoxTeacher's score. Masker swap only; ranking identical to A.
- **B\*.** The val-selected variant (below), same boxes, ranked by BoxTeacher's score.
- **Secondary (record):** B0 and B\* ranked by LACE's s′ = s·mean|2r−1|·mean(r), with s = BoxTeacher's score and r = the 4 px cell posterior of the masker box. The candidate set is the same top 600 by s, so s′ only re-orders within it. Scored at floor 0, the LACE convention; the floor already ran upstream on s.

**Val selection (B\*).** BoxTeacher's boxes are not LACE's boxes. On val, matched to GT (IoU ≥ 0.5), BoxTeacher's boxes have median size ratio 1.02 to GT with side margin +1.3% (IQR −2% to +5%). LACE s0's boxes have 0.97 and −1.1% (IQR −5% to +4%). The masker's knobs were chosen on LACE's boxes.
- Grid, 9 configs: box inflation e ∈ {0, 0.05, 0.10} × `c_peak` ∈ {0.25, 0.33, 0.40}.
  - e is the fraction of w and h added per side before masking, clipped to the image. The mask is clipped to the inflated box. The detection's reported box stays BoxTeacher's box, so box AP is unchanged.
- Score: 108 val tiles, `score_lite` crowd arm (parity-gated equal to `score_coco`), mask AP50, ranking by BoxTeacher's score.
- Rule: B\* = the config with the highest val mask AP50. If it is within 0.001 of the frozen config (e 0, c 0.33), B\* = the frozen config, and B\* is not run separately.
- Test is run once for B0 and once for B\* and reported whatever it gives.

**Comparison.** AP50 / AP75 / AP50:95 for A, B0 and B\*. A paired tile-bootstrap 95% CI of the AP50 difference B − A, using the legacy canopy arm via `fastap` as in `test_eval.boot_diff`, 2000 resamples, same ranking.

**Gates.**
1. Masker code: `btmask.render_boxes` on LACE s0's published boxes reproduces the published `lace4px_simple` masks byte-identically on 3 test tiles.
2. Arm A re-scored from the top-600 file equals the published BoxTeacher row.

## RESULTS (2026-11-02; run once)

**Gates:** masker 498/498 byte-identical, s′ exact (`gate_masker.json`). Arm A re-scored from the top-600 file reproduces the published BoxTeacher row exactly.

**Val (108 tiles, `val_grid.json`).** BoxTeacher masks: 0.7002 / 0.2967 / 0.3450 (AP50 / AP75 / AP50:95). Frozen LACE (e 0, c 0.33): **0.7040 / 0.3235 / 0.3565**, the best of the 9 configs, so B\* = B0. Every box inflation hurts, monotonically: e 0.10 / c 0.25 gives 0.593. This is consistent with `softbox`: the box clip is a regulariser.

**Test (439 tiles, frozen `score_coco`, 2048², maxDets 600, crowd; `compare.json`). Same detections, same boxes:**

| arm | ranking | AP50 | AP75 | AP50:95 | box AP50 |
|---|---|---|---|---|---|
| A: BoxTeacher masks | BoxTeacher score | 0.6642 | 0.2231 | 0.3049 | 0.7033 |
| **B0: LACE `lace4px_simple` masks** | BoxTeacher score | **0.6669** | **0.2528** | **0.3171** | 0.7033 |
| B0, secondary | LACE s′ (s = BoxTeacher score) | 0.6607 | 0.2385 | 0.3098 | 0.6992 |

- **Masker swap at fixed boxes and ranking:** AP50 +0.0027, AP75 +0.0297, AP50:95 +0.0122. Paired tile bootstrap of AP50 (legacy arm) B0 − A = +0.0028, 95% CI [−0.000, +0.0062]: borderline at AP50. The gain is concentrated at tight IoU.
- **LACE's s′ re-ranking hurts on these boxes** (−0.006 AP50 vs BoxTeacher's score). BoxTeacher's score already folds in its mask quality.
- **System:** BoxTeacher detector + LACE masker = 0.667 / 0.317. That is above every Table 1 row on AP50, and second to Restor (0.333) on AP50:95.

Files: `preds_*_r2048.json` / `scored_*_r2048.json` in this directory; per-tile masks in `feat_cache/btmask_{val,test}_*`. Logs: `native_raster/logs/btmask_{gate,val,test}_2026-11-0*.log`.
