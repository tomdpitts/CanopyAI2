# btmask: LACE's masker on BoxTeacher's detections (local only; Tom 2026-11-01)

<!-- TABLE1_CURRENT:begin (auto, results_439/mark_superseded.py) -->
> ⚠️ **SUPERSEDED 439 / sparse236 NUMBERS (marked 2026-10-27; table updated 2026-11-04).** Current headline table: [`TABLE1_CURRENT.md`](../../../results_439/TABLE1_CURRENT.md). Test-set numbers in this file are a historical experiment record (512 raster, pre-seam-fix baselines, 8 px masker or earlier models) and must not be quoted unless they match that table.
<!-- TABLE1_CURRENT:end -->

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

## Pre-registration BTMASK-EF (2026-11-26, BEFORE any number from this re-run)

Purpose: the 11-02 run above used `lace4px_simple` BEFORE the 2026-11-04 top/left border fix (`GA.EDGE_FILL`), so its numbers cannot be quoted. Re-run with the CURRENT deployed LACE masker. Script `btmask_ef.py` (this directory); log `native_raster/logs/btmask_ef_2026-11-26.log`; results `compare_ef.json`, `gate_ef.json`; per-tile masks `feat_cache/btmask_ef_<split>/` (resume); new output names only (`*_ef*`), no 11-02 file is overwritten.

**Deployed masker (confirmed from code, not from summaries).** `btmask.render_boxes` with FROZEN = e 0, c_peak 0.33 → `hunt/render.post4` (4 px DINOv3 features `feat_cache/hunt_z4p_<split>`, `em_model_4p_fix.npz` with s 4 / origin 8, α 0.3, κ ×1.6, colour-guided r 12, ε 1e-4, τ = 0.33 × in-box peak, mask clipped to box) and `guided_ablate_modal.EDGE_FILL = True` (the module default since 11-04; asserted True at run time). This is exactly what `dino_det/edgefix_test.py` and `guide_controls_test.py` ran for the deployed D1 masks.

**Boxes.** BoxTeacher B2's stitched detections, top 600 per tile by BoxTeacher's score, exactly `btmask.bt_top` (`box_supervised_baselines/tierb/results/20261101_192510/boxteacher/results/preds_boxteacher_sub_{val,test}_r2048.json`; sparse: `tierb/results_sparse/preds_boxteacher_sub_sparse_r2048.json`). Min BoxTeacher score is ≥ 0.187 on every split, so the 0.05 floor never binds.

**Arms (frozen `score_coco`, 2048², maxDets 600, canopy-neutral crowd arm):**
- **A.** BoxTeacher masks + BoxTeacher score (floor 0.05, as published). Gate: must reproduce 0.6642 / 0.3049 (AP50 / AP50:95) on test exactly; on sparse, 0.7206 / 0.3756.
- **B (primary).** Deployed LACE masks (EDGE_FILL True) on the same boxes, ranked by BoxTeacher's score (floor 0.05). Masks are the only change from A.
- **C (secondary, record only).** The B masks ranked by ŝ (R3), refitted on BoxTeacher VAL detections with the frozen R3 recipe (`dino_det/rerankR3_final.json` chosen config; code `rerank3.fit` / `oof_fits`, features `rerank._feats`): MLP 1×64, 30 epochs, dropout 0.1, wd 1e-3, AdamW lr 1e-3, batch 1024, BCE; inputs [s, log(s′/s) = log(p̄m), log box h, max box IoU with a higher-s det], s = BoxTeacher's score, s′ = s·p̄·m from the 4 px in-box posterior of the B masker; labels box50 (greedy in s′ order, box IoU ≥ 0.5 vs GT tree bbox; unmatched dets with ≥ 50% of the LACE mask in the canopy union are crowd-ignored and dropped from training); 25 models = seeds 0–4 × the 5 round-robin tile folds (sorted 108 val tiles); ŝ = mean of the 25 predicted probabilities. Val candidate set = top 600 per tile by BoxTeacher's score (same rule as test). Floor 0 (LACE convention). No hyperparameter is chosen here or on test; nothing new is selected.
- **No new config selection:** the 11-02 val grid chose the frozen config (e 0, c 0.33), so it is kept. The val split is rendered only to fit C's ŝ.

**Splits.** Test 439 primary. Sparse-236 secondary (BoxTeacher sparse predictions exist), same arms, the same val-fitted ŝ.

**Metrics.** Mask AP50, AP75, AP50:95, box AP50 from `score_coco`. Paired tile bootstrap (2,000 resamples, seed 0; `dino_det/s2_final.boot` on `fasteval` per-tile matches, crowd arm) of B − A and C − A at AP50 and AP50:95.

**Gates, before any test number:**
1. With EDGE_FILL True, `render_boxes` on D1's boxes reproduces the deployed edge-fixed D1 masks (`dino_det/results/edgefix/preds_d1_edgefix_test_sprime_r2048.json`) byte-identically on 3 test tiles (first 3 sorted tiles with detections), and s′ to 1e-6.
2. Arm A re-scored from the top-600 file reproduces the published BoxTeacher row exactly (test; sparse likewise).
3. (C only) `rerank._feats` on `feat_cache/dino_swinl_s0_val_mask.json` reproduces cached `rerankR1_val.npz` rows on 3 val tiles; and the R3 recipe code refit on `rerankR1_val.npz` reproduces the published R3 test scores (|Δ| < 1e-6). If either fails, C is skipped (A/B unaffected).
4. `fasteval` reproduces `score_coco` (|Δ| < 1.5e-4, AP50 and AP50:95) for every arm used in the bootstrap.
If gate 1 or 2 fails: stop and report. Test is looked at once; results appended below under "RESULTS, BTMASK-EF" whatever they show.

## RESULTS, BTMASK-EF (2026-11-26; ONE look per split; all gates PASS; single BoxTeacher run)

**Gates.** (1) Deployed masker (EDGE_FILL True) on D1 test boxes = `preds_d1_edgefix_test_sprime` masks byte-identical, 879/879 on 3 tiles, s′ |Δ| 0. (2) Arm A from the top-600 file = published BoxTeacher row exactly (test 0.6642 / 0.2231 / 0.3049; sparse 0.7206 / 0.3590 / 0.3756). (3) `rerank._feats` reproduces `rerankR1_val.npz` (3 val tiles); R3 recipe refit reproduces published R3 test scores, max |Δ| 0.0 → arm C run. (4) fasteval = score_coco (4 dp) on every arm, both splits. `gate_ef.json`.

**C's ŝ fit (BoxTeacher val, top 600/tile):** 64,800 dets; box50 labels 5,921 pos / 38,618 neg / 20,261 crowd-ignored; 25 models (seeds 0–4 × rr folds), frozen R3 config.

Frozen score_coco, 2048², maxDets 600, canopy-neutral crowd; same boxes in every arm (box AP50 differs only through ranking). Bootstrap = paired tile, 2,000, seed 0, crowd arm (`s2_final.boot`), vs A. (The 11-02 CI used the legacy arm; not comparable.)

| split | arm | masks | ranking | AP50 | AP75 | AP50:95 | box AP50 | Δ AP50 vs A [95% CI] | Δ AP50:95 vs A [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| test 439 | A | BoxTeacher | BT score | 0.6642 | 0.2231 | 0.3049 | 0.7033 | | |
| test 439 | **B** | LACE (deployed, EF) | BT score | **0.6676** | 0.2560 | **0.3185** | 0.7033 | +0.0034 [+0.0008, +0.0071] | +0.0136 [+0.0107, +0.0171] |
| test 439 | C | LACE (deployed, EF) | ŝ (R3 refit on BT val) | 0.6852 | 0.2549 | 0.3240 | 0.7198 | +0.0210 [+0.0172, +0.0253] | +0.0191 [+0.0158, +0.0227] |
| sparse 236 | A | BoxTeacher | BT score | 0.7206 | 0.3590 | 0.3756 | 0.7391 | | |
| sparse 236 | **B** | LACE (deployed, EF) | BT score | **0.7232** | 0.3840 | **0.3893** | 0.7391 | +0.0026 [−0.0010, +0.0063] | +0.0137 [+0.0075, +0.0202] |
| sparse 236 | C | LACE (deployed, EF) | ŝ (R3 refit on BT val) | 0.7297 | 0.3834 | 0.3909 | 0.7451 | +0.0091 [+0.0025, +0.0167] | +0.0153 [+0.0077, +0.0238] |

**Reading (framing per Tom, 2026-11-26: ŝ is part of LACE's masker, not a separate component; LACE paper framing decided 2026-09-03).**
- **LACE's masker (masks + ŝ) on BoxTeacher's boxes = arm C.** Versus BoxTeacher's own head (A): AP50 +0.021 [+0.017, +0.025] test, +0.009 [+0.003, +0.017] sparse; AP50:95 +0.019 [+0.016, +0.023] test, +0.015 [+0.008, +0.024] sparse. All four CIs exclude 0. Box AP50 also rises (+0.017 / +0.006), because ŝ re-orders the same boxes.
- **Adaptation to BoxTeacher:** the masks use no BoxTeacher input except the box. ŝ was refitted on BoxTeacher's 108 val tiles with the frozen R3 recipe, which is the same per-detector step LACE uses for D1 (fit on D1 val). No hyperparameter was chosen on BoxTeacher val or on test.
- **Disclosure:** the pre-registration above named B as primary and C as secondary. That label is kept as written. Under the masker-includes-ŝ framing, C is the comparison that matters; B is a mask-only ablation.
- **Mask-only ablation (B − A, BoxTeacher's ranking kept):** AP50:95 +0.014 on both splits (CIs > 0), AP75 +0.033 / +0.025; AP50 +0.003 test (CI > 0), +0.003 sparse (CI includes 0). So the AP50 gain over BoxTeacher's head comes mostly from ŝ, and the strict-IoU gain comes from the masks.
- B on test (0.6676 / 0.3185) is within 0.0014 of the pre-fix 11-02 B0 (0.6669 / 0.3171). The 11-02 ranking was s′ = s·p̄·m, which hurt (−0.006); the val-refit ŝ helps.
- Single BoxTeacher run, single look per split; no seed bands.

Files (this directory): `btmask_ef.py`, `gate_ef.json`, `compare_ef.json`, `preds_{A,B,C}_ef_{test,sparse}_r2048.json`, `scored_{A,B,C}_ef_{test,sparse}_r2048.json` (gitignored). Per-tile masks + s′ `feat_cache/btmask_ef_{val,test,sparse}/`; features `feat_cache/btmask_ef_feats_{val,test,sparse}.npz`; fasteval caches `feat_cache/btmask_ef_fasteval_{test,sparse}_{A,B}.pkl`. Log `native_raster/logs/btmask_ef_2026-11-26.log`. paper.tex NOT edited.

## Pre-registration BTMASK-EF arm D (2026-11-26, Tom: "using s from boxteacher detections, but not modifying s hat"; BEFORE any arm-D number)

Question: does LACE's masker with the deployed OAM-TCD ŝ, frozen and unmodified, transfer to BoxTeacher's detections? Script `btmask_ef_d.py`; results `compare_ef_d.json`; log `native_raster/logs/btmask_ef_d_2026-11-26.log`.
- **Arm D.** The same boxes and the same deployed LACE masks as arm B/C (`preds_B_ef_<split>_r2048.json`). Ranking = the deployed ŝ (R3): the 25 published models (frozen R3 recipe trained on D1 val, `rerankR1_val.npz`), applied unchanged to the cached BoxTeacher features (`feat_cache/btmask_ef_feats_<split>.npz`). The inputs are s = BoxTeacher's score, log(p̄m) from the LACE posterior on BoxTeacher's box, log box h, and max box IoU with a higher-s det. Standardisation is D1 val's, as frozen. Nothing is fitted on BoxTeacher data.
- **Gate D1.** The refit of the R3 recipe on `rerankR1_val.npz` reproduces the published R3 test scores (|Δ| < 1e-6), i.e. these are the deployed ŝ models. **Gate D2.** fasteval = score_coco (4 dp) for D. The cached feature column 0 must equal BoxTeacher's score.
- **Scoring.** Frozen score_coco (2048², maxDets 600, canopy-neutral crowd, floor 0). Test 439 and sparse 236, one look each. Paired tile bootstrap (2,000, seed 0) D − A (primary for this question), plus D − C (frozen vs refitted ŝ, record).
- **Diagnostic (record).** Quantiles of the 4 ŝ inputs on D1 val vs BoxTeacher test (input shift).
- Disclosure: one more test look and one more sparse look on BoxTeacher's boxes; masks unchanged.

## RESULTS, BTMASK-EF arm D (2026-11-26; ONE look per split; gates D1/D2 PASS)

Gate D1: the R3 recipe refit on `rerankR1_val.npz` = published R3 test scores (max |Δ| 0.0), so these are the deployed ŝ models. Gate D2: fasteval = score_coco on both splits; feature s = BoxTeacher's score (asserted). `compare_ef_d.json`, log `native_raster/logs/btmask_ef_d_2026-11-26.log`.

Same boxes, same deployed LACE masks as B/C. Bootstrap = paired tile, 2,000, seed 0.

| split | arm | ranking | AP50 | AP75 | AP50:95 | box AP50 | Δ AP50 vs A [95% CI] | Δ AP50:95 vs A [95% CI] |
|---|---|---|---|---|---|---|---|---|
| test 439 | A | BT head + BT score | 0.6642 | 0.2231 | 0.3049 | 0.7033 | | |
| test 439 | **D** | deployed ŝ, unchanged | **0.6767** | 0.2533 | **0.3207** | 0.7125 | +0.0125 [+0.0054, +0.0205] | +0.0158 [+0.0109, +0.0212] |
| test 439 | C | ŝ refit on BT val | 0.6852 | 0.2549 | 0.3240 | 0.7198 | +0.0210 [+0.0172, +0.0253] | +0.0191 [+0.0158, +0.0227] |
| sparse 236 | A | BT head + BT score | 0.7206 | 0.3590 | 0.3756 | 0.7391 | | |
| sparse 236 | **D** | deployed ŝ, unchanged | **0.7186** | 0.3739 | **0.3823** | 0.7360 | −0.0020 [−0.0084, +0.0053] | +0.0067 [−0.0024, +0.0172] |
| sparse 236 | C | ŝ refit on BT val | 0.7297 | 0.3834 | 0.3909 | 0.7451 | +0.0091 [+0.0025, +0.0167] | +0.0153 [+0.0077, +0.0238] |

D − C (frozen − refit): AP50 −0.0085 [−0.0153, −0.0010] test, −0.0111 [−0.0163, −0.0053] sparse; AP50:95 −0.0033 [−0.0064, +0.0001] test, −0.0086 [−0.0123, −0.0041] sparse.

**Input shift (quantiles 5/25/50/75/95%, D1 val vs BoxTeacher test).**
- s: 0.057 / 0.091 / 0.147 / 0.280 / 0.647 vs 0.318 / 0.399 / 0.475 / 0.580 / 0.762. BoxTeacher's scores sit almost wholly above D1's median, far outside the range the frozen ŝ was trained on.
- log h: 3.47 vs 3.71 median (BoxTeacher's boxes are larger).
- max IoU↑: 95th percentile 0.57 vs 0.61; the 75th percentile is 0.15 vs 0.37 (BoxTeacher keeps more overlapping dets).
- log(p̄m): close (−1.28 vs −1.31 median).

**Reading.**
- **Unchanged ŝ on test:** LACE (masks + ŝ) beats BoxTeacher's head, +0.013 AP50 and +0.016 AP50:95, with both CIs > 0. That is also +0.009 AP50 over B, i.e. the frozen ŝ still adds on top of BoxTeacher's own score.
- **Unchanged ŝ on sparse:** no AP50 gain (−0.002, CI spans 0); AP50:95 +0.007 (CI spans 0). It sits below B (0.7232 / 0.3893), so on sparse the frozen ŝ costs relative to BoxTeacher's score.
- **Refit vs frozen:** refitting ŝ on the new detector's val tiles (C) is significantly better than the frozen ŝ (D) on both splits. The likely cause is the large shift in s: ŝ's strongest input (T3: dropping s costs −0.07 AP50) is a detector-specific score whose scale does not transfer.
- Single BoxTeacher run; one look per split.

### Best-F1 operating point, arms A and D (2026-11-26; `btmask_ef_f1.py` → `f1_ef.json`; routine = `ablation/results/figs/table1_best_f1.py` / `make_qualitative.best_f1_threshold`; descriptive)
| split | arm | thr | TP | FP | FN | F1 |
|---|---|---|---|---|---|---|
| test | A | 0.65 | 16,218 | 7,136 | 9,487 | 0.6612 |
| test | D | 0.85 | 16,188 | 6,651 | 9,517 | 0.6669 |
| sparse | A | 0.69 | 9,911 | 2,983 | 5,026 | 0.7122 |
| sparse | D | 0.89 | 9,944 | 2,772 | 4,993 | 0.7192 |

Test A = paper Table 1 BoxTeacher row exactly. Sparse A ≠ paper Table 2 (10,140 / 3,013 / 4,797 / 0.722): the F1 routine has no per-tile cap, and the paper row uses BoxTeacher's uncapped file. Two sparse tiles hold more than 600 BoxTeacher dets above 0.69 (264 dets beyond the cap). A and D here share the identical top-600 set (D's masks exist only for it), so A vs D is like-for-like. AP is unaffected (maxDets 600).

### AP75 interval, arm D − A, test (2026-11-26; `btmask_ef_ci.py` → `ci_ef.json`, log `native_raster/logs/btmask_ef_ci_2026-11-26.log`)
Same resampling as `s2_final.boot` (2,000, seed 0). Gate: AP50 and AP50:95 reproduce `compare_ef_d.json` exactly (PASS). AP75 +0.0302 [+0.0204, +0.0400]. Test only (sparse not run). Used in paper tab:btbox-test Δ row.
