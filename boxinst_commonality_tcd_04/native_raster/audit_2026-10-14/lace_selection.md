# Audit 2026-10-14: LACE selection history, test leakage, disclosure items

Auditor scope: what was chosen on val vs test, and when; leakage; disclosure items for Tom. Read-only. Nothing in the repo was changed. Scratch check script: `/private/tmp/claude-501/.../scratchpad/disjoint.py` (not kept).
Paths are relative to `boxinst_commonality_tcd_04/` unless they start with `/`. Times are local (+1100) unless marked UTC.

## 1. Chronology

| date | choice | selected on | evidence |
|---|---|---|---|
| 07-16 | Scoring raster 512² (pipeline constant) | a priori (never argued for) | NATIVE_RASTER.md:24-31; commit 723dde9 |
| 07-24 | 4-phase L24 detector; `mask_thr` 0.5 → 0.25 (0.5794 → 0.5831) | **TEST** 439, seed 0 | commit 573f496 msg + `phase4/README.md`@573f496:100-108 |
| 07-27 | β = 0.5 over β = 0 (+0.041 AP50) after the grid fix | **TEST** 439, seed 0 | modal_tcd_multiseed/phase4/README.md:241-248; commit 1fb2c7e |
| ~07-30 | α = 0.3, κ ×1.6. Same lever search also tried an "RGB / vegetation (ExG) guided boundary refinement (guided filter …)": **negative** | **TEST** (40-tile pilot → 130 → 439) | phase4/BOX2MASK_LEVERS.md:5-9, 27, 106-107; phase4/README.md:317-318; commit c45559d |
| 08-18 | α/κ/τ re-selected on val. Rule pre-registered with the sweep at 60/108 and no cell read, on a grid **centred on the test optimum**. The tie clause was written to favour points "already evaluated on test", but was never invoked: the val argmax is (0.3, 1.6, 0.25) outright on mAP50-95 | VAL 108 | BOX2MASK_LEVERS.md:104-149 (rule :112-115); commit bdaf3b6 |
| 08-18 | Sparse-236 built: 220 train-pool + **16 official-test** tiles | n/a | modal_sparse_tcd_multiseed/slice_manifest.json `counts.by_source`; build_slice.py:9-12 |
| 08-28 → 09-01 | Reranker `s·bimod·pfg_mean` replaces the val-fitted logreg. The form was found in an ablation that scored every candidate on **test**. Val selection is printed beside test AP, with no rule written beforehand. The val argmax on both val metrics is `s·bimod^0.5·pfg`. The chosen form ranks 3rd on val box AP50 and 2nd on val box AP50:95, and was picked by a post-hoc "exponent-free member of a 0.002 tie" rule (test 0.6615 vs 0.6595 for the val argmax, 512, s0) | val tie-break, made with test visible | confidence/ablation.py:12-16, 320-341; confidence/ablation_results.json `val_selection`; confidence/README.md:132-144; commit b39d12d |
| (all) | Detector checkpoint/early stop: val box AP50. EM masker fit: 120 tiles of the 792 train partition | VAL / TRAIN | train_detector_tiles.py:100, 154; phase4_fit_tcd.py:83; HYPERPARAMS.md §3-4 |
| 09-04 | First 2048 test number (LACE τ 0.25: 0.6204, −0.041) | TEST diagnostic | NATIVE_RASTER.md:94-98, 130-143; lace_s0_r2048_scored.json mtime 09-04 15:18 |
| 09-07 | τ@2048 = 0.40 (val optimum; rule = AP50, fixed before the run). Test τ 0.35 and 0.40 were both rendered afterwards; 0.40 kept although test AP50:95 favoured 0.35 | VAL (val json 15:34; test scored 20:53) | NATIVE_RASTER.md:99-125, 173-211; val_knobs_tau_r2048_s0_local.json (`n_val_tiles` 108) |
| 09-22 | Restor rpn1000 at 2048 on test: 0.6480 > LACE 0.6324. "the ordering flips" | TEST diagnostic | NATIVE_RASTER.md:394-404 |
| 09-22 | arXiv v1.1 uploaded at 512 (0.663 vs 0.626) | — | commit e43e4dc, arxiv/abstract_for_form.txt@e43e4dc |
| 09-26 | Decision: submit at 512, 2048 as appendix | taken with test known at both rasters | NATIVE_RASTER.md:490-500 |
| 10-06 | `fp_anatomy_test.py`, `small_crowns_test.py` (test 439) → "precision, not recall … mislocated". Goal set: "beat Restor rpn1000 at 2048 mask AP50, which needs about +0.018" | TEST diagnostic → direction **and target** of search | NATIVE_RASTER.md:534-548; HANDOFF_detector_retrain.md:17; mtimes 17:08 / 17:22 |
| 10-06/07 | Lever search, all on val: oracle boxes; soft box; decode variants; detector retrain (STOPPED on the forecast "≈ 0.637 against 0.6480 for Restor"); prior sweep (α 0.3 = argmax of 50 cells); prior decoupling | VAL; the retrain stop was judged against the test gap | NATIVE_RASTER.md:515-577, 705-777; det_val_eval/product, prior_sweep_val JSONs |
| 10-07 16:25 → 18:29 | Guided render: grey (21 cells) → colour (33) → extended grid (52 @2048 + 64 @512) → r 12, ε 1e-4, τ 0.35 (0.6636; split-half held-out +0.015 / +0.017); blur and wrong-tile controls. The grid was extended adaptively 3× after edge optima, and **no written selection rule** was found before the grids. Text: "First lever that closes the 2048 gap at val level … vs the test gap of ≈ 0.018" | VAL | guided_render_val.py, guided_rgb_val.py, guided_rgb_ext_val.py (all read `phase4/val_gt.json` + `feat_cache/feat_4p_val`); NATIVE_RASTER.md:779-860; logs/guided_*.log mtimes |
| 10-07 | Decision: re-base to 2048, remove 512, "**provided LACE is still competitive**" | conditional on the test outcome | NATIVE_RASTER.md:864-866 |
| 10-07 18:59 → 19:37 | Pre-registered single test run (r 12, ε 1e-4, τ 0.35, 3 seeds) → 0.6470 ± 0.0025 | frozen before the test number (consistent timestamps, see §2.4) | NATIVE_RASTER.md:868-903; lace_guided_modal.py:44 |
| 10-08 | Qualitative tiles: "No panel shows LACE below F1 0.8 in its window". Each model is shown at its own best-F1 threshold on the whole test set | TEST (display) | ablation/results/figs/README.md:40-48, 81-83; make_qualitative.py:12-13, 159-165 |
| — | Score floor 0 for product files | equivalence, not a choice: detector floor 0.05 is applied at decode | verified: guided s0 file has 159,687 dets, detector-score min 0.05000; PROTOCOL_439.md §1 |

## 2. Leakage verdict

**2.1 Split disjointness. Verified programmatically: PASS.**
- `train_tiles_gt.json`: 792 train + 108 val. `phase4/val_gt.json` keys = partition `val` = `feat_cache/feat_4p_val/*.npy` = `feat_cache/val_tiles.txt` (108).
- `test_gt.json`: 439 = `manifest.json` `feat_test`.
- Overlaps of val∩test, train∩test and train∩val are 0 by tile id and 0 by `image_id`. The sha1 check is weak, because the manifest carries sha1 for only 17/3/20 tiles.
- Every val script (`val_knobs_local`, `oracle_boxes_val`, `prior_sweep_val`, `prior_decouple_val`, `guided_render_val`, `guided_rgb_val`, `guided_rgb_ext_val`, `guided_control*_val`, `det_val_eval`, `softbox_val`) reads `VAL_GT` = `phase4/val_gt.json` and `FEAT_DIR` = `feat_4p_val`.
- Note that test tiles are named `tcd_val_tile_*` and val tiles `tcd_tile_*`. The naming is a trap, but the sets are disjoint.

**2.2 Scene adjacency (300 m clusters, `tile_index.json`).** 64/108 val tiles share a scene with a train tile, so val is optimistic relative to test; this is calibration, not leakage. 16/439 test tiles share a scene with a train tile, and 16/439 with a val tile.

**2.3 Fits on test. None in the deployed pipeline.**
- The detector trains on the 792 train tiles and stops early on val box AP50.
- The EM masker is fitted on train only.
- The reranker has zero fitted parameters.
- The SIZE prior and the logreg rows are ablation-only and fitted on val (`confidence/ablation.py:278, 313`).
- The pilot detector v2 has no test number.

**2.4 Pre-registration of the guided render.** Timestamps are consistent with the settings being frozen before any test number, but this is not provable from git:
- `lace_guided_modal.py` hard-codes `GF_R, GF_EPS, GF_TAU = 12, 1e-4, 0.35` (mtime 18:59:10).
- The container started at 08:02 UTC (= 19:02 local). Summaries carry `"preregistration": "NATIVE_RASTER.md 2026-10-07"`.
- Test scoring ran at 19:37. The first commit containing the PRE-REGISTRATION section is c568617 (19:48), which also contains the RESULT, so git does not order them.
- Only one guided config exists on test in the repo; the α/κ ablations reuse r 12 / τ 0.35.
- No local test features exist beyond 2 figure tiles (`figs/arch/_cache`), so local test renders were impossible.
- Modal volumes were not inspected.

**2.5 Values chosen on test, historically.**

| choice | status now |
|---|---|
| α, κ (July) | re-selected on val 08-18; same values. The val grid was the test neighbourhood |
| τ | fully re-selected on val at 2048, so the July test choice is moot |
| β = 0.5 (July, test) | **never re-selected**. The val crossing (tab:components) is flat across β, so the effect is ≈ 0, but the choice is test-made |
| Reranker form | val tie-break made with test visible. The cost of using the val argmax instead is −0.002 AP50 (512, s0) |
| RGB guided filter | tried on test in July (negative) and revived on val in October. This is a forking path, not leakage of the value |

**2.6 Test diagnostics vs test selection.**
- Test diagnostics set the **direction, target and stopping rule** of the October search: the explicit target was the +0.018 test gap to Restor, the retrain was stopped against it, and the guided render was accepted because its val gain "closes the … test gap".
- Test diagnostics also gated the reporting-raster decision ("provided LACE is still competitive").
- **No value was chosen on test** in the 2048 re-base. Val → test transfer was clean: full-val +0.021, split-half +0.015/+0.017, test +0.017.
- The selection metric varied across selections: α/κ on val mAP50-95; τ, r and ε on val AP50; the reranker on val box AP50. Each metric was stated in its record.

**2.7 Sparse-236.**
- It is **not** a subset of the 439: 220 tiles are train-pool and 16 are official test (all 16 ⊂ 439).
- It is tile- and image_id-disjoint from the 900, but 130/236 share a 300 m scene with the seen 900 (`scene_clean` = 106).
- No LACE value was selected on sparse. Its 16 test tiles were in the July test-tuning pool (build_slice.py:11-12).

## 3. Disclosure items for Tom (proposals only)

**(a) Abstract reports AP50 only.**
- Quote: "LACE reaches a mask AP$_{50}$ of $0.647 \pm 0.003$ ($3$ seeds) … level with the $0.648$ of Restor's released mask-supervised Mask R-CNN".
- Table 1: AP50:95 is LACE 0.278, Restor 0.333, Detectree2 0.301, so LACE is 3rd.
- Sparse quote: "mask AP$_{50}$ rises to $0.678$ versus $0.625$ for Detectree2". In tab:sparse236 LACE has AP50:95 0.3215 vs Detectree2 0.3460.
- Propose: "…level with the $0.648$ of Restor's … Mask R-CNN at AP$_{50}$, though behind it at AP$_{50:95}$ ($0.278$ against $0.333$)", and for sparse "…versus $0.625$ for Detectree2 (AP$_{50:95}$ $0.322$ against $0.346$)".

**(a2) "Without mask annotations" / "from boxes alone".**
- Quote: "LACE matches fully-supervised specialist baselines from boxes alone".
- Masks of the 108 val images (7,041 crowns) select α, κ, τ, r and ε. The paper says so at L517 but not in the abstract.
- Propose: "…from boxes, with five inference settings selected on $108$ held-out mask-labelled images".

**(b) Qualitative figures (L195, L664, L697).**
- Quotes: "Crown masks on six windows of the sparse-canopy holdout ($1024$~px)." / "Crown masks on four OAM-TCD holdout windows ($768$~px, native $0.1$~m/px)." / "Crown boxes on six NeonTreeEvaluation tiles, one per site."
- Thresholds are symmetric: each model is shown at its own whole-set best-F1 threshold (LACE 0.11, Restor 0.24, DT2 0.49, B2M 0.77). The captions say neither this nor how the tiles were chosen.
- The F1 ≥ 0.8 floor applies to LACE only:
  - tcd439: 5/41 eligible windows meet it, and 4 were used (ranks 1, 2, 3, 5 by LACE window F1). LACE ≥ Restor in 4/4 shown windows against 18/41 eligible, and LACE is best of the four models in 11/41. An earlier draft in which "Restor led every 439 window" was rejected (figs/README.md:46).
  - The whole-tile base rate is LACE leading on 7/43 tiles (stale bilinear ranking, 10-06).
  - Sparse is representative: LACE beats DT2 in 33/41 windows.
- Propose (tcd439): "Windows chosen from tiles with < 3 % canopy among those where LACE's window F1 ≥ 0.8 (4 of 41 eligible); each model shown at its own best-F1 score threshold over the test set. LACE leads on 11 of 41 eligible windows." Same pattern for sparse and NEON.

**(c) Chronology vs presentation.**
- Quotes: L513 "we upsample it to the native $2048$~px tile with a guided filter … so that mask boundaries follow edges visible in the image"; L942 "The guided upsampling … recovers part of this at the native raster"; L1159 "Mask raster & $2048^2$ for every row".
- These read as a priori design. In fact the guided render was added on 10-07 to close a test-observed gap. A guided filter had failed on test at 512 in July. The protocol was 512 through arXiv v1.1. No version note appears in paper.tex.
- Propose (sec:inference or the protocol appendix): "An earlier version of this work scored masks at $512^2$. After moving to the native raster, where bilinear upsampling trailed the Mask R-CNN, we added the guided render; its settings were selected on validation and frozen before a single test evaluation."

**(d) L827.**
- Quote: "$\alpha$ and $\kappa$ were chosen on validation by a rule fixed before any validation cell was read, and the test split was not used for their selection."
- This is contradicted by BOX2MASK_LEVERS.md:106 ("Everything above chose α/κ on TEST tiles").
- Propose: "$\alpha$ and $\kappa$ were first tuned on the test split during development; we re-selected them on validation under a rule fixed before any validation cell was read, and it returned the same values."

**(e) L515 and L813 (β).**
- Quotes: "they are the only quantities in the pipeline selected against mask labels" and "fixed a priori except for … $\beta$".
- β = 0.5 was chosen on test 439 (07-27) and never re-selected; the val tab:components is flat across β.
- Propose: append "…and $\beta$, chosen during development on the test split (Table~\ref{tab:components} shows validation accuracy flat in $\beta$)".

**(f) L517.**
- Quote: "…selected on the $108$ validation images … under a rule fixed in advance".
- This is true for α, κ and the bilinear τ. No pre-stated rule was found for r, ε and the guided τ 0.35, whose grid was extended adaptively.
- Propose: "…r, ε and τ by maximum validation AP$_{50}$ over an adaptively extended grid, checked by split-half selection (held-out gain $+0.015$/$+0.017$)".

**(g) L544 (reranker).**
- Quote: "the unweighted product was selected as the exponent-free member of a group that could not be separated at that precision."
- The val argmax (both val metrics) is `s·bimod^0.5·pfg`. The tie rule was applied with test AP visible (ablation.py prints both).
- Propose: add "the candidate set and tie rule were settled in an ablation that also reported test AP; the validation-best member scores $0.002$ lower on test".

**(h) L860 and L899 (sparse).**
- Quotes: "On $236$ tiles which do not overlap with the training/validation set at all" and "shares no tile with the $900$ training images, so it is a generalisation test".
- 16/236 are 439-test tiles (test-tuned in July), and 130/236 share a 300 m scene with training/val tiles.
- Propose: "…236 tiles sharing no tile with the training/validation set (16 are drawn from the 439 test split; 130 share a 300 m scene with a training tile)".

**(i) NEON, abstract.**
- Quote: "LACE reaches $0.728 \pm 0.003$ F1@0.4 … matching the authors' DeepForest model's published $0.719$".
- LACE's figure is at its best-F1 threshold on the 194 test tiles. 0.719 is DeepForest's fixed published operating point; at that recall LACE's P@R0.79 = 0.624 < 0.659 (modal_neon_multiseed/README.md:34-35). The same-protocol comparator is the v2.1.0 repro at 0.726. tab NEON (L678) discloses best-F1; the abstract does not.
- Propose: "…level with DeepForest ($0.726$, our run of v2.1.0 at its best-$F_1$ point; $0.719$ published)".

## 4. Not verified
- Modal volume contents: no other guided test outputs could be ruled out on volumes.
- Provenance of the detector loss weights and lr. `FALSEHOOD_AUDIT_2026-09-26.md`, cited in memory, is not in the repo or git history.
