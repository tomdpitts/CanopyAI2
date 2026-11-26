# dino_det: LACE (D1 variant) — DINO-Swin-L detector (SelvaBox-style architecture) + LACE masker (2026-11-03)

**Naming (Tom, 2026-11-03):** D1 *is* LACE. Externally it is just "LACE"; in repo docs "LACE (D1 variant)". The earlier frozen-DINOv3 CenterNet-style detector + masker (0.655) is "LACE (CenterNet variant)". "D1" below is shorthand for the D1 variant / its detector stage.

<!-- TABLE1_CURRENT:begin (auto, results_439/mark_superseded.py) -->
> ⚠️ **SUPERSEDED 439 / sparse236 NUMBERS (marked 2026-10-27; table updated 2026-11-04).** Current headline table: [`TABLE1_CURRENT.md`](../results_439/TABLE1_CURRENT.md). Test-set numbers in this file are a historical experiment record (512 raster, pre-seam-fix baselines, 8 px masker or earlier models) and must not be quoted unless they match that table.
<!-- TABLE1_CURRENT:end -->

**Idea (Tom, 2026-11-03).** Train SelvaBox's detector architecture ourselves, on the 792 OAM-TCD training tiles only (BoxTeacher's crops and selection set), from generic pretrained weights, never SelvaBox's. Then run the unchanged LACE masker and s′.
- Architecture: detrex DINO, Swin-L, `dino_swin_large_384_5scale_36ep`.
- Pretraining: detrex's COCO-trained checkpoint. This matches BoxTeacher's ImageNet-22k + COCO box pretraining.

**Status:** local timing smoke written (`smoke_local.py`). The toolchain install (`install_local.sh`: an isolated py3.10.20 venv at `~/dphil/dino_local/venv`, detrex + detectron2 from GitHub, the 877,585,469-byte COCO checkpoint from detrex-storage v0.2.1) is **awaiting Tom's permission** (auto mode blocked fetching external code). No number exists.

**Smoke purpose:** a measured s/iter and memory on MPS, to choose local training (estimated 1.5–5 days) vs Modal (estimated $35–70). Its loss values are not results.

## Local smoke result (2026-11-03; log `native_raster/logs/dino_local_smoke_2026-11-03.log`)
- Toolchain: detrex `e244e6c`, detectron2 `70f4543`, torch 2.14.1 (MPS); COCO checkpoint sha256 `55dbb477…` (full hash in `~/dphil/dino_local/*.sha256`).
- Fixes needed for the Mac:
  - `setuptools<81` (detrex imports `pkg_resources`);
  - DINO's hard-coded `.cuda()` / `.to("cuda")` calls redirected to MPS inside `smoke_local.py` (detrex is not edited).
- The COCO checkpoint loads; only the 80-class heads are skipped (num_classes 1), as expected.
- **Batch 1 (one 1024 crop per training step): median 2.85 s/step** (10 timed steps after 3 warm-ups).
- **Batch 2 (two crops per step): not viable.** 202 / 355 / 450 s for the first 3 steps; memory thrashing on 64 GB unified memory. Stopped by Tom's go; no JSON was written.
- **Projection at batch 1:** 3,168 crops per epoch ≈ 2.5 h per epoch, so **12 epochs ≈ 31 h, 36 epochs ≈ 92 h** locally.
- Batch 1 also changes the recipe (DINO's released recipe uses batch 16, so the learning rate and the effective noise would need scaling).

## Pre-registration D1 (2026-11-03, BEFORE any trained-DINO number): DINO-Swin-L trained on OAM-TCD boxes + LACE masker

Tom: proceed through the single test run; **budget $75** (live hard stop; the run checkpoints every epoch and resumes past the cap later if Tom allows). The smoke test is small (1 GPU), with no cost sign-off step.

**Detector.**
- detrex DINO `dino_swin_large_384_5scale_36ep` (Swin-L ImageNet-22k backbone, 5-scale deformable encoder, 900 queries, denoising, two-stage), detrex `e244e6c` / detectron2 `70f4543`.
- Initialised from the **COCO-trained DINO checkpoint** (detrex-storage v0.2.1, sha256 `55dbb477…`). `num_classes = 1`, so the COCO class heads are not loaded.
- This is generic pretraining. SelvaBox's weights are never used.

**Supervision and data: identical to Tier B BoxTeacher (`TIERB.md` §2–3).**
- LACE's train boxes (`train_tiles_gt.json`, 792 train tiles), on the staged HF pixels `tcd04-tierb-vol:rgb`.
- Tier B's jittered 2×2 partition: 3,168 crops of 1024 per epoch, scale s ∈ {0.80 … 1.00}, horizontal flip p = 0.5 (`crops.py`, `d2/d2_data.TierBMapper`, aug_type `none`). The image format is RGB, as detrex DINO expects.
- No mask or crown polygon is ever read.

**Canopy (LACE's rule, adapted to set prediction).**
- An UNMATCHED query whose predicted box is ≥ 50% covered by canopy pixels (`TierBMapper`'s canopy raster, which is LACE's `canopy_px`) contributes no negative classification loss. This holds in the final layer, every auxiliary layer and the two-stage encoder output.
- Matched queries and denoising queries are unchanged.
- This mirrors the evaluation's crowd-ignore (IoA ≥ 0.5).

**Recipe.**
- detrex DINO's released optimiser: AdamW lr 1e-4, weight decay 1e-4, backbone lr × 0.1, grad-clip max-norm 0.1, fp32 (detrex `amp.enabled = False`).
- Batch 16 = 8 × A100-80GB × 2. Seed 0.
- **Schedule:** Tier B's convergence rule (`plateau.py`; the same rule as BoxTeacher). One evaluation per epoch (198 iterations). LR × 0.1 on a plateau (Δ 0.005, patience 2, at most 2 decays), stop after the second decay plus patience, min 4 / max 20 epochs.
- **Selection metric:** val box AP50 on the **896 val 1024 subtiles**, unstitched (`tierb_data/val_sub_gt_1024.json`; `score_lite.window_dets`, maxDets 600, floor 0.05). This is BoxTeacher's selection set. Best epoch kept.

**Inference.**
- 3 × 3 subtiles of 1024 at stride 512 per 2048 tile; DINO top 300 per subtile.
- **Box stitching = BoxTeacher's seam-fixed rule on boxes:** floor 0.05; drop boxes within 10 px of an interior subtile edge; dedup in score order, dropping a box if its IoU with a kept box > 0.7 or its containment (intersection / own area) > 0.85; keep the top 600 per tile.

**Masks and ranking:**
- LACE `lace4px_simple` (FROZEN_simple_2026-10-26), unchanged, local, on the stitched boxes.
- Ranking chosen on val by full-pipeline crowd mask AP50 between **detector s** and **s′** (s = DINO score). Ties within 0.001 → detector s.

**Test:** once, on the 439 tiles, at the selected ranking; frozen `score_coco` (2048², maxDets 600, crowd arm). Both rankings are reported; the selected one is the headline.

**Smoke (no number used):** 1 × A100-80GB, about 20 iterations at batch 2, a checkpoint write and resume, a mini-eval on 9 val subtiles, and the peak memory.

**Budget:** a ledger on the volume (billed seconds × Modal list rate, CPU and memory included). Training stops and checkpoints at the first epoch end where the projected spend exceeds $75 minus a reserve for predictions ($4).

**Report:**
- mask AP50 / AP75 / AP50:95 and box AP50 / AP75;
- against BoxTeacher (0.664 / 0.703), C2 (0.6627 / 0.693), LACE (0.655 / 0.666), and SelvaBox + LACE masker (A2, 8 px masker: 0.711 / 0.767; reference only, different training data);
- val recall by size;
- the actual Modal bill.

### D1 smoke test PASS (2026-11-03; `tcd04-tierb-vol:runs/dino_swinl_s0_smoke/`)
- 1 × A100-80GB, 20 iterations at batch 2: about 1.8 s/iteration, **peak GPU memory 41.9 GB** (so 80 GB GPUs are needed).
- Val mini-eval on 9 subtiles: 5 s.
- Checkpoint written; a second launch logged `[resume] from epoch 1` and stopped correctly.
- Cost $0.54 (ledger).
- Its AP (0.255 after 20 iterations on 1 tile) is meaningless and unused.
- **Projection:** about 7–8 min and about $3 per epoch on 8 × A100-80GB ($23.4/h list incl. CPU and memory), so 10–15 epochs ≈ $30–45.
- Full training launched 2026-11-03.

### D1 launch incident (2026-11-03; no number affected)
- The first 8-GPU launch (`fc-01M6EEECTW0D01W8GEVS99VYDE`) failed twice in the first minute: the DataLoader's `lambda` collate is unpicklable under detectron2's spawned DDP workers. The 1-GPU smoke runs without spawning, so it could not catch this. The third retry was cancelled. Ledger $3.77 (Modal metered +$4.00 for everything so far).
- **Fix:** a module-level `_collate` and `multiprocessing_context="fork"` (CPU-only workers).
- **Added a 2-GPU DDP smoke** (`smoke2`, `runs/dino_swinl_s0_smoke2gpu/`): PASS (10 iterations at batch 4, 42.7 GB/GPU, eval and gather OK).
- Relaunched as `fc-01M6EF44ZCBBGFVMY2J2ANB90K`.
- The ledger's 240 s startup charge per launch over-counts slightly against Modal's actual bill, so the cap is conservative.

## RESULTS, pre-registration D1 (2026-11-03)

**Training** (`tcd04-tierb-vol:runs/dino_swinl_s0/`, `fc-01M6EF44ZCBBGFVMY2J2ANB90K`; 8 × A100-80GB, about 7.0 min/epoch).
- Val box AP50 on the 896 subtiles, epochs 1–10: 0.700 / 0.739 / 0.751 / 0.749 / 0.741 [decay 1] / **0.761** / 0.756 / 0.758 [decay 2] / 0.757 / 0.757. Plateau stop at epoch 10; **best epoch 6** (box AP75 0.458).
- BoxTeacher's best on the same metric: 0.724.

**Val selection** (108 tiles, full pipeline; `results/select_val_dino_swinl_s0.json`): **s′ selected**.

| ranking | mask AP50 | AP75 | AP50:95 | box AP50 | box AP75 |
|---|---|---|---|---|---|
| detector s | 0.7356 | 0.338 | 0.376 | 0.7546 | 0.459 |
| **s′** | **0.7384** | 0.335 | 0.376 | 0.7575 | 0.463 |

- Val box diagnostics (detector s): max recall 0.878; recall < 24 px **0.782** (BoxTeacher 0.678, C2 0.706); 24–48 px 0.872; 48–96 px 0.941; ≥ 96 px 0.930; precision at recall 0.6: 0.847 (BoxTeacher 0.829).
- Val mask AP50 references: BoxTeacher 0.700, C2 0.691.

**Test (439 tiles, single run, frozen `score_coco`; `results/summary_dino_swinl_s0_test.json`):**

| ranking | mask AP50 | AP75 | AP50:95 | box AP50 | box AP75 |
|---|---|---|---|---|---|
| **s′ (val-selected)** | **0.7035** | 0.262 | 0.335 | **0.7391** | 0.401 |
| detector s | 0.6890 | 0.256 | 0.327 | 0.7257 | 0.390 |

**Paired tile bootstrap** (2,000 resamples, test-set sampling only; `results/boot_test_mask_ap50.json`; the bootstrap reproduces every full-set AP exactly):
- D1 − BoxTeacher: **+0.039 [+0.032, +0.049]**
- D1 − C2: +0.041 [+0.032, +0.053]
- D1 − LACE s0: +0.048 [+0.039, +0.060]

All three have p(≤ 0) = 0.000.

**Reading.**
- Beats BoxTeacher (0.664) by a clear margin. That is about 10× LACE's seed sd (0.004), so the pre-registered "extra seeds only if within ~0.01" rule does not trigger.
- Clears the 0.70 target. Comparable to SelvaBox boxes + LACE 8 px masker (A2, 0.711), which used about 3× the training images.
- Box AP50 0.739, vs BoxTeacher 0.703 and SelvaBox 0.767.
- The s′ ranking adds +0.0145 on test over detector s, and +0.003 on val.

**Cost.** Modal metered $72.92 → $106.90 = **$33.98** for D1 in total (smokes $1.4, the two failed launches about $4, training, predictions). The ledger's estimate was $36.6 (it over-counts startup). Under the $75 cap. No containers left running.

**Not done (Tom decides):**
- Updating `TABLE1_CURRENT.md` / paper.tex.
- Seeds 2–3 (not triggered by the rule).
- Leave-one-out ablations.
- The LACE 4 px masker on SelvaBox boxes (supporting row).

**Documentation (2026-11-03, Tom: docs only, not paper.tex):** D1 row added to `results_439/build_table1_current.py` (values asserted) → `TABLE1_CURRENT.{md,json}` regenerated; `mark_superseded.py` re-run (front boxes in README.md, PLAN.md, NATIVE_RASTER.md, PROTOCOL_439.md; banners elsewhere). Pointers added to `dethead/README.md`, `dethead/PROPOSAL_2026-11-02.md` and `tierb/TIERB.md`.

## Pre-registration D1-sparse (2026-11-03, BEFORE any sparse number; Tom: "next priority is sparse 236")
- **Fixed from D1:** the frozen `model_best.pth` (epoch 6), the same 3×3 subtile inference and stitch, the LACE masker `lace4px_simple` and the **s′ ranking selected on val for D1**. Nothing is re-selected.
- Run once on the sparse-canopy holdout (236 tiles, `splits_manifest` ids `sparse`; 0 shared with training, 16 shared with the 439).
- Masker features: local `hunt_z4p_sparse`. GT: `data/tcd_sparse/sparse_gt.json`.
- Scorer: frozen `score_coco` (2048², maxDets 600, crowd arm), as `tab:sparse236`.
- Report s′ (headline) and detector s.

## RESULTS, pre-registration D1-sparse (2026-11-03; `results/summary_dino_swinl_s0_sparse.json`)
Single run on the sparse-canopy holdout (236 tiles), frozen D1 + LACE masker. Masker parity: the new `btmask` sparse path reproduces LACE's published sparse masks byte-identically (29/29 RLE on `tcd_tile_1024` with LACE s0 boxes).

| ranking | mask AP50 | AP75 | AP50:95 | box AP50 | box AP75 |
|---|---|---|---|---|---|
| **s′ (val-selected for D1)** | **0.7349** | 0.390 | 0.396 | 0.7496 | 0.467 |
| detector s | 0.7263 | 0.384 | 0.390 | 0.7414 | 0.456 |

`tab:sparse236` references: LACE (3 seeds) 0.6785 / 0.319 / box 0.674; Detectree2 0.636 / 0.353 / 0.644; Box2Mask Swin-L 0.604 / 0.262 / 0.627. BoxTeacher had no sparse row at the time of this pre-registration (deferred in Tier B). Modal cost $0.31 (metered $106.90 → $107.21).

**Added 2026-11-03 (after the D1-sparse result, Tom):** B2 BoxTeacher on sparse-236, frozen checkpoint, inference only: 0.7206 / 0.3756 / box 0.7391 (`box_supervised_baselines/tierb/TIERB.md`, 'B2 sparse-236'). D1 − BoxTeacher on sparse: +0.014 mask AP50; paired tile bootstrap (236 tiles, 2,000 resamples) +0.0143 [−0.0047, 0.0327], P(≤0) 0.06: **not separated on sparse** (vs LACE CenterNet s0 +0.0446 [0.0255, 0.0613]); full-set AP50 reproduced exactly by the bootstrap's own COCOeval (0.7349 / 0.7206 / 0.6903) (`results/boot_sparse_mask_ap50.json`, `boot_vs_bt.py --split sparse`). Row added to TABLE1_CURRENT and tab:sparse236.

## Diagnostic: ranking ceiling, test 439 (2026-11-03; NOT a table row; `oracle_rank.py`, `results/oracle_rank_dino_swinl_s0_test.json`, log `native_raster/logs/oracle_rank_d1_2026-11-03.log`)
Frozen D1 detections and masks (s′ preds file, 200,148 dets); only the scores change. Frozen score_coco internals (2048, maxDets 600, floor 0, crowd arm).

| scores | mask AP50 | AP75 | AP50:95 | box AP50 | box AP75 |
|---|---|---|---|---|---|
| s′ (deployed) | 0.7035 | 0.262 | 0.335 | 0.739 | 0.401 |
| oracle_iou (score = max mask IoU with a tree; duplicates not penalised) | 0.8555 | 0.435 | 0.457 | 0.866 | 0.529 |
| oracle_tp (Hungarian TP@0.5 above every FP) | **0.8614** | 0.436 | 0.458 | 0.868 | 0.530 |

- Recall ceiling at mask IoU 0.5: 22,308 / 25,705 = **0.868**. That means 0.861 is the AP50 limit for this detection set.
- Of the AP50 gap from 0.7035 to the ceiling, **0.158 is ranking** and 0.007 is the 101-point interpolation and the matching.
- Duplicates (1,182 unassigned dets with mask IoU ≥ 0.5) cost only 0.006 even under a perfect IoU predictor.
- An oracle is not an achievable target. It shows that the headroom is in ranking (scoring), not in recall.

### Gap split: within-tile order vs cross-tile calibration (2026-11-03; `oracle_split.py`, `results/oracle_split_dino_swinl_s0_test.json`, log `native_raster/logs/oracle_split_d1_2026-11-03.log`)
| scores | mask AP50 | Δ vs s′ | AP75 | AP50:95 | box AP50 |
|---|---|---|---|---|---|
| s′ | 0.7035 | — | 0.262 | 0.335 | 0.739 |
| calib_only (s′ order within each tile; score = that tile's oracle precision, by PAV) | 0.761 | +0.058 | 0.272 | 0.358 | 0.785 |
| within_only (each tile's s′ values re-dealt in oracle order) | 0.8175 | +0.114 | 0.378 | 0.418 | 0.831 |
| both (oracle_tp) | 0.861 | +0.158 | 0.436 | 0.458 | 0.868 |

- **About 2/3 of the gap is the order within tiles; about 1/3 is calibration across tiles.** The two are not additive: 0.114 + 0.058 = 0.172 > 0.158.
- Both oracles use test labels. Calibration could be fixed cheaply only if tile-level features predict per-tile precision; that is untested.

### FP breakdown at AP50 (2026-11-03; `fp_breakdown.py`, `results/fp_breakdown_dino_swinl_s0_test.json`, log `native_raster/logs/fp_breakdown_d1_2026-11-03.log`)
- Greedy matching in s′ order. Own 101-point AP reproduces the frozen 0.7035 exactly.
- Counts: 22,308 TP, 103,819 FP, 74,021 crowd-ignored.
- Gain = AP50 with that type's FPs deleted, minus 0.7035. The gains are not additive.

| FP type | n | n with s′ ≥ median TP (0.179) | gain |
|---|---|---|---|
| **bg (m < 0.1, outside canopy)** | 94,675 | 1,015 | **+0.106** |
| part (≥ 80% inside one crown) | 2,252 | 172 | +0.009 |
| near (0.3 ≤ m < 0.5; 600 with box IoU ≥ 0.5) | 2,024 | 163 | +0.0085 |
| merge (covers ≥ 2 crowns) | 1,495 | 65 | +0.004 |
| dup | 1,161 | 15 | +0.002 |
| poor (0.1 ≤ m < 0.3) | 2,212 | 30 | +0.002 |

- **Reading:** the ranking gap is a classification problem (tree vs not-tree outside the canopy polygons), not a localisation problem. Localisation-type FPs together are worth about 0.026.
- Not yet checked: what the high-scoring bg FPs are. They could be unlabelled trees, shrubs, shadows, or dets near the canopy edge (canopy fraction just under 0.5).

### What the background FPs are (2026-11-03; `fp_bg.py`, `results/fp_bg_stats_dino_swinl_s0_test.json`, contact sheet `results/fp_bg_top100_dino_swinl_s0_test.jpg`, per-det table `results/fp_det_table_dino_swinl_s0_test.npz`)
- **Not the canopy-edge quirk.** Of the 1,015 bg FPs with s′ ≥ the median TP, 972 have canopy fraction < 0.01; only 4 lie in [0.4, 0.5).
- **Not a size effect.** Median √area: 32 px for these bg FPs, 38.5 px for TPs (q10–q90: 17–55 vs 15.5–78).
- **Concentrated.** 181 tiles hold them. The top tile, tcd_val_tile_193, has 77; the next are 372 (44) and 375 (40).
- **Contact sheet, top 100 by s′ (Claude's eyeball, PROVISIONAL, not an audit):**
  - most are real vegetation with no GT nearby: landscaped shrubs and small trees along roads and buildings, young palms and agave-like plants in plantation rows, a few isolated crowns;
  - only ~5–10 look like non-vegetation (grass texture, blur).
- **Implication:** most of the +0.106 is the annotators' policy on what counts as a "tree" (unlabelled shrubs, small ornamentals, young palms), not detector confusion with non-trees. A reranker can recover only the part of that policy that is consistent in train GT. The oracle ceiling overstates the achievable gain.
- An audit by Tom of the 100 crops (tree / shrub / not vegetation) would put a number on this.

## Pre-registration R1 (2026-11-03, BEFORE any reranker number): a small learned reranker on frozen D1
Tom: "Can we train that small reranker locally to see if it helps? We might be slightly over-detecting small shrubs."
Script `rerank.py`.
- **Frozen:** the D1 detections, the LACE masks and s′ (val: `feat_cache/dino_swinl_s0_val_mask.json`; test: `preds_dino_swinl_s0_test_sprime_r2048.json`). Only the score changes.
- **Training data:** the 108 val tiles only. No D1 train-tile predictions exist, and DINO's scores on its own train tiles would be overconfident.
  - Val was already used for D1's epoch selection and the s vs s′ choice; this is a further reuse of val, noted.
- **Labels: boxes only** (LACE's supervision), with no GT mask ever read. In s′ order, greedy, a det is 1 if its BOX has IoU ≥ 0.5 with an unmatched GT tree box (the bbox of the polygon), else 0.
  - Dets with ≥ 50% of their mask inside the canopy union are crowd-ignored and dropped from training (LACE's canopy rule).
- **Features (inference-time only, per det, 24):**
  - Scores: s, s′, log(s′/s).
  - Geometry: log box w, log box h, log aspect, log mask area, mask area / box area, distance to tile border.
  - Context within the tile: log1p(#dets with s ≥ 0.3), mean s of the tile's top 50, s′ rank / n, max box IoU with a higher-s det, #dets with s ≥ 0.3 whose centre is within 1.5 × max(w, h).
  - Appearance (RGB at 2048, inside the mask): mean R, G, B; ExG = mean (2G−R−B)/(R+G+B); luminance std; mean |∇L|.
  - Appearance (ring = 0 < distance ≤ max(3, 0.25√area) px outside the mask): ring L − inside L, ring ExG − inside ExG, min − max quadrant ring L (shadow asymmetry), min quadrant ring L.
- **Models (fixed hyperparameters, no tuning):**
  - (a) logistic regression;
  - (b) MLP 24→64→64→1, ReLU, dropout 0.1.
  - Both: standardised on the training fold, AdamW (lr 1e-3, weight decay 1e-3), 30 epochs, batch 1024, BCE, seed 0, torch CPU.
  - Score = predicted probability.
- **Validation:** 5-fold tile-grouped CV on the 108 val tiles (folds by a sorted-id round-robin), out-of-fold scores, full-pipeline val mask AP50 via `score_lite` (crowd, maxDets 600, floor 0). Baseline: s′ (0.7384).
  - Model choice: the higher OOF AP50; ties within 0.002 → (a).
  - Val-only feature-group ablations (drop appearance; drop context; score+geometry only) are reported for understanding and never go to test.
- **Gate to test:** the chosen model's OOF gain is ≥ +0.005 AND its paired tile-bootstrap 95% CI on val (2,000 resamples, seed 0) excludes 0.
  - If the gate passes: apply the 5 fold models' mean probability to the test 439 ONCE; frozen `score_coco` (2048, maxDets 600, floor 0, crowd arm); paired tile bootstrap vs s′.
  - If not: stop, test never touched.
- **Not a Table 1 change without Tom.**

## RESULTS, pre-registration R1 (2026-11-03; `results/rerankR1_cv.json`, log `native_raster/logs/rerankR1_cv_2026-11-03.log`): GATE FAILED, test not touched
- Val: 51,341 dets, of which 6,143 are box-positive, 29,195 negative and 16,003 crowd-ignored.
- The s′ baseline reproduces 0.7384.

| OOF val mask AP50 | AP50 | gain vs s′ |
|---|---|---|
| logreg (all 24) | 0.7411 | +0.0027 |
| mlp (all 24) | 0.7414 | +0.0030 |
| mlp, no appearance (ablation) | 0.7443 | +0.0059 |
| mlp, no context (ablation) | 0.7374 | −0.0010 |
| mlp, score + geometry (ablation) | 0.7399 | +0.0015 |

- Chosen: logreg (the MLP is within 0.002). Paired tile bootstrap vs s′: +0.0027, 95% CI [−0.0021, +0.0076], p(≤ 0) 0.13. **Gate failed** (needs ≥ +0.005 and a CI excluding 0). **The test set was not touched.**
- **Reading:**
  - The appearance features hurt: dropping them gives +0.0059. With 108 tiles they most likely fit per-tile colour rather than "shrub vs tree".
  - Context carries what signal there is.
  - The no-appearance row is an ablation and is NOT eligible for test under this pre-registration; promoting it would be selection on val after the fact.
- **Conclusion:** with only val for training, a learned reranker does not recover a meaningful part of the oracle gap. This is consistent with the bg FPs being an annotation-policy effect, not a learnable visual one. Not tested here: more training data (D1 predictions on the 792 train tiles, cross-fitted).

## Pre-registration R2 (2026-11-03, after R1; POST-HOC variant, declared as such): the R1 reranker without appearance features
Tom: "MLP without colour texture is +0.006? Let's pull on that thread." `mlp_no_app` was one of five val ablations in R1, so its +0.0059 is selected on val and may be noise. The test set is still untouched; it stays a one-shot held-out set for this single variant.
- **Model, fixed:** R1's MLP on the 14 non-appearance features (score, geometry, context), with R1's hyperparameters, labels and val features (`feat_cache/rerankR1_val.npz`). Also reported, not eligible: `logreg_no_app`.
- **Robustness on val:** 5 seeds (0–4) × 3 fold assignments (R1's round-robin, plus two random tile permutations, seeds 1 and 2) = 15 OOF runs. Paired tile bootstrap vs s′ on the R1 configuration (seed 0, round-robin).
- **Gate to test:** mean gain over the 15 runs ≥ +0.005, AND ≥ 13/15 runs > 0, AND the bootstrap CI excludes 0.
- **If the gate passes:** test ONCE with the mean over the 25 models (5 seeds × 5 round-robin folds); frozen score_coco; paired tile bootstrap vs s′. Reported as "post-hoc R2", never as pre-registered in R1. Not a Table 1 change without Tom.

## RESULTS, pre-registration R2 val gate (2026-11-03; `results/rerankR2_cv.json`, log `native_raster/logs/rerankR2_cv_2026-11-03.log`): GATE PASSED
- **15 OOF runs** (5 seeds × 3 fold assignments): gain vs s′ **+0.0061 ± 0.0013** (mean ± sd); min +0.0039, max +0.0081; 15/15 positive.
  - By fold assignment: rr +0.0051 to +0.0061; perm1 +0.0039 to +0.0065; perm2 +0.0067 to +0.0081.
- **Bootstrap** (rr, seed 0): +0.0059 [+0.0007, +0.0114], p(≤ 0) 0.016.
- logreg_no_app: +0.0018 (not eligible).
- **Caveat:** the variant was selected from R1's val ablations, so the val gain is optimistic. The one-shot test below is the unbiased number. The test run was launched 2026-11-03 (`rerank2.py test`).

### R2 test, first run INVALID: a feature bug (2026-11-03; kept as `results/*rerankR2*_bug.json`, `feat_cache/rerankR1_test_bug.npz`, log `..._bug.log`)
- The first run gave mask AP50 0.6869, −0.0166 [−0.030, −0.002] vs s′.
- **Bug:** `test()` built the test features from the s′ preds file, whose `scores` field holds s′. So feature `s` (detector score) was s′ on test; it was the true DINO score on val (verified: test X[:,0] == X[:,1]; val not).
  - The context features built on s (log_nhi, tile_top50, max_iou_higher, n_nbr) were wrong too, so the test inputs were out of distribution.
  - It also existed in R1's `test()`, which never ran.
- **Fix:** `s` is taken from `preds_..._detscore_r2048.json`; the boxes are asserted identical, in the same order.
- Nothing else changed: same models (seeds fixed, deterministic refit), same procedure. The pre-registered run is the corrected one.
- **Disclosure:** the test set has now been scored twice for R2. The first run did not implement the pre-registration, and no choice was made from its number.

## RESULTS, R2 test (corrected run, 2026-11-03; `results/rerankR2_test.json`, `results/scored_dino_swinl_s0_test_rerankR2_r2048.json`, log `native_raster/logs/rerankR2_test_2026-11-03.log`)

| test 439 | mask AP50 | AP75 | AP50:95 | box AP50 | box AP75 |
|---|---|---|---|---|---|
| s′ (D1 headline) | 0.7035 | 0.262 | 0.335 | 0.739 | 0.401 |
| **R2 reranker (post-hoc)** | **0.7151** | 0.269 | 0.342 | 0.7485 | 0.412 |

- **Paired tile bootstrap R2 − s′:** **+0.0116 [+0.0073, +0.0159]**, p(≤ 0) 0.000.
- Test feature check: s ≠ s′ (the bug is fixed).
- **Status:**
  - POST-HOC: the variant was chosen from R1's val ablations.
  - The test set was scored twice; the first run was invalid (bug, above).
  - Single D1 seed; the reranker was trained on the 108 val tiles, which were also used for D1's selection.
- Not in TABLE1_CURRENT or paper.tex (Tom decides).
- A clean confirmation is still available: the frozen R2 models applied once to sparse-236 (pre-register before running).

### Val-only diagnostic: does R2 need s′? (2026-11-03; `rerank_nosp.py`, `results/rerank_nosp_val.json`, log `native_raster/logs/rerank_nosp_val_2026-11-03.log`; never run on test)
- The no-s′ MLP drops every s′-derived feature: sp, log_sp_s, and rank_frac (the s′ rank within the tile). It keeps the detector s, geometry (including mask area and fill) and context.
- 5 seeds × 3 fold assignments.

| val mask AP50 | mean | gain vs s′ |
|---|---|---|
| detector s | 0.7356 | −0.0028 |
| s′ | 0.7384 | — |
| MLP without s′ | ≈ 0.7408 | +0.0024 ± 0.0016 |
| R2 (MLP with s′) | ≈ 0.7445 | +0.0061 ± 0.0013 |

- Without s′ − R2: **−0.0037 ± 0.0008; worse in all 15 runs.**
- **Reading:**
  - s′ carries ranking information that the detector score plus context does not reproduce, so the masker's ranking role survives.
  - R2 is best read as s′ plus a context correction, not as a replacement for s′.
  - Without s′, an MLP beats s′ by only about +0.002.

## Pre-registration E1 (2026-11-03, BEFORE any number): a scoring head on DINO's own decoder features + s′
Tom: wants the "direct" version of the reranker. No hand-picked features, no detector retraining.
- **Extraction (Modal, 1 × A100-80GB; `dino_modal.py embed(split)`):**
  - Frozen `model_best.pth` (epoch 6), the same 3×3 windows and the same stitch.
  - A forward hook on `class_embed[L−1]` (DINO's final-decoder-layer scoring layer, L = 6) captures its 256-d input per query. The top 300 queries are re-derived from its logits exactly as `DINO.inference` does; the scores are asserted equal to the model's output.
  - The stitch is re-run, carrying the vectors. Saved per tile as float16 `[n_kept, 256]` at `tcd04-tierb-vol:runs/dino_swinl_s0/emb_<split>/<tid>.npy`, with a per-tile parity line, resumable per tile.
  - Splits: val, test, sparse. Smoke: 2 val tiles first.
- **Parity gate:** each tile's stitched boxes and scores must equal `preds_<split>.json` (same count, max |Δbox| ≤ 0.01 px, max |Δs| ≤ 1e-5). Any failure is reported. The run aborts if more than 5 tiles fail in a split. Embeddings are used only for tiles that pass.
- **Heads:** input = [256-d vector, log s′], standardised on the training fold. Labels, CV and training are as in R1/R2: val box labels (`feat_cache/rerankR1_val.npz` y), AdamW lr 1e-3, weight decay 1e-3, 30 epochs, batch 1024, BCE, torch CPU.
  - (a) logistic regression: refits DINO's scoring layer, with s′ as one extra input;
  - (b) MLP, one hidden layer of 64 (ReLU, dropout 0.1);
  - (c) MLP, two hidden layers of 64 (as R2).
- **Selection on val:** 5 seeds × 3 fold assignments (R2's), mean OOF mask AP50 gain vs s′. Highest mean wins; any simpler head within 0.002 of it wins instead (a < b < c).
- **Gate to test:** the chosen head's mean gain ≥ +0.005, ≥ 13/15 runs > 0, and the paired tile-bootstrap CI vs s′ (rr folds, seed 0) excludes 0. R2's val gain (+0.0061) is reported alongside for comparison.
- **If the gate passes:** test ONCE with the mean of the 25 models (5 seeds × 5 rr folds); frozen score_coco; paired bootstrap vs s′ and vs R2. Then sparse-236 ONCE with the same frozen models, as confirmation.
- Not a Table 1 change without Tom. Modal budget cap for E1: $10.

### E1 smoke PASS (2026-11-03; `runs/dino_swinl_s0/emb_val_smoke/`, call fc-01M6G5R31QQQBXG240VWWDR0E6)
- 2 val tiles; the hook fired exactly once per window (L = 6; 7 class_embed, the last being the two-stage encoder head); the topk scores equal the model output.
- **Parity exact:** max |Δbox| 0, max |Δs| 0.
- About 10 s per tile. The vectors are (600, 256), finite, all rows distinct.
- Added after the smoke: `embed` also packs `emb_<split>.npz` on the volume, because downloading a directory with `modal volume get` was unreliable locally.
- **Launched 2026-11-03 18:45:** val fc-01M6G5W8H3BYYM5HG5RR3B4YWZ, test fc-01M6G5W9N48WQ72RNG31ZGEPFE, sparse fc-01M6G5WATGCQFM0X1805PBRPPY.
- Revised cost estimate: about 2.2 A100-hours ≈ $7–8 (the earlier "$1–2" was too low); E1 cap $10.

## RESULTS, pre-registration E1 (2026-11-03; `results/e1_cv.json`, log `native_raster/logs/e1_cv_2026-11-03.log`): GATE FAILED, test not touched
- Extraction: val 108/108, sparse 236/236 and test (running at write-time) all at exact parity: max |Δbox| 0, max |Δs| 0.
- The npz packing step was not served (warm container ran the pre-redeploy code). The per-tile .npy files are used instead.

| head (input: 256-d decoder vector + log s′) | val gain vs s′, 15 runs (mean ± sd) | runs > 0 |
|---|---|---|
| (a) logistic (refit DINO's scoring layer) | −0.0078 ± 0.0015 | 0/15 |
| **(b) MLP 1×64 (chosen)** | **+0.0038 ± 0.0012** | 15/15 |
| (c) MLP 2×64 | −0.0064 ± 0.0018 | 0/15 |
| R2 reference (hand-crafted features) | +0.0061 ± 0.0013 | 15/15 |

- **Bootstrap (b) vs s′:** +0.0038 [−0.0072, +0.0152], p(≤ 0) 0.26.
- **Gate failed:** the mean is < +0.005 and the CI includes 0.
- **Reading:**
  - Refitting DINO's scoring layer on val is worse than s′. So is the deeper head, consistent with overfitting 257 inputs on 108 tiles.
  - The one-hidden-layer head is positive but smaller than R2 and much less stable across tiles (CI width 0.022 vs R2's 0.011).
  - DINO's per-query vector carries context only within its 1024-px window. R2's whole-tile context features appear to be what transfers.
- **Conclusion:** the "direct" head does not beat the hand-crafted R2 at this data scale. The test and sparse embeddings are kept (`feat_cache/e1_emb/`, volume `emb_*`) for any later use.
- Modal cost: three 1 × A100 calls of about 5, 12 and about 20 min plus the smoke, ≈ $2.5 (to be confirmed from the ledger).
- **E1 extraction complete (2026-11-03 19:07):** test 439/439 at exact parity. All three splits are pulled to `feat_cache/e1_emb/{val,test,sparse}/`.
- **Cost:** the ledger lines for val and sparse were lost (three containers appended concurrently to one `spend.jsonl` on the volume, so the last commit won). From billed seconds: smoke $0.25 + val ≈ $0.55 + sparse ≈ $0.92 + test $1.47 ≈ **$3.2**.
- One idle warm container (ta-01M6G5W8T6J1GQSQ8TWXGC2F3R) was stopped by hand.

## Pre-registration R2-sparse (2026-11-03, BEFORE any number): confirmation of R2 on the sparse-canopy holdout
Tom: "Yes" (run R2 once on sparse-236, locally).
- **Frozen:**
  - R2's 25 models: the R2 MLP on 14 features, 5 seeds × 5 rr folds, refit deterministically on the val cache exactly as in `rerank2.py test`;
  - D1 sparse dets and masks (`preds_dino_swinl_s0_sparse_{detscore,sprime}_r2048.json`; boxes asserted identical).
  - Nothing is re-selected.
- **Data:** 236 sparse tiles; imagery at `data/tcd_sparse/test/`. 0 tiles shared with val (checked). 16 shared with the test 439, which was already used for R2.
- **Features:** as in R2 (the appearance features are computed but not used); detector s from the detscore file.
- **Score:** frozen score_coco (2048, maxDets 600, floor 0, crowd arm) on `data/tcd_sparse/sparse_gt.json`. Paired tile bootstrap vs s′ (2,000 resamples, seed 0).
- **Reported:**
  - headline: all 236;
  - secondary: the 220 tiles not in the 439 (bootstrap only).
- **Confirmation criterion:** R2 − s′ > 0 with a 95% CI excluding 0 on all 236. Any other outcome is reported as "not confirmed".
- Run once.

## RESULTS, R2-sparse (2026-11-03; `results/rerankR2_sparse.json`, log `native_raster/logs/rerankR2_sparse_2026-11-03.log`): NOT CONFIRMED

| sparse-236 | mask AP50 | AP75 | AP50:95 | box AP50 |
|---|---|---|---|---|
| s′ (D1-sparse headline) | 0.7349 | 0.390 | 0.396 | 0.7496 |
| R2 (frozen) | 0.7351 | 0.393 | 0.3965 | 0.7493 |

- **R2 − s′:** +0.0002 [−0.0043, +0.0054], p(≤ 0) 0.45. On the 220 tiles not in the 439: +0.0007 [−0.0037, +0.0063].
- **Not confirmed.**
- **Reading:**
  - R2's +0.0116 on the 439 does not carry over to the open-canopy holdout. The context correction learned on the 108 (mostly closed-canopy) val tiles is distribution-specific, or the test gain was partly favourable sampling.
  - Combined with E1 failing and R1 failing, the learned rerankers do not give a robust improvement over s′.
- **Decision recommended:** s′ remains the system score. R2 is not a paper row.

## Pre-registration R3 (2026-11-04, BEFORE any R3 number; POST-HOC, declared as such): streamline R2 by ablation, local compute only
Tom: "ablate through this R2 to cut away all the inputs and machinery that doesn't help, autonomously, overnight, and local compute only. AP50 is paramount, try to get a streamlined R2 that uses the minimum number of inputs. On the side, try to find any opportunity to improve AP50:95 also." Script `rerank3.py`; every config result in `results/rerankR3_ledger.jsonl` (also the resume cache); log `native_raster/logs/rerankR3_2026-11-04.log`.
- **Context:** R2 passed its val gate (+0.0061) and gained +0.0116 on test but was NOT confirmed on sparse-236 (+0.0002). Test has already been scored for R2 (twice, one run invalid); R3 adds one more post-hoc test look and one sparse look. Any R3 number is post-hoc and is not a Table 1 change without Tom.
- **Evaluator:** `fasteval.py`, a re-implementation of the score_lite crowd-arm COCOeval (mask IoU 0.50:0.95, maxDets 600, 101 recall points) with det-vs-GT IoUs precomputed. Parity on val before any R3 number: s′, detector s and a reversed order all match score_lite exactly to 4 dp (`results/fasteval_parity_val.json`, log `native_raster/logs/fasteval_parity_2026-11-04.log`).
- **Unit of evidence:** a config is scored on 15 OOF val runs (5 seeds × fold sets rr/perm1/perm2; R2's protocol), gain = run mask AP50 − s′ (0.7384). The R2 reference must reproduce `rerankR2_cv.json` run-for-run (determinism check).
- **Stage 1, machinery (14 features):** model ∈ {logreg, MLP 1×8, 1×16, 1×64, 2×64 (R2)}; on 2×64 also dropout 0, epochs 10/60, weight decay 0/1e-2, crowd dets as negatives.
- **Stage 2, inputs:** greedy backward elimination 14 → 1 features for each of the five models; at each step drop the feature whose removal leaves the highest mean gain.
- **Selection rule (fixed now):** admissible = box-IoU-0.5 labels, mean gain ≥ R2's mean − 0.001, mean gain ≥ +0.005, ≥ 13/15 runs > 0 (R2's gate). Rank by fewest features, then simplest model (logreg < 1×8 < 1×16 < 1×64 < 2×64), then default training settings, then higher mean.
- **Guard against selecting on noise:** the top-ranked candidate is re-run on 15 FRESH runs (seeds 5–9 × new fold permutations 3/4/5) beside R2 on the same runs; it must stay within 0.001 of R2, mean ≥ +0.005 and ≥ 13/15 positive. If it fails, the next candidate is tried, in order, until one passes.
- **Final, ONCE:** the confirmed config, 25-model mean (5 seeds × rr folds, val-trained), applied to the test 439 and sparse-236; frozen score_coco; paired tile bootstrap vs s′ and vs R2. R3 is "confirmed" only if its sparse-236 CI vs s′ excludes 0 (R2's criterion). Reported whatever the outcome.
- **AP50:95 side (val only; nothing goes to test without Tom):** (a) box-IoU-regression and multi-threshold box labels for R2 and the chosen R3; (b) score = s·(s′/s)^a for a ∈ 0…1.75 (a = 1 is s′); (c) mask dilation/erosion by ±1–3 px at 2048 under s′ ranking. A candidate is flagged to Tom if val AP50:95 rises ≥ +0.003 without lowering val AP50.

## Pre-registration S2 (2026-11-04, BEFORE any S2 number): val-only search for AP50 / AP50:95 gains without a learned model
Tom (during R3): "If you can find an improvement to either ap50 or 50:95 that will be very welcome also." Script `side2.py`, results `results/side2_val.json`, log `native_raster/logs/side2_2026-11-04.log`. Val only, local; nothing goes to test without Tom. Already known before S2 (R3 side, `results/rerankR3_side_ap5095.json`): uniform mask dilation/erosion by 1–3 px lowers every metric; box-IoU-regression / multi-threshold labels do not raise the reranker's AP50:95; s·(s′/s)^a peaks near a = 0.5 on val (AP50 0.7406, AP50:95 0.3783 vs s′ 0.7384 / 0.3763).
- **Closed-form score:** s·(s′/s)^a·(1 − max_iou_higher)^b, a ∈ 0…1.5 (step 0.125), b ∈ 0…3 (step 0.25); a = 1, b = 0 is s′. Reported: the full-val best (optimistic) and a 5-fold tile CV of the grid choice (choose on 4/5 by AP50, score the held-out 1/5; fold sets rr/perm1/perm2).
- **Mask post-processing under s′ ranking:** fill holes, largest connected component, convex hull, Gaussian boundary smoothing σ = 1/2/3 px (threshold 0.5), clip to the detector box (margin 0/2 px).
- **Diagnostic:** matched (mask IoU ≥ 0.5) pred/GT area ratio and IoU by GT size.
- **Flag to Tom:** CV mean gain ≥ +0.003 AP50 (closed form), or val AP50:95 ≥ +0.003 without lowering val AP50 (post-processing).

## RESULTS, R3 (2026-11-04; `results/rerankR3_{ledger.jsonl,paths.json,select.json,confirm.json,final.json,side_ap5095.json}`, log `native_raster/logs/rerankR3_2026-11-04.log`): CONFIRMED on sparse-236 (post-hoc)
- **Determinism:** R2 reference reproduced `rerankR2_cv.json` run-for-run.
- **Stage 1, machinery (14 features, val 15-run mean gain vs s′):**
  - Weight decay 0/1e-3/1e-2 makes no difference (+0.0061).
  - MLP 1×64 ties R2's 2×64 (+0.0059).
  - 1×16 +0.0021; 1×8 −0.0005; logreg −0.0005.
  - Dropout 0: +0.0054; epochs 10: +0.0053; epochs 60: +0.0050.
  - Crowd detections as negatives: **−0.0146** (crowd-ignore is essential).
- **Stage 2, elimination:**
  - Inputs that survive to the end in every capable model: `s`, `log_sp_s` (= log(p̄·m)), `max_iou_higher`, plus one size term.
  - The tile-level context features (`log_nhi`, `tile_top50`) are dropped first in both 1×64 and 2×64; removing them RAISES val gain (to +0.0075–0.0079).
  - 145 configs admissible.
- **Selected (rule, fewest features → simplest model):** MLP 1×64 on 4 inputs — detector `s`, `log_sp_s`, `log_h` (log box height), `max_iou_higher` (max box IoU with any higher-s detection).
  - Val: +0.0053 ± 0.0013, 15/15 runs positive.
  - Fresh randomness (seeds 5–9 × perm3–5): +0.0059 vs R2 +0.0066 on the same runs; 15/15 → PASS.
  - Val bootstrap (rr, seed 0) vs s′: +0.0026 [−0.0009, +0.0059]. R2's gate required this CI to exclude 0; R3's pre-registered rule did not. Disclosed.
- **Final, run once (25-model mean):**

| | mask AP50 | AP75 | AP50:95 | box AP50 | R3 − s′ AP50 [95% CI] | R3 − R2 AP50 [95% CI] |
|---|---|---|---|---|---|---|
| test 439, s′ | 0.7035 | 0.2615 | 0.3347 | | | |
| test 439, R2 | 0.7151 | 0.2691 | 0.3416 | 0.7485 | | |
| test 439, **R3** | **0.7131** | 0.2685 | 0.3406 | 0.7471 | +0.0097 [+0.0069, +0.0124] | −0.0020 [−0.0048, +0.0007] |
| sparse-236, s′ | 0.7349 | 0.390 | 0.396 | 0.7496 | | |
| sparse-236, R2 | 0.7351 | 0.393 | 0.3965 | 0.7493 | | |
| sparse-236, **R3** | **0.7432** | 0.3964 | 0.4013 | 0.7570 | **+0.0083 [+0.0037, +0.0152]** | +0.0081 [+0.0048, +0.0119] |

- On the 220 sparse tiles not in the 439: +0.0089 [+0.0041, +0.0163].
- **Reading:** dropping the tile-context and most geometry inputs made the reranker transfer to open canopy; R2's failure on sparse was most likely its tile-level context features (`log_nhi`, `tile_top50`), which encode closed-canopy density. R3 keeps the test gain (+0.010) and adds +0.008 on sparse.
- **Caveats:**
  - Post-hoc: the third learned reranker tried; test and sparse were each looked at for R2 before R3 was designed. Selection used val only, and R3 was scored once on each.
  - The val bootstrap CI includes 0.
  - Not a Table 1 change without Tom.
- **AP50:95 side (val):**
  - Uniform dilation/erosion ±1–3 px lowers every metric (+1 px: AP50:95 0.362; −1 px: 0.346; s′ 0.3763). Masks are small (median equivalent diameter about 28 px at 2048), so 1 px moves area by about 15%.
  - Box-IoU-regression and multi-threshold labels do not raise AP50:95 over box-0.5 labels (R2 0.3807/0.3811/0.3806; R3 0.3810/0.3810/0.3803).
  - The rerankers themselves lift val AP50:95 by about +0.0045 (0.381 vs 0.3763); on test +0.006, on sparse +0.005.

## Pre-registration S2b (2026-11-04, after S2's first mask results; follow-up, declared as such)
S2 so far (val, s′ ranking): Gaussian boundary smoothing σ = 1 / 2 / 3 → AP50:95 0.3784 / **0.3802** / 0.3767 vs 0.3763 (AP50 0.7403 / 0.7409 / 0.7363 vs 0.7384); hole fill, largest component and box clip are neutral; convex hull hurts. Matched small crowns (< 24 px) are 11% too large on median. Script `side2b.py`, results `results/side2b_val.json`, log `native_raster/logs/side2b_2026-11-04.log`.
- **Masks:** σ = 1.5, 2.5; size-gated σ (2 at ≥ 24 px equivalent diameter, else 1); 1 px erosion of < 24 px masks only.
- **Stacking:** masks ∈ {none, blur2, best blur by val AP50:95} × ranking ∈ {s′, closed form (a = 0.625 / 0.75, b = 0.5), R3 15-run OOF mean}.
- Val only; nothing goes to test without Tom. The σ choice is a 1-D val sweep (5 values); report it as such.

## RESULTS, S2 + S2b (2026-11-04; `results/side2_val.json`, `results/side2b_val.json`, logs `native_raster/logs/side2_2026-11-04{_part1,}.log`, `side2b_2026-11-04.log`): val only, test NOT touched
- **Bug fixed mid-run:** `pycocotools.mask.area` on a list > 255 masks overflows uint8 under numpy 2 (ratio diagnostic only; per-mask calls now). It does not affect score_lite/score_coco (per-mask area) or fasteval (iou only, parity-checked). Part 1 log kept.
- **Closed-form score s·(s′/s)^a·(1 − max_iou_higher)^b (no model):**
  - Full-val best a = 0.75, b = 0.5: AP50 0.7425 (+0.0041), AP50:95 0.3780 (+0.0017).
  - Tile-CV of the grid choice: +0.0044 / +0.0042 / +0.0041 AP50 (rr / perm1 / perm2) and +0.0016–0.0020 AP50:95.
  - The choice is stable: a 0.5–0.75 (mostly 0.625), b = 0.5 in 14/15 folds.
  - **Flagged** (≥ +0.003 CV AP50).
- **Matched area ratio by GT size (pred/GT, median):** < 24 px 1.11; 24–48 px 1.02; 48–96 px 1.015; ≥ 96 px 1.014. Median IoU 0.71 / 0.75 / 0.79 / 0.82.
- **Mask post-processing, s′ ranking (AP50 / AP75 / AP50:95; none = 0.7384 / 0.3345 / 0.3763):**
  - Gaussian boundary smoothing, threshold 0.5: σ 1 → 0.7403 / 0.3396 / 0.3784; σ 1.5 → 0.7401 / 0.3426 / 0.3802; **σ 2 → 0.7409 / 0.3437 / 0.3802**; σ 2.5 → 0.7382 / 0.3445 / 0.3793; σ 3 → 0.7363 / 0.3425 / 0.3767; size-gated (2 / 1) → 0.7406 / 0.3428 / 0.3800.
  - Fill holes, largest component, clip to box (0 / 2 px): neutral (masks already inside boxes).
  - Convex hull: 0.3685. 1 px erosion of < 24 px masks: 0.3647.
  - **σ = 2 flagged** (+0.0039 AP50:95, +0.0025 AP50).
- **Stacking (val):**

| masks | ranking | AP50 | AP75 | AP50:95 |
|---|---|---|---|---|
| as is | s′ | 0.7384 | 0.3345 | 0.3763 |
| as is | closed form (0.625, 0.5) | 0.7424 | 0.3370 | 0.3783 |
| as is | R3 (15-run OOF mean) | 0.7437 | 0.3434 | 0.3810 |
| σ 2 | s′ | 0.7409 | 0.3437 | 0.3802 |
| σ 2 | closed form (0.625, 0.5) | 0.7451 | 0.3462 | 0.3823 |
| σ 2 | **R3** | **0.7463** | **0.3520** | **0.3849** |

- **Reading:** smoothing and re-ranking are additive. σ = 2 + R3 gives +0.008 AP50 and +0.009 AP50:95 on val. LACE masks are slightly ragged at the boundary relative to smooth hand-drawn GT polygons; a 2 px Gaussian fixes part of it at no cost.
- **Next (needs Tom):** one pre-registered test + sparse look at σ = 2 masks (with s′, and with R3), and optionally the closed form.

## Pre-registration S2-final (2026-11-04, Tom: "Yes, of course"; BEFORE any test/sparse number for these rows)
ONE look, test 439 and sparse-236, at the val-selected S2 changes on frozen D1 boxes (`s2_final.py`, `results/s2_final.json`, log `native_raster/logs/s2_final_2026-11-04.log`):
- **Rows:**
  - `blur2_sprime`: masks Gaussian-smoothed σ = 2 px at 2048 (threshold 0.5), s′ ranking;
  - `blur2_R3`: the same masks with the frozen R3 scores (already applied once in R3-final);
  - `closed`: masks as is, s·(s′/s)^0.625·(1 − max_iou_higher)^0.5;
  - `blur2_closed`: the two combined.
- **Scoring:** frozen score_coco (2048, maxDets 600, floor 0, crowd arm). Paired tile bootstrap (2,000, seed 0) on AP50 AND AP50:95, computed with fasteval after asserting it reproduces score_coco on every row.
- **Comparisons:** each row vs s′; blur2_R3 vs R3; blur2_closed vs closed.
- **Criterion for "confirmed" (per change, per split):** the 95% CI excludes 0 in its favour on that metric. Both splits are reported whatever the outcome. Not a Table 1 change without Tom.

## RESULTS, S2-final (2026-11-04; `results/s2_final.json`, `results/scored_dino_swinl_s0_{test,sparse}_{blur2_sprime,blur2_R3,closed,blur2_closed}_r2048.json`, log `native_raster/logs/s2_final_2026-11-04.log`)
fasteval reproduced score_coco on all 6 rows of both splits. Δ = row − reference [95% CI].

| | test AP50 | test AP50:95 | sparse AP50 | sparse AP50:95 |
|---|---|---|---|---|
| s′ (reference) | 0.7035 | 0.3347 | 0.7349 | 0.3962 |
| σ 2 masks + s′ | 0.7038 (+0.0003 [−0.0006, +0.0026]) | 0.3369 (**+0.0022 [+0.0014, +0.0034]**) | 0.7342 (−0.0007 [−0.0028, +0.0008]) | 0.3966 (+0.0003 [−0.0018, +0.0019]) |
| closed form | 0.7040 (+0.0005 [−0.0013, +0.0021]) | 0.3342 (−0.0005) | 0.7349 (+0.0001) | 0.3959 (−0.0003) |
| σ 2 + closed form | 0.7043 (+0.0008) | 0.3364 (+0.0017 [+0.0006, +0.0036]) | 0.7342 (−0.0007) | 0.3961 (−0.0001) |
| R3 (reference 2) | 0.7131 | 0.3406 | 0.7432 | 0.4013 |
| σ 2 + R3 | 0.7136 (vs R3 +0.0004 [−0.0004, +0.0027]) | 0.3430 (vs R3 **+0.0024 [+0.0016, +0.0035]**) | 0.7426 (vs R3 −0.0006) | 0.4018 (vs R3 +0.0005 [−0.0015, +0.0020]) |
| σ 2 + R3 vs s′ | **+0.0101 [+0.0072, +0.0134]** | **+0.0083 [+0.0069, +0.0102]** | **+0.0077 [+0.0023, +0.0148]** | **+0.0056 [+0.0019, +0.0099]** |

- **Closed form:** NOT confirmed on either split (val CV +0.004 did not transfer).
- **σ 2 smoothing:**
  - Confirmed for AP50:95 on test (+0.002, CI excludes 0, with s′ and with R3); NOT confirmed on sparse (+0.0003 / +0.0005).
  - AP50 neutral on both.
  - A small, closed-canopy-only boundary gain.
- **R3 remains the only change confirmed on both splits.** σ 2 + R3 vs s′ is significant on both, but the sparse part is R3's.
- Not a Table 1 change without Tom.

## Diagnostic: LACE vs SAM 3 masks on the same D1 boxes (2026-11-04; `mask_diag.py`, `results/mask_diag_lace_vs_sam3_test.json`, log `native_raster/logs/mask_diag_2026-11-04.log`; test 439, diagnostic only)
23,803 detections paired to a GT tree by box IoU ≥ 0.5 (the same pair for both maskers). Columns are LACE / SAM.
- **Mean mask IoU:** 0.724 / 0.741; SAM better on 61% of pairs.
- **Size bias:**
  - < 24 px crowns: both 11% too large.
  - ≥ 48 px: LACE unbiased (ratio 1.00), SAM 7–9% smaller.
- **Error type, ≥ 48 px:** over 0.17–0.18 / 0.08–0.11; under 0.12–0.13 / 0.16. LACE errs on both sides; SAM is tight.
- **Boundary roughness (perimeter²/4πA, median):**
  - 48–96 px: LACE 1.33, SAM 0.97, GT 1.07.
  - ≥ 96 px: 1.44 / 1.04 / 1.06.
  - LACE outlines are much rougher than GT; SAM's match GT.
- **Outright failures (IoU < 0.5):** 3.9% / 3.0%. LACE-ok-SAM-fails 1.2%; SAM-ok-LACE-fails 2.0%.
- **Reading:**
  - The AP50:95 gap is boundary raggedness, not size bias or gross failures.
  - LACE's tree knowledge does not show up as better shapes on true detections.

## Pre-registration S3 (2026-11-04, Tom: "yes"; BEFORE any number): scale-aware boundary smoothing
Gaussian smoothing with σ = c × the mask's equivalent diameter (threshold 0.5), c ∈ {0.02, 0.03, 0.04, 0.06, 0.08, 0.10}, under s′ ranking. Choose c by val AP50:95 subject to val AP50 ≥ s′ − 0.001; report it beside fixed σ = 2 px and with R3 (15-run OOF). Val only (`side3.py`, `results/side3_val.json`, log `native_raster/logs/side3_2026-11-04.log`). A test + sparse look only with Tom's go-ahead, and only if it beats σ = 2 px on val AP50:95 by ≥ 0.002.

## RESULTS, S3 (2026-11-04; `results/side3_val.json`, log `native_raster/logs/side3_2026-11-04.log`): NOT ADOPTED (val only, test not touched)
Val AP50 / AP75 / AP50:95 under s′.
- Reference: none 0.7384 / 0.3345 / 0.3763; fixed σ 2 px 0.7409 / 0.3437 / 0.3802.
- Scale-aware σ = c·d:

| c | AP50 | AP75 | AP50:95 |
|---|---|---|---|
| 0.02 | 0.7383 | 0.3371 | 0.3772 |
| 0.03 | 0.7387 | 0.3385 | 0.3779 |
| 0.04 | 0.7385 | 0.3391 | 0.3789 |
| 0.06 | 0.7382 | 0.3414 | **0.3793** |
| 0.08 | 0.7384 | 0.3360 | 0.3778 |
| 0.10 | 0.7403 | 0.3292 | 0.3752 |

- With R3: none 0.7437 / 0.3810; σ 2 px 0.7463 / 0.3849; c = 0.06 0.7437 / 0.3840.
- **Reading:**
  - Scale-aware smoothing does not beat fixed 2 px (best 0.3793 vs 0.3802), so it fails the +0.002 bar.
  - Smoothing large crowns more does not convert their rougher outlines into AP. Most matched pairs are < 48 px, and at that scale 2 px is already near the optimum.
  - The boundary gap to SAM (+0.017 mean IoU) is not closable by post-hoc smoothing of LACE's binary masks.

### Contact sheet + edge split (2026-11-04; `fig_sam_vs_lace.py` → `results/fig_sam_beats_lace_top30_test.{jpg,json}`; `mask_diag_edge.py` → `results/mask_diag_edge_split_test.json`)
- **Top 30 pairs by SAM − LACE mask IoU** (gain 0.33–0.65): 23/30 have the D1 box touching the tile border (≤ 3 px); 15/30 have a GT diameter < 30 px.
- **Typical failures:**
  - At the edge, LACE fills the truncated box where the GT crown is a sliver or a diagonal cut.
  - In the interior (large crowns), LACE leaks into adjacent canopy, shadow or ground; SAM follows the crown edge.
- **All 23,803 pairs:**
  - Edge pairs are 8%, gap +0.048 (LACE 0.709, SAM 0.757), 23% of the total gap.
  - Interior pairs are 92%, gap +0.014 (0.725 / 0.739), 77% of the total gap.
- **AP value of the gap** (`mask_swap_diag.py` → `results/mask_swap_diag_test.json`; test, s′ ranking, diagnostic only). SAM 3 masks swapped into LACE's:
  - Edge detections only (23,254 of 200,148): AP50 +0.0012, AP50:95 +0.0033.
  - Interior only: +0.0043 / +0.0147.
  - All: +0.0059 / +0.0185 (0.7094 / 0.3532).

## Pre-registration C1 (2026-11-04, Tom: "can we try to introduce a notion of shape or connectedness to the hard cut?"; BEFORE any number): val only
Re-render the frozen `lace4px_simple` masker on frozen D1 val boxes with the final cut (q ≥ 0.33 × box peak) replaced by: largest/peak-connected component; hole fill; both; hysteresis (components containing q ≥ 0.6 × peak) ± fill; cut of a Gaussian-smoothed q (σ 1/2/3); guided-filter eps 1e-3 / 1e-2; σ 2 + connected + fill. Script `shape_val.py`, results `results/shape_val.json`, log `native_raster/logs/shape_val_2026-11-04.log`.
- **Controls:** s′ unchanged (posterior untouched). `GA.EDGE_FILL` forced False so only the cut changes (the border fix is a separate line of work). The `base` re-render must reproduce the cached D1 val masks exactly.
- **Selection:** highest val AP50:95 with val AP50 ≥ base − 0.001; reported beside fixed post-hoc σ 2 px (0.3802) and with R3.
- **Next step:** flagged for a test + sparse look only if it beats base by ≥ +0.003 AP50:95; that look needs Tom's go-ahead.

## RESULTS, C1 (2026-11-04; `results/shape_val.json`, masks `feat_cache/shape_val/`, log `native_raster/logs/shape_val_2026-11-04.log`): val only, test not touched
- **Parity:** base re-render == cached D1 val masks for 51,341/51,341.
- **Val AP50 / AP75 / AP50:95, s′ ranking:**

| cut | AP50 | AP75 | AP50:95 |
|---|---|---|---|
| base | 0.7384 | 0.3345 | 0.3763 |
| conn | 0.7382 | 0.3348 | 0.3763 |
| fill | 0.7381 | 0.3360 | 0.3766 |
| conn_fill | 0.7378 | 0.3359 | 0.3766 |
| hyst | 0.7382 | 0.3348 | 0.3763 |
| hyst_fill | 0.7379 | 0.3359 | 0.3766 |
| qs1 | 0.7377 | 0.3445 | 0.3801 |
| qs2 | 0.7371 | 0.3484 | 0.3801 |
| qs3 | 0.7337 | 0.3395 | 0.3741 |
| eps1e-3 | 0.7364 | 0.3253 | 0.3705 |
| eps1e-2 | 0.7263 | 0.2960 | 0.3511 |
| qs2_conn_fill | 0.7371 | 0.3484 | 0.3802 |

- **Selected by rule:** qs1 (+0.0038 AP50:95, −0.0007 AP50). With R3: 0.7431 / 0.3847 vs R3 + base 0.7437 / 0.3810.
- **Reading:**
  - Connectedness (component, hysteresis, fill) is neutral: the dots and holes carry no AP.
  - A shape prior via smoothing the soft map ties the post-hoc σ 2 binary blur on AP50:95 (0.3801 vs 0.3802) but is slightly worse on AP50 (0.7377 vs 0.7409).
  - Larger eps (less edge-following) HURTS (eps 1e-2: −0.025): the fine colour edges carry real boundary information; the raggedness is the price.
  - No cut variant beats the existing σ 2 post-hoc option, which was confirmed on test and not on sparse.

## Pre-registration R3-edgefix (2026-11-04, Tom: "in another workflow I fixed a bug in D1. Please re run R3 and R3 + smoothing using EDGE_FILL = True"; BEFORE any number)
ONE look, test 439 + sparse-236 (`r3_edgefix.py`, `results/r3_edgefix.json`, log `native_raster/logs/r3_edgefix_2026-11-04.log`).
- **Masks:** the edge-fixed D1 masks from `edgefix_test.py` (`results/edgefix/preds_d1_edgefix_<split>_sprime_r2048.json`). s′ is asserted unchanged; R3 uses no mask feature, so the R3 scores are the frozen ones from R3-final.
- **Rows:**
  - `R3_ef`: edge-fixed masks + R3;
  - `R3_ef_blur2`: + Gaussian σ 2 px smoothing (threshold 0.5).
- **Scoring:** frozen score_coco; paired bootstrap (AP50 and AP50:95) vs edge-fixed s′, vs original s′, R3_ef_blur2 vs R3_ef, and R3_ef vs R3. Reported whatever the outcome.

## RESULTS, R3-edgefix (2026-11-04; `results/r3_edgefix.json`, `results/scored_dino_swinl_s0_{test,sparse}_{R3_ef,R3_ef_blur2}_r2048.json`, log `native_raster/logs/r3_edgefix_2026-11-04.log`)
fasteval reproduced score_coco on all 5 rows of both splits. s′ unchanged by the fix (asserted).

| | test AP50 | AP75 | AP50:95 | sparse AP50 | AP75 | AP50:95 |
|---|---|---|---|---|---|---|
| s′ (original) | 0.7035 | 0.2615 | 0.3347 | 0.7349 | 0.3900 | 0.3962 |
| s′ + edge fix | 0.7044 | 0.2644 | 0.3366 | 0.7350 | 0.3944 | 0.3980 |
| R3 (original masks) | 0.7131 | 0.2685 | 0.3406 | 0.7432 | 0.3964 | 0.4013 |
| **R3 + edge fix** | **0.7141** | 0.2713 | 0.3424 | **0.7434** | 0.4006 | 0.4031 |
| R3 + edge fix + σ 2 | 0.7142 | 0.2766 | 0.3444 | 0.7427 | 0.4004 | 0.4032 |

- **R3 + edge fix vs s′ + edge fix:** AP50 +0.0097 [+0.0069, +0.0124] test, +0.0084 [+0.0037, +0.0152] sparse; AP50:95 +0.0059 [+0.0046, +0.0073] test, +0.0051 [+0.0022, +0.0088] sparse. Significant on both.
- **σ 2 on top of R3 + edge fix:**
  - Test: AP50:95 +0.0020 [+0.0013, +0.0032], AP50 +0.0001.
  - Sparse: AP50:95 +0.0001 [−0.0016, +0.0018], AP50 −0.0007.
  - Again test-only.
- **Edge fix on top of R3:** AP50:95 +0.0019 test, +0.0017 sparse (both CIs > 0); AP50 +0.0009 / +0.0002.
- **Table (Tom, 2026-11-04: "update all local documentation with R3 edge fix results, including TABLE1_CURRENT.md"):** rows "LACE (D1 variant) + edge fix", "+ edge fix + R3" and "+ edge fix + R3 + σ2" added to `results_439/TABLE1_CURRENT.{md,json}` for test 439 and sparse236 (asserted against the scored JSONs); the plain D1 row is kept as the pre-registered reference; banners refreshed via `mark_superseded`. paper.tex NOT updated.

## Pointer: 4 px masker refit (R4P, 2026-11-04)
The masker refit on the 4 px lattice was run on D1 val boxes: n120 0.7404 and n250 0.7417, against 0.7414 for the deployed masker. It FAILED the +0.003 gate, so the deployed masker is kept. Details: `../masker_refit4p/README.md`. Note: D1's val s′ reference is 0.7414 with the shared EDGE_FILL edge fix; 0.7384 is the pre-fix value.

## Pre-registration GC-test (2026-11-04, Tom: "i think 439 is enough. run it locally"; BEFORE any number): guide controls on D1 test
Purpose: replace the CenterNet-era test evidence for the guided render (paper.tex %CHECK above "By taking the 4 px interlaced grid…": test +0.022 over 0.632) with D1 numbers. ONE look, test 439 only (sparse-236 not run). `guide_controls_test.py`, `results/guidectl/guidectl_test.json`, masks `feat_cache/guidectl_test_<arm>/`, log `native_raster/logs/guidectl_test_2026-11-04.log`.
- **Boxes / masker:** frozen D1 test boxes and scores, lace4px_simple post4 (c_peak 0.33, α 0.3, κ 1.6), `EDGE_FILL = True`. Only the upsampling step differs between arms.
- **Arms:** `guided` = deployed colour-guided r 12, ε 1e-4 (masks reused from `results/edgefix/preds_d1_edgefix_test_sprime_r2048.json`; gate: re-render of the first 3 sorted tiles must be byte-identical); `bilinear` = r 0 (posterior bilinear only); `blur` = constant guide, box filter 2r+1 (r 12); `wrongtile` = colour guide from the next test tile in sorted order (wrapping). All arms at the frozen c_peak 0.33 (no per-arm threshold tuning on test; val b29 used the same fixed 0.33).
- **Rankings:** primary s′ (= the "LACE (D1 variant) + edge fix" row; s′ asserted identical across arms); secondary R3 (frozen R3 scores from `preds_dino_swinl_s0_test_R3_ef_r2048.json`, = the "+ edge fix + R3" row).
- **Scoring:** frozen score_coco, canopy-neutral crowd arm, maxDets 600; mask AP50 / AP75 / AP50:95; paired tile bootstrap (2,000, seed 0) of guided − each control, both rankings. Reported whatever the outcome; no decision rides on it beyond what the paper prose says.

## RESULTS, GC-test (2026-11-04; `results/guidectl/guidectl_test.json`, preds/scored `results/guidectl/{preds,scored}_d1_gc_<arm>_<rank>_test_r2048.json`, masks `feat_cache/guidectl_test_<arm>/`, log `native_raster/logs/guidectl_test_2026-11-04.log`)
Gate PASS (guided re-render of 3 tiles byte-identical to the edge-fixed masks, s′ max |Δ| 0). s′ asserted identical across arms; fasteval == score_coco on all 8 rows. Masks changed vs guided: bilinear 199,955 / blur 197,685 / wrong tile 198,610 of 200,148. Test 439, canopy-neutral crowd, mask AP:

| Ranking | Arm | AP50 | AP75 | AP50:95 |
|---|---|---|---|---|
| s′ | guided (deployed) | 0.7044 | 0.2644 | 0.3366 |
| s′ | bilinear | 0.7004 | 0.2407 | 0.3254 |
| s′ | blur | 0.6704 | 0.1839 | 0.2877 |
| s′ | wrong tile | 0.6807 | 0.1966 | 0.2982 |
| R3 | guided (deployed) | 0.7141 | 0.2713 | 0.3424 |
| R3 | bilinear | 0.7098 | 0.2458 | 0.3304 |
| R3 | blur | 0.6809 | 0.1923 | 0.2944 |
| R3 | wrong tile | 0.6907 | 0.2044 | 0.3046 |

Paired tile bootstrap, guided − control (2,000, seed 0), diff [95% CI]; p(≤0) = 0 for every cell:

| Ranking | vs | ΔAP50 | ΔAP75 | ΔAP50:95 |
|---|---|---|---|---|
| s′ | bilinear | +0.0040 [0.0026, 0.0078] | +0.0237 [0.0167, 0.0315] | +0.0111 [0.0086, 0.0138] |
| s′ | blur | +0.0340 [0.0277, 0.0417] | +0.0805 [0.0689, 0.0938] | +0.0489 [0.0425, 0.0549] |
| s′ | wrong tile | +0.0237 [0.0196, 0.0304] | +0.0679 [0.0578, 0.0790] | +0.0383 [0.0334, 0.0428] |
| R3 | bilinear | +0.0043 [0.0029, 0.0080] | +0.0255 [0.0189, 0.0330] | +0.0120 [0.0096, 0.0146] |
| R3 | blur | +0.0332 [0.0270, 0.0407] | +0.0790 [0.0672, 0.0919] | +0.0481 [0.0418, 0.0539] |
| R3 | wrong tile | +0.0234 [0.0193, 0.0298] | +0.0669 [0.0572, 0.0777] | +0.0378 [0.0329, 0.0422] |

**Reading.** On D1 the guided render's AP50 gain over bilinear is small (+0.004, CI > 0), well below the CenterNet-era +0.022 cited in paper.tex; the gain is now chiefly at strict IoU (AP75 +0.024–0.026, AP50:95 +0.011–0.012). Both controls lose to bilinear on every metric, so the gain comes from the tile's own image edges, not smoothing (confirms val b29 on test). Single look, single detector seed. paper.tex NOT edited (Tom decides).

## Pre-registration IL-test (2026-11-04, Tom: "run c and update appendix with results"; BEFORE any number): interlaced vs interpolated masker features on D1 test
Purpose: replace the stale NEON / CenterNet-era `tab:grid` (paper.tex, "Feature grid: interlacing versus interpolation") with a masker-level OAM-TCD result on the current pipeline. ONE look, test 439 only. `interlace_test.py`, `results/interlace/interlace_test.json`, masks `feat_cache/interlace_test_<arm>/`, log `native_raster/logs/interlace_test_2026-11-04.log`.
- **Boxes / masker:** frozen D1 test boxes and scores, lace4px_simple post4 (c_peak 0.33, α 0.3, κ 1.6, guided r 12, ε 1e-4), `EDGE_FILL = True`, masker npz `em_model_4p_fix.npz` at s 4 / origin 8. Only the 4 px feature lattice `zn4` the masker reads differs between arms.
- **Arms:** `real4` = deployed, 16 offsets interlaced (`hunt_z4p_test`; masks reused from `results/edgefix/preds_d1_edgefix_test_sprime_r2048.json`; gate: re-render of the first 3 sorted tiles byte-identical). `interp8` = 4 offsets {0,8}² interlaced (8 px, Modal `feat_4p_test`), bilinearly interpolated to the 4 px lattice. `interp16` = native single-pass 16 px grid (phase (0,0) of `feat_4p_test`, = `[:, 0::2, 0::2]`), bilinearly interpolated to the 4 px lattice.
- **Interpolation:** on the masker's unnormalised whitened projection `(f − μ)U/scale` (affine, so identical to interpolating the 1024-d DINOv3 features then projecting), then L2-normalised and fp16-rounded as the stored `zn4`. Source cell centres map exactly onto 4 px cells (16 px cell g → 4 px cell 4g; 8 px cell g → 2g); cells past the last source centre are edge-clamped.
- **Gates:** (1) per tile, projection of `feat_4p_test` vs `hunt_z4p_test[0::2, 0::2]`: cos mean ≥ 0.999 and 1st percentile ≥ 0.99 (local t457 venv vs Modal, previously 1.0000); (2) each interpolated arm reproduces its source projection exactly at the source lattice cells (max |Δ| < 1e-3 after fp16); (3) deployed re-render gate above. Also logged: cosine of each interpolated arm to `real4` at the off-source cells (the 4 px analogue of the paper's cos ≈ 0.955 claim).
- **Rankings:** primary own s′ (each arm's s′ from its own posterior; the pipeline as deployed); secondary detector s (masks only differ; isolates mask quality).
- **Scoring:** frozen score_coco, canopy-neutral crowd arm, maxDets 600; mask AP50 / AP75 / AP50:95; fasteval asserted equal; paired tile bootstrap (2,000, seed 0) of real4 − each interpolated arm, both rankings.
- **Caveat stated in advance:** c_peak / α / κ were tuned on real-4 px features (hunt, val), which favours `real4`; no per-arm re-tuning on test. Single detector seed. Reported whatever the outcome.

## Pre-registration CAP-test (2026-11-04, Tom: "proceed" on the appendix staleness list; BEFORE any number): does the 600-per-tile stitch cap bind for LACE (D1 + edge fix + R3)?
Purpose: paper.tex appendix "Detection budget" says only Detectree2 exceeds 600 dets/tile. Check on D1 before cap. `cap_test.py`, `results/cap/cap_test.json`, masks `feat_cache/cap_test/`, log `native_raster/logs/cap_test_2026-11-04.log`. Test 439 only, ONE look.
- **Already measured (counting only, no AP):** raw subtile dets `runs/dino_swinl_s0/raw_test.jsonl` (copied to `results/modal/raw/`); `dino_modal.stitch` re-run locally reproduces the published 600-capped preds exactly (scores, all 439 tiles). Uncapped (`keep=∞`, same floor 0.05 / edge / dedup): 212 / 439 tiles exceed 600, max 1,186, +47,711 dets (median 576/tile).
- **Masks:** lace4px_simple post4, `EDGE_FILL = True`, on the uncapped box list per tile (tiles with ≤ 600 dets reuse published masks). Gate: on every re-rendered tile the first 600 masks and s′ equal `results/edgefix/preds_d1_edgefix_test_sprime_r2048.json`.
- **Ranking:** R3 refit exactly as `rerank3.final` (mlp1x64 on s, log s′/s, log h, max IoU with a higher-s det; 25 models, seeds 0–4 × rr folds, val-trained). Gate: refit scores on the published capped set equal the published R3 scores (max |Δ| < 1e-6).
- **Score:** frozen score_coco, crowd arm, floor 0; uncapped list at maxDets 1,200 vs published R3 + edge fix (0.7141 / 0.3424). Also the uncapped list at maxDets 600 (top 600 by R3 instead of top 600 by s). Paired tile bootstrap (2,000, seed 0) on AP50 and AP50:95. Reported whatever the outcome.

## RESULTS, IL-test (2026-11-04/05; `results/interlace/interlace_test.json`, preds/scored `results/interlace/{preds,scored}_d1_il_<arm>_<rank>_test_r2048.json`, masks + per-tile feature stats `feat_cache/interlace_test_<arm>/`, log `native_raster/logs/interlace_test_2026-11-04.log`)
Gates: real4 re-render of 3 tiles byte-identical to the edge-fixed masks (s′ max |Δ| 0) PASS; `feat_4p_test` projection vs `hunt_z4p_test[0::2,0::2]` cos mean 1.0000 (worst-tile p1 0.9999) PASS on all 439; interpolated arms reproduce their source at lattice cells (max |Δ| 1.2e-4, fp16) PASS; fasteval == score_coco on all 6 rows. Masks changed vs real4: interp8 180,680 / interp16 188,977 of 200,148.
Feature cosine to real4 at off-source cells (mean over tiles; worst tile): interp16 **0.950** (0.877), p1 0.731; interp8 0.981 (0.967), p1 0.884.

Test 439, canopy-neutral crowd, mask AP (s′ rows = each arm's own s′; real4 s′ = the published "+ edge fix" row):

| Ranking | Arm | AP50 | AP75 | AP50:95 |
|---|---|---|---|---|
| s′ | real4 (16 offsets, deployed) | 0.7044 | 0.2644 | 0.3366 |
| s′ | interp8 (4 offsets → 4 px bilinear) | 0.7043 | 0.2619 | 0.3356 |
| s′ | interp16 (native 16 px → 4 px bilinear) | 0.6992 | 0.2490 | 0.3292 |
| detector s | real4 | 0.6900 | 0.2589 | 0.3288 |
| detector s | interp8 | 0.6901 | 0.2566 | 0.3278 |
| detector s | interp16 | 0.6872 | 0.2454 | 0.3229 |

Paired tile bootstrap, real4 − arm (2,000, seed 0), diff [95% CI] p(≤0):

| Ranking | vs | ΔAP50 | ΔAP75 | ΔAP50:95 |
|---|---|---|---|---|
| s′ | interp8 | +0.0001 [−0.0007, 0.0018] 0.42 | +0.0025 [0.0009, 0.0050] 0.001 | +0.0009 [0.0002, 0.0015] 0.004 |
| s′ | interp16 | +0.0052 [0.0035, 0.0091] 0 | +0.0155 [0.0112, 0.0207] 0 | +0.0074 [0.0056, 0.0091] 0 |
| detector s | interp8 | −0.0001 [−0.0007, 0.0017] 0.58 | +0.0023 [0.0010, 0.0047] 0.002 | +0.0010 [0.0004, 0.0015] 0 |
| detector s | interp16 | +0.0028 [0.0017, 0.0059] 0 | +0.0135 [0.0098, 0.0184] 0 | +0.0059 [0.0044, 0.0075] 0 |

**Reading.** Interlacing beats interpolating the native grid on every metric under both rankings (CIs clear of 0), but the masker-level effect is modest at AP50 (+0.005 s′, +0.003 detector s) and larger at strict IoU (AP75 +0.014–0.016). Four offsets carry essentially all of it at AP50 (interp8 ties real4); sixteen offsets add only +0.002–0.003 AP75 / +0.001 AP50:95 over four. Caveat (pre-stated): masker knobs were tuned on real-4 px features, favouring real4. Single look, single detector seed. Replaces the NEON `tab:grid` in paper.tex (2026-11-05).

## RESULTS, CAP-test (2026-11-05; `results/cap/cap_test.json`, `results/cap/{preds_d1_R3_ef_uncapped_test_r2048.json, scored_*.json}`, log `native_raster/logs/cap_test_2026-11-04.log`)
Gates: uncapped stitch prefix = published boxes/scores (439/439); first 600 masks + s′ of all 212 re-rendered tiles = published edge-fixed preds (PASS); R3 refit vs published R3 scores max |Δ| 9.5e-9 (PASS). First run crashed after the mask gate (empty-tile reduction in the R3 helper; fixed, rerun resumed from cached masks, no number had been produced).

| test 439 | mask AP50 | AP75 | AP50:95 | box AP50 |
|---|---|---|---|---|
| published R3 + edge fix (stitch cap 600, maxDets 600) | 0.7141 | 0.2713 | 0.3424 | 0.7471 |
| uncapped, maxDets 1,200 | **0.7188** | 0.2725 | 0.3438 | 0.7531 |
| uncapped, maxDets 600 (top 600 by R3) | 0.7143 | 0.2713 | 0.3424 | 0.7473 |

- Uncapped @1,200 − published: AP50 **+0.0047 [+0.0018, +0.0058]**, AP50:95 +0.0014 [+0.0006, +0.0019].
- Uncapped @600 − published: AP50 +0.0002 [−0.0018, +0.0023], AP50:95 0.0000 (n.s.): choosing the 600 by R3 instead of by s does not matter.
- **Reading:** the 600 cap binds for LACE (212 / 439 tiles; max 1,186) and costs it ≈ +0.005 AP50, so the protocol is conservative for LACE. Headline stays at the protocol cap (0.714). paper.tex "Detection budget" paragraph updated 2026-11-05.

## RESULTS, sparse s-ranking on edge-fixed masks (2026-11-05; `sparse_dets_ef.py`, `results/edgefix/{preds,scored}_d1_edgefix_sparse_dets_r2048.json`, log `native_raster/logs/sparse_dets_ef_2026-11-05.log`)
For paper tab:factors (Tom: "A" = s / s′ / learned ranking on test + sparse, existing numbers; R2 not shown). The only sparse s row predated the edge fix, so the published edge-fixed sparse masks were re-ranked by detector s (boxes asserted identical). Pure re-ordering, no choice made from it; frozen score_coco.
- Sparse-236, s, edge fix: mask AP50 **0.7265**, AP75 0.3874, AP50:95 0.3913; box AP50 0.7414 (= pre-fix detscore file, as expected). Pre-fix s was 0.7263 / 0.3897.
- tab:factors rows (paper.tex, 2026-11-05): test s 0.6900 / 0.3288 (`results/interlace/scored_d1_il_real4_dets_test_r2048.json`), s′ 0.7044 / 0.3366, learned (R3 + edge fix) 0.7141 / 0.3424; sparse s 0.7265 / 0.3913, s′ 0.7350 / 0.3980, learned 0.7434 / 0.4031.

## Pre-registration ABL-D1 (2026-11-10, Tom: "proceed with your recommendations"; BEFORE any number): re-base the four CenterNet-era ablations onto LACE (D1 + edge fix + ŝ)
Paper items: tab:strata, touching-crowns paragraph, tab:masker, tab:components (all `% TODO POSTRUN` in paper.tex). ONE look per split per item, everything local. Common frozen inputs: D1 boxes and detector scores, lace4px_simple post4 (c_peak 0.33, r 12, ε 1e-4), `GA.EDGE_FILL = True`, 4 px features `hunt_z4p_{val,test}`, ŝ = the frozen R3 heads (refit exactly as `rerank3.final`, gated against the published R3 scores as in CAP-test; no refit on new data). Scoring: frozen score_coco (crowd arm, maxDets 600, floor 0); paired tile bootstrap (2,000, seed 0). Single detector run, so no seed bands.
1. **tab:strata** (`native_raster/ablation_r2048/run_strata_d1.py`, wraps `ablation/scripts/t03_strata_tile.py` unmodified, seed 0 only): published `preds_dino_swinl_s0_test_R3_ef_r2048.json`, canopy-fraction bins as before; mask AP50, box AP50, mask − box. Gate: the script's pooled-all mask AP50 = 0.7141 (± 0.0005).
2. **Touching crowns** (same wrapper, `t04_strata_instance.py` unmodified): same preds; box-matched arm (box IoU ≥ 0.5, greedy by ŝ): mean mask IoU, fail rate (mask IoU < 0.5), mean GT size, touching vs isolated. Gate: its AP re-score = 0.7141 (± 0.0005).
3. **tab:masker** (`masker_scalars_d1.py`): test 439, arms (α, κ) = (0, 16), (1.0, 10), (0.3, 16) = deployed (κ = 10 × kappa_scale). Deployed arm reuses `results/edgefix/preds_d1_edgefix_test_sprime_r2048.json`; gate: re-render of the first 3 sorted tiles byte-identical. Rankings: **ŝ (primary, the deployed system)** and each arm's own s′ (secondary, the current caption). Report AP50 / AP75 / AP50:95 and paired bootstrap deployed − arm. Val check for the footnote's directional claims (val 108, own s′ only, since ŝ is val-trained): α ∈ {0, 0.3, 1.0} × κ ∈ {10, 16}; claims tested: "α < 1 beats α = 1 at every κ", "κ > 10 beats κ = 10 at every α".
4. **tab:components** (`components_d1.py`): val 108, K_fg = 16, EM seed 0. M-step: β = 0 vs β = 0.5. E-step recentring on/off. Maskers (all share the deployed whitening, verified: mu/U/scale identical, so `hunt_z4p_val` is reused): β0/absolute = `phase4/em_model_4p_b0_fix.npz`; β0/recentred = `phase4_collapse/results/npz/k16_s0_b0_c1.npz`; β0.5/absolute = `.../k16_s0_b0.5_c0.npz`; β0.5/recentred = deployed `em_model_4p_fix.npz`; all applied at s 4 / origin 8, α 0.3, κ 16.
   - **Definition finding (before running):** every published components eval (`phase4_collapse/collapse_lib.eval_boxsource`, `guided_ablate_modal.components_eval`) called `box_mask` with its default `contrast=True`, so the "absolute" cells were fitted without recentring but still applied WITH it. **Primary arm "fit+infer":** the recentring flag is applied at fit (npz) and at inference (`box_mask(contrast=flag)`). **Secondary arm "fit-only":** inference always recentred, as published.
   - Box sources: GT boxes (oracle, all val crowns) and D1 val boxes (`results/modal/dino_swinl_s0_preds_val.json`), box-matched TPs at box IoU ≥ 0.5, greedy by detector s. Metric: mean per-crown mask IoU at 2048 (post4 mask vs rasterised GT polygon). Effects relative to β0/absolute, plus interaction.
- Reported whatever the outcome; paper edits only after all four, and the components definition choice goes to Tom.

## RESULTS, ABL-D1 (2026-11-10; ONE look each, all gates PASS; single detector run, no seed bands)
Logs `native_raster/logs/abl_d1_{strata,touching,masker,components}_2026-11-10.log`. Deviation from pre-reg item 4: D1 val boxes read from `feat_cache/dino_swinl_s0_val_mask.json` (the exact file `edgefix_val` rendered from; verified identical to `results/modal/dino_swinl_s0_preds_val.json`). The `label` strings inside the two strata JSONs ("knobbed masker, 3 seeds") are inherited from the unmodified t03/t04 scripts and are wrong: the input is the D1 + edge fix + ŝ Table 1 file, seed 0 only.

**1. tab:strata** (`ablation/results_r2048/strata_tile_d1_r2048.json`; gate pooled-all 0.7141 PASS)

| canopy fraction | tiles | n_gt | mask AP50 | box AP50 | mask − box |
|---|---|---|---|---|---|
| [0, 0.10) | 196 | 7523 | 0.6547 | 0.6651 | −0.0103 |
| [0.10, 0.25) | 95 | 8630 | 0.7382 | 0.7466 | −0.0084 |
| [0.25, 0.50) | 86 | 7262 | 0.7588 | 0.7598 | −0.0010 |
| [0.50, 1.00] | 62 | 2290 | 0.6999 | 0.7020 | −0.0021 |

Claims: "lowest on open tiles" HOLDS; "peaks in [0.25, 0.50)" HOLDS; "mask − box positive in all bins, largest on densest" **DIES** (negative in all four bins; least negative on the two densest bins).

**2. Touching crowns** (`ablation/results_r2048/strata_instance_d1_r2048.json`; gate AP re-score 0.7141 PASS). Box-matched arm, 22,525 box TPs; GT touching rate 39.3% of 25,705 (model-independent).

| | n | mean mask IoU | fail rate (IoU < 0.5) | mean GT size px |
|---|---|---|---|---|
| touching | 8874 | 0.7353 | 0.0241 | 68.9 |
| isolated | 13651 | 0.7330 | 0.0223 | 42.0 |

Claims: "score almost identically" HOLDS (touching now marginally higher); "fail slightly more often" HOLDS.

**3. tab:masker** (`results/abl_d1/masker_scalars.json`, preds/scored `results/abl_d1/{preds,scored}_{val,test}_*_r2048.json`, masks `feat_cache/abl_masker_{val,test}_a*_k*/`; gates: val + test deployed re-render byte-identical on 3 tiles, R3 refit max |Δ| < 1e-6, deployed ŝ 0.7141 and s′ 0.7044 reproduced)

Val 108, own s′, mask AP50 / AP50:95:

| α \ κ | 10 | 16 |
|---|---|---|
| 0 | 0.7034 / 0.3418 | 0.7130 / 0.3508 |
| 0.3 | 0.7410 / 0.3791 | **0.7414** / 0.3788 |
| 1.0 | 0.7283 / 0.3657 | 0.7299 / 0.3725 |

Claims: "κ > 10 beats κ = 10 at every α" HOLDS (margin at α 0.3 is +0.0004 AP50, and AP50:95 is −0.0003 there). "α < 1 beats α = 1 at every κ" **DIES** as worded: α = 0 is below α = 1 at both κ; α = 0.3 beats α = 1 at both κ.

Test 439, mask AP50 / AP75 / AP50:95 (box AP50):

| arm | ŝ (primary) | own s′ |
|---|---|---|
| α 0, κ 16 | 0.6923 / 0.2304 / 0.3183 (0.7296) | 0.6848 / 0.2213 / 0.3119 (0.7253) |
| α 1.0, κ 10 | 0.6989 / 0.2577 / 0.3300 (0.7394) | 0.6857 / 0.2536 / 0.3244 (0.7273) |
| α 0.3, κ 16 (deployed) | **0.7141 / 0.2713 / 0.3424** (0.7471) | 0.7044 / 0.2644 / 0.3366 (0.7391) |

Paired bootstrap deployed − arm, diff [95% CI]:

| ranking | vs | ΔAP50 | ΔAP50:95 |
|---|---|---|---|
| ŝ | α 0 | +0.0218 [0.0174, 0.0280] | +0.0241 [0.0211, 0.0268] |
| ŝ | α 1, κ 10 | +0.0152 [0.0116, 0.0213] | +0.0124 [0.0091, 0.0157] |
| s′ | α 0 | +0.0196 [0.0155, 0.0253] | +0.0247 [0.0217, 0.0272] |
| s′ | α 1, κ 10 | +0.0187 [0.0149, 0.0251] | +0.0121 [0.0089, 0.0157] |

Reading: "removing the spatial prior costs 0.021 AP50 / 0.018 AP50:95" HOLDS and strengthens (ŝ: 0.022 / 0.024). Note box AP50 also moves across arms because s′ (and hence ŝ) depends on the posterior.

**4. tab:components** (`results/abl_d1/components.json`, per-tile `feat_cache/abl_comp_val/`; gates: whitening identical for 4 npz, deployed render = `edgefix_val` on 3 tiles). Val 108; 7,041 GT crowns; 6,143 D1 box-matched TPs. Mean crown IoU:

| M-step | E-step | GT fit+infer | D1 fit+infer | GT fit-only | D1 fit-only |
|---|---|---|---|---|---|
| β 0 | absolute | 0.7412 | 0.6873 | 0.7842 | 0.7364 |
| β 0 | recentred | 0.7950 | 0.7461 | 0.7950 | 0.7461 |
| β 0.5 | absolute | 0.7585 | 0.7075 | 0.7897 | 0.7398 |
| β 0.5 (deployed) | recentred | **0.7980** | **0.7476** | 0.7980 | 0.7476 |

Effects vs β0/absolute, fit+infer (primary): GT repulsion +0.0173, recentring +0.0538, both +0.0568, interaction −0.0143; D1 repulsion +0.0202, recentring +0.0588, both +0.0603, interaction −0.0187. Claims: "substitutes, not complements" HOLDS (both regimes); "recentring slightly larger" → recentring now ~3× repulsion; deployed cell now best in both columns (published: β0/recentred beat it on GT). Fit-only (as published): effects ~5× smaller. Definition choice → Tom (recommended: fit+infer primary, caption says so).

## Pre-registration ABL-D1 item 5 (2026-11-10, BEFORE any number): tab:collapse Crown IoU column on the deployed 4 px path
Found while clearing item 4: tab:collapse's Crown IoU column (0.7537 / 0.7541 / 0.7539 / 0.7472 / 0.7486 / 0.7470) is on the old render path, while the collapse prose now cites tab:components' new cells (0.7950 / 0.7980) for the same two maskers. Fix: re-measure that one column exactly as item 4's GT arm (`components_d1._tile` logic, GT boxes only, contrast True, α 0.3, κ 16, EDGE_FILL, s 4 / origin 8), val 108, 7,041 crowns. Maskers `phase4_collapse/results/npz/`: k16 β0 = `k16_s0_b0_c1`, k16 β0.1 = `k16_s0_b0.1_c1`, k16 β0.25 = `k16_s0_b0.25_c1`, k16 β0.5 = deployed (`k16_s0_b0.5.npz` is byte-identical to `em_model_4p_fix.npz`; the old column's 0.7472 = tab:components' deployed cell), k2 β0 = `k2_s0_b0_c1`, k2 β0.5 = `k2_s0_b0.5` (last three + β0.1 pulled 2026-11-10 from `tcd04-phase4-vol:/collapse/*/em_*_fix.npz`; whitening verified identical to deployed). Gate: k16 β0 and β0.5 reproduce item 4's GT cells 0.7950 / 0.7980. Geometry columns (fit-time) unchanged. Claim re-checked: "crown IoU moves by at most 0.007 across the K16 block, collapsed β=0.5 cells lowest". ONE look. Script `dino_det/collapse_iou_d1.py` → `results/abl_d1/collapse_iou.json`.

## RESULTS, ABL-D1 item 5 (2026-11-10; `results/abl_d1/collapse_iou.json`, per-tile `feat_cache/abl_collapse_val/`, log `native_raster/logs/abl_d1_collapse_2026-11-10.log`)
Gates PASS (whitening identical ×6; k16 β0 / β0.5 = item 4 GT cells 0.7950 / 0.7980). Crown IoU on GT boxes, 7,041 crowns: k16 β0 0.7950, β0.1 0.7953, β0.25 0.7958, β0.5 (deployed) 0.7980; k2 β0 0.7980, k2 β0.5 0.7985. K16 span 0.003 (was 0.007). Claim "collapsed β=0.5 cells lowest" **FLIPS**: they are now highest. "β is a geometry knob with ~no accuracy consequence" holds. paper.tex tab:collapse column + footnote updated 2026-11-10.

## Pre-registration T3 (2026-11-10, Tom: "ok please run 1 end to end including table update"; BEFORE any number): fill paper Table 3 (tab:factors, cvpr_full numbering)
ONE look, test 439 + sparse-236, local CPU (`t3_abl.py`, `results/t3_abl.json`, preds/scored `results/preds_dino_swinl_s0_<split>_t3_<row>_r2048.json`, log `native_raster/logs/t3_abl_2026-11-10.log`). Ablations of frozen configs: nothing is selected on these numbers, and every number goes into Table 3 whatever it is.
- **Fixed in every row:** the D1 detection set (decode floor on s), the edge-fixed masks (`results/edgefix/preds_d1_edgefix_<split>_sprime_r2048.json`, boxes asserted identical), frozen score_coco (canopy-neutral crowd, 2048², maxDets 600, score floor 0). Only the ranking changes.
- **Factor rows:** `sm` = s·m and `sp` = s·p̄, with p̄ = mean r and m = mean |2r−1| over the in-box 4 px posterior r from `GA._prob` (α 0.3, κ×1.6, deployed `em_model_4p_fix.npz`, as `abl_lib.render`). **Gate 1:** s·p̄·m must reproduce the cached s′ on every box of both splits (|Δ| < 1e-7); otherwise stop.
- **ŝ input-drop rows:** `no_s`, `no_logpm`, `no_logh`, `no_iou` = the frozen R3 recipe (mlp1x64, 30 epochs, dropout 0.1, wd 1e-3, box-0.5 labels; 25 models = 5 seeds × rr folds on val; mean) refit without that one input, applied to the cached test/sparse features (`feat_cache/rerankR1_<split>.npz`). **Gate 2:** the same code with all 4 inputs must reproduce the published R3 scores (|Δ| < 1e-6); otherwise stop.
- **Gate 3:** fasteval reproduces score_coco on every row (|Δ| < 1.5e-4).
- **Report:** mask AP50 / AP75 / AP50:95 per row and split; paired tile bootstrap (2,000, seed 0) on AP50 and AP50:95: factor rows vs s′, input-drop rows vs ŝ.
- **Disclosure:** six more scored looks per split at ablation rows; ŝ itself is unchanged.

## RESULTS, T3 (2026-11-10; `results/t3_abl.json`, `results/{preds,scored}_dino_swinl_s0_{test,sparse}_t3_<row>_r2048.json`, per-tile p̄/m `feat_cache/t3_pm_{test,sparse}/`, log `native_raster/logs/t3_abl_2026-11-10.log`): ALL GATES PASS, ONE look
Gate 1: max |s·p̄·m − s′| = 5.0e-9 on both splits (8 dp rounding of s′). Gate 2: 4-input refit = published R3 exactly (0.0). Gate 3: fasteval = score_coco on all 8 rows of both splits.

| Ranking | test AP50 | AP75 | AP50:95 | sparse AP50 | AP75 | AP50:95 | Δ vs ref, AP50 [95% CI] test / sparse | Δ AP50:95 test / sparse |
|---|---|---|---|---|---|---|---|---|
| s′ (ref for factor rows) | 0.7044 | 0.2644 | 0.3366 | 0.7350 | 0.3944 | 0.3980 | | |
| s·m (drop p̄) | 0.7012 | 0.2572 | 0.3319 | 0.7304 | 0.3890 | 0.3939 | −0.0032 [−0.0046, −0.0017] / −0.0046 [−0.0096, −0.0014] | −0.0047 [−0.0054, −0.0035] / −0.0041 [−0.0068, −0.0018] |
| s·p̄ (drop m) | 0.6959 | 0.2665 | 0.3345 | 0.7327 | 0.3944 | 0.3964 | −0.0085 [−0.0126, −0.0045] / −0.0023 [−0.0075, +0.0031] | −0.0020 [−0.0040, +0.0003] / −0.0016 [−0.0050, +0.0023] |
| ŝ (ref for drop rows) | 0.7141 | 0.2713 | 0.3424 | 0.7434 | 0.4006 | 0.4031 | | |
| ŝ − s | 0.6407 | 0.2374 | 0.3036 | 0.6444 | 0.3349 | 0.3415 | −0.0734 [−0.0894, −0.0568] / −0.0990 [−0.1139, −0.0827] | −0.0388 / −0.0615 (CIs < 0) |
| ŝ − log(p̄m) | 0.7058 | 0.2701 | 0.3399 | 0.7396 | 0.3988 | 0.4011 | −0.0083 [−0.0102, −0.0062] / −0.0038 [−0.0065, −0.0015] | −0.0025 / −0.0020 (CIs < 0) |
| ŝ − log h | 0.7068 | 0.2643 | 0.3365 | 0.7373 | 0.3937 | 0.3979 | −0.0073 [−0.0097, −0.0047] / −0.0061 [−0.0130, −0.0018] | −0.0059 / −0.0051 (CIs < 0) |
| ŝ − IoU↑ | 0.7082 | 0.2705 | 0.3405 | 0.7395 | 0.4007 | 0.4021 | −0.0059 [−0.0075, −0.0044] / −0.0039 [−0.0054, −0.0026] | −0.0019 / −0.0010 (CIs < 0) |

- **Every ŝ input is needed:** dropping any one lowers AP50 on both splits with the CI excluding 0. The detector score matters most by far (−0.07 / −0.10).
- **The mask term in ŝ is real but modest:** without log(p̄m), ŝ falls to 0.706 / 0.740, about level with s′ (0.704 / 0.735). The box cues (log h, IoU↑) on their own recover roughly the s′ gain; log(p̄m) adds +0.008 test / +0.004 sparse on top.
- **Both factors of s′ help on test** (drop p̄ −0.003, drop m −0.009 AP50); on sparse, drop p̄ −0.005 (CI < 0) but drop m −0.002 (n.s.).
- Paper Table 3 (tab:factors) filled 2026-11-10 from these files (3 dp half up); bolding unchanged (ŝ best in every column).

## Pre-registration T1-boot (2026-11-11, Tom: "set up cheap half run and execute"; BEFORE any number)

Paired tile-bootstrap CIs for LACE − every baseline row of tab:oamtcd439 (test 439: Restor rpn1000 tree-only, Detectree2, Box2Mask Swin-L, Box2Mask R-50, BoxTeacher) and tab:sparse236 (sparse 236: Detectree2, Box2Mask Swin-L, BoxTeacher). LACE = deployed D1 + edge fix + R3 (ŝ), `preds_dino_swinl_s0_{test,sparse}_R3_ef_r2048.json`. Mask AP50 and AP50:95, canopy-neutral crowd arm, maxDets 600, each row's own protocol floor (LACE 0.0, baselines 0.05). 2,000 resamples, seed 0, same draw for every row. Test-set sampling uncertainty only (single training seed everywhere).
- **Gate:** each row's full-sample AP50 and AP50:95 from the bootstrap code reproduces its `scored_*.json` within 5e-4, else abort before resampling.
- **Prior record:** D1 (s′) − BoxTeacher sparse AP50 +0.014 [−0.005, 0.033] (`results/boot_sparse_mask_ap50.json`), CI included 0; the ŝ comparison (+0.022) was never bootstrapped.
- **Use:** a Δ [95% CI] column for Table 1 and the sparse table; the abstract's sparse BoxTeacher claim (changed 2026-11-11) stands or is softened on this result.
- Script `t1_boot.py`; outputs `results/t1_boot_{test,sparse}.json`; log `native_raster/logs/t1_boot_2026-11-11.log`. ONE look per split.

## Pre-registration BOXSHAPE (2026-11-11; PREPARED, NOT RUN — awaiting Tom's greenlight, after T1-boot)

Geometric-masker control. LACE's deployed test/sparse detections (D1 + edge fix + R3), boxes, ŝ scores, floor 0.0 and order held fixed; each mask replaced by (a) the filled box, (b) the inscribed ellipse (72-vertex polygon, pycocotools rasterised at 2048). No ŝ refit: the ranking is held fixed by design so only mask geometry varies (ŝ reads LACE's posterior via log(p̄m), which helps every arm equally). CPU only, local.
- **Gate:** the unchanged "lace" arm reproduces `scored_dino_swinl_s0_{split}_R3_ef_r2048.json` mask AP50 / AP50:95 to 4 dp.
- **Readout (fixed now):** LACE − box and LACE − ellipse, mask AP50 and AP50:95, paired tile bootstrap (2,000, seed 0). AP50:95 is the primary measure of masker value: a tight box around a round crown gives ≈ π/4 IoU, so AP50 is expected to discriminate weakly. A LACE − ellipse AP50:95 CI excluding 0 supports "the masker adds delineation beyond box geometry"; a CI including 0 at AP50 alone is expected and not a negative.
- Smoke (2026-11-11, no AP read): 40×20 box → area 800, ellipse → 628 (π·20·10 = 628.3).
- Script `boxshape.py`; outputs `results/boxshape_{split}.json`, `results/preds_boxshape_{box,ellipse}_{split}_r2048.json`; log `native_raster/logs/boxshape_<date>.log`. ONE look per split.

### T1-boot result (2026-11-11, one look per split; gate PASS on all 10 rows, exact to 4 dp)

LACE (ŝ) − row, paired tile bootstrap 2,000, seed 0. `results/t1_boot_{test,sparse}.json`, log `native_raster/logs/t1_boot_2026-11-11.log` (test 334 s, sparse 137 s).

| Split | Row | ΔAP50 [95% CI] | ΔAP50:95 [95% CI] |
|---|---|---|---|
| test | Restor | +0.066 [0.051, 0.082] | +0.009 [0.000, 0.017] |
| test | Detectree2 | +0.084 [0.075, 0.097] | +0.033 [0.027, 0.038] |
| test | Box2Mask Swin-L | +0.125 [0.108, 0.145] | +0.098 [0.088, 0.107] |
| test | Box2Mask R-50 | +0.264 [0.245, 0.285] | +0.178 [0.167, 0.190] |
| test | BoxTeacher | +0.050 [0.043, 0.060] | +0.038 [0.033, 0.042] |
| sparse | Detectree2 | +0.107 [0.085, 0.135] | +0.050 [0.038, 0.063] |
| sparse | Box2Mask Swin-L | +0.139 [0.095, 0.175] | +0.141 [0.109, 0.174] |
| sparse | BoxTeacher | +0.023 [0.009, 0.039] | +0.028 [0.017, 0.038] |

- Every CI excludes 0. Sparse LACE − BoxTeacher is now significant (s′-era +0.014 [−0.005, 0.033] was not); the abstract claim stands.
- Weakest: test LACE − Restor AP50:95 +0.0092 [0.0001, 0.0169], lower bound at 0 — report as marginal.
- Test-set sampling only; single training seed for every row.

### BOXSHAPE result (2026-11-11, Tom greenlit; one look per split; gate PASS both splits, lace arm exact to 4 dp)

Same D1 boxes, ŝ scores and order in every arm; only mask geometry differs. `results/boxshape_{test,sparse}.json`, log `native_raster/logs/boxshape_2026-11-11.log`.

| Split | Arm | AP50 | AP75 | AP50:95 |
|---|---|---|---|---|
| test | LACE masks | 0.714 | 0.271 | 0.342 |
| test | inscribed ellipse | 0.701 | 0.223 | 0.317 |
| test | filled box | 0.647 | 0.040 | 0.217 |
| sparse | LACE masks | 0.743 | 0.401 | 0.403 |
| sparse | inscribed ellipse | 0.739 | 0.323 | 0.369 |
| sparse | filled box | 0.714 | 0.080 | 0.276 |

LACE − arm, paired tile bootstrap (2,000, seed 0): test ellipse AP50 +0.013 [0.010, 0.018], AP50:95 +0.025 [0.020, 0.030]; test box +0.068 [0.056, 0.083], +0.126 [0.116, 0.135]; sparse ellipse +0.005 [0.002, 0.010], +0.034 [0.024, 0.044]; sparse box +0.030 [0.022, 0.040], +0.127 [0.111, 0.144].
- **Masker beats geometry on every split and metric (all CIs exclude 0).** Per the pre-registered readout, AP50:95 is primary: +0.025 test, +0.034 sparse over the ellipse.
- **But D1 boxes + ellipse alone (0.701 / 0.317 test) beat every Table 1 baseline at AP50** (BoxTeacher 0.664, Restor 0.648); at AP50:95 the ellipse (0.317) trails Restor (0.333), and LACE masks (0.342) are what put LACE ahead. Most of the AP50 margin over baselines is the detector + ranking.
- Ellipse at AP50 ≈ SAM 3 zs on the same boxes (0.701); LACE's +0.013 over each is the same size.
- Caveat: ŝ reads LACE's posterior (log p̄m), so the ellipse/box arms inherit LACE-informed ranking; with s-only ranking they would likely score lower.

## Pre-registration BOXSHAPE-2 (2026-11-11, Tom: geometric rows "shouldn't access the LACE reranking" → "ŝ without mask"; BEFORE any number)

Second look at the geometric controls, superseding BOXSHAPE for tab:masker. Filled box / inscribed ellipse on the same D1 boxes, ranked by the 3-input ŝ refit without log(p̄m) (s, log h, IoU↑; T3 `no_logpm` scores), because a geometric masker has no posterior. LACE reference = deployed 4-input ŝ with LACE masks. So LACE − ellipse now measures the mask module's full contribution (shape + its ranking signal), not shape alone.
- **Gates:** (1) `no_logpm` boxes identical and same order as deployed (checked, both splits); (2) `no_logpm` + LACE masks reproduces t3_abl 0.7058/0.3399 test, 0.7396/0.4011 sparse; (3) deployed reproduces 0.7141/0.3424, 0.7434/0.4031.
- **Disclosure:** second look, chosen after BOXSHAPE was read; the ranking choice is a definitional fix, not tuned on the result.
- Script `boxshape_nm.py`; outputs `results/boxshape_nm_{split}.json`, `results/preds_boxshape_nm_{box,ellipse}_{split}_r2048.json`; log `native_raster/logs/boxshape_nm_2026-11-11.log`.

### BOXSHAPE-2 result (2026-11-11; all three gates PASS both splits)

`results/boxshape_nm_{test,sparse}.json`. Geometric arms ranked by 3-input ŝ (no log p̄m); LACE deployed.

| Split | Arm | AP50 | AP75 | AP50:95 |
|---|---|---|---|---|
| test | ellipse | 0.694 | 0.224 | 0.316 |
| test | filled box | 0.641 | 0.047 | 0.217 |
| sparse | ellipse | 0.735 | 0.324 | 0.368 |
| sparse | filled box | 0.712 | 0.081 | 0.277 |

LACE − ellipse: test AP50 +0.020 [0.017, 0.026], AP50:95 +0.027 [0.022, 0.032]; sparse +0.008 [0.004, 0.015], +0.035 [0.025, 0.045]. LACE − box: test +0.074 [0.062, 0.090], +0.126 [0.116, 0.136]; sparse +0.032 [0.024, 0.044], +0.127 [0.111, 0.143].
- Ellipse drops 0.007 AP50 test / 0.003 sparse vs BOXSHAPE (the log p̄m ranking signal), close to the tab:factors −log(p̄m) cost.
- Ellipse on D1 boxes (0.694) still beats every Table 1 baseline at AP50; trails Restor at AP50:95 (0.316 vs 0.333).
- tab:masker geometric rows, caption and footnote updated to these numbers (supersedes BOXSHAPE).

## Pre-registration SAMEBOX (2026-11-13; BEFORE any number)

Main-text "same boxes, different maskers" table (CVPR, replaces §SAM3 Masker; Tom 11-13). Adds the one missing interval: mask shape alone at identical ranking. Arms all on D1 boxes, all ranked by the 3-input ŝ (no log p̄m; T3 `no_logpm` scores, identical order): LACE masks (`preds_dino_swinl_s0_{split}_t3_no_logpm_r2048.json`), inscribed ellipse and filled box (`preds_boxshape_nm_{ellipse,box}_{split}_r2048.json`). Paired tile bootstrap (t1_boot N=2000, seed 0) of LACE-masks − ellipse and − box, AP50 / AP75 / AP50:95.
- **Gate:** every arm re-scores to the `boxshape_nm_{split}.json` values to 4 dp.
- **Disclosure:** arms and ranking already scored 11-11 (BOXSHAPE-2); only the intervals are new. No new predictions.
- Script `samebox.py`; outputs `results/samebox_{split}.json`; log `native_raster/logs/samebox_2026-11-13.log`.

### SAMEBOX result (2026-11-13; gate PASS both splits)

`results/samebox_{test,sparse}.json`. LACE masks − ellipse, identical boxes and 3-input ŝ ranking (mask shape only): test AP50 +0.012 [0.010, 0.017], AP75 +0.046 [0.035, 0.057], AP50:95 +0.024 [0.019, 0.029]; sparse +0.005 [0.001, 0.009], +0.075 [0.052, 0.098], +0.033 [0.023, 0.043]. LACE masks − box: test +0.065 [0.054, 0.081], +0.223 [0.206, 0.248], +0.123 [0.114, 0.133]; sparse +0.028 [0.021, 0.038], +0.318 [0.276, 0.360], +0.125 [0.108, 0.141]. Merged into paper.tex as tab:samebox, §"Same boxes, different maskers" (replaces "SAM3 Masker" heading), 2026-11-13.

## Pre-registration LACE-0 (2026-11-26, Tom: "yes, then run"; BEFORE any number): LACE with no crown-mask label setting any tuned quantity

Purpose: a "LACE (no mask labels)" row for Table 1 / fig:cost at 300 annotator-hours (box-only cost, same as BoxTeacher/Box2Mask). The gap to deployed LACE (0.714) is what the 108 val masks buy.
- **Detector:** D1 detections unchanged (selected on val box AP50; box-only).
- **Masker:** deployed fit `feat_cache/em_model_4p_fix.npz`, β = 0.5, kept on design grounds (β = 0 removes the repulsion step). Disclosure: β = 0.5 vs 0 was compared on masks historically (phase4 README, +0.041 mAP50 post grid fix; tab:components/tab:collapse ≤ 0.003 crown IoU with mean-centring on).
- **Inference, fixed a priori (no tuning):** α = 1 (prior as fitted), κ scale 1 (κ = 10, fit value), τ = 0.5 absolute on the posterior, no guided filter (r = 0: bilinear posterior at 2048), mask clipped to box; EDGE_FILL on. Render = `hunt/render.post4` with `{alpha 1, kappa 1, r 0, tau 0.5}`; s' = s · mean|2r−1| · mean(r) at α = 1, κ = 10.
- **ŝ:** same box-match target (`box50`, box IoU ≥ 0.5, greedy in s' order; crowd = predicted mask ≥ 50 % in canopy), same R3 machinery defaults (epochs 30, dropout 0.1, wd 1e-3), but selected by **val box AP50** (OOF, 15 runs = folds rr/perm1/perm2 × seeds 0–4) instead of mask AP50. Candidate features: all 24 R1 features (not R2's 14, which were mask-selected). Stage 2 backward elimination 24 → 1 for each of logreg, mlp1x8, mlp1x16, mlp1x64, mlp2x64. Reference = mlp2x64 on all 24. Admissible = box50, gain over s' box AP50 ≥ ref gain − 0.001, gain ≥ 0.005, n_pos ≥ 13/15; order fewest features, simplest model, higher gain. Confirm on fresh randomness (perm3/4/5 × seeds 5–9) walking the list (same thresholds vs fresh ref). If nothing is admissible or confirms: ŝ := s (detector score). Final = mean of 25 models (5 seeds × rr folds).
- **Look:** ONE look on test 439 and sparse 236, frozen `score_coco` (2048, maxDets 600, floor 0, canopy iscrowd). Primary: LACE-0 (ŝ). Secondary: LACE-0 ranked by s. Paired tile bootstrap (2000, seed 0) deployed LACE − LACE-0, AP50 and AP50:95.
- **Still uses polygons (disclosed):** OAM-TCD canopy polygons as ignore regions (detector training, background fit, ŝ crowd labels, box-AP crowd). Architecture choices (4 px lattice, mean-centring, registration fix) were developed with mask feedback; no mask label sets any value in this run.
- **Gates:** (G1) render wrapper with deployed spec reproduces `feat_cache/edgefix_val` masks byte-identical on 3 val tiles; (G2) box fasteval == `score_lite` box AP50 to 4 dp on LACE-0 val under s, s' and reversed order; (G3) `rerank._feats` on `dino_swinl_s0_val_mask.json` reproduces cached `rerankR1_val.npz` rows for 3 tiles.
- Script `lace0.py`; outputs `results/lace0/` (ledger `ledger.jsonl` = resume cache, `lace0.json`), masks `feat_cache/lace0_<split>/`, features `feat_cache/lace0_feats_<split>.npz`; log `native_raster/logs/lace0_2026-11-26.log`.

### LACE-0 result (2026-11-26; ONE look; gates G1/G2/G3 PASS; single detector run)

`results/lace0/lace0.json`, preds/scored `results/lace0/{preds,scored}_lace0_{test,sparse}_{shat,s}_r2048.json`, ŝ ledger `results/lace0/ledger.jsonl` (1,500 configs), masks `feat_cache/lace0_{val,test,sparse}/`, log `native_raster/logs/lace0_2026-11-26.log`.
- **Naming:** the box-selected ŝ is **R4** (R1–R3 = earlier ŝ versions, R3 = deployed; not to be confused with R4P, the 4 px masker refit). LACE-0 = D1 + edge fix + a-priori inference (α 1, κ 10, τ 0.5, bilinear) + R4.
- **R4 selection (val box AP50, base s' 0.7522, s 0.7546):** reference mlp2x64 × 24 gain +0.0060 (15/15); 812 admissible; first four candidates failed fresh confirm (fresh ref +0.0081); chosen **mlp2x64 on [s, log_marea, max_iou_higher, exg]**, fresh gain +0.0087 (15/15). Drops the posterior input log(p̄D); mask-aware only through predicted-mask area and ExG.
- **Test 439:** LACE-0 (R4) mask AP50 **0.694**, AP75 0.154, AP50:95 **0.292**, box AP50 0.744; ranked by s 0.675 / 0.142 / 0.281. Deployed − LACE-0 (R4): AP50 +0.020 [0.017, 0.025], AP50:95 +0.051 [0.048, 0.055]; deployed − LACE-0 (s): +0.039 [0.033, 0.047], +0.062 [0.058, 0.067].
- **Sparse 236:** LACE-0 (R4) 0.722 / 0.193 / 0.321, box 0.749; ranked by s 0.715 / 0.184 / 0.316. Deployed − LACE-0 (R4): AP50 +0.021 [0.015, 0.030], AP50:95 +0.082 [0.071, 0.093]; (s): +0.029 [0.020, 0.040], +0.088 [0.077, 0.097].
- **Read:** the 108 val masks buy +0.020 AP50 and +0.05–0.08 AP50:95, mostly boundary precision (AP75 0.271 → 0.154 on test without guided filter / box-relative τ). LACE-0 still leads every Table 1 baseline at test AP50 (BoxTeacher 0.664, Restor 0.648) but trails BoxTeacher, Detectree2 and Restor at AP50:95; on sparse it ties BoxTeacher at AP50 (0.722 vs 0.721) and trails at AP50:95 (0.321 vs 0.376). Not in paper.tex.
