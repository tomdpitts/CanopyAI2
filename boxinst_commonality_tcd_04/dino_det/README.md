# dino_det: LACE (D1 variant) — DINO-Swin-L detector (SelvaBox-style architecture) + LACE masker (2026-11-03)

**Naming (Tom, 2026-11-03):** D1 *is* LACE. Externally it is just "LACE"; in repo docs "LACE (D1 variant)". The earlier frozen-DINOv3 CenterNet-style detector + masker (0.655) is "LACE (CenterNet variant)". "D1" below is shorthand for the D1 variant / its detector stage.

<!-- TABLE1_CURRENT:begin (auto, results_439/mark_superseded.py) -->
> ⚠️ **SUPERSEDED 439 / sparse236 NUMBERS (marked 2026-10-27; table updated 2026-11-03).** Current headline table: [`TABLE1_CURRENT.md`](../results_439/TABLE1_CURRENT.md). Test-set numbers in this file are a historical experiment record (512 raster, pre-seam-fix baselines, 8 px masker or earlier models) and must not be quoted unless they match that table.
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

`tab:sparse236` references: LACE (3 seeds) 0.6785 / 0.319 / box 0.674; Detectree2 0.636 / 0.353 / 0.644; Box2Mask Swin-L 0.604 / 0.262 / 0.627. BoxTeacher has no sparse row (deferred in Tier B). Modal cost $0.31 (metered $106.90 → $107.21).
