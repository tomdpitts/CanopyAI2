# Tier B: B1 BoxInst, B2 BoxTeacher, B3 MAL head-to-head with LACE

<!-- TABLE1_CURRENT:begin (auto, results_439/mark_superseded.py) -->
> ⚠️ **SUPERSEDED 439 / sparse236 NUMBERS (marked 2026-10-27).** Current headline table: [`TABLE1_CURRENT.md`](../../results_439/TABLE1_CURRENT.md). Test-set numbers in this file are a historical experiment record (512 raster, pre-seam-fix baselines, 8 px masker or earlier models) and must not be quoted unless they match that table.
<!-- TABLE1_CURRENT:end -->

**Status 2026-10-27: SET UP, NOTHING RUN on Modal. §00 (scope and order, 2026-10-27) governs over §0, and §0 over everything below.**

## 00. Amendment 2026-10-27: B2 then B3; B1 not planned (Tom, before any Tier B number)

- **Order: B2 BoxTeacher → B3 MAL. B1 BoxInst is not planned.**
- **Why drop B1.** BoxTeacher is BoxInst plus a self-training teacher: the same CondInst architecture and box losses, and about 7 COCO mask AP higher. It also outperforms DiscoBox on COCO. B2, together with Box2Mask (TPAMI'24), therefore stands for the box-supervised system family. Paper wording claims representativeness by citation, never an upper bound: COCO rankings need not transfer to small, dense aerial crowns. Box2Mask Swin-L is the strongest of these on COCO at 42.5 mask AP, yet reaches 0.589 on the 439.
- **Revisit B1 (~$8–20, ~1 h) only if B2 lands close to LACE,** so that "an older baseline might do better here" cannot be raised.
- **Why B2 first.** It is the only Tier B row that is a full system, so it gives a Table 1 row, like Box2Mask.
- **What B3 adds.** B3 (MAL) is a box-to-mask module with no detector. It always needs boxes supplied (GT, or LACE s0 detections), so it is the like-for-like comparison with LACE's masker (the tab:gtbox protocol), not a Table 1 row.
- **B3 resolution is not unfair to MAL.** MAL crops each box with a 0.3 margin and resizes to 512² (ViT-MAE-B/16: 32×32 patches per box). For the median ~36 px crown that is ~2 px per patch, against LACE's fixed 4 px lattice; MAL's patch exceeds 4 px only for crowns above ~175 px. The compute asymmetry (a ViT-B forward + CRF per box, against one shared tile encoding + a closed-form E-step) favours LACE and is worth one line if B3 is reported.
- **Estimates (scaled from §8 to §0's schedule on 8×A100-40GB; unmeasured, replaced by the smokes' projection before HOLD is released):**

| row | what | hardware | expected epochs | wall-clock | cost |
|---|---|---|---|---|---|
| B2 | BoxTeacher Swin-B 3x, full system → Table 1 row | 8×A100-40GB | 8–20 | ~0.7–2 h | ~$12–30 |
| B3 | MAL ViT-MAE-B/16 labeler on GT + LACE s0 boxes (all training crowns) | 8×A100-40GB | 6–20 | ~0.8–3.5 h | ~$12–50 |
| — | 1-GPU gates ×2 + 2-GPU smokes ×2 → HOLD | 1–2×A100 | — | ~1–1.5 h | ~$9–14 |
| — | image builds + staging | CPU | — | ~0.5 h | ~$2–3 |
| — | predictions (B2: 439; B3: GT + LACE s0 boxes) | training container | — | included | ~$1–3 each |
| — | sparse-236 for B2 (deferred) | 1×A100 | — | ~30 min | ~$2–4 |
| **total** | | | | **~3.5–7.5 h** | **~$40–100** (§0 hard stop $72: may need raising, decide at HOLD) |

- **Code changed 2026-10-27.** `tierb_modal.PLAN` is now: stage, fetch_weights, gate B2, gate B3, smoke B2, smoke B3, HOLD, then B2 train/predict/finalize, then B3 train/predict/finalize (checked with `tierb_ctl.py plan`). B1's functions are kept, and `fetch_weights` still downloads its checkpoint. To revisit B1, re-add its four PLAN entries (comment at `PLAN`). Run `tierb_ctl.py deploy` before `start`, as after any edit.
- **LACE references re-pointed 2026-10-27 (Tom):** `lace_gate.py` now holds the lace4px_simple values (test s′ 0.6545 / 0.2803, seed 0 0.6556 / 0.2844; sparse 0.6785 / 0.3192, seed 0 0.6903 / 0.3258; GT boxes 0.7861). B3's paired GT-box diff reads `native_raster/hunt/test/gtbox4_lace4px_simple_percrown.jsonl`, which is added to the image.
- **LACE is only ever compared with its full s′ ranking (Tom).** `evaluate` asserts that every reference is a `product*` row. B3 masks on LACE boxes are ranked by the detector score s only. MAL's labeler has no confidence output; its published AP comes from phase-2 detectors trained on its pseudo-masks. So s is its native ranking, and no ranking is invented for it (a maskness variant was considered and rejected 2026-10-27 as arbitrary). The B3 s row is gated against LACE s′.
- **MAL min_obj_size 0 (Tom 2026-10-27; was 920).** B3 trains on every crown box LACE trains on: 54,184 of 54,191, with only MAL's own w, h > 2 px rule applied (verified against MAL's `datasets/voc.py` filter at the pinned commit). Gate G2 now expects 54,184. Released purpose: MAL is an offline COCO auto-labeler that crops each box and resizes it to 512² for the ViT. Tiny boxes upscaled about 10× carry little detail and give noisy MIL/CRF training signal, so the authors train on objects above 2,048 px² (about 45×45 px) and still label all objects at inference. Our median crown (about 36 px) is below that floor, a domain mismatch that favours LACE and should be stated if B3 is reported. Epoch length rises about 1.73× (warm-ups scale automatically), so B3's estimate rises to about $12–50.
- **BoxTeacher teacher post-NMS cap removed (Tom 2026-10-27).** The released teacher runs in train mode and keeps at most `POST_NMS_TOPK_TRAIN = 100` detections per 1024 crop for pseudo-masks. On our crops that leaves 4.5% of training boxes with no possible pseudo-mask (2.7% of crops exceed 100). `d2_run.uncap_teacher` sets the teacher's `post_nms_topk_train` / `_test` to 0, which FCOS treats as no cap; it is applied at every launch and in the gate, and `check_released_test_attrs` expects it. Teacher only: the student, pre-NMS top-k (1,000) and thresholds stay as released. No other baseline has a teacher, so this is disclosed as a deviation. A cap of 150, protocol density, was considered; it would leave 1.2%.
- **Pre-NMS top-k kept as released (Tom 2026-10-27):** 1,000 per level per 1024 subtile (the released value and AdelaiDet's default), 4,000 per whole tile (area-scaled). This is the same rule as Restor (restore the framework default, go no further).
- **Score floor:** B2 is scored at the 0.05 floor like the other baselines; add a floor-0 sensitivity row from the saved preds (CPU) after the run.
- **Unchanged:** the plateau schedule, budget stop, `lace_gate.py` auto-STOP if a baseline clearly beats LACE, A100 hardware (consistent with all other OAM-TCD compute), and sparse-236 deferred.

---

**Status 2026-10-15 (superseded by §00 for scope and order): SET UP, NOTHING RUN on Modal. §0 (convergence-driven schedule, budget) governs where sections disagree.** Tom approved B1–B3 on 2026-10-14 and asked for exactly
LACE's data, a clean subtiling strategy, dropout-proof Modal work, saved predictions, and a spend stop if any
baseline clearly beats LACE. This file is the dated amendment to `../PLAN.md` §13 (Tier B). It was written
before any Tier B number exists. Where it differs from the §13 draft, this file governs, and §13 points here.

Code: this directory. Local gates already run and passed: `gates/`. Modal: `tierb_modal.py` (app
`tcd04-tierb`, volume `tcd04-tierb-vol`), driven by `tierb_ctl.py`.

---

## 0. Amendment 2026-10-15: convergence-driven schedule and budget (before any Tier B number)

**This section governs wherever later sections disagree.** Tom set a target of about $75 USD while keeping the comparison fair.

- **Schedule, all three methods** (`plateau.py`, unit-tested in `gates/local_tests.json`):
  - LR is held at the released base after the released warm-up;
  - the val metric is evaluated every epoch: box AP50 on the 108 whole val tiles for B1/B2; mean per-crown mask IoU on the val GT boxes for B3;
  - improvement means a gain of more than 0.005, LACE's own early-stopping threshold;
  - 2 epochs without improvement drop the LR ×0.1, at most twice, matching the released two-step schedules;
  - 2 more such epochs after the second drop stop training;
  - minimum 4 epochs, maximum 20 as a budget cap. A run stopped by the cap while still improving is flagged as a possible lower bound;
  - selection is the best val epoch.

  This replaces the fixed 36-epoch schedule (§4) and MAL's 10-epoch warm-up-plus-cosine (§5), both disclosed. Every LR drop is driven by convergence, so the Box2Mask defect (stopping before any drop) cannot recur.
- **Warm-ups** keep their released fraction of the released 1x length (12 epochs = 2,376 iterations): LR warm-up 26 iterations; BoxInst pairwise and BoxTeacher pseudo-mask warm-ups 264.
- **B3 boxes:**
  - GT boxes for val, test and sparse (the main mask-vs-mask comparison, paired against LACE per crown);
  - LACE **seed 0's stored** boxes, gated against LACE seed 0's full s′ row (lace4px_simple, AP50 0.6556 / AP50:95 0.2844; §00. Was the 8 px det-score row 0.6101 / 0.2579).
  - LACE is never re-run. LACE s1/s2, SelvaBox and LACE-sparse boxes are deferred (each ~$1–3 later).
- **Sparse-236 for B1/B2 is deferred.** The best checkpoints are kept; it can be predicted later for about $2–4 each.
- **Hardware:** A100-40GB (gate G5 proves the real per-GPU batch fits). 8-GPU containers request 24 cores / 128 GiB.
- **Speed:** TF32 matmuls on Ampere, as in torch ≤ 1.11 and the authors' torch 1.9.1. Prediction runs in the training container, saving a second start-up.
- **Smokes:** B3 and B2 only, on 2 GPUs at the released per-GPU batch, with an 8-tile predict (16 until 2026-10-27; the predict-kill rehearsal now fires at 2 tiles, the same 1/4 point). Each smoke asserts that the plateau rule fired its LR drops and that the logged LR fell ×0.1. B1 uses the same detectron2 machinery; its 1-GPU gate still runs.
- **Budget:** a live ledger of billed minutes × container rate, CPU and memory included. A hard stop at $72 (budget $75) sets STOP; everything is resumable.
- **Expected total ~$55–75 USD.** The smokes report the measured projection at 8 / 12 / 20 epochs (B3: 6 / 10 / 20) before the review pause.

## 1. What runs

| id | method | released checkpoint (COCO, box-only) | what it is compared with |
|---|---|---|---|
| B1 | BoxInst R-101 3x (AdelaiDet, CVPR'21) | `BoxInst_MS_R_101_3x.pth`, 33.0 val / 33.2 test-dev mask AP | LACE's system row (Table 1) |
| B2 | BoxTeacher Swin-B 3x (CVPR'23) | `boxteacher_swin_base_aug_3x.pth`, 40.2 / 40.5 | LACE's system row (Table 1) |
| B3 | MAL phase-1 labeler, ViT-MAE-B/16 (CVPR'23) | `epoch=6-arch=vit-mae-base-16-...ckpt` (the only released labeler) | LACE's masker, on identical boxes |

- **B1 and B2 are full systems**, a detector plus a mask head. They cannot be prompted with a box (PLAN §6).
- **B3 is a box-to-mask module only.** It is evaluated like LACE's masker:
  - on ground-truth boxes, the tab:gtbox / A4 protocol;
  - on LACE's own boxes (seeds 0–2), ranked by LACE's detector score, which is the matched-ranking comparison with LACE's masker (0.614);
  - on the seam-fixed SelvaBox boxes, matching A2/A3;
  - on sparse-236, with the GT boxes and LACE's sparse boxes.
- **Backbones.**
  - B2 uses the strongest released backbone (Swin-B), which answers §13 decision 2.
  - B1 uses R-101, as registered in §13. A stronger BoxInst exists: R-101-BiFPN-dcni3, 35.0 test-dev. It is not run. BoxInst is the seminal-reference row, and B2 is the strongest box-supervised system.
- **One seed per method (seed 0).** This matches every other baseline. Comparisons are made against LACE's 3-seed band and its worst seed.

## 2. Data identity: the literal LACE images, by ID and by pixel hash

`splits.py` writes `splits_manifest.json`. All of the following were checked on 2026-10-14:

- **Counts.** 792 train / 108 val (the `partition` field of `train_tiles_gt.json`, the file LACE's detector and masker read), 439 test (`test_gt.json`) and 236 sparse.
- **Disjointness.** Train and val are disjoint, and neither overlaps test or sparse. The 16 sparse tiles that are also test tiles are byte-identical in both local copies.
- **Val GT.** `val_gt.json` keys equal the 108 val IDs.
- **Independent LACE evidence.** The listing of LACE's deployed feature cache `tcd04-phase4-vol:feat_4p_train` (`lace_feat_4p_train_listing.txt`) is exactly the 900 train+val IDs.
- **Image IDs.** Every tile's HF `restor/tcd` image_id (from its meta) equals the phase-4 manifest's. That manifest is the join LACE's features were extracted by.
- **Hashes.** A file SHA-256 and a decoded-RGB SHA-256 are recorded for every local image.

**One local file is empty.** `data/tcd/train/tcd_tile_1878.tif` is a training tile (HF image_id 1) and is 0 bytes locally. LACE never read it locally: LACE's pixels came from the HF copy on `canopyai-deepforest-data`. Tier B therefore stages all 1,559 unique tiles from that same HF copy (`tierb_modal.py::stage`):
- each tile's pixel hash must equal the local file's (1,558 tiles), or the stage aborts;
- tile 1878 is recorded from HF, the only copy that exists;
- PNG round-trips are hash-checked;
- every container re-verifies the pixels it uses against the stage record (`tierb_common.verify_split`; training does this at every launch).

## 3. Supervision and canopy: LACE's, exactly

- **Boxes.** Training boxes are `train_tiles_gt.json` `boxes`, the list LACE's detector trained on. No crown polygon is ever read for training. The mapped training instances carry only `gt_boxes` and `gt_classes`, which gate G4 asserts.
- **Canopy is IGNORED, not background (B1/B2, patch P1).** This is LACE's own rule (`dapt/head.py::focal_heatmap_loss` with `canopy_cell_mask`):
  - a location whose stride cell is more than 50% canopy contributes no negative classification loss;
  - positives are always supervised;
  - the loss normaliser is unchanged.

  Canopy polygons come from `train_tiles_gt.json`, rasterised by LACE's own `canopy_px`.
- **Proof.** Local test: the cell rule equals LACE's `canopy_cell_mask` at all five FPN levels. Gate G6: without canopy, the loss is bitwise equal to the released loss.
- **Why not the Box2Mask approach.** Box2Mask blacked out canopy pixels using `canopy AND NOT crown`. That changes the test-time imagery distribution. It also writes crown-polygon edges into the training pixels, which leaks mask information to colour-affinity losses. The ignore rule gives these baselines the same canopy semantics as LACE, on intact imagery. (This answers §13 decision 4.)
- **B3 (MAL)** trains on crown boxes only, so canopy has no role. It is not an instance class.

## 4. B1/B2 protocol

**Training crops: a jittered 2×2 partition** (`crops.py`). The released recipes train on ~1 MP images; a 1024² crop has the same pixel count.

- **Per crop:**
  - a scale s is drawn from the released multi-scale set, taken relative to the released test size (640–800 / 800, so s ∈ {0.80, …, 1.00});
  - the tile is resized to `int(2048·s+0.5)`;
  - one quadrant is taken at offset 0 or L−1024 on each axis;
  - a horizontal flip (released p = 0.5) is applied *after* the crop.
- **Coverage.** Every training pixel is seen at least once and at most twice per epoch, for every combination of scales (proved locally). This is the property LACE's whole-tile training has; uniform random crops starve the tile frame.
- **Epoch arithmetic.** One epoch is 792 tiles × 4 = 3,168 crops = 198 iterations at batch 16. detectron2's sampler consumes exactly one permutation per epoch.
- **Empty crops are kept.** 15.7% of quadrants hold no crown. LACE trains on those pixels. The released code crashes on empty images, which patch P3 fixes.

**Schedule.**
- The released 3x is 36 COCO epochs. Tier B runs 36 epochs over the 792 tiles (LACE runs up to 40, early-stopped), which is 7,128 iterations.
- Every iteration landmark keeps its released *fraction* of the schedule:
  - decays at 5,544 and 6,600;
  - LR warm-up 26 iterations;
  - BoxInst pairwise warm-up and BoxTeacher pseudo-mask warm-up 264 iterations each. On a fresh start, the checkpoint's saturated `_iter` buffers are reset so these warm-ups run; they are never reset on resume.
- Every scheduled decay happens. There is no early stopping and no wall-clock stop; these were the Box2Mask defects (audit #1–2).

**Optimiser.** Released, at batch 16 = 8 GPUs × 2. That is the same per-GPU batch as the release, which matters because the CondInst mask branch uses plain BN.
- B1: SGD 0.01, momentum 0.9, wd 1e-4, detectron2 default param groups (AdelaiDet).
- B2: AdamW 1e-4, wd 0.05, BoxTeacher's own param groups.
- B2 also keeps its released student colour augmentation (`strongv2`) and EMA teacher (0.999, released ramp `min(0.999, 1−1/(k+1))`), with inference by the teacher.
- The EMA ramp is the one landmark *not* rescaled. It is per-step smoothing of the weights, so its horizon in gradient steps is kept (~700 steps); rescaling it would change what the teacher averages, not when.

**Resumes.** After a resume, the data order and augmentation RNG are seeded from the resume iteration (cfg.SEED = start_iter), never re-seeded at 0. Otherwise epoch 0's order would be replayed.

**Initialisation.** From the released COCO box-only checkpoint. Only the 80-way `cls_logits` is re-initialised, asserted key-by-key (G3).

**Selection.**
- After every epoch: val box AP50 on the 108 whole val tiles at 2048, maxDets 600, floor 0.05, canopy crowd.
- The scorer is `score_lite.py`, the frozen scorer's crowd arm, parity-gated: identical on LACE s0 and Box2Mask Swin-L in every AP and detection count (`gates/score_lite_parity.json`).
- The best checkpoint and `best.json` are persisted together, so a resume can never re-select (the Box2Mask resume trap). Ties go to the earlier checkpoint.
- **`model_best.pth` is authoritative.** It is written first, then `best.json`, then the `evals.jsonl` line; a kill between the writes is reconciled at the next launch.
- **An eval interrupted by a preemption is re-run on resume** (`before_train`), so no epoch silently drops out of selection.
- **Evaluation uses rank 0's model.** DDP runs with `broadcast_buffers=False`, so mask-branch BN statistics differ per rank. Every rank evaluates with rank 0's buffers, the model that is saved and predicted, and then restores its own.

**Inference geometry: both run, primary chosen on val.**
- Native 0.1 m/px.
- `whole` = the 2048 tile in one pass (LACE's and Restor's geometry; no seams).
- `sub` = 3×3 1024 subtiles at stride 512, stitched by the seam-fixed stitcher with the same parameters as the DT2/Box2Mask rows.
- **Pre-registered rule:** the primary geometry is the one with the higher val mask AP50 on the selected checkpoint; a tie goes to whole. The other geometry is reported as a sensitivity row.
- Why both: whole-tile avoids stitching entirely, but trains on 1024 crops and tests on a 2048 canvas, where GroupNorm sees a larger canvas. Subtiles match the training canvas but need stitching. Letting val choose means neither artefact can handicap the baseline, and test is never consulted.

**Inference settings.** These are aligned to the protocol, as Restor's were; released values are kept elsewhere.

| setting | value | why |
|---|---|---|
| post-NMS cap | 600 | protocol |
| pre-NMS top-k, whole | 4000 per level | the released 1000 per ~1 MP, area-scaled |
| pre-NMS top-k, sub | 1000 per level | released |
| FCOS candidate threshold | B1 0.0025, B2 6.25e-6 on cls·ctr | chosen so every detection that can reach a reported 0.05 is produced. B1 reports sqrt(cls·ctr); B2 reports sqrt(sqrt(cls·ctr) × mask score), with mask score ≤ 1. The released 0.05 on cls·ctr would floor B1 at 0.22 and B2 at 0.16. The scorer's own 0.05 floor then decides. |
| NMS | 0.6 | released |
| mask threshold | 0.5 | released |

These overrides are applied only at inference. Training (including BoxTeacher's teacher) always sees the released values, which is asserted after every eval.

**Saved predictions.**
- Per tile: model boxes (unrounded), scores (BoxTeacher also keeps its pre-mask FCOS score) and masks as RLE at 2048.
- For `sub`: per-subtile files in the stitcher's own format.
- Rescoring never needs re-inference.

## 5. B3 protocol (MAL)

MAL's own CLI defaults are those of the released labeler. Fine-tuning starts from that labeler, on LACE's 792-tile crown boxes. MAL never reads a mask in phase 1. Changes are made only where our data forces them, each by a rule:

- **`--min_obj_size 920`.** The released 2048 px² removes 42.18% of COCO's valid training boxes, but would remove 61.49% of ours (median crown about 36 px across). 920 removes 42.16% (`gates/coco_min_obj_size.json`, computed from `instances_train2017.json`). MAL keeps 31,341 of 54,184 boxes, asserted in G2.
- **Loss warm-ups** `--loss_mil_step` / `--loss_crf_step`. The released 1000 / 4000 steps are 6.44% / 25.77% of a COCO epoch-0 at 8 GPUs × 4 (15,522 steps). The same fractions are kept of our epoch-0, about 980 steps, giving 63 / 252.
- **`--strategy ddp`** instead of `ddp_sharded`. The arithmetic is identical; OSS only shards optimiser state.
- **Box clipping (protocol).** MAL assumes in-image boxes. For a box with x0 or y0 < 0, its box-mask slice `[y0:y1+1, x0:x1+1]` counts from the far end and comes out empty, so the output mask is empty: a guaranteed false positive.
  - 1–3% of LACE's boxes reach past the tile frame (6 test GT boxes too).
  - Every box is clipped to the tile before MAL sees it. Boxes already inside pass through as the identical floats; a box that clips to nothing gets an empty mask.
  - Scoring keeps the original boxes and scores.
- **Data workers.** `--num_workers_per_gpu 1`, i.e. 8 per rank. MAL multiplies by the GPU count per rank, so the released 2 would give 16 per rank, 128 in total.
- **Released and unchanged:**
  - LR rule 1.5e-6 × GPUs × batch (8 × 4, so 4.8e-5);
  - 10 epochs, 1 warm-up epoch, cosine to 0.2·lr;
  - fp16, EMA 0.996, MIL weight 4, CRF weight 0.5;
  - inference by the student, threshold 0.1;
  - test crop margin 0.3 per side, with the wrap-around crop (their deliberate choice: padding pixels damage the CRF).

**Selection.** After each epoch:
- that epoch's weights are saved first;
- then the val mean per-crown IoU on the GT boxes of the 108 val tiles is computed (gtbox_lib validity rule);
- the best epoch is kept.

This uses val masks, as LACE's inference knobs did (tab:masker) and as MAL itself did (its released labeler is a selected epoch). Disclosed.

**Inference.**
- MAL's test path, re-implemented tile-batched and gated **bitwise** against MAL's own `main.py --box_inputs --label_dump_path` on real boxes (G5), with the input tensors bitwise equal to MAL's transform (G6).
- ToTensor is reproduced through a CPU-computed lookup table. CUDA's `x.div(255)` by a CPU scalar is not bitwise equal to it.
- The per-box path is MAL's own, in its forward batch size of 4. Batch composition differs from a whole-set MAL CLI run (tile tails can be smaller than 4), and GEMM algorithm choice depends on batch size, so this is the same per-box computation, not a promise of identical last bits on every box.
- **Resume.** The Modal parent picks the newest *loadable* `epoch=N.ckpt` once and passes it to every DDP rank.
- Saved per box: the binary RLE at 2048 *and* the soft pasted crop (uint8), so any re-threshold is CPU-only.

**Box sets** (`prep_local.py`, sha256 in `tierb_data/DATA_MANIFEST.json`):

| set | source | count |
|---|---|---|
| GT boxes | val / test / sparse | 7,041 / 25,705 / 14,937 |
| LACE s0–2 | `guided_ablate_inputs/detscore_s*.json`, the A3 source | 159,687 / 162,017 / 177,598 |
| SelvaBox | seam-fixed, the A2/A3 source | 307,500 |
| LACE sparse s0–2 | — | 79,562 / 86,784 / 90,526 |

**Finalisation** is CPU-only with the frozen code:
- `score_coco` for the box sets, ranked by the detector's own score;
- `gtbox_lib` per-crown IoU on GT boxes, paired against LACE's stored per-crown IoUs (`native_raster/hunt/test/gtbox4_lace4px_simple_percrown.jsonl`, 4 px masker, §00; was `gtbox_lace_guided_r2048.jsonl`) with A4's tile bootstrap;
- A4's SAM numbers carried as reference.

## 6. Every departure from the released code

Patches are applied by monkeypatching at import, and each is refused unless the target file's SHA-256 equals the pinned file (`d2/d2_patches.py`).

| | kind | what | proved by |
|---|---|---|---|
| P1 | protocol | canopy ignored in the FCOS classification loss (§3) | G6 + local test |
| P2 | neutral | hands the per-crop canopy raster to P1 (training only) | G6 |
| P3 | neutral | empty-crop branches in `add_bitmasks_from_boxes` / `add_pseudo_bitmasks` / `pseudo_mask_pairwise_affinity` (released code crashes: `torch.stack`/`cat` of an empty list, BoxTeacher issue #7) | G5 (dense+empty, empty+empty batches, finite grads) |
| P4 | protocol | BoxTeacher's teacher sees each crop at the test (native) scale, ×1/s, which is the released semantics. The released code derives the size from MIN_SIZE_TEST, which would be wrong on our crops. | G9 |
| P5 | neutral | mask post-processing in chunks of 64 instances. At 2048² × 600 the released code overflows int32 in `aligned_bilinear` (BoxTeacher issue #11) and needs ~10 GB. | G7: masks and boxes bitwise equal to the released postprocess. B2's mask-rescored scores agree to ≤1e-6 relative; that is a CUDA sum whose block split depends on chunk size. |
| P6 | neutral | BoxTeacher's pseudo-label assignment loop runs on CPU copies. It is the same loop, with the same GPU `argsort` order for ties; the released version does ~3 GPU syncs per (pred, GT) pair, about 1 s/iter on dense crops. | local test (2,000 cases) + G8 (random + real teacher outputs) |
| P7 | neutral | BoxTeacher's mask head all-reduces two counters only when its GPU has proposals. A GPU whose two crops are both empty (~18% of 8-GPU iterations here) would skip them and desynchronise DDP, giving a hang or corrupted gradients. The same two all-reduces are now issued, with values discarded. | G5: identical collective count for dense, mixed and all-empty batches |

Config changes:
- `NUM_CLASSES 1`;
- `BOTTOM_PIXELS_REMOVED 0` (a COCO bottom-bar artefact);
- `FILTER_EMPTY_ANNOTATIONS False`;
- the schedule and inference settings of §4.

Build and environment fixes, none of which changes behaviour:
- NVIDIA's stale CUDA apt list is removed before apt.
- pip is pinned to 24.0, because PL 1.8.1's metadata is refused by pip ≥ 24.1.
- detectron2 is installed with `setup.py develop`.
- mmcv-full is installed with `--no-deps`, because it would otherwise pull OpenCV 5 + numpy 2; the build asserts cv2 4.5.5 and numpy 1.23.5.
- `NCCL_ASYNC_ERROR_HANDLING=1`, so a hung collective crashes rather than hangs.
- The scorer image is pinned to the local environment that produced every published row (py3.10, numpy 2.2.6, Pillow 12.3.0, pycocotools 2.0.11). `finalize` re-scores a stored LACE row and requires **exact** equality before it scores any Tier B file; a mismatch sets STOP.

Source fixes (`tierb_modal.py::_SED`):
- `PROJECTS={}` (issue #9);
- AdelaiDet `ml_nms` CUDA removed (THC, unbuildable on torch ≥ 1.11, never called);
- unused `turtle` / `cv2.log` imports removed;
- the EMA hook's `.module` made single-GPU-safe (issue #3).

Environment pins:
- **d2 image:** torch 1.12.1/cu113, py3.10, numpy 1.23.5, Pillow 9.5.0, fvcore 0.1.5.post20221221, iopath 0.1.9.
- **MAL image:** PL 1.8.1, torchmetrics 0.10.2, fairscale 0.4.13, mmcv-full 1.7.0, timm 0.6.13, numpy 1.23.5. MAE weights are baked in, because MAL downloads them at every model build otherwise.

## 7. Spend stop: "clearly beats LACE" (`lace_gate.py`)

The margin is 0.010 AP, about 4× LACE's seed SD at AP50. A row clearly beats LACE if either:
- AP50 ≥ LACE + 0.010; or
- AP50 ≥ LACE **and** AP50:95 ≥ LACE + 0.010.

This is checked on **every** reported row: both geometries, test and sparse. Each row is compared against the LACE row with the same ranking rule:

| comparison | LACE AP50 / AP50:95 | trigger at AP50 |
|---|---|---|
| B2 test (LACE own s′, 3-seed mean) | 0.6545 / 0.2803 | 0.6645 |
| B2 sparse (LACE own s′) | 0.6785 / 0.3192 | 0.6885 |
| B3 on LACE seed-0 boxes, test (B3 ranked by s; vs LACE seed 0 s′, §00) | 0.6556 / 0.2844 | 0.6656 |
| B3 on LACE seed-0 boxes, sparse (deferred) | 0.6903 / 0.3258 | 0.7003 |
| B3 on GT boxes (test, 366 tiles) | — | paired mean IoU diff ≥ +0.010 with CI above 0 (LACE 0.7861) |

(Re-pointed 2026-10-27 to lace4px_simple; the 8 px values were 0.6470 / 0.2778, 0.6775 / 0.3215, 0.6101 / 0.2579, 0.6410 / 0.2946, GT 0.7477. LACE is always compared at its full s′, §00.)

If any row triggers, `finalize` does two things:
- it sets the global STOP flag, so running jobs checkpoint and exit and nothing new starts;
- it sets a persistent `GATE_TRIGGERED` latch.

`unstop` alone cannot release the latch. Only `tierb_ctl.py override-gate <method> "reason"` can, and only on Tom's decision; the reason is logged.
- The override is **per method**: releasing B3 never releases a later B2 or B1 trigger.
- A re-run of an already-finished `finalize` re-latches, unless that method was overridden.

**The order is B3 → B2 → B1**, cheapest and most likely to trigger first. If B3 triggers, B2/B1 never start. Note that A4 already shows zero-shot SAM 3 beating LACE on GT boxes by 0.041, so a B3 trigger is plausible.

## 8. Dropout-proofing

- **Deployed app, spawned calls.** Each step spawns the next when it finishes, so nothing depends on the laptop.
- **Idempotent steps.** Every step has DONE markers and resumes from what is already on the volume:
  - B1/B2 checkpoint every epoch (198 iterations, minutes), writing atomically (tmp + `os.replace` for both the file and `last_checkpoint`). A torn `last_checkpoint` is repaired to the newest loadable file.
  - MAL checkpoints at epoch ends only (its LR step counter restarts each epoch). A dropout costs at most one epoch.
  - Prediction is per tile.
  - Finalisation is per output file.
- **Retries.** GPU steps retry (8×); Modal also restarts preempted inputs.
- **Duplicate spawns.** A heartbeat lock in a Modal Dict makes a duplicate spawn exit rather than race the running call.
- **No retry loops.** A step that fails 3 times in a row from the same checkpoint is marked FAILED (preemptions don't count).
- **Commits.** The volume is committed every 2 min, and at every new checkpoint and every new best.
- **Logs.** Full logs live on the volume (`runs/logs/`), streamed line by line and never through `tail`. Progress JSON is written every 20 iterations.
- **Watchdog.** A child silent for 45 min (no log line, no progress change, no new checkpoint) is killed and counted as a failure. NCCL async error handling turns a hung collective into a crash. A container that vanishes without an exit code and without having advanced its checkpoint also counts. No hang can burn more than about 45 min.
- **STOP** (`tierb_ctl.py stop`, or the gate).
  - B1/B2 training checkpoints at the next 10-iteration boundary and exits; it is killed if not out within 15 min.
  - Everything else (MAL training, all predicts, gates) is killed. Their outputs are per epoch or per tile, so nothing is lost.
  - Nothing new spawns.
- **The smoke steps rehearse all of this** before any real run, on **2×A100** at the released per-GPU batch (2 images for B1/B2, 4 crops for B3). Per-GPU work and memory therefore equal the real 8-GPU run, and DDP collectives (including P7's empty-GPU path) are exercised. Smoke prediction covers the first **16 val tiles**. Projections scale measured eval and predict time by 2/8 GPUs and to the full tile and box counts. Each phase asserts that its kill or stop actually happened.
  - **d2:**
    - STOP lands on a 10-iteration boundary;
    - SIGKILL *during* the epoch-end eval of iteration 80;
    - the resume re-runs that eval;
    - evals land exactly at {40, 80, 120};
    - prediction is SIGKILLed once 2 of the 8 tiles are written, then resumed.
  - **MAL:**
    - SIGKILL in epoch 1 after epoch 0 is checkpointed;
    - the resume continues without replaying (global step after epoch 1 = 60);
    - losses are finite after the resume;
    - evals cover {0, 1} exactly once.
  - Both smokes write measured seconds per iteration, eval and predict, and the projected hours and USD for the full run.

## 9. Gates

**Local, done** (`gates/`):
- `splits_manifest.json` checks;
- `score_lite_parity.json` (PASS);
- `gtbox_pure_parity.json` (PASS, 6 tiles / 468 crowns);
- `coco_min_obj_size.json`;
- `local_tests.json` (PASS: crop coverage, P6 on 2,000 cases, the canopy rule at 5 levels, gate boundaries).

**Modal 1-GPU gates** (`d2/d2_gate.py`, `mal/mal_gate.py`): G0–G10 for d2 and G0–G7 for MAL, as listed in those files. They cover:
- environment and pinned sources;
- pixel hashes and checkpoint keys (MAL: no extra keys);
- mapper alignment against the tile;
- forward/backward on empty and dense crops, with the collective count identical across batch types;
- the P1/P4/P5/P6/P7 equalities;
- MAL CLI parity (run before the in-process PL fit, in a clean environment);
- inference on 2 val tiles.

Any failed assertion stops the chain.

**Modal 8-GPU smokes** (§8). These also produce the measured cost projection.

## 10. Run book

```bash
cd boxinst_commonality_tcd_04/box_supervised_baselines/tierb
PY=../../../.venv/bin/python
$PY tierb_ctl.py plan
$PY tierb_ctl.py deploy        # builds 3 images (first build ~20-40 min, CPU only)
$PY tierb_ctl.py start         # stage -> weights -> 3 gates -> 3 smokes -> HOLD
$PY tierb_ctl.py status        # read the gates' and smokes' reports; projected cost is in status
$PY tierb_ctl.py pull          # copy reports into tierb/results/<timestamp>/ (same session)
$PY tierb_ctl.py continue      # only after review: B3 train/predict/finalize -> B2 -> B1
$PY tierb_ctl.py stop "why"    # at any time
$PY tierb_ctl.py unlock        # after a hard `modal app stop` (else locks expire in 3 min)
$PY tierb_ctl.py reset-attempts <key>   # only before a deliberate retry of a FAILED step
```

**Cost to reach HOLD** (cumulative; HOLD itself costs nothing): 3 one-GPU gates (~$1 each) plus 3 two-GPU smokes (~$3–5 each), roughly **$13–20** in total, plus a few dollars of CPU for staging and image builds.

**If the Google Drive download of the MAL labeler hits a quota page,** `fetch_weights` fails cleanly. Download it manually and `modal volume put tcd04-tierb-vol <file> weights/mal_vit-mae-base-16_coco_epoch6.ckpt`. The step then only hashes it.

**Cost (to be replaced by the smokes' measurement before HOLD is released).** Assumes 8×A100-80GB at an assumed $2.50/GPU-h.

| item | basis | estimate |
|---|---|---|
| B2 | ~0.6–1.3 s/iter (BoxTeacher's own log shows 0.95 s/iter for R-101 on 8 GPUs) plus 36 evals | ~2–3.5 h, ~$40–70 |
| B1 | — | ~1.5–2.5 h, ~$30–50 |
| B3 | ~10k steps plus 10 val passes | ~1–2 h, ~$20–40 |
| predict | — | ~$5–10 each |
| smokes / gates | — | ~$25 together |
| **Total** | | **~$130–230** |

This is above the §13 draft's $10–30 per method. The draft assumed single-GPU short schedules. The released per-GPU batch (8×2) and full schedules are what make these runs defensible.

## 11. Asymmetries to state, not hide

1. **One seed per baseline.**
2. **Pretraining differs.**
   - B1: ImageNet R-101, then COCO boxes.
   - B2: ImageNet Swin-B, then COCO boxes.
   - B3: MAE ViT-B, then COCO boxes.
   - LACE: frozen DINOv3. LACE trains about 4M parameters; these train 50–110M.
3. **Selection signals differ.**
   - B1/B2 select on val **box** AP50, as LACE's detector does.
   - B3 selects on val mask IoU, as LACE's inference knobs and MAL's own release did.
4. **Box sources differ.** B1/B2 `whole` reports the model's own boxes; `sub` reports mask-derived boxes (the stitcher's, as for DT2/Box2Mask).
5. **Exposure differs.** B1/B2 see every training pixel 1–2× per epoch, for 36 epochs. LACE sees 1× per epoch, for 15–20 epochs until its best and 25–30 in total. This favours the baselines.
6. **Licences.** AdelaiDet and MAL code are non-commercial, and MAL's weights are CC-BY-NC-SA. This is research use, and should be cited as such.

## 12. Independent review (2026-10-15, before anything ran)

Three reviewers audited the code against the pinned upstream sources (BoxTeacher@6692d92 and its vendored detectron2/AdelaiDet, MAL@0ab9be4, PL 1.8.1, torch 1.12.1, Modal 1.5.4). They found 10 blockers, about 20 majors and minors. All are fixed above.

The blockers:
- the app module listed a local directory at import, so no container could start;
- BoxTeacher's 8-GPU collective mismatch on empty crops (P7);
- three d2 gate bugs: EventStorage, G9 shape pairing, G7 reduction order;
- MAL resume crashing on PL's checkpoint names;
- CUDA `div(255)` not bitwise ToTensor;
- the MAL CLI-parity subprocess inheriting stale DDP env vars;
- pip/PL 1.8.1 metadata;
- an unbounded-hang path costing up to 9 × 24 h.

The majors that change protocol rather than plumbing:
- B2's score floor (§4);
- rank-0 BN buffers at eval (§4);
- MAL box clipping (§5);
- the scorer-environment parity check (§6).

Re-checked locally after the fixes: syntax, container-style import of the app, `gates/local_tests.json` (now including MAL clipping).

**Second pass**, on the fixed code: 1 blocker, 1 major and 5 minors, all fixed.
- **Blocker:** the MAL gate's parity tile had two GT boxes past x = 2048, so its no-clip assertion could never pass. It now picks `tcd_tile_1784`, 56 boxes, none clipped.
- **Major:** a smoke kill could be delayed behind a blocking volume commit and miss its window for good. Commits are now requested from the background thread, never awaited, and watch/STOP run first.
- **Minors:**
  - lock leaks on early returns;
  - the predict progress token missed `whole` outputs;
  - the gate override is now per method;
  - smokes are STOP-safe and retry-safe, with per-phase completion records;
  - `start` now reports HOLD.
