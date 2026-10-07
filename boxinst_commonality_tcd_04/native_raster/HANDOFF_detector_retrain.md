# Handoff — LACE detector retrain (localisation) and the 2048 re-base

Written 2026-10-06 for an agent picking this up cold. Tom (DPhil, author of the LACE paper) owns every decision marked **PENDING**. Read this, then the sections of [`NATIVE_RASTER.md`](NATIVE_RASTER.md) it points to, before doing anything.

## 1. Where things stand

LACE is box-supervised tree-crown instance segmentation: frozen DINOv3 features, a CenterNet detector head, and an EM "commonality" masker that turns each box into a mask. The paper scores masks on a 512² raster. The native-raster campaign re-scored everything at the native 2048² raster (0.1 m/px). All of it is gated against the published 512 files and recorded in `NATIVE_RASTER.md` (Phase A, Phase C, and the 2026-10-06 sections).

Headline at 2048, canopy-neutral mask AP50, 439 test tiles:

| row | 512 | 2048 |
|---|---|---|
| LACE, 3 seeds | 0.6625 ± 0.0012 | **0.6300 ± 0.0033** |
| Restor Mask R-CNN, rpn2000 | 0.6605 | 0.6846 |
| Restor Mask R-CNN, rpn1000 (the paper's comparator) | 0.6255 | **0.6480** |
| DetecTree2 | 0.5969 | 0.6143 |

LACE is first at 512 and third at 2048. Tom's goal: **beat Restor rpn1000 at 2048 mask AP50**, which needs about +0.018. Higher-IoU metrics are out of scope because of the 8 px lattice.

What the 2026-10-06 diagnosis established (details in `NATIVE_RASTER.md`):
- **LACE loses on precision, not recall.** Its max recall is higher than rpn1000's (0.775 vs 0.744). At recall 0.6 it carries ~1,430 more false positives, about two-thirds of them *mislocated*: on a real crown, mask IoU 0.1–0.5. See `fp_anatomy_test.py`.
- **Localisation is the lever.** On val, snapping LACE's own boxes to GT gives +0.025 AP50 for existing hits and +0.10 including near-misses (`oracle_boxes_val.py`).
- **Training-free fixes are exhausted.** Size calibration gives +0.001 box AP50 on a held-out half. Removing the box clip ("soft box") is negative at every setting, because the clip is a regulariser the masker needs. No decode change helps by more than 0.002.
- So the remaining route is **retraining the detector head for tighter boxes**. That is the task.

## 2. The task: detector retrain pilot (seed 0), then adopt or stop

The detector is `Detector4Phase` (`modal_tcd_multiseed/phase4/phase4_lib_tcd.py`), a Detector8 head on cached 4-phase L24 features `(1024, 256, 256)` at stride 8. Training is `train_4p` → `train_detector_tiles.train` with `detector.det_loss`: focal heatmap + smooth-L1 offset + smooth-L1 log-size ×0.1 + GIoU ×1, all at **the single centre cell** per crown. Checkpoints are selected and early-stopped on **val box AP50**, with min 12 epochs, patience 2 and eval every 5; seed 0 peaked at epoch 20. The Modal entrypoints are `modal_tcd_multiseed/phase4/phase4_modal.py::train_eval_4p` / `band_4p`. `train_4p` **skips if `det_{tag}.pt` exists**, so new runs need new tags.

**Configuration decisions — PENDING Tom.** These are the suggestions he was given. Confirm his choices before any GPU spend.

| # | decision | suggested |
|---|---|---|
| 1 | checkpoint/ES metric | val **box AP50:95** (box-only, so the "no mask supervision in training" claim stays clean). Not mask AP. |
| 2 | training horizon | min 20 epochs, patience 3, cap 60 |
| 3 | box loss weights | GIoU ×2, size ×0.5 (one setting, not a sweep) |
| 4 | dense box supervision | regress at every cell with target heatmap ≥ 0.5, loss weighted by heatmap; overlap → the stronger heatmap (smaller crown). **New module.** `dapt/targets.py` is frozen by design: do not edit it. |
| 5 | small-crown heatmap min radius | leave unchanged |
| 6 | pilot ladder, seed 0 | A = 1+2; B = A + 3; C = B + 4 (~6 A100-h) |
| 7 | adopt/stop rule | render each pilot's val masks at 2048 (τ 0.40) and compare mask AP50. Adopt the best only if ≥ **+0.01 over 0.6045**; else stop. Write the rule into `NATIVE_RASTER.md` **before** running. |
| 8 | after adopting | re-select τ, α, κ on val at 2048; apply the same recipe to NEON |

**Evaluating a pilot locally.** Download its checkpoint to `/Users/tompitts/dphil/feat_cache/` and point the val harness at it. `oracle_boxes_val.py` / `softbox_val.py` already do detector → masker → score_coco on the 108 local val tiles (features in `feat_cache/feat_4p_val/`). Copy one and parametrise `DET_PATH`; don't edit the originals. Keep the gate: the current detector must still reproduce **0.6045 / 40,058 dets**.

**Estimated cost.** Pilot ~6–8 A100-h; 2 more seeds ~4 A100-h; full LACE regeneration afterwards ~3 A100-h plus ~15 Modal CPU-h. About $40–45 in total. The $4.5/seed training figure is from `phase4/README.md`.

## 3. NEON, if the recipe is adopted (Tom's instructions)

- Use **L24 only** (not L21–24) and **3 seeds** (not 5).
- No re-extraction is needed. The NEON 4-phase cache is `(4096, 64, 64)` from layers (21, 22, 23, 24) (`modal_neon_multiseed/phase4/`). L24 is channels `[3072:4096]`, sliced exactly as `phase4_features_tcd.py` does (`L24_LO, L24_HI`).
- Risk to flag to Tom: the NEON headline margin is thin (0.728 vs DeepForest 0.719, F1@0.4). Check L24-only on NEON's own validation split before committing. NEON scoring uses the isolated `.venv_df` (see the `project_environment` memory).

## 4. Hard rules (each was learned the expensive way)

- **Never overwrite.** New files and new volume paths only; `assert not os.path.exists` before every write. **Do not edit `ablation/results/paper.tex`**: Tom decides paper changes after all data is in.
- **Gate every regenerated row** against its published 512 file before reporting a new number (byte-identical where deterministic). Every script in `native_raster/` shows the pattern.
- **Modal budget was exhausted on 2026-09-25.** Do not launch Modal jobs until Tom confirms it is topped up.
- **Launch Modal only via `native_raster/detach.py`** (`deploy <file>`, then `spawn <app> <fn> --k v`). Never stop a shell attached to `modal run`: Modal then cancels the call. Every job writes per-tile checkpoints and a `run.log` on its volume, resumes, and uses a new output dir. Poll the log with a Monitor; never pipe through `tail`.
- **Copy every GPU-produced file into the repo in the same session.** Check `modal volume ls` before declaring anything lost.
- **Quiet local compute**, since Tom is often in a quiet space: `VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2`. No local process pools without checking `uptime` first. Orphaned workers survive their parent: kill them with `pkill -f "multiprocessing.spawn import spawn_main"`.
- **Scoring:** `score_coco.py --res 2048`. Use `--score_floor 0` for any `*_product.json` (reranked key) and 0.05 otherwise. Compare a 2048 number only with 2048 numbers.
- **Environment parity:** local and Modal DINOv3 features differ (cos 0.86), so never put both in one table. Only scalar knob values travel between environments.
- **zsh does not word-split** unquoted variables: pass multi-word arguments explicitly.
- **Training must stay box-only.** No mask supervision anywhere in detector training or checkpoint selection.

## 5. Other open items (not this task, but don't lose them)

- **Paused GPU rows (budget):**
  - SelvaBox → SAM 3: 145/439 tiles on `tcd04-baselines-vol:out/native_raster/selvabox_bench_r2048/`. Resume with `detach.py spawn tcd04-selvabox-native predict`, then `assemble`.
  - SAM 3 on LACE boxes: 214/439 on `tcd04-phase4-vol:out/native_raster/sam3_s0_r2048/`. Resume with `spawn tcd04-sam3-native predict`.
  - If the detector changes, the SAM 3 row must be re-run on the *new* LACE boxes.
- **Phase B:** joint α, κ, τ on val at 2048 (`val_knobs_local.py`, extended). This folds into decision 8.
- **Remaining ablations at 2048:** tab:rpn recall; tab:masker α=1 row (Modal); tab:components and tab:collapse (feasibility unchecked); t01, t02, t09, t13, t14.
- **Consolidated table:** `results_439/build_table_r2048.py` → `results_439/r2048/table_r2048.md`. Rerun it after any new scored file.

## 6. Key files

| what | where |
|---|---|
| campaign record (read first) | `native_raster/NATIVE_RASTER.md` |
| approved re-base plan | `~/.claude/plans/ok-if-we-needed-merry-treehouse.md` |
| 2048 preds and scored files | `native_raster/preds_*_r2048*.json`, `native_raster/scored_*_r2048*.json` |
| ablation outputs at 2048 | `ablation/results_r2048/` |
| val harnesses (detector → masker → score) | `native_raster/oracle_boxes_val.py`, `native_raster/softbox_val.py` |
| diagnostics | `native_raster/fp_anatomy_test.py`, `native_raster/small_crowns_test.py` |
| detector, loss, training | `boxinst_commonality_tcd_04/detector.py`, `train_detector_tiles.py`, `modal_tcd_multiseed/phase4/phase4_lib_tcd.py`, `phase4_modal.py` |
| local caches | `/Users/tompitts/dphil/feat_cache/` (val features, seed-0 detector, raw baseline subtiles) |
| memory | `project_native_raster_campaign`, `feedback_modal_detach_and_local_pools`, `project_environment` |
