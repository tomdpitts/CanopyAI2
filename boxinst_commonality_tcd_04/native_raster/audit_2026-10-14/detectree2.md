# Detectree2 s0v2 baseline audit (2026-10-14)

<!-- TABLE1_CURRENT:begin (auto, results_439/mark_superseded.py) -->
> ⚠️ **SUPERSEDED 439 / sparse236 NUMBERS (marked 2026-10-27).** Current headline table: [`TABLE1_CURRENT.md`](../../results_439/TABLE1_CURRENT.md). Test-set numbers in this file are a historical experiment record (512 raster, pre-seam-fix baselines, 8 px masker or earlier models) and must not be quoted unless they match that table.
<!-- TABLE1_CURRENT:end -->

Read-only audit. No Modal job launched. Only Modal access was `modal volume ls tcd-detectree2-vol out`.

Released source used throughout: `PatBall1/detectree2` tag **v2.1.2** (`d9fb07f`, 2026-02-14, the first PyPI release). `git diff v2.1.2 HEAD -- detectree2/` is empty, so the line numbers below hold for both. Detectron2 defaults: `facebookresearch/detectron2@1e3e13b` (`config/defaults.py`, `configs/Base-RCNN-FPN.yaml`). Both were cloned to the session scratchpad. They are not in the repo, and no local venv has detectree2 or detectron2 installed.

Paths: "ours" paths are relative to `boxinst_commonality_tcd_04/`. "dt2:" paths are inside the detectree2 repo.

## Table

| # | Item | Our setting (file:line) | Released default (source) | Effect on baseline | Severity |
|---|---|---|---|---|---|
| 1a | Init weights | `250312_flexi.pth`, md5 asserted (`detectree2_baseline/detectree2_modal.py:84-85,436-446`) | Recommended for fine-tuning (dt2:`docs/source/tutorials/03_training_and_evaluation.rst:59-67`; `model_garden/README.md:89-101`) | neutral | OK |
| 1b | Base config | R101-FPN 3x yaml (`dt2_recipe.py:27,36`) | Same (dt2:`models/train.py:990`) | neutral | OK |
| 1c | LR / WD / mom / warmup / gamma / freeze | 3.389e-4 / 1e-3 / 0.9 / 120 / 0.1 / 3 (`dt2_recipe.py:30-50`); constant LR because STEPS > MAX_ITER (`train_s0v2.log:641`) | Same values (dt2:`train.py:994-1003,1055-1062`); no STEPS set, so also constant | neutral | OK |
| 1d | RPN batch/img | 1024 (`dt2_recipe.py:45`) | 1024 (dt2:`train.py:1060`) | neutral | OK |
| 1e | ROI-heads batch/img | not set, so d2 default 512 | not set, so 512 (d2 `defaults.py:269`) | neutral | OK |
| 1f | **IMS_PER_BATCH** | **4** (`detectree2_modal.py:482,541`; log `train_s0v2.log` "batch_size=4") | **2** (dt2:`train.py:995`; tutorial `03_…rst:57,67` passes no override) | unknown (same LR, so half the steps per epoch) | **disclosure** |
| 1g | Train-time RPN topk pre/post | **3000/2000** (`detectree2_modal.py:545,547`) | 2000/1000 (`Base-RCNN-FPN.yaml:14,19`; detectree2 sets nothing) | likely helps / neutral | **disclosure** |
| 1h | Augmentation | rot 0-360, hflip, brightness/lighting/contrast/saturation, ResizeShortestEdge(1000,1333) (`dt2_recipe.py:61-68`; confirmed `train_s0v2.log:628`) | Identical for `resize="fixed"` (dt2:`train.py:661-677`) | neutral | OK |
| 1i | Eval period / patience / max_iter | 2000 / 2 / 16000; stopped by the rule at 14000, best = iter 10000 (`train_s0v2.log:2429,2436`) | setup_cfg 100 / max 1000 (dt2:`train.py:1003-1004`); tutorial 100 / patience 5 / max 3000 (`03_…rst:57,92`) | likely helps (4000-iter patience vs 500), but selection grid is coarser | disclosure |
| 1j | Early-stop rule | strict `ap > max_ap`, stop at counter >= patience, reload best (`dt2_recipe.py:105-130`) | Same (dt2:`train.py:302,310,316-329`) | neutral | OK |
| 1k | In-training val input scale | ResizeShortestEdge 1000 (`train_s0v2.log:766`) | **Native, no resize**: `MyTrainer.build_test_loader` uses `FlexibleDatasetMapper(cfg,is_train=False)` with no augs (dt2:`train.py:715-729`, `79-81`) | neutral (1024 vs 1000) | OK |
| 1l | Training data | 792 tiles split into 3x3 subtiles of 1024 at stride 512; empty subtiles dropped, giving 6692 subtiles and 127,692 crowns (`train_s0v2.log:630-634`). Crowns clipped to the subtile, kept if >=40 px (`build_coco.py:26,93-104`) | `tile_data` clips crowns to the buffered tile (dt2:`preprocessing/tiling.py:171`). Defaults it applies that we do not: `use_convex_mask=True` (pixels outside crown hull + 3 m blacked, `:197-221`), `enhance_rgb_contrast=True` (0.2/99.8 pct stretch, `:235-247`), tutorial `threshold=0.6` coverage filter (`02_…rst:59`) | unknown. Note that contrast stretch post-dates flexi (commit 6bffb3a, 2025-10-15), so raw RGB matches flexi's pretraining. The 0.6 threshold would remove sparse subtiles, so omitting it likely helps | OK (note) |
| 1m | **Canopy in training** | Emitted as `iscrowd=1` "IGNORE" (`build_coco.py:7-9,104`). detectron2 **drops** iscrowd annotations in training (d2 `data/dataset_mapper.py:124-130`), so canopy is trained as **background** | detectree2 has no ignore mechanism, so unlabelled canopy is background there too (inside the convex hull) | unknown. Code comment is false. LACE's training treatment of canopy not verified here (`cache_train_tiles.py:68-70` carries canopy polygons) | disclosure |
| 1n | Resize train | MIN_SIZE_TRAIN 1000 (1024 to 1000, x0.977) | Same (dt2:`train.py:1076`, aug `:677`) | neutral | OK |
| 2 | **INPUT.MIN_SIZE_TEST** | **1000** / MAX 1333 (`detectree2_modal.py:679-681`, run with `min_size_test=1000`, `predict_s0v2.log:59`). This is a 1024 to 1000 (x0.977) downscale, the same as training | **800** (d2 `defaults.py:63`; detectree2 never assigns it; DefaultPredictor resizes with it, d2 `engine/defaults.py:322-323`). Released tiles are ~100 m, i.e. ~1000 px at 0.1 m/px (`model_garden/README.md:94`; tutorial 40 m + 2x30 m buffer, `04_prediction.rst:24,38-40`), so **800 also DOWNscales detectree2's own tiles**. The released pipeline itself trains at 1000 and tests at 800 | probably helps, but **unverified**: `eval_val` at 800 was never run (no `eval_val_s0v2_*.json` and no `eval_s0v2_*` dir on `tcd-detectree2-vol:out/`; no local log) | **disclosure**: our docs are wrong (see notes) |
| 3a | RPN PRE/POST_NMS_TOPK_TEST | **2000/1500** (`detectree2_modal.py:677-678`) | 1000/1000 (`Base-RCNN-FPN.yaml:15,20`) | likely helps | **disclosure (undisclosed in paper)** |
| 3b | TEST.DETECTIONS_PER_IMAGE | **500** per subtile (`:676`) | 100 (d2 `defaults.py:614`) | **helps**. Raw max is 389 per subtile, so 500 never binds. At 100, 2018/3951 subtiles (51%) would be truncated and 185,589/464,868 raw dets (40%) lost | **disclosure (undisclosed in paper)** |
| 3c | SCORE_THRESH_TEST | 0.05 (`:675`) | 0.05 (d2 `defaults.py:280`) | neutral | OK |
| 3d | ROI NMS / RPN NMS | 0.5 / 0.7, untouched | 0.5 / 0.7 (d2 `defaults.py:283,245`) | neutral | OK |
| 4a | Stitch floor | 0.05. Meta `min_score 0.05` in `native_raster/preds_dt2_s0v2_r2048.json` and `preds_dt2_sparse_s0v2_r2048.json`; observed min score 0.0500001 in both raw and stitched | `clean_crowns(confidence=0.2)` (dt2:`outputs.py:325,362-363`); prediction tutorial passes `confidence=0` (`04_…rst:76`) | helps (vs 0.2) | OK. Paper text is wrong ("its 0.1"), see 8 |
| 4b | Dedup thresholds | IoU 0.7, containment 0.85 (`native_raster/stitch_res.py:80,100`) | IoU 0.7, containment 0.85 (dt2:`outputs.py:324,328`); tutorial uses IoU **0.6** (`04_…rst:76`, `01_…rst:176`) | neutral | OK |
| 4c | **Dedup algorithm** | Greedy by score. Containment is `inter / area(candidate)`, so a lower-scored crown is dropped only if **it** sits inside a kept one (`stitch_res.py:80-105`) | Union-find clustering over all pairs. Containment is `inter / min(area_i, area_j)`, i.e. symmetric. Keeps one max-score crown per **transitive** cluster (dt2:`outputs.py:392-433`) | unknown. Ours keeps a low-score big crown that encloses a high-score small one; released drops it | disclosure ("clean_crowns rule" claim is inexact) |
| 4d | Area floor | none | `area > 2` m², i.e. 200 px at 0.1 m/px (dt2:`outputs.py:326,360`) | unknown | OK (note) |
| 4e | **Edge-crown cull** | none | `stitch_crowns(folder, shift=1)` keeps only crowns lying **within** the buffered tile shrunk by 1 m (dt2:`outputs.py:282-305`) | hurts at interior seams (the KNOWN seam defect) and helps at the outer frame | DEFECT, known and owned by the seam fix |
| 4f | Mask to polygon | full mask kept | `polygon_from_mask` keeps only the **first** external contour (dt2:`outputs.py:39-55`) | helps / neutral | OK |
| 5 | Checkpoint selection | best of 7 segm-AP50 evals on **all 896 val subtiles** (108 val tiles); iter 10000 (`model_5.pth`) reloaded (`train_s0v2.log:2436`) and saved as `model_best_s0v2.pth`; 10000x4/6692 = 5.98 epochs | Same rule (dt2:`train.py:289-329`) | neutral. No test leakage: train/val/test image_id overlap is 0/0 (manifest check) | OK |
| 6 | Test subtile geometry | 3x3 grid of 1024 at stride 512 on 2048, full 9/9 coverage (3951 = 439x9; 2124 = 236x9) (`build_coco.py:30-38`, `detectree2_modal.py:235`) | Tutorial 40 m tile + 30 m buffer gives 100 m (~1000 px) tiles at stride 40 m (60% overlap). On a 204.8 m tile, the default grid (`full_coverage=False`, `tiling.py:586-597`) gives minx in {30,70,110} m: 9 tiles covering only [0,180] m, leaving a 24.8 m strip right/bottom **uncovered**. `full_coverage=True` gives 36 nodata-padded tiles. Function defaults (200 m / 30 m) give **0 tiles** | neutral. Same pixel scale; any crown <=51.2 m lies whole in some subtile | OK |
| 7 | Sparse-236 | Same `model_best_s0v2.pth` and the same test config. **Proven** by byte-identical scores+boxes on all 144 subtiles of the 16 tiles shared with the 439 (whose run logs `min_size_test=1000`); stitched with the same `stitch_res`, floor 0.05. Train∩sparse = 0 tiles | n/a | neutral | OK. No v2 sparse predict log is saved locally (only the superseded Aug-18 logs); this is a provenance gap |
| 8 | Paper claims | see notes | | | **disclosure** (3 text errors, 1 incomplete deviation list) |

## Evidence notes

- **CONFIG_AUDIT §3 is NOT closed.** `CONFIG_AUDIT.md:158` is still unchecked ("Re-run DetecTree2 predict at released proposal budget"). `tcd-detectree2-vol:out/` has no released-budget predict dir: the listing is train_s0, pred_s0, pred_raw_{,sparse_}s0, train_s0v2, pred_s0v2, pred_raw_{,sparse_}s0v2, and the preds/model files. `paper.tex:1187` (`tab:geometry`, captioned "Deviations … from each method's released configuration") and note b at `:652` list only MIN_SIZE_TEST, the floor and clean_crowns. They omit the test RPN topk (1000/1000 raised to 2000/1500), DETECTIONS_PER_IMAGE (100 raised to 500), the train RPN topk (2000/1000 raised to 3000/2000) and IMS_PER_BATCH (2 raised to 4). All four likely favour DT2, so this is not a hindrance, but the table is incomplete.
- **`paper.tex:1187` "stitched at the protocol floor 0.05 rather than its 0.1" is false.** 0.1 was our own superseded `stitch.py` default (`detectree2_baseline/stitch.py:58`). Released `clean_crowns` floors at 0.2 (`outputs.py:325`), and the prediction tutorial uses 0 (`04_prediction.rst:76`). `PROTOCOL_439.md` and `results_439/table_439.md:48` repeat "not 0.1".
- **`paper.tex:652` "Trained under Detectree2's own recipe"** is inexact because of 1f and 1g. `RETRAIN_V2.md:107-109` says "`ims_per_batch=4` … are detectree2's own recipe". That is false: `train.py:995` defaults to 2.
- **MIN_SIZE_TEST rationale is wrong in our docs.** `RETRAIN_V2.md:43-44`, `PROTOCOL_439.md:143` and `detectree2_modal.py:496-497` say "detectree2's own tiles are smaller than 800 and get UPscaled". In fact released tiles are ~100 m, about 1000 px at 0.1 m/px and about 1430 px at the tutorial's 7 cm ortho, so 800 downscales them too. Our 1000 is therefore a departure in DT2's favour, not a restoration of as-shipped behaviour. The paper's wording at `:652,1187` avoids this false claim and is acceptable. `RETRAIN_V2.md:89-90` says "800 measured as a sensitivity on the selected checkpoint (`eval_val`)", but no artefact exists. **Close it:** run `eval_val --tag v2 --min-size-test 800` and `1000` (val only, ~$0.5). If 800 wins on val, then 1000 hinders DT2 and becomes a DEFECT.
- **`paper.tex:1215` "Detectree2 emits up to 1,511 detections per tile"** is stale (the 512 stitch). The r2048 file gives max **1521**, mean 449.7. AP 0.6143 rising to 0.6187 at maxDets 1200 matches `scored_dt2_s0v2_r2048{,_maxdets1200}.json`. The seam fix will change all of these.
- Table values check out: 439 = 0.6143 / 0.3014 / box 0.6387 and sparse = 0.6252 / 0.3460 / 0.6331, both canopy_neutral_crowd in the scored r2048 files, matching `paper.tex:637,718`.
- **Canopy (1m):** `build_coco.py` and the `stitch.py` docstring say canopy is an ignore region. In training it is not, because detectron2 discards iscrowd before building `Instances`. The training histogram counts only the 127,692 crowns. In val/selection (pycocotools) and in our scorer it is a true ignore. If LACE's detector ignores canopy in training, this is an asymmetry against DT2. That is unverified, and checking it is the next step.
- Grep of v2.1.2 for `MIN_SIZE_TEST|MAX_SIZE_TEST|SCORE_THRESH_TEST|DETECTIONS_PER_IMAGE|NMS_THRESH_TEST|TOPK_TEST` finds nothing in `detectree2/` except `tests/test_models.py:18`. So every test-time knob in released DT2 is the detectron2 default.
- Raw stats were computed from the local cache `/Users/tompitts/dphil/feat_cache/dt2_raw/`. 439: 3951 subtiles, 464,868 dets, mean 117.7 / median 104 / max 389 per subtile. Sparse: 2124 subtiles, 239,714 dets, max 430.

## Released stitch rule (detectree2 v2.1.2)

Pipeline per tutorial `04_prediction.rst:38-76`:

1. **Tiling:** `tile_data(img, out, buffer=30, tile_width=40, tile_height=40)`. Grid of minx at stride `tile_width`. Each tile spans `[minx-buffer, minx+width+buffer]`, i.e. 100 m with 60 m overlap between neighbours. The name encodes `minx_miny_width_buffer_crs` (`tiling.py:159-164,586-597`).
2. **Inference:** `predict_on_data(tiles, predictor=DefaultPredictor(setup_cfg(update_model=…)))` writes per-tile COCO-RLE JSON with no post-filter (`predict.py:62-94`). Settings are the d2 defaults: MIN_SIZE_TEST 800, topk 1000/1000, 100 dets, score 0.05, NMS 0.5.
3. **Vectorising:** `project_to_geojson` converts each RLE to the **first** external contour only (`outputs.py:39-55,202-203`), georeferenced at pixel centres (`:208-210`).
4. **`stitch_crowns(folder, shift=1)`** (`outputs.py:282-313`). For each tile it builds `box(minx-buffer+shift, miny-buffer+shift, minx+width+buffer-shift, miny+width+buffer-shift)` (`box_filter`/`box_make`, `:243-279`) and keeps crowns with `sjoin(..., "within")` (`:304`). **It is NOT a core-only rule.** It keeps every crown lying wholly inside the full buffered tile shrunk by `shift` = 1 m (10 px at 0.1 m/px). It drops any crown reaching within 1 m of its own tile's edge. In pixels, the crown's pixels must lie in columns and rows 10…W-11. The cull also applies at the orthomosaic's outer frame: edge tiles start at the raster bound because `start = bound + buffer` and the extent is `minx - buffer`. Crowns from all tiles are then concatenated, so duplicates in the 60 m overlaps remain.
5. **`clean_crowns(crowns, iou_threshold, confidence=…)`**, global across tiles (`outputs.py:322-438`). Steps:
   - Drop invalid/empty geometry, `area <= 2` m² and, if `confidence` is truthy, `score <= confidence` (default 0.2; tutorial 0).
   - Find candidate pairs with `sjoin(intersects)`.
   - Union a pair if `IoU > iou_threshold` (default 0.7; tutorial 0.6) OR `inter / min(area_i, area_j) > 0.85`.
   - Keep the single highest-`Confidence_score` crown per **connected component**, so merging is transitive and the containment test is direction-free.
   - `post_clean` (gap-filling) exists (`:441-506`) but is not in the tutorial.

Differences from ours (`native_raster/stitch_res.py`):
- (a) We have no edge cull. This is the seam defect at interior edges; at the outer frame we are more lenient than released, which is in DT2's favour.
- (b) Our dedup is greedy and one-directional (candidate-in-kept), not transitive clustering with min-area containment.
- (c) We have no 2 m² area floor.
- (d) Our floor is 0.05, not 0.2.
- (e) We keep the full mask rather than the first contour.

For the seam fix: matching the released tolerance means a 10 px margin at interior subtile edges, applied before dedup.
