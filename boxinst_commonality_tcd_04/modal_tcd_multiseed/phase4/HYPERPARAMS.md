# Hyperparameters — LACE and all baselines

Every value below is transcribed from source, with the defining file and symbol named. Where a
default is overridden at a call site, both are given. **If you change a number here, change it in the
code — this file is a record, not a config.**

Scope: §1–9 the settled 4-phase L24 pipeline on **OAM-TCD** (`modal_tcd_multiseed/phase4/`) plus its
baselines; §10 **NEON** (`modal_neon_multiseed/`); §11 **SavannaTree** (`modal_savanna_multiseed/`).
The three datasets share the backbone, the detector architecture and the EM masker recipe, but differ
in canvas, tiling, supervision and metric — the per-dataset sections give only what differs.

---

## 1. Data and cohort

| item | value | source |
|---|---|---|
| Source dataset | HF `restor/tcd`, joined by `image_id` (never filename) | `phase4_modal.py::_load_hf` |
| Tile size | 2048 × 2048 px RGB, EPSG:3395, nominal 0.1 m/px | tile `_meta.json` |
| True ground GSD | 4.1–10 cm (Mercator scale 1/cos φ; median 9.1 cm) | measured, `bounds` |
| Train cohort | **900** tiles, sampled seed 0 from 4169, ITC-bearing + width 2048 | `cache_train_tiles.py::build_gt` |
| Train/val split | **792 / 108** (`int(0.12 × 900)`), RNG seed 1, by tile | `cache_train_tiles.py:70-77` |
| Test cohort | **439** = the entire official holdout (`validation_fold = −1`) | `prepare_test.py::main` |
| Cohort hashes | train `2064a2c73a5c2087`, val `f03602636d941a15`, test `d10d2200663b0587` | sha1 of sorted image_ids |
| Frozen records | `train_tiles_gt.json` (900 + partition), `test_gt.json` (439), `phase4/manifest.json` | write-once |

GT per tile: `category_id == 2` → individual crowns (positives); `category_id == 1` → canopy
(**ignore**, never background). Segmentations are polygon **or** RLE — 647 ITC + 3375 canopy
annotations are RLE across 2142 of the 4608 tiles, and **RLE annotations carry `area == 0`**, so the
COCO `area` field must never be summed (`tile_index.py::seg_to_rle`).

## 2. Backbone and features (frozen)

| item | value | source |
|---|---|---|
| Backbone | DINOv3 ViT-L/16 `"web"`, **frozen** | `dapt/backbone.py::FrozenDinoV3Features` |
| Layers | `(21, 22, 23, 24)` → 4096-d stack | `phase4_modal.py:202` |
| Detector slice | **L24 only**, dims `[3072:4096]` → 1024-d | `phase4_features_tcd.py:39` |
| Patch / native grid | 16 px → 128 × 128 per 2048 tile | `PATCH`, `GRID16` |
| 4-phase shifts | `((0,0), (0,8), (8,0), (8,8))` px | `PHASES` |
| Interleaved grid | **256 × 256 @ 8 px stride**; cell X ↔ pixel **8X + 8** | `interleave()` |
| Extraction window | 2 × 2 of 1024 px windows, stitched | `_extract_2048` |
| Precision | fp16 on disk, fp32 in compute | |
| Parity gate | cos > 0.5 vs `ref_feat_tcd.npz`; recorded **0.86003** (tf 4.57 vs local 5.12) | `extract_4p` |

## 3. Detector — `Detector4Phase` (= `Detector8` minus the internal upsample)

| item | value | source |
|---|---|---|
| Input | (1024, 256, 256) real-8px L24 | |
| Width / tower | `width=256`, `tower=3` | `train_4p` args |
| Target grid / stride | grid 256, `STRIDE8 = 8` | `detector.py:24` |
| Optimiser | Adam, **lr 1e-3**, **weight decay 1e-4** | `train_detector_tiles.py:104` |
| LR schedule | CosineAnnealingLR over `epochs` (default; plateau opt-in) | `:115` |
| Batch size | **3** tiles | `train_4p` |
| Epochs | 40 max | `train_4p` |
| Eval cadence | every **5** epochs on the 108 val tiles | `eval_every=5` |
| Early stop | min 12 epochs, patience 2 evals, `min_delta = 5e-3` on val box mAP50 | `train_4p` |
| Model selection | best val **box mAP50** | `:155` |
| Loss | focal heatmap + smooth-L1 offset + smooth-L1 size (**w_size 0.1**) + GIoU | `detector.py::det_loss` |
| **Canopy in training** | canopy cells (>50% covered) masked out of the **heatmap loss** | `detector.py:66`, `canopy_cell_mask` |
| Seeds | 0, 1, 2 (band); 3, 4 additionally for the vanilla 5-seed band | |

**Inference / decode**

| item | value | source |
|---|---|---|
| Score threshold | **0.05** (keeps the AP tail) | `train_detector_tiles.py:85` |
| top-k | **600** per tile | `:85` |
| NMS IoU | **0.5** | `:164` |
| Operating point `op_thr` | best-F1 on val, swept `linspace(0.05, 0.9, 18)` @ IoU 0.5 → **0.40** for seed 0 | `dapt/eval.py::pick_threshold` |

## 4. Box→mask masker — latent-commonality EM

**Fit** (`phase4_fit_tcd.py::fit_masker_4p` → `em.fit`), CPU-only, seed-independent, ~6 min:

| item | value |
|---|---|
| Fit tiles | **120** (from the 792 train partition) |
| Seed | 0 |
| PCA dims | **128**, centred + whitened on tree + clear-bg cells (canopy excluded) |
| Foreground mixture `k` | **16** |
| Background mixture `k_bg` | **12** |
| Distance bins | **8** |
| vMF concentration `kappa` | **10.0** |
| EM iterations | **30** |
| Prototype pruning | after 5 iters, `prune_frac = 0.25` |
| Contrastive `beta` | **0.5** (`no_contrast` defaults to `beta == 0`) |
| **`cell_origin_frac`** | **1.0** → origin = `s` = 8 px (the interleaved grid's true centres) |

`cell_origin_frac=1.0` is load-bearing: the default 0.5 (`s/2`) mis-registers the interleaved grid by
−4 px and inverts the β ranking. See README § *Registration bug and fix*.

**Inference knobs** (free scalars in the E-step, no refit; stored inside the npz and read by default):

| knob | value | effect |
|---|---|---|
| `prior_weight` α | **0.3** | scales the box-normalised spatial-prior logit `sigmoid(A + α·logit(psum))`; α<1 relaxes trust in imprecise predicted boxes |
| `kappa_scale` κ | **×1.6** | scales the appearance vMF concentration (both `zc` and `bg_ll`) → tighter boundaries |
| `mask_thr` | **0.25** | P(fg) cut in `pred_instance_masks` (vs 0.5: grows masks, recovers small crowns) |
| raster shift | **+1 px @ 512** | verified optimal by a render-offset sweep; 0 leaves a top-left bias |

CLI sentinel `−1.0` for α/κ means *use the masker's own stored values*. Pass `--prior-weight 1.0
--kappa-scale 1.0` for the vanilla ablation.

## 5. Evaluation protocol

| item | value | source |
|---|---|---|
| Scoring raster | **512 × 512** (`RES`), `SCALE = 2048/512 = 4.0` | `evaluate.py:34` |
| AP | COCO **101-point** interpolated, greedy one-to-one matching | `evaluate._greedy_ap` |
| IoU sweep | `arange(0.50, 0.96, 0.05)` for AP50-95 | `IOU_50_95` |
| Aggregation | **pooled** over all tiles (global `n_gt`), not per-tile mean | |
| **Canopy-ignore** | a prediction >50% inside canopy is **dropped**, neither TP nor FP | `eval_4p_selfmask:140` |
| Box-ignore | box crop mean over the canopy raster > 0.5 | same |
| Empty tiles | **kept** — 73 of the 439 have zero GT crowns | |
| Seed band | mean ± **sample** std (**ddof = 1**) | `no_ignore_sensitivity.py` |

`no-ignore` runs are identical except the ignore vector is forced all-False.

## 6. DetecTree2 baseline (fully supervised, our fine-tune)

`dt2_recipe.py::setup_cfg` — a line-for-line port of DetecTree2's published recipe.

| item | value |
|---|---|
| Architecture | Mask R-CNN **R101-FPN** (`COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml`) |
| Init weights | DetecTree2 released **`250312_flexi.pth`** (Zenodo 15014353) |
| Base LR | **3.389e-4** · momentum 0.9 · weight decay **1e-3** · gamma 0.1 |
| Warmup | 120 iters |
| Backbone freeze | `FREEZE_AT = 3` |
| RPN batch/image | 1024 · `NUM_CLASSES = 1` |
| `MIN_SIZE_TRAIN` | 1000 · `RESIZE = "fixed"` |
| Augmentations | RandomRotation [0,360] · HFlip p=0.5 · Brightness 0.7–1.5 · Lighting 0.7 · Contrast 0.6–1.3 · Saturation 0.8–1.4 · ResizeShortestEdge [1000,1000], max 1333 |
| `FILTER_EMPTY_ANNOTATIONS` | **False** |
| max_iter / eval_period / patience | **4000 / 500 / 6** |
| `IMS_PER_BATCH` | 4 |
| Model selection | best val **segm AP50**; reached **57.11** at ~4000 iters |
| Val subset | 256 of 896 val subtiles (model selection only) |
| Mask format | `bitmask` (lossless RLE) |
| Inference | `SCORE_THRESH_TEST = 0.05`, `DETECTIONS_PER_IMAGE = 500`, RPN pre/post NMS test 2000/1500 |
| Subtiling | 2048 → **1024 @ stride 512** (3×3), `MIN_AREA = 40 px²` |
| **Test coverage** | `keep_empty=True` → all **3951** subtiles (was 3189 = 80.7%; fixed 2026-08-18) |
| Stitch dedup | `clean_crowns`: IoU > **0.7** or containment > **0.85**, keep highest score |
| Stitch score floor | **0.1** |
| Canopy | emitted as `iscrowd=1` ignore regions in training COCO |
| Seeds | 1 (seed 0) |

## 7. SAM 3 box→mask ablation

| item | value |
|---|---|
| Model | `facebook/sam3` (gated), `build_sam3_image_model(enable_inst_interactivity=True)` |
| Pinned commit | `46957e47805eaa273f4aa7bbbd25a88bca9108ce` |
| Prompt | boxes only, no points, `multimask_output=False` (one mask per box) |
| Boxes | our exact seed-0 `boxes_2048`; detector never re-run |
| Batch | 64 boxes per forward; image encoded once per tile |
| Mask raster | predicted at 2048, 4×4 area-majority downsample to 512 (≥50%) |
| AP ranking | **our detector scores** (SAM-score ranking reported as secondary) |
| Cropping | uncropped (headline); box-clipped as a sensitivity check |

## 8. Sparse-canopy slice

| item | value |
|---|---|
| Biomes (WWF/Olson) | **7, 8, 9, 10, 12, 13** |
| Tiles | **236** (220 unseen train-pool + 16 from the official 439) |
| GT crowns | 14,937 |
| Exclusion | tile id **and** image_id **and** rgb-sha1 disjoint from the 900 |
| Scene clustering | 300 m single-linkage on tile centroids (report axis, not a filter) |
| No-data | **not** filtered — median valid 93%, 107 tiles <90% |
| Eval | identical to §5; detector + masker unchanged (zero-shot) |

## 9. Compute

A100 throughout (`gpu="A100"`, unpinned 40/80 GB). Feature extraction ≈ 4.8 s/tile; masker fit CPU
~6 min; eval CPU-bound ~3.5 s/tile. Modal image pins **transformers 4.57** — do not bump; the local
cache is 5.12 and features differ (cos 0.86), so local and Modal features must never be bit-compared.

---

# 10. NEON (`modal_neon_multiseed/phase4/`)

Detection only — **no masker, no mask metric.** The benchmark is DeepForest's: box F1 at IoU 0.4.

## 10.1 Data

| item | value | source |
|---|---|---|
| Source | NeonTreeEvaluation, hand-annotated RGB tiles + XML boxes | `prepare_neon.py` |
| Eval set | **194** tiles, 400 × 400 px @ **0.1 m/px** (= 40 × 40 m) | `prepare_neon.py:59` |
| Train patches | **2063** patches, `PATCH = 400`, `STRIDE = 400` (non-overlapping) | `prepare_neon_train.py:40-41` |
| Val split | `VAL_FRAC = 0.12` → every 8th patch of **each** tile; tile-level for the eval set | `:46-47` |
| Empty patches | kept as **real negatives** when inside the annotated bbox | `prepare_neon_train.py:9` |

## 10.2 Features — same backbone, smaller canvas

| item | value | source |
|---|---|---|
| Canvas | **512** (400 px tile padded; ≥112 px bottom-right pad, dropped at eval) | `phase4_features.py:27,36` |
| Patch / grids | `PATCH = 16` → `GRID16 = 32`, interleaved **`GRID8 = 64`** | `:28-30` |
| Phases | `((0,0), (0,8), (8,0), (8,8))` — identical to TCD | `:26` |
| Backbone | DINOv3 `"web"`, layers `(21,22,23,24)`, trap-guarded in `build_net` | `phase4_modal.py:81` |

## 10.3 Detector

Same `Detector4Phase`, same optimiser and schedule as §3:

| item | value | source |
|---|---|---|
| lr / wd | 1e-3 / 1e-4, Adam, CosineAnnealingLR | `phase4_lib.py:56-72` |
| Batch / epochs | `bs = 3`, `epochs = 40` | `:56` |
| Width / tower | 256 / 3 | `:57` |
| Eval cadence / early stop | every 5 epochs, `min_epochs = 12`, patience 2 | `:56-57` |
| Loss | `l_hm + l_off + l_size + l_giou`, **unweighted sum** (vs TCD's `w_size = 0.1`) | `:98` |
| NMS IoU | 0.5 | `:46` |
| Inference | `score_thr = 0.01`, `topk = 600` | `:185` |
| Seeds | **5** (0–4) | `multiseed_4p` |

## 10.4 Metric

| item | value | source |
|---|---|---|
| Scorer | `deepforest.evaluate_boxes` via `df_scorer` | `phase4_score.py:40` |
| IoU | **0.4** (the DeepForest convention, not 0.5) | `:40` |
| Operating point | best-F1 from a threshold-swept PR curve | `:43` |

**Result context:** ours 5-seed F1 **0.728 ± 0.003** (per-seed 0.722/0.728/0.729/0.730/0.730);
DeepForest published (Weinstein 2021, v1.8.0) **0.719**; our DeepForest 2.1.0 repro **0.726** on the
same 194 tiles — i.e. clearly above the *published* number, a tie with the *maintained package*.

---

# 11. SavannaTree (`modal_savanna_multiseed/v2/`)

⚠ **The headline mask number here is a negative result.** See `v2/README.md`; do not quote a
single-seed SavannaTree figure, and always state whether a number is box or mask.

## 11.1 Data

| item | value | source |
|---|---|---|
| Source | Jansen et al. 2023, Zenodo **7094916**, RPAS ortho, N. Australian savanna | `v2/data.py:49` |
| Native tile | **1024 px**, upscaled to a **2048** canvas (`UPSCALE = 2`) | `:57-58` |
| Test | **449** tiles / **1911** trees — site **S4**, exactly as released | `:63`, `v2/README.md:24` |
| Train | **2887** tiles / 12885 boxes — every non-S4 tile, S10 folded in, **no val site** | `v2/README.md:23` |
| Min train box | `MIN_TRAIN_BOX_PX = 2.0` @2048 (= 1 native px) | `v2/data.py:246` |
| Feature canvas | 512 (default) or 1024 — `c512` / `c1024` caches | `--canvas`, default 512 |

Structural caveats on record: only ~50% of woody vegetation is labelled (a *species* reference
dataset, no ignore class), 73% of test GT are tile-seam-truncated fragments, ~11% of polygons wrap
clumps rather than individuals.

## 11.2 Detector — `Detector8IoU` (adds an IoU-quality head)

| item | value | source |
|---|---|---|
| lr / wd | **2e-3** / 1e-4 (vs TCD 1e-3) | `v2/model.py:544-545` |
| Batch | **8** (vs TCD 3) | `:543` |
| Epochs | **60**, or `--visits` for tile-visit matching | `:540-542` |
| `w_size` | **1.0** (vs TCD 0.1) | `:548` |
| Optimiser / schedule | Adam + CosineAnnealingLR (`T_max = epochs`) | `:339-340` |
| Seed | 0 default | `:539` |
| Snapshots | `--snap-frac` / `--snap-every` | `:560-561` |

## 11.3 Masker

Same EM recipe as §4 — `pca 128, k 16, k_bg 12, bins 8, kappa 10.0, iters 30, beta 0.5` — with one
critical difference:

| item | value | why |
|---|---|---|
| `origin_frac` | **0.5** (not 1.0) | single-pass native 16 px grid, *not* the 4-phase interleaved lattice | 
| `mask_thr` | 0.25 | as TCD |
| α / κ | read from the npz (self-calibrated) | as TCD |

`v2/mask.py::fit_masker` uses one `origin_frac` for the label/ring loaders **and** `em.fit`'s
instance cells — that consistency is the function's whole purpose.

## 11.4 Metric — **theirs, not ours**

Read out of `azureml-automl-dnn-vision` v1.62.0, which produced the paper's baseline
(`v2/score.py` is the only file permitted to open `gt_test.json`):

| setting | value |
|---|---|
| `validation_metric_type` | VOC |
| `validation_iou_threshold` | **0.5** |
| 11-point AP | **No** (`_use_voc_11_point_metric = False`) |
| Aggregation | **POOLED** — concatenate over images, then re-sort |
| AP formula | rectangular AUC under the precision **envelope**, precision 1 prepended at recall 0 |
| Reported P / R | `precisions[-1]` / `recalls[-1]` — the **last** PR point, every retained detection, **not** a tuned operating point |

**Paper anchor:** AP50 **0.345**, endpoint P 0.255 / R 0.614 on 449 S4 tiles / 1911 trees.
`endpoint_P`/`endpoint_R` are the like-for-like analogue; `bestF1` is threshold-optimised and their
0.360 is not, so comparing the two **flatters us** and must be labelled whenever quoted. Mask is the
headline; box is diagnostic only.
