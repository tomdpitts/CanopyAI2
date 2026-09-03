# DetecTree2 row, re-investigated from the ground up (2026-09-02)

The published Table 1 DetecTree2 row (CN mask AP50 **0.5344**) came from a run with four
defects, **every one of them understating DetecTree2** — i.e. flattering us. This document
records the audit, the fix, and the measured cost of each defect. The scorer
(`score_coco.py`, `--res 512`, maxDets 600, floor 0.05, canopy `iscrowd`) is untouched:
every row is still measured identically.

Published artefacts are never overwritten. The retrain lives under `tag="v2"`:
`out/train_s0v2/`, `out/model_best_s0v2.pth`, `out/pred_raw_s0v2`,
`out/preds_dt2_s0v2*.json`.

## 0. Provenance of the initialisation checkpoint — CLEAN, no leakage

`250312_flexi.pth`, Zenodo record [15014353](https://zenodo.org/records/15014353)
("detectree2 trained models", Ball, 2025-03-12).

* **md5 asserted against the bytes we trained from**: `5451438786a4339a8b956830534cad40`,
  matching Zenodo exactly (`detectree2_modal.py::check_weights`, run 2026-09-02).
* **Training sites** (`model_garden/README.md`, verbatim): *"Training sites: Harapan, Danum,
  Paracou, Cambridge & Sepilok"*. Described there as "An RGB model that is trained on both
  closed canopy systems and urban environments".
* **OAM-TCD / OpenAerialMap / Restor appear nowhere** in the record or the model garden.

So the fine-tuned DetecTree2 arm cannot have seen any of the 439 test tiles through its
initialisation. This was the only audited item that could have meant leakage rather than
bias. It is bias: the arm starts from a checkpoint already pretrained on tree crowns at five
sites, which is an advantage over a generic frozen encoder, not a contamination. State it,
do not correct it.

## 1. The four defects

| # | Defect | Direction | Fix | Needs retrain? |
|---|---|---|---|---|
| 1 | Stitched at `min_score 0.1` while the protocol floor is 0.05; 56.5% of raw per-subtile detections fall below 0.1 | understates DT2 (may be immaterial — maxDets 600 truncates the tail anyway) | restitch the SAME raw predictions at 0.05 | no |
| 2 | `max_iter 4000` (2.39 epochs) while still improving — `ap_history` monotone 48.4 -> 57.1 over all 8 evals, patience counter 0, so early stopping **never fired**. It was a budget cap, not a stopping criterion | understates DT2, unboundedly | retrain to detectree2's own AP50 early-stopping rule | **yes** |
| 3 | In-training model selection used a strided 256-subtile subset of the 896 val subtiles | understates DT2 (noisier selection signal) | select on all 896 | **yes** (selection is internal to training) |
| 4 | *Found in this audit.* Neither detectree2's `setup_cfg` nor ours sets `INPUT.MIN_SIZE_TEST`, so detectron2's default **800** applies at eval and at predict, while training sees `MIN_SIZE_TRAIN = 1000`. Our 1024 subtiles are therefore silently downscaled at test but not at train | understates DT2 | run evals at 1000; decide 800 vs 1000 on **val**, never on test | folded into the retrain |

Defect 4 is an artefact of *our* 1024 subtiling interacting with an unset default, not of
detectree2's code: their own tiles are smaller than 800, so the same default *up*scales them.
Verified by grep over `PatBall1/detectree2@master` — `MIN_SIZE_TEST` is never assigned
anywhere in the package.

Comparators for calibration: LACE early-stops (best epoch 15-20 of max 40, patience 2); both
Box2Mask arms peaked and turned over. DetecTree2 was the only row in the table that never got
to state its own optimum.

## 2. The v2 run

Same recipe (`dt2_recipe.setup_cfg`, R101-FPN, base_lr 3.389e-4 constant, `FREEZE_AT 3`,
detectree2's augmentations), same 6,692 train subtiles from the same 792 tiles, same
initialisation, same AP50 early-stopping rule. Only the budget, the selection set and the
test-time input scale change.

| knob | published s0 | v2 |
|---|---|---|
| `max_iter` | 4000 (2.39 epochs) | 16000 (9.6 epochs) |
| `eval_period` | 500 | 2000 |
| `patience` | 6 evals (3000 iters) | 2 evals (4000 iters) |
| val set for selection | 256/896 subtiles (strided) | **896/896** |
| `INPUT.MIN_SIZE_TEST` | 800 (detectron2 default) | **1000** (= `MIN_SIZE_TRAIN`) |
| stitch `min_score` | 0.1 | **0.05** (= the protocol floor) |
| `ims_per_batch` | 4 | 4 |
| `TEST.DETECTIONS_PER_IMAGE` | 500 | 500 |
| RPN topk test (pre/post) | 2000 / 1500 | 2000 / 1500 |

Launched 2026-09-02, A100, `--detach`; log `train_s0v2.log`.

**Resume-safety.** `SOLVER.CHECKPOINT_PERIOD = eval_period` plus a volume commit at every
checkpoint makes the run restartable from `out/train_s0v2/last_checkpoint`; `APEarlyStopHook`
restores `APs`, `max_ap` and the patience counter from `ap_history.json`, so model selection
continues rather than restarting. Re-run with identical arguments to resume. detectron2 restores
model, optimiser, LR scheduler and iteration count but NOT the data-sampler position or RNG
state, so a resumed run is not bit-identical to an uninterrupted one.

**Measured cost** (Modal rates: A100-40GB $0.000583/s, CPU $0.0000131/core/s, memory
$0.00000222/GiB/s). Train loop 0.468 s/iter; a full 896-subtile val eval is ~700 s, dominated not
by the forward pass (detectron2 reports 0.061 s/iter inference) but by RLE-encoding up to 500
instances per 1024^2 image. 16k iters + 8 evals ~= $7.6.

## 2b. What is fixed, and what is still an asterisk

Fixed, and therefore no longer needs stating as a caveat:

* stitch floor 0.05, the same floor every other row is scored at;
* trained under detectree2's own AP50 early-stopping rule rather than a budget cap;
* model selection on all 896 val subtiles;
* `INPUT.MIN_SIZE_TEST = MIN_SIZE_TRAIN = 1000`, with 800 measured as a sensitivity on the
  selected checkpoint (`eval_val`), so the choice is evidenced rather than assumed;
* initialisation provenance md5-asserted against Zenodo (S0);
* full 9/9 subtile test coverage, so DetecTree2 is charged for false positives everywhere LACE
  is (fixed earlier, in `build_test_fullcov`).

Genuinely remaining, and to be stated plainly rather than buried:

1. **Early stopping must actually fire.** If `ap_history` is still monotone with the patience
   counter at 0 on reaching iter 16000, the cap has been moved rather than removed, and the
   honest sentence is "still improving at 9.6 epochs". Resolve by resuming with a larger
   `max_iter` (~$1.3 per 4k iters) rather than by rewording.
2. **One seed.** LACE reports three. PROTOCOL_439 §4.2 currently justifies single-seed baselines
   as "released checkpoints with no seed variation available" — that defence covers Restor and
   SelvaBox but NOT DetecTree2, which we train ourselves and could therefore seed. It was not
   seeded for budget (~$18 for three). Update §4.2 to say so.
3. **Recipe reproduced via detectron2 rather than by importing `detectree2`** — pre-existing and
   already disclosed in `dt2_recipe.py`; forced by the package's lack of a Python-3.8 wheel.
   `ims_per_batch=4` and the absence of an LR decay are detectree2's own recipe (their
   `setup_cfg` never assigns `SOLVER.STEPS`), not choices of ours.
4. **maxDets 600.** Previously measured at -0.0001 mask AP50 for DetecTree2, but the 0.05 floor
   raises its detection volume, so re-measure. Local rescoring, free.

## 3. Results

**Training stopped by its own criterion, not by the budget.** Seven full-val evaluations; the
patience counter reached 2 at iter 14000 and early stopping fired, 2000 iterations short of the
`max_iter` cap. The selected checkpoint is iter 10000 = **6.0 epochs**, against the superseded
run's cap at 2.39.

| eval | iter | val segm AP50 (896 subtiles, scale 1000) |
|---|---|---|
| 1 | 2000 | 58.594 |
| 2 | 4000 | 60.886 |
| 3 | 6000 | 62.175 |
| 4 | 8000 | 62.405 |
| 5 | 10000 | **63.827 <- selected** |
| 6 | 12000 | 62.727 (counter 1) |
| 7 | 14000 | 63.099 (counter 2 -> stop) |

`train_info_s0v2.json` records `stopped_iter: 9999`. That is NOT where training stopped -- it is
the iteration stored inside the checkpoint that `after_train` reloaded, i.e. corroboration that
the iter-10000 best was the model saved. Training ran to 14000.

These val figures are a SELECTION SIGNAL ONLY and are not comparable to Table 1: different set
(108 val tiles vs 439 test), different unit (1024 subtiles vs whole stitched 2048 tiles),
different raster, and detectron2's `COCOEvaluator` default maxDets 100 vs our 600.

### The published row (frozen scorer, `results_439/dt2_s0v2_coco512.json`)

| metric | superseded | **converged** | delta |
|---|---|---|---|
| CN mask AP50 | 0.5344 | **0.5969** | +0.0625 |
| CN mask AP75 | 0.1435 | **0.1748** | +0.0313 |
| CN mask AP50:95 | 0.2240 | **0.2578** | +0.0338 |
| CN box AP50 | 0.5310 | **0.5913** | +0.0603 |
| CN box AP75 | 0.1472 | **0.1831** | +0.0359 |
| detections | 152,690 | **196,927** | +44,237 |
| crowns matched (box IoU >= 0.5) | 18,103 | **19,603** | +1,500 |
| mean mask IoU (common subset) | 0.7368 | **0.7595** | +0.0227 |
| canopy-landing rate | 39.7% | 39.8% | +0.1 pt |

AP75 rose proportionally MORE than AP50 (+22% against +12%), so the retrain bought boundary
quality as well as recall -- consistent with defect 4, whose whole mechanism was pushing the
smallest third of crowns below the mask head's 28x28 ROI resolution.

### What this does to the comparison

1. **LACE still leads CN mask AP50**, but by **+0.066** (0.6625 +/- 0.0012 vs 0.5969), not the
   +0.128 previously published. Still 55x the seed sd, and still from boxes against full crown
   polygons.
2. **DetecTree2 now BEATS LACE at CN mask AP75** -- 0.1748 against 0.1649 +/- 0.0077. The claim
   that only Restor holds strict-IoU mask quality is dead; two mask-supervised models do.
3. **Box AP50:95 is a dead heat**: 0.2539 against LACE's 0.2546.
4. **DetecTree2 beats LACE on isolated mask quality**: 0.7595 against 0.7430 +/- 0.0013 on the
   10,020-crown common subset, where it previously "essentially tied". LACE's system lead rests
   on detection and ranking, not on the masker.
5. **Box2Mask Swin-L no longer out-localises it.** Swin-L's box AP50 of 0.5821 was said to exceed
   DetecTree2's 0.5310; the converged row is 0.5913, so DetecTree2 leads on every metric.

### maxDets 600 now binds harder

At the 0.05 floor the row emits 448.6 dets/tile (median 401, max 1511); maxDets 600 discards
**17.7% of detections on 30.5% of tiles**, against 7.0% before. Measured cost, scoring at 1200
instead of 600: mask AP50 0.5969 -> 0.6012 (**+0.0043**), AP75 +0.0001, AP50:95 +0.0010, box AP50
+0.0053. Quote +0.0043, not the old +0.0001. It is an order of magnitude below LACE's +0.066 lead
and cannot account for it.

### Scorer agreement

The frozen `score_coco.py` (pycocotools COCOeval) gives 0.5969; `evaluate.py` gives 0.6011 -- a
0.0042 gap, wider than the 0.0001 the two showed on the superseded row. Both are the respective
scorers behaving normally on a denser prediction set; **the published figure is the frozen
COCOeval 0.5969**. The `no_ignore_sensitivity` reproduction gate passes exactly
(0.6011 / 0.258), so the evaluate.py chain is self-consistent.
