# Box→mask comparison — minimal plan

## Why

LACE's contribution is a box→mask module, but nothing in the repo isolates the masker from the
detector against other methods. The one figure that tries — `masker_lab`'s EM 0.7473 vs SAM
0.7242 on GT boxes (`mps_tcd_multiseed_4phase/masker_lab/sam_bakeoff_results.json`) — is **not
citeable**: it is SAM **1** ViT-H against the **16 px** β=0 EM (not the shipped 8 px masker),
60 of 439 tiles, with SAM internally resized 2048→1024 in our favour. The same artifact also
shows the EM double-assigns 2× more than SAM on touching crowns (0.1058 vs 0.0524).

The cleanest experiment is **give every method the ground-truth boxes and compare mask IoU**.
No detection confound, no ranking, no score floor, no strata.

## Steps

Rate: A100-40GB ($0.000583/s) + 8 CPU + 64 GiB = **~$3.00/hr** all-in. Anchor for inference
estimates: the measured SAM 3 run — 439 tiles, 159,687 boxes, **97.3 min**
(`phase4_sam/sam3_results_439.json:wall_min`).

| # | step | GPU-h | $ | state |
|---|---|---|---|---|
| 1 | Fix the score-floor claim in `../PROTOCOL_439.md` | 0 | 0 | **done** |
| 2 | GT-box bake-off: shipped LACE masker vs SAM 3 vs SAM 3 FT | ~2 | 6 | **done** |
| 3 | Train **BoxInst** on our 792 box-only tiles | ~2 | 6 | |
| 4 | Box injection + round-trip validation for BoxInst | ~1 | 3 | |
| | debug / failed runs | ~3 | 9 | |
| | **total** | **~8** | **~$25** | |

Elapsed: 1–2 working days, dominated by the mmcv-full build, not GPU time.

**Step 2 is the deliverable.** Steps 3–4 prove the box-supervised literature can join the same
table: mmdet's `condinst_head.get_targets` picks the FPN level from box size (`regress_ranges`)
and the locations from the box extent / `center_sample_radius`, "fully computable from
ground-truth boxes without model predictions" — so BoxInst's own training-time assigner supplies
a box→mask inference path.

## Already done — no further cost

- `box_iou_strata.py` (run; `../results_439/box_iou_strata.json`) — mask IoU at matched box
  quality across LACE, Restor, DetecTree2, SelvaBox→SAM 3. Needs no prompting, so it already
  covers the methods injection cannot reach. New rows drop in as they are trained.
- `make_box_only_coco.py` — box-only COCO, with `verify()` asserting every polygon *is* its own
  bbox, so a config that silently fell back to mask supervision cannot pass unnoticed.
- `../modal_tcd_multiseed/phase4_sam/score_sam_vs_lace.py` — reproduces the published LACE rows
  exactly (0.6250 / 0.6615 / AP75 0.1718 / AP50:95 0.2821).

## Files

- new `boxinstseg_modal.py` — image + `imports_ok` gate + train/predict
- new `inject_boxes.py` — assigner-driven box→mask + round-trip validator
- reuse `/vol/data/{train,val}/images` on `tcd-detectree2-vol` — the *same PNG crops* DetecTree2
  trained on, so "same training data" is a filesystem fact, not two builders agreeing
- reuse `../detectree2_baseline/stitch.py`, `../score_coco.py`, and DetecTree2's schedule
  (`max_iter` 4000, `eval_period` 500, `patience` 6), selecting on val **box** AP50 — segm AP
  against the box-only COCO would score rectangles against rectangles

## Injection status — written and half-validated, 2026-09-01

`inject_boxes.py` implements box-conditioned inference for CondInst/BoxInst using THEIR OWN
training-time assigner: `get_targets` picks the FPN level from box size (`regress_ranges`) and
the locations from box extent / centre-sampling, using no model prediction, so it can be run at
inference to find the location the method would hold responsible for a given box. The controller
vector is read there and fed to the mask head exactly as `CondInst.simple_test` does.

| check | status |
|---|---|
| assigner places real boxes on a location | **PASS — 99.85%** of val GT boxes (`injection_gate`) |
| injected masks reproduce native masks | **OPEN** — needs the trained checkpoint |

The round trip cannot be run before training after all: it needs plausible boxes, and an
untrained regression head emits ~5x6 px degenerate boxes that the assigner correctly refuses
(median IoU 0.49 on those is meaningless). Re-run `injection_gate --trained` after training.

**Gate for the module table: median native-vs-injected IoU >= 0.95.** Below that, the two paths
disagree and BoxInst appears in the SYSTEM and STRATA tables only. The 0.15% of boxes that
receive no location at all (size outside every regress range, or centre-sampling excluding every
cell) have no defined mask under this method and must be reported, never silently dropped.

**This is our adaptation, not the authors'.** Never report an injected-box number without the
round-trip figure beside it, and without the native system number as well.

## Verification

- **Injection fidelity — the gate.** Inject BoxInst's *own* boxes and compare the resulting
  masks against its native masks. **Median IoU ≥ 0.95 to proceed.** Below that the adaptation is
  broken and its injected-box numbers are noise; BoxInst then appears in the strata table only.
- **Harness sanity.** BoxInst's stitched 439-tile predictions must score through
  `../score_coco.py` in a plausible range beside DetecTree2's 0.5344.
- **Box-only supervision.** `make_box_only_coco.verify()` on the real volume JSONs.
- **No scorer drift.** `score_sam_vs_lace.py` must keep reproducing the published LACE rows.

## Deferred — decided 2026-09-01: BoxInst only for now

Add a second method only once step 4's round-trip test passes and the BoxInst row proves
informative. Costs and constraints verified, so this does not need re-researching:

| candidate | COCO ckpt (box-supervised) | COCO mask AP | builds here? |
|---|---|---|---|
| **Box2Mask** | R-50 / R-101 / **Swin-L** | 35.9 / 38.2 / **42.5** | **yes** |
| **DiscoBox** | R-50 / R-101 | 32.2 / 33.4 | **yes** |
| BoxInst *(running)* | R-50 / R-101 | 30.7 / 33.1 | yes |
| BoxLevelSet | **none** | — | yes, but no weights |
| MAL | separate Docker | — | module table only |

**CORRECTION (2026-09-01): the THC "toolchain wall" was wrong.** An earlier version of this plan
recorded BoxLevelSet and Box2Mask as blocked because `mmdet/ops/tree_filter` includes
`<THC/THC.h>`, deleted in torch 1.11, with no torch<=1.10 cp310 wheels. Auditing which THC
*symbols* are actually called shows the includes are **vestigial** -- the only one used is
`atomicAdd`, a CUDA built-in. `patch_boxinstseg.py` strips nine include lines and the op
compiles; `imports_ok` now reports all four detectors registered. No kernel or parameter is
touched, and the patch asserts that no real THC function has appeared upstream.

**Two decisions follow from the pivot to system (Table 1) rows:**

1. **Fine-tune from their COCO checkpoints, not ImageNet.** BoxInst's COCO weights were trained
   with box supervision only -- that is the method -- so there is NO mask contamination, and
   "box-supervised end to end" survives. This removes the data-starvation objection: 792 tiles
   is a fine-tuning set, matching how the DetecTree2 row was produced from its released
   checkpoint. Note DetecTree2 and Restor both start from COCO *instance-seg* weights, so they
   have seen masks and we have not.
2. **Box2Mask goes from worst pick to best.** It was ruled out for not being injectable (moot
   for a system row) and for being data-hungry (moot given a 50-epoch COCO checkpoint). At 42.5
   COCO mask AP it is the strongest box-supervised method available; beating a weaker one
   invites the obvious question. BoxLevelSet is now the odd one out -- no COCO weights, so it
   alone would train from ImageNet on 792 tiles.

## Canopy handling — decided 2026-09-01

**The problem.** BoxInst cannot be given a canopy-ignore label. `CondInstBoxHead.loss` accepts
`gt_bboxes_ignore` in its signature and docstring but never uses it, and the train pipeline's
`Collect` passes only `gt_bboxes`/`gt_labels`, so it never arrives. Canopy — 19% of training
annotations, ~25% of tile area, and full of real but undelineated crowns — would train as
ordinary BACKGROUND, teaching the model to suppress detections exactly where the frozen scorer
later makes them free. The row would then measure our data handling, not the method.

**The remedy: canopy blackout on the TRAINING crops**, following CanopyRS / SelvaBox, who train
on OAM-TCD this way. Each arm then gets canopy-neutral training by whatever mechanism its
framework allows — ignore label (LACE), `iscrowd` honoured by detectron2 (DetecTree2), blackout
(BoxInst). Same goal, three mechanisms, none of them a modification to the method.

**One necessary departure from their recipe.** Canopy and crown polygons are NOT disjoint in
OAM-TCD: **6.36% of crown area falls inside a canopy polygon**, and a naive blackout erases
**3.4% of crowns outright** and damages **9.5%**. So the mask is `canopy AND NOT crown` — every
labelled crown pixel survives, asserted per subtile in `prep_data_blackout`. A crown sitting
inside canopy is left as an island of real imagery in a blacked field, which is what it is.

**Split treatment, deliberately:**

| split | images | annotations | why |
|---|---|---|---|
| train | canopy blacked | crowns only | canopy must not be a negative |
| val | **original** | crowns + canopy as `iscrowd` | select under the TEST condition: real imagery, canopy ignored by pycocotools, exactly as `score_coco.py` does on the 439 |
| test | **original, untouched** | frozen protocol | must stay byte-identical across rows |

Selecting on blacked val images would optimise for a distribution that exists only during
training; selecting against a canopy-free val GT would favour checkpoints that suppress canopy —
the very bias the blackout removes.

**Measured effect:** 23.24% of training pixels blacked (19.82% of val's, though val trains on
nothing). Removing the canopy annotations emptied 618 train subtiles, which mmdet's
`filter_empty_gt=True` would have dropped -- we set it to **False**, because that emptiness is an
artefact of our blackout rather than of the data: those subtiles carried canopy annotations until
we stripped them, and DetecTree2 trained on all of them. They are bimodal (292 under 50% blacked,
real ground/road with no trees -- genuine hard negatives; 152 over 90% blacked, trivial and 2.5%
of the set), so keeping all of them is simpler than any threshold and avoids letting our own
intervention shrink the baseline's data.

**Cost of this decision:** it breaks the "identical PNG crops" guarantee for the training split
only. BoxInst trains on modified imagery where DetecTree2 trained on originals. That is a real
asymmetry, but the alternative — canopy as a trained negative — is worse and not defensible.

## Asymmetries to state, not hide

- LACE's DINOv3 ViT-L is **frozen** (zero trained backbone params); BoxInst fine-tunes R-50.
  Not symmetric in any configuration.
- **BoxInst gets ~2.1x LACE's pixel exposure per epoch, and ~2.4x the crown instances.** Both
  train on the same 792 source tiles, but LACE consumes whole 2048^2 tiles (3.32 G px/epoch,
  54,191 crowns) while BoxInst uses the 1024@50%-overlap grid (6,692 subtiles, 7.02 G px/epoch,
  127,692 crown annotations) -- interior pixels fall inside up to four subtiles. KEEP IT: the
  asymmetry favours the baseline, so a LACE win survives it, and the grid is the one the
  DetecTree2 and SelvaBox rows already use. Deliberately shrinking a baseline to match our
  exposure would be worse than stating the asymmetry.
  *Contingency:* if BoxInst WINS, re-run it on a non-overlapping 2x2 grid (4 subtiles/tile =
  exactly 1x pixel exposure) as a control, ~$8. Not worth pre-paying.
- **"Epochs" are not a comparable unit across the two methods** -- LACE is 40 max epochs at
  batch 3 over whole tiles with early stopping; BoxInst's 1x is 12 epochs over overlapping
  subtiles. Each runs its own published recipe; do not present this as schedule parity.
- The "Imgs = 900" column of Table 1 is honest for a BoxInst row (same source images) but does
  not convey the exposure difference. Footnote it.
- Baselines init from ImageNet classification weights, never COCO instance-seg weights — the
  latter would inject exactly the mask supervision the comparison is about. This is BoxInstSeg's
  own default, so it costs nothing.
- Injection is our adaptation, not the authors'. Always report beside the native system number.
- The controller was trained under the model's own detection distribution, so injecting a box
  where the model sees nothing reads an out-of-distribution controller. Report objectness at
  each injected location and the coverage fraction; never average over unsupported readouts.
- LACE's 8 px grid caps mask resolution. In the box-IoU ≥ 0.9 stratum the box is nearly perfect,
  so what remains is largely drawing resolution — check `../ablation/results/grid_8v16.json`
  before attributing that gap to the model.

## Result — step 2, runs 2026-09-01

439 tiles · 25,692 valid crowns · identical GT prompt boxes · `results_439/gtbox_bakeoff_r2048.json`
(and `gtbox_bakeoff.json` for the 512 arm). Per-crown IoUs AND masks persisted per arm, so any
further resolution / ranking / canopy question is a free re-score.

**Scored at 2048 (native imagery resolution) — the headline:**

| arm | mean IoU | >=0.5 | >=0.75 | >=0.9 | double-assign | paired vs LACE |
|---|---|---|---|---|---|---|
| **LACE commonality-EM (shipped)** | **0.7698** | 0.9786 | 0.6588 | 0.0513 | 0.0302 | — |
| SAM 3, zero-shot @2048 | 0.7406 | 0.9555 | 0.5452 | 0.0368 | 0.0082 | −0.0293, wins 40.1% |
| SAM 3, SelvaMask-FT @1777 | 0.6232 | 0.7885 | 0.2741 | 0.0107 | 0.0023 | −0.1467, wins 19.9% |

**Scored at 512 (the rest of the paper's raster):** LACE 0.7856 / 0.9862 / 0.7096 / 0.0891 /
0.0358; SAM 3 0.7493 / 0.9775 / 0.5649 / 0.0373 / 0.0084 (wins 36.3%); SAM 3 FT 0.6048 / 0.7688 /
0.2343 / 0.0076 / 0.0022 (wins 14.5%).

Notes that belong in the paper:
- **LACE wins every metric at both rasters.** Paired win rate 59.9% of crowns at 2048 (63.7% at
  512) — report the paired rate, not the difference of means: the arms are matched crown-for-crown.
- **The raster cost us more than SAM.** 512 -> 2048 costs LACE 0.0158 mean IoU against SAM's
  0.0087, because a coarse raster smooths away boundary error an 8 px-lattice masker cannot avoid.
  The mean-IoU gap narrows 0.0363 -> 0.0292.
- **The >=0.9 column at 512 was largely a scorer artefact.** Our advantage there falls from 2.4x
  to 1.4x on moving to 2048. At 512 a one-pixel boundary error leaves median IoU 0.71 and only
  1.3% of crowns above 0.9, against 0.90 and 51.4% at 2048 (`ablation/results/raster_quantum.json`,
  `ablation/scripts/t15_raster_quantum.py`). Do not quote the 2.4x.
- **Fine-tuning made SAM worse** (0.6232 vs 0.7406): SelvaMask FT at 1.3–3.5 cm/px applied at
  10 cm/px, even run at the 1777 tile size CanopyRS deploys.
- **Double-assignment is the real weakness and it is worse than first reported.** LACE 0.0302 at
  2048 against a GT self-overlap floor of **0.0015** — roughly 20x. (An earlier note quoted a
  1.21% floor; that was a SINGLE TILE. Averaged over 120 tiles the floor is 0.0048 at 512 and
  0.0015 at 2048.) SAM sits at 5x the floor, SAM FT at 1.5x.
- **The posterior product plays no part** — masks byte-identical with and without it. This is a
  mask-shape result; the product's contribution is to AP, separately.
- **No seed band.** `_selfmask_npz` is not seed-scoped: one `em_model_4p_fix.npz` is shared by all
  three Table-1 seeds, which differ only in the detector — and this arm loads no detector. The
  `_fix` fit's own seed is unrecorded (`ablation/results/em_fit_report_4p_fix.json` is
  reconstructed from the npz); `fit_masker_4p` defaults to 0. Inference is deterministic.
- Written up as Table `tab:gtbox` in `../ablation/results/paper.tex`, 2048 as headline with 512
  as the raster sensitivity block.
