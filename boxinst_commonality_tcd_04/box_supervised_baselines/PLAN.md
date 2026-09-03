# Box-supervised baselines — plan & handover

**Status 2026-09-02.** Step 2 (GT-box bake-off) is DONE and written into the paper. Step 3 is
**COMPLETE for Box2Mask.** The row is scored, in `tab:oamtcd439` and in
`results_439/table_439.md`: **mask AP50 0.3865** canopy-neutral, against LACE's 0.6625. §5b has
the training curve (it peaked and turned over, so the row is NOT a lower bound) and §5c has the
result plus the five checks run before believing it.
Budget for the whole of step 3 is **$20 of Modal** (raised from $10 on 2026-09-02 so the
baseline could not be dismissed as undertrained); §5 has the schedule arithmetic. Spend to the
start of the 9-epoch run ≈ $2.9.

---

## 1. What this phase is for

Table 1 compares LACE against mask-supervised systems (Restor, DetecTree2) and a
detector+SAM pipeline (SelvaBox). It contains **no box-supervised competitor**, which is the
family LACE actually belongs to. This phase adds one.

**Scope decision, 2026-09-02: these are SYSTEM rows for Table 1, not box-to-mask module rows.**
See §6 for why — briefly, CondInst/SOLOv2-family mask heads are conditioned on a feature-map
LOCATION, not on box coordinates, so they cannot be prompted with a box the way LACE and SAM
can. The module table (`tab:gtbox`) stays LACE vs SAM 3.

## 2. Order of work

| # | step | state |
|---|---|---|
| 1 | toolchain gate (`imports_ok`) | **DONE** — all four methods register |
| 2 | data prep + canopy blackout | **DONE** — verified, §4 |
| 3a | generalise the harness, build Box2Mask's config | **DONE** — §5 |
| 3b | `fetch_ckpt` — their COCO-box weights | **DONE** — 610 tensors, cls head dropped |
| 3c | `dryrun --method box2mask` | **DONE** — all gates pass, §9 |
| 3d | fp16 feasibility probe | **DONE** — impossible, §5. Do not retry |
| 3e | `--smoke` gate on the validation/selection path | **DONE** — §5, $0.29 |
| 3f | train Box2Mask (9-epoch budget, early stop) | **DONE** — stopped itself at epoch 6, best at epoch 4, $9.41. §5b |
| 4a | predict → stitch → score → table → paper row (R-50) | **DONE** — §5c |
| 4b | Swin-L arm: their strongest backbone | **DONE** — §5d, mask AP50 0.5396 |
| 4 | predict → stitch → `score_coco.py` | harness written, untested on real preds |
| 5 | train BoxInst | DEFERRED — out of scope for the $10 budget |
| 6 | DiscoBox / MAL | optional, §7 |

**Box2Mask leads because it is the SOTA box-supervised method** (42.5 COCO mask AP with Swin-L,
TPAMI 2024) against BoxInst's 33.2 (CVPR 2021). Publishing against BoxInst alone invites "why
the 2021 method?". BoxInst is still worth having afterwards as the seminal, most-cited
reference point, and its harness is already wired and dry-run green.

## 3. Environment — built and verified

```
torch 1.11.0+cu113 · mmcv-full 1.5.3 · mmdet 2.25.0 (BoxInstSeg fork) · numpy 1.23.5 · py3.10
nvidia/cuda:11.3.1-devel base · A100 · registered: CondInst, DiscoBoxSOLOv2, BoxLevelSet, Box2Mask
```

Everything below was a real failure that cost a build cycle. Do not "simplify" any of it:

- **Python must be 3.10+** (Modal's builder) and mmcv/torch wheels must exist for that exact
  `(mmcv, torch, cu, cpXX)` triple. Missing wheel = silent source build that needs nvcc and
  usually fails. `imports_ok` asserts versions AND calls an mmcv CUDA op, because a mismatch
  typically imports fine and dies later in the registry.
- **numpy must stay 1.x.** torch 1.11's C extension is built against the numpy-1 ABI; numpy 2
  gives "_ARRAY_API not found" as a WARNING at import and an error deep in training. It is
  re-pinned as the LAST build step because BoxInstSeg's `runtime.txt` lists a bare `numpy`.
- **nvcc IS required**, despite the top-level `setup.py` saying `ext_modules=[]`. The ops live in
  sub-packages with their own setup.py: `pairwise` (BoxInst) and `tree_filter` (BoxLevelSet,
  Box2Mask). Both are imported at module scope by heads on the package init chain, so `mmdet`
  will not import without them.
- **`pairwise` builds in place** (`build_ext --inplace`, relative import); **`tree_filter` must be
  installed** (`build develop`, top-level `import tree_filter_cuda`) and needs a **GPU build
  layer**, because its setup.py gates on `torch.cuda.is_available()` and ignores `FORCE_CUDA`.
- **`CC`/`CXX` must be pinned to gcc/g++** — the base image leaves CXX pointing at clang++ and
  torch's cpp_extension probes for it and fails.
- **`patch_boxinstseg.py` strips 9 vestigial THC includes** from tree_filter so it compiles
  against torch >= 1.11. Safe: the only THC symbol used is `atomicAdd`, a CUDA built-in. The
  patch asserts no real THC function has appeared upstream. An earlier version of this plan
  wrongly recorded these two methods as blocked by a "toolchain wall"; they are not.
- **`scikit-image` is an undeclared dependency** of `condinst_head`.

## 4. Data — prepped and verified

On volume `tcd-detectree2-vol`, the SAME PNG crops DetecTree2 trained on:

| split | images | annotations | source |
|---|---|---|---|
| train | 6,692 subtiles | 127,692 crowns, **0 canopy** | `data/train/images_canopyblack/` + `coco_boxonly_black.json` |
| val | 896 subtiles | 16,505 crowns, 3,900 canopy `iscrowd` | `data/val/images/` (ORIGINAL) + `coco_boxonly.json` |
| test | 3,951 subtiles (439×9, full coverage) | — | `data/test/images/` |

**Canopy handling — do not change without reading this.** BoxInst-family heads accept
`gt_bboxes_ignore` and never use it, and the pipeline's `Collect` does not pass it, so canopy
(19% of annotations, ~25% of tile area, full of real undelineated crowns) would train as
ordinary BACKGROUND — teaching the model to suppress detections exactly where the scorer later
makes them free. Remedy: canopy blackout on the TRAINING crops, as CanopyRS/SelvaBox do on this
dataset. **23.24% of train pixels are blacked.**

One necessary departure from their recipe: canopy and crown polygons are NOT disjoint (6.36% of
crown area sits inside canopy; a naive blackout erases 3.4% of crowns and damages 9.5%), so the
mask is `canopy AND NOT crown`, asserted per subtile.

**HONEST LIMITATION — the blackout is REASONED, not MEASURED, and it is a handicap unique to the
box-supervised arms.** Restor, DetecTree2 and LACE all face canopy without degrading their input:
Restor has a canopy class, DetecTree2 trains on the original crops, LACE uses the ignore rule. The
BoxInst family cannot, so these arms alone train on imagery with 23.24% of pixels blacked out. The
claim that the alternative (canopy annotations simply dropped, imagery left intact, canopy
therefore training as background) is WORSE is an argument from how the loss works, not an
experiment — no arm has been trained that way. It is the one assumption in this phase that
disadvantages the baseline and has no measurement behind it. If a reviewer presses on the
box-supervised row, this is the weakest joint, and closing it costs one more training run
(~$12 at R-50, ~$14 at Swin-L). Say "we blacked canopy because these heads cannot ignore it, and
we did not measure the counterfactual" rather than implying the counterfactual was tested.

**Val deliberately differs from train**: original imagery, canopy as `iscrowd`. Selection must
happen under the TEST condition. Selecting on blacked val images optimises for a distribution
that only exists in training; selecting against canopy-free val GT favours canopy-suppressing
checkpoints — the bias the blackout just removed.

`filter_empty_gt=False` on train: stripping canopy emptied 618 subtiles which mmdet would drop.
That emptiness is OUR artefact — they carried canopy annotations until we stripped them, and
DetecTree2 trained on all of them. They are bimodal (292 under 50% blacked = real ground/road,
genuine hard negatives; 152 over 90% blacked = trivial, 2.5% of the set).

## 5. Training Box2Mask — what was actually built

`_build_cfg` now takes `--method` and dispatches to `_cfg_boxinst` / `_cfg_box2mask`; each loads
that method's OWN published config from `/opt/BoxInstSeg/configs/` and changes only what our data
forces. `_attach_data` is shared, so both arms see byte-identical splits.

| | BoxInst (unchanged) | Box2Mask (built) |
|---|---|---|
| base config | plain COCO instance | `coco_panoptic.py`, but their config already `_delete_`s `data` and uses `CocoDataset` — nothing to swap |
| runner | `EpochBasedRunner`, 12 ep | `IterBasedRunner`, rescaled — see below |
| optimiser | SGD, LR-rescaled for our batch | **AdamW 1e-4 exactly theirs** — see gradient accumulation |
| input size | multiscale, rebased to 1024 | `image_size = (1024, 1024)` already; LSJ 0.1–2.0 untouched |
| test scale | 1024 | (1333, 800) → **(1024, 1024)**, our native subtile, no resampling |
| classes | `num_classes = 1` | `num_thing_classes=1`, `num_stuff_classes=0`, `class_weight=[1.0, 0.1]` |
| eval / ckpt | every epoch | every `iters_per_epoch`; their `dynamic_intervals` dropped so both arms select identically |

**Supervision is box-only three times over, and the third is now asserted.** Our COCO carries
rectangles (`make_box_only_coco.verify()`), `LoadAnnotations` is called with `with_mask=False`,
and `GenerateBoxMask` then paints `gt_masks` from `gt_bboxes` alone. Box2Mask is the risky case
because it *does* carry a `gt_masks` key, so `dryrun` now pulls one real sample through the real
pipeline and checks every mask is exactly its own bounding rectangle: **249/249 rectangular, 0
otherwise.** A config that silently fell back to mask supervision cannot pass.

**Their effective batch is restored, not their LR rescaled.** Their recipe is 8 GPUs × 2 images.
We have one GPU that holds 2 at 1024² (24.8 GB measured). Rather than guess an AdamW LR for
batch 2, `GradientCumulativeOptimizerHook` accumulates 8 micro-batches, so the effective batch is
exactly 16 and `lr=1e-4`, the betas, `weight_decay=0.05`, the `0.1` backbone `lr_mult` and the
`0.01` grad clip are all theirs untouched. mmcv warns that accumulation can hurt with BatchNorm —
it does not apply here: their R-50 is `norm_cfg=dict(type='BN', requires_grad=False)` with
`norm_eval=True`, so BN is frozen and accumulation is arithmetically identical to the real batch.
An "iteration" in the logs is therefore one micro-batch of 2.

**Schedule: 9 epochs with early stopping, chosen by a RULE rather than by us.**
Their 368,750 iters × 16 ÷ 118,287 COCO images = 50 epochs; reused as-is on 6,692 subtiles that
would be ~880 epochs, so it has to be rescaled. A `--max-iters 400` timing probe measured
**0.75–0.95 s per micro-iteration** (memory 24.8 GB), i.e. **$1.46 per epoch** on an A100-40GB.

An earlier 3-epoch run was started and KILLED at iter ~1,300, because 3 epochs is defensible
against the DetecTree2 row but not against LACE:

**Quote LACE's ACTUAL `best_epoch`, never its 12-epoch early-stop floor.** LACE evaluates every
5 epochs with patience 2, so it selects at 15–20 and trains on to 25–30. The Table 1 detectors are
`phase4/preds/ref_knobbed_s{0,1,2}.json` (mask mAP50 0.625 / 0.6358 / 0.6294, matching Table 1;
the posterior product is a post-hoc rerank of the same predictions and does not change training):
**`best_epoch` = 20 / 15 / 20**, mean 18.3. An earlier draft of this section compared against the
12-epoch floor and so understated LACE's real training by ~50%.

Compare in **passes over the same 792 tiles**, because the 1024@50% grid double-counts: 127,692
annotations ÷ 54,191 unique crowns = 2.36×, and 7.02 ÷ 3.32 G px = 2.11×. Raw totals flatter the
baseline; passes do not.

| arm | crown-passes | pixel-passes | raw crown instances | raw pixels |
|---|---|---|---|---|
| DetecTree2 (fine-tuned) | 5.6 | 5.1 | ~306 k | 16.8 G |
| Box2Mask @ 3 epochs (killed) | 7.1 | 6.3 | 383 k | 21.1 G |
| **Box2Mask @ 9 epochs (run)** | **21.2** | **19.0** | **1.15 M** | 63.2 G |
| LACE s1 (worst seed, 15 ep) | 15 | 15 | 813 k | 49.8 G |
| LACE s0 / s2 (20 ep) | 20 | 20 | 1.08 M | 66.4 G |
| LACE 3-seed mean | 18.3 | 18.3 | 993 k | 60.9 G |

At 3 epochs Box2Mask got 26% more exposure than DetecTree2 but under half of LACE's, while
training 44M parameters against LACE's ~4M head params on a frozen backbone — it needs more
training, not less. At 9 epochs it is ~3.8× DetecTree2 and at or above EVERY LACE seed on
crown-passes (21.2 vs 15–20) and above the 3-seed mean on pixel-passes (19.0 vs 18.3). The one
sub-metric where it trails is raw pixels against the two 20-epoch seeds (63.2 G vs 66.4 G,
0.95×); a tenth epoch would fix that for $1.5 and was judged not worth the budget slack. **Do not read DetecTree2 as the
generous comparator**: its own val AP50 history is `48.4 → 51.3 → 52.8 → 53.5 → 54.2 → 54.7 →
56.0 → 57.1` with `counter: 0` — it hit its 4,000-iter cap while still improving and never
early-stopped (`tcd-detectree2-vol:out/train_s0/ap_history.json`).

Epoch-matching is a weak argument either way, so the row does not rest on it. **Early stopping on
val box AP50 is the actual protocol**, the same rule LACE uses (min 12 epochs, patience 2) and the
DetecTree2 row used (patience 6 evals × 500 iters ≈ 1.8 epochs): here `patience 4` half-epoch
evals ≈ 2 epochs, `min_epochs 5`, evaluated every half epoch for 18 curve points. If the curve
flattens, "undertrained" is refuted empirically at whatever epoch that happens — which is
evidence the DetecTree2 row in the same table does not have. The LR decays stay at the fractions
COCO used (327,778/368,750 = 88.89% and 355,092/368,750 = 96.30%) of the 9-epoch total.

**fp16 is IMPOSSIBLE here — do not try it again.** It looks free (mmcv 1.5.3 ships
`GradientCumulativeFp16OptimizerHook`, which composes with the accumulation above), and it is not.
`BaseDetector.forward` carries `@auto_fp16(apply_to=('img',))`, so `wrap_fp16_model` casts `img`
to half and runs the WHOLE forward under `autocast(enabled=True)` — and two of the CUDA ops on
that path are float-only:
- `ms_deform_attn_forward_cuda` (mmcv 1.5.3), under Box2Mask's `MSDeformAttnPixelDecoder`. This
  is what actually threw: `RuntimeError: "ms_deform_attn_forward_cuda" not implemented for
  'Half'`. Fixing it means patching mmcv, not our code.
- `tree_filter`, under the level-set loss, whose kernels are hard-coded `float*` /
  `.data<float>()` (`src/refine/refine.cu`, `src/mst/mst.cu`) and would have thrown next.

Measured by a `--max-iters 300 --fp16` probe for ~$0.15, which is the right way to settle it.

Two guards, both stopping at an evaluation boundary so `EvalHook` fires and `save_best` writes
before the run ends: `APEarlyStop` (the rule above) and `WallClockGuard` (`--max-hours`, pure cost
insurance). A run that trips the latter is TRUNCATED and `sched["max_hours"]` records it.

**`--smoke` exists because the selection path is otherwise unproven until 40 minutes into a paid
run.** `--max-iters N --smoke` trains N iterations then forces ONE full validation, exercising
EvalHook over Box2Mask's instance results, the `save_best` write, and `APEarlyStop` reading
`runner.meta['hook_msgs']['best_score']`. Verified for $0.29:
`Now best checkpoint is saved as best_bbox_mAP_50_iter_60.pth` · `[APEarlyStop] eval 1 · best val
bbox_mAP_50 0.074 · 0/1 non-improving`. A full validation costs only **45 s**, which is why
half-epoch evaluation is affordable. Note in-training selection uses mmdet's default
`maxDets=1000`; that is the SELECTION signal only and does not touch the frozen 439 protocol,
which applies maxDets 600 in `score_coco.py`.

**Initialised from their COCO checkpoint, not ImageNet** (`fetch_ckpt`, gdown from the Drive link
in their model zoo; R-50 35.9 mask AP). Downloaded 528.6 MB, **610 tensors kept**, and exactly
`panoptic_head.cls_embed.{weight,bias}` (81×256, 81) dropped — named and asserted rather than left
to mmcv's shape-mismatch warning. `train` then re-checks the key sets on CPU before any GPU time,
because `runner.load_checkpoint` loads with `strict=False`: a checkpoint that matched almost
nothing would train silently from scratch and only surface hours later as a bad AP. Why this is
the single biggest fairness improvement in the phase:
- It removes the data-starvation objection: 44M params from ImageNet init on 792 tiles is not a
  fair test of the method; their published numbers used 118k COCO images.
- It matches how the DetecTree2 row was produced (fine-tune of a released checkpoint).
- **It does NOT leak masks**: Box2Mask's COCO training used box annotations only — that is the
  method — so those weights have never seen a mask. "Box-supervised end to end" survives.
  DetecTree2 and Restor both start from COCO *instance-seg* weights and do not have this property.

**`num_queries = 100` is an architectural ceiling, and it binds.** Unlike Restor's RPN top-k it
cannot be raised without discarding the COCO initialisation, because the query embeddings are
learned weights of that size. With `num_classes = 1` the `max_per_image=100` top-k is a no-op, so
100 per 1024² subtile is the true cap. Measured: **81 of 6,692 train subtiles and 7 of 896 val
subtiles hold more than 100 crowns; 2,768 of 127,692 train boxes (2.2%) sit beyond it.** At test
time nine overlapping subtiles give at most 900 per 2048² tile against the protocol's maxDets 600,
so the cap is not binding at tile level for the median tile — `predict` counts and reports the
subtiles that actually saturate.

**`--detach` is mandatory for anything over a few minutes.** A local network blip killed two
running apps at once (`App state is APP_STATE_STOPPED`), losing a 350/400 probe and a 58%
checkpoint download. Detached runs cannot return a value to the client, so every long function
now also writes its result to `/vol/boxinst/reports/<name>.json`.

## 5b. Training result — the val curve PEAKED, so the row is not a lower bound

Run: seed 0, 9-epoch budget, early stop on val box AP50 (patience 4 half-epoch evals,
`min_epochs` 5). **Stopped itself at iter 20,076 (epoch 6)**; 268.8 min on one A100-40GB, $9.41.
Selected checkpoint `best_bbox_mAP_50_iter_13384.pth` = **epoch 4**.

| eval | epoch | val box AP50 | val box AP50:95 | | eval | epoch | val box AP50 | val box AP50:95 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.5 | 0.4140 | 0.2070 | | 7 | 3.5 | 0.4670 | 0.2450 |
| 2 | 1.0 | 0.4190 | 0.2170 | | **8** | **4.0** | **0.4790** | **0.2540** |
| 3 | 1.5 | 0.4460 | 0.2360 | | 9 | 4.5 | 0.4460 | 0.2360 |
| 4 | 2.0 | 0.4280 | 0.2230 | | 10 | 5.0 | 0.4400 | 0.2310 |
| 5 | 2.5 | 0.4550 | 0.2400 | | 11 | 5.5 | 0.4360 | 0.2200 |
| 6 | 3.0 | 0.4470 | 0.2350 | | 12 | 6.0 | 0.4450 | 0.2270 |

**This is the answer to "is the baseline undertrained?", and it is an empirical one.** The curve
rises to a peak at epoch 4 and then sits at ~0.44 for four consecutive evaluations, all below the
peak, while TRAIN loss keeps falling to its minimum (21.18 at eval 11) — mild overfitting, which
is what 44M trainable parameters on 792 tiles should do. We OFFERED 9 epochs; the model's own
validation curve chose epoch 4 and more training made it worse. Contrast the DetecTree2 row in
the same table, which hit its 4,000-iter cap while still improving and has no such evidence.

Consequences for how the row is written up:
- Do **not** call it a lower bound, and do not apologise for the epoch count. The defensible
  claim is "trained under the same early-stopping rule as LACE, selected at its validation
  optimum, with four subsequent evaluations confirming the optimum had passed".
- The SELECTED model saw 4 epochs = 9.4 crown-passes, fewer than LACE's 15–20 (§5). That is not
  a handicap we imposed: the budget was 9 epochs (21.2 crown-passes, above every LACE seed) and
  the method peaked early. Both rows are early-stopped on val; they simply peak at different
  points. State it that way.
- The earlier 3-epoch run would have selected ~0.455 against this 0.479, so killing it was
  correct — but the gap is small, which is worth knowing before spending on any future arm.
- Nothing here required the LR decay (scheduled at iters 26,768/28,999): the run never reached
  it. A step schedule whose decay sits at 88.9% composes awkwardly with patience-based early
  stopping — if a future arm is expected to keep improving, either shorten the schedule so the
  decay lands inside it or use a plateau/cosine schedule.

## 5c. Result — the row, and why it is trustworthy

`results_439/box2mask_s0_coco512.json`, frozen protocol (COCOeval, 439 tiles, masks at 512²,
maxDets 600, floor 0.05, canopy `iscrowd`). Canopy-neutral:

| | mask AP50 | mask AP75 | mask AP50:95 | box AP50 | dets |
|---|---|---|---|---|---|
| **Box2Mask R-50** | **0.3865** | 0.0509 | 0.1299 | 0.4707 | 117,027 |
| DetecTree2 (converged 2026-09-03) | 0.5969 | 0.1748 | 0.2578 | 0.5913 | 196,927 |
| Restor (rpn 1000) | 0.6255 | 0.1872 | 0.2766 | 0.6536 | 56,438 |
| LACE + product, 3-seed | 0.6625 | 0.1649 | 0.2790 | 0.6396 | 166,434 |

It is last by a wide margin, so the harness was checked before the number was believed:

1. **Val→test consistency.** Val box AP50 was 0.479 on 896 held-out 1024 subtiles; the stitched
   439-tile box AP50 is 0.4707. The detector transfers almost exactly, so the tiling, stitching
   and scoring path is sound — a geometry bug would not preserve that.
2. **`box_iou_strata.py` isolates the masker** (`results_439/box_iou_strata.json`). On the 10,020
   crowns every row found, mean mask IoU at matched box quality: Restor 0.7655,
   DetecTree2 0.7595, LACE 0.7430 ± 0.0013, SelvaBox→SAM3 0.6960, Box2Mask Swin-L 0.6943,
   **Box2Mask R-50 0.6436**. A smooth ordering, not the 0.2–0.3 or the erratic values a
   mask-decode bug produces, so masks decode correctly and are simply weaker.
3. **Its boxes are NOT the problem.** Mean box IoU on matched detections is 0.7365; the AP gap is
   recall (14,293 crowns matched vs DetecTree2's 19,603) plus mask quality.
4. **No truncation artefacts.** `tiles_over_maxdets_600` is **0** (mean 266.6 dets/tile, max 522),
   so the maxDets cap never binds and no sensitivity row is needed. The 100-query architectural
   ceiling binds on only **2.7%** of subtiles (108 of 3,951).
5. **Canopy landing 51.0%**, against LACE's 42.5% and Restor-pooled's 42.2% — higher, as expected
   from training on blacked canopy and testing on real imagery, but free here because no tile
   approaches the 600 budget. This was the risk PLAN §9 flagged; it is clear.

The honest reading: Box2Mask's level-set masks degrade at 0.1 m/px on crowns far smaller than the
COCO objects it was developed on. That is a resolution effect, not a refutation of level-set box
supervision — say it that way.

**Mean mask IoU is now a column in every 439 table** (`tab:oamtcd439`, `PROTOCOL_439.md`,
`results_439/table_439.md`), sourced from `box_iou_strata.json` and generated by `build_table.py`
rather than typed. It reorders the table relative to AP and that is worth saying out loud:
**Both** mask-supervised models beat LACE's masker on it — Restor 0.7655 and DetecTree2 0.7595
against LACE's 0.7430 ± 0.0013 — while DetecTree2 trails LACE by 0.066 AP50; SelvaBox is 0.6960,
Box2Mask Swin-L 0.6943 and R-50 last at 0.6436. LACE's system lead therefore rests on detection
and ranking, NOT on mask quality; the masker's outright win is on GT prompt boxes (`tab:gtbox`),
a different question. See PROTOCOL_439.md §0b for the caveats (no recall term, row-set
dependence — the subset moved 11,238 → 10,020 when DetecTree2 was retrained).

**Total phase spend ≈ $15.4 of the $20 budget** (train $9.41, predict $1.53, the rest probes,
smoke, dry-runs and the killed 3-epoch run).

## 5d. Swin-L arm — the strongest published configuration

R-50 is Box2Mask's WEAKEST released backbone (35.9 COCO mask AP vs R-101 38.2, Swin-L 42.5), and
§7 justified choosing Box2Mask over BoxInst by the Swin-L number — so shipping only R-50 was an
internal inconsistency. Swin-L was run 2026-09-02/03.

**Two defects in their published Swin-L config, both fixed deliberately (see `_cfg_box2mask`):**
1. `custom_keys` (per-layer AdamW decay/LR multipliers) is generated from `depths` in the Swin-T
   file; Swin-L redefines `depths [2,2,6,2] -> [2,2,18,2]` AFTER that dict is built, and mmcv
   config inheritance is a dict merge, not a re-execution. Twelve of stage 2's blocks would
   silently miss the norm no-decay rule. Rebuilt for the real depths.
2. The file is named `8x1` but inherits `samples_per_gpu=4` and `max_iters=184376` from Swin-T,
   which is 50 COCO epochs only at effective batch 32. The config is self-inconsistent, so there
   is NO "their recipe" for the batch/LR pair. **We chose effective batch 16 and lr 1e-4 —
   identical to our R-50 arm — so the backbone is the ONLY difference between our two rows.**
   Report this as our choice, never theirs.

Hardware: A100-**80GB** (`train80`), micro-batch 1, accumulation 16, 27.5 GB peak. 40GB looks
sufficient from the early reading but is NOT: the R-50 run's memory grew 21,360 -> 32,907 MB
(+54%) over its run as denser subtiles arrived, and the same growth on Swin-L projects past 40 GB.

**Val curve (evals every half epoch). Peak at epoch 3.5; selected checkpoint `iter_23422`.**

| eval | ep | AP50 | | eval | ep | AP50 |
|---|---|---|---|---|---|---|
| 1 | 0.5 | 0.5370 | | 6 | 3.0 | 0.5780 |
| 2 | 1.0 | 0.5420 | | **7** | **3.5** | **0.5960 ← peak** |
| 3 | 1.5 | 0.5670 | | 8 | 4.0 | 0.5900 |
| 4 | 2.0 | 0.5720 | | 9 | 4.5 | 0.5900 |
| 5 | 2.5 | 0.5730 | | 10 | 5.0 | 0.5900 |

Phase 1 stopped on the 5 h `WallClockGuard` at epoch 4.5 ($13.55). It was resumed to reach the LR
decays (epochs 5.33/5.78) but killed at epoch 4.8 once evals 8–10 showed a flat 0.5900 below the
peak — **the resume bought nothing** and that is worth remembering: the decision was made on
eval 6→7 (`0.578 -> 0.596`, still climbing) and the peak was already in.

**A RESUME TRAP THAT NEARLY MIS-SELECTED THE MODEL.** The resumed run's `EvalHook` started
`best_score` from -inf and wrote a SECOND `best_*.pth` at 0.590 — worse than the pre-resume 0.596.
It did not delete the older file (it had no record of it), so both survive. `predict` originally
took `sorted(glob("best_*.pth"))[-1]`, which is the lexicographically LAST filename, i.e. the
latest iteration — it would have silently scored the worse checkpoint. **Fixed: `predict` now
reads `meta['hook_msgs']['best_score']` from every candidate and selects the maximum**, printing
all candidates and recording them in the report. Verified:
`candidate iter_23422 best_score=0.596 <- SELECTED` / `candidate iter_30114 best_score=0.59`.

**Result, frozen protocol** (`results_439/box2mask_swinl_s0_coco512.json`), canopy-neutral:

| | mask AP50 | mask AP75 | mask AP50:95 | box AP50 | dets |
|---|---|---|---|---|---|
| **Box2Mask Swin-L** | **0.5396** | 0.0884 | 0.1980 | 0.5821 | 138,820 |
| Box2Mask R-50 | 0.3865 | 0.0509 | 0.1299 | 0.4707 | 117,027 |

**+0.153 mask AP50 over R-50** — a far larger backbone effect than the COCO margin implied. An
earlier extrapolation in this plan and in the paper footnote put Swin-L near 0.46 "still below
every other entry"; that was WRONG and has been removed. Swin-L also inverts the R-50 finding that
Box2Mask's boxes were the weak part — though against the CONVERGED DetecTree2 (0.5913) its box
AP50 of 0.5821 no longer leads, it merely draws close.

Read it with two cautions:
- **AP50 up, strict IoU still poor.** AP75 0.0884 against the converged DetecTree2's 0.1748 and
  AP50:95 0.1980 against 0.2578. It finds crowns well and delineates them coarsely — the same
  mask weakness the strata analysis found for R-50, not cured by a better backbone.
- **RESOLVED 2026-09-03: DetecTree2 retrained and Box2Mask Swin-L does NOT beat it.** The
  provisional caution recorded here was correct. Converged DetecTree2 scores 0.5969 / 0.1748 /
  0.2578 / 0.5913 against Swin-L's 0.5396 / 0.0884 / 0.1980 / 0.5821 — DetecTree2 leads on every
  one, including the box AP50 on which Swin-L had appeared to lead. Nothing claiming otherwise
  may go in the paper.

Diagnostics: `tiles_over_maxdets_600` = 0 (mean 316.2 dets/tile, max 584), canopy landing 52.1%.
Predict 25.8 min, stitch at the protocol floor 0.05.

**The `num_queries=100` ceiling, measured properly.** `predict` reports 28.8% of subtiles "at
cap" for Swin-L against 2.7% for R-50, but that counter means *all 100 output slots scored above
the 0.05 floor* — a statement about the score distribution, NOT about lost crowns. What matters is
how many subtiles genuinely hold more than 100 crowns: **46 of 3,951 test subtiles (1.16%)**, and
1,160 of 61,098 subtile-instances (1.90%) sit beyond the cap. Train side is the same order (81 of
6,692 subtiles, 2.2% of boxes). The cap sits ~6x above typical demand (median 5 crowns/subtile,
mean 15.5) and the 50%-overlap grid partly recovers the rest. It is a footnote, not a confound.
Raising it would mean randomly initialising new query embeddings and discarding the COCO init, so
it stays; a `num_queries=200` sensitivity would cost ~$14 and is not worth it at 1.16%.

## 6. Why these are system rows, not module rows

BoxInst/Box2Mask cannot be prompted with a box. Their mask heads take
`(mask_feat, det_params, det_coors, det_level_inds)` — a controller vector read at a feature
LOCATION. There is no box argument anywhere in the path.

`inject_boxes.py` exists and implements box-conditioned inference using their OWN training-time
assigner (`get_targets` picks the FPN level from box size and the location from box extent, using
no model prediction). It is **written and half-validated**: the assigner places **99.485%** of val
GT boxes on a location (85 of 16,505 fail — all tiny, median side 8.0 px = exactly the finest
stride; 34.6% of sub-8px crowns fail. LACE avoids this by padding a half cell, which is why it
covers 100%).

**But the box only weakly conditions the mask**: it picks the location, and nothing more. The
mask's extent comes from the network's belief at that point, so two different boxes with the
same centre and size band give the IDENTICAL mask. LACE and SAM receive the whole box; this
receives "an object about this big, near here". Cropping to the box afterwards would fake
box-conditioning and is NOT done.

So an injected row would be **location-prompted, not box-prompted**, and a poor result could not
be separated from "it was given less information". That is why the module table stays LACE vs
SAM. `inject_boxes.py` is kept because it is written, validated, and cheap to revisit — but it is
not on the critical path. If ever used, gate it on the round trip (inject the model's own boxes,
require median IoU >= 0.95 against its native masks) and report unassignable boxes as IoU 0 with
coverage stated, never silently dropped.

## 7. Method selection

| method | venue | COCO mask AP | COCO ckpt | verdict |
|---|---|---|---|---|
| **Box2Mask** | TPAMI 2024 | **42.5** (Swin-L) | R-50/R-101/Swin-L | **run first** |
| BoxInst | CVPR 2021 | 33.2 | R-50/R-101 | run second — seminal reference |
| DiscoBox | ICCV 2021 | 37.9 paper / **33.4 repro** | R-50/R-101 | skip — reproduction gap, adds no argument |
| BoxLevelSet | ECCV 2022 | — | **none** | skip — no weights, would starve on 792 tiles |
| MAL | CVPR 2023 | **44.1** | separate Docker | stretch; the only one that could enter the MODULE table |
| BoxSeg | 2025 | ~35 (R-101) | — | newest but weaker |

No published BoxInst/Box2Mask for tree crowns exists — searched. The nearest is OBBInst (BoxInst
for oriented boxes, general remote sensing). **That absence is a claim the paper can make**: the
box-supervised literature has never been applied to tree crowns.

## 8. Commands

Every function takes `--method {box2mask,boxinst}`; `box2mask` is the default.

```bash
cd boxinst_commonality_tcd_04/box_supervised_baselines
M=../../.venv/bin/modal

$M run          boxinstseg_modal.py::imports_ok                      # toolchain gate, ~2 min
$M run          boxinstseg_modal.py::dryrun     --method box2mask    # config+data+model, no GPU-h
$M run --detach boxinstseg_modal.py::fetch_ckpt --method box2mask    # 529 MB from Drive, ~15 min
$M run --detach boxinstseg_modal.py::train      --method box2mask --epochs 3 --max-hours 2.9
$M run --detach boxinstseg_modal.py::predict    --method box2mask    # 3,951 subtiles, ~20 min
$M run          boxinstseg_modal.py::stitch     --method box2mask    # CPU; min_score=0.05

# a timing probe before committing to a schedule: runs N iterations of the REAL config,
# no validation, and reports projected hours and dollars for the full run.
$M run          boxinstseg_modal.py::train --method box2mask --max-iters 400

# pull and score locally, with the frozen protocol scorer
$M volume get tcd04-baselines-vol boxinst/preds_box2mask_s0_test.json /tmp/
../../.venv/bin/python -m boxinst_commonality_tcd_04.score_coco \
    --preds /tmp/preds_box2mask_s0_test.json --res 512 \
    --out ../results_439/box2mask_s0_coco512.json
../../.venv/bin/python -m boxinst_commonality_tcd_04.results_439.build_table
```

**`--detach` for anything over a few minutes.** A local network blip stops a non-detached app
server-side mid-run; it has already cost a probe and a checkpoint download. Every long function
also writes its result to `/vol/boxinst/reports/<name>.json`, because a detached run cannot hand
a return value back to the client — pull those with `modal volume get`.

**Never pipe a long Modal run through `tail`/`grep`** — it buffers until exit and hides all
progress. Redirect to a file and read that.

## 9. Verification gates

- `imports_ok` — versions, an mmcv CUDA op, both compiled ops, all four detectors registered.
- `dryrun` — asserts train has **0** ignore boxes (canopy gone), val has **>0** (canopy retained
  as `iscrowd`), classes `('tree',)`. Builds the model. Catches path/class errors for pennies.
  It also pulls one real sample THROUGH the pipeline and asserts every `gt_masks` entry is
  exactly its own bounding rectangle (249/249 for Box2Mask), and reports the `num_queries`
  ceiling against the true per-subtile crown counts.
- `make_box_only_coco.verify()` — every polygon IS its own bbox, so a config that silently fell
  back to mask supervision cannot pass. `dryrun`'s tensor check is the second half of this: the
  FILE being box-only and the TENSOR being box-only are different claims.
- `train` — before any GPU time, asserts the COCO checkpoint's key set matches the model's with
  no shape mismatches and nothing missing except the classifier we dropped (610/612 loaded).
  `runner.load_checkpoint` is `strict=False`, so without this a near-total mismatch would train
  from scratch in silence.
- `predict` — counts subtiles that saturate the 100-query ceiling above the score floor.
- `stitch` — reports dets/tile and canopy-landing rate. **Watch this**: the model trains on
  blacked canopy but is TESTED on real imagery, so like SelvaBox it will fire freely in canopy,
  and `maxDets=600` is applied BEFORE canopy-ignore. PROTOCOL_439.md measured this for SelvaBox
  (cost 0.008 AP50). If most detections land in canopy, report a maxDets sensitivity.
- Sanity: the stitched 439-tile score should land plausibly beside DetecTree2's 0.5969. A wildly
  off number means the harness, not the method.

## 10. Asymmetries — state, do not hide

1. **BoxInst-family gets ~2.1x LACE's pixel exposure per epoch** and ~2.4x the crown instances:
   both train on the same 792 tiles, but LACE consumes whole 2048² tiles (3.32 G px/epoch,
   54,191 crowns) while these use the 1024@50%-overlap grid (7.02 G px/epoch, 127,692 crown
   annotations). KEEP IT — it favours the baseline, and the grid is the one DetecTree2 and
   SelvaBox already use. If a baseline WINS, re-run on a non-overlapping 2×2 grid (exactly 1x
   exposure) as a control, ~$8.
2. **Blackout breaks "identical PNG crops" for the training split only.** Unavoidable — the
   alternative is canopy as a trained negative.
3. **Frozen DINOv3 ViT-L vs fine-tuned R-50/Swin-L.** Not symmetric in any configuration; LACE
   trains ~4M head params, these train 34M+.
4. **"Epochs" are not commensurable across methods.** Each runs its own published recipe.
5. **Pretraining differs across every row** and already does in Table 1 — LACE's DINOv3 saw 1.7B
   unlabelled images. The "Imgs = 900" column means OAM-TCD images seen, not total pretraining,
   and needs a footnote for any fine-tuned row (DetecTree2's included).
6. **Single seed**, as for Restor/DetecTree2. Compare against LACE's WORST seed.
7. **The Box2Mask row is a 9-epoch fine-tune of their released COCO model, not their 50-epoch
   from-scratch recipe** (§5). Report the val AP50 curve with it: the claim to defend is that
   the curve had flattened under the same early-stopping rule LACE uses, NOT that the epoch
   counts match. If the curve is still climbing at the end, say so and call the row a lower
   bound — exactly as is true of the DetecTree2 row already in the table.
8. **We ran Box2Mask R-50, its WEAKEST released backbone, and the weakest representation in
   Table 1.** Their zoo is R-50 35.9 / R-101 38.2 / Swin-L 42.5 COCO mask AP, and §7 justified
   choosing Box2Mask over BoxInst by the Swin-L number — so reporting R-50 is an internal
   inconsistency unless stated. Every other row has a stronger backbone: Restor R-50 (equal),
   DetecTree2 R-101, SelvaBox DINO-Swin-L, LACE frozen DINOv3 ViT-L/16. The asymmetry favours
   us. It reverses on TRAINABLE capacity (Box2Mask R-50 trains 44M against LACE's ~4M head), so
   the axis that matters is pretrained representation quality, not parameter count. Naively
   transferring their COCO R-50→Swin-L margin gives ~0.46 AP50, still last, but that is an
   extrapolation. **If budget allows, Swin-L is the single most valuable follow-up in this
   phase** — estimated ~$25–30 (A100-80GB, ~$4/epoch, ~6 epochs to reach a peak, plus a slower
   predict pass and a ~1.3 GB Drive checkpoint). R-101 is the cheap middle at ~$12.
9. **Box2Mask cannot emit more than 100 instances per 1024² subtile** (§5). Restor's analogous
   cap was raised to the framework default because it was a config knob; this one is a learned
   weight shape and raising it would forfeit the COCO initialisation, so it stays and is
   reported instead.

## 11. Files

| file | role |
|---|---|
| `boxinstseg_modal.py` | the app: image, `imports_ok`, `prep_data*`, `fetch_ckpt`, `dryrun`, `train`, `predict`, `stitch`, `injection_gate`, `assigner_coverage`. All take `--method` |
| `patch_boxinstseg.py` | strips vestigial THC includes so `tree_filter` builds on torch >= 1.11 |
| `make_box_only_coco.py` | box-only COCO + `verify()` guard |
| `inject_boxes.py` | assigner-driven box→mask + round-trip validator (NOT on the critical path, §6) |
| `box_iou_strata.py` | mask IoU at matched box quality — needs no prompting, so every method can join |
| `gtbox_{lib,modal}.py`, `combine_gtbox.py` | step 2, DONE — the GT-box bake-off |
| `PLAN.superseded.md` | the pre-pivot plan, kept for provenance |

Reused, do not duplicate: `../detectree2_baseline/{build_coco,stitch}.py`, `../score_coco.py`,
`../test_gt.json`.

## 12. Step 2 result (done, in the paper)

439 tiles · 25,692 crowns · identical GT prompt boxes · `results_439/gtbox_bakeoff_r2048.json`.
Scored at native 2048; 512 in brackets.

| arm | mean IoU | >=0.75 | >=0.9 | double-assign | paired vs LACE |
|---|---|---|---|---|---|
| **LACE commonality-EM** | **0.7698** (0.7856) | 0.6588 (0.7096) | 0.0513 (0.0891) | 0.0302 | — |
| SAM 3 zero-shot | 0.7406 (0.7493) | 0.5452 (0.5649) | 0.0368 (0.0373) | 0.0082 | wins 40.1% (36.3%) |
| SAM 3 SelvaMask-FT | 0.6232 (0.6048) | 0.2741 (0.2343) | 0.0107 (0.0076) | 0.0023 | wins 19.9% (14.5%) |

Written up as `tab:gtbox` in `../ablation/results/paper.tex` (2048 headline, 512 as raster
sensitivity). Carried caveats: **never quote the >=0.9 advantage as 2.4x** — that was a
512-raster artefact and is 1.4x at 2048 (`ablation/results/raster_quantum.json`);
double-assignment 0.0302 vs a GT self-overlap floor of **0.0015** is ~20x and is the real
weakness; no seed band is possible (one masker npz is shared by all three Table-1 seeds, which
differ only in the detector).
