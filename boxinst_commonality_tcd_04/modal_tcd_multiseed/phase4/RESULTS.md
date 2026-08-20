# TCD 439 — best results (condensed)

Weakly-supervised individual-tree instance segmentation on the OAM-TCD 439-tile holdout.
Condensed from `README.md` (which keeps the full history, ablations, bug forensics and ops notes).

## Headline

**4-phase real-8px L24 detector (frozen DINOv3-web) + β=0.5 self-mask EM masker on the
geometry-fixed grid (`cell_origin=s`), `mask_thr=0.25`, box-robustness knobs α=0.3 / κ×1.6.**
Single-scale. Supervision = **ITC boxes + canopy-ignore only** — training never sees a polygon.

| | 3-seed (0,1,2) |
|---|---|
| **mask mAP50** | **0.630 ± 0.005** |
| mask mAP50-95 | 0.257 ± 0.001 |
| box mAP50 | 0.603 |
| box mAP40 / best-F1@IoU0.4 | 0.669 / 0.682 |

## Benchmark table — apples-to-apples on the identical cohort + metric

Same 792 train / 108 val / 439 test tiles, same GT (`test_gt.json`), same COCO-101pt mask-AP
with the >50%-canopy-ignore scorer (`evaluate._greedy_ap`, RES 512).

| method | supervision | ign AP50 | ign AP50-95 | no-ign AP50 | no-ign AP50-95 | box AP50 |
|---|---|---|---|---|---|---|
| **OURS** (β=0.5-fix + α/κ knobs, 3-seed) | **boxes + canopy-ignore (no mask labels)** | **0.630 ± 0.005** | **0.257 ± 0.001** | **0.485 ± 0.002** | **0.202 ± 0.002** | 0.603 |
| **DetecTree2** (Mask R-CNN R101-FPN, fine-tuned) | full crown masks | 0.535 | 0.224 | 0.421 | 0.180 | 0.529 |
| Restor Mask R-CNN (OAM-TCD paper) | full crown masks | | | 0.432 | | |

**Our box-weak method beats full-mask-supervised DetecTree2 by +0.096 mask AP50 (+0.033 AP50-95)**
despite DetecTree2 getting strictly more supervision, and the win holds without canopy-ignore too
(+0.064 / +0.021). The gap is mostly the **detector** (box 0.603 vs 0.529); both convert boxes→masks
comparably. DetecTree2 is a single seed.

Restor's 0.432 is **no-canopy-ignore AP50**, so it lines up against the no-ign column: **we beat it
(0.485 vs 0.432)**, while our DetecTree2 reproduction lands just below it (0.421) — a faithful
repro landing beside the dataset authors' own Mask R-CNN, which is the sanity check that matters.
The full 10-model 439 table, including SelvaMask/CanopyRS, is below.

> ⚠️ **The Restor 0.432 is the OAM-TCD paper's published figure — we have never re-scored their
> released `restor/tcd-mask-rcnn-r50` checkpoint ourselves**, and no such replication exists in this
> repo (the checkpoint is in the local HF cache but no code reads it). It is a **no-canopy-ignore
> AP50**, so it is column-matched against our 0.485 and DetecTree2's 0.421, but the rest of its
> protocol (tiling, AP aggregation) is unverified. Treat the alignment as indicative, not exact.

> 📌 **2026-08-18:** the DetecTree2 439 figures were restated (0.5448 → 0.5345 mask mAP50) after a
> prediction-coverage bug was fixed — its test subtiles with no annotations were never inferred on,
> so it was never charged for false positives there (80.7% coverage). Ours are unchanged. Full
> account in `README.md` § DetecTree2 baseline.

## Protocol sensitivity — canopy-ignore (2026-08-19)

**Our published numbers use the canopy-ignore rule; several external baselines do not.** A
prediction sitting >50% inside an OAM-TCD `canopy` region is IGNORED by our scorer rather than
counted a false positive — canopy is real tree cover whose individual crowns were never delineated,
so a detection there is valid-but-unlabelled. **41–42.5% of our predictions fall in canopy**, so this
is not a rounding-level convention.

Measured by scoring the identical saved knobbed predictions twice (`no_ignore_sensitivity.py`;
the with-ignore pass is gated to reproduce each recorded per-seed result exactly, and did):

| seed | mask mAP50 (ignore → none) | mask mAP50-95 (ignore → none) | % preds in canopy |
|---|---|---|---|
| 0 | 0.6250 → 0.4867 | 0.2574 → 0.2041 | 42.5 |
| 1 | 0.6358 → 0.4838 | 0.2567 → 0.1995 | 42.5 |
| 2 | 0.6294 → 0.4846 | 0.2564 → 0.2013 | 41.0 |
| **3-seed** | **0.630 ± 0.005 → 0.485 ± 0.002** | **0.257 ± 0.001 → 0.202 ± 0.002** | |

**Dropping the rule costs 0.055 mAP50-95 and 0.145 mAP50.** (The vanilla α=1/κ=1 seed-0 arm behaves
the same: 0.2436 → 0.1931.)

**DetecTree2 under both protocols (same cohort, same scorer, full-coverage preds):**

| method | mask mAP50 (ignore → none) | mask mAP50-95 (ignore → none) | % preds in canopy |
|---|---|---|---|
| **OURS** (knobbed 3-seed) | 0.630 → 0.485 | 0.257 → 0.202 | 42.5 |
| **DetecTree2** (s0, fullcov) | 0.535 → 0.421 | 0.224 → 0.180 | 39.7 |
| **our margin** | **+0.096 → +0.064** | **+0.033 → +0.021** | |

**We win under BOTH protocols — but the rule is not ranking-neutral.** We lose more by dropping it
than DetecTree2 does (−0.055 vs −0.043 mAP50-95), so **roughly a third of the headline margin is
protocol**: +0.096 → +0.064 mAP50. Mechanism: we emit more predictions (159,687 vs 152,690, topk 600
@ score 0.05) and a larger share land in canopy (42.5% vs 39.7%), so our low-confidence tail converts
to false positives under no-ignore. **Report both columns** — the win survives the stricter protocol,
and disclosing the narrowing pre-empts the obvious objection.

### SelvaMask / CanopyRS — corrected reading (2026-08-19)

**Correction to an earlier note in this file.** It previously said SelvaMask applies "no canopy-ignore
rule", making our **no-ignore 0.202** the matched column. That was wrong. They do not score canopy
predictions as false positives — **they delete the canopy category and black out those pixels**, so a
canopy false positive is impossible. That is the same *effect* as our ignore rule, reached by a
different mechanism, and **the matched column is therefore our with-ignore 0.257, not 0.202.**

Verified from source and from the released artifact, not from summaries:
`raw_datasets/OAM_TCD/oam_tcd.py` — `remaining_annotations = [ann for ann in annotations if
ann['category_id'] != category_id]`, then `img_array[mask == 1] = 0`, with `remove_empty_tiles=True`
hardcoded at all five call sites and a `black/total > 0.8` tile drop in `_cut_and_filter_coco`. Rows
pulled from `CanopyRS/OAM-TCD` confirm the path was taken: `categories = ['tree']` only, `iscrowd = 0`
only, tile names `..._tile_test_sf1p0_0_1024.tif` (native 0.1 m/px, 1024 px @ 0.5 overlap).

**Their 18.5 is legitimate — no methodological problem found.** The 439 is a clean holdout for them:
`oam_tcd.py` splits on OAM-TCD's *official* `validation_fold` (folds 0–3 train, 4 valid, holdout −1
test) and `dino_swinL_multi_NQOS.yaml` sets `test_dataset_names: []`. All 439 of our test tiles are
`validation_fold: None`; our 900 span folds 0–4. And their score thresholds are not a self-handicap:
`base_benchmarker.py::_benchmark` computes tile-level mAP **before** the aggregator, unthresholded —
only raster-level RF1 uses the aggregated output.

**Their test set is nonetheless easier than ours**, in three nameable ways: crown-to-crown boundaries
inside canopy (the hard case) are removed with the pixels; tiles left with no annotations are deleted;
tiles >80% black are dropped. **Our 439 retains 73 tiles with zero GT crowns**, which can only cost us.

**Their OAM-TCD baselines are zero-shot; ours are fine-tuned.** Their Table 4 marks Detectree2 (flexi)
0.123, Detectree2 (resize) 0.023 and DeepForest→SAM3 0.065 as **OOD ✓**, against their own
**OOD ✗** (in-distribution) 0.185 — a trained model against untrained baselines. Disclosed via their
OOD column, and a choice rather than a limitation: they *did* fine-tune baselines on their own
SelvaMask benchmark. **We have the fine-tuned DetecTree2 they did not run.**

**No re-run was performed.** Putting their released weights through our scorer would close the protocol
gap but not the training-data gap (~3,335 in-domain OAM-TCD images vs our 900), so it still would not
be supervision-matched — and our end-to-end-controlled DetecTree2 comparison is already stronger than
anything in their OAM-TCD table.

### Master table — OAM-TCD 439 holdout, all models

Mask AP, decimals. **ign** = canopy neutralised (our canopy-ignore rule, or SelvaMask's equivalent
deletion of canopy); **no-ign** = canopy predictions counted as false positives. Every number sits in
the column matching its own protocol, so columns line up. "OAM-TCD imgs" = in-domain training images
the model saw. Blank = never measured under that protocol.

| # | Method | Supervision | OAM-TCD imgs | ign AP50 | ign AP50-95 | no-ign AP50 | no-ign AP50-95 | box AP50 |
|---|---|---|---|---|---|---|---|---|
| 1 | **OURS — LACE** (β=0.5-fix, α=0.3 κ×1.6) | boxes + canopy-ignore (no mask labels) | 900 | **0.630 ± 0.005** | **0.257 ± 0.001** | **0.485 ± 0.002** | **0.202 ± 0.002** | **0.603** |
| 2 | OURS boxes → SAM 3 (box-prompt, frozen) | boxes (ours) + SAM pretrain (frozen) | 900 (detector) | 0.633 | 0.268 | | | 0.605 † |
| 3 | Restor Mask R-CNN (OAM-TCD paper) \*\* | full crown masks | 4,169 | | | 0.432 | | |
| 4 | **DetecTree2** R101-FPN, fine-tuned by us | **full crown masks** | 900 | 0.535 | 0.224 | 0.421 | 0.180 | 0.529 |
| 5 | **SelvaBox → SAM 3**, both FT (SelvaMask) \* | boxes + **SelvaMask masks** + SAM pretrain | ~3,335 | | **0.185** | | | |
| 6 | SelvaBox → SAM 3, frozen SAM \* | boxes + SAM pretrain (frozen) | ~3,335 | | 0.122 | | | |

† box AP is masker-invariant — same detector as rows 1–2.
**Seeds:** row 1 is a 3-seed band (0,1,2), mean ± **sample** std (ddof=1, the repo convention —
numpy's default ddof=0 would print 0.004/0.000 from the same per-seed values); rows 2 and 4 are single-seed (seed 0);
rows 3, 5 and 6 are as published, seed count not stated (row 5 reports ±1.0 over an unstated number).
Our 5-seed vanilla ablation (α=1, κ=1) is kept in `README.md` § 5-seed VANILLA band, not here.

**Rows 1, 2 and 4 share our protocol** — the same 439 whole 2048 px tiles, the same GT (`test_gt.json`),
pooled `evaluate._greedy_ap` (COCO-101pt), masks at 512, topk 600 @ score 0.05 — and are mutually
comparable.

**\* SelvaMask protocol (rows 5–6).** Same 439 source images, **different test set**: the canopy
category is deleted and its pixels blacked out (`img_array[mask == 1] = 0` — equivalent in effect to
our ignore rule, hence the `ign` column), cut to 1024 px @ 0.5 overlap, tiles left empty or >80% black
dropped (**2,527 of 3,951 subtiles survive**), AP averaged **per tile** rather than pooled, maxDets
400. Our 439 by contrast retains **73 tiles with zero GT crowns**. Rows 5–6 are therefore *not*
comparable to rows 1, 2 and 4 — their test set is materially easier. (SelvaMask's zero-shot rows — Detectree2
flexi 0.123, Detectree2 resize 0.023, DeepForest→SAM3 0.065 — are omitted: no OAM-TCD training data,
so they measure generalisation, not this benchmark.)

**\*\* Restor protocol (row 3).** The OAM-TCD paper's own Mask R-CNN, **no canopy-ignore**, mask AP50.

**Reading the supervision column.** *No method here is box-only.* The detector side is box-supervised
throughout — SelvaBox/DINO is a detection architecture and consumes `bbox`, so OAM-TCD's mask
annotations reach it only as box extents (`geodataset` `DetectionLabeledRasterCocoDataset` reads
`bbox`). The mask side differs sharply:
- **row 1 (ours)** — no mask labels at any stage. Canopy enters as an *ignore* region in the detector's
  heatmap loss (`train_detector_tiles.py:65` → `detector.py:66` `focal_heatmap_loss(..., ignore)`),
  which is weaker than mask supervision but more than boxes alone.
- **rows 2 and 6** — frozen SAM, so the mask prior is SA-1B pretraining, never tree-specific labels.
- **row 5** — SAM 3 fine-tuned end-to-end on **SelvaMask crown masks** (`sam3_multi_selvamask_FT.yaml`
  → SelvaMask rasters); its detector is additionally FT on SelvaMask boxes. No OAM-TCD masks are used
  for the segmenter in either row 5 or 6.
- **rows 3 and 4** — full OAM-TCD crown masks, the strongest supervision in the table.

**Per-seed detail for row 1** (canopy share is the fraction of predictions >50% inside canopy):

| seed | ign AP50 | ign AP50-95 | no-ign AP50 | no-ign AP50-95 | n_pred | canopy % |
|---|---|---|---|---|---|---|
| 0 | 0.6250 | 0.2574 | 0.4867 | 0.2041 | 159,687 | 42.5 |
| 1 | 0.6358 | 0.2567 | 0.4838 | 0.1995 | 162,017 | 42.5 |
| 2 | 0.6294 | 0.2564 | 0.4846 | 0.2013 | 177,598 | 41.0 |

**The comparison that carries weight** is rows 1 vs 4 — both trained on the identical 900 tiles, scored
identically, reported under both protocols:

| | ign AP50 | ign AP50-95 | no-ign AP50 | no-ign AP50-95 |
|---|---|---|---|---|
| OURS (row 1) | 0.630 | 0.257 | 0.485 | 0.202 |
| DetecTree2 (row 4) | 0.535 | 0.224 | 0.421 | 0.180 |
| **margin** | **+0.096** | **+0.033** | **+0.064** | **+0.021** |

We win under both protocols. The margin narrows by ~⅓ without canopy-ignore, because we emit more
predictions (159,687 vs 152,690) and a larger share fall in canopy (42.5% vs 39.7%) — so our
low-confidence tail converts to false positives when the rule is dropped. Report both columns.

## Why it works — the two load-bearing pieces

**1. Grid registration fix (2026-07-27).** The interleaved 8px `asm` grid puts cell X's feature at
pixel `8X+8`, but the masker computed centers as `8X+4` (`mgrid*s + s/2`). That −4px up-left shift
made every mask fill the top-left and carve the bottom-right (worst on small crowns), and made
β=0 (box-fill, robust to the shift) spuriously beat β=0.5 (carve, which compounds it). Fixed via a
per-model `cell_origin` (+ a `+1px@512` raster shift, verified optimal by a render-offset sweep).
Post-fix the carve wins by **+0.041 mAP50 / +0.044 mAP50-95** over β=0.

**2. Box-robustness knobs α/κ.** Two free inference-time scalars in the E-step — no re-fit, no
re-extraction. α = `prior_weight` = 0.3 relaxes trust in the imprecise *predicted* box so the
box-independent appearance term drives; κ = `kappa_scale` = 1.6 sharpens the vMF concentration for
tighter boundaries. **+0.007 mAP50 / +0.016 mAP50-95 on every seed** (paired), so robust rather
than within-noise. Motivation: our EM beats SAM on *GT* boxes (crown IoU 0.747 vs 0.724), so the
masker was box-imprecision sensitive, not weaker. Full lever map in `BOX2MASK_LEVERS.md`.

## Per-seed detail (knobbed, α=0.3 κ×1.6, mask_thr 0.25)

| seed | mask mAP50 | mask mAP50-95 | box mAP50 |
|---|---|---|---|
| 0 | 0.6250 | 0.2574 | 0.6050 |
| 1 | 0.6358 | 0.2567 | 0.5981 |
| 2 | 0.6294 | 0.2564 | 0.6066 |
| **mean ± std** | **0.630 ± 0.005** | **0.257 ± 0.001** | **0.603** |

Seeds 3,4 are trained but not yet re-evaluated with the knobs. The 5-seed *vanilla* (α=1, κ=1)
band is 0.615 ± 0.011 — see `README.md` for it and the full β ablation.

## Reproduce

From `modal_tcd_multiseed/phase4/` (knobs are the eval default):

```bash
modal run --detach phase4_modal.py::eval_selfmask --beta 0.5 --fix          # seed 0, best config
modal run --detach phase4_modal.py::band_selfmask --seeds 0,1,2 --beta 0.5 --fix   # the 3-seed band
```

Artifacts on Modal Volume `tcd04-phase4-vol` (`modal volume get tcd04-phase4-vol <path> <dest>`):

| artifact | path |
|---|---|
| detector ckpt (seed 0) | `out/det_phase4_L24_s0.pt` |
| masker β=0.5 FIXED (the one) | `out/em_model_4p_fix.npz` |
| predictions (439, boxes + RLE masks) | `out/preds_selfmask_fix_thr025_phase4_L24_s0/preds.json` |
| metrics | `out/results_selfmask_fix_thr025_phase4_L24_s0.json` |

DetecTree2 baseline: code in `../../detectree2_baseline/`, metrics in `results_dt2_s0.json`,
artifacts on Volume `tcd-detectree2-vol`.

## Ceiling

Oracle-perfect masks on the *same* boxes score 0.737 AP50 (maxR 0.818), and box→mask caps at ~0.633
on these detections. **Reaching mask mAP50 0.70 needs better detection (small-crown recall), not a
better masker.**
