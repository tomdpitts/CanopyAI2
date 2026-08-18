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

| method | supervision | mask mAP50 | mask mAP50-95 | box mAP50 |
|---|---|---|---|---|
| **OURS** (β=0.5-fix + α/κ knobs, 3-seed) | **box + canopy (weak)** | **0.630 ± 0.005** | **0.257** | 0.603 |
| **DetecTree2** (Mask R-CNN R101-FPN, fine-tuned) | full crown masks | 0.535 | 0.224 | 0.529 |
| Restor Mask R-CNN (OAM-TCD paper, *their* protocol) | full masks | 0.432 | — | — |

**Our box-weak method beats full-mask-supervised DetecTree2 by +0.085 mask mAP50**, and on
mAP50-95 and box mAP50 too, despite DetecTree2 getting strictly more supervision. The gap is
mostly the **detector** (box 0.605 vs 0.539); both convert boxes→masks comparably. The DetecTree2
reproduction is credible rather than a strawman: 0.535 lands above the paper's Restor number and
below ours. DetecTree2 is a single seed.

> ⚠️ **The Restor 0.432 row is the paper's published figure under Restor's own protocol — we have
> never re-scored their released `restor/tcd-mask-rcnn-r50` checkpoint under our canopy-`iscrowd=1`
> scorer.** No such replication exists anywhere in this repo (the checkpoint is in the local HF
> cache but no code reads it). The only Mask R-CNN we have run under our exact protocol is the
> DetecTree2 row (0.535). Treat 0.432 as cross-protocol context, not a like-for-like comparison.

> 📌 **2026-08-18:** the DetecTree2 439 figures were restated (0.5448 → 0.5345 mask mAP50) after a
> prediction-coverage bug was fixed — its test subtiles with no annotations were never inferred on,
> so it was never charged for false positives there (80.7% coverage). Ours are unchanged. Full
> account in `README.md` § DetecTree2 baseline.

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
