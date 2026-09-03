# 4-phase real-8px L24 — TCD instance seg (Modal A100)

> ## ⭐ HEADLINE (2026-08-28) — every baseline re-run by us, one frozen scorer
> **LACE + posterior-product confidence, 3 seeds: OAM-TCD 439 canopy-neutral mask AP50 =
> 0.6625 ± 0.0012; held-out sparse 236 = 0.6913 ± 0.0095 (converged DetecTree2 0.6118).**
> Every row below was produced and scored by us under one protocol (`score_coco.py`,
> pycocotools COCOeval, 439 whole 2048² tiles, masks 512², maxDets 600, floor 0.05, canopy as
> `iscrowd`). **No number is quoted from another paper.** Full spec + table:
> **`../../PROTOCOL_439.md`**; artifacts in **`../../results_439/`**.
>
> | Method | Superv. | Imgs | CN AP50 | CN AP75 | CN AP50:95 | Box AP50 |
> |---|---|---|---|---|---|---|
> | **LACE + posterior product, 3-seed** | box | 900 | **0.6625 ±0.0012** | 0.1649 | **0.2790** | 0.6396 |
> | LACE 3-seed baseline | box | 900 | 0.6300 ±0.0054 | 0.1409 | 0.2573 | 0.6092 |
> | **Restor Mask R-CNN @ rpn 1000** | masks | 4169 | 0.6255 | **0.1872** | 0.2766 | 0.6536 |
> | SelvaBox → SAM 3 (both FT) | box+SAM | 3024 | 0.5687 | 0.0746 | 0.2028 | **0.7297** |
> | DetecTree2 (fine-tuned, converged) | masks | 900 | 0.5969 | **0.1748** | 0.2578 | 0.5913 |
>
> **Four corrections to the previous headline, all against us:**
> 1. **Restor's published 0.432 was an under-configured model.** Their shipped config caps the
>    RPN at `topk 512` — half detectron2's FPN default — starving it to 98 dets/tile and
>    recall 0.6442. Re-run at the framework default (1000) the released checkpoint scores
>    **0.6255**, not 0.432. "Beats Restor by a wide margin" no longer holds.
> 2. **The un-reranked LACE baseline TIES Restor** — 0.6250 (s0) vs 0.6255. The entire margin
>    in the headline comes from the confidence fix, not from the detector or masker.
> 3. **Restor still wins strict-IoU mask quality** — CN AP75 0.1872 vs our 0.1718. Pixel-level
>    mask supervision buys boundary precision that reranking cannot manufacture.
> 4. **DetecTree2 was hamstrung and is now retrained (2026-09-03): 0.5344 → 0.5969.** The old
>    row was stopped by a budget cap at 2.39 epochs while still improving, selected on a
>    strided 256/896 val subset, evaluated at an inherited COCO input scale of 800 that
>    downscaled our 1024 subtiles, and stitched at 0.1 against a protocol floor of 0.05 — four
>    defects, every one understating it. Retrained to detectree2's own AP50 early-stopping rule
>    it stops itself at 6.0 epochs and **beats us at CN AP75 (0.1748 vs 0.1649)** and on
>    isolated mask quality (0.7595 vs 0.7430). Our lead on CN AP50 falls from +0.128 to
>    **+0.066**. Account: `../../detectree2_baseline/RETRAIN_V2.md`.
>
> **What stands:** beating DetecTree2 (0.5969) on the identical split/test/metric, from boxes
> only, by +0.066 CN AP50 — 55× the seed sd, though no longer the 0.128 previously claimed; and
> matching/beating a model trained on 4.6× the images with full crown polygons. **What no longer
> stands:** any claim that LACE's masker is competitive with mask supervision at strict IoU —
> both Restor and DetecTree2 now beat it at AP75 and on mask IoU.
>
> **The confidence fix is banded and generalises** (2026-08-28): +0.0325 CN AP50 on all three
> seeds of the 439, **+0.0354 on the held-out sparse 236** where no tile is shared with the 900
> training images — the gain is *larger* off-distribution, not smaller. It also cuts seed
> variance 4.5× (sd 0.0054 → 0.0012). Sparse table: `../../results_439/sparse236/`.
>
> ⚠️ **Caveats carried, not hidden:** ~two thirds of the confidence gain is a box-size prior,
> so the masker-specific increment is ≈ +0.012 (`../../confidence/README.md`). SelvaBox's
> canopy=FP figures are not comparable (it trains with canopy deleted; 83% of its detections
> land in canopy here). Restor keeps strict-IoU mask quality (CN AP75 0.1872 vs our 0.1649).

> ## 🔬 BOX→MASK MODULE, ISOLATED (2026-09-01) — read before citing any masker number
> **Given the SAME ground-truth boxes, our masker beats SAM 3 at BOTH rasters: mean per-crown
> IoU 0.7698 vs 0.7406 scored at native 2048 (0.7856 vs 0.7493 at 512), winning on 59.9% of
> 25,692 crowns individually (63.7% at 512).** 439 tiles, no detector, no ranking,
> no score floor — prompt box *i* is GT crown *i*, so the mask is scored against that crown and
> nothing else. Fine-tuned SAM 3 is far worse (0.6232). Full spec, caveats and costs:
> **`../../box_supervised_baselines/PLAN.md`**; artifacts in
> **`../../results_439/gtbox_bakeoff.json`** (+ `gtbox_{lace,sam3_base,sam3_ft}.json`, which keep
> the per-crown IoU arrays so size/stratum re-analysis needs no re-run). Code:
> `../../box_supervised_baselines/gtbox_{lib,modal}.py`, `combine_gtbox.py`.
>
> ⚠️ **`masker_lab/sam_bakeoff_results.json` (EM 0.7473 vs SAM 0.7242) IS SUPERSEDED — do not
> cite it.** It used SAM **1** ViT-H against the **16 px** β=0 EM (not the shipped 8 px masker),
> 60 of 439 tiles, with SAM internally resized 2048→1024 in our favour. Both corrections went
> against us and the margin still widened (+0.023 → +0.036).
>
> ⚠️ **NEVER quote the ≥0.9 advantage as 2.4×** — that was a 512-raster artefact and is 1.4× at
> 2048. At 512 a one-pixel boundary error leaves median IoU 0.71 and only 1.3% of crowns above
> 0.9, versus 0.90 and 51.4% at 2048 (`../../ablation/results/raster_quantum.json`). The 512
> raster also cost LACE more than SAM on switching (−0.0158 vs −0.0087 mean IoU), i.e. it was
> flattering us.
>
> ⚠️ **Two limits that travel with this result.** (1) **Double-assignment 0.0302 (2048) vs SAM's
> 0.0082, against a GT self-overlap floor of just 0.0015 — about 20×.** (An earlier version of
> this block said the floor was 0.0121; that was a single tile. Over 120 tiles it is 0.0048 at
> 512 and 0.0015 at 2048.) We claim pixels shared between touching crowns far more often than the
> ground truth does. (2) **No seed band, and one is not cheaply quotable**:
> `_selfmask_npz` is not seed-scoped, so one `em_model_4p_fix.npz` is shared by all three
> Table-1 seeds (which differ only in the *detector*), and this arm loads no detector. The
> `_fix` fit's own seed is unrecorded — `../../ablation/results/em_fit_report_4p_fix.json` is
> reconstructed from the npz and says so — though `fit_masker_4p` defaults to 0.
>
> The posterior product plays no part here: masks are byte-identical with and without it
> (verified). This is a mask-*shape* result; the product's contribution is to AP, separately.
> Scope the claim to *box→mask modules that accept a box prompt* — against mask-supervised
> maskers we still trail at high box IoU (`../../results_439/box_iou_strata.json`).
> Written up as Table `tab:gtbox` in `../../ablation/results/paper.tex`.

> ## Previous headline — LACE numbers unchanged, baseline comparisons superseded
> *The per-seed LACE figures below reproduce EXACTLY under the new scorer (0.625 / 0.636 /
> 0.629 → mean 0.630). Only the claims about Restor are superseded — see the 2026-08-28 block
> above and the dated update at the end of this file.*

> ## ⭐ BEST RESULT — the settled pipeline
> **4-phase real-8px interleave (L24) detector + β=0.5 self-mask masker, geometry-FIXED grid
> (`cell_origin=s`), @ `mask_thr=0.25`, with the box-robustness knobs (`prior_weight`=α=0.3,
> `kappa_scale`=κ×1.6)**, single-scale, OAM-TCD 439. **3-seed knobbed band (seeds 0,1,2):**
> - **Instance-seg:** mask mAP50 **0.630 ± 0.005** (per-seed 0.625 / 0.636 / 0.629) · mAP50-95 **0.257 ± 0.001**
> - **Detection:** box mAP50 **0.603** (3-seed; masker-invariant · seed-0 0.605) / mAP40 0.669 · best-F1@IoU0.4 **0.682**
>
> The two α/κ knobs are free inference-time scalars in the E-step (relax the imprecise-box spatial prior +
> sharpen the appearance vMF); they add **+0.007 mask AP50 / +0.016 mAP50-95 on EVERY seed** over the
> un-tuned masker, and their values are **selected on the held-out 108-tile val split** (pre-registered
> rule; (0.3, 1.6) is the val argmax — so the band below is select-on-val / report-on-test) — see
> **[box-robustness knobs](#box-robustness-knobs--the-α-κ-levers-2026-07-30)** + `BOX2MASK_LEVERS.md`. The **5-seed VANILLA band (α=1, κ=1) = 0.615 ± 0.011** is kept below as historical
> context / ablation (seeds 3,4 not yet re-evaluated with the knobs).
>
> **Beats fully-supervised DetecTree2 (mask 0.535), reproduced on the IDENTICAL split/test/metric** — i.e.
> our BOX+canopy-weak method beats a full-mask-supervised Mask R-CNN. ~~Also beats the OAM-TCD paper's Restor
> Mask R-CNN (0.432)~~ — **WITHDRAWN 2026-08-28: 0.432 is their published figure for an under-configured
> model (RPN capped at half detectron2's default). Re-run by us at the framework default the released
> checkpoint scores 0.6255, and the un-reranked LACE baseline TIES it. See the headline block above.**
> Still beats the vaulted multiscale headline (0.504) and the old β=0 headline (0.583). Full tables
> + per-seed bands in **[SETTLED PIPELINE + tables](#settled-pipeline--tables)**; the head-to-head in
> **[DetecTree2 baseline](#detectree2-baseline--apples-to-apples-2026-07-30)**.
>
> **⚠️ β=0.5 only wins AFTER the grid-registration fix (2026-07-27).** Before the fix, a 4px
> half-cell mis-registration made β=0.5's carve collapse (biased top-left), so β=0 looked best —
> an artifact, not the real masker ranking. See **[Registration bug and fix](#registration-bug-and-fix--the-reversal-2026-07-27)**.

## 🧭 HANDOFF — state, artifacts & paths (read this first)

**Current state:** detector **settled** (4-phase 8px L24, box mAP50 0.605); masker settled to
**β=0.5 self-mask on the geometry-FIXED grid + box-robustness knobs (α=0.3, κ×1.6)** @ mask_thr=0.25.
**3-seed knobbed band (0,1,2) = mask mAP50 0.630 ± 0.005 / mAP50-95 0.257 ± 0.001** (the deployable
headline). The **5-seed VANILLA band (α=1, κ=1) = 0.615 ± 0.011** is the historical/ablation baseline —
all 5 detectors are trained, seeds 3,4 just not yet re-evaluated with the knobs. **The α/κ knobs LIVE IN
THE MASKER npz** (`prior_weight`/`kappa_scale`, like `cell_origin`): `em_model_4p_fix.npz` stores 0.3/1.6,
`TCDMasker` reads them, and the eval chain DEFERS by default (Modal sentinel `-1.0` → the masker's stored
knobs). Pass `--prior-weight 1.0 --kappa-scale 1.0` for the vanilla masker; `em.set_masker_knobs(npz,α,κ)`
re-writes them post-tune without a refit; `fit`/`fit_masker_4p` write them so a new-dataset masker carries
its own calibration. Everything on Modal A100; detector features cached — **do not re-extract**. Maskers
are seed-independent (reuse across seeds).

**Grid fix verified complete.** The β=0-FIXED prior is dihedrally symmetric (quadrants ≈0.25),
and a local render-offset sweep confirms the `+1px@512` raster shift is OPTIMAL (|asym| minimized
at delta=1; delta=0 leaves masks strongly TL-biased — do NOT reduce it).

**5-seed VANILLA band DONE (2026-07-27; α=1, κ=1 — the pre-knob ablation baseline):** β=0.5-fixed, mask
mAP50 **0.615 ± 0.011** (per-seed 0.620 / 0.625 / 0.624 / 0.605 / 0.601), box mAP50 0.597 ± 0.010. Ran as
4 parallel single-seed A100s (seed 0 reused). The knobbed headline (α=0.3/κ×1.6) is the 3-seed band below.
**GPU per seed (from `band_selfmask` logs):** seed 1 & 2 = A100 80GB PCIe; seed 3, 4 & seed-2-redo =
A100-SXM4-40GB; seed 0 (earlier standalone eval) A100 variant not logged. (`band_selfmask` prints
`gpu=…`; a re-run records it per seed.)
Caveat: seed 2's first run was silently undertrained by a **Modal spot preemption** — `train_4p`'s
`if ckpt exists: skip` reused the preemption-partial ep10 checkpoint; deleting it + rerunning gave 0.624.
A preemption-safe completion marker for `train_4p` is a known (unaddressed) follow-up.

**Modal Volume `tcd04-phase4-vol`** (pull any: `modal volume get tcd04-phase4-vol <path> <dest>`):

| artifact | path on volume |
|---|---|
| detector ckpt (seed 0) | `out/det_phase4_L24_s0.pt` |
| **masker β=0.5 FIXED (THE one)** | `out/em_model_4p_fix.npz` |
| masker β=0 FIXED (compare) | `out/em_model_4p_b0_fix.npz` |
| **predictions β=0.5 FIXED** (439, boxes+RLE masks) | `out/preds_selfmask_fix_thr025_phase4_L24_s0/preds.json` |
| predictions β=0 FIXED (439, same boxes) | `out/preds_selfmask_b0_fix_thr025_phase4_L24_s0/preds.json` |
| best-model metrics (β=0.5 fixed, both tables) | `out/results_selfmask_fix_thr025_phase4_L24_s0.json` |
| — superseded (mis-registered) maskers, kept on-volume only | `out/em_model_4p.npz` (β0.5), `out/em_model_4p_b0.npz` (β0) |
| detector features | `feat_4p_train/` (900), `feat_4p_test/` (439), `native_test/` (439, 4096-d) |

Local copies of the fixed results + preds are pulled into `phase4/` (`results_b05_fix_thr025_439.json`,
`preds_b05_fix_thr025_439.json`, and the β0 equivalents).

**Key commands** (from `modal_tcd_multiseed/phase4/`):
- **Best eval (β=0.5, fixed grid, mask_thr 0.25, knobbed α=0.3/κ×1.6 — now the DEFAULT):**
  `modal run phase4_modal.py::eval_selfmask --beta 0.5 --fix` · add `--save-preds` to dump boxes+masks ·
  `--limit N` for a subset · `--beta 0` for the fill masker · `--prior-weight 1.0 --kappa-scale 1.0` for
  the VANILLA (pre-knob) masker · override knobs via `--prior-weight/--kappa-scale`.
- **Refit a fixed masker:** `modal run phase4_modal.py::fit_masker_4p --beta {0|0.5} --fix` (CPU, ~6 min).
- **Multi-seed band (β=0.5-fixed + α/κ knobs, now the default):** `::band_selfmask --seeds 0,1,2` (knobbed
  0.630 band), or `--seeds N` per seed for parallel A100s; done seeds are skipped/reused. Add
  `--prior-weight 1.0 --kappa-scale 1.0` for the vanilla 0.615 band (`_pw`-tag keeps them separate).
- `--fix` routes to `_fix` npz/preds/results so the pre-fix mis-registered models are preserved.
- Preds JSON format: `{meta, preds[tile]}` → `boxes_2048` (xyxy@2048px), `scores`, `canopy_ignore`,
  `masks_rle` (pycocotools RLE @512; boxes/`meta.scale_box_to_mask` to align). Decode:
  `pycocotools.mask.decode({"size", "counts": counts.encode("ascii")})`.

**Ops gotchas:** cancel a Modal run with **`modal app stop <ap-id> --yes`** — `pkill` only kills the
local client and leaves the remote A100 billing. Image pins **transformers 4.57** (do not bump; it's
cross-env vs the local 5.12 cache — never bit-compare Modal features to local).

---

**Question (the original go/no-go).** Does *real* 8px feature sampling (run the frozen DINOv3-web
backbone 4× on the tile shifted by every (dy,dx)∈{0,8}px, interleave into a real 256-grid) beat the
native *interpolated* 8px (Detector8's internal bilinear upsample) at layer 24? → **GO on
detection** (box mAP50 0.605 vs interp 0.540 / native 0.555).

**Recipe.** Detector4Phase (= Detector8 minus the internal interpolate) on (1024,256,256)
real-8px L24 features; det_t8 recipe + aggressive early-stop; single-scale box+mask eval;
box→mask by the self-mask EM masker refit on the SAME 4-phase L24 cells. Cohort = the 900/439
OAM-TCD set, joined from HF `restor/tcd` by image_id.

## Detection go/no-go (seed 0) — the original result, unchanged

| variant | env | box mAP50 | mask mAP50 (as-then) | best ep |
|---|---|---|---|---|
| **4-phase L24 real-8px** | **Modal (tf 4.57)** | **0.605** | (see masker sections) | 20 |
| interp L24 (probe) | local (tf 5.12) | 0.540 | 0.502 | 20 |
| native full-4096 | local (tf 5.12) | 0.555 | 0.499 | 20 |

**Detection clearly improves: box mAP50 0.605** vs interp-L24 0.540 / native-4096 0.555 (+0.05–0.065,
≫ the ~0.025 5-seed σ). Real-8px lets the detector resolve more crowns — the hypothesis's core
prediction. Caveat: the box gain is **cross-environment** (Modal tf-4.57 vs the local 5.12 cache the
baselines trained on; feature cos 0.86), so it conflates real-vs-interp with the version difference; a
same-env interp-L24 baseline is still un-run. The mask side is settled below.

Cost: extract ~$4.1 + train ~$4.5 + eval $2.0/masker. Train streamed 106 GB/epoch (~1.5–3.8 min/epoch);
eval masker is CPU-bound ~56 min.

## Registration bug and fix — the reversal (2026-07-27)

The user spotted, from output images, that every mask over-included the **top-left** corner and
carved the **bottom-right** — a distinctive shape not borne out by the image — and that β=0 was
winning *spuriously* (its wins were just larger, box-fill masks that IoU rewards).

**Root cause (confirmed).** The interleaved 8px `asm` grid places cell X's feature at physical pixel
**`8·X + 8`** (union of two 16px phase grids offset by 8; see `phase4_features_tcd.interleave`). But
the masker computed cell centers as `np.mgrid*s + s/2` = **`8·X + 4`** — off by **−4px (up-left)**.
Correct for the native 16px grid (`16g+8`), silently wrong for the interleaved 8px grid (needs `+s`,
not `+s/2`). This mis-placed crown appearance toward the top-left of every box → the learned prior
`pi` became TL-heavy → thresholded masks filled TL / carved BR; worst on **small crowns** (error =
4px / crown-width → the fixed-pixel-error signature).

**Evidence (all from the buggy `+s/2` state — diagnostic, no valid metric).** (1) The learned prior
`pi[s].sum(0)` was strongly top-left-heavy, and for the carve masker the bottom row + right column went
near-dead (the "collapse"). (2) The output masks filled the top-left and carved the bottom-right, worst on
small crowns (the fixed-pixel-error signature: error = 4px / crown-width). (3) Local proof (no Modal): the
crown's appearance-fg centroid sat up-left of the box centre under `+s/2` and recentred to the GT-mask
centroid (≈0.50, 0.50) under `+s`.

**Fix.** Per-model `cell_origin` (default s/2 = standard grid, backward-compatible; interleaved = s),
threaded through `em.py` (`cell_labels_canopy`/`ring_cells_canopy`/`TCDMasker.box_mask`; `fit` reads
`cell_origin_frac`, saves `cell_origin`), `phase4_fit_tcd.py` (loaders origin=s, `cell_origin_frac=1.0`),
and `evaluate.pred_instance_masks` (+1px@512 raster shift; 0 for native → byte-identical). Existing
pre-fix npz have no `cell_origin` → default s/2 → unchanged.

**Result of the fix (seed 0, 439 tiles, mask_thr 0.25, same detector/boxes → box mAP50 0.605 for all):**

| masker | mask mAP50 | mask mAP50-95 | box→mask gap | note |
|---|---|---|---|---|
| **β=0.5 self-mask, FIXED** | **0.6203** | **0.2436** | **−0.015** | **winner** — masks beat boxes @IoU0.5 |
| β=0 self-mask, FIXED | 0.5790 | 0.2003 | +0.026 | fill baseline |
| fixed vaulted 4096/16px (OOD, native masker) | 0.504 | 0.159 | +0.101 | earlier baseline (different masker, valid) |

**What this overturns.** The old masker forensics concluded *"the `contrastive_update` (β=0.5) helps only
large crowns and DESTROYS small ones; β=0 is the robust default; the contrastive is a novelty out of its
resolution regime on TCD."* **That verdict was driven by this bug** — the 4px seed gave the EM a consistent
TL direction to collapse toward. Once the grid is registered correctly, the contrastive carve **wins by
+0.041 mAP50 / +0.044 mAP50-95** over β=0, and the carve's collapse disappears. The user's hypothesis —
that a β near 0.5 should win — is confirmed. β=0 (near box-fill) is inherently robust to a half-cell
mis-registration; that robustness is exactly why it looked best before the fix.

**Grid fix verified complete.** Corner probe + a local render-offset sweep + direct prior inspection
confirm: the β=0-FIXED prior is now **dihedrally symmetric** (quadrants ≈0.25) and the `+1px@512` raster
shift is **optimal** (|asym| min at delta=1; delta=0 leaves a strong TL bias — do not reduce). The
mis-registration is fully corrected.

## SETTLED PIPELINE + tables

**Masker = β=0.5 self-mask, geometry-fixed grid, @ `mask_thr=0.25`.** `mask_thr` is the P(fg) cut in
`pred_instance_masks`; 0.25 (vs 0.5) grows masks to recover small crowns that under-cover under imprecise
PREDICTED boxes. Seed-0, 4-phase L24, single-scale.

### Table 1 — Detection (box), seed-0, single-scale, 439 TCD (masker-invariant)

| IoU | box AP | P@op | R@op | F1@op | **P (best-F1)** | **R (best-F1)** | **F1 (best-F1)** | maxR |
|---|---|---|---|---|---|---|---|---|
| **0.4** (NEON conv.) | 0.669 | 0.783 | 0.560 | 0.653 | 0.683 | 0.682 | **0.682** @thr0.33 | 0.854 |
| **0.5** | 0.605 | 0.751 | 0.537 | 0.626 | 0.670 | 0.625 | **0.647** @thr0.35 | 0.787 |

box mAP50-95 = 0.256. *NEON link (IoU 0.4, best-F1):* our NEON 4-phase F1 **0.728** (P0.727/R0.729),
DeepForest published 0.719 — vs TCD 4-phase F1 **0.682** (denser/smaller crowns).

### Table 2 — Instance segmentation (mask), seed-0, single-scale, β=0.5 FIXED @ 0.25

| metric | β=0.5 FIXED (winner) | β=0 FIXED (compare) |
|---|---|---|
| mask mAP50 | **0.6203** | 0.5790 |
| mask mAP50-95 | **0.2436** | 0.2003 |
| mask P/R/F1 @0.5 (op) | 0.759 / 0.541 / 0.632 | 0.732 / 0.523 / 0.610 |
| mask P/R/F1 @0.5 (best-F1) | 0.670 / 0.640 / **0.654** @thr0.34 | 0.652 / 0.608 / 0.629 @thr0.35 |
| semantic F1 | 0.577 | 0.578 |
| box→mask mAP50 gap | −0.015 | +0.026 |

The negative box→mask gap for β=0.5 means the carved masks score *higher* than the boxes at IoU 0.5 —
the carve sculpts crowns tighter than their bounding boxes match GT. mask_thr 0.25 (vs 0.5) mainly buys
semantic recall (fatter masks recover crown pixels); free (a threshold, no re-fit).

### Ablation — β=0 (fill) vs β=0.5 (carve), fixed grid

Seed 0, 439 TCD, single-scale, mask_thr 0.25, geometry-fixed grid. **Same detector/boxes → box mAP50 =
0.605 for both** (masker-invariant), so this is a clean masker A/B:

| masker | mask mAP50 | mask mAP50-95 | mask F1@0.5 (best) | box→mask gap |
|---|---|---|---|---|
| β=0 (fill) | 0.579 | 0.200 | 0.629 | +0.026 |
| **β=0.5 (carve)** | **0.620** | **0.244** | **0.654** | **−0.015** |

The carve wins across every mask metric: **+0.041 mAP50, +0.044 mAP50-95, +0.025 F1**. Its negative
box→mask gap means the carved masks match GT crowns *better* than the boxes match GT boxes. β=0.5 requires
the geometry fix — on a mis-registered grid the carve compounds the placement error and loses; see the
[reversal section](#registration-bug-and-fix--the-reversal-2026-07-27).

### Box-robustness knobs — the α/κ levers (2026-07-30) — THE HEADLINE RESULT

Two free inference-time scalars in the E-step, no re-fit / no re-extraction / novelty intact
(`prior_weight`=α, `kappa_scale`=κ; threaded through `estep`→`box_mask`→`pred_instance_masks`→eval):
- **α = prior_weight = 0.3** — scales the box-normalized spatial-prior logit `sigmoid(A + α·logit(psum))`.
  α<1 relaxes trust in the imprecise PREDICTED box so the box-INDEPENDENT appearance term `A` drives.
- **κ = kappa_scale = 1.6** — scales the appearance vMF concentration (both `zc` and `bg_ll`) → sharper /
  tighter boundaries. Orthogonal to α; they stack. (mask_thr stays 0.25.)

**Selected on VAL, not on test (2026-08-18).** The original sweep chose α/κ on TEST tiles (40-tile
pilot of the 439 → 130 → all 439), which made 0.630 a tuned-on-test number. `sweep_val_knobs` re-ran
the same grid on the held-out **108-tile val split** (7041 crowns, `val_gt.json` built by
`make_val_gt.py` in the prepare_test convention, identical AP core) under a rule pre-registered in
`BOX2MASK_LEVERS.md` before any val cell was read. **(α=0.3, κ×1.6, thr=0.25) is the outright val
argmax on both mask AP50 and mAP50-95** → the 3-seed 439 band below is select-on-val /
report-on-test, and **no test eval was re-run**. Caveat, stated plainly: the val surface is a broad
plateau (runner-up within 0.0003 mAP50-95; the whole α≤0.4/κ≥1.3 corner within 0.002), so this shows
the knobs were not chosen on the eval set — NOT that (0.3, 1.6) is uniquely optimal. What is robust
across the surface: α<1 beats α=1 at every κ, κ>1 beats κ=1 at every α. Val is held out for the
masker knobs but was used for detector early-stopping (box AP is masker-invariant, so selection is
uncontaminated), and the 2 scalars are fitted against 108 tiles of MASK labels — describe as "two
inference scalars selected on 108 held-out images", not as fully mask-label-free.

**Why:** SAM-3 box-prompt scores 0.633 on our EXACT boxes (a *floor*, not a method we use), but our EM
BEATS SAM on GT boxes (crown IoU 0.747 vs 0.724) — so the masker isn't worse, it's **box-imprecision
sensitive**. α/κ restore box-robustness. (RGB/vegetation-guided refinement FAILED: TCD boundaries are
crown-to-CROWN, image-invisible. Box-jitter TTA gained +0.009 but was DROPPED — 9× compute + unfair vs
single-pass SAM. Full lever map in `BOX2MASK_LEVERS.md`.)

**3-seed knobbed band (seeds 0,1,2), α=0.3 κ×1.6 mask_thr 0.25, 439 TCD** — paired vs the same-seed vanilla:

| seed | mask mAP50 (vanilla → **knobbed**) | mAP50-95 (vanilla → **knobbed**) | box mAP50 (invariant) |
|---|---|---|---|
| 0 | 0.6203 → **0.6250** | 0.2440 → **0.2574** | 0.6050 |
| 1 | 0.6253 → **0.6358** | 0.2362 → **0.2567** | 0.5981 |
| 2 | 0.6241 → **0.6294** | 0.2411 → **0.2564** | 0.6066 |
| **3-seed mean ± std** | 0.623 → **0.630 ± 0.005** | 0.240 → **0.257 ± 0.001** | **0.603** |

**Every seed improves on both metrics** (+0.007 AP50 / +0.016 mAP50-95, paired) → robust, not within-noise.
Box AP50 unchanged (masker-invariant). Seeds 3,4 not yet re-evaluated with the knobs (would complete a
5-seed knobbed band ≈ 0.622, matched to the vanilla 0.615). Command:
`::band_selfmask --seeds 0,1,2 --beta 0.5 --fix` (knobs are now the default; `_pw030_ks160`-tagged files).

### 5-seed VANILLA band — α=1, κ=1, β=0.5 FIXED, mask_thr 0.25, 439 TCD (historical / ablation)

The pre-knob baseline (no fine-tuned hyperparams) — kept for the full 5-seed variance and the DetecTree2
head-to-head:

| seed | mask mAP50 | mask mAP50-95 | box mAP50 | best_epoch |
|---|---|---|---|---|
| 0 | 0.6203 | 0.2440 | 0.6050 | 20 |
| 1 | 0.6253 | 0.2362 | 0.5981 | 15 |
| 2 | 0.6241 | 0.2411 | 0.6066 | 20 |
| 3 | 0.6045 | 0.2368 | 0.5942 | 25 |
| 4 | 0.6011 | 0.2300 | 0.5819 | 20 |
| **mean ± std** | **0.615 ± 0.011** | **0.238 ± 0.005** | **0.597 ± 0.010** | |

Tight spread (σ ≈ 0.011, well within the ~0.025 5-seed expectation); every seed beats the old β=0 headline
(0.583). Run as 4 parallel single-seed A100s + reused seed 0. Per-seed result JSONs on the volume are the
source of truth (the single-seed runs overwrite the shared summary). To reproduce vanilla now that knobs
are the default: add `--prior-weight 1.0 --kappa-scale 1.0`.

### Method lesson

Evaluate masker knobs on **PREDICTED boxes, not GT boxes** (the GT-box proxy mis-called the mask_thr
gain — perfect boxes don't under-cover). And: **a symmetric-looking geometry assumption (`+s/2`) can be
silently wrong for a non-standard grid** — the interleaved 8px lattice needed `+s`; the appearance-centroid
probe (crown mass should sit at box-centre 0.5) is the cheap check that catches it.

## DetecTree2 baseline — apples-to-apples (2026-07-30; **corrected 2026-08-18**)

> ### ⛔ SUPERSEDED 2026-09-03 — the whole arm was retrained; every number below is HISTORICAL
>
> The model this section describes was stopped by a `max_iter 4000` budget cap at 2.39 epochs
> while its validation AP50 was still rising monotonically (patience counter 0 — early stopping
> never fired). Three further defects compounded it: model selection on a strided 256 of 896 val
> subtiles; `INPUT.MIN_SIZE_TEST` left at detectron2's inherited COCO default of 800, silently
> downscaling our 1024 subtiles at test but not at train; and stitching at score 0.1 against a
> protocol floor of 0.05. **All four understated DetecTree2.**
>
> Retrained to detectree2's OWN AP50 early-stopping rule, it stops itself at iter 14000 and
> selects iter 10000 (6.0 epochs). **CN mask AP50 0.5344 → `0.5969`**, AP75 `0.1435 → 0.1748`,
> AP50:95 `0.2240 → 0.2578`, box AP50 `0.5310 → 0.5913`, dets `152,690 → 196,927`.
>
> The dated notes below record earlier, real fixes to a model that no longer exists. They are kept
> because deleting a changelog falsifies the record — **but do not quote any DetecTree2 figure
> from this section.** Current numbers: `../../PROTOCOL_439.md` and
> `../../detectree2_baseline/RETRAIN_V2.md`.

> ### 📌 2026-08-18 — prediction-coverage bug fixed; DetecTree2 439 numbers restated
>
> **DetecTree2's 439 figures dropped: mask mAP50 0.5448 → `0.5345`, mAP50-95 0.2277 → `0.2235`,
> box mAP50 0.5392 → `0.5290`, box mAP40 0.5951 → `0.5836`.** Our numbers are UNCHANGED — the bug
> was in the DetecTree2 subtiling path only.
>
> **Cause.** `build_coco.build_tile` dropped any 1024 subtile carrying no crown and no canopy
> annotation. That is correct for TRAINING (empty-sky subtiles skew background stats) but it was
> also applied to the TEST set, so DetecTree2 was never inferred on those regions and never charged
> for false positives there: **3189/3951 subtiles = 80.7% coverage**. Our method has no subtiling —
> it runs on the whole 2048 tile (256-cell grid × 8px = 2048px) and keeps zero-GT tiles (73 of the
> 439) — so it was always charged everywhere. The comparison was asymmetric in DetecTree2's favour.
>
> **Fix.** `build_tile(..., keep_empty=True)` for prediction builds (default `False`, so training
> and the original build stay byte-identical). Re-predicted at full 9/9 coverage: **3951 subtiles,
> 147310 → 152690 crowns (+5380)**.
>
> **The mechanism checks out exactly:** recall is IDENTICAL to 4 dp (mask best-F1 R 0.5544 →
> 0.5544; box 0.5508 → 0.5508) while precision falls (0.6004 → 0.5847). The added predictions match
> no new GT — they are pure false positives in regions that were previously invisible. That is the
> signature this bug must produce, and it does.
>
> **Also fixed (same date): a silent tile-drop in both scorers.** `score_detectree2.py` and
> `compare_subsets.py` selected `[t for t in sorted(gt) if t in preds]`, so a tile a model produced
> nothing for was removed from the GT denominator rather than scored as zero recall — which would
> hide a partial prediction run. Now every GT tile is scored, missing ones as zero-prediction, with
> a loud warning. Verified: dropping 4 of 12 tiles holds `n_gt` at 1090 and moves mask AP50
> 0.6528 → 0.5996 (previously it would have silently rescored on the surviving 8). **This did not
> change any published number** — every run here had predictions for all tiles.
>
> **Artifacts.** New: `out/preds_dt2_s0_fullcov.json` + `detectree2_baseline/results_dt2_s0_fullcov.json`.
> The published `out/preds_dt2_s0.json` (0.5448) is untouched for provenance. ⚠ `out/pred_raw_s0/`
> is now the full-coverage superset (3951 files), so it no longer reproduces the old 0.5448 preds
> exactly — the 762 added subtiles are the difference. Restated numbers come from the fullcov files.
> Cost of the correction ≈ $0.6 (762 new subtiles on A100 + CPU stitch; the 3189 existing raw
> predictions were reused).

Fully-supervised **DetecTree2** (Mask R-CNN R101-FPN) fine-tuned and evaluated on the **IDENTICAL**
cohort/metric as ours: same 792 train / 108 val / 439 test tiles, same GT (`test_gt.json`), same
COCO-101pt mask-AP + >50%-canopy-ignore scorer (`evaluate._greedy_ap`, RES 512). Code:
`boxinst_commonality_tcd_04/detectree2_baseline/` (isolated Modal app `tcd-detectree2`).

| method | supervision | OAM-TCD imgs | ign AP50 | ign AP50-95 | no-ign AP50 | no-ign AP50-95 | box AP50 |
|---|---|---|---|---|---|---|---|
| **OURS** (β=0.5-fix + α/κ knobs, 3-seed) | **boxes + canopy-ignore (no mask labels)** | 900 | **0.630 ± 0.005** | **0.257 ± 0.001** | **0.485 ± 0.002** | **0.202 ± 0.002** | 0.603 |
| OURS boxes → SAM 3 (frozen, box-prompt) | boxes (ours) + SAM pretrain (frozen) | 900 | 0.633 | 0.268 | | | 0.605 |
| Restor Mask R-CNN (OAM-TCD paper) \*\* ~~0.432~~ | full crown masks | 4,169 | | | **see ↑ headline: 0.6255 re-run by us** | | |
| **DetecTree2** (R101-FPN, fine-tuned by us) | **full crown masks** | 900 | **0.535** | 0.224 | 0.421 | 0.180 | 0.529 |
| SelvaBox→SAM3, both FT (SelvaMask) \* | boxes + **SelvaMask masks** + SAM pretrain | 3024 | | **0.185** | | | |
| SelvaBox→SAM3, frozen SAM \* | boxes + SAM pretrain (frozen) | 3024 | | 0.122 | | | |

**ign** = canopy neutralised; **no-ign** = canopy predictions count as false positives. Ours is a
3-seed band (mean ± std); DetecTree2 and the SAM 3 row are single-seed; published rows do not state
seed counts. The 5-seed vanilla band (α=1, κ=1) is tabulated separately below. SelvaMask's zero-shot
rows are omitted — no OAM-TCD training data, so they measure generalisation, not this benchmark.

**\* SelvaMask protocol — same 439 source images, different and easier test set.** They delete the
canopy category and black out its pixels (equivalent in effect to our ignore rule, hence the `ign`
column), cut to 1024 px @ 0.5 overlap, drop tiles left empty or >80% black (2,527 of 3,951 survive),
and average AP per tile. Our 439 keeps 73 zero-GT tiles. Their 439 is a genuine holdout for them
(official `validation_fold`), and their OAM-TCD baselines are **zero-shot** while ours are fine-tuned.
**\*\* Restor protocol** — the OAM-TCD paper's own Mask R-CNN, no canopy-ignore, AP50.
Full account, evidence and the master table: `RESULTS.md` § SelvaMask / CanopyRS.

**Our box-weak method beats full-mask-supervised DetecTree2 by +0.080 mask AP50 (vanilla 5-seed) /
+0.096 (knobbed 3-seed)**, and the win survives dropping canopy-ignore (+0.064 AP50 / +0.021 AP50-95),
despite DetecTree2 getting strictly more supervision. The gap is mostly the **detector**
(box 0.603 vs 0.529); both convert boxes→masks comparably.
**Reproduction is credible:** on the column-matched no-ignore AP50, our DetecTree2 (0.421) lands just
beside the OAM-TCD paper's own Restor Mask R-CNN (0.432 — **their published figure for an
under-configured model; we now measure 0.6255 for the released checkpoint at detectron2's default
proposal budget, see the 2026-08-28 headline**), while ours clears both (0.485) — a faithful
repro, not a strawman. (Earlier text here compared our ign 0.535 against Restor's no-ign 0.432, which
was not like-for-like.) DetecTree2 is a single seed; our knobbed number is 3-seed, our vanilla 5-seed —
a seed-matched knobbed band and a DT2 band are the obvious follow-ups.

**How DetecTree2 was run (faithful):** its published `setup_cfg` (R101-FPN, `base_lr=3.389e-4`,
`backbone_freeze=3`, its augmentations, AP50 early-stop) + released `250312_flexi.pth` weights,
reproduced via detectron2 (its py3.10 package won't install; `detectree2_baseline/dt2_recipe.py` ports
the recipe line-for-line). Full regime (max_iter 4000, ran clean to 57.1 val AP50), canopy=ignore in
training, native full-res mask supervision, fixed-pixel 1024@50% subtiles (matches our per-pixel regime),
`clean_crowns` dedup on stitch, **full 9/9 subtile coverage (corrected 2026-08-18)**.
**Caveats:** single DetecTree2 seed (ours is a 5-seed band — a DT2 band
is the obvious follow-up); tropical-tuned model applied to global OAM-TCD; DT2 preds floored at score 0.1
for tractable stitching (best-F1 op-point ~0.52, so no material AP effect). Artifacts on Volume
`tcd-detectree2-vol`: `out/model_best_s0.pth`, `out/preds_dt2_s0.json`; local `detectree2_baseline/results_dt2_s0.json`.
**GPU:** train + predict ran on Modal `gpu="A100"` (unpinned) — the 40 vs 80 GB variant was **not logged**
for these specific runs; `train`/`predict` now record `torch.cuda.get_device_name` (`out/train_info_s{seed}.json`
+ preds `meta.gpu`) so any re-run (e.g. the DT2 band) captures it. Stitch + scoring were CPU.

## Open next steps

- **Confidence (posterior product) — DONE 2026-08-28:** `s · bimod · pfg_mean`, zero fitted
  parameters. 3-seed 439 band **0.6625 ± 0.0012** (+0.0325, every seed and metric); held-out
  sparse 236 **0.6913 ± 0.0095** (+0.0354, larger off-distribution). Seed variance cut 4.5×.
  `../../confidence/README.md`, `../../results_439/sparse236/`.
- **All Table-1 baselines re-run by us — DONE 2026-08-28:** Restor (rpn 512/1000/2000, tree-only
  and pooled), SelvaBox → SAM 3 (both FT), DetecTree2, under one frozen scorer.
  `../../PROTOCOL_439.md`.
- **Open:** SelvaBox → SAM 3 (frozen SAM) is NOT re-run — their detector pairing is not
  recoverable from source; cite with an asterisk or omit.
- **Open:** the `preds/knobbed_s*.json` meta strings say "β=0 self-mask masker"; the files are
  **β=0.5**. Stale label predating the 2026-07-27 registration fix; worth a one-line correction
  since these are the paper's provenance record.

- **α/κ box-robustness knobs — DONE (headline):** 3-seed knobbed band 0.630 ± 0.005 / 0.257 ± 0.001, every
  seed improves (see [box-robustness knobs](#box-robustness-knobs--the-α-κ-levers-2026-07-30)). Phase4 eval
  defaults now = α=0.3/κ×1.6. **Open:** seeds 3,4 knobbed → a seed-matched 5-seed knobbed band (~$4,
  eval-only; `::band_selfmask --seeds 3,4 --beta 0.5 --fix`) to replace the 5-seed vanilla 0.615 cleanly.
- **Ceiling reality:** oracle-perfect masks on the SAME boxes = 0.737 AP50, maxR 0.818 → **mask mAP50 0.70
  needs BETTER DETECTION (small-crown recall), not the masker.** Box→mask caps ~0.633 on these detections.
- **5-seed vanilla variance — DONE:** band 0.615 ± 0.011 (ablation table above). Preds for all 5 seeds:
  `out/preds_selfmask_fix_thr025_phase4_L24_s{0..4}/`.
- **Preemption-safe `train_4p` (follow-up, unaddressed):** the `if ckpt exists: skip` guard reuses a
  preemption-partial checkpoint as if training finished (silently undertrained seed 2 on its first run).
  Mark completion (store/check the stop-criterion, or a `.done` sentinel) before treating a ckpt as reusable.
- **α/κ selected off-test — DONE (2026-08-18):** val-grid sweep (`::sweep_val_knobs`, 108 tiles) picks
  (0.3, 1.6, 0.25) = the settled values, under a pre-registered rule → no test re-runs. Surface + caveats
  in `BOX2MASK_LEVERS.md`. **Open:** the sweep ran on seed 0 only (knobs are masker-level, so seed-0
  selection is applied to all seeds); a full val thr sweep was not run (only thr 0.30 spot-checks).
- **Same-env interp-L24 baseline** (never run): clean same-env real-vs-interp box A/B.
- **Multiscale arm** (not tried): add the 0.5× downscale detection arm — likely lifts detection further.

## Files (isolated; `rm -rf` this folder + the `tcd04-phase4-vol` Volume undoes everything)

| file | role |
|---|---|
| `phase4_features_tcd.py` | 2048-tile 4-phase shift + interleave (256-grid; cell X ↔ pixel 8X+8) + L24 slice + native-4096 byproduct + reg self-test |
| `phase4_lib_tcd.py` | `Detector4Phase` + `train_4p` + evals: `eval_4p_selfmask` (β-masker @ mask_thr, `save_preds_dir`, `limit`) · `eval_4p` (native masker) · `_full_metrics`/`_instance_pr` (the two tables) |
| `phase4_fit_tcd.py` | `fit_masker_4p` — refit the EM masker on 4-phase L24 8px cells (`contrastive_beta`/`no_contrast`; **origin=s for the interleaved grid**, `cell_origin_frac=1.0`) |
| `phase4_modal.py` | A100 app: `verify` · `extract_4p` · `train_eval_4p` · `fit_masker_4p (--fix)` · **`eval_selfmask (--beta/--fix/--save-preds/--mask-thr/--limit)`** · **`band_selfmask`** (β=0.5-fixed 5-seed) · `band_4p` |
| `../../../boxinst_commonality_tcd_04/em.py` | `TCDMasker` (reads `cell_origin`) + `fit` (writes `cell_origin`, reads `cell_origin_frac`) + `cell_labels_canopy`/`ring_cells_canopy` (origin param) — the geometry fix lives here + in `evaluate.pred_instance_masks` (raster shift) |
| `stubs/boxinst_commonality/em.py` | Modal masker deps (REAL `logsumexp`/`estep`/`spherical_kmeans`/`contrastive_update`/`softmax`) — **keep** |
| `test_interleave_tcd.py`, `ref_feat_tcd.npz` | pure-numpy geometry test · layers-trap parity ref |
| `make_val_gt.py` | build `val_gt.json` — crown polygons for the 108 VAL tiles only (prepare_test convention) for OFF-TEST knob selection |
| `run_*.sh` | autonomous orchestrators w/ heartbeats + `app stop` on runaway |
| `results_*_fix_*.json`, `preds_*_fix_*.json` (local pulls) | geometry-fixed metrics + predictions; canonical copies on the Volume |

## Caveat log (things that bit us)

- **Grid-registration bug (2026-07-27):** the interleaved 8px masker used `+s/2` cell centers; the true
  centers are `+s` (8X+8). Half-cell up-left mis-registration → TL-fill/BR-carve, and β=0.5 "collapse".
  Fixed via `cell_origin`. See [Registration bug and fix](#registration-bug-and-fix--the-reversal-2026-07-27).
  The `+1px@512` raster shift in `pred_instance_masks` is **verified optimal** by a local render-offset
  sweep (|asym| min at delta=1; delta=0 → strong TL bias).
- **Vanished local caches:** `phase4_research/` and `corner_drizzle/` (~6.6 GB of gitignored 4-phase
  feature caches + probe scripts) disappeared mid-session 2026-07-27 (cause unknown; not in git → no
  record). Re-derivable by re-extracting the 50/20 test tiles' 4-phase features. Needed for FREE local
  render-offset tuning; otherwise tuning is Modal-only.
- **Layers-trap guard** fired benign: Modal (tf 4.57) vs local (tf 5.12) DINOv3 differ by cos 0.86 —
  *not* the catastrophic layers-default trap (cos 0.17). Gate relaxed to >0.5 after reg-test passes.
- **Mask-eval crash**: feat_ablation's `boxinst_commonality.em` stub was a no-op (box-only); replaced with
  a minimal real one (`logsumexp`+`estep`, parity-checked).

## Update 2026-08-28 — all baselines re-run under one scorer (`PROTOCOL_439.md`)

Everything in Table 1 is now measured by us. Previously the Restor and SelvaBox rows were
published figures under their own protocols; both are replaced.

**Frozen scorer.** `score_coco.py`: pycocotools `COCOeval` (not the hand-rolled
`evaluate._greedy_ap`), 439 whole 2048² tiles, masks at 512², maxDets 600, score floor 0.05,
`iouThrs = linspace(0.5, 0.95, 10)`, canopy as `iscrowd=1`. Validation: the canopy=FP arm
reproduces `results_dt2_s0_fullcov_noignore.json` to **4 dp**, and the residual +0.0008 on
AP50:95 is exactly the `np.arange`-vs-`linspace` ULP defect recorded in
`ablation/results/cocoeval_parity.json`. COCO's crowd rule was confirmed identical to our
>50%-in-canopy rule at IoU 0.50 on all four rows (Δ = 0.0000).

**Restor Mask R-CNN — run by us for the first time.** `restor/tcd-mask-rcnn-r50`, whole 2048²
tiles (their own test config), `SCORE_THRESH_TEST` 0.2→0.05, `DETECTIONS_PER_IMAGE` 512→600.
The decisive knob was `RPN.PRE/POST_NMS_TOPK_TEST`:

| rpn topk | dets/tile | recall@0.5 | CN AP50 |
|---|---|---|---|
| 512 (their shipped cfg) | 98 | 0.6442 | 0.5706 |
| **1000 (detectron2 default — published point)** | **163** | — | **0.6255** |
| 2000 (2× default, sensitivity only) | — | 0.7804 | 0.6605 |

512 proposals against tiles holding up to 450 GT crowns is proposal starvation, not a
scoring choice: lifting it recovers **3,503 true positives** while the top of the ranking is
untouched (max score identical at 0.9964). 1000 is the framework default — chosen by neither
party — and is the number we publish.

**SelvaBox → SAM 3 (both FT) — run by us.** Their OAM-TCD benchmark tiling (1024²@0.5, native
0.1 m/px), IoU-NMS 0.7, no edge-band cull, SAM 3 FT at their `target_tile_size` 1777.
439/439 tiles, 491,847 crowns. **Best detector in the table (box AP50 0.7297) and the worst
masker (CN AP75 0.0746).** Our measured AP50:95 of 0.2050 sits close to their published
0.185, indicating a faithful reproduction. Their maxDets budget (1600 per 2048²-equivalent)
vs our 600 is worth only **+0.008**, so the shared cap needs no asterisk.

**Confidence fix (`../../confidence/README.md`).** LACE ranks masks by the CenterNet heatmap
peak alone, which answers "is there a crown centre here", not "is this box real". Multiplying
the masker's own posterior back in — `score' = s · bimod · pfg_mean`, **zero fitted
parameters** — gives +0.0365 CN AP50 with no training, no mask labels and no change to what
is detected or segmented. Honest floor: a box-size prior alone recovers +0.0244 of that, so
the masker-specific increment is ≈ +0.012.

**Also corrected here:** the `preds/knobbed_s*.json` meta strings say "β=0 self-mask masker";
they are **β=0.5** (`em_model_4p_fix.npz`, per `RESULTS.md:133/228/237`, and the scored
numbers reproduce the β=0.5 row exactly). Stale label from before the 2026-07-27 registration
fix — worth correcting since these files are the paper's provenance record.
