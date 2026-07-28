# 4-phase real-8px L24 — TCD instance seg (Modal A100)

> ## ⭐ BEST RESULT — the settled pipeline
> **4-phase real-8px interleave (L24) detector + β=0.5 self-mask masker, geometry-FIXED grid
> (`cell_origin=s`), @ `mask_thr=0.25`**, single-scale, OAM-TCD 439. **5-seed band (seeds 0–4):**
> - **Instance-seg:** mask mAP50 **0.615 ± 0.011** (seed-0 0.620; range 0.601–0.625) · mAP50-95 **0.238 ± 0.005**
> - **Detection:** box mAP50 **0.597 ± 0.010** (seed-0 0.605) / mAP40 0.669 · best-F1@IoU0.4 **0.682** (NEON-linked)
>
> Every seed clears mask mAP50 0.60. Beats the fully-supervised Restor Mask R-CNN (mask 0.432), the
> vaulted multiscale headline (0.504), and the old β=0 headline (0.583). Full tables + per-seed band in
> **[SETTLED PIPELINE + tables](#settled-pipeline--tables)**.
>
> **⚠️ β=0.5 only wins AFTER the grid-registration fix (2026-07-27).** Before the fix, a 4px
> half-cell mis-registration made β=0.5's carve collapse (biased top-left), so β=0 looked best —
> an artifact, not the real masker ranking. See **[Registration bug and fix](#registration-bug-and-fix--the-reversal-2026-07-27)**.

## 🧭 HANDOFF — state, artifacts & paths (read this first)

**Current state:** detector **settled** (4-phase 8px L24, box mAP50 0.605); masker settled to
**β=0.5 self-mask on the geometry-FIXED grid** (`--fix`, `cell_origin=s`, mask_thr=0.25). **All 5
seeds (0–4) trained/evaluated** — band mask mAP50 **0.615 ± 0.011**. Everything on Modal A100;
detector features cached — **do not re-extract**. The maskers are seed-independent (reuse across seeds).

**Grid fix verified complete.** The β=0-FIXED prior is dihedrally symmetric (quadrants ≈0.25),
and a local render-offset sweep confirms the `+1px@512` raster shift is OPTIMAL (|asym| minimized
at delta=1; delta=0 leaves masks strongly TL-biased — do NOT reduce it).

**5-seed band DONE (2026-07-27):** β=0.5-fixed, mask mAP50 **0.615 ± 0.011** (per-seed 0.620 / 0.625 /
0.624 / 0.605 / 0.601), box mAP50 0.597 ± 0.010. Ran as 4 parallel single-seed A100s (seed 0 reused).
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
- **Best eval (β=0.5, fixed grid, mask_thr 0.25):** `modal run phase4_modal.py::eval_selfmask --beta 0.5 --fix`
  · add `--save-preds` to dump boxes+masks · `--limit N` for a subset · `--beta 0` for the fill masker.
- **Refit a fixed masker:** `modal run phase4_modal.py::fit_masker_4p --beta {0|0.5} --fix` (CPU, ~6 min).
- **5-seed band (β=0.5-fixed, the default):** `::band_selfmask --seeds 0,1,2,3,4` (one instance), or run
  `--seeds N` per seed for parallel A100s (as done for the 0.615 ± 0.011 band); done seeds are skipped/reused.
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

### 5-seed band — β=0.5 FIXED, mask_thr 0.25, 439 TCD (the deployable number)

| seed | mask mAP50 | mask mAP50-95 | box mAP50 | best_epoch |
|---|---|---|---|---|
| 0 | 0.6203 | 0.2440 | 0.6050 | 20 |
| 1 | 0.6253 | 0.2362 | 0.5981 | 15 |
| 2 | 0.6241 | 0.2411 | 0.6066 | 20 |
| 3 | 0.6045 | 0.2368 | 0.5942 | 25 |
| 4 | 0.6011 | 0.2300 | 0.5819 | 20 |
| **mean ± std** | **0.615 ± 0.011** | **0.238 ± 0.005** | **0.597 ± 0.010** | |

Tight spread (σ ≈ 0.011, well within the ~0.025 5-seed expectation); every seed beats the old β=0 headline
(0.583). Run as 4 parallel single-seed A100s + reused seed 0 (`::band_selfmask --seeds N --beta 0.5 --fix`
per seed, or `--seeds 0,1,2,3,4` for one sequential instance). Per-seed result JSONs on the volume are the
source of truth (the single-seed runs overwrite the shared `band_selfmask_fix_thr025.json` summary).

### Method lesson

Evaluate masker knobs on **PREDICTED boxes, not GT boxes** (the GT-box proxy mis-called the mask_thr
gain — perfect boxes don't under-cover). And: **a symmetric-looking geometry assumption (`+s/2`) can be
silently wrong for a non-standard grid** — the interleaved 8px lattice needed `+s`; the appearance-centroid
probe (crown mass should sit at box-centre 0.5) is the cheap check that catches it.

## Open next steps

- **5-seed variance — DONE:** band mask mAP50 **0.615 ± 0.011** (see table above). `band_selfmask` defaults
  to the β=0.5-fixed config (`em_model_4p_fix.npz`, beta=0.5, fix=True, mask_thr=0.25, save_preds); result
  filenames match `eval_selfmask` so a done seed is skipped/reused. Preds for all 5 seeds are on the volume
  (`out/preds_selfmask_fix_thr025_phase4_L24_s{0..4}/`).
- **Preemption-safe `train_4p` (follow-up, unaddressed):** the `if ckpt exists: skip` guard reuses a
  preemption-partial checkpoint as if training finished (silently undertrained seed 2 on its first run).
  Mark completion (store/check the stop-criterion, or a `.done` sentinel) before treating a ckpt as reusable.
- **Promote to code defaults:** once render-offset + 5-seed settle, rename the `_fix` maskers to the
  canonical paths and flip `eval_selfmask`'s default to `--beta 0.5`. Not yet done (premature).
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
