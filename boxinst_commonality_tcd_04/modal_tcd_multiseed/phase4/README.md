# 4-phase real-8px L24 — TCD seed-0 go/no-go (Modal A100)

> ## ⭐ BEST RESULT — the settled pipeline
> **4-phase real-8px interleave (L24) detector + β=0 self-mask masker @ `mask_thr=0.25`**, seed-0, single-scale, OAM-TCD 439:
> - **Detection:** box mAP50 **0.605** / mAP40 0.669 · best-F1@IoU0.4 **0.682** (NEON-linked)
> - **Instance-seg:** mask mAP50 **0.583** · mAP50-95 0.200 · semantic F1 0.578
>
> Beats the vaulted multiscale headline (mask 0.504) and the fully-supervised Restor Mask R-CNN (0.432).
> Full tables + how it was reached in **[SETTLED PIPELINE + tables](#settled-pipeline--tables-2026-07-24)** below.
> Deployable 5-seed band: `modal run phase4_modal.py::band_selfmask --seeds 0,1,2,3,4`.
> _(The rest of this doc is the chronological investigation that led here — the go/no-go, the β=0 payoff, and the sigmoid-γ scoped-negative.)_

## 🧭 HANDOFF — state, artifacts & paths (read this first)

**Current state:** detector + masker **settled** (β=0 self-mask @ mask_thr=0.25, code defaults).
Only **seed 0** is trained/evaluated. Everything runs on Modal A100; features are cached — **do
not re-extract**. Numbers/tables in [SETTLED PIPELINE + tables](#settled-pipeline--tables-2026-07-24).

**Modal Volume `tcd04-phase4-vol`** (pull any: `modal volume get tcd04-phase4-vol <path> <dest>`):

| artifact | path on volume |
|---|---|
| detector ckpt (seed 0) | `out/det_phase4_L24_s0.pt` |
| **masker β=0 (THE one)** | `out/em_model_4p_b0.npz` |
| masker β=0.5 (carve, for compare/negative) | `out/em_model_4p.npz` |
| best-model metrics (both tables) | `out/results_selfmask_b0_thr025_phase4_L24_s0.json` |
| **predictions β=0** (boxes+RLE masks) | `out/preds_selfmask_b0_thr025_phase4_L24_s0/preds.json` |
| predictions β=0.5 (same boxes, carved masks) | `out/preds_selfmask_thr025_phase4_L24_s0/preds.json` |
| detector features | `feat_4p_train/` (900), `feat_4p_test/` (439), `native_test/` (439, 4096-d) |

⚠️ **Predictions are currently the first 50 tiles only** (last run used `--limit 50`). For the full
439, re-run without `--limit` (see below) — it overwrites the 50-tile files at the same paths.
⚠️ The `results_selfmask_b0_thr025_*.json` on the volume is currently the **50-tile** version (the
limit run overwrote it); the **439** numbers are in this README + `phase4/results_*.json` local copies.

**Key commands** (from `modal_tcd_multiseed/phase4/`):
- Best eval (defaults β=0, mask_thr=0.25): `modal run phase4_modal.py::eval_selfmask`
  · add `--save-preds` to dump boxes+masks · `--limit N` for a subset · `--beta 0.5` for the carve masker.
- **Deployable 5-seed band:** `modal run phase4_modal.py::band_selfmask --seeds 0,1,2,3,4` (~$1.5–2/seed).
- Sigmoid-γ blend (scoped-negative): `::eval_blend`.  Refit a masker: `::fit_masker_4p --beta {0|0.5}`.
- Preds JSON format: `{meta, preds[tile]}` → `boxes_2048` (xyxy@2048px), `scores`, `canopy_ignore`,
  `masks_rle` (pycocotools RLE @512; boxes/`meta.scale_box_to_mask` to align). Decode:
  `pycocotools.mask.decode({"size", "counts": counts.encode("ascii")})`.

**Ops gotchas:** cancel a Modal run with **`modal app stop <ap-id> --yes`** — `pkill` only kills the
local client and leaves the remote A100 billing. Image pins **transformers 4.57** (do not bump; it's
cross-env vs the local 5.12 cache — never bit-compare Modal features to local).

**Open threads (none blocking):** 5-seed variance (`band_selfmask`, only seed 0 done) · full-439
predictions (only 50 saved) · same-env interp-L24 baseline (never run) · multiscale arm (not tried).
The sigmoid-γ carve is a **closed negative** — don't reopen without new box-precision.

**Question.** Does *real* 8px feature sampling (run the frozen DINOv3-web backbone 4×
on the tile shifted by every (dy,dx)∈{0,8}px, interleave into a real 256-grid) beat the
native *interpolated* 8px (Detector8's internal bilinear upsample) at layer 24?

**Recipe.** Detector4Phase (= Detector8 minus the internal interpolate) on (1024,256,256)
real-8px L24 features; det_t8 recipe + aggressive early-stop; single-scale box+mask eval;
box→mask by the FIXED vault EM masker (full-4096 native features). Cohort = the 900/439
OAM-TCD set, joined from HF `restor/tcd` by image_id.

## Result (seed 0)

| variant | env | mask mAP50 | box mAP50 | box→mask | semF1 | best ep |
|---|---|---|---|---|---|---|
| **4-phase L24 real-8px** | **Modal (tf 4.57)** | **0.5039** | **0.605** | **0.101** | 0.548 | 20 |
| interp L24 (probe) | local (tf 5.12) | 0.502 | 0.540 | 0.038 | 0.563 | 20 |
| native full-4096 | local (tf 5.12) | 0.499 | 0.555 | 0.056 | 0.587 | 20 |

Cost: extract ~$4.1 + train ~$4.5 + eval $2.0 ≈ **$10.6**. (Train streamed 106 GB/epoch
from the Volume → ~1.5–3.8 min/epoch; eval masker is CPU-bound ~56 min.)

## Reading it — two signals, both with caveats

1. **Detection clearly improves: box mAP50 0.605** vs interp-L24 0.540 / native-4096 0.555
   (+0.05–0.065, ≫ the ~0.025 5-seed σ). Training val boxAP50 also led (0.498 vs the
   probe's 0.475). Real-8px lets the detector resolve more crowns — the hypothesis's
   core prediction. **BUT** this is **cross-environment** (Modal transformers 4.57 vs the
   local 5.12 cache the baselines were trained on; feature cos 0.86), so the box gain
   conflates real-vs-interp with the version difference. Not yet a clean A/B.

2. **Mask mAP50 only ties (0.5039 vs 0.502)** — and it is **not trustworthy here**. The
   box→mask gap **doubled to 0.101** (vs 0.038–0.056 local). Two reasons, both masker-side:
   the vault EM masker was **fit on local (5.12) features** and is applied to **Modal (4.57)
   features it wasn't fit for** (OOD), and it was fit on native 16px cells, not the denser
   4-phase boxes. So the masker — not the detector — caps the mask metric. semF1 down
   (0.548, recall 0.42) is the same story: masks under-cover.

**Bottom line.** The go/no-go leans **GO on detection** (box mAP50 jump is large and in the
predicted direction), but the **mask metric is inconclusive** — confounded by (a) the
transformers-version drift vs the local baselines and (b) the fixed masker being OOD on
Modal features. "Beats 0.502 mask" is technically true but within noise and confounded;
don't over-claim it. The clean, trustworthy signal is the box side.

## Masker investigation — RESOLVED (full forensics in `../../mps_tcd_multiseed_4phase/masker_lab/README.md`)

Refitting the masker on the 4-phase L24 cells made mask mAP **worse** (0.449 vs the
vaulted's 0.504), which triggered a deep forensic dig. Conclusion:

> **TCD mask mAP is DETECTION-DOMINATED (crowns fill ~71% of their box; box-fill floor =
> 0.710 mean IoU). The commonality masker's `contrastive_update` helps only WELL-RESOLVED
> (large) crowns and DESTROYS under-resolved (small) ones — the controlling variable is
> CELLS-PER-CROWN (resolution), not layer/density/8px. β=0 is the robust default; the
> contrastive is a real, dryland-validated novelty out of its resolution regime on TCD's
> small crowns.**

Key evidence (GT-box mean IoU isolates the masker): Δ(β0−β0.5) by crown size = **+0.233
(<25px) → +0.097 → +0.009 → −0.024 (>70px)**, perfectly monotonic. β=0 (0.741) > box-fill
(0.710) > vaulted β=0.5 (0.657, *over-carves below doing nothing*). The refit's fg
prototypes collapsed (pairwise cosine 0.96) via a `contrastive_update` positive-feedback
loop, worst at 8px (correlated cells). "density/isolation" was a size confound (isolated
crowns median 29px vs 61px). Dryland proxies confirm the contrastive carves well on
well-resolved crowns (centre 0.96 / corner 0.27, beats the boxinst head) — so the novelty
is scoped, not wrong.

**Design decision:** detector settled (4-phase 8px L24, box 0.605); masker = **β=0**
(fill + light carve, no collapse, fit & apply at 8px). The `fit_masker_4p`/`eval_selfmask`
functions and the β=0.5 self-mask result (`results_selfmask_phase4_L24_s0.json`,
mask 0.449) are kept as the negative-result record.

**PAYOFF — RUN (2026-07-23): β=0 self-mask unlocks the detection win.** Real-pipeline eval,
4-phase L24 seed-0 detector boxes (box mAP50 0.605), single-scale, masker refit β=0
(no_contrast, 8px, diverse prototypes cos 0.223 — no collapse):

| masker (4-phase boxes, single-scale) | mask mAP50 | mask 50-95 | box→mask gap | semF1 |
|---|---|---|---|---|
| **β=0 self-mask (`em_model_4p_b0.npz`)** | **0.5794** | **0.1948** | **0.0256** | 0.5772 |
| fixed vaulted 4096/16px (OOD) | 0.5039 | 0.1591 | 0.101 | 0.548 |
| β=0.5 self-mask (collapse) | 0.449 | 0.145 | 0.156 | — |

**box→mask gap collapsed 0.101 → 0.026** (nearly lossless) — the box-0.605 win now flows to
mask. **mask mAP50 0.579**: +0.075 over the vaulted multiscale headline (0.504), +0.080 over
native single-scale (0.499) and interp-L24 (0.502); mask 50-95 also up (0.195 vs 0.159).
Confirms the forensic verdict: TCD mask is detection-dominated + β=0 is the robust masker.
Caveats: single-scale, **seed-0 only** (run `band_4p` for variance), Modal tf-4.57 features
(box cross-env vs local, but the mask number is now fully same-env / in-distribution).

## SETTLED PIPELINE + tables (2026-07-24)

**Masker = β=0 self-mask @ `mask_thr=0.25` (the default in the code).** The mask threshold
is the P(fg) cut in `pred_instance_masks`; lowering it 0.5→0.25 grows masks to recover
small crowns that under-cover under imprecise PREDICTED boxes. Seed-0, 4-phase L24,
single-scale:

| β=0 masker | mask mAP50 | mask mAP50-95 | semantic F1 | sem R | box→mask |
|---|---|---|---|---|---|
| mask_thr 0.50 | 0.5794 | 0.1948 | 0.550 | 0.429 | 0.026 |
| **mask_thr 0.25 (default)** | **0.5831** | **0.1998** | **0.578** | **0.509** | 0.022 |

The instance-AP gain is small (+0.004 / +0.005) — the 4-phase β=0 masker at 8px is already
tight (near-lossless box→mask), so little under-covering remains. The real lift is
**semantic recall +0.08 (F1 +0.028)** — fatter masks recover crown *pixels*. Free (a
threshold, no re-fit / no extra inference).

### Table 1 — Detection (box), seed-0, single-scale, 439 TCD (masker-invariant)

| IoU | box AP | P@op | R@op | F1@op | **P (best-F1)** | **R (best-F1)** | **F1 (best-F1)** | maxR |
|---|---|---|---|---|---|---|---|---|
| **0.4** (NEON conv.) | 0.669 | 0.783 | 0.560 | 0.653 | 0.683 | 0.682 | **0.682** @thr0.33 | 0.854 |
| **0.5** | 0.605 | 0.751 | 0.537 | 0.626 | 0.670 | 0.625 | **0.647** @thr0.35 | 0.787 |

*NEON link (IoU 0.4, best-F1):* our NEON 4-phase F1 **0.728** (P0.727/R0.729), DeepForest
published 0.719 — vs TCD 4-phase F1 **0.682**. Same convention, different dataset (TCD
denser/smaller crowns).

### Table 2 — Instance segmentation (mask), seed-0, single-scale, β=0 @ 0.25

| metric | value |
|---|---|
| mask mAP50 | **0.5831** |
| mask mAP50-95 | 0.1998 |
| mask P/R/F1 @0.5 (op) | 0.735 / 0.525 / 0.612 |
| mask P/R/F1 @0.5 (best-F1) | 0.655 / 0.610 / 0.632 |

### Sigmoid-γ carve — RUN, NEGATIVE (kept as a scoped result)

The sigmoid-γ refinement (weight the carve by crown resolution) was built (`BlendMasker`,
a resolution-gated fill↔carve ensemble) and evaluated. On **GT boxes** it gained +0.037
mask mAP50-95; on the **real 4-phase pipeline it LOST** (−0.006 mAP50 / −0.016 mAP50-95).
The carve only helps when the box tightly bounds the crown, and weak-sup detector boxes
are imprecise, so carving corner-background removes real crown pixels and hurts tightness.
**Scoped result:** the mechanism works with precise boxes, not through weak-sup detection —
consistent with "TCD mask is detection-dominated; the masker can't beat its boxes' precision."

### Method lesson (both directions)

Evaluate masker knobs on **PREDICTED boxes, not GT boxes**. The GT-box proxy *over*-predicted
the sigmoid-γ gain (perfect boxes let the carve work) and *under*-predicted the mask_thr gain
(perfect boxes don't under-cover, so it said lower thr hurts). Only the predicted-box regime
(det_t8 sweep) gave the right call on both.

### Deployable 5-seed band

`modal run phase4_modal.py::band_selfmask --seeds 0,1,2,3,4` — 4-phase detector + β=0 masker
@ mask_thr=0.25, per-seed idempotent, reuses the seed-independent masker + cached features
(~$1.5–2/seed). Seeds 1–4 extend the seed-0 headline for variance.

## Open next steps (nothing blocking; see Handoff "Open threads")

- **5-seed variance** (the main gap): `modal run phase4_modal.py::band_selfmask --seeds 0,1,2,3,4`
  — β=0 @ 0.25, per-seed idempotent, reuses cached features (~$1.5–2/seed). Only seed 0 done.
- **Same-env interp-L24 baseline** (never run): train Detector8 on the native phase-(0,0) L24 slice
  on Modal, eval identically → a clean same-env real-vs-interp box A/B (the box 0.605 is currently
  cross-env vs local). Cheap (phase-(0,0) native already extracted).
- **Multiscale arm** (not tried): add the 0.5× downscale detection arm (as native did, 0.499→0.504)
  to the 4-phase detector — likely lifts detection further.
- *Done:* masker refit on Modal/4-phase cells (→ β=0 self-mask); mask_thr sweep (→ 0.25 default).

## Files (isolated; `rm -rf` this folder + the `tcd04-phase4-vol` Volume undoes everything)

| file | role |
|---|---|
| `phase4_features_tcd.py` | 2048-tile 4-phase shift + interleave (256-grid) + L24 slice + native-4096 byproduct + reg self-test |
| `phase4_lib_tcd.py` | `Detector4Phase` + `train_4p` + evals: `eval_4p_selfmask` (β-masker @ mask_thr, `save_preds_dir`, `limit`) · `eval_4p_blend` (sigmoid-γ, negative) · `eval_4p` (fixed vault masker) · `_full_metrics`/`_instance_pr` (the two tables) · `BlendMasker` |
| `phase4_fit_tcd.py` | `fit_masker_4p` — refit the EM masker on 4-phase L24 8px cells (`contrastive_beta`/`no_contrast`) |
| `phase4_modal.py` | A100 app: `verify` · `extract_4p` (reg+layers-trap gate) · `train_eval_4p` · `fit_masker_4p` · **`eval_selfmask`** (best; `--save-preds`/`--limit`/`--beta`/`--mask-thr`) · **`band_selfmask`** (deployable 5-seed) · `eval_blend`/`band_blend` (sigmoid-γ) · `band_4p` (fixed-masker) |
| `stubs/boxinst_commonality/em.py` | Modal masker deps (REAL `logsumexp`/`estep`/`spherical_kmeans`/`contrastive_update`/`softmax`) — **keep** |
| `test_interleave_tcd.py`, `ref_feat_tcd.npz` | pure-numpy geometry test · layers-trap parity ref |
| `run_*.sh` | autonomous orchestrators (extract→train→eval, save_preds, blend, thr) w/ heartbeats + `app stop` on runaway |
| `results_*.json`, `preds_*` (local pulls) | metrics + prediction copies; canonical copies live on the Volume (see Handoff) |
| `../../mps_tcd_multiseed_4phase/masker_lab/` | local/free forensics: β-sweep, size-control, gamma sweeps, det_t8 predicted-box mask_thr sweep |

## Caveat log (things that bit us)

- **Layers-trap guard** fired benign: Modal (tf 4.57) vs local (tf 5.12) DINOv3 differ by
  cos 0.86 — *not* the catastrophic layers-default trap (that was cos 0.17). Gate relaxed
  to >0.5 after confirming reg-test (invertibility + real≠interp) passes.
- **Mask-eval crash**: feat_ablation's `boxinst_commonality.em` stub is a no-op (it was
  box-only); replaced with a minimal real one (`logsumexp`+`estep`, parity-checked). The
  crash was post-training, so the committed checkpoint let the re-run skip straight to eval.
