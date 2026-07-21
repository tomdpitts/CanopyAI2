# 4-phase "center-registration" — NEON seed-0 go/no-go

**Hypothesis.** The native detector runs the frozen DINOv3-web backbone on ONE 16px patch
grid, then `DetectorS(up=2)` **bilinearly upsamples** those 16px features to an 8px grid.
That 8px grid is *interpolated* — no new sub-patch information. Here we instead run the
backbone **four times**, on the image shifted by every `(dy,dx) ∈ {0,8}px`, and
**interleave** the four 16px grids into a **real** 8px grid where each cell is filled by
the phase whose patch is centered nearest it ("pick-one", stationary). Same 4096-dim,
same targets/decode/stride-8 as native — **the only change is real-8px vs interpolated-8px
features.** Question: does that raise the recall ceiling, especially at NIWO?

Prediction from the native upscale A/B (README.md §Multiscale): NIWO's ceiling is largely
*window*-limited (2× upscale doubled NIWO maxR 0.348→0.701), and 4-phase does **not** shrink
the 16px window — so expect a *modest* NIWO lift at best. A null here means NIWO is
window-limited → pivot to **train-time upscale**. A clear lift means there was a
precision-safe *sampling* win the (precision-crashing) upscale arm masked.

## Files (all isolated in this folder)

| file | role |
|---|---|
| `phase4_features.py` | 4-phase shift + `interleave()` (pixel-unshuffle inverse) + `registration_self_test` (real-DINO abort-fast guard) |
| `phase4_lib.py` | `Detector4Phase` (= `Detector8` minus the internal interpolate) + `train_seed` (native recipe verbatim) + `predict_boxes_4p` |
| `phase4_modal.py` | Modal H100 app: `extract_4p`, `train_eval_4p` |
| `test_interleave.py` | local NO-GPU geometry test (invertibility, blob registration ≤1 cell, monotonic ordering, sub-lattice) |
| `phase4_score.py` | local apples-to-apples scoring vs native seed-0 (global-194 + NIWO-12) with `df_scorer` |

## Run

```bash
# 0) geometry test (free, no GPU)
../../.venv/bin/python phase4/test_interleave.py

# 1) 4-phase features (H100, ~15-18 min, ~$1.5) — reg-test aborts fast if interleave wrong
../../.venv/bin/modal run phase4/phase4_modal.py::extract_4p

# 2) seed-0 train + eval (H100) — writes /vol/phase4/out/preds_phase4_s0.json
../../.venv/bin/modal run phase4/phase4_modal.py::train_eval_4p

# 3) pull preds + score apples-to-apples
../../.venv/bin/modal volume get neon-multiseed-vol phase4/out/preds_phase4_s0.json phase4/out/
.venv_df/bin/python phase4/phase4_score.py
```

## RESULT — seed 0 (2026-07-21): GO. Real-8px sampling helps, precision-safe.

Apples-to-apples vs native seed 0 (`df_scorer`, IoU 0.4). Extract 18.8min (~$1.5) +
train 30.5min/best-ep25 (~$2.5) = **~$4.06 total**, one H100, seed 0 only.

| scope | arm | P | R | **maxR** |
|---|---|---|---|---|
| global (194) | native | 0.731 | 0.679 | 0.784 |
| global | **4-phase** | 0.720 | 0.724 | **0.850** |
| NIWO (12) | native | 0.533 | 0.343 | 0.348 |
| NIWO | **4-phase** | **0.625** | **0.552** | **0.589** |

Δ (paired, seed 0): global ΔP −0.011 ΔR +0.044 **ΔmaxR +0.066**;
NIWO ΔP **+0.092** ΔR **+0.210** ΔmaxR **+0.241**.

- **Recall ceiling cleared**: global maxR 0.784→0.850, now **past DeepForest's R 0.790**
  (native literally couldn't reach it). ΔmaxR ≈ 3× seed noise (±0.021); NIWO Δ ≈ 10×.
- **NIWO up on BOTH axes, precision-safe** — the win the (precision-crashing) inference
  upscale arm masked. Global precision held (−0.011) vs upscale's 0.731→0.523.
- **Not yet a DeepForest *beat*** on the strict dominance test: P@R0.79 = **0.615**
  (need >0.659; native: can't-reach) and R@P0.66 = **0.758** (need >0.790; native 0.720).
  Closes ~½–⅔ of the gap and reaches DF's recall regime for the first time.
- **Seed-0 only** — magnitude is well outside seed noise, but a 5-seed band is needed for
  error bars before a firm claim. Follow-ups queued: 5-seed band, 2-phase fuse, mid-layer
  ×4-phase, train-time upscale (the window lever).

## Design notes / correctness

- **Registration.** Encode/decode are byte-identical to native (`TargetConfig(grid=64,
  stride=8)`), so the comparison is purely feature quality. The assembled cell `X` sits at
  patch-center pixel `8X+8` vs the target cell center `8X+4` — a uniform **half-cell**
  offset absorbed by the offset head (init bias 0.5), exactly as native absorbs its own
  interpolation offset. Verified: `test_interleave.py` blob test ≤1 cell across 200 spots;
  real-DINO `reg-test` shows invertibility (maxabs ~2e-4 fp16) and real≠interp cos≈0.955.
- **Edge.** An 8px shift zero-fills a ≤8px top/left strip; NEON 400px tiles have ≥112px
  bottom-right pad so this is lossless there, and the pad region is dropped at eval anyway.
- **Cost.** Extract (one-time) ~$1.5; seed-0 train+eval ~$1.5–3 → total ≈ **$3–4.5** on
  H100. `train_eval_4p` prints a live `est_cost_usd` from wall-time.

## Cleanup (risk-free — nothing shared is modified)

```bash
rm -rf phase4/                                   # this whole folder
../../.venv/bin/modal volume rm neon-multiseed-vol phase4   # the /vol/phase4 subtree
```
No native file, default, or Volume input is touched — the native pipeline is unaffected.
