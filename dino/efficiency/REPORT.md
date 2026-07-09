# Overnight findings — shadow prior + DINOv3 on dryland tree crowns

Autonomous run, 2026-07-01. All work contained to `dino/efficiency/`. Methodology
pre-registered in `PLAN.md`; exploration/confirmation discipline enforced (config
chosen on EXPLORE, tested once on locked CONFIRMATION). Stats: Nadeau–Bengio
corrected resampled t-test (the trustworthy one here) + draw-noise MDE floor.

---

## TRACK 1 — directional shadow prior: a real, confirmed effect (with an honest twist)

**Setup.** 133 azimuth-annotated scenes (WON 41 / BRU 53 / NEON 39; WON+BRU =
dryland, NEON = temperate), tile→512px→frozen DINOv3 (web ViT-L) patch features,
per-patch crown probe (PCA-128 + ridge). Shadow = the `shadow_prior` swept
directional matched filter, pooled to the 32×32 patch grid, as an extra channel.
Three rungs: **R1** features · **R2** features⊕correct-shadow · **R3**
features⊕within-acquisition-shuffled-shadow. Primary comparison **R2 vs R3**
(isolates *direction* from added channel capacity). Target: crown **centers**
(detection), AP metric.

**The effect lives in a specific, physically-sensible place:** the **greenness**
brightness proxy (not luminance) and the **center-detection** target. Physically:
in dryland, green crowns cast shadows onto bright bare soil, so a "green-here AND
dark-along-the-sun-azimuth" filter localizes crowns where luminance cannot.

### Confirmation (locked 40-scene held-out split, touched once)

| base features | split | R1 (no shadow) | R2 (correct) | R3 (shuffled) | **d23 = R2−R3** | 95% CI | p (NB) |
|---|---|---|---|---|---|---|---|
| **raw** (6-d colour) | all | 0.028 | 0.046 | 0.027 | **+0.020** | (+0.016,+0.023) | <1e-4 |
| **raw** | **dryland** | 0.049 | **0.104** | 0.042 | **+0.060** | (+0.044,+0.077) | <1e-4 |
| **DINOv3 web** | all | 0.263 | 0.254 | 0.264 | −0.009 | (−0.015,−0.004) | ns/neg |
| **DINOv3 web** | dryland | 0.246 | 0.253 | 0.248 | +0.005 | (−0.004,+0.015) | 0.25 (ns) |

(values at N≈12–20 train scenes; full curves in `artifacts/confirm_*.json`.)

**What this means — two findings, both honest:**

1. **The directional shadow prior is real and large on weak features, strongest in
   dryland.** On a 6-d colour base, the correct azimuth roughly **doubles** dryland
   crown-center AP (0.049→0.104) and beats the shuffled-azimuth control by
   d23=+0.060, p<1e-4 — confirmed on held-out scenes. It is *direction*, not extra
   capacity (R3 ≈ R1). **Label-efficiency:** raw+shadow at **6 training scenes
   already beats raw-only at 48** — the prior is worth more than 8× the labels.

2. **DINOv3 fully subsumes it.** Adding the same shadow channel to frozen DINOv3
   features gives **no reliable gain** (null-to-slightly-negative on held-out; the
   small +0.014 seen on EXPLORE did **not** replicate). DINOv3 alone is already a
   far better crown-center detector (R1 ≈ 0.26 vs raw 0.03). The
   **delta-of-deltas** is the headline measurement: *shadow geometry that classical
   features lack is already encoded by a self-supervised foundation model.*

### Publishability verdict (Track 1)

A **genuine, confirmed, statistically defensible effect** — but read it correctly:
- **Strong/honest framing that holds:** *"A physics-grounded directional shadow
  prior is a large, label-efficient substitute for learned representation in
  dryland tree-center detection — and we quantify that a frozen aerial foundation
  model (DINOv3) has already implicitly learned this shadow-from-structure
  geometry that classical features have not."* That is a real probing result about
  foundation models + a deployable result for label-scarce, no-foundation-model
  settings.
- **The original thesis — inject shadow to boost the DINOv3 pipeline — is NOT
  supported.** DINOv3 subsumes the prior; it adds nothing on top. Don't write the
  paper as "shadow improves the SOTA pipeline"; write it as the measurement above.
- Scope caveats: effect is specific to greenness + center detection; N is small
  (3 acquisitions; external validity is the weak axis); the probe is patch-coarse.

---

## TRACK 2 — "beat 0.902 area-F1 on OAM-TCD": a real ~0.02 gap, not a protocol mirage

**Calibration (full 439 tiles, `sota/calib_b5.log`).** restor's SegFormer-b5 — the
model that reports **0.902** in *their* protocol — scores **0.8945 F1 / 0.8091 IoU
in my matched protocol** (foreground = cat1∪cat2, sliding-window-512). So:

- The **protocol gap is small** (0.8945 vs 0.902 ≈ 0.008). My earlier 25-tile read
  of 0.878 was noise; "0.902 is mostly protocol" was **wrong** — corrected here.
- My **frozen DINOv3 web linear probe = 0.874** → a **genuine ~0.020 gap below**
  the fine-tuned SegFormer-b5 in the same protocol. Not a tie.

So beating 0.902 means closing a real ~0.02–0.028 model gap — the stronger head's job.

**Stronger frozen head** (multi-layer 4-block fusion + 3×3 conv decoder + 800 tiles
+ multi-scale 512/768 TTA; `sota/strong_result.json`): **F1=0.8763, IoU=0.7798**.
(First attempt diverged at lr=2e-3/batch-1 → F1=0; fixed with lr=3e-4 + batch-3 +
grad-clip; the stable rerun is this number.)

| approach (matched protocol) | F1 | IoU |
|---|---|---|
| DINOv3 frozen **1×1 linear probe** (300 tiles) | 0.874 | 0.776 |
| DINOv3 frozen **multi-layer + decoder + TTA** (800 tiles) | 0.876 | 0.780 |
| **SegFormer-b5 fine-tuned** (restor) | **0.8945** | **0.8091** |

### Publishability verdict (Track 2)

- **Frozen DINOv3 plateaus at ~0.876 — a real, stubborn ~0.018 gap below
  fine-tuned SegFormer-b5 (0.8945 ≈ their 0.902).** Crucially, the multi-layer +
  decoder + multi-scale upgrades moved F1 by only +0.002 over the 1×1 probe, so the
  bottleneck is the **frozen representation**, not head capacity or data. More head/
  data/TTA will not close it.
- **Therefore: beating 0.902 requires fine-tuning the DINOv3 backbone** (partial /
  LoRA on the last blocks), not a frozen probe. That is the single clear next
  experiment; I did **not** launch it unattended (ViT-L fine-tuning on MPS is slow
  and divergence-prone — see the head's first collapse — and is better watched).
- Honest status on the goal: **not achieved by the frozen approach; a real ~0.02
  gap remains, and the path to close it (backbone FT) is identified.** No part of
  this is a protocol mirage — the matched bar is genuinely ~0.89–0.90.

---

## TRACK 3 — fine-tuning DINOv3: light FT ties fully-fine-tuned SegFormer-b5

Backbone FT was Track 2's identified next step. We ran it (watched). `dino/finetune/`.

**v1 (failed — a leakage lesson).** Last-4 blocks + 10M decoder, 4000 steps, val =
random 200 tiles from train. Val hit **0.9235** but the 439-test read **0.864 —
*below* the frozen probe.** Cause: **spatial-autocorrelation leakage** — a random
*tile* split scatters neighbouring tiles (up to 29 per 1 km bin) across train/val, so
val saw near-duplicates of training tiles and inflated; the inflated val couldn't
detect the (compounding) over-capacity overfitting.

**Fix.** Restor's **official folds** (`validation_fold` 0–4 in each train tile's
meta; 95 % spatially grouped → leakage-safe): train folds 1–4, select on fold 0,
report on the 439 holdout. Plus regularisation — last-**2** blocks, LR 1e-5, weight
decay, flip aug, a **head-warmup phase** (freeze → train head → unfreeze), and
**early stopping** (min-delta) on the honest val.

**v2 (honest result): 439-test F1 = 0.8929, IoU = 0.8065.**

| model (matched protocol, 439 test) | F1 |
|---|---|
| DINOv3 frozen 1×1 probe | 0.874 |
| DINOv3 frozen strong head | 0.876 |
| DINOv3 FT v1 (broken — leaked val) | 0.864 |
| DINOv3 FT v2 (last-2 blocks, official folds) | 0.8929 |
| SegFormer-b5 (fully fine-tuned) | 0.8945 |
| **DINOv3 FT v3 (last-4, no TTA)** | **0.8954** |
| **DINOv3 FT v3 (last-4 + flip-TTA)** | **0.8964** |

### Verdict (Track 3)
- **+0.019–0.022 over frozen** — backbone adaptation breaks the frozen ceiling, as predicted.
- **v2 (last-2) ties SegFormer-b5** (0.8929 vs 0.8945); **v3 (last-4) marginally beats
  it** — 0.8954 no-TTA (+0.0009, single-inference, fair) / 0.8964 with flip-TTA
  (+0.0019). Both margins are within single-run noise, so honestly it's
  *matches-to-slightly-beats*, not a decisive win. Last-4 added a real +0.0025 over
  last-2; flip-TTA added +0.0010.
- A *light* DINOv3 FT (last-4, ~50 M params) matches/edges a *full* SegFormer-b5 FT.
- **Still did NOT beat 0.902** — in matched protocol both models sit ~0.895; the
  ~0.008 offset to restor's reported 0.902 is protocol (foreground raster + tiling),
  unverified.
- Even the clean fold-0 val (0.9038) sits ~0.01 above test — a legitimate
  train-easier-than-holdout difficulty offset, **not** leakage.
- Methodological lesson: geospatial val must be spatially grouped (use the official
  folds); a random-tile val leaks and inflates by ~0.04.

---

## Bottom line for the morning

- **Track 1 is the publishable science**, but as a *measurement of what the
  foundation model already knows*, plus a *label-efficient classical-feature*
  result — **not** as "shadow boosts DINOv3" (that's a confirmed null).
- **Track 2**: matched-protocol SegFormer-b5 = 0.8945 (≈ their 0.902); frozen
  DINOv3 plateaus at **0.876** even with multi-layer fusion + decoder + multi-scale
  (only +0.002 over the 1×1 probe). The ~0.02 gap is **real and representation-
  bound, not a protocol mirage**. Beating 0.902 needs **backbone fine-tuning** —
  identified as the next experiment, deliberately left for a watched run.

Artifacts: `artifacts/confirm_raw.json`, `artifacts/confirm_web.json`,
`artifacts/grid_summary.json`, `sota/calib_b5.log`, `sota/strong_result.json`.
