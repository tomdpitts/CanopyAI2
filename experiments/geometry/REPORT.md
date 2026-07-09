# Deterministic crown→shadow filter — proof-of-mechanism report

**Question.** In arid tree-crown detection, does injecting the *correct* global
solar azimuth via a **deterministic, zero-parameter** crown→shadow geometric
filter improve detection precision, and does a *wrong* azimuth degrade it? The
sought result is the asymmetry (correct helps, wrong hurts).

**Verdict (one line).** The geometric signal is **real, directional, and
discriminative** — but it is **redundant with, and weaker than, ordinary
appearance**, so it does **not** improve precision over the raw detector at any
fusion strength. This is the *CORRECT ≈ NONE* branch of the decision rule → **do
NOT build the learned conditioning head expecting a precision gain.** It
independently reproduces the earlier "shadow prior is subsumed by a strong
appearance model" finding via a completely different, parameter-free route.

Seed = 0 (numpy `default_rng(0)`) throughout. All numbers regenerable from
`experiments/geometry/{gdata,candidates,gfilter,evaluate,diagnostics}.py`.

---

## Data / cohort
- Combined `phase22X_train.csv` + `phase22X_val_fixed.csv` →
  `data/finetune/phase22X_combined.csv` (1847 rows, 166 images).
- **Azimuth cohort = 131 images** with both GT crowns and a usable shadow
  direction: **WON 41, BRU 53, NEON 37** (1685 GT crowns, median crown √area
  ≈ 56 px). 2 NEON placeholder azimuths (exactly 0°, 270°) and 13 NEON images
  with no azimuth were excluded.
- WON/BRU are synthetic rotations → azimuth spans the full circle, **decorrelated
  from site** (so azimuth cannot be a site shortcut; WON/BRU are the clean
  controls). NEON azimuths are real but **clustered near solar-noon (~270–345°)**,
  so a within-NEON shuffle does not decorrelate direction — for NEON the valid
  WRONG condition is a 180°/90° rotation, not a shuffle.

## Conventions verified (not assumed)
- Shadow direction taken directly from `shadow_x, shadow_y` via
  `shadow_prior.geometry.vector_to_azimuth` (the convention previously fixed and
  verified against WON base=215°, BRU base=118°). No sun elevation is used or
  needed — magnitude is marginalised over a distance bracket; direction is the
  prior.
- **Rotation padding confound handled.** WON tiles are ~10% pure-black rotation
  triangles (BRU ~0.5%, NEON 0%). A validity mask excludes padding (+3 px erosion)
  so the luminance filter never reads padding as shadow.

## Pipeline (fully parameter-free)
1. **Candidates** — training-free multi-scale DoG blob detector on darkness
   (−luminance) and greenness, low threshold → high recall + rich shadow-FP pool.
   Pool recall ≈ **1.00** (WON/BRU) / 0.98 (NEON); ~35.8k candidates, ~55% of FPs
   are dark (shadow-like). Score = DoG response.
2. **Filter** — signed directional contrast along the anti-solar vs solar ray,
   `delta = mean_lum(+u) − mean_lum(−u)` over d∈[15,60] px; keep-prob
   `g = expit(−2·delta)`. Real crown → shadow ahead / lit behind → delta<0 → g high;
   cast-shadow FP → dark crown behind / lit soil ahead → delta>0 → g low. (One
   signed quantity implements both shadow-presence and self-shadow rejection, and
   is domain-agnostic to the arid "dark crown on bright soil" inversion.)
3. **Three arms** re-score the *same* candidates through the *same* filter; only
   the azimuth differs (NONE g=1 / CORRECT / WRONG = flip180, rot90, shuffle).
   Because arms share one candidate pool and the filter has no learned parameters,
   the CORRECT-vs-WRONG asymmetry cannot be manufactured by the detector.

---

## Results

### 1. Signal exists and is directional (three independent measures)
- **Directional-contrast probe** (GT crowns, `signal_probe.py`):
  correct Δ = **−0.378** (95% CI −0.426,−0.330); shuffled −0.041 (−0.090,+0.008);
  flipped +0.378 (exact mirror). Per site: WON −0.29, **BRU −0.90**, NEON −0.59.
- **Three-arm PR** (`evaluate.py`, scene-bootstrap CIs): at every recall,
  NONE − WRONG_flip = **+0.32 … +0.55** (CI excludes 0). CORRECT ≫ all WRONG arms.
- **TP-vs-FP AUC of the cue** (`diagnostics.py`; NONE = 0.500 by construction):
  ALL correct **0.738**, shuffle 0.513, flip **0.262**, rot90 0.500.
  WON **0.794**, BRU **0.624**, NEON **0.455 (inverted)**.

### 2. …but it does NOT beat the raw detector, at any fusion strength
| arm | AP | P@R0.5 | P@R0.7 |
|---|---|---|---|
| **NONE** | **0.579** | **0.622** | **0.470** |
| CORRECT | 0.531 | 0.582 | 0.434 |
| WRONG_rot90 | 0.381 | 0.369 | 0.191 |
| WRONG_flip | 0.207 | 0.083 | 0.050 |

CORRECT − NONE is **negative** at every recall (−0.04 to −0.10, CI excludes 0 on
the wrong side). Symmetric fusion sweep `raw·exp(−λ·delta)`: **best λ = 0 in every
domain** (any λ>0 lowers AP). See `claude_outputs/geometry/pr_three_arm.png`,
`lambda_sweep.png`.

### 3. Why: the cue is redundant with appearance
The raw candidate score already separates crowns from shadow-FPs **much** better
than the geometry does, so adding geometry only injects noise:

| scope | AUC_raw | AUC_−delta | AUC_sum(equal) | Δ vs raw |
|---|---|---|---|---|
| ALL | **0.948** | 0.738 | 0.931 | −0.017 |
| WON | **0.964** | 0.794 | 0.956 | −0.007 |
| BRU | **0.909** | 0.624 | 0.842 | −0.066 |
| NEON | **0.853** | 0.455 | 0.811 | −0.042 |

Even in the hardest subset (dark arid candidates) AUC_raw 0.962 vs sum 0.949.

---

## Diagnosis (the null is not an artifact)
The decision rule asked, if null, to rule out bad provenance / orientation / weak
shadows / tiny crowns. None apply:
- **Azimuth provenance** — convention verified three independent ways; WON/BRU
  controls behave perfectly (shuffle→chance, flip→inverse).
- **North-up / orientation** — per-tile `shadow_x/y` are already rotation-consistent;
  no reliance on north-up.
- **Shadows too weak / crowns too small** — probe Δ is large and crowns (~56 px)
  resolve their shadows fine; the cue's AUC is 0.62–0.79 in arid domains.

The real cause is **redundancy**: appearance already solves the crown/shadow
separation the geometry was meant to supply. NEON additionally **inverts** — the
arid "dark crown / dark shadow / bright soil" model does not transfer to dense
forest + buildings with clustered illumination.

## Recommendation
- **Do not build the learned azimuth-conditioning head** expecting a precision
  gain on this data. A frozen DINOv3 backbone is an even stronger appearance model
  than the blob score used here, so the geometry will be subsumed *a fortiori*
  (consistent with the prior overnight result).
- The signal is genuine, so the *only* regime where conditioning could pay off is
  one where appearance genuinely cannot separate crowns from shadows
  (AUC_appearance → ~0.5 on the confusable set). We did not reach that regime even
  in BRU (AUC_raw 0.909); with DINOv3 it is even less likely. If pursued, that
  hypothesis must be tested *first* — before any conditioning is built.
