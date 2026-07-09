# Directional shadow prior for tree-crown instance segmentation

This repository builds an **input feature** and an **ablation harness** for testing
whether a sun-azimuth-derived directional shadow prior measurably changes tree-crown
instance-segmentation performance. It deliberately stops at a clean `(C, H, W)`
tensor stack: the detector (a PolarMask-style RetinaNet + FPN + polar/radial mask
head, initialised from DeepForest weights — see *Reproducibility*) is out of scope.

This README doubles as the seed for the paper's methods section, so the rationale is
written out in full. Where the implementation and this description could drift, the
code is the source of truth and the relevant function is named inline.

## Objective

We test whether a sun-azimuth-derived **directional** shadow prior, injected at the
input representation, produces a measurable and statistically defensible change in
tree-crown instance-segmentation performance. The honest target is a *credible
effect estimate with a confidence interval*: a precise null — "the shadow prior
changes mAP by 0.00 ± 0.4 points" — is a valid and reportable outcome. This is not a
hunt for `p < 0.05`. We fix the analysis (unit of analysis, folds, tests, and the
seed-noise floor) before looking at any comparison, and we report effect size and
interval regardless of which side of any threshold it lands on.

The applied objective is the Restor **OAM-TCD** benchmark (439-image test split) as
the dense-canopy regime, with an Australian arid-rangeland set as the sparse target
regime. The shadow rungs require a per-image azimuth, so the ablation is run on the
azimuth-annotated cohort (~130 images at time of writing); see *Validation protocol*
and *Threats to validity* for how that interacts with the benchmark and with
DeepForest-initialised fine-tuning.

## The feature

We are given the sun/shadow **azimuth only** — no solar elevation and no timestamps
from which to derive it. Shadow displacement is `d = h / tan(elevation)`; without
elevation, the crown height `h` (hence `d`) is unknown. We therefore do **not**
compute shadow length. We treat it as a nuisance and **marginalise it out** by
sweeping the offset (design decision #1).

Concretely, `compute_shadow_feature(rgb, azimuth_rad, cfg)`
([`shadow_prior/shadow_feature.py`](shadow_prior/shadow_feature.py)) is a **swept
directional matched filter**. For each pixel `p` it aggregates, over a bracket of
offsets `s` along the annotated azimuth `φ`, a *"lit-here AND dark-at-offset-s"*
score:

```
crown(p) = aggregate over s of   lit(p) · dark(p + s·φ̂)
```

where `lit = σ(gain·z)` and `dark = σ(−gain·z) = 1 − lit` are soft thresholds on a
**robustly standardised** brightness field `z` (median/MAD, with a std fallback for
near-flat tiles, clipped to `±z_clip`). Robust standardisation rather than a
min-max/percentile rescale is deliberate: a rescale collapses to a constant whenever
the bright/dark features cover a small fraction of the tile — exactly the small-crown
regime of the sparse target — and would silently zero the feature there.

- **Offsets** are `linspace(offset_min, offset_max, offset_steps)` pixels. This
  bracket is a **site prior**, passed via config and frozen across the whole
  experiment. It is never tuned on test data and never tuned per acquisition (doing
  either would leak test information into the feature).
- **Aggregation** is switchable (`cfg.aggregation`): `"max"` is the MAP-style choice
  (single best displacement); `"logsumexp"` is the soft marginalisation over the
  nuisance displacement (decision #1).
- **Brightness proxy** is configurable: `"luminance"` (Rec.601) by default, or
  `"greenness"` (excess green) for vegetation contrast.
- **Azimuth convention** (`cfg.azimuth_points_to`) is a flag, not an assumption: the
  annotated vector is taken to point along the shadow displacement (anti-solar) by
  default, or toward the sun (`"sun"`, internally a π rotation). Because getting the
  sign wrong silently inverts the prior, `shadow_prior.verify.recommend_convention`
  checks both signs against annotated crowns and reports which makes the crown
  response actually fire on real crowns — the convention is confirmed from data
  before any run, not guessed.
- **Channels** (`cfg.n_channels`): 1 emits the crown response; 2 additionally emits
  the **dual** "shadow" response `dark(p)·lit(p − s·φ̂)` (this pixel is itself in a
  cast shadow). We deliberately do **not** emit the arg-max offset as a channel —
  that would be *computing* shadow length, which decision #1 forbids.

The contribution is scoped as a **directional** shadow prior: direction is the
informative quantity, magnitude is integrated out. In the informed-ML taxonomy of
von Rueden et al. this is exogenous knowledge injected at the **input
representation** (a hand-designed feature channel), as opposed to in the
architecture, the loss, or the training data.

## Experimental design

A **three-rung ablation** with the architecture held identical across rungs; only the
input stack varies (design decision #3):

| Rung | Input stack | Role |
|------|-------------|------|
| 1 | RGB | baseline |
| 2 | RGB + correct shadow feature | treatment |
| 3 | RGB + **shuffled** shadow feature | control |

**Rung 2 vs rung 3 is the primary scientific comparison.** It isolates *direction
information* from *added channel capacity*: both rungs add the same number of
channels with the same marginal statistics, so any difference between them is
attributable to the azimuth being correct rather than to the network simply having
more input channels to fit. **Rung 2 vs rung 1 alone is not interpretable**, because
adding the shadow vector also supplies an absolute orientation reference that a plain
translation-equivariant CNN otherwise lacks; a rung-2-over-rung-1 gain could be that
orientation cue rather than shadow geometry. Rung 1 is still reported as the absolute
baseline, but the claim rests on rung 2 vs rung 3.

**Why the shuffle is within-acquisition.** Rung 3 permutes the azimuth labels
*within each acquisition group* (`shuffle_azimuths`, used by the dataset). A *global*
shuffle could be defeated by a model that first infers acquisition identity (sensor,
season, illumination signature) and then recovers the typical azimuth for that
acquisition — winning rung 3 without using shadow geometry at all. Stratifying the
permutation holds acquisition constant, so the only way rung 2 can beat rung 3 is
through the per-scene direction being right. (Caveat: singleton acquisitions cannot
be shuffled and are necessarily unchanged in rung 3; this weakens the control for
those scenes and is reported, not hidden.)

## Direction-variance has two sources

The azimuth variation in the data comes from two qualitatively different places, and
they are **not pooled into one effect**:

- **Across acquisitions** = genuinely different sun positions. This is the
  external-validity axis, but it is **confounded** by everything that co-varies with
  capture context — sensor, GSD, season, phenology, atmosphere. A shadow effect
  measured across acquisitions is real-world relevant but not cleanly attributable.
- **Within acquisition, via rotation augmentation** = synthetic direction-variance
  with the scene held fixed. Rotating the tile and recomputing the feature from the
  rotated azimuth varies direction while holding sensor/scene/phenology constant.
  This is the clean-mechanism axis.

These form a **nested argument** (clean mechanism inside, external validity outside),
reported separately. A mechanism that works within-acquisition but fails
across-acquisition is itself a finding about confounds, not noise to average away.

## Validation protocol

**Cohort split.** Rungs 2 and 3 require a per-image azimuth, which only the
azimuth-annotated cohort (~130 images) carries; the OAM-TCD 439-image test split does
not. The shadow comparison therefore lives **entirely on the ~130 cohort**, evaluated
by scene-clustered cross-validation: each fold fine-tunes the DeepForest-initialised
detector on the training scenes and evaluates on the held-out scenes, with fine-tune
and evaluation scenes kept disjoint by the fold guard (so the comparison is not
circular). The OAM-TCD 439 split is used for **RGB-only** absolute benchmark numbers
(rung 1) and for sanity that the fine-tuned detector is competitive; it cannot host
the shadow rungs. This bounds statistical power: with ~130 images the effective N is
the number of distinct *scenes* among them (reported once the data is in), and the
seed-variance MDE below states what effect that N can actually detect.

- **Unit of analysis = source scene** (design decision #4). The effective sample
  size `N` is the number of **distinct scenes**, stated explicitly in every result.
  Tile counts and rotation counts are **not** sample sizes — treating them as such is
  pseudoreplication (tiles from one scene are not independent) and rotation-inflated
  `N` (rotations of one scene are the same scene). These are the two named failure
  modes the code is built to prevent.
- **Scene-clustered folds.** `make_scene_clustered_folds`
  ([`shadow_prior/folds.py`](shadow_prior/folds.py)) is a GroupKFold-style split with
  the scene as the group key; every tile and every rotation of a scene is confined to
  one fold. The dataset additionally **refuses to construct** if any crown's records
  span more than one fold (`CrownTileDataset._assert_no_crown_fold_leakage`) — a hard
  leakage guard that raises, not warns. `FoldAssignment.effective_n()` returns the
  distinct-scene count per fold, which is what the statistics treat as `N`.
- **Leave-one-acquisition-out** (`leave_one_acquisition_out`) gives the
  generalisation number: each held-out acquisition contributes one paired estimate,
  so `N` for that comparison is the number of acquisitions.
- **Acquisition as a random effect.** The recommended top-level analysis is a
  mixed model with a random intercept for acquisition **and a random slope on the
  shadow effect**, so per-context consistency is visible. "Shadow helps in 4 of 5
  acquisitions and hurts in 1" is a reportable finding about where the prior works,
  not variance to be averaged into a single mean. (The harness in `stats.py` provides
  the paired per-fold/per-acquisition deltas that feed such a model; fitting the
  mixed model itself is left to the analysis layer, e.g. `statsmodels`/`lme4`.)

## Statistics

See [`shadow_prior/stats.py`](shadow_prior/stats.py). We report **effect size and a
confidence interval**, never a bare p-value.

- **Seed variance first.** `seed_variance` runs one fixed config across seeds and
  reports the seed-to-seed std and the **minimum detectable effect** (MDE) at a stated
  α/power. If the shadow effect is smaller than the MDE — or simply smaller than the
  seed std — the comparison is underpowered and no test rescues it. We establish this
  floor before trusting any rung comparison.
- **Primary test: Nadeau & Bengio (2003) corrected resampled t-test**
  (`corrected_resampled_ttest`). Cross-validation folds share training data, so the
  per-fold deltas are positively correlated and a plain paired t-test is
  anti-conservative (its p-values are too small). Nadeau & Bengio correct the
  variance by `(1/J + n_test/n_train)` instead of `1/J`, where the sizes are in the
  unit of analysis (scenes). The same corrected standard error defines the reported
  CI, so the interval and the test agree.
- **Backup test: permutation over paired deltas** (`permutation_test_paired`). An
  assumption-light sign-flip test; sign assignments are enumerated **exactly** when
  the number of folds is small. Reported alongside the corrected t-test so the
  conclusion does not hinge on the t-test's distributional assumptions.

## Metric sensitivity differs by domain (pre-specified)

The mechanism by which a shadow prior could help is **opposite** in the two regimes,
so the two are analysed separately and **never averaged** (the effects would cancel):

- **Dense canopy (OAM-TCD).** Crowns touch; the prior's value is in instance
  *separation*. Pre-specified secondary metric: merge / under-segmentation rate on
  high-density tiles.
- **Sparse rangeland.** Crowns are isolated; separation is not the bottleneck, so the
  prior's value is in *detection* of faint raised objects via their shadows.
  Pre-specified secondary metric: recall on small / low-contrast crowns.

These are declared as a stratified secondary hypothesis up front, not chosen after
seeing results.

## Threats to validity

- **Single global offset bracket.** `offset_min..offset_max` is one site prior for
  the whole experiment. Tuning it per acquisition, or on the test split, would leak
  test information into the feature. Mitigation: it is config, set from external site
  knowledge, and frozen; sensitivity to the bracket is reported as an ablation over
  fixed brackets, not a fit.
- **Recompute-not-rotate invariant.** The feature is always recomputed from the
  rotated tile and the rotation-updated azimuth, never produced by rotating a
  precomputed raster (decision #2). Rotating a raster would smear the feature by
  interpolation in a way correlated with the rotation angle, leaking an orientation
  cue. Mitigation: enforced in `CrownTileDataset` (`_featurize(..., post_rotation=True)`)
  and locked by tests (`test_rotating_precomputed_raster_leaks_interpolation`,
  `test_recompute_required_when_azimuth_not_updated`).
- **Azimuth-only scoping.** We claim a *directional* prior, not a shadow-length one;
  magnitude is integrated out. Any result is scoped to direction.
- **Azimuth sign convention.** A flipped convention (sun- vs shadow-pointing) would
  silently invert the prior and could masquerade as "shadow hurts". Mitigation:
  `azimuth_points_to` is a config flag confirmed empirically against annotated crowns
  (`verify.recommend_convention`) before the experiment, not assumed.
- **Across-acquisition confounds.** Sensor/season/phenology co-vary with sun
  position, so the across-acquisition axis cannot attribute a change to shadows
  alone. Mitigation: the within-acquisition (rotation) axis isolates the mechanism;
  the two axes are reported separately, not pooled.
- **Azimuth availability bounds the ablation cohort.** Rungs 2 and 3 require a
  per-image azimuth, so they can only be evaluated on azimuth-annotated images. The
  full OAM-TCD test split can host the RGB-only rung and absolute benchmark numbers,
  but the shadow comparison lives on the annotated cohort, cross-validated at the
  scene level. Fine-tuning data and evaluation data must stay disjoint at the scene
  level (enforced by the fold guard) so the comparison is not circular.

## Reproducibility

- The full configuration (`ShadowFeatureConfig`, `DatasetConfig`, `FoldConfig`) is a
  frozen dataclass and is dumped per run (`ShadowFeatureConfig.to_json()`).
- Seeds are logged; augmentation rotation is a deterministic, seeded function of the
  sample index.
- Fold assignments **and the effective-N per fold** are persisted as artifacts
  (`FoldAssignment.save_json()`), alongside results.
- **DeepForest initialisation.** The detector reuses DeepForest's pretrained
  RetinaNet (ResNet-50 + FPN) weights to avoid full pretraining, then fine-tunes on
  the azimuth-annotated cohort. Because the emitted stack is **RGB-first** (channels
  0–2 RGB, shadow channels appended), the pretrained 3-channel input stem can be
  reused by *inflating* the first conv with extra zero-initialised input channels for
  the shadow feature: at initialisation the RGB response is identical to stock
  DeepForest, and the shadow channels start from zero influence and are learned during
  fine-tuning. (DeepForest is a box detector; the polar/radial mask head is added on
  top and trained — that head, and the inflation itself, live in the out-of-scope
  detector code, but the channel ordering here is what makes them clean.)

## Layout

```
shadow_prior/
  config.py          # frozen dataclasses; every tunable, no inline magic numbers
  geometry.py        # azimuth<->vector + rotation update (single source of truth)
  shadow_feature.py  # compute_shadow_feature, shuffle_azimuths  (pure, no I/O)
  verify.py          # recommend_convention: confirm azimuth sign against crowns
  folds.py           # scene-clustered folds, leave-one-acquisition-out, effective N
  dataset.py         # CrownTileDataset: rotate->update azimuth->recompute; rungs
  stats.py           # Nadeau-Bengio, permutation, seed-variance / MDE
tests/               # peak response, recompute invariant, shuffle, folds, leakage, stats, convention
```

## Install & test

```bash
python3.10 -m venv .venv && . .venv/bin/activate
pip install numpy scipy pytest torch
pytest -q
```
