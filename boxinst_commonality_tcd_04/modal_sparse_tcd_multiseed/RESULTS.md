# Zero-shot on unseen open-canopy TCD — results

**The settled phase-4 model, unmodified, on 236 open-canopy tiles it has never seen.**
Same checkpoints (`det_phase4_L24_s{0,1,2}.pt`), same masker (`em_model_4p_fix.npz`, α=0.3/κ×1.6),
same `mask_thr` 0.25. No retraining, no refit, no re-tuning.

## Headline

| | sparse 236 (3-seed) | official 439 (3-seed) | Δ | |
|---|---|---|---|---|
| mask mAP50 | **0.656 ± 0.010** | 0.630 ± 0.005 | **+0.026** | 2.3σ |
| mask mAP50-95 | **0.2885 ± 0.0054** | 0.257 ± 0.001 | **+0.032** | 5.7σ |
| box mAP50 | **0.631 ± 0.011** | 0.603 | **+0.028** | 2.6σ |

Per-seed sparse 0.6699 / 0.6476 / 0.6504 · per-seed 439 0.6250 / 0.6358 / 0.6294.

⚠ **Seed 0 alone overstates this.** Seed 0 is the *high* seed on the sparse slice and the *low*
seed on the 439, so a seed-0-only read gives +0.045 — nearly double the true +0.026. AP50 at 2.3σ
over 3 seeds is suggestive, not settled; the mAP50-95 gain (5.7σ) is the robust one.

## The aggregate number is misleading — read the per-biome table

Mask mAP50, 3-seed, vs the 439 band. `crowns/tile` is the mean over the unseen pool for that biome.

| biome | n | crowns/tile | mask mAP50 | Δ vs 439 |
|---|---|---|---|---|
| 13 Desert & Xeric Shrubland | 24 | 86.9 | 0.7141 ± 0.0233 | **+0.084** |
| 12 Mediterranean Forest, Woodland & Scrub | 56 | 103.8 | 0.7102 ± 0.0017 | **+0.080** |
| 8 Temperate Grassland, Savanna & Shrubland | 31 | 66.6 | 0.6576 ± 0.0099 | +0.028 |
| 7 Trop/Subtrop Grassland, Savanna & Shrubland | 102 | 43.0 | 0.6232 ± 0.0154 | **−0.007** |
| 9 Flooded Grassland & Savanna | 7 | 31.9 | 0.5439 ± 0.0154 | −0.086 |
| **10 Montane Grassland & Shrubland** | 16 | 16.2 | **0.1736 ± 0.0225** | **−0.456** |

**Spearman ρ(crowns/tile, mask AP50) = 0.943.** Performance tracks crown density almost
monotonically, and the box column moves in lockstep (masker-invariant → this is *detection*
behaviour, not a box→mask artifact).

**So the model does not handle sparse canopy well. It handles well-separated crowns well.**
The aggregate +0.026 is carried entirely by biomes 12 and 13 — which are labelled "sparse" by
geography but are the crown-*densest* in the slice (chaparral, olive/oak woodland; visible in the
contact sheet). Archetypal savanna (biome 7, the largest group at 102 tiles) sits at **parity**
with closed-canopy forest. The genuinely sparse biomes degrade, and **montane grassland collapses
to 0.174 — a 3.6× drop, consistent on all three seeds (0.1786 / 0.1439 / 0.1982).**

That collapse lands squarely in the dryland regime the AusDryland / SavannaTree thread targets.

## Head-to-head vs DetecTree2 — same 236 tiles, same scorer

> ⛔ **SUPERSEDED 2026-09-03.** The DetecTree2 checkpoint used throughout this document
> (`model_best_s0.pth`) was stopped by a budget cap at 2.39 epochs while still improving, and had
> three further defects, all understating it. It was retrained to detectree2's own early-stopping
> rule (`model_best_s0v2.pth`, 6.0 epochs) and re-run on this slice. **Every DetecTree2 figure
> below is historical** — current sparse numbers are in `../results_439/sparse236/README.md`;
> the account is in `../detectree2_baseline/RETRAIN_V2.md`.

Fully-supervised **DetecTree2** (Mask R-CNN R101-FPN), the checkpoint fine-tuned on the SAME
792/108 tiles (superseded `model_best_s0.pth`, 0.5345 on the 439 — see the coverage-bug note
below), run zero-shot on this slice. Its 900
training tiles are excluded from the slice by construction, so it is a valid transfer test for it
too. Both models scored by the identical `compare_subsets.py` (`evaluate._greedy_ap` +
canopy-ignore), which was cross-checked against the Modal-side scorer to 4 dp.

Mask mAP50. Ours is the 3-seed band; DetecTree2 is single-seed (s0).

| cut | n | OURS | DetecTree2 | Δ |
|---|---|---|---|---|
| all | 236 | **0.6560** | 0.5672 | +0.089 |
| **scene_clean** | 106 | **0.6787** | 0.6309 | **+0.048** |
| train_pool | 220 | **0.6557** | 0.5680 | +0.088 |
| scene_clean ∩ train_pool | 90 | **0.6788** | 0.6359 | +0.043 |
| biome 13 Desert & Xeric | 24 | **0.7141** | 0.6006 | +0.114 |
| biome 12 Mediterranean | 56 | **0.7102** | 0.6344 | +0.076 |
| biome 8 Temperate Grassland | 31 | **0.6576** | 0.5106 | +0.147 |
| biome 7 Trop/Subtrop Savanna | 102 | **0.6231** | 0.5358 | +0.087 |
| biome 9 Flooded Grassland | 7 | **0.5440** | 0.4716 | +0.072 |
| **biome 10 Montane Grassland** | 16 | 0.1735 | **0.2651** | **−0.092** ← DetecTree2 wins |

Our margin is **+0.089 here vs +0.096 on the 439** (both now full-coverage), i.e. essentially
unchanged — slightly *narrower* on open canopy, not wider. Both models improve on this slice over
their own 439 result (ours +0.026, DetecTree2 +0.033), which independently supports the "open canopy
is easier" reading rather than anything specific to our method — DetecTree2 in fact gains slightly
more. On the honest `scene_clean` cut our margin narrows to +0.048, about half the aggregate figure.

**The exception is the finding.** In biome 10 — the sparsest biome, and the only place our method
collapses — **DetecTree2 beats us, 0.265 vs 0.174**. Montane grassland is DetecTree2's worst biome
too (it degrades from ~0.60 to 0.27), but it degrades *far less*. That partially resolves the
"is it the metric?" caveat above: if the low-density collapse were purely pooled-AP behaviour at
small object counts, DetecTree2 would collapse alongside us. It does not. **Something specific to
our detector fails on genuinely sparse crowns**, and a conventional supervised Mask R-CNN is more
robust there. That is the single most actionable result in this file.

### Prediction-coverage bug — found, fixed, and the 439 restated (2026-08-18)

`build_coco.build_tile` dropped any 1024 subtile with no crown and no canopy annotation. Correct
for training ("empty-sky subtiles skew background stats") — but it was also applied to the TEST
set, so DetecTree2 was never inferred on those regions and never charged for false positives there,
while our method (whole-tile, no subtiling) always is.

Measured on this slice: **16.4% of tile area uncovered on average** (median 0%, so concentrated —
32 of 236 tiles >50% uncovered), and worst in exactly the biomes that carry the finding: **34.4% in
biome 10**, 19.9% in biome 7. Left unfixed it would have handed DetecTree2 a free pass precisely
where the comparison matters.

Fixed via `build_tile(..., keep_empty=True)` for prediction builds (default `False` keeps training
and the existing 439 build byte-identical). The sparse run above uses **full 9/9 subtile coverage**
(2,124 subtiles).

**The 439 has since been re-run at full coverage (2026-08-18), so both rows are now like-for-like.**
DetecTree2's 439 figures dropped 0.5448 → **0.5345** mask mAP50 (mAP50-95 0.2277 → 0.2235, box
0.5392 → 0.5290): +5,380 predictions in the previously-invisible regions, with recall IDENTICAL to
4 dp and precision falling 0.6004 → 0.5847 — pure false positives, exactly the expected signature.
Ours are unchanged (no subtiling). Those artifacts (`preds_dt2_s0_fullcov.json`,
`results_dt2_s0_fullcov.json`) were themselves superseded by the 2026-09-03 retrain and deleted
from the working tree; they remain in git history. Full account in
`phase4/README.md` § DetecTree2 baseline.

**Scorer hardening (same date).** Both scorers previously selected `[t for t in sorted(gt) if t in
preds]`, silently dropping a tile a model produced nothing for instead of scoring it as zero recall.
Now every GT tile is scored, missing ones as zero-prediction, with a warning. No published number
changed — every run had predictions for all tiles.

### Restor Mask R-CNN — excluded, and why

`restor/tcd-mask-rcnn-r50` cannot be evaluated on this slice. Its card states it was *"trained on
all `train` images"*, and **220 of the 236 tiles here come from the HF `train` split** — it has
seen 93% of this test set. Only the 16 tiles from the official 439 would be valid holdout for it,
which is far too few. This is structural: unseen open-canopy tiles only *exist* in the train split,
since the official 439 contains just 16 of them. The paper's 0.432 stays where it is — on the 439,
under Restor's own protocol, still never re-scored under our canopy-ignore scorer.

## Exclusion held, and scene overlap turned out not to matter

| cut | n | mask mAP50 | box mAP50 | Δ mask |
|---|---|---|---|---|
| all | 236 | 0.6560 ± 0.0099 | 0.6310 ± 0.0108 | +0.026 |
| **scene_clean** | 106 | **0.6787 ± 0.0042** | 0.6580 ± 0.0080 | **+0.049** |
| train_pool | 220 | 0.6557 ± 0.0103 | 0.6304 ± 0.0113 | +0.026 |
| scene_clean ∩ train_pool | 90 | 0.6788 ± 0.0042 | 0.6556 ± 0.0085 | +0.049 |

The slice is tile-disjoint from training but not scene-disjoint (130 tiles share a 300 m ortho
cluster with a training tile). **Dropping every such tile makes the result better, not worse**
(+0.049 vs +0.026), on all three seeds and with a *tighter* σ (0.0042) than the 439's own band.
Leakage would have pushed the other way.

Nor is that a composition artifact: `scene_clean` is *depleted* in the high-scoring biomes
(biome 13 is 25% clean, biome 12 38%) and *enriched* in the lower-scoring biome 7 (54%) — the mix
works against the gain and it still gains. The overlapping tiles differ by scene structure, not
difficulty: they come from 30 large orthos (median 7 tiles each) vs 65 small ones (median 2) for
the clean set, with matched no-data (86.2% vs 86.1% valid) and crown count (64.2 vs 62.2/tile).

**Conclusion: sharing a scene with training conferred no measurable advantage here, so tile-id-only
exclusion was sufficient — shown empirically, not assumed.** The 16 tiles from the official 439
(which tuned `mask_thr` and α/κ) are immaterial: `train_pool` ≡ `all` to 3 decimal places.

## Caveats

- **Biomes 9 and 10 are small** (7 and 16 tiles; 223 and 260 crowns). The *ranking* is solid and
  the biome-10 collapse reproduces on every seed, but those two AP values are individually noisy.
- **Pooled AP is harsher at low object counts** — with few GT per tile, each false positive costs
  proportionally more precision. Part of the low-density degradation may be metric behaviour rather
  than model failure. The saved predictions support a per-tile FP-rate check to separate these;
  **not yet run.**
- GT density varies 6× across biomes (16.2 → 103.8 crowns/tile), so cross-biome AP comparison is
  not like-for-like.
- Coverage figures are *labelled* coverage; a tile can read sparse because its closed part went
  unlabelled.

## Provenance

Slice: `data/tcd_sparse` (236 tiles, 14,937 GT crowns, biomes 7/8/9/10/12/13), built and gated by
`build_slice.py` — see `data/tcd_sparse/README.md`. All 236 pixel-verified against HF on Modal.
Feature parity gate cos = 0.86003, the exact value phase-4 recorded.

The subset scorer reuses phase-4's `_full_metrics` and was validated by re-scoring the saved 439
predictions: reproduces the published 0.6203 / 0.2436 / 0.605 / 0.6692 **exactly**.

Artifacts on Volume `tcd-sparse-vol`: `out/results_sparse_L24_s{0,1,2}.json`,
`out/preds_sparse_L24_s{0,1,2}/preds.json`, `out/band_sparse.json`, `out/subsets_sparse.json`.
Local pulls: `subsets_sparse_s0.json`, `subsets_sparse_band.json`.
Cost: ~$0.5 extract + ~$2/seed eval. Nothing was written to `tcd04-phase4-vol`.
