# Zero-shot on unseen open-canopy TCD — results

**The settled phase-4 model, unmodified, on 236 open-canopy tiles it has never seen.**
Same checkpoints (`det_phase4_L24_s{0,1,2}.pt`), same masker (`em_model_4p_fix.npz`, α=0.3/κ×1.6),
same `mask_thr` 0.25. No retraining, no refit, no re-tuning.

## Headline

Ranked by `s' = s · bimod · pfg_mean` (`confidence/`), the deployed ranking key — the posterior
product is part of the method, not an add-on, so every AP here is the product-ranked arm. The
base-ranking (heatmap score alone) figures this file previously carried are in git history.

| | sparse 236 (3-seed) | official 439 (3-seed) | Δ |
|---|---|---|---|
| mask mAP50 | **0.6913 ± 0.0095** | 0.6625 ± 0.0012 | **+0.029** |
| box mAP50 | **0.6668 ± 0.0117** | — | — |

Per-seed sparse 0.7018 / 0.6831 / 0.6891. Positive on every seed.

## The aggregate number is misleading — read the per-biome table

Mask mAP50, 3-seed, vs the 439 band. `crowns/tile` is the mean over the unseen pool for that biome.

| biome | n | crowns/tile | mask mAP50 | Δ vs 439 |
|---|---|---|---|---|
| 13 Desert & Xeric Shrubland | 24 | 86.9 | 0.7314 ± 0.0147 | **+0.069** |
| 12 Mediterranean Forest, Woodland & Scrub | 56 | 103.8 | 0.7265 ± 0.0018 | **+0.064** |
| 8 Temperate Grassland, Savanna & Shrubland | 31 | 66.6 | 0.6860 ± 0.0097 | +0.024 |
| 7 Trop/Subtrop Grassland, Savanna & Shrubland | 102 | 43.0 | 0.6809 ± 0.0127 | +0.018 |
| 9 Flooded Grassland & Savanna | 7 | 31.9 | 0.6310 ± 0.0134 | −0.032 |
| **10 Montane Grassland & Shrubland** | 16 | 16.2 | **0.2768 ± 0.0421** | **−0.386** |

**Spearman ρ(crowns/tile, mask AP50) = 0.943.** Performance tracks crown density almost
monotonically, and the box column moves in lockstep (masker-invariant → this is *detection*
behaviour, not a box→mask artifact).

**So the model does not handle sparse canopy well. It handles well-separated crowns well.**
The aggregate +0.029 is carried by biomes 12 and 13 — which are labelled "sparse" by geography but
are the crown-*densest* in the slice (chaparral, olive/oak woodland; visible in the contact sheet).
Archetypal savanna (biome 7, the largest group at 102 tiles) sits slightly *above* closed-canopy
forest. The genuinely sparse biomes still degrade, and **montane grassland collapses to 0.277 — a
2.4× drop, consistent on all three seeds (0.2886 / 0.2301 / 0.3118).**

That collapse lands squarely in the dryland regime the AusDryland / SavannaTree thread targets.

## Head-to-head vs DetecTree2 — same 236 tiles, same scorer

> **Restated 2026-09-03.** The DetecTree2 checkpoint originally used here (`model_best_s0.pth`)
> was stopped by a budget cap at 2.39 epochs while still improving, and had three further defects,
> all understating it. It was retrained to detectree2's own early-stopping rule
> (`model_best_s0v2.pth`, 6.0 epochs) and re-run on this slice; every figure below is the
> converged model (`preds/dt2_sparse_s0v2.json` → `cuts_dt2_v2.json`). The superseded column is in
> git history; the account is in `../detectree2_baseline/RETRAIN_V2.md`.

Fully-supervised **DetecTree2** (Mask R-CNN R101-FPN), the checkpoint fine-tuned on the SAME
792/108 tiles (`model_best_s0v2.pth`, 0.5969 on the 439), run zero-shot on this slice. Its 900
training tiles are excluded from the slice by construction, so it is a valid transfer test for it
too. Both models scored by the identical `compare_subsets.py` (`evaluate._greedy_ap` +
canopy-ignore), which was cross-checked against the Modal-side scorer to 4 dp.

Mask mAP50. Ours is the 3-seed product-ranked band; DetecTree2 is single-seed (s0v2).

| cut | n | OURS | DetecTree2 v2 | Δ |
|---|---|---|---|---|
| all | 236 | **0.6913 ± 0.0095** | 0.6281 | +0.063 |
| train_pool | 220 | **0.6908 ± 0.0100** | 0.6281 | +0.063 |
| biome 13 Desert & Xeric | 24 | **0.7314** | 0.6469 | +0.085 |
| biome 12 Mediterranean | 56 | **0.7265** | 0.6906 | +0.036 |
| biome 8 Temperate Grassland | 31 | **0.6860** | 0.5995 | +0.087 |
| biome 7 Trop/Subtrop Savanna | 102 | **0.6809** | 0.5910 | +0.090 |
| biome 9 Flooded Grassland | 7 | **0.6310** | 0.5773 | +0.054 |
| **biome 10 Montane Grassland** | 16 | 0.2768 | **0.3250** | **−0.048** ← DetecTree2 wins |

The retrain moved DetecTree2 +0.061 on this slice (0.5672 → 0.6281), so our margin narrows from
+0.089 to **+0.063** — still wider than the +0.029 we gain over our own 439 result, and wider than
the +0.066 margin on the 439 itself. Both models improve on this slice over their own 439 figure
(ours +0.029, DetecTree2 +0.031 against its converged 0.5969), which independently supports the
"open canopy is easier" reading rather than anything specific to our method.

**The exception is still the finding, and it survives the retrain.** In biome 10 — the sparsest
biome, and the only place our method collapses — **DetecTree2 beats us, 0.325 vs 0.277**. Montane
grassland is DetecTree2's worst biome too, but it degrades far less (~0.60 → 0.33 against our
0.66 → 0.28). If the low-density collapse were purely pooled-AP behaviour at small object counts,
DetecTree2 would collapse alongside us. It does not. **Something specific to our detector fails on
genuinely sparse crowns**, and a conventional supervised Mask R-CNN is more robust there. That is
the single most actionable result in this file.

Two things the restatement changed. The gap is **narrower than previously recorded** (−0.048, not
−0.092): the posterior product lifts our biome-10 figure 0.174 → 0.277 more than the retrain lifts
DetecTree2's 0.265 → 0.325, so the reranker recovers roughly half the deficit. And biome 9, which
under the old base-ranking numbers looked like a second loss, is not one — we lead it by +0.054.
The failure is confined to biome 10.

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
since the official 439 contains just 16 of them. Its 439 figure is unaffected by that exclusion:
Restor has since been re-run and re-scored under our canopy-ignore scorer at **0.626** CN mask
AP50 (`../PROTOCOL_439.md`), superseding the 0.432 this section previously cited as never
re-scored.

## Exclusion held, and scene overlap turned out not to matter

> Numbers in this section are the **base-ranking** arm (heatmap score alone), the state in which
> the scene-overlap check was run. It is a check on the slice's construction, not a performance
> claim, and the conclusion is a ranking-invariant property of which tiles are in which cut — but
> do not read these AP values alongside the product-ranked figures above.

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

Artifacts behind the current numbers (2026-09-03):

| file | contents |
|---|---|
| `preds/ours_prod_s{0,1,2}.json` | product-ranked LACE predictions, `apply_product.py` over `preds/ours_s*.json` with `confidence/em_post_sparse_s*.json`. The detection set and the masks are unchanged from the base files; only `scores` differ. |
| `cuts_ours_prod.json` | the cuts above, 3 seeds |
| `preds/dt2_sparse_s0v2.json` | converged DetecTree2 |
| `cuts_dt2_v2.json` | its cuts |
| `cuts_dt2.json` | superseded DetecTree2 cuts, kept for the deltas quoted above; its predictions file no longer exists |

The `all` cut of `cuts_ours_prod.json` reproduces the published 0.6913 ± 0.0095 exactly, so the
greedy scorer used for the cuts and the pycocotools protocol used for `results_439/sparse236/`
agree to 4 dp for LACE. They do **not** for DetecTree2 — 0.6281 greedy against 0.6118 pycocotools
on the same predictions. Quote the pycocotools figure in the paper and the greedy one only
within this file, where both columns come from the same scorer.

Artifacts on Volume `tcd-sparse-vol`: `out/results_sparse_L24_s{0,1,2}.json`,
`out/preds_sparse_L24_s{0,1,2}/preds.json`, `out/band_sparse.json`, `out/subsets_sparse.json`.
Local pulls: `subsets_sparse_s0.json`, `subsets_sparse_band.json`.
Cost: ~$0.5 extract + ~$2/seed eval. Nothing was written to `tcd04-phase4-vol`.
