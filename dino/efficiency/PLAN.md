# Overnight autonomous plan — pre-registration

Goal: find a **genuine, statistically defensible** result worth trying to publish.
Two tracks. Everything is contained to `dino/efficiency/`.

Methodology guardrails (decided BEFORE looking at any result):
- **No p-hacking.** Track 1 uses an **exploration / confirmation** split: all
  trail-following and config selection happen on the EXPLORE scenes; the single
  best pre-specified config is then tested **once** on the locked CONFIRMATION
  scenes. Only the confirmation result is reported as "the effect".
- **Unit of analysis = scene** (133 scenes; 3 acquisitions WON/BRU/NEON). Stats are
  Nadeau–Bengio corrected resampled t-test + sign-flip permutation + a seed-variance
  MDE floor (reused from `shadow_prior.stats`). Effect size + CI always, never a
  bare p. A precise null is a valid, reportable outcome.
- **Local only** (MPS/CPU). Frozen-feature linear probes keep Track 1 cheap enough
  for many seeds/folds (real power) and immune to the step1 detector-collapse.

## Track 1 — does a directional shadow prior improve label-efficiency of crown detection on frozen DINOv3 features?

Pipeline: tile→512px→DINOv3 (web ViT-L, the bake-off winner) frozen features
(32×32×C, cached) ⊕ shadow channel pooled to 32×32. Per-patch crown-occupancy
linear probe. Primary metric = patch-level Average Precision (stable, threshold-
free); secondary = point-detection F1 (impact translation).

Three rungs (architecture identical; only the input stack varies):
- R1 features only · R2 features ⊕ correct shadow · R3 features ⊕ within-acquisition-shuffled shadow.
- **Primary comparison R2 vs R3** (isolates direction info from added channel capacity).

Primary analysis = **label-efficiency curve**: vary N train scenes; the hypothesis
is the R2−R3 gain is positive and *larger at small N*, shrinking as labels grow
(physics substitutes for labels). Report the curve with CIs.

Bounded variant grid (the only trails I may follow, fixed up front):
- shadow cfg: aggregation {max, logsumexp}, n_channels {1,2}, brightness {luminance, greenness}, offset bracket {(2,20),(2,30),(4,40)}
- probe: {linear, small-MLP}
- features: {DINOv3-web, DINOv3-sat, raw-RGB-patch-stats}  → enables the **delta-of-deltas**: does shadow help MORE on raw features than on DINOv3 (how much does the FM already encode shadow geometry?)
- per-acquisition stratification (is the effect dryland-specific: WON/BRU vs temperate NEON?)

## Track 2 — convincingly beat 0.902 area F1 on OAM-TCD test (target 0.91+)

0. **Calibrate**: re-score cached `restor/tcd-segformer` through `dino/eval.py`.
   "Beat 0.902" only means something in a matched protocol; this defines the real bar.
1. Stronger but still-frozen DINOv3 head: multi-layer feature fusion + light decoder
   (vs 1×1) + full 5072-tile train + multi-scale/overlap/TTA inference.
2. If frozen plateaus < 0.91, partial/LoRA fine-tune (last blocks).
Report the matched-protocol number, not paper-vs-mine across protocols.

## Deliverable
`dino/efficiency/REPORT.md`: methodology, efficiency curve, confirmation result,
per-acquisition + delta-of-deltas, Track-2 calibrated SOTA number, honest nulls,
and a publishability verdict for each track.
