# Per-tile AP inflates as tiles shrink — OAM-TCD 439 (2026-08-21)

**Corrects a conclusion in `../aggregation_sensitivity/RESULTS.md`.** That study compared
pooled vs per-tile AP averaging at *our* granularity — whole 2048 px tiles, ~70 GT crowns
each — found box AP50-95 essentially unmoved (0.2555 → 0.2553 for seed 0), and concluded
that aggregation "accounts for ~0.002 of a ~0.19 gap: essentially none of it."

**That conclusion was wrong**, because tile size and tile aggregation are not separable.
SelvaMask does not average over 2048 px tiles; it averages over **1024 px subtiles at 0.5
overlap**. Per-tile AP is a *different estimator at every granularity*, and 2048 px is the
one granularity where it happens to be a no-op.

## Mechanism

Per-tile AP normalises precision **within** a tile. A subtile holding 1 GT crown and 20
false positives scores **AP = 1.0** provided the true positive outranks that subtile's own
FPs — those 20 FPs are never weighed against any other tile's true positives. Pooled AP
charges every one of them against the global ranking.

So as GT-per-tile falls, per-tile AP degenerates from "average precision" toward "did the
top-ranked box in this tile hit", and becomes progressively **blind to false positives**.
That matters most for a detector with a large FP surface: ours emits ~360 boxes per 2048
tile against ~70 GT crowns.

## Sweep — box AP, canopy-ignore on, uncapped maxDets

Identical saved predictions throughout; only the subtile grid over which per-tile APs are
averaged changes. Seed 0.

| subtile grid | GT/subtile | OURS AP50 | OURS AP50-95 | DT2 AP50 | DT2 AP50-95 |
|---|---|---|---|---|---|
| **pooled** (published) | — | 0.6050 | **0.2555** | 0.5290 | **0.2425** |
| 2048 (no subtiling) | 70.2 | 0.5770 | 0.2553 | 0.5194 | 0.2659 |
| **1024 @ 0.5 ov — theirs** | 21.3 | 0.6230 | **0.2880** | 0.5723 | **0.2984** |
| 1024 @ 0 ov | 21.6 | 0.6146 | 0.2834 | 0.5744 | 0.2992 |
| 768 @ 0.5 ov | 13.7 | 0.6516 | 0.3091 | 0.6031 | 0.3198 |
| 768 @ 0 ov | 13.8 | 0.6507 | 0.3083 | 0.6048 | 0.3211 |
| 512 @ 0.5 ov | 7.6 | 0.6836 | 0.3310 | 0.6376 | 0.3407 |
| 512 @ 0 ov | 7.8 | 0.6773 | 0.3253 | 0.6370 | 0.3394 |
| 256 @ 0.5 ov | 3.2 | 0.7313 | 0.3654 | 0.6833 | 0.3681 |
| 256 @ 0 ov | 3.3 | 0.7274 | 0.3620 | 0.6816 | 0.3670 |

**Box AP50-95 rises from 0.2553 to 0.3654 — +43% relative — purely by changing the tile
size at which per-tile APs are averaged.** Nothing about the predictions changed. AP50
rises 0.577 → 0.731 on the same axis.

**Overlap is not the driver — tile size is.** Every 0.5-overlap arm sits within 0.006 of
its no-overlap twin. The variable that matters is GT density per scored tile.

## The comparison point is confirmed by arithmetic

439 tiles × a 3×3 grid of 1024 px subtiles at 0.5 overlap = **3,951 subtiles**, exactly the
subtile total recorded for SelvaMask's pipeline (`../modal_tcd_multiseed/phase4/RESULTS.md`:
"2,527 of 3,951 subtiles survive"). Our `1024|ov0.5` arm generates that same grid and finds
**2,683 non-empty** subtiles against their 2,527 surviving — the difference being their
additional >80%-black drop. So `1024|ov0.5` is the right row to read against their numbers.

## Restated bottom line for SelvaBox

| term | box AP50-95 |
|---|---|
| ours, published (pooled, 2048) | 0.2555 |
| + their aggregation granularity (1024 @ 0.5) | **0.2880** (+0.033) |
| SelvaBox reported | 0.4429 |
| **residual gap** | **0.155** |

The granularity term is worth **+0.033**, roughly **20% of the gap** — not the ~0.002
("essentially none") claimed in the aggregation study. The earlier figure measured the
right quantity at the wrong granularity.

The remaining **0.155** is not a scoring convention. It is some combination of detector
architecture (a DETR with iterative box refinement vs our frozen-DINOv3 + 8 px CenterNet),
~3,335 in-domain training images vs our 900, and the easier imagery their pipeline produces
(canopy blacked out, empty and mostly-black tiles dropped). Those are not separable from
outside their codebase; running their released weights through our scorer is the measurement
that would split them.

## Caveats

- **Centre assignment, not clipping.** A GT crown or prediction belongs to the subtile its
  box centre falls in; IoU is computed on the original uncropped geometry. `geodataset`
  additionally *clips* annotations to the subtile and filters by an area ratio, which would
  change the IoUs near borders as well. This isolates the aggregation-granularity term alone
  and is a **lower bound** on the full tiling effect.
- Where a crown and its matching prediction straddle a subtile border they can land in
  different subtiles, costing a match. That penalty is included, so the measured rise is
  conservative.
- Empty subtiles are NaN-skipped, matching per-tile averaging's own convention — their
  false positives go uncharged, exactly as in the coarse study.
- Seed 0 only; a protocol sweep does not need the seed band, and the aggregation study
  already established the 3-seed spread (±0.009 box AP50-95).

## Cross-check

The `2048|ov0.0` row reproduces `../aggregation_sensitivity/results_aggregation.json`'s
`per_tile|all439|ign|mdall` cell **exactly** for both runs — ours 0.5770 / 0.2553, DetecTree2
0.5194 / 0.2659 — so the two independent harnesses agree where they overlap.

## Reproduce

```bash
.venv/bin/python -m boxinst_commonality_tcd_04.tiling_sensitivity.tiling_sensitivity
```

CPU only. Reads committed prediction JSONs and `test_gt.json`; writes only
`results_tiling.json` in this folder.
