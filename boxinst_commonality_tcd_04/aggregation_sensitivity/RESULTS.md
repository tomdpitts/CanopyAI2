# AP aggregation sensitivity — OAM-TCD 439 (2026-08-20)

**Question.** Our published 439 numbers are **pooled** AP: per-tile greedy matching, then all
tiles' TP/FP flags concatenated into one globally score-sorted PR curve against a global
`n_gt` (25,705). SelvaMask/CanopyRS compute AP **inside each subtile and average over tiles**.
How much of the gap is aggregation rather than detection quality?

**Method.** Rescore the *identical* saved predictions under a full cross of
`aggregation × tileset × canopy-ignore × maxDets`. Nothing is retrained and nothing is
re-inferred; only the scoring reduction changes. Code: `aggregation_sensitivity.py`.
Output: `results_aggregation.json` (36 cells × {mask, box} × 4 runs).

Runs: ours knobbed seeds 0/1/2 (`modal_tcd_multiseed/phase4/preds/knobbed_s*.json`) and
DetecTree2 s0 full-coverage (`detectree2_baseline/preds_dt2_s0_fullcov.json`).

**SENSITIVITY ONLY — published numbers are unchanged** (`pooled | all439 | ign | uncapped`).

---

## Headline: aggregation is not a small convention

3-seed mean, canopy-ignore on, uncapped, all 439 tiles. DetecTree2 is single-seed.

| metric | OURS pooled | OURS per-tile | DT2 pooled | DT2 per-tile | margin pooled → per-tile |
|---|---|---|---|---|---|
| mask AP50 | 0.6301 | 0.5983 | 0.6011 | 0.5858 | **+0.029 → +0.013** |
| mask AP50-95 | 0.2561 | 0.2594 | 0.2572 | 0.2854 | **−0.001 → −0.026** |
| box AP50 | 0.6033 | 0.5703 | 0.5973 | 0.5822 | **+0.006 → −0.012** |
| box AP50-95 | 0.2495 | 0.2492 | 0.2819 | 0.3068 | **−0.032 → −0.058** |

> **Re-run 2026-09-03** against the converged DetecTree2 (`preds_dt2_s0v2.json`). The DT2 columns
> and every margin moved; the OURS columns are unchanged. Note these are the **un-reranked** LACE
> seeds, so this table is a protocol study, not the paper's headline — that is the
> posterior-product row under the frozen COCOeval scorer (`../PROTOCOL_439.md`).

Three things fall out, one of them unwelcome:

1. **Per-tile averaging is not uniformly kinder.** It *lowers* AP50 for both models and
   *raises* AP50-95 for both. The prior expectation that it simply reads higher was wrong.
2. **It is strongly non-neutral between models.** DetecTree2 gains +0.028 mask AP50-95 from
   the switch; we gain +0.003. Against the converged DetecTree2 we no longer lead on mask
   AP50-95 under either aggregation: −0.001 pooled, −0.026 per-tile.
3. **We do not lead on box AP50-95 under any aggregation.** DetecTree2 is 0.032 ahead pooled
   and 0.058 ahead per-tile. (Before the 2026-09-03 retrain this read as a +0.007 pooled lead
   flipping to −0.017 per-tile, i.e. an aggregation artefact; against the converged model there
   is no lead to explain away.) The only margin that survives both aggregations is mask AP50,
   +0.029 pooled and +0.013 per-tile — and that is the un-reranked baseline; the posterior
   product leads by +0.066 pooled.

The mask AP50 headline (+0.096 → +0.072) and box AP50 (+0.074 → +0.051) survive both
protocols with room to spare. **Those are the claims to lead with.**

## Mechanism: the crossover sits at IoU ≈ 0.65

Per-IoU pooled → per-tile deltas, seed 0, mask, ignore on:

| IoU | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 |
|---|---|---|---|---|---|---|---|---|---|
| OURS | −0.027 | −0.021 | −0.012 | −0.001 | +0.023 | +0.038 | +0.032 | +0.018 | +0.003 |
| DT2 | −0.009 | +0.003 | +0.016 | +0.036 | +0.054 | +0.068 | +0.059 | +0.042 | +0.011 |

At **loose** IoU, per-tile averaging costs: every tile gets one vote, so sparse tiles — where
missing 1 of 3 crowns costs 33% — outweigh the dense closed-canopy tiles that dominate the
pooled `n_gt`. At **strict** IoU it pays, and pays much more: pooled AP requires the handful
of surviving TPs to outrank ~160k globally-ranked false positives, so precision collapses,
whereas per-tile each tile's few TPs only compete with that tile's ~360 predictions. **The
tighter the IoU, the rarer the TPs, the more per-tile averaging rescues** — which is exactly
why AP50-95 moves far more than AP50, and why the model with more scattered errors
(DetecTree2) is rescued hardest.

## Empty tiles

Pooled, ignore on, uncapped. `nonempty` drops the 73 zero-GT tiles.

| | mask AP50 | mask AP50-95 | box AP50 | box AP50-95 |
|---|---|---|---|---|
| OURS all439 → nonempty | 0.6301 → 0.6345 | 0.2561 → 0.2578 | 0.6033 → 0.6075 | 0.2495 → 0.2513 |
| DT2 all439 → nonempty | 0.6011 → 0.6090 | 0.2572 → 0.2605 | 0.5973 → 0.6052 | 0.2819 → 0.2855 |

Keeping the 73 zero-GT tiles costs us **0.004 mask AP50** and DetecTree2 **0.009** — a real
handicap against protocols that delete such tiles, but a *smaller* one for us than for
DetecTree2, so it is not the source of our margin. 4.0–5.6% of our predictions land in
zero-GT tiles vs 6.1% of DetecTree2's.

**Under per-tile averaging the axis vanishes**: a zero-GT tile has no defined AP, so it is
NaN-skipped and its false positives are *uncharged*. `per_tile|all439` and `per_tile|nonempty`
are identical in every cell (asserted in the script). The strict alternative — score a
zero-GT tile 0 if it fired at all — is the `per_tile_empty0` row, and it is punishing:
mask AP50 0.5983 → 0.5058 for us, 0.5260 → 0.4477 for DetecTree2.

## maxDets

Their cap is 400/tile; our detector emits top-600 and DetecTree2 has 80 tiles above 600
(max 968). Dropping to 400 costs **0.002–0.003 AP50-95** for every model under either
aggregation. Negligible next to the aggregation term.

> ⚠️ **2026-08-21 — the SelvaBox conclusion below is superseded.** It compares per-tile
> averaging at OUR 2048 px granularity, but SelvaMask averages over 1024 px subtiles, and
> per-tile AP is a different estimator at every granularity. Measured properly in
> `../tiling_sensitivity/`, the aggregation term is worth **+0.033 box AP50-95** (~20% of
> the gap), not the ~0.002 stated here. The pooled-vs-per-tile findings above, which are all
> at fixed 2048 px granularity, are unaffected.

## What this does and does not settle for SelvaBox / SelvaMask

SelvaBox reports **box mAP50-95 44.29 ± 0.33** on OAM-TCD. Ours, moved onto their
aggregation *and* their maxDets (`per_tile | ign | md400`), is **0.2477** — against our
published 0.2495. **Aggregation plus maxDets accounts for ~0.002 of a ~0.19 gap: essentially
none of it.** The distance to SelvaBox is not a scoring convention.

What is still **not** simulated, and remains the honest residual: their 1024 px @ 0.5-overlap
subtiling (which changes what an "instance" is near tile borders), their >80%-black tile drop,
and their ~3,335 in-domain training images vs our 900. A subtiled rescore is the next lever
if the comparison ever needs to be tight.

## Side finding: the published scorer is knife-edge on tied IoUs

`evaluate.mask_iou` computes mask intersections with a **float32 SGEMM**
(`evaluate.py:56`). On Apple Accelerate that call raises `overflow`/`invalid value`
RuntimeWarnings and lands up to **3e-8** off the true IoU. Mask IoU is a ratio of small
integers, so values sitting *exactly* on a sweep threshold are common and `iou >= thr` is
decided by the last ulp: **151 threshold-crossing flips per 250 tiles**.

Consequences, all confined to IoU ≥ 0.65:

- Rescoring the identical saved predictions with the identical shipped code gave 0.4042 vs
  the recorded 0.4029 at IoU 0.65 and 0.1445 vs 0.1379 at IoU 0.75 — mask AP50-95 0.2585 vs
  the recorded 0.2574.
- It is not stable run to run. The shipped `no_ignore_sensitivity.py` reproduced 0.2574
  exactly on this machine while a harness with byte-identical per-tile IoU matrices (verified
  over 120 tiles) got 0.2585 — the SGEMM's last ulp depends on the process's allocation state.
- **AP50 is unaffected** (0.6250 either way), so every published headline AP50 stands. The
  exposure is ~0.001 on AP50-95.

This study therefore scores on **exact integer RLE IoU** (pycocotools, C integer arithmetic),
deterministic by construction, so all 36 cells share one basis and the cell-to-cell deltas are
clean. The gate measures the residual against the recorded figures rather than asserting
equality:

| run | mask AP50 | mask AP50-95 | box AP50 | box AP50-95 |
|---|---|---|---|---|
| ours s0 | 0.6250 (=) | 0.2568 (−0.0006) | 0.6050 (=) | 0.2555 (=) |
| ours s1 | 0.6358 (=) | 0.2560 (−0.0007) | 0.5982 (+0.0001) | 0.2396 (=) |
| ours s2 | 0.6294 (=) | 0.2554 (−0.0010) | 0.6066 (=) | 0.2534 (−0.0001) |
| DetecTree2 s0 (v2) | 0.6011 (=) | 0.2572 (−0.0008) | 0.5973 (=) | 0.2819 (=) |

Every AP50 and every box AP50-95 reproduces exactly; mask AP50-95 sits 0.0006–0.0010 below
the recorded value, in the direction and of the size the tie-breaking analysis predicts.
Fixing `evaluate.mask_iou` to accumulate exactly would make the published mask AP50-95 figures
reproducible — it would restate them by ~0.001, and is a separate change, not made here.

## Reproduce

```bash
.venv/bin/python -m boxinst_commonality_tcd_04.aggregation_sensitivity.aggregation_sensitivity
```

~25 min on the M-series laptop, CPU only. Reads only committed prediction JSONs and
`test_gt.json`; writes only `results_aggregation.json` in this folder. No file outside this
folder is modified.
