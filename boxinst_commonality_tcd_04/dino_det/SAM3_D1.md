# SAM 3 zero-shot masker on LACE (D1 variant) boxes

<!-- TABLE1_CURRENT:begin (auto, results_439/mark_superseded.py) -->
> ⚠️ **SUPERSEDED 439 / sparse236 NUMBERS (marked 2026-10-27; table updated 2026-11-04).** Current headline table: [`TABLE1_CURRENT.md`](../results_439/TABLE1_CURRENT.md). Test-set numbers in this file are a historical experiment record (512 raster, pre-seam-fix baselines, 8 px masker or earlier models) and must not be quoted unless they match that table.
<!-- TABLE1_CURRENT:end -->

Approved by Tom 2026-11-03. Code: `dino_det/sam3_on_d1.py`. Status: **DONE 2026-11-03** (results below).

## Question
Same frozen D1 boxes, two maskers: LACE masker (`lace4px_simple`, s′) vs SAM 3 zero-shot. What does SAM 3 give on our boxes?

## Frozen inputs
- Boxes + detector score s: `results/preds_dino_swinl_s0_{test,sparse}_detscore_r2048.json` (test 439 tiles / 200,148 boxes; sparse-236 / 106,514 boxes; s ≥ 0.05).
- LACE comparison row: `results/preds_dino_swinl_s0_{split}_sprime_r2048.json` (frozen D1, test mask AP50 0.7035 / AP50:95 0.335). D1 refinements done elsewhere after 11-03 are **not** in this comparison; if they keep the same boxes, the SAM masks here can be re-scored against them without re-running SAM.

## Method (= Tier A1, `native_raster/tierA/a1_lace_sam3_persub.py`)
`facebook/sam3` via HF Sam3Tracker, no fine-tuning, local MPS. Each box on the most central 1024 subtile (3×3, stride 512) that contains it, else whole 2048 tile; `multimask_output=False`; masks pasted at 2048. Sparse tiles whose D1 boxes equal the test tile's (16 overlap) copied from the test run.

Rankings (SAM's own pipeline only): **√(s·SAM)** (floor 0) and **D1 detector s** (floor 0.05). SAM masks at LACE's s′ are not a SAM baseline (Tom, 10-16) and are not reported.

## Gate
`parity`: re-prompt A1 s0's stored boxes on tcd_val_tile_0/1; mean mask IoU vs stored A1 masks ≥ 0.99 → `results/sam3_d1/parity_vs_a1_s0.json`.

## Scoring
Frozen `score_coco` (res 2048, maxDets 600, canopy as crowd). Paired tile bootstrap (2,000, seed 0) of LACE(D1) − SAM for mask AP50 and AP50:95; bootstrap full-set AP must reproduce the scorer. Test-set sampling only; one detector seed.

## Artefacts
- Per-tile checkpoints: `~/dphil/feat_cache/sam3_d1_{test,sparse}/` (RLE at 2048, SAM scores, subtile, `_env.json`).
- `results/sam3_d1/{preds,scored}_d1_sam3zs_{gmean,det}_{split}_r2048.json`, `summary_sam3_d1_{split}.json`.
- Logs: `native_raster/logs/sam3_d1_{parity,test,sparse,score_test,score_sparse}.log`.

## Prediction (written before results)
AP50 ≈ tie near 0.70; SAM ahead on AP50:95 (GT boxes: tie at AP50, SAM ahead on boundary; CenterNet boxes: LACE +0.007 AP50, SAM +0.02 AP50:95).

## Results (2026-11-03, complete)
Parity gate PASS: 226/226 masks IoU 1.0, SAM scores identical. Sparse: 16 tiles reused from test (identical boxes). Bootstrap full-set AP reproduces the scorer exactly for every row. Summaries: `results/sam3_d1/summary_sam3_d1_{test,sparse}.json`.

Mask AP (box AP50 in last column). Same D1 boxes in every row.

| Split | Row | AP50 | AP75 | AP50:95 | box AP50 |
|---|---|---|---|---|---|
| test 439 | LACE (D1) s′ | 0.7035 | 0.2615 | 0.3347 | 0.7391 |
| test 439 | SAM 3 zs, √(s·SAM) | 0.7013 | 0.3107 | 0.3553 | 0.7301 |
| test 439 | SAM 3 zs, s | 0.6950 | 0.2950 | 0.3464 | 0.7257 |
| sparse 236 | LACE (D1) s′ | 0.7349 | 0.3900 | 0.3962 | 0.7496 |
| sparse 236 | SAM 3 zs, √(s·SAM) | 0.7399 | 0.4505 | 0.4252 | 0.7481 |
| sparse 236 | SAM 3 zs, s | 0.7328 | 0.4378 | 0.4157 | 0.7414 |

LACE − SAM, paired tile bootstrap (2,000; test-set sampling only):

| Split | vs | ΔAP50 [95% CI] | ΔAP50:95 [95% CI] |
|---|---|---|---|
| test | √(s·SAM) | +0.0022 [−0.0033, +0.0094] | −0.0206 [−0.0253, −0.0163] |
| test | s | +0.0084 [+0.0032, +0.0159] | −0.0117 [−0.0163, −0.0077] |
| sparse | √(s·SAM) | −0.0050 [−0.0118, +0.0042] | −0.0289 [−0.0369, −0.0197] |
| sparse | s | +0.0020 [−0.0039, +0.0115] | −0.0195 [−0.0282, −0.0104] |

**Reading.** Against SAM's best own ranking √(s·SAM): AP50 tie on both splits (CIs straddle 0); SAM clearly ahead on AP50:95 (−0.021 test, −0.029 sparse) and AP75 (+0.05/+0.06 for SAM). Matches the pre-registered prediction and the GT-box / CenterNet-box pattern: equal at detection-level overlap, SAM's edge = boundary precision. LACE's only significant AP50 win is over SAM ranked by bare s (+0.008), not SAM's best ranking. Limits: one detector seed; LACE row is frozen D1 s′, not later refinements (same boxes, so re-score only).

## Mask IoU, LACE vs SAM 3 (2026-11-04)
`dino_det/sam3_d1_iou.py` → `results/sam3_d1/mask_iou_{test,sparse}.json`. Same D1 detection, two masks. Box match to a GT tree = box IoU ≥ 0.5. The vs-GT rows use, for each GT tree, the single detection with the highest box IoU, so both maskers are scored on the same detection.

| | test 439 | sparse 236 |
|---|---|---|
| LACE↔SAM, all detections (mean) | 0.757 (n 200,148) | 0.760 (n 106,514) |
| LACE↔SAM, detections matching a GT tree (mean) | 0.828 (n 23,803) | 0.842 (n 13,769) |
| LACE vs GT (mean, matched crowns) | 0.733 (n 22,612) | 0.759 (n 12,967) |
| SAM 3 vs GT (mean, same crowns) | 0.749 | 0.779 |
| LACE − SAM vs GT | −0.016 (LACE better on 39%) | −0.020 (36%) |

## Where LACE trails SAM 3 on IoU with GT (2026-11-04)
Scripts `dino_det/sam3_d1_gap.py` (per-crown table + single-variable strata → `results/sam3_d1/gap_crowns_*.npz`, `gap_strata_*.json`) and `dino_det/sam3_d1_gap2.py` (exclusive groups + error by distance from GT boundary → `gap2_*.json`). Same crowns/pairing as above. Gap = SAM IoU − LACE IoU; share = fraction of the summed gap.

Exclusive groups (first match wins: tile edge → touching → elongated → rest):

| Group | test: crowns / LACE / SAM / gap / share | sparse: crowns / LACE / SAM / gap / share |
|---|---|---|
| Tile edge (det box within 2 px of tile border) | 7% / 0.727 / 0.776 / +0.048 / 22% | 7% / 0.760 / 0.805 / +0.045 / 15% |
| Touching (det box ≥10% covered by other GT trees) | 9% / 0.685 / 0.728 / +0.043 / 23% | 7% / 0.681 / 0.731 / +0.050 / 16% |
| Elongated (GT aspect ≥ 1.5) | 4% / 0.699 / 0.721 / +0.022 / 6% | 4% / 0.731 / 0.749 / +0.018 / 4% |
| Rest | 80% / 0.740 / 0.750 / +0.010 / 50% | 82% / 0.766 / 0.782 / +0.016 / 65% |

Findings:
- Overall the deficit is false-positive pixels (LACE FP 0.202 vs SAM 0.164 per GT area, test); FN is equal or better for LACE (0.136 vs 0.142). LACE over-extends.
- Touching: LACE FP 0.318 vs SAM 0.182; the excess lies ≥4 px from the GT boundary (4–8 / 8–16 / >16 px: +0.038 / +0.038 / +0.020 test) = leakage into neighbouring crowns, not boundary resolution. LACE has fewer FN there (0.123 vs 0.159).
- Tile edge: LACE misses 6.0% of GT area within 8 px of the tile border vs SAM 1.9% (sparse 5.4% vs 1.7%) → LACE masks stop short of the tile border; plus extra far FP (>8 px).
- Rest: excess FP is a 2–8 px band at the boundary (test 2–4 +0.009, 4–8 +0.009, 8–16 +0.006) = resolution/boundary placement, diffuse.
- Closing the gap only in tile-edge + touching would close 45% of it on test (31% on sparse). These are IoU numbers, not AP effects, and "touching" uses GT neighbours, which aren't available at inference (overlapping detections are the observable proxy).

## AP75 (2026-11-04)
`dino_det/sam3_d1_ap75.py` → `results/sam3_d1/ap75_{split}.json` (bootstrap full-set AP75 reproduces the scorer). Mask AP75, LACE (D1) s′ vs SAM 3 zs; LACE − SAM paired tile bootstrap (2,000):

| Split | LACE s′ | SAM √(s·SAM) | SAM s | LACE − SAM √(s·SAM) | LACE − SAM s |
|---|---|---|---|---|---|
| test | 0.261 | 0.311 | 0.295 | −0.049 [−0.059, −0.040] | −0.034 [−0.042, −0.025] |
| sparse | 0.390 | 0.450 | 0.438 | −0.060 [−0.078, −0.042] | −0.048 [−0.066, −0.029] |

## Tile-border diagnostic (2026-11-04)
`dino_det/sam3_d1_border.py` → `results/sam3_d1/border_{split}.json`. GT-pixel miss rate within 16 px of each tile side (matched crowns), test:

| Side | LACE 0–2 / 2–4 / 4–6 / 6–8 / 8–12 / 12–16 px | SAM 3 same bands |
|---|---|---|
| top | 0.549 / 0.499 / 0.382 / 0.283 / 0.216 / 0.183 | 0.121 / 0.127 / 0.119 / 0.122 / 0.123 / 0.125 |
| left | 0.535 / 0.480 / 0.363 / 0.265 / 0.208 / 0.172 | 0.108 / 0.116 / 0.112 / 0.114 / 0.124 / 0.128 |
| bottom | 0.361 / 0.335 / 0.293 / 0.250 / 0.200 / 0.165 | 0.091 / 0.091 / 0.091 / 0.095 / 0.100 / 0.109 |
| right | 0.181 / 0.175 / 0.165 / 0.153 / 0.136 / 0.125 | 0.084 / 0.089 / 0.091 / 0.094 / 0.098 / 0.109 |

Sparse: same ordering (top 0.452, left 0.528, bottom 0.339, right 0.143 at 0–2 px). SAM is flat (~0.1) to the border; LACE rises toward every border, worst top/left. Code hypotheses: H1 top/left = 6 px zero-fill in `guided_ablate_modal._prob` (delta shift), softened by the r12 guided filter (hence ~55%, not 100%, at 0–2 px, decaying over ~16 px); H2 bottom/right = black zero-fill in `phase4_features_tcd._shift` for the 16-phase features. Bottom ≫ right is not explained by H2 alone (shifts are symmetric) — open. No fix applied yet.

## Top/left border fix, VAL (2026-11-04)
`dino_det/edgefix_val.py` → `results/edgefix/edgefix_val.json`, masks `feat_cache/edgefix_val/`. Fix: edge-replicate the 6 px band `_prob` zero-fills (top/left), applied by wrapping `GA._prob` in the script's workers only (shared code untouched). s′ unchanged (verified), so ranking is identical; 3,625 / 51,341 val masks change. Parity: baseline re-render of 3 val tiles = cached D1 val masks (1,800/1,800 identical). Decision rule written before running: carry to one pre-registered test run iff val AP50 ≥ baseline − 0.001 and tile-edge crown IoU improves → **both met**.

| Val 108, s′ | AP50 | AP75 | AP50:95 |
|---|---|---|---|
| D1 baseline | 0.7384 | 0.3345 | 0.3763 |
| + top/left fix | 0.7414 | 0.3392 | 0.3788 |
| Δ [95% CI, paired tile bootstrap] | +0.0030 [+0.0005, +0.0038] | +0.0047 [+0.0026, +0.0079] | +0.0025 [+0.0016, +0.0032] |

Tile-edge crown mean IoU 0.750 → 0.776 (445 crowns; interior unchanged 0.745). GT miss rate at 0–4 px: top 0.509 → 0.175, left 0.495 → 0.154 (now ≈ the 8–16 px level). Test not run.

### Pre-registration, top/left fix on TEST (written 2026-11-04 before any test render; Tom approved "yes")
Frozen: D1 test/sparse boxes and detector scores, lace4px_simple, s′ ranking (unchanged by the fix, asserted), fix exactly as val. Run ONCE on test 439 (primary) and sparse 236 (secondary). Report frozen `score_coco` mask AP50/AP75/AP50:95, paired tile bootstrap (2,000, seed 0) fix − baseline, and LACE(fixed) − SAM 3 zs √(s·SAM). Whatever it gives is reported. Script `dino_det/edgefix_test.py`.

### Top/left fix, TEST result (run once, 2026-11-04)
`dino_det/edgefix_test.py` → `results/edgefix/edgefix_{test,sparse}.json`, `{preds,scored}_d1_edgefix_{split}_sprime_r2048.json`; masks `feat_cache/edgefix_{test,sparse}/`. s′ identical to published (asserted); masks changed: test 13,171 / 200,148, sparse 6,121 / 106,514.

| Split | D1 baseline AP50 / AP75 / AP50:95 | D1 + fix | Δ AP50 [CI] | Δ AP75 [CI] | Δ AP50:95 [CI] |
|---|---|---|---|---|---|
| test 439 | 0.7035 / 0.2615 / 0.3347 | 0.7044 / 0.2644 / 0.3366 | +0.0009 [+0.0005, +0.0032] | +0.0029 [+0.0024, +0.0048] | +0.0019 [+0.0016, +0.0027] |
| sparse 236 | 0.7349 / 0.3900 / 0.3962 | 0.7350 / 0.3944 / 0.3980 | +0.0001 [−0.0001, +0.0007] | +0.0044 [+0.0018, +0.0055] | +0.0018 [+0.0012, +0.0023] |

LACE (D1 + fix) − SAM 3 zs √(s·SAM): test AP50 +0.0031 [−0.0019, +0.0106], AP75 −0.0463 [−0.0546, −0.0371], AP50:95 −0.0187 [−0.0232, −0.0141]; sparse AP50 −0.0049 [−0.0117, +0.0043], AP75 −0.0561, AP50:95 −0.0272. Conclusion unchanged: AP50 tie; SAM ahead at strict IoU.

**Adopted (Tom, 2026-11-04):** fix moved into shared `native_raster/guided_ablate_modal._prob` behind `EDGE_FILL = True` (False reproduces every pre-11-04 render). Parity `results/edgefix/shared_flag_parity.json`: flag on = fixed test masks, flag off = published D1 masks (2 test tiles, 475/475 each way, incl. 2 masks that differ). `evaluate.pred_instance_masks` (older 8 px path) still has the zero-fill and is untouched. TABLE1_CURRENT / paper.tex NOT updated — Tom decides. **2026-11-04 (Tom):** TABLE1_CURRENT now carries "LACE (D1 variant) + edge fix" (0.704 / 0.337 test, 0.735 / 0.398 sparse) plus the + R3 and + R3 + σ2 rows (`dino_det/README.md` R3-edgefix); the plain D1 row is kept. paper.tex still NOT updated.
