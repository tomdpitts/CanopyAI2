# boxfix — training-free render fix for LACE's box→mask module (2026-10-15)

Goal: beat SAM 3 zero-shot (per-1024-subtile placement) on identical GT boxes (tab:gtbox protocol, gtbox_lib).
Masker (posterior) unchanged: em_model_4p_fix.npz, knobs 0.3/1.6. Only the render (posterior → mask) changes.

## Frozen setting (`FROZEN_val_2026-10-15.json`, chosen on 108 val tiles before any test look)
- guided filter radius r = min(12, max(1, ⌊0.4 × box short side⌋)) (was fixed 12), eps 1e-4, colour guide
- threshold = 0.3 × peak of filtered posterior inside the box (absolute τ 0.35 never binds) (was τ 0.35)
- mask ∩ box clip ∩ box-inscribed superellipse |2dx/w|^2.5 + |2dy/h|^2.5 ≤ 1 (new)

## Results (mean per-crown IoU, valid crowns; Δ CI = tile bootstrap 95%)
| | val (108 tiles, 7,032) | test (366 tiles, 25,692) |
|---|---|---|
| LACE deployed render | 0.7481 | 0.7477 (parity vs published: 13/25,705 crowns differ, max 4.9e-3) |
| **LACE boxfix_v1** | **0.8195** | **0.8128** |
| SAM 3 zs per-subtile | 0.7973 | 0.7890 |
| boxfix − SAM | +0.022 | **+0.024 [0.019, 0.029]**; small +0.055, mid +0.008, large +0.009 |
| superellipse only, no image (val p3; test p2.5) | 0.7915 | 0.7973 |
| ellipse only (no image) | 0.8018 | 0.7980 |

**Caveats.** (1) A box-inscribed ellipse with no image already beats SAM zero-shot on this metric (val +0.004, test +0.009): GT-box mean IoU is dominated by box geometry for small crowns. (2) One seed of nothing — the render is deterministic; selection over ~90 variants on val (mild optimism; test confirms the main effect).

## Mechanism (val sweeps v1–v8)
- r = half-box alone: −0.050 → −0.013 vs SAM (tiny crowns: 12 px filter wider than the crown erased the posterior).
- + box-relative τ: → −0.004 (small crowns at parity).
- κ 1.0 instead of 1.6: +0.002 (not kept once the shape prior was added; deployed knobs won).
- fill holes / largest component: ≤ +0.0003 (dropped, pre-declared tie rule).
- residual error is over-segmentation (recall > precision; losers GT box-fill 0.60 vs predicted 0.77) → superellipse prior: +0.016.

## Files
cache_windows.py (per-crown posterior windows → feat_cache/boxfix_{val,test}), render_lib.py, sweep.py, compare.py,
parity_val.py / parity_test.py (gates), diag_prec_rec.py, sam_val_masks.py (SAM masks with RLE → feat_cache/boxfix_sam_{val,test}, gated vs stored IoUs),
Outputs: sweep_val_v{1..8}.json, compare_val_v8.json, sweep_test_frozen.json, compare_test_frozen.json. Logs: ../logs/boxfix_*.log.

## Detections: val AP50 check (2026-10-15) — boxfix does NOT transfer to detected boxes
`ap_val.py` → `ap_val_s0_r2048.json` (108 val tiles, det_phase4_L24_s0, 40,058 dets; gate: deployed product AP50 0.6636 == guided_rgb_ext_val_r2048.json, PASS).
| arm | product AP50 / AP75 / AP50:95 | s AP50 | half A / B product AP50 |
|---|---|---|---|
| deployed | **0.6636** / 0.2156 / 0.2977 | 0.6275 | 0.6699 / 0.6567 |
| boxfix_v1 | 0.6590 / 0.2037 / 0.2921 | 0.6218 | 0.6594 / 0.6562 |
| boxfix_noSE | 0.6524 / 0.2068 / 0.2883 | 0.6128 | 0.6565 / 0.6502 |
| deployed_SE | 0.6607 / 0.2040 / 0.2932 | 0.6243 | 0.6644 / 0.6539 |
| hybrid30 (boxfix only for box short side < 30 px; `ap_val_hybrid.py`) | 0.6607 / 0.2103 / 0.2949 | 0.6232 | 0.6599 / 0.6617 |

Matched-pair mask IoU on detections (`pair_iou.py` → `pair_iou_val_s0.json`; 5,428 pairs fixed by box IoU ≥ 0.5, product order): deployed 0.7208, boxfix_v1 0.7220 (+0.001 [−0.001, 0.004]); small +0.012, mid −0.003, large −0.003. The GT-box gain (+0.065 vs deployed) mostly vanishes on detected boxes: the superellipse and box-relative threshold assume the box is the crown's extent.
**Decision (pre-agreed rule): no val arm beats deployed → no render swap; test AP stage NOT run.** `ap_test.py` is written and its gate passed (`ap_test_gate.json`: 498/498 masks byte-identical, product |diff| 1.5e-6) should a test run be wanted anyway.

## LACE boxes, test 439, 3 seeds: LACE masks vs SAM 3 zero-shot masks (2026-10-16)
Same boxes (published LACE detections), same ranking (LACE product, published scores). `a1_same_rank.py` → `scored_a1_sam3zs_lace_product_s{0,1,2}_r2048.json`; `pair_iou.py` → `pair_iou_test_s{s}_lace_vs_sam3zs.json`.
| masks | ranking | AP50 (mean ± sd, ddof 1) |
|---|---|---|
| LACE (deployed render) | LACE product | 0.647 ± 0.003 |
| SAM 3 zs per-subtile | LACE product (HYBRID — needs LACE's posterior; not a SAM baseline, Tom 2026-10-16) | 0.664 ± 0.003 (s0 0.6653, s1 0.6610, s2 0.6657) |
| SAM 3 zs per-subtile | sqrt(s × SAM) (A1) | 0.648 ± 0.003 |
| SAM 3 zs per-subtile | detector s (A1) | 0.633 ± 0.002 |
| LACE | detector s | 0.614 ± 0.003 |

Matched-pair mask IoU (pairs fixed by box IoU ≥ 0.5, LACE product order; ~20k pairs/seed): SAM − LACE = +0.030 (s0 [0.028, 0.034], s1 +0.030, s2 +0.030); small +0.042–0.046, mid +0.031–0.033, large +0.018–0.021. ≥0.5: LACE 0.96 vs SAM 0.98.
Reading: at a FIXED ranking SAM's masks are better (masks-only diagnostic); the fair SAM baseline is its own ranking (0.648) — see `../hunt/README.md` L30, where the 4 px LACE masker beats it; LACE's product ranking is the better ranking for either mask set (+0.016 for SAM masks over SAM's own ranking).
Val box geometry (`box_geometry_val.py` → `box_geometry_val_s0.json`, 5,428 matched pairs): detected box size within ±5% of the crown box (mid/large slightly small, small slightly large), centre offset ~8% of crown size, mean 7.5% of crown pixels outside the box; deployed masks recall 0.84 / precision 0.85.

## Step 2: mask arms on val detections (`ap_val2.py` → `ap_val2_s0_r2048.json`, 2026-10-16)
Boxes and product ranking fixed; masks only. Gate: deployed 0.6636, 40,058 dets, PASS. Rule: qualify if AP50 ≥ deployed + 0.003 and not worse on both halves.
| arm | AP50 all / A / B | AP75 | AP50:95 |
|---|---|---|---|
| deployed | 0.6636 / 0.6699 / 0.6567 | 0.2156 | 0.2977 |
| rhalf | 0.6596 / 0.6609 / 0.6558 | 0.2102 | 0.2944 |
| a0.15 | 0.6603 / 0.6654 / 0.6521 | 0.2089 | 0.2945 |
| a0.5 | 0.6628 / 0.6682 / 0.6571 | 0.2165 | 0.2972 |
| a1.0 | 0.6584 / 0.6618 / 0.6520 | 0.2064 | 0.2907 |
| k1.0 | 0.6622 / 0.6680 / 0.6559 | 0.2128 | 0.2964 |
| k1.3 | 0.6633 / 0.6689 / 0.6572 | 0.2151 | 0.2976 |
| enl1.05 | 0.6605 / 0.6680 / 0.6520 | 0.2115 | 0.2921 |
| enl1.10 | 0.6475 / 0.6569 / 0.6366 | 0.1862 | 0.2761 |
| clip+4 | 0.6580 / 0.6675 / 0.6446 | 0.2106 | 0.2915 |
**No arm qualifies; none beats deployed.** The deployed render and masker knobs sit at a local optimum for val AP50 on detections; the +0.017 AP50 / +0.030 IoU gap to SAM masks is not reachable by render or α/κ/box tweaks. Test not run.
