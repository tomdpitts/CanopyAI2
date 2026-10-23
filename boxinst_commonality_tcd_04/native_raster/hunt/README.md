# hunt — a LACE-style masker that beats SAM 3 zero-shot on AP50 (started 2026-10-16)

Goal: LACE masks on LACE's own boxes beat SAM 3 zero-shot masks on the same boxes, mask AP50, 439 test tiles, 3 seeds. Original target 0.664 ± 0.003 was the HYBRID (SAM masks ranked by LACE's product — not computable by a SAM pipeline); the fair SAM baseline is its own ranking √(s·SAM), **0.648 ± 0.003** (see L30). LACE deployed 0.647 ± 0.003.
Rules: everything local CPU/MPS; every setting chosen on the 108 val tiles (det_phase4_L24_s0, 40,058 dets); test once per frozen candidate; no file overwritten; masker npz and knobs unchanged unless stated.

## Infrastructure (gated)
- `fastap.py` — legacy-canopy-arm AP on RLEs (evaluate.match_tile/_greedy_ap). Gate `check_fastap.py`: deployed val masks 0.6636 / A 0.6699 / B 0.6567 == COCO crowd (ap_val_s0_r2048.json) exactly.
- `sam_val_dets.py` — SAM 3 zs per-subtile masks on the val detections (A1 placement), feat_cache/hunt_sam_val_dets/. **Val target: SAM AP50 0.6776 (A 0.6844, B 0.6721), AP75 0.2839** at LACE product ranking (legacy arm).
- `pxlace.py` — LACE E-step at pixel resolution (interpolated projected features). Gate `check_pxlace.py`: matches the cell posterior near cell centres (median |diff| 0.003).
- `arms.py` + `render.py` — mask arms on val detections → `results/<batch>.json` (gate: deployed = 0.6636).
- `diag_tp.py`, `diag_pair.py` — TP gain/loss decomposition between two mask sets (outputs in `diag/`).

## Lab notebook
### L1 — where SAM beats LACE on val detections (`diag_tp_val.json`, `diag/pair_deployed_vs_sam.json`)
TPs at mask IoU 0.5: SAM-only 281, LACE-only 117 → net +164 for SAM; small crowns net +123 (179 vs 56), mid +37, large +4.
Small SAM-only: LACE mask under-segments (median area 0.45× GT, recall 0.43, 11% empty); SAM 1.1× GT. SAM rarely leaves the box (2–5% of its pixels).
Large SAM-only: LACE over-segments (1.7× GT). Shared TPs: SAM IoU higher by ~0.03 in every size (→ AP75 0.284 vs 0.217).
### L2 — why making small masks bigger failed (`diag/pair_deployed_vs_rhalf.json`)
Half-box radius: +79 small TPs (fixes under-seg) but −103 small TPs where the box is looser than the crown (box IoU 0.47): the deployed mask already over-filled (1.5× GT) and rhalf fills the box (2.1×).
Two small-crown regimes: tight box + weak posterior → under-seg; loose box → mask fills box. Box-shaped knobs cannot fix both; needs pixel-level appearance separation inside the box (what SAM does).
### L3 — first arms (`results/b01_explore.json`): nothing beats deployed
Pixel-resolution E-step on interpolated features (`pxlace`): raw −0.012 to −0.016 AP50 (AP75 collapses to 0.14–0.16: ragged boundaries); + guided r12 ties (+0.0007). Per-box Gaussian colour EM (GrabCut-like): −0.003 to −0.010. Interpolation adds no information.
### L4 — fill/carve signal (`diag_fillcarve.py` → `diag/fillcarve_signals.json`; figure `diag/fig_sam_gain_small.png`)
Small crowns SAM wins: posterior strong (peak 0.77–0.99) but the r12 colour-guided filter carves a partial blob. Among small dets where half-box radius gains (79) vs loses (103): deployed mask box-fill 0.29 vs 0.59 (AUC 0.82); inner-box posterior 0.65 vs 0.77 (AUC 0.83); colour contrasts useless (AUC ~0.45). SAM keeps 72/103 of the loose-box TPs that filling destroys.
### L5 — conditional fill (`results/b02_cond.json`, `b03_fill.json`): first gain, +0.004
Re-render only when the deployed mask fills < θ of its box: best `cond0.5_cell_r12_t0.25` (alt = r12, τ 0.25) **0.6677 (+0.0041; A +0.0011, B +0.0048)**; rhalf/ellipse/quantile alts all ≈ +0.003–0.004 (any fuller mask works). Gap to SAM still −0.010.
### L6 — what remains (`diag/pair_cond05_vs_sam.json`)
Small: SAM +131 / −51. LACE mask now right-sized (1.03× GT) but MISPLACED (prec 0.54, rec 0.63; box IoU 0.47): offset boxes, LACE centres the blob, SAM finds the crown. Mid: SAM +60/−24 (LACE under, 0.75× GT). Large: ~even (+41/−37).
### L7 — weaker spatial prior on small boxes (`results/b04_prior.json`): no help
α 0 / 0.15, κ 1.0 for short < 20/30 px: −0.007 to +0.000 alone; with conditional fill ≤ +0.0041 (= L5). The 8 px cell grid (a 12–15 px crown spans ~2 cells) is the limit, not the prior.
### L8 — real 4 px features (`feat16p.py`)
DINOv3-web L24, 16 shifted passes ({0,4,8,12}²), interleaved to a 512² lattice (s 4, origin 8); stored as the masker's 128-d projection. Requires Modal's library pins: local transformers 5.12.1 gives cos 0.59 vs the Modal feat_4p (gate FAIL, as documented in memory); isolated venv `feat_cache/venv_feat_t457` (transformers 4.57.1, torch 2.12.1, numpy 2.2.6; freeze `requirements_feat_t457.txt`) gives **cos 1.0000** on the {0,8}² sub-lattice (gate per tile in `feat_cache/hunt_z4p_val/_gate.jsonl`). 64 s/tile on MPS.
### L9 — 4 px prototype on 11 val tiles (`results/b05_p4_proto11.json`; SUBSET, compare only with in-batch `sam`)
Deployed 0.7018, SAM 0.7191 (+0.017). Best 4 px arm r12 τ0.30: 0.7047 (+0.003); τ0.40 −0.023 (very τ-sensitive). Not a breakthrough on its own.
### L10 — contrast centre (`recentre.py`, `results/b06_recentre.json`): no
Centring the appearance log-ratio on the ring outside the box, or the core/ring midpoint: −0.0003 to −0.097. LACE's in-box centring is right.
### L11 — ORACLE CEILINGS (`results/b07_oracle.json`, `b08_select.json`; use GT, ceiling only)
Per-detection best of {τ0.35, τ0.25}: **0.6806 > SAM 0.6776**. Best of 6 LACE variants: 0.6963; of 7 τ values: 0.7036; deployed-or-SAM: 0.6946.
→ The LACE family already contains SAM-beating masks; the problem is choosing per box without GT. (Caveat: an oracle over k options also harvests noise.)
### L12 — training-free selectors (`selectors_lace.py`; b08, b09, b10)
Otsu −0.037, stability (SAM-style) −0.025, boundary edge −0.020 (narrow ladder −0.002), colour contrast −0.023; area-calibrated τ (mask area = posterior expected area) −0.004 to −0.075.
Per-detection τ table (`tau_table.py`, `tau_mine.py` → `diag/tau_mine.json`; 6,534 near-crown dets, IoU≥0.5 hits: deployed 5,606, oracle 5,970): best single predictor of τ* = posterior peak in the box (held-out +53 hits): low peak → τ 0.25, high → 0.40–0.45, i.e. τ ≈ c·peak both ways.
Two-way relative threshold τ = 0.4·peak: **0.6677 (+0.0041)** = same ceiling as conditional fill (L5); combos do not add. All 8 px fixes repair the same under-filled small boxes and saturate at +0.004; the remainder (misplaced masks in offset boxes; mid under-seg) needs resolution.
### L13 — shape regularisation (`results/b12_shape.json`; figure `diag/fig_sam_gain_mid.png`)
Mid crowns SAM wins: same place/size, but LACE edges ragged (notches, bitten corners), many at IoU 0.40–0.50. Closing (disk 0.2·short) + fill on τ = 0.4·peak: **0.6692 (+0.0056)**; hull −0.006 on deployed; smoothing ≈ closing.
### L14 — per-tile test-time EM (`tta.py`, `results/b13_tta.json`): no
MAP re-estimation of fg/bg prototypes per tile (λ 50–1000 cells, 2–4 iters): −0.004 to +0.0004; with τ = 0.4·peak +0.0053 (= threshold alone). The 4-tile smoke test (+0.011/+0.021) was noise — never trust < full val.
### L15 — real 4 px features on full val (`results/b11_p4_full.json`, `b14_p4_combo.json`)
Same masker on the 4 px lattice (s 4, origin 8). τ 0.30: 0.6694 (+0.0058, AP75 0.228 best); τ = c·peak: c 0.30 0.6726, **c 0.33 0.6759 (+0.0123)**, c 0.35 0.6713, c 0.37 0.6700 (sensitive); +closing c0.35 0.6751; κ 1.3 c0.35 0.6756; r10 c0.35 0.6745; 8/4 px posterior mix no better. SAM 0.6776 — gap −0.0017.
### L16 — test 4 px features extracting (`logs/hunt_feat16p_test.log`; gate cos 1.0000 on test tiles too), ~8 h.
### L17 — 4 px factorial (`results/b15_p4_factorial.json`): plateau
c ∈ {0.33, 0.35} × κ ∈ {1.3, 1.6} × r ∈ {10, 12} × closing: 0.6713–0.6759; no combination exceeds c0.33/κ1.6/r12 (0.6759).
### L18 — pixel E-step on 4 px features (`results/b16_px4.json`, `b17_px4_gate.json`)
`px4` (interpolated 4 px features → LACE E-step per pixel) + guided r12 + τ = 0.33·peak: **0.6770** (A 0.6812, B 0.6717). c 0.28–0.36: 0.6736–0.6770.
Size gate (px4 for box short side < T, 4 px cell posterior otherwise): **T 30 → 0.6787 (A 0.6833, B 0.6730) vs SAM 0.6776 (A 0.6844, B 0.6721)**; T 20/45 0.6778. Reverse gates worse.
### L19 — ensembles (`results/b18_ensemble.json`): no gain (vote/medoid/union/intersection of 8 px, 4 px, px4: 0.6754–0.6765).
### L20 — self-consistent re-boxing (`results/b19_rebox.json`): fails
Refit box to mask0 extent and re-run the posterior: masks run away when the box may grow (0.20–0.59); constrained within the detector box it is a no-op (0.6786).
### L21 — remaining SAM wins vs g30 (`diag/pair_g30_vs_sam.json`, figure `diag/fig_g30_sam_gain_mid.png`)
Small balanced (75/75). Mid SAM +56/−29: near-threshold (LACE 0.41–0.49 vs SAM 0.50–0.59), crown often exceeds a slightly small box, SAM fills box edges/corners. Large: LACE +45/−33 (SAM-won large: LACE spills into a neighbour, 1.65× GT).
### L22 — instance competition (`results/b21_compete.json`): hurts (−0.0016 to −0.025). Large-branch knobs (`b22_large.json`): α/κ/c/closing all 0.6778–0.6788 — robust parity: every g30 variant ≥ SAM 0.6776, none clearly above.
### L23 — joint-bilateral 4 px feature upsampling (`results/b23_jbu.json`): no gain (σ 0.05–0.2: 0.6740–0.6786).
### L24 — learned-free per-detection SELECTION among saved variants (`variant_mine.py` → `diag/variant_mine.json`)
Masks of 6 variants saved (`b24_save_variants`); quintile lookups on intrinsic signals (box size, score, agreement IoU g30↔deployed/e8, area ratios, fill), fit on one val half, scored on the other: never beats g30 held-out (A→B ≤ 0.6730 = g30; B→A ≤ 0.6837 vs 0.6833). g30 is the local optimum.
### L25 — CANDIDATE 1 FROZEN (`FROZEN_cand1_2026-10-16.json`, before any test look): `lace4px_g30`
Val 0.6787 (A 0.6833, B 0.6730) vs SAM 0.6776 (A 0.6844, B 0.6721). Paired tile bootstrap (`val_boot.py` → `diag/val_boot_g30_vs_hunt_sam_val_dets.json`): **+0.0010, 95% CI [−0.0064, +0.0068], P(≤0) 0.44 — a tie on val.**
Test harness `test_eval.py` (gate: deployed spec = published masks 498/498 byte-identical; dry render OK). Ran on test 2026-10-16: see L30.
### L26 — posterior-supported edge fill for mid/large (`results/b25_edgefill.json`): hurts AP50 (0.6679–0.6772), helps AP75 slightly.
### L27 — 4 px posterior scored past the box edge (prior clamped; `post4pad`, `results/b26_pad.json`; gate pad0 = g30 exactly): margin 2 px 0.6781, 4 px 0.6730–0.6773, 8 px 0.6474–0.6753. No.
### L28 — 4 px masker refit: NOT pursued (reasoning)
The masker's parameters (PCA whitening, vMF prototypes, background mixture, box-normalised spatial prior) are functions of the per-cell feature distribution and box geometry, both stride-invariant (4 px and 8 px cells are the same backbone sampled at different shifts). A refit would converge to ≈ the same model; and a 1024-d refit at 4 px needs ~160 GB RAM.
### L29 — flip test-time augmentation: considered and ABANDONED (2026-10-19)
Not part of the method, triples backbone cost, and SAM 3 could equally use it. Code, spec and cached views deleted.
### L30 — CANDIDATE 1 ON TEST (`test/`, `test/compare_lace4px_g30.json`, `logs/hunt_test_cand1.log`) — all 3 seeds, 439 tiles
Gate: deployed spec = published masks 498/498 byte-identical. Same LACE boxes for every row; mask AP50 (score_coco: 2048, maxDets 600, floor 0, canopy crowd).
| masks (ranking) | s0 | s1 | s2 | mean ± sd |
|---|---|---|---|---|
| **LACE cand1 `lace4px_g30`** (LACE product) | 0.6569 | 0.6518 | 0.6589 | **0.6559 ± 0.0037** |
| LACE deployed (LACE product) | 0.6472 | 0.6444 | 0.6493 | 0.6470 ± 0.0025 |
| **SAM 3 zs, its OWN ranking √(s·SAM)** (A1) | 0.6451 | 0.6479 | 0.6507 | 0.6479 ± 0.0028 |
| SAM 3 zs, detector score s (A1) | | | | 0.6329 ± 0.0024 |
| *hybrid, NOT a SAM baseline:* SAM 3 masks ranked by LACE's product | 0.6653 | 0.6610 | 0.6657 | 0.6640 ± 0.0026 |

**Headline (fair: each method with a ranking it can compute itself): cand1 − SAM 3 = +0.0118 / +0.0039 / +0.0082, mean +0.0080 ± 0.0040; positive on all 3 seeds.** Evidence = the official 439-tile test set × 3 detector seeds (no tile bootstrap: the fixed official set is the comparison other papers use). The CIs inside `compare_lace4px_g30.json` are against the hybrid row only.
The hybrid row uses LACE's box→mask posterior (bimod × p̄) to rank SAM's masks — a SAM-only pipeline cannot compute it (Tom, 2026-10-16), so it is a masks-at-fixed-ranking diagnostic, not the SAM baseline. Throughout L1–L27 "SAM val ref 0.6776" is this hybrid on val.
AP50:95 (same files): cand1 0.2856 / 0.2722 / 0.2844 vs SAM 3 own ranking 0.3027 / 0.2971 / 0.3027 → **SAM wins AP50:95 by 0.017–0.025 on every seed** (tighter outlines; LACE finds/ranks more crowns at IoU 0.5).
Val→test shrinkage: gain over deployed +0.015 (val) → +0.009 (test); selection over ~170 val arms inflated it.
### Status (2026-10-19)
Closed. Candidate 1 `lace4px_g30` is the result (L30). Nothing running, nothing pending.
