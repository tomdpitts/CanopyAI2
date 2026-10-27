# Paper-number audit — paper.tex (working copy) vs scored sources (2026-10-14)

<!-- TABLE1_CURRENT:begin (auto, results_439/mark_superseded.py) -->
> ⚠️ **SUPERSEDED 439 / sparse236 NUMBERS (marked 2026-10-27).** Current headline table: [`TABLE1_CURRENT.md`](../../results_439/TABLE1_CURRENT.md). Test-set numbers in this file are a historical experiment record (512 raster, pre-seam-fix baselines, 8 px masker or earlier models) and must not be quoted unless they match that table.
<!-- TABLE1_CURRENT:end -->

Scope: every number in `ablation/results/paper.tex` (working copy, 1264 lines), rendered text only (comments listed separately). Read-only; no re-scoring. Recomputation = means and ddof=1 sds from stored JSON keys, GT counts from `test_gt.json` / `data/tcd_sparse/sparse_gt.json`, per-tile det counts from `scored_*.json:areas`.

Abbreviations: `NR/` = `boxinst_commonality_tcd_04/native_raster/`; `AR/` = `boxinst_commonality_tcd_04/ablation/results_r2048/`; `crowd` = key `canopy_neutral_crowd`; `G{s}` = `NR/scored_lace_s{s}_r2048_cgf_r12_tau035_guidedrun.json`; `D{s}` = `NR/scored_lace_s{s}_r2048_cgf_r12_tau035_detscore.json`; `SG{s}` / `SD{s}` = sparse equivalents; EXT = external literature/config fact, not a scored quantity (not audited beyond internal consistency).

Totals: 248 trace rows, ~800 numeric tokens (incl. config/protocol constants), ~60 EXT-only. Findings: 6 MISMATCH (A1–A6), 2 NO SOURCE / narrative-only (A7, A8), 2 internal inconsistencies (A9, A10), 4 provenance/stale-raster flags (A11–A14). Every Table 1, tab:sparse236, tab:scorer, tab:masker, tab:factors, tab:strata, tab:gtbox, tab:strata-maskiou, tab:rpn, tab:neon-detection cell MATCHES its source.

## (A) MISMATCH / NO SOURCE / inconsistency

| # | line | quoted text | expected | source |
|---|---|---|---|---|
| A1 | 1215 | "Detectree2 emits up to $1{,}511$ detections per tile" | **1,521** at 2048 | `NR/preds_dt2_s0v2_r2048.json` max per tile = 1,521 (also `NR/scored_dt2_s0v2_r2048.json:areas`). 1,511 is the stale 512 stitch `detectree2_baseline/preds_dt2_s0v2.json`. SEAM row. |
| A2 | 1215 | "Restor and Box2Mask ... at most $377$ and $584$" | **592** (Swin-L) at 2048 | `NR/preds_box2mask_swinl_s0_r2048.json` max = 592 (R-50 527). 584 is stale 512 (`box_supervised_baselines/preds_box2mask_swin-l_s0_test.json`). 377 MATCH. SEAM row. |
| A3 | 655 | "only $1.2\%$ actually hold more than $100$ ground-truth crowns. Removing every tile containing such a subtile improves by $+0.007$" | 1.0% (39/3,951) for the definition that gives +0.007 | `NR/b2m_cap_subset.json`: n_over_cap 39 of 3,951 → 0.99%, delta_AP50 (2048) 0.007. The 1.2% is `box_supervised_baselines/PLAN.md` L414 (46/3,951 = 1.16%, different definition). Two definitions in one sentence. SEAM row. |
| A4 | 1118 | "mean crown IoU spans $0.0013$" (across the $K_{fg}=16$ block) | **0.0012** | tab:collapse K16 pred cells 0.7205/0.7210/0.7217/0.7208 → 0.0012; unrounded (`NR/components_guided_raw/em_k16_s0_b{0,0.1,0.25,0.5}_c1_fix_pred.jsonl`) 0.72048–0.72165 = 0.00118. 0.0013 is the span over all six rows (incl. K=2). Fix wording or number. |
| A5 | 737 | tab:sparse-biomes, biome 12 "Canopy px $7.6\%$" | **7.8%** | `modal_sparse_tcd_multiseed/tile_index.json` biome 12 ∩ `sparse_gt.json` (56 tiles): canopy 0.0777. 7.6% (0.0764) includes the dropped unreadable `tcd_tile_231` (57 tiles). Crown px 4.8% unaffected. |
| A6 | 1157 | "Score floor $0.05$, applied to every row before scoring" | LACE product rows scored at floor **0** | `G{0,1,2}:protocol.score_floor = 0.0`, also SG*, `cgf_a0/a1` files. Floor is applied to $s$ at decode (L542 says so), not before scoring. Wording contradicts the protocol blocks. |
| A7 | 1149 | "placed $0.60$, $0.75$ and $0.85$ one ulp high ... worth $0.0005$ AP$_{50:95}$" | — | NO SOURCE FOUND. Closest: `ablation/results/cocoeval_parity.json` +0.0008 AP50:95 for arange+float32 combined, at 512, canopy=FP arm. |
| A8 | 800, 801, 803 | NIWO site F1 $0.427 \pm 0.010$, $0.588 \pm 0.017$, $+0.161$ | — | NARRATIVE ONLY: `modal_neon_multiseed/phase4/README.md` L77. No per-seed per-site JSON persisted (would need re-scoring `phase4/dl/preds_phase4_s*.json`). Global cells of the same table do trace (pr_curves.json). |
| A9 | 358 vs 654 | "Swin-L ... achieves $42.4$ mask AP on COCO" vs "(Swin-L, $42.5$ COCO mask AP)" | one value | `~/dphil/refs/lace_bib_pdfs/box2mask.pdf`: 42.4% (paper). 42.5 = model-zoo figure (`box_supervised_baselines/PLAN.md` L342/L543). Internal inconsistency. |
| A10 | 555 vs 1155 | "densest tile holds $422$ individual crowns" vs "densest test tile holds $450$ ... ($421$ crowns plus $29$ canopy)" | both correct | `test_gt.json`: tile_104 = 422 crowns; tile_113 = 421+29 = 450 instances. Different tiles; reads as a contradiction. Suggest "the tile with most instances". |
| A11 | 515, 827 | "$16$ is simply the best of $\{10,13,16,20\}$ on validation ... varies by under $0.005$"; "$\alpha<1$ beats $\alpha=1$ at every $\kappa$, and $\kappa>10$ beats $\kappa=10$ at every $\alpha$" | values MATCH, raster STALE | `modal_tcd_multiseed/phase4/sweep_val_knobs_s0.json` (α=0.3: 0.6348/0.6366/0.6392/0.6382, range 0.0044) is the **512-raster, bilinear, τ 0.25** val sweep. α, κ never re-selected at 2048 (NATIVE_RASTER L487, "Phase B not yet run"). Paper presents all selection as current. |
| A12 | 706 | "median tile is $7.5\%$ ... $59.3\%$ of its tiles" | values MATCH on 236; stored file disagrees | `ablation/results/slice_fao_cover.json` is computed on **237** tiles (0.0744 → 7.4%, 0.5949 → 59.5%). Recomputed on the 236 (`tile_index.json` ∩ `sparse_gt.json`): 0.0747 → 7.5%, 0.5932 → 59.3%. Committed source file does not reproduce the paper. 31.0% MATCH. |
| A13 | 833 | tab:components caption "$K_{fg}=16$" | generative/absolute cell is K_init 16, **K_final 14** | `modal_tcd_multiseed/phase4_collapse/results/factorial_summary.json` (β=0 absolute: k_final 14); own comment L1121 says K=14. |
| A14 | 879–888 | tab:factors AP values | crowd arm differs at 4th dp | `AR/confidence_ablation_results_r2048_guided.json:forms` = score_coco **legacy** arm (product 0.6473 / 0.2825; s 0.2584). Crowd arm G0/D0: 0.6472 / 0.2821; 0.2579. Contradicts L1133 "every OAM-TCD figure ... one scorer under one configuration". Values themselves MATCH the file. |

Rounding notes (not mismatches): L899 "+0.0365 against +0.0333" = differences of 4-dp-rounded means (0.6775−0.6410; 0.6470−0.6137); unrounded 0.0366 / 0.0332; at the 3-dp of tab:scorer 0.037 / 0.033. tab:components effects are differences of the tabulated 4-dp cells; unrounded oracle effects 0.0080 / 0.0087 / 0.0022 / −0.0145.

Stale text in comments (not rendered; stale if uncommented): L893–897 (512-era 0.5613, 0.6250, 0.6616, 0.6615, 0.2797, 0.2822, −0.00005, [−0.0016,+0.0016], 0.6587, 0.6495, +0.0244, +0.0365, +0.012, 0.6605; guided-2048 equivalents in `AR/confidence_ablation_results_r2048_guided.json`: logreg8 0.647 vs 0.6473, bootstrap −0.00031 [−0.00203,+0.00117], s+SIZE 0.6352, no-detector 0.5482); L1121 (0.2680, 0.0014, 0.6406, 0.6392, 0.5790, 0.6203, 0.605 are 512 sources); L1071 (claims both mask-supervised models trail LACE on AP50 — fine now, seam-dependent).

Provenance gap: `results_439/r2048/table_r2048.md` (+`build_table_r2048.py`), named in NATIVE_RASTER as the consolidated Table 1 source, predates the guided render: no guided LACE row, SelvaBox 2048 "NOT PRODUCED". Table 1 traces directly to scored JSONs instead (all MATCH).

Protocol blocks (check 2): every file below has `res 2048`, `max_dets 600` (except the two declared budget files), `n_tiles 439` (236 sparse), crowd key present. Floor 0.0: G*, SG*, `NR/scored_lace_s*_r2048_bilinear_tau040_guidedrun.json`, `cgf_a0_k1.6`, `cgf_a1_k1`. Floor 0.05: D*, SD*, Restor tree/pooled rpn512/1000, DT2 (439, sparse, maxdets1200), Box2Mask ×2, SelvaBox md600/md1600. maxDets 1200: `scored_dt2_s0v2_r2048_maxdets1200.json`; 1600: `scored_selvabox_bench_r2048_md1600.json` — both used only where the paper says so (L653, L1215). All consistent with the paper except A6 wording. The crowd rule is not recorded in the protocol block itself, only by the key name.

## (B) Trace table

| line | number | source file:key | status |
|---|---|---|---|
| 126 | 0.1 m/px | OAM-TCD | EXT |
| 126 | 0.647 ± 0.003 (3 seeds) | G0–2:crowd.mask.AP50 [0.6472,0.6444,0.6493] → 0.64697 ± 0.00246 | MATCH |
| 126 | 900 | train split | MATCH |
| 126 | 0.648 | NR/scored_restor_rpn1000_r2048_tree.json:crowd.mask.AP50 0.6480 | MATCH |
| 126 | ~4.2k | 4169, PROTOCOL_439.md L25 | MATCH |
| 126 | 0.678 | SG0–2:crowd.mask.AP50 [0.6889,0.6671,0.6766] → 0.67753 | MATCH |
| 126 | 0.625 | NR/scored_dt2_sparse_s0v2_r2048.json:crowd.mask.AP50 0.6252 | MATCH (SEAM) |
| 126 | 0.728 ± 0.003 (5 seeds) | modal_neon_multiseed/phase4/dl/pr_curves.json:phase4 best-F1 [0.7219,0.7282,0.7294,0.7296,0.7297] → 0.7278 ± 0.0033 | MATCH |
| 126 | 23,424 | NeonTreeEvaluation/annotations/*.xml objects over the 18 `src_tile` of train_patches_gt.json | MATCH (recounted) |
| 126 | 0.719 | Weinstein 2021 T3; band_df/results_deepforest.json target_paper | MATCH/EXT |
| 126 | <0.1% | 23,424 / >30.01M = 0.078% | MATCH |
| 126 | 30M | DeepForest | EXT |
| 188 | 1/4; 29% | crowther2015; reiner2023 | EXT |
| 195 | 1024 px | figure | n/a |
| 202 | 1.8 bn; 1.3 M km²; 0.5 m; 9.9 bn; 0.25 m; 48.6%; 38.4%; 25.5% | cited works | EXT |
| 211 | ≤40% | ball2023 | EXT |
| 218 | 5–10 m; <1; ≤0.1 | oamtcd | EXT |
| 218 | 439; 25,705 | test_gt.json (439 keys, 25,705 trees) | MATCH |
| 218 | 25.0% | ablation/results/gsd_downsample.json:rows.1.0m.frac_zero_enclosed_px 0.2502 | MATCH |
| 218 | 57.0% | rows.2.0m.frac_zero_enclosed_px 0.5700 | MATCH |
| 218 | 22.6% | rows.2.0m.frac_bbox_side_lt_1px 0.2263 | MATCH |
| 220 | 9.9 bn | tucker2023 | EXT |
| 227 | 39–47%; 2011; 41.5%; 10–23%; 2100 | cited works | EXT |
| 239, 266, 300 | 1.7 bn | dinov3 | EXT |
| 245 | 3.2M params | detector.py Detector8 (stem 1024→256 + 3 tower + 2 up 3×3 + heads) = 3,217,157 | MATCH (computed) |
| 247 | 0.647 ± 0.003; 0.728 ± 0.003; 0.4 | as L126 | MATCH |
| 249 | 13 of 22 sites | train_patches_gt.json (13 sites) / neon_gt.json (22 sites) | MATCH |
| 249 | 0.719; 1 m | DeepForest / NEON CHM | EXT |
| 280 | 118k; 1.28M | COCO / ImageNet | EXT |
| 284 | 30 M; 22; 10,000 | DeepForest | EXT |
| 307 | 83,000 | selvabox | EXT |
| 330–345 | GSD cm 10/5/<5/1–4/1–5/≤10 | tab:gap | EXT |
| 352 | WWF codes 1–14 | WWF | EXT |
| 358 | 33.2; 39.1 | boxinst | EXT |
| 358 | 42.4 | box2mask.pdf "42.4% mask AP" | EXT; INCONSISTENT with L654 (A9) |
| 384, 388 | p=16; 8 px; i,j ≤ 127 | 2048/16 = 128 | MATCH |
| 392–394 | 8X+8, 4 px border | phase4_features.py (comment) | MATCH (doc) |
| 396 | 24; 1024; 21–24; 4096; 256×256; 2048 | config | MATCH |
| 396 | 200 positions; ≈0.955 | modal_neon_multiseed/phase4/README.md L95 | MATCH |
| 403 | 1×1→256; five 3×3; 3.2M; 4.0M | detector.py; NEON stem 4096→256 → 4,003,589 | MATCH (computed) |
| 406 | 0.1; 0.7; 0.05; 600; 0.4 | config / protocol | MATCH |
| 418 | d = 128 | ablation/results/em_fit_report_4p_fix.json:pca 128 | MATCH |
| 440 | π/4 ≈ 0.785; 0.95 | config | MATCH |
| 440 | 0.690 | em_fit_report_4p_fix.json:mean_fg_mass_per_bin 0.6899 | MATCH |
| 477 | β = 0.5 | em_fit_report | MATCH |
| 494–508 | T = 30; t > 5; 0.25/K_fg; two | config (alg:fit) | not traced (code) |
| 513 | 2048; r 12 px; 1.5 cells; ε 1e-4 | NATIVE_RASTER prereg L876 | MATCH |
| 515 | κ 10 → 16 | em_fit_report kappa 10 × kappa_scale 1.6 | MATCH |
| 515 | {10,13,16,20}; <0.005 | modal_tcd_multiseed/phase4/sweep_val_knobs_s0.json α=0.3 range 0.0044 | MATCH, STALE-RASTER (A11) |
| 517 | 0.5; 108; 900; 792 | split | MATCH |
| 542, 544 | 0.05; 108 | protocol | MATCH |
| 553 | 5072; 2048; 0.1; 280k; 56k | oamtcd | EXT |
| 555 | 5072; 439 | oamtcd / test_gt.json | MATCH |
| 555 | 25,705; 366; 73; 6,782; 422; 29 | test_gt.json (trees total; ≥1 crown; 0 crowns; canopy total; max = tile_104; median) | MATCH ×6 |
| 562 | 22 | neon_gt.json | MATCH |
| 565 | @1.8.0; 194; 6,634; 22 | modal_neon_multiseed/deepforest_repro.json (pinned_tag, n_plots, n_gt_boxes); neon_gt.json | MATCH ×4 |
| 565 | 18 tiles; 13 sites; 23,424; 2,063; 400 px; 26,800 | train_patches_gt.json + annotations XML | MATCH ×6 |
| 565 | 0.3× | prepare_neon_train.py | not traced |
| 565 | SJER 41.8%; TOOL 0% | modal_neon_multiseed/README.md L267–268 | MATCH (narrative) |
| 565 | every 8th; 1,807 / 256 | train_patches_gt.json:partition (1807/256) | MATCH |
| 567 | 0.1; 400; 400; 512; 32×32; 64×64; 21–24; 4096 | NEON pipeline | MATCH |
| 584–606 | layer 24; 1024; 21–24; 4096; 4 offsets {0,8}; 8 px; 256×256; 2048; width 256; 5×; 3×3; 4-ch; stride 8; grid 256; 0.7; ≥1 | config / detector.py | MATCH |
| 592 | Adam lr 1e-3; wd 1e-4; batch 3; max 40; min 12; patience 2 | neon_train_lib.py recipe (lr, wd, bs 3, min12/pat2); 40 epochs not traced | MATCH (5) / not traced (1) |
| 594 | 0.05; 600; 0.4 | protocol | MATCH |
| 596–604 | 120 tiles; d 128; K_fg 16; K_bg 12; κ 10; β 0.5; N 8; 3 bands; π/4; 0.95; 30 its; 5; 0.25/K; α 0.3; κ 16 (10); r 12; ε 1e-4; τ 0.35 | em_fit_report_4p_fix.json (k_effective 16, k_bg 12, pca 128, kappa 10, bins 8, size_bands 3, prior_weight 0.3, kappa_scale 1.6); prereg NATIVE_RASTER L874–877 | MATCH (120 tiles: RESULTS.md phase4_collapse "120 training tiles") |
| 619 | cat=2; cat=1; 10; 1; 2; 5; 0.5 | definitions | n/a |
| 624 | 439; 2048² | protocol | MATCH |
| 636 | Restor 4169 | PROTOCOL_439.md | EXT/MATCH |
| 636 | 0.648 / 0.333 / 0.665 | NR/scored_restor_rpn1000_r2048_tree.json crowd mask AP50 0.6480, AP50_95 0.3332, box AP50 0.6646 | MATCH ×3 |
| 637 | DT2 900; 0.614 / 0.301 / 0.639 | NR/scored_dt2_s0v2_r2048.json 0.6143 / 0.3014 / 0.6387 | MATCH ×3 (SEAM) |
| 639–640 | SelvaBox 3024; 0.571 / 0.211 / 0.732 | NR/scored_selvabox_bench_r2048_md600.json 0.5710 / 0.2111 / 0.7318 | MATCH ×3 (SEAM) |
| 642 | B2M Swin-L 900; 0.563 / 0.231 / 0.602 | NR/scored_box2mask_swinl_s0_r2048.json 0.5625 / 0.2309 / 0.6017 | MATCH ×3 (SEAM) |
| 643 | B2M R-50 900; 0.433 / 0.157 / 0.497 | NR/scored_box2mask_r50_s0_r2048.json 0.4332 / 0.1566 / 0.4972 | MATCH ×3 (SEAM) |
| 644–647 | LACE 3-seed; 0.647 ± 0.003 | G0–2 → 0.64697 ± 0.00246 | MATCH |
| 645–647 | 0.278 ± 0.005 | G0–2:crowd.mask.AP50_95 [0.2821,0.2722,0.2791] → 0.27780 ± 0.00508 | MATCH |
| 645–647 | 0.667 ± 0.011 | G0–2:crowd.box.AP50 [0.6748,0.6543,0.6706] → 0.66657 ± 0.01083 | MATCH |
| 651 | 0.634 / 0.328 | NR/scored_restor_rpn1000_r2048_pooled.json 0.6342 / 0.3279 | MATCH |
| 651 | 3; 0.2→0.05; 512→600; 512→1000 | restor_native_modal config | MATCH (config) |
| 651 | 0.585 / 0.308 | NR/scored_restor_rpn512_r2048_tree.json 0.5847 / 0.3076 | MATCH |
| 652 | 15014353; 6.0 epoch; 1000; 800 | detectree2_baseline/RETRAIN_V2.md L116 (iter 10000 = 6.0 ep) | MATCH |
| 653 | 1024²; 0.5; 0.1; T19; 3024 | PROTOCOL_439.md L35–38 | EXT/MATCH |
| 653 | 400; 1600 | PROTOCOL_439.md L188 (area-equivalent) | MATCH |
| 653 | 0.579 / 0.213 / 0.752 | NR/scored_selvabox_bench_r2048_md1600.json 0.5789 / 0.2132 / 0.7521 | MATCH ×3 (SEAM) |
| 654 | 42.5; 35.9 | box_supervised_baselines/PLAN.md L342 (zoo) | EXT; 42.5 INCONSISTENT with L358 (A9) |
| 654 | R-50 stop ep 6, select ep 4 | box_supervised_baselines/train_box2mask_s0.json best iter 13384 / 3346 = 4.0; valcurve 12 half-epoch evals = 6 | MATCH |
| 654 | Swin-L peak 3.5; three later evals | train_box2mask_swin-l_s0.json best iter 23422 / 6692 = 3.5; valcurve logs 0.596 then 0.590 ×3 | MATCH |
| 655 | batch 16; lr 1e-4 | train_box2mask_*.json:sched.effective_batch 16, lr 0.0001 | MATCH |
| 655 | 100; 1024²; 50%; 3×3; 2048² | config | MATCH |
| 655 | 28.8%; 2.7% | box_supervised_baselines/PLAN.md L411–412 | MATCH (SEAM context) |
| 655 | 1.2% | PLAN.md L414 (46/3951) vs NR/b2m_cap_subset.json (39/3951 = 1.0%) | MISMATCH-def (A3) |
| 655 | +0.007 | NR/b2m_cap_subset.json:res.2048.delta_AP50 0.007 | MATCH (SEAM) |
| 664 | 768 px; 0.1 | figure | n/a |
| 670 | 0, 1, 2; 3 | seeds | MATCH |
| 672 | 439 | test_gt.json | MATCH |
| 674 | 900; 0.563; 0.647 | Swin-L file 0.5625; G mean | MATCH (SEAM) |
| 674 | 0.647 ± 0.003 vs 0.648; box level | G vs Restor tree; box 0.667 vs 0.665 | MATCH |
| 678 | 0.4; 194 | protocol | MATCH |
| 684 | 0.659 / 0.790 / 0.719 | Weinstein 2021 T3; band_df/results_deepforest.json target_paper_P/R 0.659/0.79; F1 = 0.7186 | MATCH/EXT |
| 685 | 0.745 / 0.709 / 0.726 / 0.765 | modal_neon_multiseed/band_df/results_deepforest.json best_f1_point P 0.7451, R 0.7085 (F1 0.72636); pr_curve max R 0.7647 | MATCH ×4 |
| 686 | 0.727 ± 0.009 | pr_curves.json:phase4 best-F1 P 0.7272 ± 0.0093 | MATCH |
| 686 | 0.729 ± 0.009 | R 0.7285 ± 0.0088 | MATCH |
| 686 | 0.728 ± 0.003 | F1 0.7278 ± 0.0033 | MATCH |
| 686 | 0.849 ± 0.003 | maxR 0.8491 ± 0.0027 | MATCH |
| 690 | 2021; T3; v2.1.0; 194 | — | MATCH |
| 706 | 236; 14,937; 900 | sparse_gt.json 236 keys / 14,937 trees; SG*:n_gt_trees 14937 | MATCH |
| 706 | 7.5% | recomputed on 236 = 0.0747; stored slice_fao_cover.json (237) = 0.0744 | MATCH on 236 / source STALE (A12) |
| 706 | 10% | FAO | EXT |
| 706 | 59.3% | recomputed 236 = 0.5932; stored 0.5949 | MATCH on 236 / source STALE (A12) |
| 706 | 31.0%; six | slice_fao_cover.json:forest_biomes_1_to_6.frac_below_10pct 0.3097 | MATCH |
| 706 | 366 | results_439/gtbox_bakeoff_r2048.json:n_tiles | MATCH |
| 709 | 236; 14,937; 900; 0–2 | as above | MATCH |
| 718 | DT2 0.6252 / 0.3460 / 0.6331 | NR/scored_dt2_sparse_s0v2_r2048.json crowd 0.6252 / 0.3460 / box 0.6331 | MATCH ×3 (SEAM) |
| 719–722 | 0.6775 ± 0.0109 | SG0–2 AP50 → 0.67753 ± 0.01093 | MATCH |
| 719–722 | 0.3215 ± 0.0079 | SG0–2 AP50_95 [0.3282,0.3128,0.3236] → 0.32153 ± 0.00791 | MATCH |
| 719–722 | 0.6783 ± 0.0152 | SG0–2 box AP50 [0.69,0.6612,0.6838] → 0.67833 ± 0.01516 | MATCH |
| 725 | 2048²; 600 | protocol | MATCH |
| 725 | 0.678 vs 0.647 | SG mean / G mean | MATCH |
| 725 | 0.625 vs 0.614 | DT2 sparse / DT2 439 | MATCH (SEAM) |
| 729 | 900; 12; 236 | README data/tcd_sparse L75 (tcd_tile_231 dropped) | MATCH |
| 736–741 | tiles 102 / 56 / 31 / 24 / 16 / 7 | tile_index.json ∩ sparse_gt.json | MATCH ×6 |
| 736–741 | crown px 2.3 / 4.8 / 4.8 / 6.1 / 1.1 / 1.0 | tile_index.json:crown_frac means (236) | MATCH ×6 |
| 736–741 | canopy px 8.6 / 7.6 / 11.5 / 15.2 / 1.9 / 0.8 | tile_index.json:canopy_frac means (236) | MATCH ×5; biome 12 7.6 → 7.8 MISMATCH (A5) |
| 743 | 236 | sum | MATCH |
| 746–755 | excluded crown/canopy px (20 values) | tile_index.json unseen tiles per biome | MATCH ×20 |
| 763 | 439; 25,692; 13; 2048² | NR/gtbox_guided_raw/gtbox_lace_guided_r2048.jsonl (25,692 valid, 13 invalid) | MATCH |
| 763 | 0.6 m² | gtbox_lib threshold | not traced |
| 771 | 0.748 / 0.940 / 0.647 / 0.051 | gtbox_lace_guided_r2048.jsonl recomputed 0.7477 / 0.9396 / 0.6467 / 0.0510 | MATCH ×4 |
| 773 | 0.741 / 0.956 / 0.545 / 0.037 | results_439/gtbox_sam3_base_r2048.json:summary 0.7406 / 0.9555 / 0.5452 / 0.0368 | MATCH ×4 |
| 774 | 0.623 / 0.789 / 0.274 / 0.011 | results_439/gtbox_sam3_ft_r2048.json:summary 0.6232 / 0.7885 / 0.2741 / 0.0107 | MATCH ×4 |
| 778–779 | 2048²; 1777; 685 | PROTOCOL_439.md L225, L292 | MATCH |
| 791 | 16×16; 4; 8 px; grid 64; stride 8 | phase4/README.md | MATCH |
| 794 | 194; 5; 0.4 | — | MATCH |
| 800 | 0.689 ± 0.013 | pr_curves.json:native F1 0.6887 ± 0.0125 | MATCH |
| 800 | 0.779 ± 0.009 | native maxR 0.7786 ± 0.0088 | MATCH |
| 800 | 0.427 ± 0.010 | phase4/README.md L77 only | NO SCORED SOURCE (A8) |
| 801 | 0.728 ± 0.003; 0.849 ± 0.003 | pr_curves.json:phase4 | MATCH |
| 801 | 0.588 ± 0.017 | phase4/README.md L77 only | NO SCORED SOURCE (A8) |
| 803 | +0.039; +0.070 | 0.728−0.689; 0.849−0.779 | MATCH |
| 803 | +0.161 | 0.588−0.427 (README) | MATCH arithmetic / A8 |
| 806 | cos ≈ 0.955; ≤1 cell; 200 | phase4/README.md L95 | MATCH |
| 806 | ~10× | 0.161 / 0.017 = 9.5 | MATCH |
| 813 | 10 → 16; τ 0.35 | config | MATCH |
| 816 | 439; 108; 0–2 | — | MATCH |
| 822 | 0.6240 ± 0.0029 / 0.2581 ± 0.0031 | NR/scored_lace_s{0,1,2}_r2048_cgf_a0_k1.6.json → 0.62397 ± 0.00291 / 0.25807 ± 0.00306 | MATCH ×2 |
| 823 | 0.6193 ± 0.0081 / 0.2558 ± 0.0029 | NR/scored_lace_s{0,1,2}_r2048_cgf_a1_k1.json → 0.61927 ± 0.00812 / 0.25577 ± 0.00294 | MATCH ×2 |
| 824 | 0.6470 ± 0.0025 / 0.2778 ± 0.0051 | G0–2 | MATCH ×2 |
| 827 | 0.023 AP50; 0.020 AP50:95 | 0.6470−0.6240; 0.2778−0.2581 = 0.0197 | MATCH ×2 |
| 827 | α/κ directional claims; 108 | sweep_val_knobs_s0.json (512) | MATCH, STALE-RASTER (A11) |
| 833 | K_fg 16; seeds 0, 0; 108; 7041; 5428 | NR/components_guided_raw/*_summary.json:n_tiles 108, n_matched 7041 / 5428 | MATCH (K caveat A13) |
| 839 | 0.7449 / 0.6896 | em_model_4p_b0_fix_{gt,pred}_summary.json:mean_crown_iou_2048_guided | MATCH ×2 |
| 840 | 0.7537 / 0.7205 | em_k16_s0_b0_c1_fix_{gt,pred} | MATCH ×2 |
| 841 | 0.7530 / 0.7156 | em_k16_s0_b0.5_c0_fix_{gt,pred} | MATCH ×2 |
| 842 | 0.7472 / 0.7208 | em_k16_s0_b0.5_c1_fix_{gt,pred} | MATCH ×2 |
| 844–847 | +0.0081 / +0.0260; +0.0088 / +0.0309; +0.0023 / +0.0312; −0.0146 / −0.0257 | differences of tabulated cells | MATCH ×8 |
| 850 | +0.03; +0.031; +0.0569; +0.0312 | 0.0260+0.0309; table | MATCH ×4 |
| 850 | 0.37 → 0.61; 0.37 → 0.41 | boxinst_commonality/README.md L105–120 (dryland dev set, 16 px); phase4_collapse/RESULTS.md L34 | MATCH (dev set, disclosed) |
| 857 | 3 seeds | — | MATCH |
| 866 | 0.614 ± 0.003 / 0.256 ± 0.003 | D0–2 (floor 0.05) → 0.61373 ± 0.00317 / 0.25570 ± 0.00255 | MATCH ×2 |
| 867 | 0.647 ± 0.003 / 0.278 ± 0.005 | G0–2 | MATCH ×2 |
| 869 | DT2 0.625 / 0.346 | NR/scored_dt2_sparse_s0v2_r2048.json | MATCH ×2 (SEAM) |
| 870 | 0.641 ± 0.014 / 0.295 ± 0.008 | SD0–2 → 0.64097 ± 0.01362 / 0.29460 ± 0.00829 | MATCH ×2 |
| 871 | 0.678 ± 0.011 / 0.322 ± 0.008 | SG0–2 | MATCH ×2 |
| 874 | 3; ddof 1 | — | MATCH |
| 876 | +0.0237; +0.0228; +0.0372; "a third" | AR/confidence_ablation_results_r2048_guided.json:forms (0.6338/0.6329/0.6473 − 0.6101); loss 36% / 39% | MATCH ×4 |
| 879 | seed 0; 439; 108; 159,687 | n_boxes 159687; G0 n_det | MATCH |
| 885 | 0.6101 / 0.2584 / 0.4982 | forms."s  (baseline)" | MATCH ×3 (legacy arm, A14) |
| 886 | 0.6338 / 0.2714 / 0.5148 | forms."s * bimod" | MATCH ×3 |
| 887 | 0.6329 / 0.2741 / 0.5094 | forms."s * pfg_mean" | MATCH ×3 |
| 888 | 0.6473 / 0.2825 / 0.5177 | forms."s * bimod * pfg_mean" | MATCH ×3 (legacy; crowd 0.6472 / 0.2821) |
| 899 | 236; +0.0365; +0.0333 | 4-dp means (see rounding note) | MATCH |
| 906 | 39.3%; 25,705 | AR/strata_instance_r2048_guided.json:gt_level.touching_rate 0.3932 | MATCH |
| 906 | 0.7078 ± 0.0026 / 0.7118 ± 0.0018 | band_mean_std_boxmatched.mean_mask_iou | MATCH ×2 |
| 906 | 0.0373 ± 0.0031 / 0.0403 ± 0.0007; 0.5 | band_mean_std_boxmatched.mask_fail_rate | MATCH ×2 |
| 906 | 70.2 / 44.1 px | per_seed.0.by_touching_boxmatched.mean_gt_size_px 70.18 / 44.08 | MATCH (seed 0 only, unstated) |
| 911 | 439; 3 | — | MATCH |
| 917–920 | tiles 196 / 95 / 86 / 62 | AR/strata_tile_product_r2048_guided.json:band_mean_std.closure_canopy_frac.*.n_tiles | MATCH ×4 |
| 917–920 | mask 0.5667 / 0.6750 / 0.6993 / 0.6391 | …mask_mAP50[0] | MATCH ×4 |
| 917–920 | box 0.5610 / 0.6653 / 0.6899 / 0.6227 | …box_mAP50[0] | MATCH ×4 |
| 917–920 | +0.0057 / +0.0097 / +0.0094 / +0.0163 | …mask_minus_box_AP50[0] | MATCH ×4 |
| 917–923 | bin edges 0.10 / 0.25 / 0.50 / 1.00; 439 | closure_edges (last 1.01) | MATCH |
| 926 | 0.5667; 0.6993; +0.006 to +0.016 | as table | MATCH ×3 |
| 940 | 439; 0.647; 0.648; 0.333 vs 0.278; 900; 4169 | G / Restor tree / PROTOCOL_439 | MATCH ×6 |
| 940 | 1.7 bn; 7B | dinov3 | EXT |
| 942 | 8 px; eight | — | MATCH |
| 942 | 0.017 | 0.64697 − 0.63000 = 0.0170 | MATCH |
| 942 | 0.630 ± 0.003 | NR/scored_lace_s{0,1,2}_r2048_bilinear_tau040_guidedrun.json AP50 → 0.63000 ± 0.00333 | MATCH |
| 942 | AP75 0.173 vs 0.297 | G0–2:crowd.mask.AP75 → 0.17300; Restor tree AP75 0.2968 | MATCH ×2 |
| 942 | blur / wrong-tile < bilinear (val) | NR/guided_control_val_r2048.json, guided_control_lowtau_val_r2048.json (NATIVE_RASTER L851–858) | MATCH |
| 945 | 1.3–3.5 cm/px; 10 cm/px | selvamask | EXT |
| 945 | best-in-class box AP50 | SelvaBox 0.732 = max Table 1 box column | MATCH (SEAM) |
| 1074 | 439; 11,076 | AR/box_iou_strata_guided_r2048.json:n_common | MATCH |
| 1084 | Restor 0.797 / 0.789 / 0.611 / 0.695 / 0.765 / 0.825 / 0.872 | common_subset."Restor MRCNN rpn1000" (0.797, 0.7894, 0.6113, 0.6949, 0.7646, 0.8254, 0.8715) | MATCH ×7 |
| 1085 | DT2 0.783 / 0.776 / 0.599 / 0.682 / 0.755 / 0.825 / 0.881 | common_subset."DetecTree2 s0" | MATCH ×7 (SEAM) |
| 1086 | SelvaBox 0.807 / 0.690 / 0.538 / 0.608 / 0.656 / 0.709 / 0.783 | common_subset."SelvaBox -> SAM 3" | MATCH ×7 (SEAM) |
| 1087 | B2M Swin-L 0.782 / 0.716 / 0.548 / 0.640 / 0.701 / 0.756 / 0.813 | common_subset."Box2Mask Swin-L" | MATCH ×7 (SEAM) |
| 1088 | B2M R-50 0.769 / 0.662 / 0.521 / 0.589 / 0.645 / 0.712 / 0.786 | common_subset."Box2Mask R-50" | MATCH ×7 (SEAM) |
| 1089 | LACE 0.752 / 0.748 / 0.614 / 0.691 / 0.753 / 0.802 / 0.836 | common_subset."LACE s0 (product)" | MATCH ×7 |
| 1092 | 371 to 4,659; seed 0; 8 px | min/max of table rows' bin n (Restor 371, 4659) | MATCH |
| 1097 | β 0.5; 0.994; 1.66 of 16 | modal_tcd_multiseed/phase4_collapse/RESULTS.md L45 | MATCH ×3 |
| 1100 | 108; seeds 0, 0 | — | MATCH |
| 1109–1115 | K init/final 16/16, 16/15, 16/13, 16/16, 2/2, 2/2 | RESULTS.md L42–45, L53–56 | MATCH ×12 |
| 1109–1115 | eff. rank 11.43 / 9.41 / 6.37 / 1.66 / 2.00 / 1.30 | RESULTS.md (factorial_fits.json 11.4339) | MATCH ×6 |
| 1109–1115 | cos 0.234 / 0.334 / 0.525 / 0.994 / 0.053 / 0.988 | RESULTS.md | MATCH ×6 |
| 1109–1115 | oracle 0.7537 / 0.7541 / 0.7539 / 0.7472 / 0.7486 / 0.7470 | components_guided_raw em_k16_s0_b{0,0.1,0.25,0.5}_c1, em_k2_s0_b0_c1, em_k2_s0_b0.5 _gt_summary | MATCH ×6 |
| 1109–1115 | pred 0.7205 / 0.7210 / 0.7217 / 0.7208 / 0.7204 / 0.7210 | same, _pred_summary | MATCH ×6 |
| 1118 | sevenfold | 11.43 / 1.66 = 6.9 | MATCH |
| 1118 | 0.0013 | K16 block span 0.0012 | MISMATCH (A4) |
| 1118 | 0.0021 | EM seeds 0.7208 / 0.7229 / 0.7221 (em_k16_s{1,2}_b0.5_fix_pred) unrounded 0.00212 | MATCH |
| 1118 | ≈0.007 | 0.7537–0.7541 vs 0.7472 | MATCH |
| 1118 | 0.9942; 0.9915; 0.9922; three | phase4_collapse/results/stage1_gate.json:em_init_variance | MATCH ×3 |
| 1122 | 0.5; three; two; ≥0.9908; sixteen | stage1_gate.json within_k2_cos_min 0.9908 (global min) | MATCH |
| 1122 | 0.7208 → 0.6896 | tab:components | MATCH |
| 1126 | 0.7208 to 0.7229; three | components_guided_raw EM seeds | MATCH |
| 1133 | 0.432 vs 0.648 | PROTOCOL_439.md L53 / Restor tree | MATCH/EXT |
| 1143–1145 | 439; 2048² | — | MATCH |
| 1147 | cat=2 | — | MATCH |
| 1149 | linspace(0.5,0.95,10); 0.60 / 0.75 / 0.85 | protocol.iou_thrs | MATCH |
| 1149 | 0.0005 | — | NO SOURCE (A7) |
| 1151 | 101 | protocol.rec_thrs | MATCH |
| 1155 | 600 vs 100; 450 = 421 + 29 | protocol.max_dets; test_gt.json tile_113 | MATCH ×4 (A10 wording) |
| 1157 | 0.05 "every row" | LACE product protocol.score_floor 0.0 | MISMATCH wording (A6) |
| 1159 | 2048²; 0.1 | protocol.res | MATCH |
| 1161 | cat=1; iscrowd=1 | crowd key | MATCH |
| 1163 | 439 | n_tiles | MATCH |
| 1168 | 0.50 | — | MATCH |
| 1170 | 439; 73; 51; 25,705 | test_gt.json | MATCH ×4 |
| 1170 | 4.0% LACE | G0:areas on the 73 = 6,453 / 159,687 = 4.04% (s1 3.95%, s2 5.61%; seed unstated) | MATCH (s0) |
| 1170 | 2.1% Restor | scored_restor_rpn1000_r2048_tree.json:areas 1,180 / 56,438 = 2.09% | MATCH |
| 1170 | 4.7% DT2 | scored_dt2_s0v2_r2048.json:areas 9,287 / 197,423 = 4.70% | MATCH (SEAM) |
| 1170 | 6.6% Box2Mask | scored_box2mask_swinl_s0_r2048.json:areas 9,151 / 139,328 = 6.57% (R-50 6.76%; SelvaBox 5.60% unlisted) | MATCH (SEAM) |
| 1183 | 2048²; 0.05; 600 | protocol | MATCH |
| 1185 | 2048²; 0.2→0.05; 512→600; 512→1000 | restor preds meta | MATCH |
| 1187 | 1024²; 0.5; 3×3; 1000; 800; 0.05; 0.1; 0.7; 0.85 | NR/preds_dt2_s0v2_r2048.json:meta.dedup {iou_thr 0.7, cont_thr 0.85, min_score 0.05} | MATCH (SEAM) |
| 1189 | 1024²; 0.5; 0.1; 1777; 0.045; 2.22×; 0.7 | PROTOCOL_439.md L170–171; 0.1/0.045 = 2.22 | MATCH (SEAM) |
| 1191 | 100; 1024² | config | MATCH (SEAM) |
| 1196 | 512; 1000; 422 | config; test_gt.json | MATCH |
| 1205 | 512: 97.9 / 0.6596 / 0.5847 / 0.3076 | AR/restor_rpn_recall_r2048.json + NR/scored_restor_rpn512_r2048_tree.json | MATCH ×4 |
| 1206 | 1000: 128.6 / 0.7441 / 0.6480 / 0.3332 | AR/restor_rpn_recall_r2048.json + NR/scored_restor_rpn1000_r2048_tree.json | MATCH ×4 |
| 1211 | 0.05; 2048²; 4,108 / 25,705; 16.0%; 0.84 | PROTOCOL_439.md L178; recomputed from test_gt.json (51.2 px frame band) = 4,108, 0.1598 | MATCH ×4 (SEAM context) |
| 1215 | 600; 422 | protocol; test_gt.json | MATCH |
| 1215 | 1,511 | 2048 file max 1,521 | MISMATCH (A1, SEAM) |
| 1215 | 1,600 | scored_selvabox_bench_r2048_md600.json:areas max 1600 | MATCH (SEAM) |
| 1215 | 1,200; 1,600; 0.004 (0.6143 → 0.6187) | NR/scored_dt2_s0v2_r2048{,_maxdets1200}.json | MATCH ×3 (SEAM) |
| 1215 | 0.008 (0.5710 → 0.5789) | NR/scored_selvabox_bench_r2048_md{600,1600}.json | MATCH ×3 (SEAM) |
| 1215 | ranking unchanged | 0.6187 < 0.647; 0.5789 vs B2M 0.5625 / DT2 0.6143 | MATCH (SEAM) |
| 1215 | 377 | scored_restor_rpn1000_r2048_tree.json:areas max 377 | MATCH |
| 1215 | 584 | 2048 Swin-L max 592 | MISMATCH (A2, SEAM) |
| 1217 | 0, 1, 2; five; 0–4; 0 | — | MATCH |

## (C) Seam-affected locations (Detectree2 = DT2, SelvaBox → SAM 3 = SB, Box2Mask Swin-L / R-50 = B2M)

| line | text | baseline |
|---|---|---|
| 126 | abstract "mask AP$_{50}$ rises to $0.678$ versus $0.625$ for Detectree2"; also "matches fully-supervised specialist baselines" | DT2 (sparse) |
| 194–195 | fig:qual-sparse `figs/qualitative_sparse.pdf` (GT, LACE, Detectree2 panels; figs/README L9, L36 best-F1 thresholds) | DT2 (sparse) — regenerate |
| 251 | "LACE is ... the strongest performing box-supervised model in the TCIS space" | B2M, SB (ordering claim) |
| 637 | Table 1 DT2 row $0.614$ / $0.301$ / $0.639$ | DT2 |
| 639–640 | Table 1 SB row $0.571$ / $0.211$ / $\mathbf{0.732}$ (bold box) | SB |
| 642 | Table 1 B2M Swin-L $0.563$ / $0.231$ / $0.602$ | B2M |
| 643 | Table 1 B2M R-50 $0.433$ / $0.157$ / $0.497$ | B2M |
| 653 | note c: "$0.579$ / $0.213$ and $\mathbf{0.752}$ box AP$_{50}$" at maxDets 1600; "the same grid as the Detectree2 row" | SB |
| 655 | note d: "$28.8\%$ ... ($2.7\%$) ... $1.2\%$ ... $+0.007$ AP$_{50}$" (cap analysis; +0.007 depends on stitched output; see A3) | B2M |
| 663–664, 674 | fig:qual `figs/qualitative_tcd439.pdf` (Detectree2 + Box2Mask Swin-L panels; tile 125 seam cuts); "Figure~\ref{fig:qual} shows four models" | DT2, B2M — regenerate |
| 674 | "it reaches $0.563$ mask AP$_{50}$ against LACE's $0.647$" | B2M |
| 718 | tab:sparse236 DT2 $0.6252$ / $\mathbf{0.3460}$ / $0.6331$ (bolding of AP50:95 and LACE AP50/box depends on it) | DT2 (sparse) |
| 725 | "Detectree2 $0.625$ against $0.614$" | DT2 (sparse + 439) |
| 869 | tab:scorer "Detectree2 (FT) & $0.625$ & $\mathbf{0.346}$" | DT2 (sparse) |
| 945 | "it nevertheless exhibits best-in-class scores for Box AP$_{50}$" | SB |
| 1074 | tab:strata-maskiou caption "$11{,}076$ crowns that every row matched" (common subset changes when any row changes) | DT2, SB, B2M |
| 1085 | tab:strata-maskiou DT2 row (7 cells; bold 0.825, 0.881) | DT2 |
| 1086 | SB row (7 cells; bold box IoU 0.807) | SB |
| 1087 | B2M Swin-L row (7 cells) | B2M |
| 1088 | B2M R-50 row (7 cells) | B2M |
| 1084, 1089 | Restor and LACE rows of the same table (common subset recomputed → their cells move too; bolding of 0.611/0.614, 0.695, 0.765, 0.825) | indirect |
| 1092 | "range from $371$ to $4{,}659$"; "LACE leads the lowest stratum" | indirect (all rows) |
| 1170 | "$4.7\%$ for Detectree2 and $6.6\%$ for Box2Mask" (empty-tile exposure); "no row is an outlier" | DT2, B2M (SB 5.60% unlisted) |
| 1187 | tab:geometry DT2 row: "stitched at the protocol floor $0.05$ rather than its $0.1$; \texttt{clean\_crowns} merge (IoU~$>0.7$ or containment~$>0.85$)" (stitch rule text changes with fix) | DT2 |
| 1189 | tab:geometry SB row: "IoU-NMS $0.7$ merge; their edge-band cull not applied" | SB |
| 1191 | tab:geometry B2M row: "same crops, grid and stitcher" | B2M |
| 1211 | "SelvaBox Edge trim" paragraph ($0.05$; $4{,}108$; $16.0\%$; $0.84$) — must be reworded if the cull is re-enabled at interior seams only | SB |
| 1215 | detection budget: "$1{,}511$" (already stale, A1), "$1{,}600$", "$0.004$ ($0.6143 \rightarrow 0.6187$)", "$0.008$ ($0.5710 \rightarrow 0.5789$)", "does not change any ranking order", "$584$" (already stale, A2) | DT2, SB, B2M |
| 1217 | "Detectree2 and Box2Mask are trained by us but we did not run multiseed" | text only (no number) |
| 1071 (comment) | commented paragraph: "both mask-supervised models exceed LACE on it while trailing it on AP$_{50}$" | DT2 (if uncommented) |

Derived deltas: no rendered "+0.033 over Detectree2" or other LACE-minus-baseline delta exists in the current text; the only baseline-derived differences are L725 (0.625 vs 0.614), L1215 (0.004, 0.008) and L655 (+0.007). The cost-Pareto figure (`figs/cost_pareto*`, uses Table 1 AP) is not \includegraphics'd in paper.tex but will also need regeneration if used in the CVPR build.
