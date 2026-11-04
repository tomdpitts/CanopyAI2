# masker_refit4p: refit the LACE masker on the 4 px lattice (2026-11-04)

## Pre-registration R4P (2026-11-04, BEFORE any refit number)

Tom: "let's explore doing this again with the 4px lattice. It's a cleaner story, and might help with performance?" The arms are 120 and 250 tiles, run locally on the M4 Max.

**Motivation.** The deployed masker (`em_model_4p_fix.npz`, sha1 `2f0eaa3f…`, β 0.5, K 16) was fitted on the **8 px** lattice: 4 passes, `feat_4p_train`, the first 120 sorted train tiles, on a Modal CPU. It is applied unchanged on the **4 px** lattice (16 passes, `hunt_z4p_*`). That mismatch is the question a reader asks first (paper Method draft `ablation/results/method_v2.tex`). It was never refitted because the parameters were argued to be offset-invariant (`native_raster/hunt/README.md` L28), but that argument was never tested.

**Naming caution.** `feat_4p_*` = 4 passes = 8 px lattice. `z4p_*` / `hunt_z4p_*` = 4 px lattice (16 passes). The cache names are not changed (Tom, 2026-11-04).

**Fit (`fit4p.py`).** This is `em.fit` verbatim except for:
- the PCA block, which is removed: cells are the deployed whitening's projection, already L2-normalised, not re-normalised, exactly as inference loads them;
- the 4 px geometry: cell centre 4X + 8;
- the ring / clear-background margin, which stays 8 px (2 cells);
- a checkpoint after init and after each EM iteration;
- mu/U/scale, which are copied from the deployed npz.

All hyperparameters are unchanged: K 16, K_bg 12, 8 bins, κ 10, 30 iterations, prune after 5 at 0.25/K, β 0.5, seed 0.
- **Input:** `tcd04-phase4-vol:z4p_train`, pulled locally to `~/dphil/feat_cache/z4p_train/`. This is the first 250 sorted train ids, by the same rule as the deployed fit's first 120. It is (512, 512, 128) fp16, gated against local `hunt_z4p_val` at cosine 1.000 (dethead/README.md G2).
- **Arms:**
  - **n120:** the same 120 tiles as the deployed fit. Only the lattice changes.
  - **n250:** the first 250 tiles. This changes the lattice and adds data.
- Boxes only. Canopy is ignored. No polygon is read. Validation tiles are never used in fitting.

**Evaluation (`eval4p.py`).**
- **Boxes:** the frozen D1 boxes (`dino_det/results/modal/dino_swinl_s0_preds_*.json`).
- **Render:** the lace4px_simple path (α 0.3, κ ×1.6, τ 0.33 × box peak, guided r 12, ε 1e-4), unchanged. Only the masker npz is swapped.
- **Ranking:** s′. R3 is not involved, because its inputs depend on the masker.
- **Parity gate, first:** the deployed masker must reproduce D1's val s′ mask AP50 through this script (±5e-5). If it fails, nothing else is scored. *Amended 18:47, before any refit number: the reference is 0.7414 (D1 + the shared top/left edge fix, `GA.EDGE_FILL = True`, adopted 2026-11-04, dino_det/SAM3_D1.md), not the pre-fix 0.7384. The test/sparse comparison is against `dino_det/results/edgefix/preds_d1_edgefix_*`.*
- **Val metric:** mask AP50 on the 108 val tiles. score_lite crowd arm, maxDets 600, floor 0.
- **Rule:** an arm passes if its val AP50 ≥ deployed + 0.003. Among the passing arms, choose the smallest within 0.001 of the best. If none passes, the deployed masker stays, and test and sparse are not run.
- **Final, ONCE, only if an arm passes:** the selected masker on test 439 and sparse 236.
  - Scoring: frozen score_coco (2048, maxDets 600, crowd arm, floor 0).
  - Comparison: a paired tile bootstrap against the deployed masker on the same D1 boxes (2,000 resamples, seed 0).
  - Adoption into the paper is Tom's decision. If adopted, R3 must be retrained on val, since it takes log(p̄·m) as input.
- Reported whatever the outcome.

**Cost.** Local, $0. Pulling 250 tiles from the Modal volume is about 16 GB of egress.

## Log
- 2026-11-04 18:23: pull of the first 250 z4p_train tiles started (`feat_cache/z4p_train/_pull.sh`, log `_pull.log`).
- 2026-11-04 18:27: parity run started (`eval4p val --tag deployed`, log `native_raster/logs/refit4p_val_deployed_2026-11-04.log`).
- 2026-11-04 18:40: parity FAILED against 0.7384 (got 0.7414 / 0.3392 / 0.3788; record kept as `results/val_deployed_vs7384_2026-11-04.json`). Cause: another session moved the top/left edge fix into the shared `guided_ablate_modal._prob` (`EDGE_FILL = True`, 15:20). s′ is identical on all 51,341 val detections; 3,625 masks differ, and the re-rendered masks are identical to the edge-fixed D1 masks (next line), so the difference is exactly the edge fix. 0.7414 equals SAM3_D1.md's "+ top/left fix" val row exactly.
- 2026-11-04 18:47: parity reference amended (above). Re-run: 0.7414, PASS. All 51,341 masks are identical to `feat_cache/edgefix_val`.

## RESULTS, R4P (2026-11-04 19:02; `results/{val_deployed,val_n120,val_n250,select}.json`, fits `results/fit_n{120,250}/`, log `native_raster/logs/refit4p_run_2026-11-04.log`): NO ARM PASSES, deployed masker kept

| masker (D1 val boxes, s′) | K after pruning | boxes | val mask AP50 | AP75 | AP50:95 | box AP50 |
|---|---|---|---|---|---|---|
| deployed (8 px fit, 120 tiles) | 16 | — | **0.7414** | 0.3392 | 0.3788 | 0.7575 |
| n120 (4 px refit, same 120 tiles) | 15 | 9,153 | 0.7404 (−0.0010) | 0.3341 | 0.3772 | 0.7574 |
| n250 (4 px refit, first 250 tiles) | 8 | 17,385 | 0.7417 (+0.0003) | 0.3390 | 0.3789 | 0.7606 |

- **Gate:** +0.003 is needed; neither arm reaches it. Test and sparse were NOT run (pre-registered).
- **Reading:**
  - Refitting on the 4 px lattice changes val AP50 by −0.001 to +0.0003. That is well inside the noise the gate was set to exclude.
  - This supports the L28 argument (the masker's parameters are offset-invariant) with a measurement instead of an argument.
  - More data (250 tiles) neither helps nor hurts. The 250-tile fit pruned to 8 prototypes and still matched.
- **For the paper:** the 8 px fit applied at 4 px can be justified in one sentence: "refitting on the 4 px lattice (120 or 250 training tiles) changes validation mask AP50 by at most 0.001".
- **Engineering notes:**
  - The first n250 attempt died after load (out of memory: em.fit materialises every clear-background cell). fit4p (f) now gathers only the subsampled rows; it was verified bit-identical on n120 before n250 was rerun.
  - The local 4 px train cells are kept: `~/dphil/feat_cache/z4p_train/` (250 tiles, 16 GB).
