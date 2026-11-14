# 4-shift LACE at the learned confidence ŝ (test 439 + sparse 236)

**Question.** The IL-test (11-04) showed 4 shifts tie 16 at AP50 on test, at the old s′ ranking. What does 4-shift LACE score at the current ranking ŝ (R3), on both splits?

## Pre-registration (2026-11-14, before any number)

- **4-shift masks and s′:** IL-test arm `interp8` exactly (`dino_det/interlace_test._job`: Modal `feat_4p_<split>` = 4 offsets {0,8}², projection bilinearly interpolated to the 4 px lattice, lace4px_simple, EDGE_FILL on). Test: the stored IL-test masks (`feat_cache/interlace_test_interp8/`). Sparse: rendered here by the same function → `feat_cache/shift4_sparse_interp8/`.
- **ŝ:** the frozen R3 recipe refit exactly as `cap_test.py` (rerank3.final, 25 models, val-trained on 16-shift posteriors), gated to reproduce the published R3 scores (max |Δ| < 1e-6), then applied unchanged to the 4-shift s′. The reranker is NOT retrained on 4-shift val features, which if anything disadvantages 4 shifts.
- **Rows:** 16-shift ŝ (published R3_ef; gate to 4 dp), 4-shift ŝ; and at s′ ranking 16-shift vs 4-shift (reproduces IL-test on test; new on sparse).
- **Scoring:** `sam3_on_d1.per_tile` / `ap_at` (crowd arm, maxDets 600, floor 0), AP50 / AP75 / AP50:95; paired tile bootstrap (2,000, seed 0) of 4-shift − 16-shift at each ranking.
- One look per split. Reported whatever the outcome. Single detector seed.

```
cd boxinst_commonality_tcd_04/lace_4shift && PYTHONPATH=/Users/tompitts/dphil/CanopyAI2 ../../.venv/bin/python -u run.py > logs/run.log 2>&1
```

## RESULTS (2026-11-14; `results/shift4.json`, `logs/run.log`)

Gates PASS: R3 refit reproduces the published scores (max |Δ| 9.5e-9, both splits); the 16-shift ŝ row reproduces the published R3_ef row; the test s′ rows reproduce the IL-test (0.7044 / 0.7043); sparse feature gate (cos ≥ 0.999) on all 236 tiles. Mask AP50 / AP75 / AP50:95:

| ranking | shifts | test 439 | sparse 236 |
|---|---|---|---|
| ŝ (R3) | 16 (deployed) | 0.7141 / 0.2713 / 0.3424 | 0.7434 / 0.4006 / 0.4031 |
| ŝ (R3) | **4** | **0.7138 / 0.2684 / 0.3411** | **0.7436 / 0.3973 / 0.4022** |
| s′ | 16 | 0.7044 / 0.2644 / 0.3366 | 0.7350 / 0.3944 / 0.3980 |
| s′ | 4 | 0.7043 / 0.2619 / 0.3356 | 0.7368 / 0.3927 / 0.3982 |

Paired tile bootstrap, 4 shifts − 16 shifts (diff [95 % CI]):

| ranking | split | ΔAP50 | ΔAP75 | ΔAP50:95 |
|---|---|---|---|---|
| ŝ | test | −0.0003 [−0.0021, 0.0004] | −0.0029 [−0.0053, −0.0016] | −0.0013 [−0.0018, −0.0007] |
| ŝ | sparse | +0.0002 [−0.0005, 0.0010] | −0.0033 [−0.0061, −0.0006] | −0.0009 [−0.0017, −0.0001] |
| s′ | test | −0.0001 [−0.0018, 0.0007] | −0.0025 [−0.0050, −0.0009] | −0.0010 [−0.0015, −0.0002] |
| s′ | sparse | +0.0018 [0.0000, 0.0043] | −0.0017 [−0.0049, 0.0012] | +0.0002 [−0.0011, 0.0016] |

**Reading.** At the current ranking, 4 shifts tie 16 at AP50 on both splits (CIs span 0) and lose 0.003 AP75 / 0.001 AP50:95 (CIs below 0, small). Sixteen shifts buy about 0.001 AP50:95 for 2.6× the masker compute (66 s vs 25 s per tile, `masker_cost/`). Caveats: the reranker was not retrained on 4-shift features; masker knobs were tuned on 16-shift features; single detector seed.
