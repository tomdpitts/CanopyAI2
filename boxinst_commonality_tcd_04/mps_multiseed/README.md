# 5-seed MPS variance, fixed 900-tile cohort + fixed EM, detector seed varied, det_t8 recipe

| metric | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std |
|---|---|---|---|---|---|---|---|
| mask mAP50 | 0.5041 | 0.4583 | 0.5218 | 0.4720 | 0.5022 | 0.4917 | 0.0231 |
| mask mAP50-95 | 0.1594 | 0.1451 | 0.1683 | 0.1532 | 0.1639 | 0.1580 | 0.0082 |
| mask P@50 (instance) | 0.6537 | 0.6083 | 0.6848 | 0.6780 | 0.6605 | 0.6571 | 0.0269 |
| mask R@50 (instance) | 0.5217 | 0.4908 | 0.5074 | 0.4298 | 0.5124 | 0.4924 | 0.0329 |
| box mAP50 | 0.5562 | 0.5010 | 0.5640 | 0.5159 | 0.5541 | 0.5382 | 0.0250 |
| box mAP50-95 | 0.2271 | 0.2002 | 0.2257 | 0.2095 | 0.2296 | 0.2184 | 0.0115 |
| box P@50 (instance) | 0.6877 | 0.6354 | 0.7161 | 0.7122 | 0.6945 | 0.6892 | 0.0289 |
| box R@50 (instance) | 0.5509 | 0.5149 | 0.5327 | 0.4529 | 0.5411 | 0.5185 | 0.0349 |
| semantic F1 (pixel) | 0.6453 | 0.6208 | 0.6550 | 0.5832 | 0.6494 | 0.6307 | 0.0265 |
| semantic P (pixel) | 0.7853 | 0.7536 | 0.7871 | 0.7829 | 0.7619 | 0.7742 | 0.0137 |
| semantic R (pixel) | 0.5477 | 0.5278 | 0.5609 | 0.4647 | 0.5659 | 0.5334 | 0.0368 |

**Headline: mask mAP50 = 0.4917 ± 0.0231** (n=5 seeds).

Per-seed operating threshold (val-picked, used for the instance P/R rows): s0=0.40, s1=0.40, s2=0.35, s3=0.35, s4=0.40.

Vaulted single-run reference (seed 0, multiscale): **mask mAP50 = 0.5041** (0.504).
Seed-0 rerun here: 0.5041 (+0.0000 vs vaulted) — reproduces within MPS noise (±0.01).

## Bottom line
The headline **0.504 is reproducible and representative, not cherry-picked**: seed 0 reproduces it to 4 dp, and it sits +0.54σ from the 5-seed mean (well inside 1σ). The honest expected value for this pipeline is **mask mAP50 ≈ 0.492 ± 0.023** (1σ, detector-seed + MPS noise); the published 0.504 is a favourable-but-typical draw from that distribution. All five seeds beat the fully-supervised Restor Mask R-CNN baseline (0.432) — the weakest, seed 1 at 0.458, still clears it by +0.026. No seed collapsed; nothing needed fixing.

> ⚠️ **Outlier seeds (mask mAP50 > 0.02 from mean):** seed 1 (0.4583), seed 2 (0.5218). Flagged rather than silently averaged.

## Design
- **What varies:** only the detector-training seed (numpy+torch). The 900-tile cohort (train/val partitions), the DINOv3-web feature caches, and the box→mask EM masker (`vault/em_model.npz`) are all held fixed.
- **What this isolates:** detector-training + MPS non-determinism only (MPS is not bitwise-reproducible). It does **not** estimate cohort variance — the data split is identical across seeds.
- **Recipe:** det_t8 (width 256, tower 3, Adam lr 1e-3 wd 1e-4, cosine, bs 3, eval_every 5, best-on-val checkpoint) **+ aggressive early stopping** (min_epochs 12, es_patience 2 → stop after ~10 flat epochs). ES trims dead tail epochs only; the checkpoint stays best-on-val. It deviates slightly from the exact full-40 recipe behind 0.504, so the seed-0 repro is a loose (not byte-faithful) check.
- **Eval:** full OAM-TCD 439 test, multiscale (native + 0.5× downscale arm), same canopy-ignore matching as the headline.
- **P/R rows:** *instance* P@50 / R@50 are single-operating-point detection precision/recall at IoU 0.5, greedy-matched with the same canopy-ignore rule as AP, at each seed's val-picked score threshold (`op_thr`, shown per seed below). *Semantic* P/R are pixel-level foreground agreement (canopy-excluded), not instance-level. Recall denominator = 25 705 GT trees. Computed by `eval_pr.py` (reuses the evaluator's exact prediction pipeline; predictions weren't cached, so P/R required re-running detector+EM).
- **Isolation:** every checkpoint / eval json / result lives under `mps_multiseed/` (`_out/artifacts/` via monkeypatched `T.ART`/`E.OUT`; `_out/test_gt.json` symlinks the read-only test GT). `rm -rf mps_multiseed/` undoes the whole experiment; `../artifacts/` and `../vault/` are untouched.
