# Feature-compression ablation (tcd_04 detector) — stopped after stage 1

Question: Detector8 trains on 4096-dim DINOv3-web features (134 MB/tile), so a
full-4,169-tile local cache would be ~484 GB. Its first layer is a 1×1 conv
4096→256 — does detection actually need the width, or would a compressed cache
(unlocking full-4k training locally) be near-lossless?

Self-contained and deletable: `rm -rf feat_ablation/` undoes everything. The
parent pipeline (det_t8.pt, train_detector_tiles.py, evaluate.py, caches) was
never modified — wrappers here monkeypatch paths only. Current footprint ~10 GB
(the PCA-256 train+test cache; block-1024 needs no cache, it's an mmap slice of
the parent .npy — the last 1024 channels are contiguous on disk).

## What ran (2026-07-16, seed 0 throughout)

1. **Baseline validation** — `eval_box.py` (box-only, single-scale, identical
   decode + canopy-ignore + matching to the parent `evaluate.py`) on the
   existing det_t8 / full-4096: **box mAP50 0.555, mAP50-95 0.2271** on the 439
   test — exactly reproduces the parent number, so the eval path is validated.
   → `results/eval_box_full4096_t8.json`
2. **PCA-256 basis** — fit on 200k patch vectors sampled across all 900 cached
   train tiles (streaming f64 covariance → eigh). Basis in `pca256.npz`;
   projected fp16 caches in `cache/pca256/`. → `logs/build_pca.log`
3. **Stage-1 gate (linear probe, no training)** — in-box (positive) vs clear
   background (negative) cells at the 128×128 feature grid, canopy dropped
   (pipeline >50% rule); ≤100+100 cells/tile balanced; logistic regression
   (torch LBFGS, standardized) fit on train-partition tiles, AUC on the
   held-out val partition; all variants scored on identical cells.
   → `results/linear_probe.json`, `linear_probe.py`

## Stage-1 results

| Variant | dim | probe val AUC | Δ vs full | var. retained | MB/tile | full 4,169-tile cache |
|---|---|---|---|---|---|---|
| full-4096 | 4096 | 0.9480 | — | 100% | 134 | ~484 GB |
| block-1024 (last block, L24) | 1024 | **0.9491** | +0.001 | n/a (slice) | 33.6 | ~121 GB |
| PCA-256 | 256 | 0.9426 | −0.005 | **84.0%** | 8.4 | ~33 GB |

Secondary: full-4096 overfits the probe (train 0.967 → val 0.948) while
PCA-256 does not (0.942 → 0.943); block-1024's val AUC is indistinguishable
from full (noise).

## Verdict

Gate rule was: proceed to stage 2 iff PCA-256 AUC within ~0.01 of full AND
top-256 variance ≥ ~95%. **AUC passed for both variants (block-1024 exactly
matches full; PCA-256 −0.005), but the 95% variance gate failed (84.0%)** —
the DINO spectrum is heavy-tailed, so the two criteria disagree for PCA-256.
Decision (user, 2026-07-16): **stop here — no detector training.** Nothing in
the probe says detection needs the rich features, but linear presence-vs-
background separability is not localization; the mAP question is unanswered.

## Resuming stage 2 later

Sequential (one MPS job at a time); MPS is not bitwise-reproducible, treat
mAP50 gaps < ~0.02 as noise vs the 0.555 baseline:

```bash
.venv/bin/python -u -m boxinst_commonality_tcd_04.feat_ablation.train_variant \
    --variant pca256 --epochs 40 --bs 3 --eval_every 5 --seed 0
.venv/bin/python -u -m boxinst_commonality_tcd_04.feat_ablation.eval_box --variant pca256
# then the same with --variant block1024
```

Checkpoints land in `artifacts/det_fa_<variant>.pt`; eval jsons in `results/`.
If `cache/pca256/` was deleted, `build_pca_cache.py` rebuilds it in ~5 min
(the saved `pca256.npz` basis is reused; delete it too to refit).
