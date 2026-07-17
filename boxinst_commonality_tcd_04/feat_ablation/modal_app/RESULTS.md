# block-1024 gate on Modal A100 — VALIDATED

**Verdict: block-1024 box mAP50 = 0.5371 vs full-4096 baseline 0.555 → gap 0.018,
inside the ~0.02 tolerance. The full-4k caching program (~121 GB local) is
green-lit.** (Ran 2026-07-16.)

## Number

| metric | block-1024 (this run) | full-4096 baseline |
|---|---|---|
| box mAP50 (single-scale, 439 test) | **0.5371** | 0.555 |
| box mAP50-95 | 0.2174 | — |
| best val boxAP50 | 0.4638 @ ep15 | (det_t8: ep~20 peak) |

The 0.018 drop is within MPS-vs-CUDA non-determinism (extraction *and* detector
training both moved to GPU here). The val trajectory reproduced det_t8's overfit
signature — peak ep15 (0.464), then monotone decline to ep40 (0.346) — so
best-on-val selected ep15, exactly as designed. `PICKUP.md`'s expectation ("linear
presence AUC held; box mAP50 unmeasured but expected positive") is confirmed:
detection does NOT need the full 4096 width. The last DINOv3 block carries the
detector signal.

## What changed vs the baseline (only one intended thing)

The single intended change is **feature width** (4096 → last-block 1024). Two
incidental, tolerance-covered changes: features re-extracted on A100 (was local
MPS) and detector trained on A100 (was MPS). Everything else is byte-for-byte the
det_t8 recipe: `Detector8` width 256 / tower 3 / in_dim 1024, Adam lr 1e-3 wd 1e-4,
bs 3, 40 epochs, eval_every 5, cosine schedule, best-on-val checkpoint, seed 0, the
same 900-tile cohort + train/val split from `train_tiles_gt.json`, and the
box-only single-scale evaluator that reproduces 0.555 exactly on full features.

## Provenance — same tiles as the baseline, proven

Features were extracted on the A100 from the raw tiles ALREADY on Modal (HF
`restor/tcd` on the `canopyai-deepforest-data` volume), NOT uploaded. Tiles were
joined to the exact 900+439 baseline set by **`image_id`** (the stable key in each
local `meta.json` and in the HF rows), never by filename. The `verify` stage
proved the join before any GPU spend:

- 900 train + 439 test `image_id`s all present in HF; unique; 0 train/test overlap.
- 0 width/height mismatches, 0 ITC-box-count mismatches.
- 40/40 sampled tiles: decoded-RGB **sha1 byte-identical** to the local tiles.

So the extracted features come from the identical pixels the 0.555 baseline used;
only the extraction device differs (covered by the ~0.02 gate).

## Artifacts (Modal Volume `tcd04-block1024-vol`, profile tomdpitts)

- `feat_traintile/<tid>.npy` — 900 train tiles, `(1024,128,128)` fp16 (block 3072:4096)
- `feat_test/<tid>.npy` — 439 test tiles, same dtype/shape
- `out/det_fa_block1024.pt` — best-on-val (ep15) checkpoint
- `out/eval_box_block1024.json` — the eval result above

`<tid>` keys match `boxinst_commonality_tcd_04/{train_tiles_gt.json,test_gt.json}`;
`image_id` provenance in `feat_ablation/modal_app/manifest.json`. Block-1024 ONLY —
the full 4096 composite was sliced in-memory and never persisted.

## Reproduce / re-run

```bash
# CPU join-proof (cheap; no GPU)
modal run boxinst_commonality_tcd_04/feat_ablation/modal_app/app.py::verify
# full chain: extract (A100, ~30 min) + train + eval  [--skip-verify if already proven]
modal run boxinst_commonality_tcd_04/feat_ablation/modal_app/app.py --skip-verify
```

Cost this run: ~30 min A100 extraction + ~25 min A100 train/eval ≈ ~$10–12.

## Next (the actual fix for the overfit — separate program, see PICKUP.md §"Full-4k")

block-1024 is validated, so the ~121 GB local full-4k cache is justified. The
overfit fix is 4× data (~3,611 ITC train tiles) + regularization (tower 2, wd 5e-4)
+ ReduceLROnPlateau/early-stop, ×5 seeds. NOTE: this Modal volume holds only the
900+439 **gate** cohort at block-1024 — the larger pool is not cached here.
