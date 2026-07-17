# vault — frozen backup of the 0.504 mask-mAP50 model

Read-only backup of the exact artifacts behind the headline result, kept here so a
stray training/eval run in `../artifacts/` can never overwrite them. **Do not train
or write into this folder.** Integrity: see `SHA256SUMS.txt` (`shasum -a 256 -c
SHA256SUMS.txt` to verify).

## The result

Full OAM-TCD 439-tile test, web backbone, seed 0, **multiscale** pipeline:

| metric | value |
|--------|-------|
| **mask mAP50 (ITC, headline)** | **0.504** |
| mask mAP50-95 | 0.159 |
| box mAP50 | 0.556 |
| semantic F1 (canopy-excluded) | 0.645 |

Single-scale (no downscale arm) is mask mAP50 0.499 / box 0.555 / F1 0.587.
`43.2` = the fully-supervised Restor Mask R-CNN baseline this beats, box-only + frozen backbone.

## What's in here

| file | what |
|------|------|
| `det_t8.pt` | the 8px CenterNet detector (`Detector8`); `{state, cfg}`. cfg: seed 0, in_dim 4096, width 256, tower 3, best_epoch 20, score_thr 0.40. |
| `em_model.npz` | the training-free box-to-mask commonality EM (`TCDMasker`). |
| `SHA256SUMS.txt` | checksums of both. |
| `eval_multiscale_439.json` | authoritative full-439 multiscale metrics (written by the verify run). |

Not vaulted (regenerable, too large): the DINOv3-web feature caches
`../cache/web/{feat_test, feat_test_down}` (~55 GB + ~14 GB). Rebuild them with the
commands below if missing — the models depend on them at eval time.

## How to use / replicate 0.504

Run from the repo root (`/Users/tompitts/dphil/CanopyAI2`). The model files here are
byte-identical to `../artifacts/det_t8.pt` and `../artifacts/em_model.npz`; the
evaluator loads the detector by tag from `../artifacts/`, so either restore the
vaulted copies there or point `--model` at the vaulted EM.

```bash
V=.venv/bin/python

# 0. (if missing) rebuild the frozen-feature caches — one-time, backbone-bound
$V -m boxinst_commonality_tcd_04.prepare_test        # 439 GT -> test_gt.json
$V -m boxinst_commonality_tcd_04.cache_test          # native 128x128 feats  -> feat_test/
$V -m boxinst_commonality_tcd_04.cache_test_down     # 0.5x feats (big-tree arm) -> feat_test_down/

# 1. reproduce the headline (detector det_t8 + vaulted EM + downscale arm)
$V -m boxinst_commonality_tcd_04.evaluate \
     --det det_t8 \
     --model boxinst_commonality_tcd_04/vault/em_model.npz \
     --multiscale
# -> mask mAP50 0.504 ; writes ../artifacts/eval_t8.json
```

To restore the model into the working folder after an accidental overwrite:

```bash
cp -p boxinst_commonality_tcd_04/vault/det_t8.pt   boxinst_commonality_tcd_04/artifacts/
cp -p boxinst_commonality_tcd_04/vault/em_model.npz boxinst_commonality_tcd_04/artifacts/
shasum -a 256 -c boxinst_commonality_tcd_04/vault/SHA256SUMS.txt   # confirm intact
```

The pipeline is: frozen DINOv3-web features → `det_t8` boxes (native + 0.5× downscale,
cross-scale NMS) → per-box `em_model` mask. Single-scale (drop `--multiscale`) = 0.499.

## How these models were trained

Both were produced from **bounding boxes only** — no mask ever touches training.
Backbone (DINOv3-web ViT-L/16) is frozen throughout; only the detector's ~4 M-param
head is trained. Reproduce from scratch (needs the train caches too):

```bash
V=.venv/bin/python
# a. sample + cache 900 full 2048 train tiles as stitched 128x128 web features
$V -m boxinst_commonality_tcd_04.cache_train_tiles --n 900        # -> train_tiles_gt.json + feat_traintile/
# b. fit the training-free EM masker on train-tile box interiors (canopy ignored)
$V -m boxinst_commonality_tcd_04.em --seed 0                      # -> em_model.npz
# c. train the 8px detector on whole tiles (boxes only, canopy-ignore)
$V -m boxinst_commonality_tcd_04.train_detector_tiles --tag t8 --seed 0 \
     --epochs 40 --bs 3 --eval_every 5                            # -> det_t8.pt
```

**Detector (`det_t8.pt`).** `Detector8` — 1×1 stem (4096→256), 3× 3×3 conv tower,
bilinear ×2 to an 8px grid, CenterNet heads (heatmap + offset + log-size). Trained
on the 900-tile cohort (792 train / 108 val), all ITC boxes per tile as targets,
canopy cells excluded from the negative focal loss. Adam, lr 1e-3, weight decay
1e-4, batch 3, cosine schedule over 40 epochs; **model selected by best validation
box AP50 (early stopping)** — the reported checkpoint is epoch 20 (val boxAP50
0.471), before the run overfits. Losses: penalty-reduced focal (heatmap) +
masked smooth-L1 (offset, log-size×0.1) + GIoU at positive cells. Decode:
3×3 max-pool peak-pick + IoU-NMS 0.5; operating score threshold 0.40 (val-picked).
Note MPS training is not bitwise-reproducible (~±0.01), so a re-run lands near, not
exactly on, 0.471/0.504.

**EM masker (`em_model.npz`).** Training-free — *fit*, not gradient-trained. On the
train-tile box interiors: centre + PCA-128 + whiten the DINOv3 features, fit a frozen
`K_bg=12` background mixture (clear + near-box-ring patches), then EM for `K` crown
prototypes with size-conditioned spatial priors and contrastive prototype repel
(β=0.5); K auto-prunes to ~7. No polygon is read at any point. Deterministic given
`--seed 0` and the fixed cohort.

**Downscale arm.** Not a trained artifact — `evaluate --multiscale` runs the same
`det_t8` on 0.5×-downsampled tiles (`cache_test_down`) and cross-scale NMS-merges,
recovering large crowns. +0.005 mask mAP50 / +0.058 semantic F1 over single-scale.
