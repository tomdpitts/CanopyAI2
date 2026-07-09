# DINOv3 on OAM-TCD (backbone-benchmark thread)

Isolated from the shadow-prior work (`shadow_prior/`, DeepForest detector). This
thread asks: **how well does a frozen DINOv3 backbone + light head do on OAM-TCD
tree-crown segmentation**, and does the satellite (SAT-493M) or web (LVD-1689M)
pretraining transfer better at 0.1 m? It reuses the local TCD data only.

## Status

| Piece | File | State |
|-------|------|-------|
| TCD label decoder (RLE-aware) | `tcd_data.py` | ✅ validated vs provenance (180/180 crown counts exact, canopy_frac Δ<5e-5) |
| Metrics: area F1/IoU + COCO mAP50 | `eval.py` | ✅ GT builds: 439 imgs, 25705 crowns + 4958 canopy-ignore |
| Frozen DINOv3 + linear head | `dinov3_seg.py` | ✅ validated (ViT-S/L forward; feature grid + ImageNet norm checked) |
| Semantic run (train head → eval F1/IoU) | `run_semantic.py` | ✅ runs end-to-end on MPS (~149 ms/window, 2.4 s/tile @ ViT-L) |
| Instance run (crown mAP50) | `run_instance.py` | ⬜ next rung (reuses `eval.coco_map50`) |

### Data availability
- `test` (439) and `sparse` (180) resolve (sparse symlinks → iCloud `.../tcd/val/`).
- **`dryland` (144) is unavailable**: all symlinks dangle (→ `../by_id/`, absent).
  Restore `data/tcd/experimental/by_id/` from the iCloud `CanopyAI/data` tree to
  evaluate the true dryland subset; until then `sparse` is the dryland proxy.

## ⚠️ Blocker: DINOv3 weights are gated

All `facebook/dinov3-*` checkpoints are `gated=manual` on HuggingFace, and this
environment has no HF token, so weight download fails with `GatedRepoError 401`.
Everything except the two `⏳` files above is therefore done and validated.

To unblock (one-time, ~2 min):
1. Accept the license at the model page(s) you want, e.g.
   <https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m> and
   <https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m> (click
   "Agree and access"; manual approval is usually instant).
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. Make it visible to the run, either:
   - `export HF_TOKEN=hf_xxx` before running, or
   - `.venv/bin/huggingface-cli login` (after `pip install huggingface_hub[cli]`).

## Run (after unblocking)

```bash
# semantic area-F1 / IoU, satellite vs web bake-off on the full 439 test split
.venv/bin/python dino/run_semantic.py --model facebook/dinov3-vitl16-pretrain-sat493m --eval-split test
.venv/bin/python dino/run_semantic.py --model facebook/dinov3-vitl16-pretrain-lvd1689m --eval-split test
# dryland-domain read (144-tile experimental subset)
.venv/bin/python dino/run_semantic.py --model facebook/dinov3-vitl16-pretrain-sat493m --eval-split dryland
```

`transformers` is required at run time (`pip install transformers`); `pycocotools`
and `einops` are already installed.

## Defaults chosen (change freely)

- **Variant:** the SAT-vs-web bake-off at **ViT-L/16**. Prediction from the 0.1 m
  resolution argument: the *web* (LVD-1689M) variant may win despite "sat" matching
  the domain on paper — that comparison is the point. Drop to `vitb16`/`vits16`
  for faster first passes on MPS.
- **Metric staging:** semantic **area F1/IoU first** (cheap linear probe), then
  instance **crown mAP50** (needs an instance head — heavier, next rung).
- **Baselines to beat:** `restor/tcd-segformer-mit-b0` (semantic) and
  `restor/tcd-mask-rcnn-r50` (instance) are already in the HF cache.

## Reuse from the existing project

Labels/metrics here are self-contained, but the **azimuth finetune cohort** and
the shadow feature live in `shadow_prior/` and `data/finetune/` — the eventual
shadow-prior-on-DINOv3 experiment would compose `shadow_prior.compute_shadow_feature`
with this backbone via the same RGB-first channel-inflation trick.
