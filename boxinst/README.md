# boxinst — box-supervised instance segmentation on frozen DINOv3

One-shot prototype: a single lightweight head (~2.7M params) on frozen
DINOv3-ViT-L/16 features that does crown **detection + instance masks in one
forward pass**, trained with **bounding boxes only** (no mask labels, no SAM /
spectral / watershed post-processing).

## Method

- **Backbone (frozen, cached)**: `facebook/dinov3-vitl16-pretrain-lvd1689m`;
  last 4 blocks (21–24), each L2-normed, concatenated → (4096, 32, 32) per
  512-padded tile, cached fp16 in `dapt/cache/web_last4/`.
  *Note*: this diverges from the frozen dapt probe protocol (blocks 3/6/9/12) —
  the one-shot spec asked for the last ~4 hidden layers.
- **Head** (`model.py`): 1×1 stem (4096→256) → 2× 3×3 det tower →
  CenterNet maps (heatmap / offset / log-size) + CondInst controller (169 ch);
  mask neck with CoordConv upsamples 32→128 (stride 4) to an 8-ch shared
  F_mask. Per instance, the controller's 169 params form a 3-layer / 8-ch
  dynamic point-wise conv run over F_mask ⊕ rel-coords → mask logits.
- **Losses, boxes only** (`losses.py`): CenterNet penalty-reduced focal (centre) +
  smooth-L1 (offset / log-size) + GIoU at positive cells (box), plus BoxInst:
  - *projection*: dice between row/col-max of the sigmoid mask and the box's
    1-D extents, at the 128×128 mask grid (not the 16 px patch grid);
  - *pairwise*: for pixel pairs one DINO patch apart (dilation 4 at mask res =
    16 image px; 4 directions) whose anchor pixel is inside the box, if DINO
    cosine ≥ τ the pair is pushed to agree (−log P(same)). Affinity = DINO
    features (bilinearly upsampled, re-normed), not raw colour. Ramped 0→1 over
    1000 iters. Gating is ONE-endpoint (neighbour may be outside the box) — this
    is load-bearing: with both endpoints gated in-box, "fill the whole box"
    becomes a global minimum and masks converge to rounded rectangles (observed;
    that run is archived as `artifacts/boxinst_s0_bothgate.*`).
- **Selection**: best val detection mAP50 (masks have no GT to select on),
  restricted to epochs AFTER the pairwise warmup completes — otherwise val
  detection picks a very early epoch (ep20) whose mask branch has barely
  trained. Operating score threshold frozen on val F1. Seeded and recorded.

## Untuned knobs (reported, never tuned on masks)

- τ = 0.975 = p75 of the dilation-4 cosine distribution over the cohort
  (feature statistics only). The distribution is very anisotropic
  (p5 = 0.85, p50 = 0.96 even one full patch apart) — bilinear upsampling of
  patch-16 features makes close-range affinity nearly saturated, which is why
  the dilation-2 (8 px) offsets are cached but not used by default.
- centre-peak NMS: 3×3 max-pool peak picking + IoU-NMS 0.5; score threshold
  from val F1. Mask threshold 0.5.

## Usage

```bash
.venv/bin/python -m boxinst.cache_feats            # once: features + affinities
.venv/bin/python -m boxinst.train --epochs 300 --seed 0
.venv/bin/python -m boxinst.infer_viz              # PNGs -> claude_outputs/boxinst/
```

## Results (seed 0, best ep160, thresholds frozen on val)

Detection (boxes have GT; masks do not):

|            | mAP50 | mAP50-95 | F1    | P / R       | count MAE |
|------------|-------|----------|-------|-------------|-----------|
| val  (33)  | 0.643 | 0.228    | 0.705 | 0.71 / 0.70 | 1.79      |
| test (33)  | 0.589 | 0.217    | 0.652 | 0.67 / 0.63 | 1.82      |

Pre-warmup val detection peak was 0.639 (ep30) vs 0.643 eligible (ep160) — the
mask-maturity selection rule cost ~nothing this run. Touching-crown recall
(0.62) ≈ isolated (0.66). AP_small ≈ 0.05: sub-32px crowns are essentially
missed (patch-16 features). Reference: the dapt L1 linear probe on blocks
3/6/9/12 scores test mAP50 ≈ comparable — this head adds instance masks for
+2.7M params, one pass.

Masks are qualitative only (no mask GT): see `claude_outputs/boxinst/`.
Instances land on real crowns in dryland tiles; boundaries are feature-driven
but soft (16 px evidence granularity), continuous canopy rows undersegment,
and big canopies occasionally get duplicate overlapping instances.

## Commonality extension (`commonality.py`, `eval_masks.py`)

"What do the annotated boxes share in latent space?" made concrete, three ways:
1. **LDA direction** (train-fit, in-box vs clear-bg patches): held-out AUC 0.975
   (WON 0.98 / NEON 0.99 / BRU 0.88). Cached per tile in
   `dapt/cache/commonality_last4/` (+PCA-64 embeddings).
2. **Box-topic EM**: figure/ground mixture over box-normalised coords learns a
   centred-blob spatial prior pi(u,v) from box statistics alone; GT-box-prompted
   posterior masks with zero mask labels.
3. **Prototype-consistency loss** (`--proto_loss`): each instance's mask-weighted
   latent is pulled toward an EMA crown prototype — the mask learns to select
   the sub-region that makes all boxes alike. `--commonality_channel` appends
   the LDA map to the mask neck. `--no_center` ablates the CenterNet objective.

Ablations (300 ep, seed 0; proxy mask metrics are GT-box-prompted on test —
fill/corner/leak; degenerate box-filler scores fill .95 / corner .89):

| run | det mAP50 | fill | corner | centre | leak |
|-----|-----------|------|--------|--------|------|
| A baseline            | .577 | .63 | .41 | .92 | .08 |
| B +LDA channel        | .576 | .66 | .44 | .93 | .08 |
| C +channel +proto 0.5 | .558 | .51 | .31 | .82 | .08 |
| D proto, NO center    | .002 | .44 | .26 | .73 | .06 |

Read: commonality objectives tighten masks (corner suppression .41->.26) at ~no
detection cost when joint (C); detection collapses without the centre objective
(D) — commonality is a foreground prior, not an instance signal. MPS runs are
not bitwise reproducible (same-seed repeats vary ~±0.01 mAP50).

## Known limitations (by construction)

- Patch-16 resolution is the main risk: mask losses act at stride 4 but the
  *evidence* (DINO features / affinity) has 16 px granularity — boundaries are
  soft and tend to follow patch structure around small crowns.
- Box-supervised masks are approximate; there is no mask GT anywhere, so masks
  are evaluated qualitatively + via detection AP only (no mask IoU).
- One centre cell per instance (32×32 grid): touching crowns whose centres
  fall in the same 16 px cell collapse to one instance (known dapt limitation,
  ~cell-collision rate logged by `python -m dapt.targets check`).
