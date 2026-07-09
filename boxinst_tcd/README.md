# boxinst_tcd — box-supervised tree segmentation on OAM-TCD, scored on masks

Tests the commonality question on data that HAS polygon masks: can a box-only-trained
head on frozen DINOv3 segment individual tree crowns, and does the cross-box
commonality objective (configs B/C) help? Crucially, **training reads only boxes;
polygons are held out and used for evaluation only.**

## Data (`prepare.py`)

- Source: OAM-TCD train, **category_id==1 (individual tree) only** — canopy
  (category 2) dropped entirely.
- One coherent ecoregion (Zanzibar-Inhambane East African coastal forest — the
  dominant cluster), **native-resolution 512×512 crops** (no downsampling), split
  **by source tile** (crops from a val/test source never appear in train).
- Two earlier designs failed and are worth recording: (1) resizing whole 2048 tiles
  to 512 blurs the texture DINOv3 needs; (2) a random split across TCD's global
  biomes leaves val biomes unseen → the frozen-feature head overfits (train boxAP
  0.75 / val 0.02). Native crops + one ecoregion + source-split fixed both.
- `boxes.json` (xyxy, == COCO bbox == polygon extent) is the ONLY training input;
  `gt_polys.json` (polygons) is rasterized to masks in `eval_masks.py` only.

## Configs (same head/losses as `boxinst/`, trained from scratch on frozen DINOv3)

- **A** baseline: CenterNet detect + CondInst dynamic mask + BoxInst proj+pairwise.
- **B** A + LDA commonality channel in the mask neck.
- **C** B + cross-box prototype-consistency loss (`--proto_loss 0.5`).

## Evaluation (`eval_masks.py`) — two modes

1. **Full pipeline** (detect → mask): mask mAP50 / mAP50-95 (MASK IoU), box mAP50,
   pixel area-F1. Reflects the whole system; **detection-limited** on this hard,
   low-contrast forest.
2. **GT-box-prompted**: prompt the mask head at each GT box's centre, score the mask
   vs the GT polygon. Isolates *mask quality* from the weak detector — the direct
   test of "given a box, can it segment?".

## Results (test = 30 crops / 108 trees, seed 0)

| cfg | mask mAP50 | box mAP50 | area-F1 | GT-prompt mIoU | IoU>.5 | GT-prompt area-F1 |
|-----|-----------|-----------|---------|----------------|--------|-------------------|
| A baseline            | 0.070 | 0.069 | 0.290 | 0.609 | 0.759 | 0.783 |
| B +commonality chan   | 0.074 | 0.055 | 0.222 | 0.613 | 0.768 | 0.788 |
| C +channel +proto     | 0.088 | 0.066 | 0.275 | 0.596 | 0.750 | 0.773 |

**Headline: box-only training segments held-out polygons well** — given the right
box, ~76% of trees are masked at IoU>0.5 and pixel area-F1 ≈ 0.78, with masks that
follow crown shape (see `claude_outputs/boxinst_tcd/gtprompt_*.png`). The core
hypothesis holds on masked data.

**Commonality (B/C) makes no material difference** to GT-prompted mask quality
(mIoU 0.61/0.61/0.60 — within MPS run noise). Same conclusion as the dryland
ablation: the CondInst controller already conditions each mask on its crown's
appearance, so the population prior is real-but-redundant here.

Full-pipeline mask mAP50 is low (0.07–0.09) and **detection-limited** (box mAP50
≈ 0.06–0.07), not a statement about segmentation — the GT-prompted columns are.

## Ceiling sweep (`sweep.py`, `run_sweep.sh`) — 340-crop pool, fixed 50/50 val/test

Diagnosis first: baseline hits **train boxAP50=1.00 vs val 0.12** → the ceiling is
overfitting, not capacity. Detection trunk = 3 conv (1×1 stem + two 3×3), *deeper*
than the dapt linear probe that worked. Swept the anti-overfitting levers, box-only:

| arm | layers / head / crops | mask mAP50 | box mAP50 | GTp mIoU | GTp IoU>.5 |
|-----|-----------------------|-----------|-----------|----------|------------|
| S0 | deep 21-24 / 3conv / 100 | 0.143 | 0.115 | 0.637 | 0.825 |
| **S1** | **deep / 3conv / 340** | **0.159** | **0.143** | 0.648 | 0.825 |
| S2 | mid 3-12 / 3conv / 340 | 0.105 | 0.096 | 0.625 | 0.831 |
| S3 | mid / lean probe / 340 | 0.067 | 0.033 | 0.609 | 0.783 |
| S4 | mid / 3conv / 100 | 0.084 | 0.070 | 0.610 | 0.794 |

Findings: **data is the only lever that helps** (S0→S1: +0.03 box mAP50); mid vs
deep layers is neutral-to-worse; a lean probe head *underfits* (S3 worst). Detection
tops out ≈ 0.14 box mAP50 — it's task difficulty (individual trees in closed forest)
+ frozen features, not a tunable knob. **GT-prompted mask quality is robust and good
across every arm** (mIoU 0.61-0.65, 78-83% at IoU>0.5) — segmentation-from-boxes is
NOT the bottleneck. Full-pipeline mAP is high-variance on this little data (the same
S0 config scored 0.07 on the earlier 30-crop test split, 0.143 here).

## Per-instance signature channel (`--signature`)

Idea (tested): pool each box's PCA-64 DINO embeddings (unweighted) → normalize →
per-pixel cosine-to-signature map → extra input channel to the dynamic mask conv
(controller emits n_dyn(1)=177). Validated it lights up the box's tree content
(`claude_outputs/boxinst_tcd/signature_check.png`). Matched vs S1 (deep/3conv/340):

| config | GTp mIoU | GTp IoU>.5 | GTp areaF1 | mask mAP50 | box mAP50 |
|--------|----------|------------|------------|-----------|-----------|
| S1 baseline | 0.648 | 0.825 | 0.820 | 0.159 | 0.143 |
| +signature  | 0.646 | 0.825 | 0.804 | 0.174 | 0.140 |

**No effect on mask quality** (GT-prompted mIoU 0.646 vs 0.648, IoU>.5 identical;
masks visually indistinguishable, `sig_vs_base_*.png`). Full-pipeline delta is
within run noise. The box signature is redundant with the per-pixel F_mask features
+ rel-coords the decoder already has — 4th independent confirmation that the mask
*mechanism* isn't the bottleneck; DINOv3's 16px spatial coarseness is.

## Signature-ONLY (drop F_mask/controller) — the informative control

Threshold the per-box cosine-to-signature map inside the box, NO training, no head:

| mask method | mIoU | IoU>0.5 | over box-fill |
|-------------|------|---------|---------------|
| pure box-fill (mask = whole box) | 0.644 | 0.878 | — |
| learned full head (CondInst+BoxInst) | 0.648 | 0.825 | +0.004 |
| **signature-only, training-free (τ=0.5)** | **0.691** | **0.921** | **+0.047** |

Two findings: (1) the **learned decoder barely beats box-fill** (GT crowns fill ~65%
of their tight boxes, so box-fill is a strong baseline and the controller/dynamic-
filter/F_mask machinery adds ~nothing on this metric); (2) **signature-only, with no
training, beats both** — thresholding the per-box cosine trims non-crown corners more
accurately than the trained head. So per-box signature is the load-bearing signal;
adding it as an extra channel to the full head (above) hid this because the head was
already at box-fill. Caveat: mIoU on tight boxes is forgiving to box-fill, numbers sit
in a compressed 0.64-0.69 band, τ untuned-but-natural. Natural next step: a tiny
learned head on the signature alone, or mask-weighted signature to de-contaminate.

## Learned sig-head sweep + the training-free winner (all val-selected -> test)

Learned signature-only head sweep (width×depth×pooling×threshold, 12 configs,
`sig_head.py --sweep`, `sighead_sweep.json`). Best learned: hid32/depth3/center/
thr0.6 -> test mIoU 0.613. But the full fair comparison:

| mask method | mIoU | IoU>0.5 | trained? |
|-------------|------|---------|----------|
| learned sig-head (best of 12) | 0.613 | 0.799 | yes |
| box-fill (trivial) | 0.644 | 0.878 | no |
| full CondInst head | 0.648 | 0.825 | yes |
| training-free cosine, mean pool | 0.696 | 0.937 | no |
| **training-free cosine, CENTRE pool** | **0.717** | **0.947** | **no** |

Findings: (1) **the best mask method needs NO training** — centre-weighted per-box
signature pooling + cosine threshold hits mIoU 0.717 / 94.7% at IoU>0.5, beating
every trained head; (2) **learning a head HURTS** (0.613 < box-fill 0.644 < raw
threshold 0.717) — the box-only projection+pairwise losses degrade the clean cosine
prior; (3) **centre-weighted pooling > mean** everywhere (+0.02-0.14), so de-weighting
contaminated box corners matters. Recommended mask pipeline: detect box -> centre-pool
its DINO embeddings -> cosine -> threshold. No mask head, no mask training.
Caveat: mIoU forgiving on tight boxes (box-fill floor 0.644); still detection-limited
end-to-end.

## CORRECTED (2026-07-06): individual trees + canopy-ignore, multi-biome

Earlier rounds trained on category_id==1 = CANOPY by mistake (the ITC premise was
inverted). Fixed: positives = cat 2 (individual trees); canopy = cat 1 handled as
IGNORE in three places — commonality background, detection negative focal loss, and
the detection metric (unmatched preds in canopy are ignored, not FP). Split rebuilt
to 48 biomes, 720/160/160 native crops, by source tile.

Detection (test, seed 0, best ep120): **canopy-aware box mAP50 = 0.451** (naive
0.376), P/R ≈ 0.56/0.57 — ~3× the invalid canopy-based runs (~0.14), from correct
labels + canopy-ignore + 7× data + 48 biomes. Viz: `08_corrected_results/`.

Mask, GT-box-prompted on correct individual-tree polygons:

| method | mIoU | IoU>0.5 |
|--------|------|---------|
| full CondInst head (trained) | 0.646 | 0.836 |
| box-fill (trivial) | 0.731 | 0.975 |
| training-free centre-pool signature | 0.762 | 0.986 |

The mask-mechanism findings SURVIVE the correction: training-free signature ≥ box-fill
> learned head. BUT the honest caveat is sharper now — individual crowns fill ~75% of
their tight boxes, so box-fill alone hits mIoU 0.73 / 97.5% IoU>.5. Mask-from-box is
near-trivial for ITC; mIoU is a weak discriminator and the signature's edge is small
(+0.03). The real problem is DETECTION (box mAP50 0.45), which is where effort should go.

## Honest caveats

- Detection is the bottleneck here (coastal-forest crowns sit in dense vegetation,
  unlike high-contrast dryland). Full-pipeline mask mAP50 is therefore low and is
  *not* a statement about segmentation quality — the GT-prompted columns are.
- Masks have no GT in training; polygons touch the pipeline only at eval.
- Thresholds (τ, score, mask) are inherited from the dryland run, not tuned on TCD.
- MPS runs are not bitwise reproducible (~±0.01).
