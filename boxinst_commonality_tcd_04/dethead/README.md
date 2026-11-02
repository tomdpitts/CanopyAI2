# dethead: FCOS-style head for LACE's detector (local only; Tom 2026-11-02)

Question: is LACE's weak localisation (box AP75 0.197 vs BoxTeacher 0.368; box AP50 0.666 vs 0.703) a property of the CenterNet head, or of the frozen features? Swap ONLY the head, the target assignment and the losses. Keep everything else. Report AP50 and box AP50 on the 439, through the full LACE pipeline.

## Pre-registration (written 2026-11-02, BEFORE any number from this experiment)

**Held fixed (= the deployed `Detector4Phase`, `modal_tcd_multiseed/phase4/HYPERPARAMS.md` §3):**
- Features: the Modal-cached L24 4-phase 8 px DINOv3 features (1024 × 256 × 256), train `tcd04-phase4-vol:feat_4p_train`, pulled verbatim. Val 108 and test 439 are the existing local copies of the same Modal caches.
- Trunk: stem 1×1 → 256, GN, ReLU; tower 3 × (3×3, GN, ReLU); `up` 2 × (3×3, GN, ReLU). Identical modules.
- Recipe: Adam lr 1e-3, wd 1e-4, CosineAnnealing T_max 40, batch 3 tiles, 40 epochs max. Eval every 5 epochs on the 108 val tiles. Early stop: min 12 epochs, patience 2 evals, min_delta 0.005 on val box mAP50 (`dapt.eval.average_precision`, IoU 0.5). Model selection: best val box mAP50. Seed 0.
- Canopy: canopy cells (> 50 % covered) are excluded from the classification NEGATIVES, LACE's rule. Positives are always supervised.
- Downstream: the LACE masker `lace4px_simple` (FROZEN_simple_2026-10-26) with its own 4 px s′ ranking, s = the detector score. Scored by the frozen `score_coco`, 2048², maxDets 600, canopy crowd arm, floor 0. That is the LACE protocol: the detector-score floor 0.05 is applied upstream.

**Changed (FCOS, Tian et al. 2019, with its published defaults where they apply):**
- Heads on the shared 8 px map: classification (1), box distances l, t, r, b (4; d = 8 · exp(x)), centre-ness (1).
- Anchor point of cell (X, Y): pixel (8X + 8, 8Y + 8), the feature grid's true cell centre.
- Assignment: a cell is positive for a GT box if its anchor is inside the box AND within the centre-sampling radius 1.5 strides (12 px) of the box centre. If a box gets no such cell, the cell whose anchor is nearest its centre is made positive for it. Ambiguous cells go to the smallest-area box.
- Losses: sigmoid focal loss on classification (α 0.25, γ 2, normalised by #positives). GIoU loss on positive boxes, weighted by the centre-ness target and normalised by its sum. BCE on centre-ness. Equal weights.
- Inference: s = √(σ(cls) · σ(ctr)). Candidates: s > 0.05, top 4000, then NMS IoU 0.6 (FCOS default), keeping the top 600 per tile.

**Arms on the 439 (all through the identical masker + scorer):**
1. **FCOS head**, seed 0, the primary.
2. **Gate / reference:** the published LACE s0 detector checkpoint (`det_phase4_L24_s0.pt`, Modal-trained), decoded locally on the same test features. Its boxes must reproduce the published s0 detector boxes, and its masked row the published 0.6556 / box 0.6748.
3. **Control (run after 1, if time allows):** the CenterNet `Detector4Phase` re-trained LOCALLY with the identical recipe, seed 0. This separates the head effect from local-MPS-vs-Modal-CUDA training.

**Report:** mask AP50, AP75, AP50:95 and box AP50, box AP75 for each arm, with the s′ ranking (LACE protocol). The detector-score ranking is a secondary row. One seed: a difference within about ±0.01 of the reference is not distinguishable from seed noise (LACE's seed sd is 0.004 AP50 and 0.011 box AP50).

## RESULTS (2026-11-02)

**Gate:** a local decode of the published `det_phase4_L24_s0.pt` reproduces the published s0 test boxes: counts equal, 99.4% of 498 boxes within 0.05 px, scores within 2.5e-4 (`results/gate_decode.json`). Two stricter earlier gate versions failed on CPU-vs-CUDA tie re-ordering and are kept, renamed.

**Training (val box mAP50, `dapt` metric).** FCOS early-stopped at epoch 35, best epoch 25: **0.5497**. The local CenterNet control stopped at epoch 25, best epoch 20: 0.4935. The Modal-trained deployed s0 scored 0.4975.

**Test, 439 tiles, single seed, as pre-registered (FCOS NMS 0.6):**

| arm | ranking | mask AP50 | AP75 | AP50:95 | box AP50 | box AP75 |
|---|---|---|---|---|---|---|
| LACE s0, Modal (published) | s′ | 0.6556 | 0.181 | 0.284 | 0.6748 | 0.197 |
| CenterNet, local control | s′ | 0.6501 | 0.166 | 0.276 | 0.6610 | 0.169 |
| **FCOS** | s′ | 0.6386 | 0.203 | 0.289 | 0.6754 | 0.297 |
| FCOS | detector s | 0.6453 | 0.202 | 0.289 | 0.6823 | 0.289 |

Against the local control (same machine and recipe): FCOS box AP50 +0.014 and box AP75 +0.128, but mask AP50 −0.012 (s′). The local control sits inside LACE's seed band (0.650–0.658).

**Val diagnostics (test-informed, `results/diag_val.json`, `results/diag_box_val.json`).**
- FCOS boxes are not small: 1.02× GT, like BoxTeacher; CenterNet's are 0.975×.
- At NMS 0.6, 35% of FCOS detections are near-duplicates (IoU 0.5–0.6 with a higher-scored detection). NMS 0.5 removes them.
- Val mask AP50:
  - CenterNet s′: 0.675.
  - FCOS NMS 0.6: s′ 0.666, detector s 0.679.
  - FCOS NMS 0.5: s′ 0.676, **detector s 0.680**.
  - s′ helps CenterNet (+0.034) and hurts FCOS.
- Box AP50 gap, FCOS NMS 0.5 (0.696) vs BoxTeacher (0.721) on val:
  - Max recall: 0.822 vs 0.851.
  - Recall by GT size: < 24 px (26% of crowns) **0.577 vs 0.678**; 24–48 px 0.883 vs 0.872; 48–96 px 0.937 vs 0.936; > 96 px **0.872 vs 0.951**.
  - Touching vs isolated: no specific deficit.
  - Precision at recall 0.6: 0.804 vs 0.829. That is 1,030 vs 871 false positives ranked above recall 0.6, with the same mix: 2/3 background, 1/3 mislocalised.

**Seam diagnostic (val, `results/diag_seam_val.json`, `results/diag_seam_val_sizematched.json`).** LACE features are extracted per 1024 window (2×2 per tile), so x=1024 / y=1024 are feature seams. The comparison is recall at IoU 0.5 for GT crowns straddling a seam vs far crowns re-weighted to the straddlers' size distribution:

| row | straddle | far, size-matched | diff |
|---|---|---|---|
| LACE CenterNet s0 | 0.801 | 0.840 | −0.039 |
| FCOS NMS 0.5 | 0.830 | 0.887 | −0.057 |
| BoxTeacher (overlapping 3×3) | 0.912 | 0.908 | +0.004 |

The penalty is real for LACE features, about 2–2.6 SE, and largest at 24–48 px (−0.10 to −0.12). But straddlers are only 4.9% of crowns (342 / 7,041), so a full fix is worth about +0.25% recall overall, roughly +0.002–0.003 AP50. It is not the main box AP50 gap.

## Pre-registration 2 (2026-11-02, after the FCOS test run, the val diagnostics and the seam diagnostic; BEFORE any pyramid number)

**Arm P: FCOS + feature pyramid on the same frozen 8 px features.**
- P3 = the unchanged trunk output (8 px, 256²). P4 = 3×3 stride-2 conv + GN + ReLU on P3 (16 px, 128²). P5 = the same on P4 (32 px, 64²).
- Shared 1×1 cls / box / ctr heads on all levels (the same heads as flat FCOS), with a learnable per-level scale on the box log-distances.
- Anchors: pixel stride·index + 8 per level.
- Level assignment as FCOS / AdelaiDet, the ranges BoxTeacher was released with: max(l, t, r, b) ∈ [0, 64] → P3, (64, 128] → P4, > 128 → P5. Centre-sampling radius 1.5 × level stride. The tiny-box fallback applies on P3 only. Ambiguities go to the smallest area, per level. The canopy ignore is applied per level (> 50% covered).
- Training recipe, losses and seed: identical to flat FCOS.

**Decode, chosen on val for each head separately** (108 tiles, full LACE pipeline, `score_lite` crowd mask AP50): NMS ∈ {0.5, 0.6} × ranking ∈ {s′, detector s}. Pick the highest val mask AP50; ties within 0.001 go to NMS 0.5 and detector s.

**Test (439, frozen `score_coco`):** arm P at its val-selected decode, AND flat FCOS at its val-selected decode. The val diagnostics already showed NMS 0.5 + detector s, so this is a second, test-informed look at flat FCOS, disclosed as such. Each is run once.

**Report:** mask AP50 / AP75 / AP50:95 and box AP50 / AP75, plus val recall by size (the > 96 px stratum is the pyramid's target).

## Pre-registration 3 (2026-11-02, BEFORE any ConvNeXt number): frozen DINOv3 ConvNeXt-L backbone + ATSS / GFL head

Tom's choice: ConvNeXt-L, the strongest released distilled DINOv3 ConvNeXt, for the best AP50. Route A: the detector changes, the masker is unchanged (`lace4px_simple` on ViT-L 4 px features).

- **Backbone:** `facebook/dinov3-convnext-large-pretrain-lvd1689m`, frozen, fp32 compute.
  - Input: the full 2048² tile in one pass, ImageNet mean / std normalisation, no resize.
  - Features: stage outputs (`hidden_states[1..4]`) at strides 4 / 8 / 16 / 32, 192 / 384 / 768 / 1536 channels.
  - Train + val are cached at fp16. Test is computed live.
  - Same tile pixels as LACE: local tifs verified against the phase4 manifest sha1. The one local-missing train tile (`tcd_tile_1878`) comes from the Tier B volume's HF-staged PNG.
- **Neck:** per level, a 1×1 conv → 192 + GN + ReLU.
- **Head:** a shared tower of 3 × (3×3 conv 192, GN, ReLU) over all levels, then 1×1 outputs for classification / quality (1) and box distances l, t, r, b (4; d = stride · exp(scale_level · x)). No centre-ness.
- **Assignment: ATSS** (Zhang et al. 2020), anchor-free form.
  - Each location carries a square anchor of side 8 × stride, centred at stride · index + stride / 2.
  - Per GT: the top-9 anchors by centre distance on each level form the candidates. Threshold = mean + std of their IoUs with the GT. Positives = candidates with IoU ≥ threshold AND centre inside the GT box.
  - A location claimed by several GTs goes to the one with the highest IoU.
- **Losses (GFL, Li et al. 2020):** quality focal loss on the classification output (target = IoU of the predicted box with its GT for positives, 0 for negatives; β = 2; normalised by #positives). GIoU loss on positives, weighted by the detached quality target. Canopy cells (> 50% covered, per level) are excluded from the negatives (LACE's rule).
- **Recipe:** identical to pre-registrations 1–2 (Adam 1e-3, wd 1e-4, cosine 40 epochs, batch 3 tiles, val every 5 epochs, early stop min 12 / patience 2 / min_delta 0.005 on val box mAP50, seed 0).
- **Decode:** score = σ(cls). Per level: s > 0.05, top 1000. Then NMS over levels, keeping 600 per tile.
- **Selection and test:** NMS ∈ {0.5, 0.6} × ranking ∈ {s′, detector s}, chosen on val by the full LACE pipeline's mask AP50 (ties → NMS 0.5, detector s). Then a single test run on the 439, frozen `score_coco`.
- **Report:** mask AP50 / AP75 / AP50:95, box AP50 / AP75, and val recall by size, against BoxTeacher (test 0.664 / box AP50 0.703) and LACE (0.655 / 0.666).

## RESULTS, pre-registration 2 (pyramid; 2026-11-02)

FPN early-stopped at epoch 35, best epoch 25: val box mAP50 0.5525.
- **Val selection:** NMS 0.5 + detector s. Val mask AP50 0.691, AP75 0.255; box AP50 0.704, box AP75 0.308.
- **Val recall by size:** < 24 px 0.582, > 96 px **0.925** (flat FCOS 0.872, BoxTeacher 0.951). Precision at recall 0.6: 0.815.

**Test (439, NMS 0.5):**

| ranking | mask AP50 | AP75 | AP50:95 | box AP50 | box AP75 |
|---|---|---|---|---|---|
| detector s (val-selected) | 0.6481 | 0.206 | 0.292 | 0.6875 | 0.295 |
| s′ | 0.6479 | 0.207 | 0.293 | 0.6873 | 0.302 |

- **Box AP50:** +0.013 vs LACE s0 (0.675) and +0.027 vs the local CenterNet control (0.661). Still −0.016 vs BoxTeacher (0.703).
- **Mask AP50:** 0.648, still below LACE (s0 0.656, 3-seed 0.655). The val → test drop is −0.043 (CenterNet's is −0.019).
- **Mask − box AP50, by box source:**
  - CenterNet boxes: +0.009 (val), −0.019 (test).
  - FCOS-family boxes: −0.013 (val), −0.039 (test).
  - BoxTeacher boxes + LACE masker: −0.036 (test).

  With accurate boxes, the masker gives back much of the box gain, so mask AP50 is now limited by the masker on tight boxes, not by detection. (The masker's knobs were chosen on CenterNet boxes; the btmask val grid found no better `c_peak` or inflation for tight boxes.)

**Flat FCOS re-test at its val-selected decode (NMS 0.5 + detector s; a test-informed second look, disclosed):** mask AP50 0.6489, AP75 0.200, AP50:95 0.289; box AP50 0.6857, box AP75 0.286 (s′: mask 0.6499, box 0.6865). That is level with the pyramid on test (0.6481 / 0.6875). The pyramid's val gain (+0.011 mask, +0.008 box) did not reach test.

### Amendment 3a (2026-11-02, after epoch 10 of the first ConvNeXt-L run; BEFORE any test number)

The first run (`runs/cnxl_s0_noLN`) was stopped at epoch 10 with val box mAP50 0.362 (ViT-L heads at the same epoch: 0.478–0.508). On a val subset, the score floor was ruled out (AP identical for thresholds 0.001–0.1) and recall was about 0.71.

The cause found: the raw ConvNeXt stage outputs were fed unnormalised. They carry outlier channels: s3 and s4 have their top-5 channels at about 44–52% of the variance, a 26–32× channel-std ratio, and max |x| about 300 (s3) and 15–36 (s4). The ViT-L features, by contrast, are L2-normalised patch tokens.

**Change (revised before any run of it):** frozen per-channel standardisation of every stage output before the lateral 1×1, `(x − μ_c) / σ_c`, affine-free, as MAE's linear-probe BatchNorm (`cnx.CNXHead(stage_norm="chanstd")`).
- The statistics come from 200 training tiles (every 4th), spatially subsampled: `feat_cnxl/chan_stats_train200.npz`. Train only; never val or test.
- A channel LayerNorm was tried first and checked on one tile: it fixed the scale across stages but left s4's channel-std ratio at 31×, so it was replaced.
- Nothing else changes: features, ATSS, GFL, recipe, selection and the single test run all stay as in pre-registration 3. The new run is tagged `cnxl_ln_s0` (name kept for "normalised").

## Diagnostic: where mask AP50 loses against box AP50 (val, 2026-11-02; `diag_maskbox.py`, `results/diag_maskbox_val.json`)

No selection or tuning. Same detections and ranking in both evals, `score_lite` crowd arm. Each detection gets a box status and a mask status (TP / FP / IGN, where IGN means ignored by a canopy crowd region). Each change of status is applied alone to the box statuses and AP50 is recomputed. A per-detection AP rebuild reproduces COCOeval exactly for every row. The CenterNet s′ row gives 0.6751 against FROZEN_simple 0.6752: the same 0.0001 render-path offset `diag_val.py` already showed. It is not material.

| row | box AP50 | mask AP50 | diff | TP→FP | FP→TP | IGN→FP (canopy) | IGN→TP |
|---|---|---|---|---|---|---|---|
| BoxTeacher native | 0.7214 | 0.7002 | −0.021 | −0.023 (181) | +0.012 (93) | −0.016 (5,777) | +0.003 |
| BT boxes + LACE masker, BT score | 0.7214 | 0.7040 | −0.017 | −0.023 (205) | +0.016 (107) | −0.016 (5,688) | +0.003 |
| BT boxes + LACE masker, s′ | 0.7094 | 0.6892 | −0.020 | −0.024 (208) | +0.017 (109) | −0.018 (5,687) | +0.003 |
| LACE CenterNet s0, s′ | 0.6657 | 0.6751 | +0.009 | −0.019 (151) | **+0.036** (230) | −0.014 (3,871) | +0.011 |

- **About 80% of the val gap is a scoring asymmetry, not the masker.** In the box eval, a detection is ignored if it overlaps a canopy region's bounding box (IoA ≥ 0.5). The mask eval uses the exact canopy mask, so about 5,700 detections inside canopy bounding boxes but off the canopy count as false positives. That costs −0.016 to −0.018 for every row, BoxTeacher's own included. Box AP50 is therefore inflated relative to mask AP50, and box AP50 gains in those zones do not carry over to mask AP50.
- **The masker's own net cost is small, and no worse than BoxTeacher's.** TP→FP plus FP→TP on BoxTeacher boxes: LACE masker −0.007, BoxTeacher's own masks −0.011.
- **Masker failures sit at box IoU 0.5–0.6.** About 25% of box-TPs in that band fail as masks (median mask IoU 0.47), against 4% at 0.6–0.7 and ≤ 1% above. Mask AP50 is governed by box quality at IoU ≈ 0.6, not 0.5.
- **The masker rescues loose CenterNet boxes** (FP→TP +0.036), which is why CenterNet's mask AP50 exceeds its box AP50. Tight boxes leave less to rescue.
- **Implication:** box AP50 is the wrong proxy. Select detectors on full-pipeline mask AP50 (as pre-registrations 2–3 already do) or on box AP at IoU 0.6–0.7, not on box AP50. There is little left to gain inside the masker on tight boxes (−0.007 net).

### Pre-registration 3 / 3a outcome (2026-11-02): stopped, no test run
`cnxl_ln_s0` was stopped during epoch 6 on Tom's go. Val box mAP50 at epoch 5 was 0.333, against 0.313 for the unnormalised run and 0.44–0.47 for the ViT-L heads at the same epoch. Per-channel standardisation recovered only about +0.02. No val selection and no test number exist for ConvNeXt-L. Checkpoints are kept in `runs/cnxl_ln_s0/` and `runs/cnxl_s0_noLN/`; the feature cache `feat_cnxl/` is kept until Tom decides.
The ConvNeXt-L stage caches `feat_cnxl/s1..s4` (161 GB) were deleted on Tom's go (2026-11-02). They can be regenerated with `cnx.py` extraction from the phase4-manifest tiles. The channel statistics are kept in `results/cnxl_chan_stats_train200.npz`.
