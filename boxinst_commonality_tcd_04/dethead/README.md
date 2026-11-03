# dethead: FCOS-style head for LACE's detector (local only; Tom 2026-11-02)

<!-- TABLE1_CURRENT:begin (auto, results_439/mark_superseded.py) -->
> ⚠️ **SUPERSEDED 439 / sparse236 NUMBERS (marked 2026-10-27; table updated 2026-11-03).** Current headline table: [`TABLE1_CURRENT.md`](../results_439/TABLE1_CURRENT.md). Test-set numbers in this file are a historical experiment record (512 raster, pre-seam-fix baselines, 8 px masker or earlier models) and must not be quoted unless they match that table.
<!-- TABLE1_CURRENT:end -->

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

## Pre-registration 4 (2026-11-02, BEFORE any 4 px feature or arm-C number): arm C = 8 px + 4 px P2 + pixel branch

Plan and rationale: `PROPOSAL_2026-11-02.md`. Tom chose arm C only (budget); arms A, B and D are deferred. Steps 1–3 run on Modal with Tom's OK. Training also runs on Modal (step 5).

**4 px features.**
- `feat16p.py`'s definition, verbatim: DINOv3-web ViT-L/16, block 24, L2-normed; 16 passes at shifts {0,4,8,12}², each windowed 2×2 at 1024; interleaved to a 512² lattice with cell (Y, X) centred at (8 + 4X, 8 + 4Y); projected with `em_model_4p_fix.npz` (sha1 2f0eaa3f…) to 128-d, unit norm; stored fp16.
- Image pins as `phase4_modal.py` (torch 2.12.1, transformers 4.57.1, numpy 2.2.6), the same as the local `venv_feat_t457`.
- Tiles: HF `restor/tcd` joined by `manifest.json` image_id, with the RGB sha1 checked per tile. 900 train (incl. the 108 val) + 439 test.
- Written to `tcd04-phase4-vol:z4p_train/`, `z4p_test/`.
- Gates:
  - G1, per tile: the {0,8}² sub-lattice vs the projection of the same volume's `feat_4p_*`, cosine mean ≥ 0.995 and p1 ≥ 0.98, else abort.
  - G2, before the full run: 2 val tiles vs the local `hunt_z4p_val`, full 512² lattice, cosine mean ≥ 0.999.
- Used by the detector only. The masker keeps its local `hunt_z4p_{val,test}` (published inputs).

**Arm C (`fpnC_s0`).** Pre-registration 2's pyramid head, unchanged where not stated.
- P3 = the unchanged trunk on the 1024-d 8 px features, plus a zero-initialised 1×1 projection of the pixel branch's stride-8 map.
- P4 and P5 are stride-2 3×3 conv + GN + ReLU from the fused P3, as before.
- **P2** (stride 4, 512²) = 1×1 conv 128 → 256 + GN + ReLU on the 4 px features, plus a nearest-neighbour 2× upsampling of the fused P3, plus a zero-initialised 1×1 projection of the pixel branch's stride-4 map; then 3×3 conv 256 + GN + ReLU.
- **Pixel branch:**
  - Input: the RGB tile in [0, 1], ImageNet mean/std.
  - Layers, each 3×3 conv + GN + ReLU: s2 3 → 32, s1 32 → 32 (stride 2); s2 32 → 64, s1 64 → 64 (stride 4, feeds P2); s2 64 → 128 (stride 8, feeds P3).
- Shared 1×1 cls / box / ctr heads on P2–P5, with a learnable per-level box scale.
- **Level ranges** on m = max(l, t, r, b): P2 [0, 32], P3 (32, 64], P4 (64, 128], P5 > 128. Anchors at stride · index + 8 on every level. Centre radius 1.5 × stride. The tiny-box fallback (nearest anchor) is on P2 only. Ambiguities go to the smallest area. Canopy cells > 50% covered are excluded from the negatives on every level, P2 included.
- Losses as FCOS / pre-registration 2.

**No augmentation.** Flipping cached ViT feature maps is not equivalent to features of a flipped image. Aerial shadows have a consistent direction, and the 8 px and 4 px lattices (origin 8) are not flip-symmetric. It would train on input/geometry combinations never seen at test, and it would add a second variable against `fpn_s0` (no augmentation).

**Recipe:**
- As pre-registrations 1–3: Adam lr 1e-3, wd 1e-4, cosine T_max 40, batch 3 tiles, 40 epochs max, eval every 5, early stop min 12 / patience 2 / Δ 0.005, seed 0, best checkpoint kept.
- **Changed selection metric:** val box AP at **IoU 0.6**, crowd arm (`score_lite` GT, canopy = iscrowd), maxDets 600, floor 0.05. This replaces the `dapt` box mAP50, which counts canopy as FP. IoU 0.6 follows the mask/box diagnostic (`results/diag_maskbox_val.json`).

**Decode:** per level, s = √(σ(cls) · σ(ctr)) > 0.05, top 4000; then NMS over all levels, top 600 per tile.

**Selection (val, full LACE pipeline):** NMS {0.5, 0.6} × ranking {s′, detector s}, by `score_lite` crowd mask AP50 (masker `lace4px_simple`, local, unchanged). Ties within 0.001 → NMS 0.5, detector s.

**Final retrain on train + val:** only if Tom approves BEFORE the test run. It would use the selected epoch count, the same cosine T_max 40, the last epoch and the selected decode. Otherwise the val-selected checkpoint is tested.

**Test:** once, on the 439 tiles. Detector inference on Modal (8 px `feat_4p_test`, `z4p_test`, RGB); masker and frozen `score_coco` locally (2048², maxDets 600, crowd arm).

**Report:**
- mask AP50 / AP75 / AP50:95 and box AP50 / AP75;
- val recall by size (< 24, 24–48, 48–96, ≥ 96 px);
- the mask/box decomposition on val;
- against `fpn_s0` (0.648 / 0.6875), LACE (0.655 / 0.666) and BoxTeacher (0.664 / 0.703);
- single seed (sd ≈ 0.004 mask / 0.011 box AP50);
- the actual Modal bill.

### Amendment 4a (2026-11-02, after gate G2; BEFORE any detector number)
- **G2 PASS** (A100, 2 val tiles): Modal 4 px vs local `hunt_z4p_val`, cosine mean 1.000000, p1 0.99987 / 0.99988, min 0.99972 / 0.99954. G1 cosine 1.00000.
- Measured speed: 20 s/tile on A100 (fp32 ViT-L), about 7.4 A100-h for 1,339 tiles (about $20).
- **Change (Tom):** the extraction GPU is now **H100** (about 3× fp32 throughput at about 1.9× the price, about $12). Everything else is unchanged. G1 still gates every tile.
- 6 shards: train 4, test 2. Container cpu 4, memory 32 GB.

### Amendment 4b (2026-11-02, during extraction; no detector number exists)
`manifest.json` carries `rgb_sha1` for only a 20-tile sample per split; the original phase4 `verify()` checks only that sample. The first H100 launch asserted it for every tile, so all train shards failed with a KeyError on the first unhashed tile. Tiles already written are valid: they were hashed and passed G1.
- **Change:** the sha1 is asserted where the manifest has one. Every tile asserts width and height against the manifest and is joined by image_id, as the original extraction was.
- Pixel identity for every tile is still evidenced by G1: cosine against `feat_4p_*` from the same image_id join, observed 1.00000.
- Shards are relaunched; resume skips completed tiles.
- The 2 G2 tiles (`tcd_tile_1012`, `tcd_tile_1041`, extracted on A100) are kept in `z4p_train`; G1 for them was 1.00000.
- **Arm C smoke test PASS** (A100, 6 train / 6 val tiles, 1 epoch, tag `fpnC_s0_smoke`): HF join, 8 px + 4 px + RGB loading, P2–P5 targets, loss, checkpoint, val metric all run; peak GPU memory 12.0 GB at batch 3. Its metric (0.0003) is meaningless and unused.
- Code: `trainC_modal.py` (train / predict on Modal), `evalC.py` (local masker + scoring for selection and the single test), `fcos.FCOSC` / `LEVELS_C` (the edit only turns the level table into a parameter of `encode_fpn` / `loss_fpn` / `decode_fpn`, defaulting to `LEVELS`, so existing heads are unchanged).
- **Decision (Tom, 2026-11-02, before any arm-C number): no train+val retrain.** The val-selected `best.pt` (792 training tiles) is the model tested, as `fpn_s0` and BoxTeacher were.

## RESULTS, pre-registration 4 (arm C `fpnC_s0`; 2026-11-02)

**Training (Modal A100):** early stop at epoch 35, best epoch 30. Val crowd box AP@0.6 by epoch: 0.419 / 0.521 / 0.575 / 0.589 / 0.594 / **0.597** / 0.591 (epochs 5–35). Epochs take 66 s once the page cache is warm. BoxTeacher on the same metric: 0.654 (val crowd box AP@0.5 0.721).

**Val selection (108 tiles, full LACE pipeline, `select_val_fpnC_s0.json`):** selected NMS 0.5 + detector s.

| decode | mask AP50 | AP75 | AP50:95 | box AP50 | box AP75 |
|---|---|---|---|---|---|
| NMS 0.5, s′ | 0.6841 | 0.240 | 0.319 | 0.6999 | 0.278 |
| **NMS 0.5, detector s** | **0.6844** | 0.227 | 0.314 | 0.7011 | 0.269 |
| NMS 0.6, s′ | 0.6623 | 0.243 | 0.314 | 0.6779 | 0.282 |
| NMS 0.6, detector s | 0.6746 | 0.231 | 0.313 | 0.6905 | 0.273 |

**Val, arm C vs `fpn_s0` (both NMS 0.5 + detector s) vs BoxTeacher:**

| | mask AP50 | box AP50 | max recall | recall < 24 px | 24–48 | 48–96 | ≥ 96 | precision at recall 0.6 | FP above recall 0.6 (background share) |
|---|---|---|---|---|---|---|---|---|---|
| `fpn_s0` | 0.6907 | 0.7036 | 0.837 | 0.582 | 0.893 | 0.955 | 0.925 | 0.815 | 959 (66%) |
| **arm C** | 0.6844 | 0.7011 | **0.856** | **0.678** | 0.888 | 0.946 | 0.917 | 0.799 | 1,065 (70%) |
| BoxTeacher | 0.7002 | 0.7214 | 0.851 | 0.678 | 0.872 | 0.936 | 0.951 | 0.829 | 871 |

- **Arm C closed the small-crown recall gap:** < 24 px recall is 0.678, equal to BoxTeacher and +0.096 over `fpn_s0`. Max recall now exceeds BoxTeacher's.
- **But it added background false positives:** +106 above recall 0.6, with the background share up from 66% to 70%. Precision fell, so box AP50 is flat (−0.003) and val mask AP50 is −0.006 against `fpn_s0`.
- Single seed: these differences are within about 1–2 seed sd.

### Diagnostic: arm C false positives (val, after selection; `diag_fp.py`, `results/diag_fp_val.json`)
Crowd bbox, IoU 0.5. "Top" = detections ranked above the score where recall reaches 0.6. Entries are TP / FP (precision).

| det size | arm C, top | BoxTeacher, top |
|---|---|---|
| < 24 px | 539 / **270** (0.666) | 252 / 58 (0.813) |
| 24–48 | 1,529 / 527 (0.744) | 1,483 / 416 (0.781) |
| 48–96 | 1,860 / 209 (0.899) | 2,016 / 336 (0.857) |
| ≥ 96 | 297 / 59 (0.834) | 474 / 61 (0.886) |

- **The extra false positives are small detections ranked too high.** Arm C puts +287 more small TPs and +212 more small FPs above the recall-0.6 cut; its small-box precision there is 0.666 against 0.74–0.90 for larger boxes. Over all detections, arm C actually has fewer small FPs than BoxTeacher (9,543 vs 11,836). It is a ranking (calibration) problem, not a volume problem.
- **Canopy is not a factor:** 1,001 of 1,065 top FPs touch no canopy.
- **Candidate fix (needs its own pre-registration):** per-level or per-size score calibration fitted on TRAIN-tile detections, never val, then a single val check and a single test run.

**Test (439 tiles, single run, frozen `score_coco`, crowd arm; `summary_fpnC_s0_nms05.json`).** The first three test-predict calls produced no output (an NMS-argument parse bug, then a stale warm container still running the old code); the fourth is the run.

| arm C ranking | mask AP50 | AP75 | AP50:95 | box AP50 | box AP75 |
|---|---|---|---|---|---|
| **detector s (val-selected)** | **0.6485** | 0.187 | 0.286 | 0.6862 | 0.249 |
| s′ (not selected; val tie 0.6841 vs 0.6844) | 0.6587 | 0.201 | 0.296 | 0.6947 | 0.262 |

- **Headline (val-selected): 0.6485 / box 0.6862**, level with `fpn_s0` (0.6481 / 0.6875). Below LACE s0 (0.6556 / 0.6748) and BoxTeacher (0.664 / 0.703).
- The s′ row is better on test by +0.010, but the pre-registered tie rule picked detector s on val. Reporting s′ as the arm-C result would be a test-informed choice; it is recorded, not adopted.
- **Arm C does not beat BoxTeacher.** The 4 px level and pixel branch closed the small-crown recall gap on val, but extra high-ranked small false positives offset it (`diag_fp_val.json`).

## Pre-registration 5 (2026-11-02, after arm C's test; BEFORE any calibrated number): per-level score calibration for arm C

**Motivation:** `diag_fp_val.json` shows arm C's small detections ranked too high (top-ranked < 24 px precision 0.666 vs 0.74–0.90 for larger boxes). This is a second look at the arm-C detector on test, **disclosed as test-informed**: it was chosen after arm C's test row (0.6485) was known.

**Detector and boxes unchanged:** `fpnC_s0` `best.pt`, NMS 0.5 (val-selected), the same detection set (raw detector s ≥ 0.05, top 600).
- `predict` is re-run with each detection's pyramid level recorded (P2–P5).
- Gate: the re-run boxes and scores must equal the tested ones (same counts per tile, ≥ 99% of boxes within 0.05 px, scores within 1e-4), else abort.

**Calibration (Platt, per level):** s_cal = σ(a_l · logit(s) + b_l), with s the detector score (clipped to [1e-6, 1 − 1e-6]) and l ∈ {P2, P3, P4, P5}; 8 parameters.
- Fitted by unregularised logistic regression (L-BFGS) on the **val** 108 tiles' detections.
- Label: 1 if the detection is a TP in the crowd-arm **mask** eval at IoU 0.5 (`score_lite`, segm, maxDets 600, LACE masks `lace4px_simple` as rendered for selection); 0 if FP. Crowd-ignored detections are excluded.
- A level with fewer than 50 val detections keeps the identity (a = 1, b = 0).
- Applied after NMS: it only re-orders detections across levels; within a level the order is unchanged.

**Evaluation:**
- Val mask AP50 with s_cal (in-sample, reported as such).
- **Test once:** the same masks (`feat_cache/dethead_fpnC_s0_nms05_mask`), ranking s_cal, the detection set = raw s ≥ 0.05, frozen `score_coco` (2048², maxDets 600, crowd arm, floor 0 on s_cal since the set is fixed upstream).

**Report:** vs arm C detector s (0.6485) and s′ (0.6587), `fpn_s0` (0.6481), LACE (0.6556 s0 / 0.655) and BoxTeacher (0.664); the fitted (a_l, b_l); val precision by size above recall 0.6 (as `diag_fp`). Single seed.

### Amendment 5a (2026-11-02, before any calibrated number; Tom: run it locally)
- **No re-run of `predict`, so no gate.** The level is taken from the saved boxes by the training level ranges. FCOS assigns a box to a level by m = max(l, t, r, b) from the anchor, and m ≈ max(w, h) / 2 for a centred anchor. So the levels are P2: max(w, h) ≤ 64 px; P3: (64, 128]; P4: (128, 256]; P5: > 256.
- **This is a proxy:** off-centre anchors and the regression make it approximate.
- Everything else as pre-registration 5. It uses the saved val preds and masks (`results/modal/fpnC_s0_val_nms0.5.json`, `feat_cache/dethead_fpnC_s0_val_nms05_mask.json`) and the tested test preds and masks. $0, local.
- The `return_level` / `with_level` code paths added to `fcos.decode_fpn` and `trainC_modal.predict` are kept but unused (their defaults reproduce the previous outputs exactly; checked on random maps).

## RESULTS, pre-registration 5 / 5a (per-level calibration; 2026-11-02; `calib_fpnC_s0_nms05.json`, `summary_fpnC_s0_nms05_cal.json`)

**Fitted (val, mask-TP labels):**

| level (max side, px) | a | b | n | TP rate |
|---|---|---|---|---|
| P2 (≤ 64) | 3.499 | −0.536 | 27,456 | 0.139 |
| P3 (64–128] | 4.121 | +0.190 | 8,960 | 0.212 |
| P4 (128–256] | 3.908 | −0.200 | 1,926 | 0.129 |
| P5 (> 256) | 3.611 | −0.943 | 1,035 | 0.030 |

**Val (in-sample):** mask AP50 0.6844 → **0.6922** (+0.008); box AP50 0.7011 → 0.7069. The uncalibrated row reproduces the recorded 0.6844 (parity). Precision above recall 0.6 for < 24 px rose from 0.666 to 0.685.

**Test (single run; test-informed second look at arm C, disclosed):** mask AP50 **0.6602**, AP75 0.196, AP50:95 0.294; box AP50 0.6962, box AP75 0.259.
- +0.0117 over arm C's detector-s row (0.6485) and +0.0015 over its s′ row (0.6587).
- Above LACE s0 (0.6556) and the 3-seed LACE (0.655); **below BoxTeacher (0.664) by 0.004**, which is within single-seed noise.

## Pre-registration 6 (2026-11-02, after pre-registration 5's test; BEFORE any C2 number): arm C2 = arm C + IoU-aware score + P2 stack + EMA

Tom: "one and done". The motivation is test-informed (arm C's test and the calibration test are known) and disclosed. Three changes at once, so attribution needs the later leave-one-out runs. ATSS was considered and rejected (Tom: added complexity; FCOS assignment already gives every crown a positive).

**Unchanged from arm C (pre-registration 4):** the inputs (8 px 1024-d, 4 px 128-d, RGB pixel branch), the trunk, P3–P5, the pixel-branch fusion, the level ranges (`LEVELS_C`), FCOS assignment with centre sampling and the P2 fallback, the canopy ignore on every level, the recipe (Adam 1e-3, wd 1e-4, cosine T_max 40, batch 3, 40 epochs max, eval every 5, early stop min 12 / patience 2 / Δ 0.005 on val crowd box AP@0.6 with decode NMS 0.6), seed 0, no augmentation.

**Changed:**
1. **Quality focal loss (GFL, Li et al. 2020), β = 2.** The classification target is the IoU of the predicted box with its assigned GT (detached) for positives, and 0 for negatives (canopy-ignored cells excluded). Loss = BCE(σ(x), q) · |σ(x) − q|^β, summed over the valid cells of all levels, divided by #positives.
   - **The centre-ness branch is removed.** Score s = σ(cls).
   - GIoU box loss on positives, weighted by the detached predicted score σ(cls) (as the official GFL head) and normalised by its sum.
2. **P2 stack:** P2 = lat2(z4) + up2(P3) + proj4(pixel s4), then **3 ×** (3×3 conv 256, GN, ReLU), mirroring the 8 px trunk (arm C had 1 ×).
3. **EMA of weights,** updated every step with decay d_n = min(0.999, (1 + n) / (10 + n)) (timm / TF warm-up form). Val evals, `best.pt` and all predictions use the EMA weights. `last.pt` stores both, for resume.

**Decode:** per level s = σ(cls) > 0.05, top 4000; then NMS over all levels, top 600.

**Selection (val, full LACE pipeline, crowd mask AP50):**
- NMS {0.5, 0.6} × ranking {detector s, s′, per-level calibrated s (pre-registration 5 / 5a method, refitted on the val detections of that NMS)}.
- Ties within 0.001 → NMS 0.5, then detector s > s′ > calibrated (simplest first).
- Known bias: the calibrated row is fitted in-sample on val.

**Test once** at the selected setting (frozen `score_coco`, 2048², maxDets 600, crowd arm).

**Report:** against arm C (0.6485; s′ 0.6587; calibrated 0.6602), LACE (0.6556 s0 / 0.655), BoxTeacher (0.664 / box 0.703), `fpn_s0` (0.6481); val recall by size and the `diag_fp` breakdown; actual Modal bill. Tag `fpnC2_s0`.

## RESULTS, pre-registration 6 (arm C2 `fpnC2_s0`; 2026-11-02)

**Training (Modal A100, 85 s/epoch):** ran the full 40 epochs; best epoch 35. Val crowd box AP@0.6 by epoch: 0.463 / 0.544 / 0.570 / 0.591 / 0.599 / 0.604 / **0.6045** / 0.602 (epochs 5–40). Arm C's best was 0.597.

**Val selection (`select_val_fpnC2_s0.json`):** **NMS 0.5 + s′ selected** (0.6907).

| decode | mask AP50 | AP75 | box AP50 |
|---|---|---|---|
| **NMS 0.5, s′** | **0.6907** | 0.255 | 0.7087 |
| NMS 0.5, detector s | 0.6818 | 0.243 | 0.7009 |
| NMS 0.5, calibrated (in-sample) | 0.6876 | 0.248 | 0.7051 |
| NMS 0.6, s′ | 0.6716 | 0.259 | 0.6890 |
| NMS 0.6, detector s | 0.6705 | 0.247 | 0.6887 |
| NMS 0.6, calibrated | 0.6756 | 0.252 | 0.6924 |

**Val box diagnostics (NMS 0.5, detector s):**
- Recall < 24 px **0.706** (arm C 0.678, BoxTeacher 0.678, `fpn_s0` 0.582).
- Max recall 0.861.
- Precision at recall 0.6: 0.789 (arm C 0.799, BoxTeacher 0.829).

**Test (439 tiles, single run at NMS 0.5; `summary_fpnC2_s0_nms05.json`):**

| ranking | mask AP50 | AP75 | AP50:95 | box AP50 | box AP75 |
|---|---|---|---|---|---|
| **s′ (val-selected)** | **0.6627** | 0.205 | 0.297 | 0.6930 | 0.261 |
| detector s | 0.6482 | 0.194 | 0.288 | 0.6810 | 0.249 |
| calibrated | 0.6581 | 0.197 | 0.293 | 0.6893 | 0.255 |

**Comparison of the headline 0.6627 (single seed):**
- vs LACE: +0.007 over s0 (0.6556) and 3-seed 0.655.
- vs arm C: +0.014 over its detector-s row (0.6485); +0.0025 over calibrated arm C (0.6602).
- vs BoxTeacher: **−0.0013** (0.664), within single-seed noise, so parity rather than a win.
- Box AP50: 0.693, vs BoxTeacher 0.703 (−0.010).

**What did and didn't work:**
- The IoU-aware score alone did not fix the ranking: the detector-s row is 0.6482, level with arm C's 0.6485.
- The masker's s′ ranking fixed it on both val and test.

**Modal cost of C2:** $3.98 (metered $68.96 → $72.94).

### Diagnostic: mask-AP50 levers on the 439 TEST tiles (test-informed, Tom's request; `diag_levers.py`, `results/diag_levers_test.json`)
TIDE-style oracle fixes, each applied alone (they overlap, so they do not sum). Crowd arm, segm, IoU 0.5. Removing every FP equals the perfect-ranking oracle by construction.

| lever (ΔAP50 if fully fixed) | C2 (0.6627) | BoxTeacher (0.6642) |
|---|---|---|
| FN, GT < 24 px | **+0.065** (recall 0.631, 2,317 missed) | +0.059 (0.657, 2,152) |
| FN, 24–48 | +0.033 (0.843) | +0.035 (0.832) |
| FN, 48–96 | +0.020 (0.914) | +0.019 (0.921) |
| FN, ≥ 96 | +0.005 (0.888) | +0.003 (0.942) |
| FP, det < 24 | +0.027 (35.6k) | +0.015 (52.8k) |
| FP, det 24–48 | **+0.047** (26.9k) | +0.052 (50.7k) |
| FP, det 48–96 | +0.027 (21.7k) | +0.041 (28.6k) |
| FP, det ≥ 96 | +0.007 (11.3k) | +0.006 (22.3k) |
| FP background (IoU < 0.1) | +0.082 (66.7k) | +0.116 (139.9k) |
| FP localisation (0.1–0.5) | +0.030 (**20.9k**) | +0.023 (12.1k) |
| FP duplicate / other (≥ 0.5) | +0.006 (**7.8k**) | +0.001 (2.4k) |
| all FN | +0.143 | +0.134 |
| all FP = ranking oracle | +0.159 | +0.168 |

- **Where C2 trails BoxTeacher on test:**
  - Small-crown recall: 0.631 vs 0.657, reversing val, where C2 was 0.706 on box recall; this is mask-level matching on test.
  - Localisation FPs: 20.9k vs 12.1k.
  - Duplicates: 7.8k vs 2.4k.
- **Where C2 is ahead:** background FPs, 66.7k vs 139.9k, and the ≥ 24 px FP cost.
- **Small crowns, box vs mask recall on test** (all detections, IoU 0.5; same test-informed diagnostic). Test BOX recall for GT < 24 px is C2 **0.653** vs BoxTeacher 0.670; MASK recall 0.631 vs 0.657.
  - **The detector's small-crown lead on val did not carry over to test:** box recall was 0.706 vs 0.678 on val and is 0.653 vs 0.670 on test.
  - The masker adds only a small extra loss on small crowns: box → mask −0.022 for C2 vs −0.013 for BoxTeacher.
  - Recall here is ranking-independent: every tile has ≤ 600 detections, so the maxDets cap drops nothing.

## Superseded as the best detector (2026-11-03)
The frozen-DINOv3 detector line (`fpn_s0`, arm C, C2) plateaued at test box AP50 ≈ 0.69. **C2 + LACE masker = 0.6627**, parity with BoxTeacher (0.664). Our own DINO-Swin-L detector trained on OAM-TCD boxes (`../dino_det/README.md`, pre-registration D1) + the same LACE masker reaches **0.7035** test mask AP50 (box 0.739; +0.039 [0.032, 0.049] vs BoxTeacher, paired tile bootstrap). D1 is now a row in `results_439/TABLE1_CURRENT.md`. C2 remains the lightweight frozen-feature reference.
