# OAM-TCD 439 re-evaluation protocol (Table 1)

## Results

Every row was produced and scored **by us** under one frozen protocol — no figure here is
quoted from another paper. `pycocotools COCOeval` · 439 whole 2048² tiles · masks at 512² ·
maxDets 600 · score floor 0.05 · `iouThrs = linspace(0.5, 0.95, 10)` · canopy as `iscrowd`.
**mask IoU** is the masker isolated: mean per-crown mask IoU on the 9,823 crowns EVERY row we
still hold predictions for matched, at matched detection quality (box IoU ≥ 0.5), from
`results_439/box_iou_strata.json`. It carries NO recall term and is not a system metric — see §5.
CN = canopy-neutral: canopy polygons are `iscrowd` ground truth, so a prediction landing
in unlabelled canopy is ignored rather than scored. It is the only arm reported.

| Method | Superv. | Imgs | CN AP50 | CN AP75 | CN AP50:95 | Box AP50 | mask IoU | dets |
|---|---|---|---|---|---|---|---|---|
| LACE (ours), seed 0 | box | 900 | 0.6250 | 0.1433 | 0.2578 | 0.6211 | -- | 159,687 |
| LACE (ours), seed 1 | box | 900 | 0.6358 | 0.1376 | 0.2575 | 0.5835 | -- | 162,017 |
| LACE (ours), seed 2 | box | 900 | 0.6293 | 0.1417 | 0.2567 | 0.6230 | -- | 177,598 |
| _LACE 3-seed mean ± sd_ | box | 900 | 0.6300 ±0.0054 | 0.1409 ±0.0029 | 0.2573 ±0.0006 | 0.6092 ±0.0223 | -- | 166,434 |
| **LACE + posterior product, 3-seed** | box | 900 | **0.6625 ±0.0012** | 0.1649 ±0.0077 | **0.2790 ±0.0035** | 0.6396 ±0.0240 | 0.7430 ±0.0013 | 166,434 |
| LACE + posterior product, seed 0 | box | 900 | 0.6615 | 0.1718 | 0.2821 | 0.6535 | 0.7445 | 159,687 |
| LACE + posterior product, seed 1 | box | 900 | 0.6639 | 0.1566 | 0.2752 | 0.6119 | -- | 162,017 |
| LACE + posterior product, seed 2 | box | 900 | 0.6622 | 0.1662 | 0.2798 | 0.6533 | -- | 177,598 |
| LACE + 8-feat logreg *(superseded)* | box | 900 | 0.6616 | 0.1648 | 0.2795 | 0.6525 | -- | 159,687 |
| Restor Mask R-CNN, shipped cfg (rpn 512) | masks | 4169 | 0.5706 | 0.1806 | 0.2575 | 0.5868 | -- | 42,987 |
| **Restor Mask R-CNN, rpn 1000** | masks | 4169 | 0.6255 | 0.1872 | 0.2766 | 0.6536 | 0.7655 | 56,438 |
| Restor, shipped cfg, classes pooled | masks | 4169 | 0.5620 | 0.1811 | 0.2555 | 0.5825 | -- | 61,182 |
| Restor, rpn 1000, classes pooled | masks | 4169 | 0.6137 | 0.1893 | 0.2738 | 0.6392 | -- | 79,201 |
| **DetecTree2 (fine-tuned, converged)** | masks | 900 | 0.5969 | 0.1748 | 0.2578 | 0.5913 | 0.7595 | 196,927 |
| **SelvaBox → SAM 3 (both FT)** | box+SAM | 3024 | 0.5687 | 0.0746 | 0.2028 | 0.7297 | 0.6960 | 491,847 |
| SelvaBox → SAM 3, maxDets 1600 | box+SAM | 3024 | 0.5767 | 0.0754 | 0.2050 | 0.7480 | -- | 491,847 |
| **Box2Mask Swin-L (box-supervised)** | box | 900 | 0.5396 | 0.0884 | 0.1980 | 0.5821 | 0.6943 | 138,820 |
| Box2Mask R-50 (backbone sensitivity) | box | 900 | 0.3865 | 0.0509 | 0.1299 | 0.4707 | 0.6436 | 117,027 |

**Imgs, SelvaBox rows.** `3024`, not the earlier estimate `~3335` (which was 4169 × 4/5).
Source: SelvaBox (arXiv:2507.00170v1) Appendix E.1 + Table 19 — "OAM-TCD provides five predefined
cross-validation folds; we train on folds 0–3 and use fold 4 exclusively for validation", and
Table 19 gives `# Train Images = 3024` at 2048 px for OAM-TCD, after their zero-annotation and
pixel-cover tile removal. Tree-group (canopy) pixels are masked out of their training data. The
detector additionally trains on NeonTreeEvaluation (912), QuebecTrees (148) and SelvaBox (585)
tiles, which this column does not count.

### What the table says

1. **LACE leads on mask AP50 at the defensible operating point** — **0.6625 ± 0.0012**
   (3 seeds) vs Restor's 0.6255 at detectron2's default proposal budget, from 900
   box-labelled images against Restor's 4,169 with full crown polygons. Even the weakest
   seed (0.6615) clears it by +0.036, thirty times the seed sd. The margin comes entirely
   from the confidence fix: the un-reranked baseline (0.6300 ± 0.0054) ties Restor.
2. **The lead is contingent on Restor's proposal budget.** Restor ships `rpn topk 512`, half
   the framework default, which starves it (98 dets/tile, recall 0.6442). At 1000 it reaches
   0.6255; at 2× default it reaches 0.6605 and the lead vanishes. §2 explains why 1000 is the
   published point. Anyone citing Restor's published 0.432 is citing an under-configured model.
3. **SelvaBox is the best detector and the worst masker.** Box AP50 0.7297 beats every other
   row by ~0.076, but mask AP75 0.0746 is under half everyone else's — DINO-Swin-L localises
   well, SAM 3 fine-tuned at 1.3–3.5 cm/px delineates poorly at 10 cm/px.
4. **Restor holds strict-IoU mask quality** — CN AP75 0.1872 vs LACE's 0.1718. Mask
   supervision at pixel resolution buys boundary precision that reranking cannot manufacture.
5. **The confidence fix is banded and generalises.** +0.0325 CN AP50 on all three seeds
   (439), and **+0.0354 on the held-out sparse 236** — a slice sharing no tile with the 900
   training images, where the gain is *larger* than on the test set. It also cuts seed
   variance 4.5× (sd 0.0054 → 0.0012). Caveat that stands: roughly two thirds of the gain is
   a box-size prior, so the masker-specific increment is ≈ +0.012 (`confidence/README.md`).
6. **The box-supervised family is far behind, and it is not a training artefact.** Box2Mask
   (TPAMI 2024), the strongest published box-supervised segmenter and the only row from LACE's
   own supervision class, reaches 0.3865 CN AP50 against LACE's 0.6625 on identical crops, grid
   and scorer. It was given a 9-epoch budget under LACE's own early-stopping rule, stopped
   itself at epoch 6 and selected epoch 4, with four later evaluations below the peak while
   train loss kept falling — its validation optimum, not a truncation. The gap is recall
   (14,293 crowns matched vs DetecTree2's 19,603) and mask quality (0.6436, last at every
   stratum). Full account: `box_supervised_baselines/PLAN.md` §5b–5c.
7. **mask IoU is the masker isolated, and it reorders the table.** BOTH mask-supervised models
   beat LACE's masker on it — Restor **0.7655** and DetecTree2 **0.7595** against LACE's
   **0.7430 ± 0.0013** — while DetecTree2 trails LACE by 0.066 on CN AP50 and Restor by 0.037.
   LACE's system lead therefore rests on detection and ranking, NOT on mask quality; state it
   that way. (Before the 2026-09-03 DetecTree2 retrain this read as an "essential tie" at
   0.7368 vs 0.7378 — the converged model overturns it.) The one place the masker leads
   outright is on GT prompt boxes (`tab:gtbox`), a different question.
8. **Sparse 236 (`results_439/sparse236/`)**: LACE + product **0.6913 ± 0.0095** vs the converged
   DetecTree2 **0.6118** — **+0.080** CN AP50, and ahead on every other metric too. Worth stating
   next to item 7: on the 439 the converged DetecTree2 beats LACE at AP75 (0.1748 vs 0.1649), but
   on this open-canopy slice LACE still leads it (0.2326 vs 0.2155). The masker's strict-IoU
   disadvantage against mask supervision is a dense-canopy effect, absent in the regime the paper
   is about — and the margin is larger here (+0.080) than on the 439 (+0.066).

Full per-row artifacts: `results_439/` (`build_table.py` regenerates `table_439.md`).
The `rpn 2000` sensitivity row is omitted here; see §2.


One scorer, one protocol. Every row re-scored by us **except** SelvaBox → SAM 3 (frozen SAM),
which stays a cited figure under their protocol and is marked with an asterisk — or dropped.
Footnotes a and b in `ablation/results/paper.tex` collapse to "all rows re-scored by us under
one protocol", plus that single asterisk.

## 0b. The mask-IoU column — what it is and what it is not

`box_iou_strata.py` → `results_439/box_iou_strata.json`. Every prediction is paired with the GT
crown it matches on **box** IoU (greedy by score, threshold 0.5) and its mask IoU is recorded as
the DEPENDENT variable, so mask quality plays no part in choosing the pairing. Stratifying on the
DETECTION box, never the tight box of the predicted mask, is deliberate: a mask-derived box is
partly determined by the mask, so an under-segmenting masker would be credited with a worse box,
which is backwards. Canopy-landing predictions are dropped first by the project's >50% rule.

Read it with three caveats:
1. **No recall term.** It is mask quality GIVEN a detection at box IoU ≥ 0.5. A method that finds
   few crowns is not charged for the ones it missed. AP is the system metric; this is not.
2. **The value depends on the row set.** It is computed on the intersection — the 10,020 crowns
   EVERY listed row matched — because the rows otherwise match different crowns and a
   higher-recall method would be charged for the extra hard ones only it found. Add or remove a
   row and every number moves. Recompute the whole column, never one cell.
3. **All three LACE seeds are now included** (0.7430 ± 0.0013). Seeds 1–2 were briefly believed
   lost to a session scratchpad; they were recovered from `tcd04-phase4-vol` on 2026-09-02 and
   re-verified to reproduce their published AP and detection counts exactly. See
   `confidence/PREDICTIONS.md` and `results_439/ARTIFACTS.md`.

Recomputed WHOLE 2026-09-03 after the DetecTree2 retrain, per caveat 2: every cell in the column
moved because the common subset shrank from 11,238 to 10,020 crowns. Per-bin counts sum to the
common-subset size for every row; the top stratum (box IoU ≥ 0.9, i.e. near-GT boxes) sits slightly
above the independent GT-box bake-off for the comparable arms, the expected direction since
near-perfectly localised crowns are the easier ones. DetecTree2's own stratum profile rises
monotonically 0.6163 → 0.8654 across the five box-IoU bins on 10,020 matched crowns.

## 1. Scorer (frozen)

| knob | value | justification |
|---|---|---|
| implementation | `pycocotools` / `faster_coco_eval` **`COCOeval`**, `iouType='segm'` | standard implementation; removes the "hand-rolled AP core" objection outright. Parity with the old `evaluate._greedy_ap` already established: AP50 identical to 4 dp, AP50:95 +0.0008 (float knife-edge, `np.arange` vs `np.linspace`) — `ablation/results/cocoeval_parity.json` |
| scoring unit | 439 whole 2048² tiles | the benchmark's native unit |
| mask raster | **512²**, RLE throughout | DECIDED: keep 512². Measured cost of the 4× downsample: +0.005 IoU on average and +2.9 pts of TP rate at IoU 0.75, and it flatters the COARSEST masker — ours (LACE's 8 px grid vs DetecTree2's 28×28 ROI head, finer than the 512 raster for 90% of crowns). AP50 is unaffected (~0.8 pts). Re-basing to 2048² is blocked by `phase4_lib_tcd.eval_selfmask`, which accumulates dense masks for all 439 tiles (~49 GB at 512² in a 64 GB container; 2048² would need ~780 GB). State the limitation in the paper rather than hide it |
| maxDets | **600** | densest test tile holds 450 GT instances (421 trees + 29 canopy); only 3 of 439 tiles exceed 400. 600 is ~1.3× the densest tile and ~10× the median (42), i.e. comfortably past saturation |
| score floor | 0.05, **applied to the detector score** | COCO convention; no aggregator/deployment floors (0.2 / 0.4 / 0.5 / 0.6) anywhere in the path. The floor decides which detections EXIST; how the survivors are RANKED is a separate question, and AP is invariant to any monotone rescaling of the ranking key. So a reranked row is floored on the detector score `s` and then ranked by the reranked key — not floored on the reranked key. This matters only for the posterior-product rows: `s · bimod · pfg_mean` is a product of three sub-1 numbers, so flooring it directly would discard ~half of the 159,687 detections purely for living on a smaller scale (measured: CN AP50 0.6615 → 0.6461). Here `s ≥ 0.05` for all 159,687 detections, so the floor binds on nobody and the published figures are unaffected. Note the reranked artifacts (`results_439/rerank_product_*.json`) record `score_floor: 0.0` for exactly this reason — the filtering had already happened upstream |
| canopy arms | **canopy-neutral only**, every metric, every row | The canopy=FP arm was dropped 2026-08-28. It charges a method for finding crowns the annotators chose to group rather than delineate, so its cost scales with detection volume: LACE emits 364 dets/tile against Restor's 129 and is penalised for the recall that the CN arm rewards. It is also not comparable across rows — Restor is 2-class, so scoring its tree class alone lets a learned canopy classifier pre-filter its own FPs (pooled, its canopy-landing rate is 42.2% against LACE's 42.5%: it is not avoiding canopy, only labelling it), while CanopyRS never saw canopy in training at all. `score_coco.py` still emits the arm and the per-row artifacts retain it; it is simply not reported |
| canopy-neutral | canopy polygons as `iscrowd=1` GT | COCO's crowd rule (intersection-over-detection-area > t) equals our >50%-in-canopy rule **exactly at t = 0.50**, so AP50 transfers unchanged. Above t = 0.50 COCO is stricter — canopy-neutral **AP50:95 will move**. Verify numerically during the run |

## 2. Prediction geometry — each model's native configuration

No model is forced out of distribution; only the measurement is held fixed.

| row | geometry | notes |
|---|---|---|
| LACE (ours), 3 seeds | whole 2048² | decoder capped at `topk 600` — see §4 |
| Restor Mask R-CNN | whole 2048² | their released test config already does this (`MIN_SIZE_TEST: 0`, `MAX_SIZE_TEST: 2048`). Align `SCORE_THRESH_TEST` 0.2 → 0.05, `DETECTIONS_PER_IMAGE` 512 → 600, **`RPN.PRE/POST_NMS_TOPK_TEST` 512 → 1000** (see below). `NUM_CLASSES: 2` — emit all classes and choose at scoring time |
| DetecTree2 (fine-tuned, **retrained 2026-09-03**) | 1024² @ 0.5 overlap, 3×3 = 9 subtiles → stitch, `INPUT.MIN_SIZE_TEST` 1000 | its native operating resolution. `MIN_SIZE_TEST` is set to detectree2's own `MIN_SIZE_TRAIN` of 1000 rather than left at detectron2's inherited COCO default of 800, which downscaled our 1024 subtiles at test but not at train — an artefact of OUR subtiling, since detectree2's own tiles are smaller than 800 and the same default UPscales them. Stitched at the protocol floor 0.05. See `detectree2_baseline/RETRAIN_V2.md` |
| SelvaBox → SAM 3 (both FT) — **RUN 2026-08-28** | 1024² @ 0.5 overlap at native 0.1 m/px → stitch | their **OAM-TCD benchmark** config (`OamTcdDataset`: `ground_resolution 0.1`, `test_tile_size 1024`, overlap 0.5) — identical grid to the DetecTree2 arm. **Not** the shipped 1777 px @ 0.045 m/px deployment preset, which is their dense-canopy UAV setting and would resample OAM-TCD 2.22× up at 64 passes/tile |
| SelvaBox → SAM 3 (frozen SAM) | — **not re-run** | their published figure, cited with an asterisk, or omitted. The repo ships no frozen-SAM3 preset, so which detector they paired with frozen SAM is not recoverable from source; re-running it would mean guessing their configuration |

### Restor's proposal budget — DECIDED: rpn topk 1000

Their config ships `RPN.PRE/POST_NMS_TOPK_TEST: 512`, **half** detectron2's FPN default of 1000,
against tiles holding up to 422 GT crowns (450 instances including canopy groups). Mask R-CNN can only detect what its RPN proposes, so
this is a third truncation knob of exactly the same kind as `SCORE_THRESH_TEST`, which we already
raise. Measured effect — it is proposal starvation, not rescoring (max score identical at 0.9964;
detections ≥0.5 barely move; the new detections are all low-confidence tail):

| rpn topk | dets/tile | recall@0.5 | CN AP50 |
|---|---|---|---|
| 512 (shipped) | 97.9 | 0.6442 | 0.5706 |
| **1000 (detectron2 default)** | **128.6** | 0.7227 | **0.6255** |
| 2000 (2× default) | 163.1 | 0.7804 | 0.6605 |

dets/tile are `n_det`/439 from `results_439/restor_{coco512,rpn1000_tree,rpn2000}.json`:
42,987 / 56,438 / 71,606. (Corrected 2026-09-03: the 1000 row previously read 163, which is
the 2000 figure. Recall for the 1000 row recovered 2026-09-22 from the stored predictions via `ablation/scripts/t_restor_rpn_recall.py`, gated on the 512/2000 values reproducing exactly; see `ablation/results/restor_rpn_recall.json`.)

**Published operating point = 1000**: the framework default, chosen by neither party. 512 is
Restor's deliberate reduction below it; 2000 is above it and was picked arbitrarily by us. Report
512 and 2000 as documented sensitivities, never as the headline. Note LACE has no analogous
recoverable cap — raising its decode `topk` does nothing because the 0.05 score floor binds first
on 74% of tiles, and the capped 26% are already at the scorer's maxDets 600.

**Stitching**: one stitcher, merge rule per method — `clean_crowns` (IoU > 0.7 ∨ containment > 0.85)
for DetecTree2, IoU-NMS 0.7 for SelvaBox.

**No edge-band cull.** CanopyRS's `edge_band_buffer_percentage: 0.05` drops any polygon not
*fully* inside a subtile shrunk by 51.2 px. Interior seams tolerate it (consecutive safe
intervals overlap by 921.6 − 512 = 409.6 px; only 6 of 25,705 crowns exceed that width), but
the shrink also applies to the outer frame of the 2048² tile, where no neighbouring subtile can
recover it: **4,108 of 25,705 GT crowns (16.0%) touch that frame**, capping recall at 0.84
before the model predicts anything. It is also absent from the path that produced their
published numbers — their tile-level AP is computed on the model component's raw COCO output,
pre-aggregator; the aggregator only feeds their raster-level (geo, F1) metrics.

## 3. What this replaces

The SelvaMask protocol our current row-b numbers (0.185 / 0.122) come from uses the same 439
source images but a materially easier test set: canopy annotations deleted and their pixels
blacked out, subtiles > 80% black or all-white dropped (**2,527 of 3,951 survive**), GT retained
at ≥ 40% tile overlap, scored per 1024² subtile with maxDets 400 — an effective budget of 1,600
per 2048²-equivalent, 2.7× ours.

Two corrections to the current footnote b: their tile-level AP is **pooled** COCOeval over the
merged subtile COCO, not per-tile averaged; and the aggregator's 0.5/0.6 score floors never
touch it.

## 4. Asymmetries to state, not hide

1. **Detection budget.** LACE is hard-capped at 600 proposals by its decoder (`--topk 600`;
   observed max is exactly 600 on 9–16 of 236 tiles). Baselines are free to fill the budget.
   The cap can only understate LACE. **Re-measured 2026-09-03 on the converged DetecTree2**,
   where it now binds much harder: at the 0.05 stitch floor that row emits 448.6 dets/tile
   (median 401, max 1511) and maxDets 600 discards **17.7% of its detections on 30.5% of
   tiles**, against 7.0% before. Scoring it at maxDets 1200 instead of 600 gives mask AP50
   **0.5969 → 0.6012 (+0.0043)**, AP75 +0.0001, AP50:95 +0.0010, box AP50 +0.0053. So the cap
   costs DetecTree2 ≈0.004 AP50 — 43× the previously measured 0.0001, and it should be quoted
   at the new value — but it remains an order of magnitude below LACE's +0.066 lead and cannot
   account for it. `results_439/dt2_s0v2_coco512.json` is the maxDets-600 row; the 1200 arm is
   a sensitivity, not a published figure.
2. **Seeds.** LACE has 3; Restor and SelvaBox are single released checkpoints with no seed
   variation available. **DetecTree2 and Box2Mask are different** — we train those ourselves, so
   seeds ARE available and we simply did not buy them (~$18 of A100 for three DetecTree2 seeds).
   Say that plainly rather than extending the released-checkpoint defence to cover them. Report
   LACE's band *and* its worst seed so the comparison holds pessimistically.
3. **Canopy-ignore is genuinely free.** An ignored detection is removed from both TP and FP
   counts, so a prediction landing > 50% inside unlabelled canopy costs nothing. Relevant to the
   SelvaBox row, whose model never saw canopy in training (CanopyRS deletes canopy annotations
   and blacks the pixels) and will fire there freely. Canopy-neutral is the only arm reported,
   which is what makes that row comparable at all.
4. **SAM 3 FT domain gap.** `sam3-multi-selvabox-selvamask-FT` was fine-tuned on three SelvaMask
   UAV sites (dense tropical canopy, 1.3–3.5 cm/px) and is applied here at 10 cm/px. Context for
   the discussion, not a thumb on the scale.

## 5. Open items

- [x] **FT SAM 3 load — RESOLVED 2026-08-27.** The checkpoint is a
      `transformers.Sam3TrackerModel` state dict: **685/685 tensors match exactly, zero missing,
      zero unexpected** (verified locally on CPU against `Sam3TrackerConfig.from_pretrained`).
      So the SelvaBox row must be built on **HF transformers**, NOT the
      `facebookresearch/sam3` package that `phase4_sam/sam3_modal.py` pins at `46957e4` —
      their key namespaces are disjoint (`vision_encoder.*` vs
      `detector.backbone.vision_backbone.trunk.*`).

      **Trap to guard against:** CanopyRS's own loader
      (`canopyrs/engine/models/segmenter/sam3.py:93`) calls
      `load_state_dict(..., strict=False)` and then prints "✓ Fine-tuned weights loaded
      successfully!" *unconditionally* — it would report success having loaded nothing. Our
      runner must assert the matched-key count, not trust the message.

      Their wrapper also carries `target_tile_size` (default **1777**), resizing each tile
      before SAM sees it, described in their code as a fairness knob "to match typical
      Detectree2 input". The pipeline presets do not set it, so their published numbers use
      1777. Decide explicitly whether to feed native 1024 subtiles or reproduce their 1777
      resize — and state which.
- [x] **maxDets for SelvaBox — RESOLVED 2026-08-28.** It emits **1,120 dets/tile** after
      NMS, so the 600 cap discards ~47% of them — by far the hardest the cap binds on any row.
      Measured anyway: scoring at their own budget (1,600 per 2048²-equivalent) moves
      canopy-neutral mask AP50 only **0.5687 → 0.5767 (+0.008)**. The cap costs them
      essentially nothing and 600 stands for every row without a fairness asterisk. The
      concern that maxDets is applied *before* canopy-ignore — and so would waste the budget
      on the ~83% of their detections that fall in canopy — turned out not to bite.
- [ ] **Confirm the 2048² raster switch** — it will move every AP50:95 in the table, including
      figures already published in the ablation sections.
- [ ] `ablation/results/cocoeval_parity.json`'s `verdict` field still carries the
      matching-fallback explanation that `ablation/scripts/t13_cocoeval_parity.py` retracted on
      2026-08-26. Stale artifact, not a wrong number — regenerate.

## 5b. SelvaBox result (run 2026-08-28)

439/439 tiles, 491,847 crowns, their benchmark tiling, IoU-NMS 0.7, no edge-band cull,
SAM 3 FT at their `target_tile_size` 1777. Canopy-neutral arm:

| | mask AP50 | mask AP75 | mask AP50:95 | **box AP50** |
|---|---|---|---|---|
| SelvaBox → SAM 3, maxDets 600 | 0.5687 | 0.0746 | 0.2028 | **0.7297** |
| SelvaBox → SAM 3, maxDets 1600 | 0.5767 | 0.0754 | 0.2050 | 0.7480 |

**Best detector in the table, worst masker.** Box AP50 0.7297 beats Restor (0.6536) and LACE
(0.6535) by ~0.076 — the highest of any row. But mask AP75 is 0.0746, under half everyone
else's, and its box→mask step loses 0.16 AP50 (0.73 → 0.57) where LACE gains slightly and
Restor loses 0.03. Consistent with the components: DINO-Swin-L trained across four crown
datasets localises well; SAM 3 fine-tuned on 1.3–3.5 cm/px tropical UAV imagery delineates
poorly at 10 cm/px.

**Sanity check against their own publication.** They report 0.185 AP50:95 for both-FT on
OAM-TCD; we measure 0.2050 at their maxDets on the full 439 with canopy-ignore. Same
ballpark, slightly in their favour under our protocol — evidence the pipeline was reproduced
faithfully rather than crippled.

**Why the canopy=FP arm is not reported for this row** (or any other; see §1). CanopyRS
trains with canopy deleted and blacked out, so the model detects individual crowns inside
OAM-TCD's canopy *group* polygons: 83% of its predictions centre inside canopy covering 25%
of the tile, while it still recovered 6/7 labelled crowns at IoU ≥ 0.5 on the probe tile.
That arm would measure the annotation mismatch, not the model.

## 6. Local checkpoints

All in the HF cache (`~/.cache/huggingface/hub`), 13 GB:

| repo | file | needed |
|---|---|---|
| `restor/tcd-mask-rcnn-r50` | `model.pth` + `config.yaml` (Detectron2) | yes |
| `CanopyRS/dino-swin-l-384-multi-NQOS-selvamask-FT` | `model_best.pth` + detector YAML | yes — the detector for the both-FT row |
| `CanopyRS/sam3-multi-selvabox-selvamask-FT` | `sam3_selvamask_ft_model_best.pt` (1,833,323,691 B, 685 tensors) | yes |
| `CanopyRS/dino-swin-l-384-multi-NQOS` | `model_best.pth` + tilerizer/detector/aggregator YAMLs | surplus once frozen SAM is dropped (2.6 GB) — YAMLs still useful as reference |
| `facebook/sam3` | `sam3.pt`, `model.safetensors` | surplus once frozen SAM is dropped (6.9 GB) |
