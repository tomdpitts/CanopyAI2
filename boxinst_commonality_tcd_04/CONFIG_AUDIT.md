# Config audit — every baseline's test-time knobs vs its released source

**Why.** Table 1 claims each baseline runs in "its own native prediction geometry". That claim
was being checked against *our runners' comments*, not against upstream code. This file checks
it against upstream source, fetched and grepped. Started 2026-09-03.

**Rule for reading this file.** "Released" = the value in the authors' own published config or
code, cited by file and line. "Ours" = what our runner actually sets, cited by file and line.
Anything not yet verified against upstream is marked UNVERIFIED and must not be described as
audited.

---

## 1. Restor Mask R-CNN — VERIFIED

Released config: `https://huggingface.co/restor/tcd-mask-rcnn-r50/resolve/main/config.yaml`
(fetched 2026-09-03). Ours: `restor_baseline/restor_modal.py:106-123`.

| Knob | Released | Ours | Verdict |
|---|---|---|---|
| `ROI_HEADS.SCORE_THRESH_TEST` | 0.2 (L201) | 0.05 | deviation, forced by protocol (AP over a list truncated at 0.2 is not AP) |
| `TEST.DETECTIONS_PER_IMAGE` | 512 (L321) | 600 | deviation, forced (= frozen maxDets) |
| `RPN.PRE/POST_NMS_TOPK_TEST` | 512 (L254, L256) | 1000 | deviation, discretionary — see §2 |
| `INPUT.MIN_SIZE_TEST` / `MAX_SIZE_TEST` | 0 / 2048 (L30, L28) | 0 / 2048 | unchanged |
| `RPN.NMS_THRESH` | 0.7 (L252) | untouched | unchanged |
| `ROI_HEADS.NMS_THRESH_TEST` | 0.5 (L197) | untouched | unchanged |
| `ROI_HEADS.NUM_CLASSES` | 2 (L198) | untouched | unchanged |

No undisclosed deviation. The three changes are the three the paper states.

## 2. The `rpn topk 1000` justification — CORRECTED

Note a in the paper said 1000 was chosen "to match Detectree2 RPN config". Checked:

- detectree2's `models/train.py` sets **only** `RPN.BATCH_SIZE_PER_IMAGE` (L1060). It sets no
  test-time proposal budget, no `DETECTIONS_PER_IMAGE`, no `SCORE_THRESH_TEST`.
- `models/predict.py` (98 lines) wraps `DefaultPredictor` and sets nothing.
- So a released-config detectree2 run inherits `Base-RCNN-FPN.yaml`:
  `PRE_NMS_TOPK_TEST 1000`, `POST_NMS_TOPK_TEST 1000`, and detectron2's
  `TEST.DETECTIONS_PER_IMAGE 100`.

**So 1000 is simultaneously detectron2's FPN default and detectree2's effective released
budget. The justification in note a is sound.** What is *not* sound is our own DetecTree2 run —
see §3.

## 3. DetecTree2 — VERIFIED, TWO UNDISCLOSED DEVIATIONS

Released: `PatBall1/detectree2` `models/train.py`, `models/predict.py`, `models/outputs.py`
(fetched 2026-09-03). Ours: `detectree2_baseline/detectree2_modal.py` (train L545-548, val-eval
L623-624, predict L677-678), `detectree2_baseline/stitch.py`, `dt2_recipe.py`.

| Knob | Released | Ours | Verdict |
|---|---|---|---|
| `RPN.PRE_NMS_TOPK_TEST` | 1000 (FPN default; they set nothing) | **2000** | **undisclosed deviation, ours** |
| `RPN.POST_NMS_TOPK_TEST` | 1000 (FPN default) | **1500** | **undisclosed deviation, ours** |
| `TEST.DETECTIONS_PER_IMAGE` | 100 (d2 default) | **500** per subtile | **undisclosed deviation, ours** |
| `RPN.PRE/POST_NMS_TOPK_TRAIN` | 2000 / 1000 (FPN default) | 3000 / 2000 | deviation at TRAIN time, ours |
| `INPUT.MIN_SIZE_TEST` | 800 (d2 default; they set nothing) | 1000 | deviation, disclosed and justified |
| `clean_crowns` iou / containment | 0.7 / 0.85 (`outputs.py:324,328`) | 0.7 / 0.85 | unchanged |
| `clean_crowns` confidence floor | **0.2** (`outputs.py:325`) | 0.05 | deviation, forced by protocol |

Direction of the error: the raised budgets can only *help* DetecTree2, which still scores
0.5969. No published number of ours is inflated by this. But it breaks "released config" and
must be either reverted or disclosed.

`dt2_recipe.py`'s docstring ("line-for-line port of setup_cfg") is accurate for that file; the
deviations were added on top, in the Modal driver.

Documentation error to fix: `PROTOCOL_439.md` says DetecTree2 is "stitched at the protocol floor
0.05, not 0.1". 0.1 is our `stitch.py` default, not theirs — theirs is 0.2.

**Action:** re-run predict at 1000/1000 and `DETECTIONS_PER_IMAGE 100` per subtile, same
checkpoint, and compare. Inference-only. Keep the train-time raise (retraining is not worth the
GPU, and it favours the baseline) but disclose it.

## 4. Box2Mask — VERIFIED

Released: `LiWentomng/BoxInstSeg` `configs/box2mask/box2mask_r50_lsj_8x2_50e_coco.py`,
`mmdet/models/seg_heads/panoptic_fusion_heads/maskformer_fusion_head.py` (fetched 2026-09-03).
Ours: `box_supervised_baselines/boxinstseg_modal.py:381-460`.

| Knob | Released | Ours | Verdict |
|---|---|---|---|
| `num_queries` | 100 (L28) | 100 | unchanged (the 100-instances/subtile cap the paper discloses) |
| `test_cfg.max_per_image` | 100 (L120) | 100 | unchanged |
| `test_cfg.iou_thr` | 0.8 (L121) | 0.8 | unchanged |
| `test_cfg.filter_low_score` | True (L122) | True | unchanged |
| `test_cfg.score_thr` | **absent** | set to 0.05 | **no-op** — the fusion head reads `iou_thr`, `filter_low_score`, `max_per_image` only (L45-46, L133); it never reads `score_thr` |
| test pipeline `img_scale` | (1333, 800) (L158) | (1024, 1024) | deviation, ours, justified (resizing would move the arm off 0.1 m/px) |
| Swin-L batch / LR | config self-inconsistent | 16 / 1e-4 | ours, disclosed in note d |

The 0.05 floor still reaches this row — our stitcher applies `min_score=0.05` (confirmed in
`preds_box2mask_swin-l_s0_test.json` meta) — so results are unaffected. The runner's *stated*
mechanism is wrong and should be corrected.

Provenance defect: `preds_box2mask_swin-l_s0_test.json` carries the model string
"Box2Mask-T R-50" — a copy-paste from the R-50 arm. The runs are distinct (138,820 vs 117,027
dets), only the label is wrong.

## 5. SelvaBox -> SAM 3 (CanopyRS) — VERIFIED, ONE MATERIAL UNDISCLOSED OMISSION

Released: `hugobaudchon/CanopyRS`, presets
`config/pipelines/preset_seg_multi_NQOS_selvamask_SAM3_FT_{fast,quality}.yaml`, dataset class
`OamTcdDataset` in `canopyrs/data/detection/preprocessed_datasets.py:439-457` (fetched
2026-09-03). Ours: `selvabox_baseline/selvabox_modal.py`.

Their benchmark dataset config, which we follow: `ground_resolution 0.1`, `valid_tile_size 1024`,
`test_tile_size 1024`, `tile_level_eval_maxDets 400`. Our geometry claim checks out.

Their deployment presets, which we do not follow, run two aggregators:

| Stage | Released (fast preset) | Ours | Verdict |
|---|---|---|---|
| tilerizer | 1777 px @ 0.5, gr 0.07 | 1024 px @ 0.5, gr 0.1 | deviation, justified: preset would resample OAM-TCD; theirs is the benchmark config |
| aggregator 1 (post-detector) | iou NMS 0.7, `score_threshold 0.4`, `edge_band 0.05` | iou NMS 0.7, floor 0.05, no edge band | NMS matches; floor forced by protocol; edge band omission disclosed and justified (16.0% of GT crowns touch the tile frame) |
| aggregator 2 (post-segmenter) | `ioa-disambiguate`, `score_threshold 0.6`, `nms_threshold 0.05`, score = weighted geometric mean of detector and segmenter scores | **absent** | **undisclosed omission** |

The second aggregator is a scoring and mask-disambiguation stage of their published pipeline.
Omitting it (a) denies them a rerank that fuses SAM's own mask score with the detector score,
and (b) denies them the overlap disambiguation. Both plausibly move their mask AP and their
double-assignment figure in `tab:gtbox`. It is also worth noting for Related Work that their
pipeline already fuses a segmenter score into detection ranking — adjacent to our own
`s' = s p m`.

**Action:** decide whether to add aggregator 2 with the protocol floor substituted for their
0.4/0.6 thresholds, or to disclose the omission in note c. Do not leave it unstated.

## 6. LACE (ours) — not a baseline, but state it for symmetry

Decode floor 0.05 and `topk 600` are the protocol values, so there is no "released config" to
deviate from. Note the asymmetry flagged separately: if the protocol rule is "nothing is
truncated upstream of the scorer", it applies to our own floor too, which binds (minimum
observed detector score is exactly 0.050; 9.3% of detections sit in [0.05, 0.06)).

---

## 7. Stale descriptions of our OWN pipeline — FIXED 2026-09-03

Two files described a superseded LACE configuration and were read as if they described the
published one (by me, in this session — the same failure mode as §3, applied to ourselves).

* `modal_tcd_multiseed/phase4/phase4_lib_tcd.py` docstring said the masker "stays the FIXED
  4096-dim vault model". True of that A/B arm, where the masker is held fixed on purpose so the
  detector comparison is clean. **Not** true of the published pipeline: `phase4_fit_tcd.py`
  refits the masker on the 4-phase L24 cells at 1024-dim, so detector and masker share one
  feature space. Docstring now carries a DEPRECATED banner pointing at `phase4_fit_tcd.py`.
* `artifacts/FROZEN_paper_s0/PROVENANCE.txt` was headed "FROZEN headline artifacts ... mask
  mAP50 0.499" with no indication that this is the old native-4096 result. Header now opens
  with a DEPRECATED banner. Nothing in the repo references this directory, so it is retained
  for reproducibility only.

**Method note.** Both were caught by checking parameter shapes (`em_model.npz`: `mu (4096,)`,
`U (4096,128)` for the deprecated fit; `(1024,256,256)` arrays in `phase4_fit_tcd.py` for the
deployed one), not by reading prose. Verify shapes and released source; never a docstring.

## Open items

- [ ] Re-run DetecTree2 predict at released proposal budget (§3).
- [ ] Decide on CanopyRS aggregator 2 — reproduce or disclose (§5).
- [ ] Fix note a's "to match Detectree2 RPN config": true of their released config, false of our
      DetecTree2 run until §3 is actioned.
- [ ] Fix `PROTOCOL_439.md`'s "not 0.1" -> upstream `clean_crowns` floor is 0.2.
- [ ] Fix the model string in `preds_box2mask_swin-l_s0_test.json`.
- [ ] Fix `boxinstseg_modal.py`'s `test_cfg.score_thr` line or comment it as inert.
- [ ] UNVERIFIED: Restor's training-time config (we only audited inference).
- [ ] UNVERIFIED: SAM 3 zero-shot arm in `tab:gtbox` against the `facebook/sam3` released
      defaults.
- [ ] UNVERIFIED: DeepForest v2.1.0 row in the NEON table.
