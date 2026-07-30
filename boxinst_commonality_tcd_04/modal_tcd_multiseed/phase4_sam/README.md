# phase4_sam — SAM 3 vs our commonality masker (same boxes, OAM-TCD 439)

Clean ablation sister of `phase4/`. **Question:** given the *identical* seed-0 detector
boxes, how does **SAM 3** (box-prompted) compare to our latent-commonality EM masker at
turning those boxes into crown instance masks? `rm -rf` this folder undoes it; it writes
only `out/sam3_results*.json` on the shared `tcd04-phase4-vol`.

## Apples-to-apples protocol (what is held fixed)
- **Same boxes.** SAM 3 is prompted with the exact `boxes_2048` from the saved seed-0
  β=0.5-fixed predictions (`out/preds_selfmask_fix_thr025_phase4_L24_s0/preds.json`). The
  detector and its boxes are never re-run — only the box→mask module changes.
- **Same scoring.** `sam3_eval_tcd.py` copies the COCO-101pt mask-AP core (`_greedy_ap`,
  `mask_iou`, `raster`, `_instance_pr`) **verbatim** from `evaluate.py` (parity-guarded by
  `_parity_check`), with the identical GT rasterisation (512px), canopy-ignore rule, and
  operating threshold (`op_thr` from the preds meta).
- **Same AP ranking.** Masks are ranked by **our detector scores** (primary), so the only
  variable is mask shape. A SAM-score-ranked AP is also reported (secondary).
- **No cropping (headline).** Crowns can spill past the tight box, so SAM masks are scored
  **uncropped**. A **box-clipped** variant is reported as a sensitivity check only.

## Model
SAM 3 image model (`build_sam3_image_model(enable_inst_interactivity=True)` — the SAM-1
box/point task head), checkpoint `facebook/sam3` auto-downloaded from HF. SAM 3.1's changes
are video Object-Multiplex, not image-mask quality, so the base SAM 3 image checkpoint is
the right box-prompt predictor. Pinned to `_sam3_repo` commit `46957e4`.

## Run (Modal A100)
```
modal run sam3_modal.py::eval_sam3 --limit 20   # smoke test (~few min)
modal run sam3_modal.py::eval_sam3              # full 439
```
**Prereq:** the `huggingface` secret's token must have accepted the gated `facebook/sam3`
license, else the checkpoint download 403s.

## Compare target (same seed-0 boxes)
| masker | mask mAP50 | mAP50-95 |
|---|---|---|
| ours β=0.5 (geometry-fixed) | **0.620** | 0.244 |
| ours β=0 (geometry-fixed) | 0.579 | 0.200 |
| **SAM 3 (box-prompt)** | _this experiment_ | |

## Files
| file | role |
|---|---|
| `sam3_eval_tcd.py` | verbatim scoring core + SAM box→mask post-proc (`sam_masks_for_tile`) + `evaluate_sam` (uncropped & clipped, det- & SAM-score ranked) |
| `sam3_modal.py` | A100 app: builds SAM 3, reads seed-0 boxes from the volume, RGB tiles via the manifest image_id join, scores, writes `out/sam3_results.json` |
