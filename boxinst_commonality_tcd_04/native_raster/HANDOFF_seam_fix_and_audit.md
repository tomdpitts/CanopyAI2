# Handoff: subtile-seam fix for the baselines + fairness audit (written 2026-10-13)

For a fresh Claude Code instance. Tom (DPhil, author of the LACE paper) is about to submit. He asked
for **as-shipped, unimpeachable, apples-to-apples baselines**, and found by eye that three baselines are
damaged by **our** stitching. Fix that first; run a full fairness audit in parallel. Do not take the
previous instance's claims on trust: verify against code and files.

Read first: `native_raster/NATIVE_RASTER.md` (the campaign record; the last ~6 sections cover
2026-10-07/08), memory files `project_native_raster_campaign`, `feedback_modal_detach_and_local_pools`,
`project_environment`, `project_latex_build`, and project `CLAUDE.md`.

## 0. Hard rules (each learned the expensive way)

- **Never overwrite** data or results. New files and new volume paths only; `assert not os.path.exists`
  before every write. Back up any figure before regenerating it.
- **Gate every regenerated row**: the new code with the fix switched OFF must reproduce the current
  file byte-for-byte (or the published AP exactly) before any fixed number is read.
- **`paper.tex` may be edited directly** (Tom reviews the git diff). LaTeX prose on one line. Simple,
  clear prose, no LLM tells. Mark anything not yet computed `PENDING`. Build: `tectonic --synctex
  --keep-logs paper.tex` in `ablation/results/` (0 undefined refs/cites required).
- Modal: launch only via `native_raster/detach.py deploy|spawn` (never stop a shell attached to
  `modal run`), per-tile checkpoints + resume, logs on the volume, copy outputs into the repo the same
  session. Local compute: cap BLAS threads (`VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 ...`), check
  `uptime` before pools, kill orphans with `pkill -f "multiprocessing.spawn import spawn_main"`.
- zsh does not word-split: pass multi-word arguments explicitly (bit us on `--gt`).
- Scoring: `score_coco.py --res 2048`; `--score_floor 0` for LACE product files, 0.05 otherwise.

## 1. Current paper state (all at native 2048, LACE = colour-guided render)

`ablation/results/paper.tex` builds clean. Table 1 (439 test tiles, canopy-neutral crowd):

| row | AP50 | AP50:95 | box AP50 | prediction geometry |
|---|---|---|---|---|
| Restor Mask R-CNN rpn1000 | 0.648 | 0.333 | 0.665 | whole 2048 tile |
| LACE 3 seeds | 0.647 ± 0.003 | 0.278 ± 0.005 | 0.667 ± 0.011 | whole 2048 tile |
| Detectree2 s0v2 | 0.614 | 0.301 | 0.639 | 3×3 1024 subtiles, stitched |
| SelvaBox → SAM 3 | 0.571 | 0.211 | 0.732 | 3×3 1024 subtiles, box-NMS, SAM on whole tile |
| Box2Mask Swin-L | 0.563 | 0.231 | 0.602 | 3×3 1024 subtiles, stitched |
| Box2Mask R-50 | 0.433 | 0.157 | 0.497 | same |

Sparse-236: LACE 0.678 ± 0.011 / 0.322; Detectree2 0.625 / 0.346.

## 2. PRIORITY: the seam defect

**Finding** (`native_raster/seam_analysis.py` → `seam_analysis_r2048.json`). A prediction is
"seam-cut" if its mask bbox edge lies on an interior subtile edge (x or y ∈ {512, 1024, 1536}).
Chance rate (whole-tile models) ≈ 0.5%:

| row | seam-cut | AP50 | AP50 with seam-cut preds simply deleted |
|---|---|---|---|
| LACE s0 (control) | 0.5% | 0.6472 | 0.6419 |
| Restor rpn1000 (control) | 0.6% | – | – |
| Detectree2 | 9.7% | 0.6143 | 0.6139 |
| Box2Mask Swin-L | 20.8% | 0.5625 | 0.5643 |
| Box2Mask R-50 | 24.9% | 0.4332 | 0.4329 |
| SelvaBox → SAM 3 | 35.7% | 0.5710 | 0.5878 |

Visible in `ablation/results/figs/qualitative_tcd439.pdf`, tile 125 (Box2Mask panel: straight red cuts).

**Mechanism (Detectree2, Box2Mask).** `native_raster/stitch_res.py` (a res-parametrised copy of
`detectree2_baseline/stitch.py`) does greedy dedup by score: drop a crown if IoU with a kept crown
> 0.7 or if > 0.85 of the *candidate* lies inside it. A crown truncated at one subtile's edge can
outscore the full crown from the overlapping subtile; the fragment is kept, and the full crown is
ALSO kept (only ~half of it overlaps the fragment). Deleting fragments afterwards is the wrong
counterfactual: often the fragment is the only surviving copy. The fix must act **before** dedup.

**Mechanism (SelvaBox).** `native_raster/selvabox_native_modal.py`: detector per subtile →
`raw_dets` (saved per tile, boxes in 2048 coords, with subtile origin `oy, ox`) → box IoU-NMS 0.7 →
SAM 3 prompted on the whole tile (resampled to 1777). Seam damage is truncated **boxes**. CanopyRS's
released preset has an edge-band cull (`edge_band_buffer_percentage = 0.05`), which we disabled
(paper appendix "SelvaBox Edge trim") because on the outer frame it caps recall. Applied only at
**interior** subtile edges it is the correct, as-released behaviour.

**Steps**
1. **Establish the released rules from source, not memory.** Detectree2: read `detectree2` package
   `models/outputs.py` (`stitch_crowns`, `box_filter`, `clean_crowns`); the previous instance
   *believes* `stitch_crowns(folder, shift)` keeps only crowns within each tile's core but did not
   verify it. CanopyRS: the edge-band code. Box2Mask has no released tiling, so use the same rule as
   Detectree2 (same crops/grid/stitcher). Write down exactly what each released pipeline does.
2. **New stitcher (copy; never edit `stitch_res.py` / `detectree2_baseline/stitch.py`)** with a
   pre-dedup filter: drop a subtile prediction whose mask touches an edge of *its own* subtile that
   is interior to the 2048 tile (keep outer-frame edges). Match the released rule's tolerance/buffer.
   **Gate:** filter off ⇒ byte-identical to `native_raster/preds_{dt2_s0v2,box2mask_r50_s0,
   box2mask_swinl_s0}_r2048.json`.
3. **Re-stitch locally** from raw subtile predictions: `/Users/tompitts/dphil/feat_cache/dt2_raw/s0v2`,
   `feat_cache/b2m_raw/{r50,swinl}` (check which dirs are the 2048 inputs; `native_raster/
   run_stitch_b2m.py` shows the Box2Mask invocation, min_score 0.05). Detectree2 sparse-236 raw:
   `tcd-detectree2-vol:out/pred_raw_sparse_s0v2` (2,124 files; pull if not local). Dense Detectree2
   tiles are slow (~3,000 raw crowns); use the bbox-window AND already in `stitch_res.dedup`.
4. **SelvaBox:** from each tile's `raw_dets` on `tcd04-baselines-vol:out/native_raster/
   selvabox_bench_r2048/tiles/<tid>.json`, drop interior-edge boxes per the CanopyRS rule, redo the
   0.7 NMS, then re-run SAM 3 **only for boxes not already prompted** (SAM masks per box should be
   independent given the image embedding — verify on one tile that re-prompting an unchanged box
   reproduces its stored mask before reusing). New Modal output dir; A100 via detach; full rerun is
   ~6 A100-h if reuse is not valid.
5. **Rescore** (2048, maxDets 600; plus 1,200 for Detectree2 and 1,600 for SelvaBox as in the
   detection-budget paragraph). Rerun `seam_analysis.py` on the new files: seam-cut share should
   fall to ~chance.
6. **Paper updates:** Table 1 (Detectree2, SelvaBox, Box2Mask ×2 + note c own-budget numbers + note d
   Box2Mask cap delta via `native_raster/b2m_cap_subset.py`), tab:sparse236 + tab:scorer Detectree2
   rows, abstract "0.625", Results text (Box2Mask 0.563), detection-budget paragraph, empty-tiles
   percentages (appendix), tab:geometry (state the seam rule per row), "SelvaBox Edge trim"
   paragraph (now: cull applied at interior edges only), tab:strata-maskiou
   (`native_raster/ablation_r2048/box_iou_strata_res_guided.py`, point at the new files),
   qualitative figures (`figs/make_qualitative.py --set tcd439|sparse`; back up first; delete the
   baseline entries from `qualitative_thresholds_*.json` so their best-F1 thresholds recompute;
   LACE panels must stay F1 ≥ 0.8 — Tom's rule).
7. Record in `NATIVE_RASTER.md` + memory.

## 3. In parallel: fairness audit (Tom opted in; use subagents/workflow)

Goal: find anything else that hindered a baseline or oversold LACE. Each check reads code + files and
reports evidence, not opinion. Suggested split:

- **Box2Mask (R-50, Swin-L):** config vs released defaults, train AND test resize/scales
  (`box_supervised_baselines/boxinstseg_modal.py` ~L338–448: test is MultiScaleFlipAug
  `img_scale=(1024,1024)`, keep_ratio — i.e. no test downscale; **check the train `scales` list**),
  num_queries=100 cap, checkpoint selection, score floor (0.05 confirmed in the 2048 file).
- **Detectree2:** recipe vs its released one, `INPUT.MIN_SIZE_TEST=1000`, stitch floor (**verified
  0.05**: `preds_dt2_s0v2*.json` meta and min score), clean_crowns thresholds vs released.
- **SelvaBox → SAM 3:** subtile size/overlap/GSD vs CanopyRS OAM-TCD benchmark preset, NMS, SAM tile
  size (1777, their wrapper default), maxDets, edge-band.
- **Restor Mask R-CNN:** released checkpoint, deviations (score floor, dets/image, RPN topk 1000),
  tree-only filter. **Tom questions the rpn2000 number (0.685):** previous instance says it is the
  same checkpoint and pipeline with only `RPN.PRE/POST_NMS_TOPK_TEST=2000` (`restor_native_modal.py
  --rpn-topk 2000`), tree-filter byte-identical to `restor_baseline/preds_restor_rpn2000.json`
  (71,606 dets, 163/tile, cap 600 not binding). Verify independently: identical config otherwise,
  same scorer/floor/tree filter, where the original `preds_restor_rpn2000.json` came from, and
  whether anything else differs between the rpn1000 and rpn2000 runs (NATIVE_RASTER notes the rpn1000
  files predate the rpn-parametrised script). Report whether it is a like-for-like comparison.
- **LACE selection history / leakage:** what was chosen on val vs test, and when (the 2048 switch
  and the guided render were motivated by test diagnostics; render settings were val-selected and
  pre-registered — NATIVE_RASTER "PRE-REGISTRATION"). Check no test data entered any selection.
- **Paper claims vs source JSON:** every number in `paper.tex` traced to a scored file; flag mismatches.
- **Disclosure items to put to Tom (not to change unilaterally):** abstract reports AP50 only (LACE
  trails at AP50:95); qualitative figures selected with an F1 ≥ 0.8 floor for LACE only (captions
  do not say so); chronology of the 2048/guided decisions.

Tom's position (do not re-litigate): LACE is the proposed method, so its components are tuned; the
baselines are representative and need not be optimised by us, **but must not be hindered by us**.
Anything we did that departs from a baseline's released behaviour to its detriment is a defect.

## 4. Key files

| what | where |
|---|---|
| campaign record | `native_raster/NATIVE_RASTER.md` |
| seam analysis | `native_raster/seam_analysis.py`, `seam_analysis_r2048.json` |
| stitcher (current) | `native_raster/stitch_res.py`, `native_raster/run_stitch_b2m.py`, `detectree2_baseline/stitch.py` |
| SelvaBox app | `native_raster/selvabox_native_modal.py` (tiles on `tcd04-baselines-vol`) |
| Restor app | `native_raster/restor_native_modal.py` |
| Box2Mask train/predict | `box_supervised_baselines/boxinstseg_modal.py`, `PLAN.md` |
| Detectree2 | `detectree2_baseline/detectree2_modal.py`, `dt2_recipe.py` |
| scorer | `boxinst_commonality_tcd_04/score_coco.py` |
| current 2048 preds | `native_raster/preds_*_r2048*.json`, scored `scored_*_r2048*.json` |
| paper | `ablation/results/paper.tex`, `lace.bib`, figures `ablation/results/figs/` |
| Overleaf bundle (stale after fix) | `ablation/results/LACE_overleaf_2026-10-08.zip` |
