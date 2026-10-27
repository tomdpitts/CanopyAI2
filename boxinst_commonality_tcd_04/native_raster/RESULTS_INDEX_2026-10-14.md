# Results index — seam fix, fairness audit, SAM placement, Tier A (2026-10-14)

<!-- TABLE1_CURRENT:begin (auto, results_439/mark_superseded.py) -->
> ⚠️ **SUPERSEDED 439 / sparse236 NUMBERS (marked 2026-10-27).** Current headline table: [`TABLE1_CURRENT.md`](../results_439/TABLE1_CURRENT.md). Test-set numbers in this file are a historical experiment record (512 raster, pre-seam-fix baselines, 8 px masker or earlier models) and must not be quoted unless they match that table.
<!-- TABLE1_CURRENT:end -->

For an agent analysing these results. **Read-only: never overwrite a file here; write new files.**
All paths are relative to `boxinst_commonality_tcd_04/native_raster/` unless absolute. Python env:
`/Users/tompitts/dphil/CanopyAI2/.venv/bin/python`, run from the repo root with `PYTHONPATH=.`.
Narrative record: `NATIVE_RASTER.md` (sections "SEAM FIX" onward) and `audit_2026-10-14/README.md`
(sections A–E; E holds the latest verified results and supersedes earlier sample numbers).

## Status of each result (as of writing)
| result | status |
|---|---|
| seam-fixed Detectree2 / Box2Mask rows | final; filter-off gates byte-identical |
| seam-fixed SelvaBox → SAM 3 (0.598), per-subtile SAM (0.712 / 0.724) | provisional: 9% / 100% of SAM masks made on local MPS; A100 re-run pending (Modal paused) |
| A4: masker comparison on GT boxes, full 366 tiles | final for FT SAM (parity exact); zero-shot SAM parity approximate (HF vs `sam3` package, +0.006) |
| A3 ellipse, A2 SelvaBox boxes + LACE masker | scored |
| A1 LACE boxes + zero-shot SAM 3 | RUNNING (3 seeds, local MPS); do not read `tierA/a1_tiles_s*` until `tierA/summary_tierA.json` has `a1_*` keys |

## File formats
- **preds** (`preds_*.json`): `{"meta": {...}, "preds": {tile_id: {"boxes_2048": [[x0,y0,x1,y1],...], "scores": [...], "masks_rle": [{"size":[2048,2048],"counts": str}, ...], ...}}}`; masks are COCO RLE at 2048.
- **scored** (`scored_*.json`, output of `score_coco.py`): `protocol` (res, max_dets, score_floor), `canopy_neutral_crowd` → `mask`/`box` → `AP50`, `AP75`, `AP50_95`, `n_det` (the paper's headline arm). No per-tile breakdown: recompute from preds + GT with `boxinst_commonality_tcd_04.score_coco` (`build_gt`, `_score_one`).
- **GT**: `../test_gt.json` (`{tile_id: {"trees": [flat polygon], "canopy": [...], "W", "H"}}`, 2048 px coords); sparse-236: `../../data/tcd_sparse/sparse_gt.json`. Images: `../../data/tcd/test/<tile_id>.tif`.

## Where things are
| what | path |
|---|---|
| Table 1 rows, current paper | LACE `scored_lace_s{0,1,2}_r2048_cgf_r12_tau035_guidedrun.json`; Restor `scored_restor_rpn1000_r2048_tree.json`; seam-fixed baselines `seamfix/scored_{dt2_s0v2,box2mask_r50_s0,box2mask_swinl_s0}_r2048_seamfix.json`, `seamfix/scored_selvabox_seamfix_on_r2048_md{600,1600}.json`; sparse DT2 `seamfix/scored_dt2_sparse_s0v2_r2048_seamfix.json` |
| pre-fix baselines (for comparison) | `scored_{dt2_s0v2,box2mask_*,selvabox_bench}_r2048*.json` |
| SelvaBox, released SAM placement | `selvabox_persub/scored_selvabox_persub_r2048{,_rankgmean}_md{600,1600}.json`; preds alongside; per-tile `selvabox_persub/tiles_mps/` (with `src_subtile`, `sam_scores`) |
| A4 masker comparison (per crown) | `audit_2026-10-14/selvabox_evidence/gtbox_sam_persub.json` (summary, CIs, parity); per-tile per-crown IoUs `.../gtbox_sam_persub_tiles/<tid>.json` (`valid`, `ft_whole`, `ft_sub`, `base_whole`, `base_sub`, crown order = `test_gt.json` polygons); LACE per-crown `gtbox_guided_raw/gtbox_lace_guided_r2048.jsonl`; paper SAM arms `../results_439/gtbox_sam3_{base,ft}_r2048.json` |
| Tier A rows | `tierA/summary_tierA.json` (all scored rows), `tierA/scored_*.json`, `tierA/preds_*.json`; A2 per-tile `tierA/a2_tiles/` (`scores` = LACE product, `scores_det` = SelvaBox score); env records `tierA/*/_env.json`; A2 gate `tierA/a2_gate.json` |
| SAM field-of-view tests (25-tile, superseded by A4 for the masker claim) | `audit_2026-10-14/selvabox_evidence/sam_fov_{gtbox,control}.json` |
| seam analysis | `seamfix/seam_analysis_seamfix_r2048.json` (vs `seam_analysis_r2048.json`) |
| strata-maskiou (seam-fixed) | `../ablation/results_r2048/box_iou_strata_seamfix_r2048.json` |
| audit reports | `audit_2026-10-14/{box2mask,detectree2,selvabox,restor,lace_selection,paper_numbers}.md` |
| plans | `../box_supervised_baselines/PLAN.md` §13 (Tier A/B/C pre-registration) |

## Comparators to keep in view
LACE product (deployed) 0.647 ± 0.003 AP50; LACE ranked by detector score s 0.614 ± 0.003 (`scored_lace_s*_r2048_cgf_r12_tau035_detscore.json`). Bands are ddof=1 over seeds 0–2.

## Rules
A sample result is provisional until measured on the full set, against the strongest competitor variant, with a parity check against stored paper numbers. Say which, every time.
