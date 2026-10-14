# Seam fix for the subtile-stitched baselines (2026-10-14)

Record of truth: `../NATIVE_RASTER.md`, section "SEAM FIX". This folder holds every artefact of the fix.
Nothing outside it was overwritten. Inputs that were read but not modified: `../preds_*_r2048.json`,
`/Users/tompitts/dphil/feat_cache/{dt2_raw,b2m_raw}`, the SelvaBox tile files (local copy
`feat_cache/selvabox_bench_r2048_tiles/`, original on `tcd04-baselines-vol:out/native_raster/selvabox_bench_r2048/tiles/`).

## Code
| file | role |
|---|---|
| `../stitch_seam.py` | `stitch_res` + pre-dedup interior-edge filter (detectree2 `stitch_crowns(shift=1 m)` = 10 px) |
| `../run_stitch_seam.py` | pooled driver with checkpoint/resume and the byte-level gate (`--gate_ref`) |
| `run_all.sh` | the 8 stitches (gate + fixed for DT2, DT2 sparse, Box2Mask R-50 / Swin-L) |
| `../selvabox_seam.py` | SelvaBox selection replay from `raw_dets` + CanopyRS edge band (5% of 1024 px) at interior edges |
| `../selvabox_seam_plan.py` | box-level replay check + reuse plan -> `selvabox_seam_plan.json` |
| `../selvabox_seamfix_local.py` | SAM 3 refill on MPS (verify / refill / assemble) |
| `../selvabox_seamfix_modal.py` | the same on A100; written, NOT deployed (Modal workspace paused 2026-10-14) |
| `selvabox_env_gate.py` | local-vs-A100 SAM masks at AP level, 44 tiles |
| `score_rest.sh` | scoring of the SelvaBox and sparse rows |
| `seam_analysis_seamfix.py`, `empty_tile_share.py`, `b2m_cap_subset_seamfix.py` | downstream numbers |
| `../ablation_r2048/box_iou_strata_res_seamfix.py` | tab:strata-maskiou on the fixed files |

## Outputs
- `preds_{dt2_s0v2,dt2_sparse_s0v2,box2mask_r50_s0,box2mask_swinl_s0}_r2048_seamfix.json` (+ `.jsonl` checkpoints), `*_GATE.json`
- `preds_selvabox_seamfix_{on,off}_r2048.json` (2048 masks), tile dirs `selvabox_seamfix_{on,off}_tiles_mps/` (1777/2048/512 masks, reuse flags)
- `scored_*_seamfix*.json`, `scored_selvabox_seamfix_{on,off}_r2048_md{600,1600}.json`
- `seam_analysis_seamfix_r2048.json`, `empty_tile_share.json`, `b2m_cap_subset_seamfix.json`, `selvabox_env_gate.json`,
  `selvabox_verify_reuse_local_{mps,cpu}.json`; strata: `../../ablation/results_r2048/box_iou_strata_seamfix_r2048.json`
  (+ gate `box_iou_strata_guided_r2048_GATE_rerun_2026-10-14.json`)
- Logs: `../logs/seamfix_*.log`, `../logs/selvabox_*`, `../logs/score_*seamfix*.log`
