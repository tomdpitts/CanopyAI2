# Modal storage audit — 2026-10-15

Trigger: about $1.30/day in Modal volume charges. The account has one workspace (`tomdpitts`), one environment (`main`) and 20 volumes, holding **about 1.58 TB** in total. `../volsizes.py <vol>...` re-measures any volume. It lists file metadata only and runs no compute.

## Inventory (GB)
| volume | GB | content |
|---|---|---|
| dinov3-dapt-arid-vol | 317 | DCP ckpts 283 (7 GB each) + export/ teachers 34 |
| tcd04-phase4-vol | 247 | feat_4p_train 121, feat_4p_test 59, native_test 59, rgb 7 — ACTIVE |
| savanna-phase4-vol | 196 | features only |
| tcd04-multiseed-vol | 194 | features (see below — NOT duplicates) |
| dinov3-dapt-v3-vol | 162 | DCP ckpts 140 + export/ 19 |
| neon-multiseed-vol | 129 | features ~124, outputs ~1 |
| canopyai-deepforest-data | 81 | images 53, hf_cache 20 — ACTIVE (tierb) |
| canopyai-deepforest-checkpoints | 68 | golf…tango, manual_s*, phase3–27 runs, SAM weights |
| tcd04-block1024-vol | 45 | features |
| tcd-detectree2-vol | 44 | data 28, out 16 |
| tcd-sparse-vol | 35 | features 32 |
| canopyai-checkpoints | 20 | run_alpha…run_echo (every-250-iter d2 ckpts) |
| 8 others | <16 each | baselines (15), treecloud, canopyai-data, wm2, benchmark-results, shadow, projectb, savanna-raw |

## T1 (approved by Tom): DAPT DCP checkpoints
`t1_dapt_plan.json` lists all 60 checkpoint dirs.
- **44 dirs, 308.6 GB, already have an exported teacher** (`/export/teacher_*.pth`, 1.21 GB each) → safe to delete. The optimizer and student state are lost, so training can't resume from those iterations.
- **16 dirs, 112.2 GB, have NO export** → KEEP: `out_s123` iterations 1499/2499/3499/3999/4499, `out_s42_shuf` iterations 1499–4999 (including the final one), `v3_s401` and `v3_s501` iterations 249/749. To export them, run `extract` from `dapt/ssl/modal/app.py` (seed=123 or 42, tag="" or "_shuf") or `dapt/v3/ssl/app.py`, then delete. That needs compute.

Status: **NOT deleted.** Claude Code's auto-mode classifier blocked `modal volume rm`, so Tom must run the deletion. One command per dir: `.venv/bin/modal volume rm -r <vol> <dir>`, for each row with `"action": "delete"` in the plan.

Pre-deletion file manifests (path, size, mtime): `manifest_pre_*.json`.

## tcd04-multiseed-vol is NOT a duplicate (T1 item withdrawn)
Every file matches its twin by name and size (900 against phase4 `feat_4p_train`, 439 against phase4 `feat_4p_test`, 439 against block1024 `feat_test`), but **0/12 sampled files are byte-equal** (`multiseed_spot_hash.json`). The arrays have different shapes: multiseed is (4096,128,128) and phase4 is (1024,256,256), with equal byte counts. It is a distinct feature representation. The volume's `out/` (result_s0.json, which is all-NaN with n_tiles=0, plus det_ms_s0.pt and a flag) is copied to `tcd04-multiseed-vol_out/`. No code in the repo references this volume. Deleting it is now a T2 decision, because the features can be regenerated.

## Blockers seen
- The workspace has **exceeded its spend limit**, so no deploy or compute is possible. Teacher exports and server-side hashing are blocked until the limit is raised.

## 2026-10-20 update: DAPT abandoned → both DAPT volumes deletable whole
Tom: "we've moved away from dapt completely". In paper.tex DAPT survives only as a commented-out line (l.397: no material improvement over frozen weights). All logs, configs and training_metrics (52 files, 4.1 MB) are copied to `dapt_provenance/`, and the full file lists are in `manifest_pre_dinov3-dapt-*.json`. With both volumes gone, 479 GB is freed (about $0.39/day). This replaces the partial T1 plan, including the 16 unexported checkpoints, which are no longer needed. Deletion: `modal volume delete dinov3-dapt-arid-vol` and `modal volume delete dinov3-dapt-v3-vol` (Tom to run).
- 2026-10-20: Tom chose 3 v3 seeds to keep → `dapt/teachers_keep/` (3.6 GB, sha256-verified). After that, both DAPT volumes can be deleted whole.
