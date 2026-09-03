# Where every Table 1 artefact lives

**Rule (learned the hard way, three times):** a prediction that cost GPU time never lives only in
a session scratchpad. It goes on the Modal volume that produced it AND its stitched form is
committed here. Modal volumes are durable — the LACE seed 1/2 predictions were recovered from one
on 2026-09-02 after being presumed lost. **Search the volume before declaring anything gone.**

Two artefact levels matter:
- **stitched preds** (`{meta, preds{tile: {boxes_2048, scores, masks_rle@512}}}`) — what
  `score_coco.py` consumes. All are in the repo.
- **raw per-subtile preds** (one JSON per 1024² subtile, masks at native 1024) — lets you
  re-stitch at a different score floor or dedup rule **without re-inference**. This is what made
  the DetecTree2 score-floor question answerable at no GPU cost.

| row | stitched preds (in repo) | raw per-subtile | scored |
|---|---|---|---|
| LACE s0/s1/s2 (+product) | `confidence/preds_knobbed_s{0,1,2}*.json` | n/a — LACE infers whole 2048² tiles, so the stitched file IS native | `results_439/rerank_product_*.json` |
| Restor Mask R-CNN | `restor_baseline/preds_restor_rpn1000.json` (+ rpn512/2000, pooled) | n/a — whole-tile inference, no subtiling | `results_439/restor_*.json` |
| **DetecTree2 s0 (converged, 2026-09-03)** | `detectree2_baseline/preds_dt2_s0v2.json` | `tcd-detectree2-vol:out/pred_raw_s0v2` (3,951) | `results_439/dt2_s0v2_coco512.json` |
| SelvaBox → SAM 3 | `selvabox_baseline/preds_selvabox_bench.json` | **MISSING** — see gap below | `results_439/selvabox_bench_*.json` |
| **Box2Mask R-50** | `box_supervised_baselines/preds_box2mask_s0_test.json` | `tcd04-baselines-vol:boxinst/pred_raw_box2mask_s0_test` (3,951) | `results_439/box2mask_s0_coco512.json` |
| **Box2Mask Swin-L** | `box_supervised_baselines/preds_box2mask_swin-l_s0_test.json` | `tcd04-baselines-vol:boxinst/pred_raw_box2mask_swin-l_s0_test` (3,951) | `results_439/box2mask_swinl_s0_coco512.json` |

## DetecTree2 supporting artefacts (`detectree2_baseline/`)

| file | what |
|---|---|
| `RETRAIN_V2.md` | the full re-investigation: four defects found, provenance check, what is fixed and what remains |
| `train_info_s0v2.json` | run report: 7 evals, early-stop at iter 14000, selected iter 10000, config |
| `train_s0v2_valcurve.json` | the val AP50 curve — the convergence evidence (`ap_history.json` from the volume) |
| `train_s0v2.log`, `predict_s0v2.log` | raw run logs |
| `results_dt2_s0v2.json` | `evaluate.py` scorer (feeds the `t01_boundary` reproduction gate) |
| `results_dt2_s0v2_noignore.json` | canopy-ignore sensitivity + per-IoU curves (feeds `t02_curves`) |

Checkpoint stays on `tcd-detectree2-vol:out/model_best_s0v2.pth` (503 MB, too large for the repo);
its per-eval checkpoints are in `out/train_s0v2/`. The initialisation checkpoint
`weights/250312_flexi.pth` is md5-asserted against Zenodo 15014353 by
`detectree2_modal.py::check_weights`.

**Superseded and DELETED from the working tree** (recoverable from git history, and deliberately
absent so any missed consumer fails loudly rather than silently emitting a stale number):
`preds_dt2_s0.json`, `preds_dt2_s0_fullcov.json`, `results_dt2_s0.json`,
`results_dt2_s0_fullcov.json`, `results_dt2_s0_fullcov_noignore.json`,
`results_439/dt2_s0_coco512.json`.

## Box2Mask supporting artefacts (`box_supervised_baselines/`)

| file | what |
|---|---|
| `train_box2mask_s0.json` | R-50 run report: schedule, COCO-init key check, cost |
| `train_box2mask_s0_valcurve.log.json` | R-50 val curve, 12 evals — the convergence evidence |
| `predict_box2mask_s0_test.json`, `stitch_box2mask_s0_test.json` | R-50 inference/stitch diagnostics |
| `train_box2mask_swin-l_s0*.json`, `*_valcurve.log.json` | same for Swin-L (curve spans two phases: iters 0–30,114 then the resume) |
| `predict_box2mask_swin-l_s0_test.json` | records **both** best-checkpoint candidates and which was selected |

Checkpoints stay on `tcd04-baselines-vol:boxinst/{box2mask_s0,box2mask_swin-l_s0}/` (2.5 GB for
Swin-L — too large for the repo). The COCO-init checkpoints with the 81-class head stripped are at
`boxinst/ckpt/`.

## Known gap — SelvaBox raw predictions

SelvaBox runs the same 1024²@0.5 grid as DetecTree2 and therefore HAS a raw per-subtile stage, but
only the stitched output was persisted. Consequence: any question needing a different score floor,
dedup rule or mask raster for that row costs a full re-run rather than a CPU re-stitch. Do not
repeat this pattern; see `feedback-persist-masks`.

## Retrieval

```bash
.venv/bin/modal volume ls  <vol> <path>
.venv/bin/modal volume get <vol> <path> <dest>     # --force to overwrite; a directory
                                                   # destination that does not exist yet gets
                                                   # written as ONE concatenated file, so make
                                                   # the dest dir first and check the file count
```
