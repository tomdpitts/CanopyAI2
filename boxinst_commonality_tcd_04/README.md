# boxinst_commonality_tcd_04 — weakly-supervised ITC instance segmentation on OAM-TCD

Individual-tree-crown (ITC) **instance segmentation from bounding boxes only**, on
a **frozen** DINOv3 backbone, scored on the official OAM-TCD 439-tile test set.
Thematic-arm question: can a frozen DINOv3 (web now; DAPT-finetuned vs web later)
drive competitive instance segmentation under weak supervision?

**Result: mask mAP50 = 0.499** on the 439 test — **beats 0.432**, the fully-supervised
Restor Mask R-CNN benchmark (mask labels + trained backbone) — using **only ITC
boxes** and a **frozen** backbone. Training never sees a polygon; canopy (category 1)
is COCO-ignore throughout; polygons touch the pipeline only inside `evaluate.py`.

## Results (full 439 test, web backbone, seed 0, `det_t8` + `em_model.npz`)

| metric | single-scale | + downscale arm (`--multiscale`) |
|--------|-------------|----------------------------------|
| **mask mAP50 (ITC, headline)** | 0.499 | **0.504** — beats supervised 0.432 |
| mask mAP50-95 | 0.156 | 0.159 |
| box mAP50 (detector ceiling) | 0.555 | 0.556 |
| semantic F1 (canopy-excluded) | 0.587 | **0.645** |
| semantic P / R | 0.81 / 0.46 | 0.79 / 0.55 |

box→mask cost is only ~0.05 (what the training-free EM masks cost vs the box
ceiling). Semantic F1 is at the val-picked operating threshold; mAP uses the full
low-threshold decode.

**Multiscale (downscale big-tree arm, `cache_test_down.py` + `evaluate --multiscale`).**
Recall by GT crown size showed the detector has a **32–128px sweet spot** (recall
~0.65–0.70) and fails at *both* tails: <32px 0.31 (sub-16px-patch), 128–256px 0.36,
>256px 0.03. The downscale arm runs `det_t8` on the tile at 0.5× (a 256px crown
looks ~128px, back in the sweet spot), scales boxes ×2, and cross-scale NMS-merges;
masks stay native. Inference-only, no retrain. Effect: **128–256px recall 0.48→0.66**,
mask mAP50 +0.005 (safe — big trees are rare so instance-mAP barely moves), **semantic
F1 +0.058** (big trees = big area).

**Upscale arm for the <32px tail: TESTED AND REJECTED** (isolated experiment,
folder since deleted). On a 60-tile tiny-crown-rich subset — upscale's *best case*
— running `det_t8` on 2× tiles recovered <32px recall 0.28→0.61 (total recall
0.46→0.68) but **dropped mask mAP50 0.44→0.39**: the recovered crowns bring FPs,
and a <32px crown is <8px at the 512 eval raster, so its mask often can't clear
IoU 0.5 even when the box is right — recall rises, AP falls. Since the full 439
has proportionally fewer tiny crowns (only more FP surface), it can only look
worse; not folded in. Upscale is a *coverage/recall* lever (e.g. tree counting),
not an AP lever. Cost also steep: 2× features are 537MB/tile (~235GB for 439) or
on-the-fly recompute.

Crop-trained baseline `det_d8` (superseded): mask 0.254 / box 0.275. The **+0.245**
came from full-tile training (below). Still box-limited (box→mask only 0.056), and
`det_t8` overfits — val boxAP50 peaks ep20 (0.471) then falls as train loss → 0 on
792 train tiles, so 0.499 is **overfit-limited**; regularization + more of the 3,611
available tiles (we use 900) is the open lever, plus precision (R 0.46 < P 0.81).

## Architecture: detection decoder + training-free masks

```
frozen DINOv3-web (patch-16, cached)  ─►  8px DETECTION DECODER  ─►  ITC boxes
                                            (trained, boxes only)        │
                                                                         ▼
                                                    training-free COMMONALITY EM ─► masks
```

Two decoupled modules (deliberate): the **detector** is the only trained part and
the sole lever on mask mAP50, because on TCD ITC the mask-given-a-box problem is
near-saturated (a learned CondInst mask head *underperformed* box-fill in
`boxinst_tcd`). So masks stay training-free and effort goes to detection.

- **Detector** (`detector.py` → `Detector8`): frozen 16px features → bilinear ×2 →
  **8px-grid CenterNet** (heatmap + offset + log-size). The 8px grid halves the
  small/adjacent-crown peak-merging that a 16px grid suffers in closed canopy (ITC
  median crown 46px, p5 13px). Fully convolutional → same weights on any tile size.
- **Commonality EM** (`em.py` → `TCDMasker`): the `boxinst_commonality` method
  (whiten + within-box contrast + contrastive prototype repel + size-conditioned
  spatial priors + auto-K), fit on the `boxinst_tcd` 720 ITC crops' box interiors
  (its default `--feat_dir`) with **canopy as an ignore label**. No polygon, no
  gradient. Grid-agnostic (stores stride, not size), so it applies unchanged to the
  128×128 test-tile grid.

**Why full-tile training (the +0.245 unlock):** train on whole 2048 tiles, exactly
like the test — *not* curated crops. This removed both distribution problems at
once: (1) the old crop filters (size floor 20px, density ceiling 28 trees/window)
excluded exactly the tiny + dense crowns the test is full of; (2) the train/test
grid-size mismatch (crop 32→64 vs tile 128→256). Box AP jumped 0.275 → 0.555.

## Repository map (code)

| file | role |
|------|------|
| `prepare_test.py` | 439 test GT → `test_gt.json` (ITC polys + canopy ignore). Decodes RLE `segmentation` dicts (pycocotools → zero-padded contourpy contour → polygon); the pad closes border-touching canopy. |
| `cache_test.py` | Stitched 128×128 web features per 2048 test tile (2×2 of 1024 windows — avoids the O(N²) 2048 pass). `tile_feature()` is reused by train caching. |
| `cache_train_tiles.py` | Samples N ITC-bearing full 2048 train tiles, extracts GT (`train_tiles_gt.json`) + stitched features. |
| `detector.py` | `Detector8` (8px CenterNet), `det_loss` (focal + smooth-L1 + GIoU), `canopy_cell_mask`, `CFG8`. |
| `train_detector_tiles.py` | **Winning trainer** — `Detector8` on full 2048 tiles (256-grid), all ITC boxes/tile, canopy-ignore, incremental best-checkpoint save. |
| `train_detector.py` | *Superseded* crop trainer (reused `boxinst_tcd`'s 720 curated 512 crops → `det_d8`). Kept for the baseline comparison. |
| `em.py` | Commonality EM fit (`fit`, boxes-only, canopy-ignore) + `TCDMasker` posterior applier. Reuses pure-math core from `boxinst_commonality.em`. |
| `evaluate.py` | 439 eval: detect → EM masks → mask mAP50 / box mAP50 (shared matching, canopy-ignore) + semantic F1. `--multiscale` adds the downscale big-tree arm. |
| `cache_test_down.py` | 0.5× downscaled test features (one 1024 pass/tile, 64×64 grid) for the `--multiscale` big-tree arm. |

Shared infra imported from the repo: `dapt.backbone` (frozen DINOv3), `dapt.targets`
(`encode`, `TargetConfig`), `dapt.decode` (peak-pick), `dapt.eval` (AP), and
`boxinst_commonality.em` (EM math).

## Artifacts & paths

| path | what |
|------|------|
| `artifacts/det_t8.pt` | **headline detector** (full-tile, ep20). `{state, cfg}`; cfg has `score_thr`, `width`, `tower`, `in_dim`. |
| `artifacts/det_d8.pt` | crop-trained baseline detector. |
| `artifacts/em_model.npz` | EM masker (mu, U, scale, C, Gbg, wbg, pi, kappa, size_edges, s_px). |
| `artifacts/em_fit_report.json`, `artifacts/eval_t8.json` | EM fit log; 439 eval metrics. |
| `test_gt.json` | 439 test GT (`{tid: {trees:[poly@2048], canopy:[poly@2048], W, H}}`). EVAL ONLY. |
| `train_tiles_gt.json` | sampled train-tile GT + partition. |
| `cache/web/feat_test/` (55G) | 439 test tile features. |
| `cache/web/feat_traintile/` (113G) | 900 train tile features. |
| `claude_outputs/boxinst_commonality_tcd_04/` | qualitative renders (`detcheck_*`, `peak_separation/`). |

## Replicate the TCD result

```bash
V=.venv/bin/python
# 1. test GT + features (once)
$V -m boxinst_commonality_tcd_04.prepare_test
$V -m boxinst_commonality_tcd_04.cache_test                       # 439 tiles → feat_test/
# 2. train-tile GT + features (sample 900 of ~3611 ITC tiles)
$V -m boxinst_commonality_tcd_04.cache_train_tiles --n 900        # → feat_traintile/
# 3. fit the training-free EM masker (boxes only)
$V -m boxinst_commonality_tcd_04.em --seed 0                      # → em_model.npz
# 4. train the 8px detector on full tiles (best-ckpt saved each eval; ~ep20 peak)
$V -m boxinst_commonality_tcd_04.train_detector_tiles --tag t8 --epochs 40 --bs 3 --eval_every 5
# 5. evaluate on the 439 (mask mAP50 + box + semantic F1)
$V -m boxinst_commonality_tcd_04.evaluate --det det_t8            # → eval_t8.json
# 5b. (optional) big-tree downscale arm: recovers 128-256px recall, lifts F1
$V -m boxinst_commonality_tcd_04.cache_test_down                  # → feat_test_down/ (~14GB)
$V -m boxinst_commonality_tcd_04.evaluate --det det_t8 --multiscale
```

Key CLI flags: `train_detector_tiles` — `--width 256 --tower 3 --lr 1e-3 --wd 1e-4`
(raise `--wd` / lower `--tower` to fight the overfit); `evaluate` — `--eval_score_thr
0.05` (AP recall tail, keep low), `--op_thr` (semantic-F1 operating threshold;
defaults to the checkpoint's val-picked `score_thr`), `--limit N` (subset for quick
runs). All modules take `--arm web` (swap for a DAPT checkpoint registered in
`dapt/ssl/checkpoints.json` to run the backbone comparison).

## Run on a new dataset

The pipeline is dataset-agnostic given **COCO-style boxes (train) + polygons/masks
(test-eval only)** at a fixed tile size. To port:

1. **Data adapters** — point `prepare_test.py` / `cache_train_tiles.py` at the new
   tiles. They read per-tile COCO metas (`coco_annotations` JSON, category ids for
   the positive class + an optional ignore class). Adjust `TCD_TRAIN`/`TCD_TEST`
   paths, the category ids (2=ITC, 1=canopy here), and tile size if ≠ 2048.
2. **Tiling** — `cache_test.py` assumes 2048→2×2 of 1024 (16px grid). For a
   different tile size, change `STARTS`/`WIN`/`GRID2K` so windows stay ≤1024 (the
   MPS O(N²) ceiling) and stitch to `tile/16`.
3. **Detector grid** — `CFG_TILE` in `train_detector_tiles.py` = `TargetConfig(grid=
   tile/8, stride=8)`; `evaluate.py` decodes with `stride=8`. No code change if you
   keep 8px.
4. **Fit + train + eval** — same 5 commands. The EM auto-selects K; the detector is
   fully convolutional so nothing is hard-coded to 2048.
5. **No ignore class?** Pass empty canopy lists; canopy handling no-ops.

## Caveats

- Backbone frozen (web); DAPT-vs-web is the next comparison (swap `--arm`).
- Masks never see a polygon; EM is fit on box interiors, canopy ignored.
- `det_t8` overfits (0.499 is the early-peak, overfit-limited number).
- Test tiles stitched from 1024 windows — seam patches lose one-patch attention
  context (negligible for detection).
- Eval rasters at 512px (2048/4) for memory; mask IoU is at that resolution.
