# Qualitative figures (Results section)

Made 2026-09-26. All three are 400 dpi JPEG-wrapped PDFs (matplotlib's lossless embed was 26 MB).
Each has a `.jpg` preview and a `*_meta.json` with the tiles, thresholds, crop origins and per-window F1 and TP/FP/FN (panel labels show F1 first).

| file | paper label | content |
|---|---|---|
| `qualitative_tcd439.pdf` | `fig:qual` | OAM-TCD 439 holdout, tiles 24, 146, 431, 258 (<3 % canopy; LACE modestly ahead or level); GT, LACE, Restor, Detectree2, Box2Mask Swin-L; 768 px windows, native 2048 masks |
| `qualitative_sparse.pdf` | `fig:qual-sparse` | sparse-236 holdout, tiles 426, 2009, 3569, 413, 1762, 3969 (savanna/xeric ecoregions, <0.5 % canopy; LACE modestly ahead); GT, LACE, Detectree2; 1024 px windows |
| `qualitative_neon.pdf` | `fig:qual-neon` | NeonTreeEvaluation boxes, one tile per site, LACE modestly ahead (SJER_057, TEAK_057, NIWO_012, OSBS_003, MLBS_069, DSNY_025); GT, LACE 4-phase s0, DeepForest 2.1.0 |

## Regenerate

```
.venv/bin/python boxinst_commonality_tcd_04/ablation/results/figs/make_qualitative.py --set tcd439
.venv/bin/python boxinst_commonality_tcd_04/ablation/results/figs/make_qualitative.py --set sparse
.venv/bin/python boxinst_commonality_tcd_04/ablation/results/figs/make_qualitative_neon.py
```
Env overrides: `TILES=` (comma list, tile numbers for TCD, full keys for NEON), `CROP_PX=` (2048 = full tile), `SUFFIX=`.
~1 min each once thresholds are cached.

## Inputs

- TCD 439: `test_gt.json`, `data/tcd/test/*.tif`, `native_raster/preds_lace_s0_r2048_tau040.json` (product-reranked score),
  `preds_restor_rpn1000_r2048_treeonly.json`, `preds_dt2_s0v2_r2048.json`, `preds_box2mask_swinl_s0_r2048.json`.
  SAM 3 / SelvaBox has no persisted 2048 masks, hence absent.
- Sparse: `data/tcd_sparse/sparse_gt.json`, `data/tcd_sparse/test/*.tif`, `native_raster/preds_lace_sparse_s0_r2048_tau040.json`,
  `preds_dt2_sparse_s0v2_r2048.json` (only rows in `tab:sparse236`).
- NEON: `modal_neon_multiseed/neon_gt.json`, `NeonTreeEvaluation/evaluation/RGB/*.tif` (400 px),
  `phase4/dl/preds_phase4_s0.json`, `preds_deepforest.json` (2.1.0 repro).

## Thresholds (display only; AP tables use the full ranked list)

`qualitative_thresholds_<set>.json`, cached; delete to recompute. Per model, the score cut maximising F1 over the whole set:
- TCD sets: pooled F1 at mask IoU 0.5, COCO-style greedy score-ordered matching, >50 %-in-canopy predictions ignored (and not drawn).
  439: LACE 0.11, Restor 0.24, Detectree2 0.49, Box2Mask 0.77. Sparse: LACE 0.12, Detectree2 0.61.
- NEON: F1 of macro-averaged P/R at box IoU 0.4 (benchmark convention). LACE s0 0.30 (F1 0.7227), DeepForest 0.24 (F1 0.7265);
  these reproduce the paper's NEON table to 3 dp.

Tile selection (2026-10-06, Tom's call: reasonably favourable to LACE, not the best possible; captions are one-line
descriptors and do not state the rule): `select_tiles.py --set <set>` ranks every eligible tile (>=10 GT crowns; canopy
area <0.5 %, or <3 % for tcd439 via `CANOPY_MAX=0.03`) by LACE's whole-tile F1 margin over the best baseline at the display
thresholds -> `qualitative_tile_ranking_<set>.json`. Base rates: LACE leads on 7/43 (tcd439; 3/14 at <0.5 %), 30/43
(sparse), 66/126 (neon). Current tiles are mid-ranked: margins -0.012..+0.06 (tcd439), +0.000..+0.11 (sparse),
+0.01..+0.10 (neon), picked for site/ecoregion spread. Earlier drafts: density-and-diversity pick (180,122,112,48 /
290,3569,426,2976,583,1762 / SJER_025,TEAK_057,NIWO_011,OSBS_029,MLBS_069,JERC_059; Restor led every 439 window) and the
top-margin pick (125,146,24,407 / 2321,919,290,2976,3770,2009 / SJER_053,TEAK_334,NIWO_001,OSBS_033,HARV_041,DSNY_012;
rejected as too strong). Crop window = CROP_PX square with the most GT centroids (stride 64).

# Architecture figure (Methods)

Made 2026-10-05, revised the same day. `lace_architecture.pdf` (18.2 cm wide, prints 1:1 at MDPI `\fulllength`), preview
`lace_architecture_preview.png`. Not yet referenced by `paper.tex`; `lace_overview.pdf` (`fig:overview`) is untouched.
Top row: input, dense encoding (four 16 px patch grids offset by 8 px, interlaced), detection, box-to-mask, output.
Bottom: the box-to-mask module as Fit (whiten, frozen background mixture, foreground init, EM, frozen masker) and
Inference (one E-step per predicted box, rescore).

Every inset is ONE holdout crop, `tcd_val_tile_147` at x0=128, y0=768, 512 px, chosen by a crop search over all 439 tiles
for a sparse, canopy-free window where LACE is near-perfect at the paper's display cut (s' >= 0.11, the LACE row of
`qualitative_thresholds_tcd439.json`): 14 GT crowns, 14 TP, 0 FP, 0 FN at mask IoU 0.5. Boxes, s' and native-2048 masks in
the detection/output insets are the published ones (`native_raster/preds_lace_s0_r2048_tau040.json`). The heatmap and the
zoomed box's per-cell A, A-bar, p_fg come from re-running the deployed s0 detector + `em_model_4p_fix.npz` on the deployed
features. Zoomed box = the top-ranked crown in the crop (100x111 px, s=0.49, p-bar=0.65, m=0.81, s'=0.258, published 0.258).
The Fit "background mixture" inset applies the fit's own partition helpers (in-box / ring / clear) to this crop's GT boxes:
the tile is holdout, so it illustrates the partition rule, it was not part of the fit. Say so in the caption.

Features must be the **Modal** ones (local transformers 5.12 drifts to cos ~0.86; tile 122 gave 204 boxes vs 445 locally):
```
.venv/bin/modal volume get tcd04-phase4-vol feat_4p_test/tcd_val_tile_147.npy \
  boxinst_commonality_tcd_04/ablation/results/figs/arch/_cache/tcd_val_tile_147_4p_L24.npy
.venv/bin/python boxinst_commonality_tcd_04/ablation/results/figs/arch/make_arch_panels.py
cd boxinst_commonality_tcd_04/ablation/results/figs && tectonic lace_architecture.tex
```
Font: Calibri when installed (macOS: Font Book > Calibri > Download), else Carlito, its metric-identical open clone
from the TeX bundle; the `.tex` picks automatically, so rebuild after installing Calibri.
Each run prints the reproduction check (tile 147: 596/596 boxes, 99.8 % at IoU>0.99, max |s' local - published| 2.6e-4).
`_cache/` is gitignored (128 MB per tile). Tile 122 files there are from the first draft; `*_localtf512.npy` is the drifted
local extraction, kept only as the counter-example.
