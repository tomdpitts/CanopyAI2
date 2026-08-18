# `data/tcd_sparse` — unseen open-canopy OAM-TCD test slice

**236 tiles · 14,937 GT crowns · 6 biomes · zero overlap with the 900 the detector was trained on.**

Built by `boxinst_commonality_tcd_04/modal_sparse_tcd_multiseed/build_slice.py`. Full provenance
per tile in `slice_manifest.json`.

## Why

The settled phase-4 model (mask mAP50 0.630 ± 0.005 on the official 439) was fitted on 900 tiles that
are ~80% closed-canopy forest — 401 tropical moist broadleaf, 315 temperate broadleaf, and only 58 in
any open-canopy biome. That headline therefore says nothing about the sparse/dryland regime. This
slice measures the **existing, unmodified** model there: no retraining, no masker refit.

## What is in it

```
test/  tcd_tile_*.tif + tcd_tile_*_meta.json     (real copies, 2.8 GB — annotations travel in the meta)
sparse_gt.json        {tid: {trees, canopy, W, H}}   — same shape as the frozen test_gt.json
slice_manifest.json   criteria + counts + per-tile provenance
```
Plus `../../boxinst_commonality_tcd_04/modal_sparse_tcd_multiseed/manifest_sparse.json`, in the
phase-4 manifest shape, so the Modal chain can join to HF `restor/tcd` by `image_id`.

There is **no train split** — this slice is evaluation-only by design.

| biome | | tiles |
|---|---|---|
| 7 | Trop/Subtrop Grassland, Savanna & Shrubland | 102 |
| 12 | Mediterranean Forest, Woodland & Scrub | 56 |
| 8 | Temperate Grassland, Savanna & Shrubland | 31 |
| 13 | Desert & Xeric Shrubland | 24 |
| 10 | Montane Grassland & Shrubland | 16 |
| 9 | Flooded Grassland & Savanna | 7 |

Sources: 220 from the unseen part of `data/tcd/train`, 16 from the official 439.

## How exclusion is enforced

Four gates, all of which must pass or the build aborts. Each is independently exercised by
`--inject-seen <tid> --inject-gate {1,2,3}`, which smuggles a real training tile in and requires the
named gate to fail:

1. **tile id** vs both seen-900 records (`phase4/manifest.json` and `train_tiles_gt.json`, asserted
   identical to each other)
2. **image_id** vs the seen-900 image_ids
3. **rgb sha1** vs the seen-900 pixels — the only gate that catches the same ortho crop republished
   under a different `image_id`. The hasher self-checks against the 20 sha1s recorded in
   `phase4/manifest.json`.
4. **biome** ∈ the chosen set

## Caveats — read before quoting a number

- **Tile-disjoint, NOT scene-disjoint.** Only 106 of 236 are `scene_clean` (their 300 m
  single-linkage spatial cluster contains no training tile). The other 130 are adjacent crops of an
  OpenAerialMap ortho the model has already seen. **Report the scene-clean cut as the honest
  headline**; the full 236 is the optimistic one. One filter: `[t for t in tiles if t.scene_clean]`.
- **16 tiles come from the official 439.** Never trained on, but they *were* used to tune `mask_thr`
  and the α/κ knobs. `source == "train_pool"` (220) drops them.
- **Biome is geography, not canopy density.** Biome 13 (Desert & Xeric) averages 99.5 crowns/tile
  and 17.8% labelled canopy here — denser than several closed-canopy biomes; visually its tiles are
  olive/oak woodland. Biome 7 is the genuinely open one (8.4% canopy, 2.4% crown pixels).
- **`canopy_frac` / `crown_frac` are labelled coverage, and are UNION areas.** OAM-TCD stores
  segmentations as polygons *or* RLE, and **every RLE annotation carries `area == 0`** — summing the
  COCO `area` field silently drops the large canopy regions. These are decoded and unioned. They
  still measure what was *annotated*: a tile can read as sparse because its closed part went
  unlabelled (mangrove reads 2.2% for exactly this reason).
- **No-data wedges: kept deliberately.** OAM orthos are rotated, clipped mosaics — median
  `valid_frac` 93%, with 107 tiles <90% and 43 <70% non-black. Recorded per tile, **not** filtered:
  this slice is a fixed benchmark that other models will be run against, so every method must face
  the identical challenge. Do not add a `valid_frac` floor without re-running every model on the
  filtered set.
- **One candidate dropped:** `tcd_tile_231` (biome 12, 0 crowns) — zero-byte tif in the local mirror.
  `tcd_tile_121` and `tcd_tile_1878` are likewise zero-byte (HF `image_id` 0/1/2, a truncated
  download); `tcd_tile_1878` is one of the 900, so it sits outside gate 3's pixel comparison, though
  gates 1 and 2 still cover it. The HF rows are fine — only the local copies are empty.

## Reproduce

```
.venv/bin/python -m boxinst_commonality_tcd_04.modal_sparse_tcd_multiseed.tile_index      # index + inventory
.venv/bin/python -m boxinst_commonality_tcd_04.modal_sparse_tcd_multiseed.build_slice --dry-run
.venv/bin/python -m boxinst_commonality_tcd_04.modal_sparse_tcd_multiseed.build_slice
```
Non-destructive: reads `data/tcd/{train,test}`, writes only here. `rm -rf data/tcd_sparse` undoes it.
