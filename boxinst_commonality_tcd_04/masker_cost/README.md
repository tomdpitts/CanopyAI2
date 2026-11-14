# Masker cost: LACE vs SAM 3 (local, M4 Max MPS)

**Question.** In a world where SAM 3 exists, is the LACE masker worth having on cost grounds? What does each masker cost per 2048 px OAM-TCD tile, end to end as deployed in the paper, on identical boxes and the same machine?

**Status:** DONE 2026-11-14. All gates pass. Source: `results/summary.json`, `logs/summarise.log`.

## Result (20 test tiles, 10,431 D1 boxes, median 600 per tile; M4 Max, MPS)

| per 2048 tile | median s | range |
|---|---|---|
| LACE features (16 phases × 4 windows = 64 DINOv3 ViT-L/16 passes) | 59.9 | 59.4–63.3 |
| LACE render (every box, CPU) | 6.1 | 0.7–6.4 |
| **LACE total** | **66.0** | 60.2–69.4 |
| SAM 3 encoder (8–9 subtile encodes) | 5.8 | 5.0–6.6 |
| SAM 3 decode + post-processing | 4.3 | 0.5–8.5 |
| **SAM 3 total** (pass 1) | **10.1** | 7.1–14.3 |
| SAM 3 total (pass 2, drift check after LACE) | 9.7 | 6.1–13.9 |

- **LACE ÷ SAM 3 per tile: 6.7× (median; range 4.7–9.1×).** SAM 3 is taken per tile as the mean of its two bracketing passes. LACE as deployed is more expensive than SAM 3, not cheaper.
- **The cost is the 4 px feature lattice:** 64 ViT-L passes per tile against SAM 3's 9 encodes, about 91 % of LACE's time.
- **Per box, LACE's render is no cheaper than SAM 3's decoder:** 10.3 ms against 9.1 ms. The render alone (no features) is 0.55× SAM 3's total per tile.
- **Parameters:** DINOv3 ViT-L/16 303.1M plus a fitted masker of 0.139M numbers, against SAM 3 (HF Sam3Tracker) 458.3M.
- **Drift:** SAM 3 pass 2 − pass 1 = −0.66 s median, so there was no thermal penalty against SAM 3.
- **Gates:**
  - LACE feature cosine to the deployed cache ≥ 0.9999996 on every tile.
  - LACE masks are byte-identical to the deployed D1 test masks (10,431/10,431).
  - SAM 3 masks have mean IoU 1.0000 with the paper's SAM 3 run on every tile, in both passes.

### Fewer shifts (added 2026-11-14; measured, same 20 tiles)

Arms `lace4` / `lace1` in `bench.py` = the IL-test arms `interp8` / `interp16` (`dino_det/interlace_test.py`): features on 4 offsets {0,8}² or the native grid, bilinearly interpolated to the 4 px lattice, same render. AP is the published IL-test result on the full test 439 (D1 boxes, s′ ranking; `dino_det/results/interlace/interlace_test.json`), not re-run here. SAM 3 here = per-tile mean of three passes (p1, p2, p3 after these arms; p3 median 10.5 s).

| LACE config | ViT passes | features s | render s | total s (range) | ÷ SAM 3 (range) | test AP50 / AP75 / AP50:95 (s′) |
|---|---|---|---|---|---|---|
| 16 shifts, real 4 px (deployed) | 64 | 59.9 | 6.1 | 66.0 (60.2–69.4) | 6.6× (4.5–9.3) | 0.7044 / 0.2644 / 0.3366 |
| 4 shifts → 4 px bilinear | 16 | 18.7 | 6.1 | 24.9 (19.1–27.6) | 2.5× (1.7–2.9) | 0.7043 / 0.2619 / 0.3356 |
| 1 shift → 4 px bilinear | 4 | 5.3 | 6.1 | 11.4 (6.0–11.6) | 1.06× (0.78–1.24) | 0.6992 / 0.2490 / 0.3292 |
| SAM 3 zero-shot, √(s·SAM) | 9 encodes | | | 10.1 | 1 | 0.7013 / 0.3107 / 0.3553 |

- **4 shifts cost 2.6× less than 16 for the same AP50** (−0.0001, CI spans 0) and −0.001 AP50:95. Sixteen shifts are not worth their compute. Even so, 4-shift LACE is 2.5× dearer than SAM 3.
- **1 shift is cost-parity with SAM 3** and loses 0.005 AP50 / 0.007 AP50:95 to deployed LACE.
- "Features" for the interpolated arms includes the projection and interpolation (about 3.7 s for 4 shifts beyond the 16 ViT passes), which is why 4 shifts is above the 15 s projected from pass count.
- The render (6.1 s, single-threaded CPU) is now the floor; no shift count gets LACE below SAM 3 without also speeding it up.
- Gates: masks vs the stored IL-test arm masks (rendered from Modal features; ours are local t457): 9,469 / 10,431 and 9,376 / 10,431 byte-identical, mean IoU ≥ 0.9999 on every tile.
- Log labels in `logs/lace4_p1.log` / `lace1_p1.log` show a wrong tile counter (cosmetic; per-tile JSONs are correct).

### Caveats

- **The venv matters for LACE speed.** Attempt 1 ran the LACE features in `.venv` (torch 2.13 / transformers 5.12.1), which took 31.7 s on the warm-up tile against about 64 s here (torch 2.12.1 / transformers 4.57.1). But those features are numerically NOT the deployed ones (raw cosine 0.41 vs Modal; attempt invalid; `logs/bench_attempt1_venv_drift.log`). Even at the faster speed LACE would be about 3.5× SAM 3. The conclusion does not depend on the venv.
- **Hardware:** a single MPS machine; the CUDA ratios are not measured. Both methods are dominated by ViT forward passes (64 vs 9 at about 1024², with ViT-L for LACE and SAM 3's larger encoder for SAM 3), so the direction should hold on CUDA. The ratio could shrink if SAM 3's encoder is relatively slower there.
- **What is not tested:** a LACE sharing its backbone with the detector (as in the CenterNet variant); its cost would approach the render alone.

### Reading

The cost argument for the LACE masker as deployed in the paper (D1 + 16-phase 4 px masker) **fails**: at the paper's operating point SAM 3 is about 6.7× cheaper, and it ties at AP50 and wins at AP50:95. Fewer shifts narrow the gap to 2.5× (4 shifts, same AP50) or parity (1 shift, −0.005 AP50) but never invert it; a cost case would need features shared with the detector plus a faster render.

## Design (fixed before any timing was seen)

- **Boxes:** the frozen D1 test boxes (`dino_det/results/preds_dino_swinl_s0_test_R3_ef_r2048.json`). These are identical, tile for tile, to the boxes the paper's SAM 3 row was prompted with (`..._detscore_r2048.json`; 439/439 lists identical). Up to 600 boxes per tile; median 576.
- **Run order:** SAM 3 pass 1, LACE, SAM 3 pass 2 (a drift bracket). One method per process, each in the venv that produced the paper's numbers: LACE in `feat_cache/venv_feat_t457` (= Modal transformers 4.57.1), SAM 3 in the repo `.venv`.
- **Tiles:** 20 test tiles, a seed-0 permutation (`results/selection.json`), plus 1 warm-up tile that is run first and discarded.
- **LACE, as deployed:**
  - 16-phase DINOv3 ViT-L/16 L24 features: `_extract_2048` (2×2 windows of 1024) on the tile shifted by {0,4,8,12}², so 64 ViT-L forward passes per tile. This is `native_raster/hunt/feat16p.py`.
  - Then `TCDMasker.project`, the fp16 storage round-trip, the lace4px_simple render of every box (the parity-gated `zeroshot_phenobench/code/lace_core`), and RLE encoding.
- **SAM 3, as deployed:** `dino_det/sam3_on_d1.sam_tile` (Tier A1). This is `facebook/sam3` via HF Sam3Tracker: each box goes on the most central of the 3×3 1024 px subtiles (else the whole tile), multimask is off, masks are pasted at 2048, then RLE-encoded. It is the paper's SAM 3 row (AP50 0.7013).
- **Not timed:**
  - The D1 detector (DINO-Swin-L), which is common to both. It produces no DINOv3 features, so neither masker shares the detector's backbone.
  - Image decode and model load, for both.
- **Timing:**
  - Wall clock, `torch.mps.synchronize()` at every timer edge.
  - Both models are resident in one process, and the order alternates per tile.
  - Splits: LACE features vs render; SAM encoder (instance-wrapped `get_image_embeddings`) vs decode + post-processing.
- **Gates (stored per tile):**
  - LACE features' cosine against the deployed `feat_cache/hunt_z4p_test` cache.
  - LACE masks' RLE identity and IoU against the deployed D1 test masks.
  - SAM masks' IoU against the paper's SAM 3 run (`feat_cache/sam3_d1_test`).
  - These prove the timed code is the code that produced the paper's numbers.
- **Report:** per-tile median and range of s/tile for each component, the ratio, and parameter counts (`results/summary.json` → env).

## Commands

```
cd boxinst_commonality_tcd_04/masker_cost; export PYTHONPATH=/Users/tompitts/dphil/CanopyAI2
../../.venv/bin/python -u bench.py sam 1                                   # resumes by listing
/Users/tompitts/dphil/feat_cache/venv_feat_t457/bin/python -u bench.py lace 1
../../.venv/bin/python -u bench.py sam 2
../../.venv/bin/python -u summarise.py
```

## Layout

- `bench.py`, `summarise.py`
- `results/{selection,summary}.json`, `results/{sam_p1,lace_p1,sam_p2}/<tid>.json` (+ `_env.json`)
- `logs/`: `{sam_p1,lace_p1,sam_p2,summarise}.log`, plus two invalid attempts kept for the record (`bench_attempt1_venv_drift.log`, `sam_p1_attempt2_manifest_keyerror.log`)
