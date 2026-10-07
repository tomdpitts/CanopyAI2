# Native-raster campaign — closing the 512² objection and two config debts

**Started 2026-09-04.** One campaign, so that no row is re-run twice.

## The question

Table 1 scores every mask at 512², a 4× downsample of the 0.1 m/px imagery. A reviewer can
say: *the coarse raster flatters your 8 px-lattice masker.* They would be right about the
direction. [`ablation/results/raster_quantum.json`](../ablation/results/raster_quantum.json)
(T1.5) already measured the size of it model-free, from GT polygons alone:

| raster | IoU after a ONE-pixel boundary error | frac ≥ 0.9 |
|---|---|---|
| 512² | median 0.71 | 1.3% |
| 2048² | median 0.90 | 51.4% |

> *Mean IoU and IoU≥0.5 are largely unaffected; IoU≥0.75 is compressed but still
> discriminative; IoU≥0.9 should be reported with this caveat or not at all.*

So **AP₅₀ is the robust column and AP₅₀:₉₅ is the exposed one**, via its 0.75+ tail.

## Why 512 was chosen (for the record)

It was never argued for. `RES = 512` entered in `723dde9` (2026-07-16), the first commit that
built the EM model, as the pipeline's scoring constant; every result since was produced at it.
By the time `PROTOCOL_439.md` was written the question was "do we re-base everything?", and the
entry reads `DECIDED: keep 512²`. That entry justifies itself partly with a ~49 GB / ~780 GB
memory wall — which is real but belongs to `phase4_lib_tcd.eval_selfmask`, which accumulates
DENSE masks for all 439 tiles. It does **not** apply to `score_coco.py`, which is RLE end to end
and handles 2048 fine. The true cost of the switch was always "re-run the baselines", a budget
argument rather than an impossibility. State it that way.

## Scope decision: sensitivity, NOT a protocol change

Re-basing the frozen protocol to 2048 would move every AP₅₀:₉₅ in the paper, including the
ablation sections (`PROTOCOL_439.md` open item says exactly this). Instead: a **self-contained
2048 block** in which every row is at the same raster. It answers "does the lead survive at
native resolution?" without touching the frozen table or any ablation number.

## What needs a GPU, and what does not

No row needs **retraining**. Checkpoints and features verified present on the volumes
2026-09-04.

| row | 2048 masks | cost | also closes |
|---|---|---|---|
| LACE s0–s2 | re-render from cached features | **CPU only** | — |
| DetecTree2 | re-run predict | GPU (needed anyway) | CONFIG_AUDIT §3 undisclosed RPN/dets deviation |
| Box2Mask ×2 | re-stitch raw 1024 subtiles | **no inference at all** | — |
| Restor | re-run predict | GPU | — |
| SelvaBox → SAM 3 | re-run | GPU | CONFIG_AUDIT §5 aggregator-2 omission; missing raw preds |

### Expected direction

**Restor up, LACE down** — both against our margin, which is why it is being measured rather
than asserted. The 512 raster quantises away detail finer than 4 tile px. Restor's 28×28 ROI
mask is finer than that for ~90% of crowns, so at 512 detail it genuinely produced is erased.
LACE's 8 px lattice is 2× *coarser* than the 512 raster, so it has nothing to recover and its
blockiness is exposed against a now-smooth GT. The one direct measurement (`tab:gtbox`, oracle
boxes, mean per-crown IoU) agrees: LACE 0.786 → 0.770 (−0.016), SAM 3 zero-shot 0.749 → 0.741
(−0.009). The SelvaMask-FT arm went *up* (0.605 → 0.623), because it is resampled from a 1777 px
tile — a reminder that the resampling path can dominate the boundary-quality effect, and why no
number is predicted for Restor here.

Note mean IoU does **not** transfer to AP₅₀: a crown only changes the AP₅₀ count if it crosses
IoU 0.5.

## Why LACE must be re-rendered rather than upsampled

`evaluate.pred_instance_masks` bilinearly resizes the 256-cell posterior grid **directly to
`res`** and thresholds afterwards. The 2048 mask is therefore not the 512 mask upsampled — the
interpolation happens at the target resolution and the thresholded boundary genuinely differs.
Nearest-upsampling the stored 512 RLE would publish a mask the model never emitted.
`score_coco.build_dt` refuses a size mismatch for exactly this reason.

## The persistence rule (why this campaign exists at all)

Every pass in this campaign MUST write:

1. masks as RLE at **native** resolution — never only the downsample;
2. **raw per-subtile** predictions, before stitching;
3. every **per-instance score** the model emits.

With those three, raster / score floor / ranking key / canopy rule / stitching are permanently
free CPU re-scores. The reason this campaign is expensive is that earlier passes persisted the
cheap summary instead of the expensive artefact —
`restor_modal.py:195` generates masks at full 2048 and throws them away, third occurrence of
the same failure.

## Run log

| date | what | result |
|---|---|---|
| 2026-09-04 | GT rasterisation at 2048 smoke-tested locally | 2.0 s / 5 tiles, 2.5 GB peak on the densest tile (422 crowns) → ~3 min for 439. Tractable. |
| 2026-09-04 | 2048 render benchmarked on real 4-phase features | 6.2 ms/det at 2048 vs 0.8 ms at 512 → ~16 min/seed compute |
| 2026-09-04 | `lace_masks_modal.py::regen --seed 0 --limit 3` | OK. `masker s=8`, 498 dets / 3 tiles / 22 s |
| 2026-09-04 | `regen --seed 0` full 439 | OK. 159,687 dets (= the published count exactly), 5,385 s, 41 MB, no projection warnings |
| 2026-09-04 | scored both arms, `--score_floor 0` | see below |
| 2026-09-07 | tau re-selected on 108 val tiles, both rasters (local M4) | 512 optimum 0.25 (reproduces Modal to 4 dp); 2048 optimum **0.40** |
| 2026-09-07 | `lace_multitau_modal.py::render --seed 0`, tau 0.35 + 0.40 @2048 | OK. Preempted once, **resumed from checkpoints**: 439/439 tiles, 0 duplicates, 0 missing, 159,687 dets in each tau file (= published count exactly), 0 failed tiles |

### RESULT — LACE seed 0 on the 439 at a raster-calibrated tau

| raster | tau | AP50 | AP75 | AP50:95 |
|---|---|---|---|---|
| 512 (published) | 0.25 | 0.6615 | 0.1718 | 0.2821 |
| 2048 | 0.25 | 0.6204 | 0.1481 | 0.2559 |
| 2048 | 0.35 | 0.6312 | **0.1463** | **0.2611** |
| 2048 | **0.40** (val-selected) | **0.6324** | 0.1397 | 0.2577 |

Re-calibrating tau recovers **+0.0120 AP50** on the test set, against **+0.0109 predicted** from the
108 validation tiles -- the extrapolation held to about one part in a thousand, which is a
meaningful check on the select-on-val / report-on-test rule.

**The raster therefore costs LACE -0.0291 AP50, not -0.0411.** The extra -0.012 in the first
measurement was tau miscalibration, not the raster.

Box AP50 is 0.6748 in BOTH tau arms, identical to the tau=0.25 arm, as it must be -- boxes are
copied verbatim and box AP cannot depend on the mask threshold. A useful internal consistency
check that the pipeline is doing what it claims.

The AP50 / AP50:95 disagreement seen on validation reappears on test: AP50 peaks at tau=0.40,
AP50:95 and AP75 at tau=0.35. The pre-registered criterion is AP50, so **tau=0.40 stands** and
the disagreement is disclosed rather than resolved by choosing the flattering metric.

**Still not comparable to Restor.** Restor's 0.6255 is a 512 figure. LACE@2048 (0.6324) against
Restor@512 (0.6255) is a meaningless comparison and must never be made. Whether the ordering
survives requires Restor re-run at 2048.

### RESULT — LACE seed 0, 512 vs native 2048

Both arms score the **identical** 159,687 detections; only the mask raster (and, with it, the
GT raster) differs. The 512 arm reproduces the published seed-0 figure `0.6615` exactly
(`tab:factors`, row `s·p̄·m`), which validates the path.

| metric | 512 | 2048 | Δ |
|---|---|---|---|
| **mask AP₅₀** | 0.6615 | **0.6204** | **−0.0411** |
| mask AP₇₅ | 0.1718 | 0.1481 | −0.0237 |
| mask AP₅₀:₉₅ | 0.2821 | 0.2559 | −0.0262 |
| box AP₅₀ | 0.6535 | 0.6748 | **+0.0213** |
| box AP₇₅ | 0.1284 | 0.1994 | +0.0710 |
| box AP₅₀:₉₅ | 0.2546 | 0.2962 | +0.0416 |

Artifacts: `lace_s0_r512_control_scored.json`, `lace_s0_r2048_scored.json`,
`preds_lace_s0_r2048.json`.

**The box row is the key to reading this.** Predicted boxes are copied verbatim and cannot
change, yet box AP rises sharply at 2048. The reason is that COCO GT boxes are derived from the
GT mask *at the scoring raster* (`maskUtils.toBbox`), so at 512 the ground truth itself is
quantised to a 4-tile-px grid and is measurably wrong. Moving to 2048 therefore does two things
at once:

1. **the GT gets sharper** — helps every row, and shows up as the box-AP gain;
2. **predicted masks are measured more strictly** — hurts coarse maskers, and shows up as the
   mask-AP loss.

So the −0.041 mask AP₅₀ is **not** a like-for-like penalty applied to LACE alone; it is what
happens when a harder, truer GT replaces a blurred one. AP₅₀ was NOT robust to the raster after
all, contrary to what T1.5's GT-only quantum analysis suggested — T1.5 measured the cost of a
one-pixel error at each raster, not the change in GT fidelity, and the second effect turns out
to matter more.

**What this does and does not license.** It does not license comparing LACE@2048 (0.6204)
against Restor@512 (0.6255) — that comparison is meaningless and must never be made. Every row
must be at the same raster. Whether the Table 1 ordering survives is **unknown until Restor and
DetecTree2 are re-run at 2048**, because a drop of this kind is expected to apply to every row;
what matters is only the *relative* change. Revise the earlier expectation accordingly: the
prediction "Restor up, LACE down" was half right at best — the gtbox measurement had *both*
LACE (−0.016) and SAM 3 (−0.009) losing mean IoU at 2048, so a uniform drop that preserves
ordering is at least as likely as a reversal.

### RESULT — tau re-selected on the 108 val tiles (2026-09-07)

Selection rule fixed before the run: sweep tau at the deployed (alpha, kappa) = (0.3, 1.6),
criterion canopy-neutral mask AP50 on the 108 val tiles. alpha/kappa frozen (see
`val_knobs_local.py` for the mechanism argument). Run on the M4 Max; the 512 arm reproduces
Modal's numbers **to 4 dp on every overlapping cell**, and the detector emits 40,058 dets in
both environments, so the local arm is a valid stand-in.

| tau | AP50 @512 | AP50 @2048 |
|---|---|---|
| 0.15 | 0.6242 | 0.5772 |
| 0.20 | 0.6339 | 0.5875 |
| **0.25** | **0.6392** ← 512 optimum | 0.5936 |
| 0.30 | 0.6387 | 0.5990 |
| 0.35 | 0.6339 | 0.6032 |
| **0.40** | 0.6234 | **0.6045** ← 2048 optimum |
| 0.45 | 0.6021 | 0.6029 |
| 0.50 | 0.5779 | 0.5944 |
| 0.55 | 0.5355 | 0.5767 |
| 0.60 | 0.4649 | 0.5457 |

**The optimum moves 0.25 -> 0.40.** Both peaks are interior to the grid (`best_is_grid_edge`
false in both arms), so each is a real optimum rather than a boundary value. This confirms the
hypothesis the median-mask-area shrink suggested: tau=0.25 under-covers at the native raster
because the posterior cut is applied AFTER an 8x bilinear upsample rather than a 2x one.

Recalibrating is worth **+0.0109 val AP50** at 2048 (0.5936 -> 0.6045), i.e. it recovers
roughly a quarter of the -0.041 test-set drop. Extrapolating, LACE on the 439 at 2048 with
tau=0.40 should land near 0.631 against 0.6204 at tau=0.25 -- but that is an EXTRAPOLATION
from validation and must be measured on the 439 before it is quoted.

Two things worth carrying into the paper independently of this campaign:
1. The 512 arm establishes tau=0.25 as the genuine 512 optimum over a 10-point grid. The paper
   currently concedes the validation grid only ever spanned {0.25, 0.30} and that the 0.5->0.25
   move predated the validation protocol. That concession can now be replaced with a
   measurement.
2. AP50 and AP50:95 disagree slightly at 2048: AP50 peaks at 0.40 while AP50:95 peaks at 0.35
   (0.2445 vs 0.2436). Selecting on AP50 is the pre-registered rule and is kept, but the
   disagreement should be disclosed rather than hidden.

Artifacts: `val_knobs_tau_r512_s0_local.json`, `val_knobs_tau_r2048_s0_local.json`.

Note: numpy emits `divide by zero / overflow / invalid value in matmul` RuntimeWarnings on
macOS during `em.project`. The stored features are clean (0 non-finite, max |x| ~0.38) and the
results are bit-identical to Modal's, so these are spurious Accelerate-backend warnings, not a
data or numerics problem.

### Scoring trap hit and recorded

The first pass of both arms was scored with the default `--score_floor 0.05`. These
`preds_knobbed_s*_product.json` files carry the **reranked** key `s·p̄·m` in `scores`, so the
floor was applied to a product of three sub-1 numbers and silently discarded half the
detections (159,687 → 78,568), giving `0.6461` instead of `0.6615`. This is exactly the trap
`PROTOCOL_439.md` documents. **Any scoring of a `*_product.json` file must pass
`--score_floor 0`** — the floor already ran upstream, on the detector score.

## Isolation

All writes land under `tcd04-phase4-vol:out/native_raster/` — a directory that did not exist
before this campaign (verified). Nothing existing is overwritten or deleted.

- `out/native_raster/src/lace_s{N}.json` — staged copies of the published 512 detections
- `out/native_raster/preds_lace_s{N}_r2048.json` — the regenerated native-raster predictions

### RESULT — Box2Mask ×2 re-stitched at native 2048 (2026-09-07)

**No inference was re-run.** Box2Mask emits its masks at 1024 per 1024² subtile, i.e. already at the native 0.1 m/px raster; the published 512 row threw that away in `_to_512_canvas`'s 1024→256 downsample. Re-stitching the persisted raw subtile predictions onto a 2048² canvas is exact: at `res=2048`, `SCALE=1.0`, the subtile resize becomes 1024→1024 and the origins {0,512,1024} place directly. Verified rather than assumed — a random bool subtile round-trips through `_to_canvas` bit-identically and lands at exactly its origin.

#### Provenance

| item | value |
|---|---|
| raw preds (R-50) | `tcd04-baselines-vol:boxinst/pred_raw_box2mask_s0_test/` — **3951** JSON files downloaded, **3951** verified locally |
| raw preds (Swin-L) | `tcd04-baselines-vol:boxinst/pred_raw_box2mask_swin-l_s0_test/` — **3951** downloaded, **3951** verified |
| RLE size in every raw instance | `[1024, 1024]` (checked over 12,541 instances across 200 files; no other size present) |
| download form | `modal volume get <vol> <dir>/ <dest>/` — trailing-slash directory form. The deprecated `/*` glob suffix silently writes a 0-byte file; file **counts** were checked afterwards, not just exit status |
| local cache | `/Users/tompitts/dphil/feat_cache/b2m_raw/{r50,swinl}/` — outside the repo |
| stitch params | `iou_thr=0.7, cont_thr=0.85, min_score=0.05` — read from each published file's `meta.dedup`, not assumed |
| tile list | `sorted(test_gt.json)` = the same 439, identical to the Modal `stitch` entrypoint |
| score floor | `0.05`, read from `protocol.score_floor` in the published scored JSONs |

The floor is a **no-op** here and cannot repeat the `*_product.json` trap: `min_score=0.05` already ran at stitch time on the raw detector score, so the stored `scores` have min exactly 0.05 and 0 detections fall below it. The ranking key is the raw detector confidence, not a rescaled product.

#### Environment

Local, CPU-only, no GPU: MacBook Pro M4 Max (16 cores, 64 GiB), macOS 26.6.2 (`macOS-26.6.2-arm64`), `.venv` Python **3.10.20**, numpy **2.2.6**, Pillow **12.3.0**, pycocotools **2.0.11**.

Note this differs from the Modal image that produced the published files, which pins **numpy 1.23.5** (required by torch 1.11's ABI) and takes unpinned `pycocotools`/`Pillow`. The gate below is what makes the difference immaterial for this code path; **the exact Pillow and pycocotools versions inside the Modal image were not recorded and were not looked up.**

#### THE 512 REPRODUCTION GATE — **PASS**

Both backbones were re-stitched locally at `RES=512` with the published parameters and compared against the published Modal outputs.

| | R-50 | Swin-L |
|---|---|---|
| tile sets equal | ✅ 439 = 439 | ✅ 439 = 439 |
| detections compared | 117,027 | 138,820 |
| tiles with a per-tile **count** mismatch | **0** | **0** |
| tiles with any **box** mismatch | **0** | **0** |
| tiles with any **score** mismatch | **0** | **0** |
| tiles with any RLE **`counts` string** mismatch | **0** | **0** |
| tiles with any RLE **`size`** mismatch | **0** | **0** |
| verdict | **byte-identical** | **byte-identical** |

So the local environment reproduces the Modal stitch exactly, despite the numpy 1↔2 difference, and the 2048 run was allowed to proceed locally. What this gate does **not** prove: it exercises only the stitch code path (RLE decode, PIL BILINEAR resize, boolean dedup, RLE encode). It says nothing about local-vs-Modal equivalence for any GPU or BLAS path.

#### Results — canopy-neutral (crowd) arm, seed 0, 439 tiles

| backbone | raster | mask AP₅₀ | mask AP₇₅ | mask AP₅₀:₉₅ | box AP₅₀ | n_det |
|---|---|---|---|---|---|---|
| Box2Mask-T R-50 | 512 (published) | 0.3865 | 0.0509 | 0.1299 | 0.4707 | 117,027 |
| Box2Mask-T R-50 | **2048** | **0.4332** | **0.0704** | **0.1566** | **0.4972** | 118,102 |
| Box2Mask-T Swin-L | 512 (published) | 0.5396 | 0.0884 | 0.1980 | 0.5821 | 138,820 |
| Box2Mask-T Swin-L | **2048** | **0.5625** | **0.1272** | **0.2309** | **0.6017** | 139,328 |

Supporting arms at 2048 — R-50: `canopy_fp` AP₅₀ 0.2158, `canopy_neutral_legacy` AP₅₀ 0.4334. Swin-L: `canopy_fp` AP₅₀ 0.2909, `canopy_neutral_legacy` AP₅₀ 0.5626. Crowd and legacy agree to ≤0.0002 at AP₅₀ in both, as the protocol says they must.

**Both backbones go UP at 2048, on every metric.** That is the opposite sign to LACE (−0.0291 AP₅₀ at its re-calibrated tau) and is consistent with the mechanism already argued in this file: 2048 makes the GT sharper (helps everyone; visible in the box-AP gain, +0.027 R-50 / +0.020 Swin-L on boxes that are *not* copied verbatim here — see the caveat below) *and* measures predicted masks more strictly (hurts coarse maskers only). Box2Mask's masks are genuinely at 0.1 m/px, so it collects the first effect and pays little of the second; LACE's 8 px lattice is coarser than even the 512 raster, so it pays the second and collects less of the first. The AP₇₅ column is where this is starkest: Box2Mask-T Swin-L AP₇₅ rises +0.039 (0.0884 → 0.1272) while LACE's falls.

#### Caveats — what limits interpretation

1. **This is NOT a "same detections, finer masks" pairing.** `clean_crowns` dedup operates on masks, so at a different raster it keeps or drops marginally different instances. Measured drift: R-50 117,027 → 118,102 (**+1,075, +0.92%**), Swin-L 138,820 → 139,328 (**+508, +0.37%**); only 74/439 (R-50) and 84/439 (Swin-L) tiles have an identical detection count, and per-tile deltas run −6…+13 and −10…+16. The 2048 row is *re-stitched at native*, not *the same row re-measured*. Some of the AP gain is therefore attributable to a slightly different (and slightly larger) detection set, and this analysis does **not** separate that contribution from the mask-fidelity contribution. Because the boxes are derived from the mask via `_bbox`, the box AP₅₀ column moves for the same reason and is **not** the clean invariant it is in the LACE arm.
2. **Do not cross-compare rasters between rows.** Box2Mask-T Swin-L @2048 (0.5625) against LACE @2048 (0.6324, τ=0.40) is legitimate — same raster, same GT rasterisation, same scorer. Against LACE @512 (0.6615), or against Restor @512 (0.6255), it is meaningless. Restor and DetecTree2 have not been re-run at 2048.
3. **AP₅₀:₉₅ carries the T1.5 caveat**, via its ≥0.75 tail; `raster_quantum.json` says IoU≥0.9 should not be read at 512 at all. It is reported here for completeness, not as the criterion.
4. **Not verified**: that the Modal stitch container's Pillow/pycocotools versions match the local ones (only that the outputs are byte-identical); anything about the training or inference of these checkpoints, which was not touched.
5. The R-50 arm's `dets_per_tile_max` was 522 at 512 against a `maxDets=600` budget; at 2048 the detection count rose ~1%, so the budget is still not binding, but a per-tile max was **not** re-computed for the 2048 files.

#### Config debt closed

`preds_box2mask_swin-l_s0_test.json` carries `meta.model = "Box2Mask-T R-50 ..."`, a copy-paste from the R-50 arm (CONFIG_AUDIT open item). The runs are genuinely distinct — 138,820 vs 117,027 detections — so only the label is wrong. **The new file `preds_box2mask_swinl_s0_r2048.json` carries the correct `"Box2Mask-T Swin-L ..."` string. The published 512 file was NOT edited** and still carries the wrong label.

#### Commands

```
# download (verify COUNTS, not exit status -- the deprecated /* glob form writes a 0-byte file)
mkdir -p /Users/tompitts/dphil/feat_cache/b2m_raw/{r50,swinl}
.venv/bin/modal volume get tcd04-baselines-vol boxinst/pred_raw_box2mask_s0_test/        /Users/tompitts/dphil/feat_cache/b2m_raw/r50/
.venv/bin/modal volume get tcd04-baselines-vol boxinst/pred_raw_box2mask_swin-l_s0_test/ /Users/tompitts/dphil/feat_cache/b2m_raw/swinl/

# 512 GATE (both backbones; output compared byte-for-byte to the published files)
.venv/bin/python -m boxinst_commonality_tcd_04.native_raster.run_stitch_b2m \
    --pred_dir <raw dir> --res 512 --min_score 0.05 --workers 12 --model "GATE-512 ..." --out <scratch>.json

# 2048 stitch
.venv/bin/python -m boxinst_commonality_tcd_04.native_raster.run_stitch_b2m \
    --pred_dir /Users/tompitts/dphil/feat_cache/b2m_raw/r50/pred_raw_box2mask_s0_test \
    --res 2048 --min_score 0.05 --workers 14 --model "Box2Mask-T R-50 ...; re-stitched at native 2048" \
    --out boxinst_commonality_tcd_04/native_raster/preds_box2mask_r50_s0_r2048.json
#   ... same with swinl/pred_raw_box2mask_swin-l_s0_test -> preds_box2mask_swinl_s0_r2048.json

# score (floor read from protocol.score_floor of the published scored JSON; GT rasterised at --res)
.venv/bin/python -m boxinst_commonality_tcd_04.score_coco \
    --preds boxinst_commonality_tcd_04/native_raster/preds_box2mask_{r50,swinl}_s0_r2048.json \
    --res 2048 --score_floor 0.05 \
    --out  boxinst_commonality_tcd_04/native_raster/scored_box2mask_{r50,swinl}_s0_r2048.json
```

New code: [`native_raster/stitch_res.py`](stitch_res.py) — a RES-parametrised copy of `detectree2_baseline/stitch.py`, algorithmically identical, written as a NEW file so the published stitcher (which produced published numbers) is untouched. [`native_raster/run_stitch_b2m.py`](run_stitch_b2m.py) — driver; shards the independent per-tile loop over a process pool and writes a per-tile JSONL checkpoint it resumes from (used in anger: a killed shell lost only the tiles in flight). Both 512 gate arms were produced with the pooled+checkpointed driver, so the gate covers the parallel path, not just the algorithm.

Cost: 272 s (R-50) and 283 s (Swin-L) at 2048 on 14 workers; peak RSS stayed far below the ~3.6 GB/tile worst case, and memory was never a constraint. No GPU, no Modal compute.

Artifacts (all NEW files; nothing existing was written to or deleted — checked before each write):
`preds_box2mask_r50_s0_r2048.json`, `preds_box2mask_swinl_s0_r2048.json`,
`scored_box2mask_r50_s0_r2048.json`, `scored_box2mask_swinl_s0_r2048.json`.

### Sidebar — why box AP rises at 2048 for every row (measured 2026-09-09)

The earlier text says the 512 GT box is "quantised to a 4-tile-px grid". Measured, it is not jitter but a **one-sided bias**: `maskUtils.toBbox` counts whole pixels, a crown's boundary only partially covers its edge pixels but they count fully, so the derived GT box rounds *outward* by ~1 pixel per dimension — 4 tile-px at 512, 1 tile-px at 2048. Model-free, from GT polygons alone (2,601 crowns / 60 tiles, seed 0 sample; derived GT box vs the exact polygon bbox):

| | 512 | 2048 |
|---|---|---|
| mean IoU, derived GT box vs true bbox | 0.8129 | 0.9441 |
| median IoU | 0.8432 | 0.9584 |
| frac ≥ 0.75 | 0.745 | 0.998 |
| mean signed width error (tile-px) | **+3.98** (std 1.70) | **+0.96** (std 0.42) |
| mean IoU, crowns < 20 tile-px | 0.62 | 0.87 |
| mean IoU, crowns > 160 tile-px | 0.96 | 0.99 |

The arithmetic closes: a tight prediction on a median 45 tile-px crown against a GT box dilated by 4 gives 45²/49² = 0.843; measured median 0.843. At 2048, 45²/46² = 0.957; measured 0.958. LACE's predicted boxes are byte-identical in 439/439 tiles between its 512 and 2048 arms, so its +0.0213 box AP₅₀ is *entirely* the GT getting truer. **Every box AP₅₀ in the 512 table is depressed by this artifact, roughly uniformly.**

### RESULT — Restor Mask R-CNN (rpn1000) re-run at native 2048 (2026-09-22)

The one GPU row in the campaign so far. `restor_baseline/restor_modal.py:193-197` generated masks at 2048 and downsampled them away; raw preds were never persisted, so this is a re-run, not a re-stitch. New script [`native_raster/restor_native_modal.py`](restor_native_modal.py): inference **identical** to the published rpn1000 arm (whole 2048 tiles, no resize, `SCORE_THRESH 0.05`, `DETECTIONS_PER_IMAGE 600`, `RPN PRE/POST_NMS_TOPK_TEST 1000`, all classes emitted with `pred_classes`); the only change is that the (2048,2048) mask is RLE-encoded as-is, **and** the 512 downsample is also written from the same forward pass with the identical PIL call, as the gate.

RPN arm: **1000** (Tom's call, 2026-09-22). Not the released cap (512) and not 2000.

#### Provenance

| item | value |
|---|---|
| Modal app | `tcd04-restor-native`, run `ap-479D83lqoumhKb1WKdXXWH`, detached |
| volume path (NEW, verified absent before the run) | `tcd04-baselines-vol:out/native_raster/restor_rpn1000_r2048/` |
| per-tile files | `tiles/<tid>.json` × **439**, atomic write, `vol.commit()` every 10 tiles, resumable (not needed: single container, no preemption) |
| container env | torch 2.1.0, detectron2 0.6, numpy 1.26.0, Pillow 10.0.1, CUDA 12.1, A100-SXM4-40GB |
| checkpoint | `restor/tcd-mask-rcnn-r50` `model.pth` + `config.yaml` from HF, as published |
| wall | ~1 h on one A100 (from polling cadence; not timed in-container) |
| copied into repo same session | `preds_restor_rpn1000_r2048.json` (32 MB), `preds_restor_rpn1000_r512_incontainer.json` (15 MB), `restor_rpn1000_r2048_run.log` |
| derived (CPU) | `preds_restor_rpn1000_r2048_treeonly.json` = `pred_classes == 1` filter, which is exactly how the published `preds_restor_rpn1000_treeonly.json` was made (verified 439/439) |

#### Gates — both **PASS, byte-identical**, 79,201 detections, 439 tiles

| gate | compares | count | scores | boxes | classes | RLE | verdict |
|---|---|---|---|---|---|---|---|
| **A** GPU determinism | in-container 512 arm vs published `preds_restor_rpn1000.json` | 0 | 0 | 0 | 0 | 0 mismatching tiles | **PASS** |
| **B** local Pillow | local decode(2048 RLE)→PIL BILINEAR≥128 vs in-container 512 | 0 | 0 | 0 | 0 | 0 | **PASS** |

A: the A100 re-run reproduced the published detections, scores, boxes and masks exactly (first-10-tile class counts also matched mid-flight, 1092/355). B: local Pillow 12.3.0 and container Pillow 10.0.1 give identical downsamples, which retroactively closes the Box2Mask "Pillow version not verified" item. The 2048 row therefore inherits the published row's provenance in full.

#### Results — canopy-neutral (crowd), 439 tiles, floor 0.05

| variant | raster | mask AP₅₀ | mask AP₇₅ | mask AP₅₀:₉₅ | box AP₅₀ | box AP₇₅ | n_det |
|---|---|---|---|---|---|---|---|
| tree-only | 512 (published) | 0.6255 | 0.1872 | 0.2766 | 0.6536 | 0.2483 | 56,438 |
| tree-only | **2048** | **0.6480** | **0.2968** | **0.3332** | 0.6646 | 0.3409 | 56,438 |
| pooled | 512 (published) | 0.6137 | 0.1893 | 0.2738 | 0.6392 | 0.2463 | 79,201 |
| pooled | **2048** | **0.6342** | **0.2946** | **0.3279** | 0.6490 | 0.3364 | 79,201 |

`canopy_fp` AP₅₀ at 2048: tree-only 0.5646, pooled 0.5296. Legacy arm agrees with crowd to ≤0.0010.

**This is a clean "same detections, finer masks" pairing** — the thing Box2Mask could not offer. Detections, scores, boxes and classes are byte-identical across rasters (Gate A), so every delta is the raster and nothing else: mask AP₅₀ **+0.0225** (tree-only), AP₇₅ **+0.1096** (0.1872 → 0.2968, ×1.6), box AP₅₀ +0.0110 (pure GT-fidelity, per the sidebar).

#### Head-to-head with LACE at native 2048 — **the ordering flips**

| raster | LACE (s0) | Restor rpn1000 tree-only | LACE − Restor |
|---|---|---|---|
| 512, mask AP₅₀ | 0.6615 (τ=0.25) | 0.6255 | **+0.0360** |
| 2048, mask AP₅₀ | 0.6324 (τ=0.40) | 0.6480 | **−0.0156** |
| 2048, mask AP₇₅ | 0.1397 | 0.2968 | −0.1571 |
| 2048, mask AP₅₀:₉₅ | 0.2577 | 0.3332 | −0.0755 |
| 2048, box AP₅₀ | 0.6748 | 0.6646 | +0.0102 |

At 512 LACE leads Restor-rpn1000 by 0.036; at native resolution Restor leads by 0.016, and by a wide margin on every stricter metric. The raster costs LACE −0.029 and gives Restor +0.023; the swing (0.052) exceeds the 512 margin. The box column stays in LACE's favour (+0.010), i.e. **the detector holds, the masker does not**: at a truthful raster LACE's 8 px lattice is measurably worse than Restor's 28×28 ROI paste. This is the campaign's question answered, and the answer goes against the paper's current framing. Note it is seed 0 only for LACE (3-seed 512 mean is 0.6300 ± 0.0054, which would already sit *below* 0.6480 — and below Restor-rpn1000's 512 figure is a separate, not-yet-made comparison the seed band would need).

#### Caveats

1. rpn1000 is neither Restor's released config (512, 0.5706@512) nor the near-ceiling 2000 (0.6605@512). Persisted at 2048 for rpn1000 only; the other arms need their own A100 pass each (~1 h).
2. LACE at 2048 is seed 0 with a val-selected τ; Restor has no tunable knob here and no seed band. A 3-seed LACE@2048 would tighten the flip but cannot plausibly reverse a 0.016 gap given the ±0.005 seed band at 512.
3. DetecTree2 and SAM 3 / SelvaBox still un-run at 2048. The Table 1 ordering at native raster is therefore known for LACE vs Restor vs Box2Mask only.
4. Not verified: pycocotools version inside the container (`__version__` absent); only that its RLE output matches local 2.0.11 byte-for-byte (Gate B).

#### Commands

```
cd boxinst_commonality_tcd_04/native_raster
../../.venv/bin/modal run --detach restor_native_modal.py::predict        # A100, ~1 h
../../.venv/bin/modal run          restor_native_modal.py::assemble       # CPU, writes the two preds files
modal volume get tcd04-baselines-vol out/native_raster/restor_rpn1000_r2048/preds_restor_rpn1000_r2048.json  boxinst_commonality_tcd_04/native_raster/
modal volume get tcd04-baselines-vol out/native_raster/restor_rpn1000_r2048/preds_restor_rpn1000_r512_incontainer.json boxinst_commonality_tcd_04/native_raster/
.venv/bin/python -m boxinst_commonality_tcd_04.native_raster.restor_gate_derive --dir boxinst_commonality_tcd_04/native_raster
.venv/bin/python -m boxinst_commonality_tcd_04.score_coco --preds .../preds_restor_rpn1000_r2048.json          --res 2048 --score_floor 0.05 --out .../scored_restor_rpn1000_r2048_pooled.json
.venv/bin/python -m boxinst_commonality_tcd_04.score_coco --preds .../preds_restor_rpn1000_r2048_treeonly.json --res 2048 --score_floor 0.05 --out .../scored_restor_rpn1000_r2048_tree.json
```

Artifacts (all NEW; nothing published modified): `preds_restor_rpn1000_r2048.json`, `preds_restor_rpn1000_r512_incontainer.json`, `preds_restor_rpn1000_r2048_treeonly.json`, `scored_restor_rpn1000_r2048_{pooled,tree}.json`, `restor_rpn1000_r2048_run.log`, `restor_native_modal.py`, `restor_gate_derive.py`.

---

## FULL RE-BASE DATA PASS — Phase A (2026-09-23 → 2026-09-26)

Plan: `~/.claude/plans/ok-if-we-needed-merry-treehouse.md` (approved 2026-09-23). **Data only; `paper.tex` untouched; no protocol decision taken.** Every output is a new file; every regenerated row gated against its published 512 file before its 2048 number was recorded. Consolidated table: [`results_439/r2048/table_r2048.md`](../results_439/r2048/table_r2048.md) (built by `results_439/build_table_r2048.py` from scored JSONs only).

### Rows produced and their gates

| row | how | gate (vs published 512) | 2048 artefact |
|---|---|---|---|
| LACE s1, s2 on the 439 | Modal CPU, `lace_multitau_modal.py --seed {1,2}` (τ 0.35 + 0.40), spawned via `detach.py` | boxes + scores byte-identical to `confidence/preds_knobbed_s{1,2}_product.json` on 439/439 tiles; in-container render path asserted equal to `evaluate.pred_instance_masks` before the run | `preds_lace_s{1,2}_r2048_tau0{35,40}.json` (raw JSONL in `lace439_r2048_raw/`) |
| LACE sparse-236 s0/s1/s2 | Modal CPU, new `lace_masks_sparse_modal.py` (τ 0.25 + 0.40) | boxes + scores byte-identical to `modal_sparse_tcd_multiseed/preds/ours_prod_s{0,1,2}.json`, 236/236 | `preds_lace_sparse_s{0,1,2}_r2048_tau0{25,40}.json` (raw in `sparse_r2048_raw/`) |
| Restor rpn512 (released cap), rpn2000 | Modal A100, `restor_native_modal.py --rpn-topk {512,2000}` | **Gate A PASS byte-identical**: rpn512 container-512 = `preds_restor_allcls.json` (61,182) and its tree filter = `preds_restor.json` (42,987); rpn2000 tree filter = `preds_restor_rpn2000.json` (71,606) | `preds_restor_rpn{512,2000}_r2048{,_treeonly}.json`, `..._r512_incontainer.json` |
| DetecTree2 s0v2, 439 | local CPU re-stitch of `tcd-detectree2-vol:out/pred_raw_s0v2` (3,951 files) | **PASS byte-identical** 512 re-stitch = `detectree2_baseline/preds_dt2_s0v2.json` (196,927) | `preds_dt2_s0v2_r2048.json` (197,423 dets) |
| DetecTree2 s0v2, sparse-236 | local CPU re-stitch of `out/pred_raw_sparse_s0v2` (2,124 files) | **PASS byte-identical** vs `modal_sparse_tcd_multiseed/preds/dt2_sparse_s0v2.json` (99,558) | `preds_dt2_sparse_s0v2_r2048.json` (99,900) |

Scoring: `score_coco.py --res 2048`; `--score_floor 0` for every LACE `*_product` file, 0.05 for every baseline (their stored scores are raw detector scores already floored at 0.05).

**Stitcher change, re-gated.** `stitch_res.dedup` now ANDs only the two boxes' overlap window instead of the whole raster — the same integer, since pixels outside that window are False in at least one mask. Needed because a dense DetecTree2 tile (~3,000 raw crowns) took ~5 min at 512 with the full-raster AND. After the change, **Box2Mask R-50 512 re-stitch is still byte-identical** to the published file (117,027 dets) and the DetecTree2 gates above ran on it. The 2048 Box2Mask files predate the change; they were not regenerated (identical algorithm, proven equal at 512).

### Headline numbers (canopy-neutral crowd, mask AP50; compare within a column only)

| row | 512 | 2048 |
|---|---|---|
| **LACE 3-seed** (τ 0.25 @512, 0.40 @2048) | **0.6625 ± 0.0012** | **0.6300 ± 0.0033** |
| Restor rpn2000 tree-only | 0.6605 | **0.6846** |
| Restor rpn1000 tree-only | 0.6255 | 0.6480 |
| DetecTree2 s0v2 | 0.5969 | 0.6143 |
| Restor rpn512 (released) tree-only | 0.5706 | 0.5847 |
| Box2Mask-T Swin-L | 0.5396 | 0.5625 |
| Box2Mask-T R-50 | 0.3865 | 0.4332 |
| SelvaBox → SAM 3 | 0.5687 | **not produced** |
| *sparse-236:* LACE 3-seed | 0.6913 ± 0.0095 | 0.6670 ± 0.0112 |
| *sparse-236:* DetecTree2 s0v2 | 0.6118 | 0.6252 |

Bands are ddof=1 (repo convention). At 2048 the LACE seed band is 0.0033, so every ordering in the 439 block is outside seed noise.

**What moves.** Every non-LACE row rises at 2048 (+0.013 to +0.047); every LACE seed falls (−0.029 to −0.038 on the 439, −0.023 to −0.027 on sparse). At 512 LACE is first on the 439; at 2048 it is third, behind Restor rpn1000 and rpn2000, and ahead of DetecTree2, released Restor and both Box2Mask. On the sparse slice LACE still leads DetecTree2 at 2048 (+0.042, down from +0.080). The AP75 column is where LACE is weakest at 2048 (0.1315 ± 0.0103 vs 0.25–0.35 for every Mask R-CNN row). Box AP50 at 2048: LACE 0.6666 ± 0.0108, second only to Restor rpn2000 (0.7050).

The sparse `s0 @ τ 0.25 both rasters` row isolates the pure raster effect at a fixed threshold: −0.0283, i.e. almost all of LACE's drop is the raster, not the τ change (τ 0.40 recovers +0.005 on sparse s0).

### Not produced — stopped by Modal budget (2026-09-25)

| row | state | resume |
|---|---|---|
| SelvaBox → SAM 3 bench (`selvabox_native_modal.py`) | **145/439** tiles checkpointed on `tcd04-baselines-vol:out/native_raster/selvabox_bench_r2048/tiles/` (masks at 1777, 2048, 512 + raw dets + SAM scores) | `python detach.py spawn tcd04-selvabox-native predict` — resumes from the tile files; then `assemble` |
| SAM 3 box-prompted on LACE s0 boxes (`sam3_native_modal.py`) | **214/439** on `tcd04-phase4-vol:out/native_raster/sam3_s0_r2048/tiles/` | `python detach.py spawn tcd04-sam3-native predict`; then `assemble` |

Both rows need ~1–1.5 A100-h more between them. Nothing already computed is lost.

### Harness incidents (and why nothing was lost)

1. **Detached Modal calls cancelled by the client.** Stopping a Bash task that was attached to `modal run --detach` cancelled the call server-side (LACE s1 at 40/439, LACE s2, Restor rpn512 early). All resumed from per-tile checkpoints with no recomputation. Every launch since goes through `detach.py deploy` + `spawn`.
2. **Orphaned multiprocessing workers** from killed local stitches kept running (macOS `spawn` workers survive the parent), driving load average to 50–80 for hours. Killed with `pkill -f "multiprocessing.spawn import spawn_main"`. The checkpoint JSONLs made every restart resume cleanly; duplicate writers produced duplicate lines only, which the loader de-duplicates by tile id.
3. **One truncated download** (`preds_lace_sparse_s2_r2048_tau025.jsonl`, 26/236 tiles) was caught by the converter's tile-set assertion and re-fetched; the regenerated file passes the pairing gate.

### Not verified in this pass

- No ablation table has been regenerated yet (Phase C). The 2048 numbers above are Table-1 and sparse-236 rows only.
- α, κ remain at their 512 selection (0.3, 1.6); only τ was re-selected at 2048 (Phase B not yet run).
- Restor rpn512 / rpn2000 used the rpn-parametrised script; the rpn1000 arm's files were produced before that edit and were not regenerated.

## PAPER INTEGRATION (2026-09-26)

Decision: **submit at the 512² protocol raster; native 2048 goes in as a sensitivity block, not a re-base.** `paper.tex` now carries:

- `\subsection{Native-raster sensitivity}` (`app:native`) + **Table A6** (`tab:native`): every Table 1 row and both sparse rows at 512 and 2048, columns AP50 / AP50:95 / box AP50 only. Numbers are the 3-dp rounding of `results_439/r2048/table_r2048.md`. SelvaBox 2048 cell is a dash (145/439, not completed).
- Abstract, intro (L247), discussion (L941) and "note of caution" (L944) lessened from "above / highest / strongest biome-agnostic" to "matches at the protocol raster; at native the Mask R-CNN mask head leads while box AP stays level; strongest *box-supervised* model". Table 1 caption points at Table A6.
- Appendix "scoring raster favours coarse maskers" paragraph now states why 512 was the protocol (pipeline constant since the first EM commit; kept because re-basing moves every ablation figure, not because 2048 is intractable).

Two decisions taken by Tom, recorded so they are not re-litigated:
1. **rpn2000 is excluded from the paper altogether.** Reason: it is a tuned arm and the other baselines had no hyperparameter tuning; rpn1000 is Detectron2's FPN default and what Detectree2 inherits, so it is the like-for-like configuration. rpn2000 stays in this file and in `table_r2048.md` as data only.
2. **AP75 appears nowhere in the paper.** Removed from tab:sparse236, tab:scorer, tab:factors, the L944 text and a commented-out paragraph. AP50 and AP50:95 only; the strict-threshold story is carried by AP50:95.

Build verified: `tectonic --synctex --keep-logs paper.tex` exit 0, no undefined refs; Table A6 renders on p.30.

### QUEUED, NOT RUN — oracle-localisation ceiling (2026-10-06)

[`oracle_boxes_val.py`](oracle_boxes_val.py), written and syntax-checked only; it has not been executed. Question: how much mask AP would a better *detector* buy LACE, with the masker fixed? On the 108 val tiles it runs LACE's seed-0 detections once, then re-renders masks twice more at 2048 (τ 0.40). In those arms each prediction matched to a GT crown at box IoU ≥ 0.5 (or ≥ 0.3) gets its box replaced by the GT box. Detection set, count and detector-score ranking are unchanged, so only localisation improves. The output is the mask AP delta versus the as-is arm.

The gate is that the as-is arm must reproduce cell 0.40 of `val_knobs_tau_r2048_s0_local.json` (AP50 0.6045, 40,058 dets), or the run aborts before the oracle arms. It runs locally, with no Modal and an estimated 10–15 min.

    .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.oracle_boxes_val          # 2048
    .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.oracle_boxes_val --res 512

Decision rule agreed in conversation: if the ceiling is under ~0.02 mask AP50, a detector retrain is not worth the GPU time.

### RESULT — oracle-localisation ceiling, 108 val tiles, 2048, τ 0.40 (2026-10-06)

Run locally with a 2-thread BLAS/OMP cap (one P-core sustained, ~7 min per arm). The **gate passed**: the as-is arm reproduces `val_knobs_tau_r2048_s0_local.json` cell 0.40 exactly (AP50 0.6045, 40,058 dets). Output: `oracle_boxes_val_r2048.json`, log `logs/oracle_boxes_val_r2048.log`. Ranking is the detector score s in every arm.

| arm | boxes replaced | their mean box IoU before | mask AP50 | AP75 | AP50:95 | box AP50 |
|---|---|---|---|---|---|---|
| as-is | 0 | — | 0.6045 | 0.1328 | 0.2436 | 0.6338 |
| snap@0.5 | 5,428 | 0.729 | 0.6292 (**+0.025**) | 0.3844 (**+0.252**) | 0.3523 (**+0.109**) | 0.6416 |
| snap@0.3 | 6,212 | 0.688 | 0.7044 (**+0.100**) | 0.4293 (**+0.297**) | 0.3990 (**+0.155**) | 0.7371 |

**Reading.** Localisation, not the masker lattice, is LACE's main limit at strict IoU. With the detections and ranking held fixed, exact boxes nearly triple AP75. This **retracts** the in-conversation claim (2026-10-06, earlier) that AP75 is masker-bound and no detector would close it. snap@0.5 changes only boxes that already count as detections, so its +0.025 AP50 is pure tightening. snap@0.3 also converts 784 near-miss boxes (IoU 0.3–0.5) into detections, so its AP50 gain is mostly recall, as box AP50 0.634 → 0.737 shows.

**Why this is a ceiling, not a forecast.**
1. Exact GT boxes are unattainable, and the masker clips its posterior to the box (`pred_instance_masks` ANDs with the box), so a perfect box also gives a perfect clip. Part of the AP75 jump is that clip, which a real detector only approximates.
2. The snapped boxes start at mean box IoU 0.73. Restor's matched boxes on the 439 average ~0.80 (box_iou_strata), so a Restor-quality detector closes roughly a quarter of the distance to the oracle, not all of it.
3. These are val tiles, seed 0, with detector-score ranking; they are not test numbers and not comparable to any test row.

**Decision rule** (pre-agreed: < ~0.02 AP50 means not worth a retrain): **exceeded** on every metric, even in the conservative snap@0.5 arm. The next cheap step before any GPU spend would be a partial oracle that moves each box toward its GT box until mean box IoU matches Restor's (~0.80), which gives a realistic forecast rather than a ceiling. Not run.

### RESULT — where LACE loses AP50 to Restor at 2048 (test 439, 2026-10-06)

Two analyses, both new files, test set, native raster.

**Per-crown-size recall** ([`small_crowns_test.py`](small_crowns_test.py) → `small_crowns_test_r2048.json`). LACE misses fewer crowns outright than Restor rpn1000 at every size (under 2 m: 34% vs 50%). Its overall mask recall at IoU 0.5 is higher than rpn1000's (0.775 vs 0.744) and below rpn2000's (0.806). At IoU 0.75 it trails by 15–21 points in every size bin. Under 2 m, 20% of LACE's crowns get only a near-miss box (vs 5%), which is why its small-crown masks rarely convert.

**Precision–recall anatomy** ([`fp_anatomy_test.py`](fp_anatomy_test.py) → `fp_anatomy_test_r2048.json`). It uses mask IoU 0.5, one-to-one greedy matching, the legacy canopy-neutral rule, and global score order; it reproduces score_coco AP50 exactly for all three rows.

| | AP50 | max recall | precision @R0.6 | precision @R0.7 | FP share "loc" @R0.6 | FP share "bg" @R0.6 | dup |
|---|---|---|---|---|---|---|---|
| LACE s0 (product) | 0.6324 | 0.775 | 0.730 | 0.564 | **0.29** | 0.71 | 0.003 |
| Restor rpn1000 tree | 0.6480 | 0.744 | 0.783 | 0.667 | 0.16 | 0.84 | 0.003 |
| Restor rpn2000 tree | 0.6846 | 0.806 | 0.788 | 0.688 | 0.16 | 0.83 | 0.003 |

"loc" means the FP overlaps a real crown at mask IoU 0.1–0.5; "bg" means IoU < 0.1 with every crown; "dup" means a second hit on an already-matched crown. **LACE's AP50 deficit is precision, not recall.** At recall 0.6 LACE carries ~1,430 more FPs than rpn1000. About two-thirds are localisation FPs (≈1,650 vs ≈690), which are double-costed because the crown they sit on also stays unmatched; the rest are background. Duplicates are negligible, so NMS is not the problem. LACE's small-box FP share is *lower* than Restor's.

### RESULT — soft box (box as prior, not wall): NEGATIVE, stopped at smoke stage (2026-10-06)

[`softbox.py`](softbox.py) replaces the three hard uses of the box: support is widened by `margin` × box size per side, the spatial prior outside the box decays as exp(−d/λ), and the clip moves to the window. Contrast recentring stays on the original in-box cells. **Gate PASS:** margin 0 reproduces the deployed masks bit for bit (4 val tiles, 240 detections). Runner [`softbox_val.py`](softbox_val.py); smoke test on the first 20 val tiles, 2048, τ 0.40, detector-score ranking → `softbox_val_r2048_lim20.json`.

| arm (margin, λ) | mask AP50 (20 tiles) | Δ | AP75 | Δ |
|---|---|---|---|---|
| deployed | 0.6661 | — | 0.1478 | — |
| 0.10, 0.05 | 0.6557 | −0.010 | 0.1230 | −0.025 |
| 0.10, 0.10 | 0.6500 | −0.016 | 0.1128 | −0.035 |
| 0.20, 0.05 | 0.6382 | −0.028 | 0.0960 | −0.052 |
| 0.20, 0.20 | 0.6047 | −0.061 | 0.0620 | −0.086 |

Every setting loses, and monotonically: the more freedom outside the box, the worse. The median mask grows 9–16% in area at (0.2, 0.2), but the growth is bleed into neighbouring canopy and background, not recovery of the clipped crown. **The box clip is a regulariser LACE depends on.** Outside the box the appearance term cannot tell the target crown from its neighbours (commonality is exactly what neighbouring crowns share), so the box is what stops the mask. This also qualifies the oracle-localisation result: its gain came from the clip being in the right place, which only a better box delivers, not from removing the clip. The full 108-tile grid was not run. Re-boxing's first pass uses the same enlarged window, so it inherits this bleed risk.

### RESULT — training-free decode variants: no gain (2026-10-06)

Seed-0 detector on the 108 val tiles, box AP against GT polygon-extent boxes (the tab:factors "val box AP50" protocol). The baseline reproduces it exactly (0.4982, 40,058 dets), and the re-implemented decode equals `dapt.decode` on every tile.

| decode variant | box AP50 | Δ | box AP75 | Δ |
|---|---|---|---|---|
| baseline (3×3 peak, NMS 0.5, topk 600) | 0.4982 | — | 0.1471 | — |
| topk 1500 (cap binds on 27/108 tiles, max 1,406 peaks) | 0.4988 | +0.0006 | 0.1471 | 0 |
| NMS 0.4 / 0.6 / 0.7 | 0.4967 / 0.4982 / 0.4981 | ≤ 0 | 0.1471 | 0 |
| 5×5 peak window | 0.4917 | −0.0065 | 0.1463 | −0.0009 |
| heatmap-weighted regression, 3×3 | 0.4969 | −0.0013 | 0.1506 | +0.0035 |
| heatmap-weighted regression, 5×5 | 0.4594 | −0.0388 | 0.1123 | −0.0348 |

Nothing in decoding moves box AP50 by more than 0.002. The localisation error is in what the head regresses, not in how it is read out, so the remaining detector levers all need a retrain.

---

## FULL RE-BASE DATA PASS — Phase C, ablations at 2048 (2026-09-26 → 2026-10-06)

All outputs are new files under `ablation/results_r2048/` (plus scored files in `native_raster/`). Published scripts are run **unmodified**, through wrappers in [`ablation_r2048/`](ablation_r2048/) that override only the raster and the input paths. Every wrapper was gated by re-running it at 512 against the published output first. `evaluate.py` was **not** modified: the plan's `TCD04_RES` env-var mechanism was replaced by these wrappers. [`ablation_r2048/rle_io.py`](ablation_r2048/rle_io.py) replaces the dense per-tile loader with an RLE one, and its IoU is **bitwise identical** to `E.mask_iou` (checked on 25 tiles); the integer intersection comes from pycocotools and the same float32 expression is evaluated in the same order.

### tab:scorer — ranking by the posterior product

Detector-ranked 2048 files `preds_lace_{,sparse_}s{0,1,2}_r2048_tau040_detscore.json` were built by [`rescore_detector.py`](rescore_detector.py). The gate: boxes byte-identical and identically ordered to `modal_tcd_multiseed/phase4/preds/knobbed_s*.json` / `modal_sparse_tcd_multiseed/preds/ours_s*.json` on every tile, PASS for all 6. Floor 0.05 for detector-ranked files, 0 for product files. 3 seeds, ddof=1.

| split | ranking | AP50 512 | AP50 2048 | AP75 2048 | AP50:95 2048 |
|---|---|---|---|---|---|
| 439 | s | 0.6300 ± 0.0054 | 0.5950 ± 0.0013 | 0.1047 ± 0.0062 | 0.2299 ± 0.0029 |
| 439 | s·p̄·m | 0.6625 ± 0.0012 | 0.6300 ± 0.0033 | 0.1315 ± 0.0103 | 0.2534 ± 0.0055 |
| sparse | DetecTree2 | 0.6118 | 0.6252 | 0.3541 | 0.3460 |
| sparse | s | 0.6560 ± 0.0121 | 0.6307 ± 0.0136 | 0.1606 ± 0.0131 | 0.2655 ± 0.0080 |
| sparse | s·p̄·m | 0.6913 ± 0.0095 | 0.6670 ± 0.0112 | 0.1965 ± 0.0151 | 0.2918 ± 0.0084 |

The reranker gain survives at 2048: +0.035 on the 439 (+0.0325 at 512) and +0.036 on sparse (+0.035). The 512 `s` row was re-scored locally with score_coco into `scored_lace_s{s}_r512_detscore.json`; per-seed 0.6250 / 0.6358 / 0.6293, matching the published per-seed values.

### tab:factors — confidence/ablation.py, seed 0

The wrapper is [`conf_ablation_res.py`](ablation_r2048/conf_ablation_res.py). **Gate PASS:** at 512 it reproduces `confidence/_ap_cache_s0.pkl` array-for-array on 439/439 tiles and every value in `confidence/ablation_results.json` except `subsets`, which the gate run omitted. The 2048 run used `--full` → `ablation/results_r2048/confidence_ablation_results_r2048.json`.

| ranking | AP50 512 → 2048 | AP75 512 → 2048 | AP50:95 512 → 2048 | val box AP50 |
|---|---|---|---|---|
| s | 0.6250 → 0.5937 | 0.1445 → 0.1109 | 0.2581 → 0.2330 | 0.4982 (unchanged) |
| s·m (drop p̄) | 0.6484 → 0.6180 | 0.1577 → 0.1252 | 0.2719 → 0.2469 | 0.5148 |
| s·p̄ (drop m) | 0.6472 → 0.6178 | 0.1638 → 0.1307 | 0.2737 → 0.2492 | 0.5094 |
| s·p̄·m | 0.6615 → 0.6324 | 0.1728 → 0.1404 | 0.2822 → 0.2580 | 0.5177 |

The product gain is +0.0387 at 2048 (vs +0.0365), and dropping either factor still costs about a third of it. The val box column uses GT polygon extents and is raster-independent. At 2048 the 8-feature logreg **ties** the product at AP50 (0.6319; paired bootstrap −0.0005, 95% CI [−0.0021, +0.0011]) and loses at AP75. s + box-size prior alone gives 0.6198 (+0.0261 of the +0.0387). Note the AP50 here is score_coco's legacy arm, which equals the crowd arm to within 1e-4 (0.5937 vs 0.5936 on the same file).

### tab:strata — t03 --product, 3 seeds

The runner is [`run_strata.py`](ablation_r2048/run_strata.py). **Gate PASS:** at 512 its output is identical to `ablation/results/strata_tile_product.json` on bands, per-seed values and Spearman. Output: `ablation/results_r2048/strata_tile_product_r2048.json`.

| closure bin | tiles | mask AP50 | box AP50 | mask − box (2048) | mask − box (512, paper) |
|---|---|---|---|---|---|
| [0, 0.1) | 196 | 0.5490 | 0.5610 | −0.0120 | +0.023…+0.033 |
| [0.1, 0.25) | 95 | 0.6570 | 0.6653 | −0.0083 | |
| [0.25, 0.5) | 86 | 0.6862 | 0.6900 | −0.0038 | |
| [0.5, 1.01) | 62 | 0.6177 | 0.6229 | −0.0052 | |

The mask-minus-box gap **flips sign** at 2048 in every bin. Box AP here uses polygon-extent GT boxes and is raster-independent, so the entire move is the mask.

### Touching vs isolated crowns — t04 (paper L892)

Same runner. Output: `ablation/results_r2048/strata_instance_r2048.json`; per-instance rows in `side_t04_r2048/`. **Gate:** at 512 the output is identical except two hand-added prose fields and two 4th-decimal values in the per-IoU sweep (s2 IoU 0.60: 0.5009 vs 0.5010; s0 IoU 0.70: 0.2762 vs 0.2763). The original *unshimmed dense* code in the current environment also gives 0.2763, so the flips are the published run's environment (2026-08-21), not the shim.

| box-matched TPs, 3 seeds | mean mask IoU 512 | mean mask IoU 2048 | fail rate (<0.5) 2048 |
|---|---|---|---|
| touching | 0.7093 ± 0.0015 | 0.6925 ± 0.0041 | 0.0418 ± 0.0024 |
| isolated | 0.7148 ± 0.0019 | 0.6944 ± 0.0025 | 0.0493 ± 0.0025 |

The conclusion holds at 2048: crown overlap does not break commonality (Δ 0.002 IoU; touching crowns fail *less*). Failure rates roughly double for both groups.

### tab:strata-maskiou — box_iou_strata

The wrapper is [`box_iou_strata_res.py`](ablation_r2048/box_iou_strata_res.py). **Gate PASS:** with the published rows at 512 the output is identical to `results_439/box_iou_strata.json`. SelvaBox has no 2048 masks and every row enters the common-subset intersection, so the comparison drops SelvaBox **at both rasters**. Outputs: `ablation/results_r2048/box_iou_strata_noselva_r{512,2048}.json`.

| row | mean mask IoU, common subset, 512 (n = 10,049) | 2048 (n = 11,126) |
|---|---|---|
| LACE s0 | 0.7442 | 0.7297 |
| Restor rpn1000 (pooled) | 0.7653 | 0.7890 |
| DetecTree2 s0v2 | 0.7593 | 0.7757 |
| Box2Mask Swin-L | 0.6942 | 0.7153 |

With detection quality controlled, LACE's masker trails Restor and DetecTree2 in **every** box-IoU stratum at 2048. At 512 it led them in the two lowest strata. It still beats both Box2Mask backbones everywhere. Caveat: GT boxes in this script are mask-derived, so the dilation bias moves strata edges between rasters.

### Not done in Phase C

tab:rpn Recall@0.5 at 2048 (`t_restor_rpn_recall`; all three Restor arms now exist at 2048). tab:masker's α=1, κ=10 row (needs test features on Modal). tab:components and tab:collapse predicted-box columns (feasibility not checked). t01 boundary_band, t02 per_iou, t09, t13 and t14 at 2048. Phase B (joint α, κ, τ on val at 2048).

---

## DETECTOR RETRAIN — decisions (2026-10-06, Tom)

Context: [`HANDOFF_detector_retrain.md`](HANDOFF_detector_retrain.md) §2. Decisions recorded before any GPU spend.

| # | decision | chosen |
|---|---|---|
| 1 | checkpoint / ES metric | val **box AP50:95** (box-only; no mask signal in training or selection) |
| 2 | horizon | min 20 epochs, patience 3 evals, cap 60, eval every 5 (cosine T_max = 60) |
| 3 | loss weights | GIoU ×2, size ×0.5 (one setting) |
| 4 | dense box supervision | chosen "heatmap ≥ 0.5, heatmap-weighted, overlap → smaller crown" — **found to be a no-op, re-decision pending** (below) |
| 5 | small-crown heatmap min radius | unchanged |
| 6 | pilot | **single run, seed 0, all changes at once** (no ladder; per-change attribution is not available) |
| 7 | adopt/stop | **post hoc**, on the 108 val tiles at 2048 (τ 0.40) vs 0.6045. Guardrail: decided on val only, before any test-split number for the new detector is computed. The pilot's Modal function trains only; it does not call `eval_4p` (which scores test). |
| 8, 9 | after adopting; paused SAM 3 rows | deferred |

**Decision 4 as specified does nothing.** Measured on the 792 training tiles (54,191 boxes) with `dapt.targets.gaussian_radius` at stride 8, overlap 0.7, min radius 1: 98.1% of crowns have the clamped radius r = 1 (median side 36 px = 4.5 cells), 1.7% r = 2, 0.2% r ≥ 3. With σ = (2r+1)/6 a Gaussian ≥ 0.5 covers only the centre cell for r ≤ 2, so **0.2%** of crowns would gain any extra regression cell (mean 1.01 cells/crown). Even ≥ 0.3 gives 1.9% (1.09 cells/crown).

**Re-decisions (Tom, 2026-10-06).** 4 → **dropped**: no dense supervision; regression stays at the single centre cell. Reason: its only evidence was the decode test (3×3 heatmap-weighted readout: box AP75 +0.0035, AP50 −0.0013), and dropping it leaves target encoding untouched. 10 (new) → ES `min_delta` **0.002** (0.005 was calibrated on box AP50 ≈ 0.5; AP50:95 runs ≈ 0.2). It affects only when training stops; the checkpoint is always the max val box AP50:95.

**Final pilot recipe (seed 0):** Detector4Phase on the 4-phase L24 cache, unchanged architecture and targets; loss = focal + offset + **0.5** × size + **2** × GIoU; Adam 1e-3, wd 1e-4, cosine T_max 60, bs 3; eval every 5 epochs; checkpoint = max val box AP50:95; early stop at ≥ 20 epochs after 3 evals without a > 0.002 gain; cap 60. Paper impact if adopted: L442 loss sentence and the tab:config "Detector training" row only (plus the already-existing omission of GIoU ×1 at L442).

### Pilot code and gate (2026-10-06) — **PASS**, nothing launched

New files only: [`det_retrain_v2.py`](det_retrain_v2.py) (training loop: loss weights, ES metric, per-epoch resume state, refuses to overwrite), [`det_retrain_modal.py`](det_retrain_modal.py) (Modal app `tcd04-det-retrain`, **train only**, writes `tcd04-phase4-vol:out/det_retrain/{tag}/` incl. `run.log`, retries ×3 with resume), [`det_val_eval.py`](det_val_eval.py) (val-only adopt/stop harness; re-gates the published detector at 0.6045 / 40,058 first).

Gate [`gate_det_retrain_v2.py`](gate_det_retrain_v2.py), CPU, 3+2 val tiles as toy train/val, 4 epochs:
- run 1 → `gate_det_retrain_v2.json`: PASS but weak (toy val AP 0, so the best checkpoint stayed at epoch 1).
- run 2, forced rising val metric so every epoch re-saves → `gate_det_retrain_v2_forced.json`: **(a)** legacy settings reproduce the published `train_detector_tiles.train` **bitwise** at epoch 4; **(b)** interrupted-at-epoch-2 then resumed equals uninterrupted, bitwise; **(c)** pilot knobs change the weights. Logs in `logs/gate_det_retrain_v2*.log`.

Launch (awaiting Modal budget confirmation):

    python detach.py deploy det_retrain_modal.py
    python detach.py spawn tcd04-det-retrain train --seed 0 --note "pilot v2 s0"

### RESULT — pilot training, seed 0 (2026-10-06)

Call `fc-01M48AGN8XGDX95NWAPPEVAVTF`, A100-40GB, 91 min, ≈ $3.19. Early-stopped at epoch 35; **best epoch 20**. Training-time val (Modal features, 108 val tiles, `dapt.eval.full_report`, no canopy ignore):

| epoch | box AP50 | box AP50:95 | |
|---|---|---|---|
| 5 | 0.4592 | 0.1837 | |
| 10 | 0.4776 | 0.1927 | |
| 15 | 0.5044 | 0.2171 | |
| **20** | **0.5137** | **0.2306** | selected |
| 25 | 0.4990 | 0.2229 | |
| 30 | 0.4788 | 0.2092 | |
| 35 | 0.4665 | 0.2064 | stop |

Same metric for the published seed-0 detector: box AP50 0.4975 at its epoch 20 (AP50:95 not recorded). Training loss kept falling after epoch 20 while val fell: overfitting, and the early stop caught it. Artefacts copied same-session to [`det_retrain_out/v2_L24_s0/`](det_retrain_out/v2_L24_s0/) (`det_*.pt`, `state_*.pt`, `run.log`, `summary_*.json`); originals remain on `tcd04-phase4-vol:out/det_retrain/v2_L24_s0/`. Local copy for the val harness: `feat_cache/det_v2_L24_s0.pt`, sha1 `1bc8560e…` (equal to the repo copy).

### RESULT — pilot vs published detector, 108 val tiles, 2048, τ 0.40 (2026-10-06)

[`det_val_eval.py`](det_val_eval.py) → `det_val_eval_v2_L24_s0_r2048.json`, predictions `preds_val_v2_L24_s0_r2048_tau040.json`, log `logs/det_val_eval_v2_L24_s0_r2048.log`. Local features, deployed masker (α 0.3, κ 1.6), **detector-score ranking** (not the s·p̄·m product), score_coco canopy-neutral crowd. **Gate PASS:** the published detector reproduces AP50 0.6045 / 40,058 dets. No test-split number exists for the new detector.

| detector | dets | mask AP50 | mask AP75 | mask AP50:95 | box AP50 | box AP75 | box AP50:95 |
|---|---|---|---|---|---|---|---|
| published s0 | 40,058 | 0.6045 | 0.1328 | 0.2436 | 0.6338 | 0.1863 | 0.2742 |
| pilot v2 s0 | 44,103 | 0.6132 | 0.1404 | 0.2495 | 0.6501 | 0.2063 | 0.2909 |
| Δ | +10% | **+0.0087** | +0.0076 | +0.0059 | +0.0163 | +0.0200 | +0.0167 |

**Reading.** Every metric improves, and box more than mask: about half the box AP50 gain reaches the masks. The mask AP50 gain (+0.0087) is **below the +0.01 floor proposed in the handoff** and about half the ~+0.018 test gap to Restor rpn1000. It is one seed against one seed; the test seed band (ddof 1) is 0.0033, so a seed draw could account for a fair part of the gain. Not yet measured: the gain under product ranking (the deployed ranking) and seed variance on val. **Adopt/stop: pending Tom.**

### RESULT — pilot under the deployed ranking, and what the extra detections are (val, 2048, 2026-10-06)

[`det_val_product.py`](det_val_product.py) → `det_val_product_r2048.json`; predictions with posterior factors `preds_val_{published_s0,pilot_v2_s0}_r2048_tau040_withpost.json`; log `logs/det_val_product_r2048.log`. **Gates PASS:** (1) `masker.box_mask` posterior = the em_diag product formula, max |Δpfg| 6e-16 over 400 boxes per detector; (2) detector-ranked AP50 and det counts reproduce `det_val_eval` (0.6045 / 40,058; 0.6132 / 44,103). Anatomy uses the fp_anatomy_test rule, and its own AP50 reproduces score_coco's to ≤ 1e-4.

| ranking | detector | mask AP50 | AP75 | AP50:95 | box AP50 | max recall | precision @R0.5 / 0.6 / 0.7 |
|---|---|---|---|---|---|---|---|
| s | published | 0.6045 | 0.1328 | 0.2436 | 0.6338 | 0.7649 | 0.777 / 0.707 / 0.574 |
| s | pilot | 0.6132 | 0.1404 | 0.2495 | 0.6501 | 0.7803 | 0.765 / 0.703 / 0.573 |
| s·p̄·m | published | 0.6427 | 0.1647 | 0.2693 | 0.6662 | 0.7649 | 0.835 / 0.762 / 0.605 |
| s·p̄·m | pilot | **0.6495** (**+0.0068**) | 0.1730 | 0.2750 | 0.6794 | 0.7803 | 0.831 / 0.749 / 0.607 |

**The gain is tail recall, not localisation.** Precision at fixed recall is flat or slightly lower for the pilot under both rankings, and its "loc" FPs at R 0.6 do not fall (s: 510 vs 477; product: 470 vs 429). The whole AP gain comes from max recall 0.765 → 0.780 (+108 matched crowns, 5,494 vs 5,386). The retrain did not fix the mislocated-FP problem that motivated it. The product gain (+0.0068) is smaller than the detector-score gain (+0.0087).

**The extra ~4,000 detections are low-score background.** By detector score: [0.05, 0.3) holds 20,977 dets (pilot) vs 17,155 (published), +3,822, of which ≈ +3,650 are "bg" (IoU < 0.1 with every crown). Above s 0.4 the pilot has *fewer* dets (4,790 vs 5,256): its scores are compressed downwards, with TPs moving from [0.5, 1) into [0.3, 0.4). The tail FPs rank last, so they cost AP almost nothing; they would matter for any fixed-threshold use and add masker work.

**Forecast.** Transferred one-for-one, +0.007 on the test 3-seed 0.6300 gives ≈ 0.637, against 0.6480 for Restor rpn1000: short by ≈ 0.011. That is single-seed and val, so not a test claim.

**DECISION (Tom, 2026-10-07): STOP.** The retrain is not adopted. The detector, the paper numbers and every 2048 row stay as published. The pilot's artefacts stay as a recorded negative. Paper action independent of this: L442 omits the deployed GIoU ×1 loss term.

### RESULT — is the spatial prior too strong? NO (val, 2048, 2026-10-07)

Hypothesis (Tom): the prior ρ is learned on GT boxes and indexed by position inside the predicted box, so on a misplaced box it pushes the mask away from the truth. The α grid of Aug 2026 (512) stopped at its edge, α 0.3, and α was never re-selected at 2048. [`prior_sweep_val.py`](prior_sweep_val.py) → `prior_sweep_val_r2048.json` (50 cells), log `logs/prior_sweep_val_r2048.log`, per-tile intermediates `feat_cache/prior_sweep_val_r2048_tiles/` (outside git). Published seed-0 detector, 108 val tiles, local features, 12 workers × 1 BLAS thread, 23 min + scoring. **Gate PASS:** cell (0.3, 1.6, 0.40) reproduces 0.6045 (s), 0.6427 (product), 40,058 dets. "noprior" = ρ set to 0.5/K everywhere at inference only (uniform per-prototype weights, fg-mass logit 0); the prototypes were still fitted with the prior.

Mask AP50, product ranking (best τ per row in bold):

| config | τ 0.30 | 0.35 | 0.40 | 0.45 | 0.50 |
|---|---|---|---|---|---|
| α 0, κ 1.6 | 0.6102 | 0.6155 | **0.6186** | 0.6103 | 0.5855 |
| α 0, κ 2.0 | 0.6152 | **0.6203** | 0.6167 | 0.6091 | 0.5886 |
| α 0.1, κ 1.6 | 0.6272 | 0.6287 | **0.6319** | 0.6257 | 0.6134 |
| α 0.1, κ 2.0 | 0.6285 | 0.6297 | **0.6310** | 0.6229 | 0.6086 |
| α 0.2, κ 1.6 | 0.6353 | 0.6382 | **0.6391** | 0.6375 | 0.6239 |
| α 0.2, κ 2.0 | 0.6352 | 0.6372 | **0.6386** | 0.6315 | 0.6227 |
| **α 0.3, κ 1.6 (deployed)** | 0.6386 | 0.6417 | **0.6427** | 0.6412 | 0.6327 |
| α 0.3, κ 2.0 | 0.6392 | **0.6410** | 0.6408 | 0.6408 | 0.6298 |
| noprior, κ 1.6 | 0.6101 | 0.6158 | **0.6184** | 0.6108 | 0.5855 |
| noprior, κ 2.0 | 0.6151 | **0.6196** | 0.6165 | 0.6091 | 0.5888 |

Detector-score ranking (s): every cell lies in 0.56–0.607; at τ 0.40 α barely matters (0.6042 at α 0, 0.6067 at α 0.1, 0.6045 at α 0.3).

**Reading.**
1. **Hypothesis rejected.** Under the deployed ranking, weakening the prior is monotonically worse: α 0.3 → 0.2 → 0.1 → 0 costs 0.004 / 0.011 / 0.024 AP50. The deployed (0.3, 1.6, 0.40) is the argmax of all 50 cells. AP50:95 follows the same order.
2. **The prior's value is almost entirely in ranking, not mask shape.** With s ranking α moves AP50 by ≤ 0.003; with product ranking by 0.024. The fg-mass term makes p̄ (mean posterior) informative about whether a box holds a crown.
3. **The per-prototype part of ρ does nothing.** noprior equals α 0 to within 0.001 in every cell, so the per-bin prototype weights ρ̂_k carry no signal (consistent with tab:collapse). All of the prior's effect is the total fg-mass map Σ_k ρ_k entering through α.
4. Untested: α > 0.3 at 2048 (the 512 grid had 0.3 > 0.4 > 0.5 under s ranking). At 2048 + product the curve is still rising at 0.3, so the optimum could sit at 0.3–0.5.

### RESULT — prior for ranking only (masks at α_M, product factors at α_R), val 2048 (2026-10-07)

[`prior_decouple_val.py`](prior_decouple_val.py) → `prior_decouple_val_r2048.json` (100 cells, scoring only, from the prior-sweep intermediates), log `logs/prior_decouple_val_r2048.log`. **Gate PASS** (M = R = (0.3, 1.6), τ 0.40 → 0.6427).

Ranking factors from (0.3, 1.6), τ 0.40, mask AP50 / AP50:95:

| mask α_M (κ 1.6) | AP50 | AP50:95 |
|---|---|---|
| 0 (mask = σ(Ā), no prior) | 0.6414 | 0.261 |
| 0.1 | **0.6440** | 0.268 |
| 0.2 | 0.6429 | 0.269 |
| 0.3 (deployed, coupled) | 0.6427 | 0.269 |
| inference no-prior | 0.6412 | 0.261 |

**Reading.** Decoupling recovers nearly all of the ranking value: prior-free masks with prior-based ranking give 0.6414, vs 0.6186 fully prior-free and 0.6427 deployed. So the prior can be confined to ranking at an AP50 cost of 0.001, but it costs **0.008 AP50:95**: at strict IoU the prior does help mask shape. No setting beats deployed by more than 0.0013 AP50 (α_M 0.1), which is noise. R from κ 2.0 changes nothing (≤ 0.0005). **No route to Restor here.** The decoupled form is only a possible simplification of the story ("the prior is a ranking prior"), and only if AP50:95 is given up.

### RESULT — guided (RGB edge-aware) upsampling of the posterior, val 2048 (2026-10-07)

[`guided_render_val.py`](guided_render_val.py) → `guided_render_val_r2048.json` (21 cells), log `logs/guided_render_val_r2048.log`, intermediates `feat_cache/guided_render_val_r2048_tiles/` (outside git). Isolated: no existing code or data edited, no packages added (guided filter via `scipy.ndimage.uniform_filter`). The deployed bilinear posterior map per box goes through a grey-guide guided filter (He et al. 2010; guide = luminance of the local RGB tile, verified per tile against `manifest.json`), then τ + box clip as before. Masker, detector, knobs and ranking unchanged. **Gate PASS:** the bilinear arm reproduces 0.6045 / 0.6427 / 40,058 dets.

Product ranking, mask AP50 (AP75 / AP50:95):

| render | τ 0.35 | τ 0.40 | τ 0.45 |
|---|---|---|---|
| bilinear (deployed) | 0.6417 | 0.6427 (0.165 / 0.269) | 0.6412 |
| guided r 4, ε 1e-4 | 0.6462 | **0.6488** (0.175 / 0.276) | 0.6418 |
| guided r 4, ε 1e-3 | 0.6460 | 0.6468 | 0.6418 |
| guided r 4, ε 1e-2 | 0.6446 | 0.6462 | 0.6382 |
| guided r 8, ε 1e-4 | 0.6473 | 0.6356 | 0.6027 |
| guided r 8, ε 1e-3 | 0.6470 (0.187 / 0.279) | 0.6322 | 0.6012 |
| guided r 8, ε 1e-2 | 0.6445 (0.192 / 0.280) | 0.6249 | 0.5893 |

**Reading.** First positive training-free result at 2048. Best cell: **+0.0061 AP50, +0.010 AP75, +0.007 AP50:95** (product); detector-score ranking +0.0072. All 6 r = 4 cells at τ 0.35–0.40 beat bilinear (+0.002 to +0.006), so the gain is not one lucky cell. r 8 smooths more and moves the τ optimum down (best at 0.35), with the largest AP75/AP50:95 gains (up to +0.027 AP75). The optimum sits at the grid edge (smallest r and ε), so r 2 or a smaller ε may do better. Caveats: 3 knobs read on the same 108 val tiles (mild optimism); 2048 only (the 512 protocol is untested); grey guide only. Forecast: one-for-one on test, ≈ 0.636 vs Restor rpn1000 0.648. A real, free gain, but on its own not enough.

### RESULT — COLOUR-guided upsampling, val 2048 (2026-10-07)

[`guided_rgb_val.py`](guided_rgb_val.py) (a copy of `guided_render_val.py`, which is untouched, with He et al.'s colour-guide form: per-pixel 3×3 covariance solve on the RGB crop) → `guided_rgb_val_r2048.json` (33 cells), log `logs/guided_rgb_val_r2048.log`, intermediates `feat_cache/guided_rgb_val_r2048_tiles/`. Filter unit checks: a guide containing p reproduces p (max err 8e-6); a colour guide with constant extra channels equals the grey filter (1e-7). **Gates PASS:** bilinear 0.6045 / 0.6427 / 40,058; grey r4 ε1e-4 τ0.40 re-gives 0.6488.

Product ranking, mask AP50 (AP75 / AP50:95 at the best τ):

| render | τ 0.35 | τ 0.40 | τ 0.45 |
|---|---|---|---|
| bilinear (deployed) | 0.6417 | 0.6427 (0.165 / 0.269) | 0.6412 |
| grey r4 ε1e-4 | 0.6462 | 0.6488 | 0.6418 |
| colour r2, ε 1e-4 / 1e-3 / 1e-2 | 0.6430 / 0.6435 / 0.6441 | 0.6440 / 0.6440 / 0.6431 | ≤ 0.6416 |
| colour r4, ε 1e-4 / 1e-3 / 1e-2 | 0.6498 / 0.6495 / 0.6457 | 0.6499 / **0.6506** / 0.6468 | ≤ 0.6435 |
| colour r8, ε 1e-4 | **0.6610** (0.211 / 0.294) | 0.6600 | 0.6467 |
| colour r8, ε 1e-3 / 1e-2 | 0.6603 / 0.6524 | 0.6557 / 0.6405 | 0.6362 / 0.6115 |

**Reading.** The colour guide is much stronger than grey. Best: colour r8, ε 1e-4, τ 0.35 → **+0.0183 AP50, +0.046 AP75, +0.025 AP50:95** over deployed (product); detector-score ranking +0.020 (0.6246 vs 0.6045). The gain grows with radius (r2 < r4 < r8) and with smaller ε, and the best τ moves down to 0.35. **All three of these optima sit at grid edges** (largest r, smallest ε, lowest τ), so the true optimum is outside the grid. The gain is not one cell: all six colour r8 cells at ε ≤ 1e-3 and τ ≤ 0.40 are ≥ +0.013.

**Caveats.** 3 knobs read on the same 108 val tiles, the best cell picked from 33 (optimism bias; needs a split-half check). 2048 only; the 512 protocol is untested. Single detector seed. If transferred one-for-one to test, 0.630 + 0.018 ≈ 0.648, a tie with Restor rpn1000; not yet a win.

### RESULT — colour-guided upsampling: extended grid, split-half, and the 512 protocol raster (val, 2026-10-07)

[`guided_rgb_ext_val.py`](guided_rgb_ext_val.py) → `guided_rgb_ext_val_r{2048,512}.json`, log `logs/guided_rgb_ext_val.log`, intermediates `feat_cache/guided_rgb_ext_val_r{2048,512}_tiles/`. Every cell is scored on all 108 val tiles and on two fixed halves (sorted ids, even/odd: A 54 tiles / 21,768 dets, B 54 / 18,290). At 512 the RGB guide is area-downsampled and r is in 512-px units. **Gates PASS:** 2048 bilinear 0.6045 (s) / 0.6427 (product) / 40,058 dets; colour r8 ε1e-4 τ0.35 re-gives 0.6610; 512 bilinear τ0.25 0.6392 (s) = `val_knobs_tau_r512_s0_local.json`.

**2048, product ranking** (AP50 / AP50:95; deployed bilinear τ0.40 = 0.6427 / 0.269):

| colour r (px) | best ε | τ 0.25 | τ 0.30 | τ 0.35 | τ 0.40 |
|---|---|---|---|---|---|
| 8 | 1e-4 | 0.6460 | 0.6536 | 0.6610 / 0.294 | 0.6600 |
| **12** | **1e-4** | 0.6506 | 0.6612 | **0.6636 / 0.298** | 0.6527 |
| 16 | 1e-5 | 0.6532 | 0.6594 / 0.293 | 0.6495 | 0.6312 |
| 24 | 1e-5 | 0.6361 | 0.6125 | 0.5788 | 0.5316 |

The optimum is now interior: r 12, ε ≤ 1e-4 (1e-5 ≈ 1e-4, a plateau), τ 0.30–0.35. **Best: 0.6636 = +0.0209 AP50, AP75 0.216 (+0.051), AP50:95 0.298 (+0.029).** Larger r wants lower τ; r 24 over-smooths.

**Split-half (select on one half by product AP50, report on the other):** select A → r12 ε1e-4 τ0.35, gain on B **+0.0150** (0.6567 vs 0.6417); select B → r8 ε1e-4 τ0.35, gain on A **+0.0173** (0.6629 vs 0.6456). **Honest held-out gain ≈ +0.016 AP50 at 2048.**

**512 (protocol raster), product ranking** (deployed bilinear τ0.25 = 0.6739 / 0.294):

| colour r (512 px) | τ 0.20 | τ 0.25 | τ 0.30 | τ 0.35 |
|---|---|---|---|---|
| 1 (ε 1e-4) | 0.6716 | 0.6777 | 0.6803 | 0.6740 |
| **2 (ε 1e-5)** | 0.6765 | 0.6791 | **0.6812 / 0.306** | 0.6737 |
| 3 (ε 1e-5) | 0.6773 | 0.6812 / 0.308 | 0.6748 | 0.6616 |
| 4 (ε 1e-5) | 0.6764 | 0.6737 | 0.6583 | 0.6316 |
| 6 (ε 1e-5) | 0.6537 | 0.6262 | 0.5892 | 0.5394 |

512 best +0.0073 AP50 / +0.012 AP50:95. Split-half: +0.0012 (select A → report B), +0.0080 (select B → report A), so ≈ +0.005 held-out. **512 is not hurt; the gain there is small.** Physical consistency: r 3 px at 512 = r 12 px at 2048 = **1.2 m**, and that radius is near-best at both rasters (512 r3 τ0.25: 0.6805–0.6812). So one physically defined radius serves both rasters, with τ re-selected per raster as before.

**Reading.** First lever that closes the 2048 gap at val level: held-out ≈ +0.016 vs the test gap of ≈ 0.018 to Restor rpn1000. Training-free, label-free, a rendering step after the masker; adds ε (plateau, any ≤ 1e-4) and r (fixable physically at 1.2 m) as inference choices; disclose as val-selected. Not yet a test number: needs a 3-seed test render at both rasters (RGB for test tiles; Modal CPU).

### CONTROL — is the colour-guide gain from image edges or from smoothing? EDGES (val 2048, 2026-10-07)

Supervisor sense check. [`guided_control_val.py`](guided_control_val.py) (τ 0.30–0.40) and [`guided_control_lowtau_val.py`](guided_control_lowtau_val.py) (τ 0.10–0.25) → `guided_control_val_r2048.json`, `guided_control_lowtau_val_r2048.json`; logs `logs/guided_control*_r2048.log`. Isolated copies; all four gates PASS (bilinear 0.6427 @ 0.40 and 0.6340 @ 0.25; colour r12 0.6636 @ 0.35 and 0.6506 @ 0.25). Arms at r 12, ε 1e-4: **blur** = constant guide, i.e. q = mean(mean(p)), what the filter reduces to with no image structure; **wrong tile** = colour guide from a different val tile (next in sorted order).

Product ranking, best τ over 0.10–0.40 (mask AP50 / AP50:95):

| render | best τ | AP50 | AP50:95 |
|---|---|---|---|
| bilinear (deployed) | 0.40 | 0.6427 | 0.269 |
| **true RGB guide** | 0.35 | **0.6636** | **0.298** |
| blur only | 0.25 | 0.5948 | 0.232 |
| wrong-tile RGB guide | 0.25 | 0.6117 | 0.238 |

**Reading.** At their own best τ both controls lose 0.03–0.05 to bilinear, and only the true image as guide gains. The +0.021 comes from the tile's own edges, not from smoothing, and not from the guide merely adding colour texture.

---

## DECISION (Tom, 2026-10-07): re-base the paper on the native 2048² raster; remove 512 entirely

Supersedes the 2026-09-26 PAPER INTEGRATION decision ("submit at the 512² protocol raster"). Tom: the 512 raster was an oversight; every trace of it is to be removed and replaced by native 2048, provided LACE is still competitive. Context: training is 900 images with boxes only, against Restor's ~4.3k images with full masks, so **a tie with Restor rpn1000 counts as a win**. No baseline is modified: the colour-guided render is part of LACE, and baselines stay as released and measured.

## PRE-REGISTRATION — LACE + colour-guided render, 439 test tiles, 2048 (frozen 2026-10-07, before any test number)

Settings, all selected on the 108 val tiles only (sections above):

| knob | value | source |
|---|---|---|
| detector, boxes, scores, ranking | published seeds 0, 1, 2, product s·p̄·m; boxes and scores verbatim from `out/native_raster/src/lace_s{0,1,2}.json` | unchanged |
| masker | `em_model_4p_fix.npz`, α 0.3, κ 1.6 | unchanged |
| render | colour-guided filter (He et al. 2010) on the bilinear posterior map, guide = the tile's RGB in [0, 1], **r = 12 px (1.2 m), ε = 1e-4**, window = box clip ± 16 px | `guided_rgb_ext_val_r2048.json` (interior optimum; split-half held-out +0.015 / +0.017) |
| mask threshold | **τ = 0.35** | same |
| box clip | unchanged | |

Reported: canopy-neutral crowd mask AP50 / AP75 / AP50:95 and box AP50, `score_coco.py --res 2048 --score_floor 0`, 3 seeds, mean ± sd (ddof 1). These settings will not be changed after test numbers are seen.

**Gate.** The same run also emits the deployed bilinear τ 0.40 masks; scoring them must reproduce the published `scored_lace_s{s}_r2048_tau040.json` AP50 exactly for each seed, before the guided number is read. In-container gates: the bilinear path equals `evaluate.pred_instance_masks` on the first tile, and every RGB tile is verified against `manifest.json` (pixel sha1 or image_id).

Outputs, new paths only: `tcd04-phase4-vol:out/native_raster/guided_r12/`; local `native_raster/preds_lace_s{s}_r2048_cgf_r12_tau035.json` (+ the bilinear gate files). The test RGB is staged to `tcd04-phase4-vol:rgb_test/`, a new directory.

### RESULT — PRE-REGISTERED TEST: LACE + colour-guided render, 439 tiles, 2048, 3 seeds (2026-10-07)

Run: [`lace_guided_modal.py`](lace_guided_modal.py), app `tcd04-lace-guided`, calls `fc-01M4AP4VHP6KQE0S5J17717D89` (s0), `fc-01M4AP4WS1A7R1F6N49X7BN48W` (s1), `fc-01M4AP4XWPD4QARYC2N921SGSD` (s2); smoke `fc-01M4AP1BN9T4FRVDX756Y3QJFT` (3 tiles, separate `_smoke_lim3` dir). Modal CPU, ~25 min per seed in parallel. Test RGB staged to `tcd04-phase4-vol:rgb_test/` (439 tif + meta; per-tile sha1/image_id verified in container). Raw JSONL, logs and summaries copied same-session to [`guided_r12_raw/`](guided_r12_raw/); originals on `out/native_raster/guided_r12/`. Converted `preds_lace_s{s}_r2048_{cgf_r12_tau035,bilinear_tau040}_guidedrun.json`, scored `scored_lace_s{s}_r2048_*_guidedrun.json` (`score_coco --res 2048 --score_floor 0`), logs `logs/score_s*_guidedrun.log`.

**Gates.** In container, every seed: filter self-guide err 8e-6; bilinear path = `evaluate.pred_instance_masks` on the first tile; RGB verified. Pairing (`jsonl_to_preds`): boxes and scores identical to the published rows, all 6 files, 439/439 tiles, det counts 159,687 / 162,017 / 177,598. Bilinear τ0.40 rerun vs published masks: s0 and s2 byte-identical on 439/439 tiles; s1 differs on 6 masks of 162,017 by **exactly 1 pixel each**, all in tiles processed after s1's container was pre-empted and retried (08:03; resumed from 15/439). Float rounding at τ on a different host CPU, not a logic difference. **Scored bilinear rerun = published AP50 / AP75 / AP50:95 for all three seeds (PASS).**

Canopy-neutral crowd, mean ± sd (ddof 1):

| row | mask AP50 | AP75 | AP50:95 | box AP50 |
|---|---|---|---|---|
| LACE bilinear τ0.40 (previous 2048 row) | 0.6300 ± 0.0033 | 0.1315 ± 0.0103 | 0.2534 ± 0.0055 | 0.6666 ± 0.0108 |
| **LACE + colour-guided render (pre-registered)** | **0.6470 ± 0.0025** | 0.1730 ± 0.0101 | 0.2778 ± 0.0051 | 0.6666 ± 0.0108 |
| per seed (s0 / s1 / s2) | 0.6472 / 0.6444 / 0.6493 | 0.1816 / 0.1619 / 0.1755 | 0.2821 / 0.2722 / 0.2791 | |
| Restor Mask R-CNN rpn1000, tree-only | 0.6480 | 0.2968 | 0.3332 | 0.6646 |
| DetecTree2 s0v2 | 0.6143 | 0.2534 | 0.3014 | 0.6387 |
| *(Restor rpn2000, data only, excluded from the paper)* | *0.6846* | *0.3070* | *0.3484* | *0.7050* |

**Reading.** The gain transfers from val to test almost one-for-one: **+0.0170 AP50** (per seed +0.0148 / +0.0182 / +0.0179; val held-out forecast +0.016). LACE vs Restor rpn1000: **−0.0010 AP50, inside one seed sd (0.0025)**; the best seed (0.6493) exceeds it. A statistical tie on mask AP50, which Tom counts as a win given 900 box-only training images vs ~4.3k mask images. LACE leads DetecTree2 by +0.033. Detections unchanged: box AP is identical by construction. **Not a tie at strict IoU:** AP75 0.173 vs 0.297 and AP50:95 0.278 vs 0.333; the 8 px feature grid still limits fine boundaries.

**Not yet done for a full re-base:** sparse-236 rows with the guided render; the mask-based ablations at 2048 with the guided render; the paper text (method paragraph, tab:config, inference-scalar count). `paper.tex` untouched.
