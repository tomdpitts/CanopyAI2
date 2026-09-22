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
