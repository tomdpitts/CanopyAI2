# Audit: Restor Mask R-CNN baseline (rpn512 / rpn1000 / rpn2000), 2026-10-14

<!-- TABLE1_CURRENT:begin (auto, results_439/mark_superseded.py) -->
> ⚠️ **SUPERSEDED 439 / sparse236 NUMBERS (marked 2026-10-27).** Current headline table: [`TABLE1_CURRENT.md`](../../results_439/TABLE1_CURRENT.md). Test-set numbers in this file are a historical experiment record (512 raster, pre-seam-fix baselines, 8 px masker or earlier models) and must not be quoted unless they match that table.
<!-- TABLE1_CURRENT:end -->

Scope: `restor/tcd-mask-rcnn-r50` rows at native 2048 on the 439 test tiles. Read-only. No Modal job launched; only `modal volume ls/get` of the three `run.log` files. Evidence (scripts, outputs, fetched upstream configs, Modal run logs) is in `restor_evidence/`. `verify.py` writes an intermediate file `arms_light.json` to the session scratchpad, which `dist.py` and `stability.py` then read.

## 1. Settings table

Abbreviations: NRM = `native_raster/restor_native_modal.py`; HFC = released `config.yaml` at HF `restor/tcd-mask-rcnn-r50@0b4a94d7` (copy: `restor_evidence/hf_restor_config_0b4a94d7.yaml`); OAM = `~/dphil/refs/lace_bib_pdfs/oamtcd.txt`.

| item | our setting (file:line) | released default (source) | effect | severity |
|---|---|---|---|---|
| Checkpoint | `restor/tcd-mask-rcnn-r50` `model.pth`, sha256 `5c33e18d…19db`, HF rev `0b4a94d7c81d…` (NRM:34,107-108). The revision is not pinned in code, but the Modal HF cache holds only this snapshot, and HF `main` is still `0b4a94d7` (lastModified 2024-07-03) | same | neutral | OK |
| Config load | `get_cfg()` + `merge_from_file(HFC)` (NRM:109-111) | Restor pipeline: model_zoo `mask_rcnn_R_50_FPN_3x` + `detectron_mask_rcnn.yaml`. Inference keys are identical to HFC (512/512/0.2/512, MIN 0/MAX 2048) | neutral | OK |
| Test geometry / resize | whole 2048², `MIN_SIZE_TEST 0`, `MAX 2048` (NRM:116-117), so `ResizeShortestEdge` returns `NoOpTransform` (d2 `augmentation_impl.py:174`) | HFC L28/L30 the same. OAM L360-361: "we do not perform tiled inference when testing on the holdout set". Training used native-scale 1024 `FixedSizeCrop` (tcd `train_instance.py:127`), so test scale 1.0 = train scale | neutral | OK |
| Deployment pipeline (`tcd-predict`) | not used | tcd `data/default.yaml`: tile 1024, overlap 256. Current `main` model cfg = `detectron_mask_rcnn_tta.yaml` (multi-scale TTA 768–2048 + flip). HFC L308 `AUG.ENABLED false` | unknown | OK. Paper says "their own test configuration" (L1185), which is true of the holdout protocol, not of the deployment tool |
| Channel order | RGB→BGR flip (NRM:143) | HFC L26 `FORMAT: BGR`. Training reads via `DatasetMapper(cfg)`, i.e. BGR | neutral | OK |
| `SCORE_THRESH_TEST` | 0.05 (NRM:114) | 0.2 (HFC L201) | helps | disclosure, done (note a, tab:geometry) |
| `DETECTIONS_PER_IMAGE` | 600 (NRM:115), applied in-model to **both classes jointly** | 512 (HFC L321) | helps marginally: rpn1000 max is 522 total dets/tile (1 tile above 512) | disclosure, done |
| `RPN.PRE/POST_NMS_TOPK_TEST` | 1000/1000 (NRM:118-119). PRE is per FPN level, POST is per image (d2 `proposal_utils.py:33-34,129`) | 512/512 (HFC L254/L256; tcd `detectron_mask_rcnn.yaml` "effectively the max number of detections"). OAM L385-386: "we increased the number of predictions to 512". d2 FPN default 1000/1000 (`Base-RCNN-FPN.yaml`); d2 global default PRE 6000/POST 1000 (`defaults.py:235,243`) | **helps**: +0.063 AP50 (0.5847→0.6480) | disclosure, done (note a, tab:geometry, tab:rpn) |
| Class filter | tree-only = `pred_classes==1` post hoc (TREE_CLASS NRM:38; derivation `restor_gate_derive.py`). All classes emitted | the model has 2 classes; Restor's own eval pools both categories | helps (tree 0.6480 vs pooled 0.6342) | disclosure, done (note a) |
| Mask paste | d2 native paste at 2048, RLE stored as-is (NRM:149-152); all three r2048 files are size 2048² | same | neutral | OK |
| Scorer | `score_coco.py` (unchanged since c77bceb, 2026-09-03), res 2048, maxDets 600, floor 0.05, canopy as crowd: identical protocol blocks in all 6 `scored_restor_rpn*_r2048_*.json` | n/a | the canopy crowd rule helps every row | OK |
| Runtime | torch 2.1.0, d2 0.6, numpy 1.26.0, Pillow 10.0.1, CUDA 12.1, A100-SXM4-40GB; identical in all three Modal `run.log`s | n/a | neutral | OK |
| Training-image count | paper "4169" (L636, L940), "~4.2k" (L126) | OAM L1042: train split = 4406 (incl. CC BY-NC/SA). Model card: "trained on CC BY-NC imagery". 4169 = CC-BY HF split only | understates Restor's data, which is conservative for LACE | disclosure (minor; verify) |
| rpn2000 measured, not reported | tab:rpn has 512 and 1000 only (L1199-1209) | n/a | n/a | **disclosure** (see §3) |
| "max score identical … additions entirely low-confidence tail" (L1196) | global max is 0.9964 in all arms, but the per-tile max differs on 34/439 tiles (512→1000; always higher, never lower). 1,034/13,451 (7.7%) of the added tree dets score ≥0.5. ~24% of the box-TP gain (503/2,136) sits at score ≥0.5 | n/a | wording | disclosure: "entirely" → "predominantly" |
| "0.432 vs 0.648 is protocol not model" (L1133) | 0.585→0.648 of that gap comes from our RPN override, which is an inference-config change, not a scoring-protocol change. The 0.432 has never been re-scored by us under Restor's protocol (`modal_tcd_multiseed/phase4/RESULTS.md:40`) | n/a | wording | disclosure |

No DEFECT. Nothing we do hinders Restor. Every deviation is in the helping direction and is disclosed.

## 2. Is rpn1000 Detectron2's default; what is the released cap

- Detectron2 FPN default: `RPN.PRE_NMS_TOPK_TEST 1000` (per level), `POST_NMS_TOPK_TEST 1000` (per image) (`Base-RCNN-FPN.yaml`, which `mask_rcnn_R_50_FPN_3x.yaml` inherits; the model card names that arch config). **Claim true.**
- Released: 512/512 in both the HF dump and the tcd pipeline cfg. OAM-TCD paper L385-386 explains the change as deliberate ("increased the number of predictions to 512", relative to d2's default of 100 detections/image). **rpn512 = released cap: true.**
- So rpn1000 departs from the released config by **raising** Restor's proposal budget (helps, +0.063 AP50). This is disclosed in paper.tex L651 (note a), L1185 (tab:geometry) and L1196-1209 (tab:rpn, with 512 as a sensitivity).

## 3. rpn2000 verification

**Verdict: like-for-like = YES.** Only the RPN topk differs. One asymmetry runs *against* rpn2000: the in-model 600 cap binds on 10 tiles. So 0.6846 is genuine and, if anything, a slight underestimate.

(a) **Code path.** `git log -p` on NRM covers e895778 (2026-09-22, the rpn1000 run) and c568617 (current, used for rpn512/2000). The diff changes only: the `RPN_TOPK` constant becomes the `rpn_topk` arg, `OUT_DIR` becomes `_dirs()`, the `_log` signature, assemble now skips instead of asserting on an existing file, and the docstring. Inference lines are untouched. The Modal `run.log` config lines are identical across arms except `rpn_pre/rpn_post` (`score_thr=0.05 dets=600 num_classes=2 min_size_test=0 max=2048 fmt=BGR`), and the env strings are identical. The launch command (transcript `ae2ea8aa…`, 2026-09-23) was `modal run --detach restor_native_modal.py::predict --rpn-topk 2000`.

(b) **Scoring.** All six scored files carry the identical protocol block `{pycocotools, res 2048, max_dets 600, score_floor 0.05, 439 tiles, iou 0.5:0.95, 101 rec thrs, area all}` and use the same `score_coco.py` (last changed 2026-09-03, before every Restor scoring). `ablation/scripts/t_restor_rpn_recall_r2048.py` re-scored all three arms in one process (`logs/rpn_recall_r2048.log`) and reproduced AP50 0.5847 / 0.648 / 0.6846. Tree filter: re-derived by me, byte-identical to each `*_treeonly.json` (scores, boxes, RLE).

(c) **Original `restor_baseline/preds_restor_rpn2000.json`** (2026-08-27 19:56). Its meta uses the *old* schema (`n_canopy_dropped`, no `pred_classes` / `emits_all_classes`; `rpn_topk_test 2000`), so it came from an uncommitted pre-b39d12d variant of `restor_modal.py` that dropped canopy at generation. The launch transcript was not retained (the oldest retained transcript is 2026-09-07). Provenance is therefore established only by reproduction, which I re-checked independently. The native rpn2000 in-container 512 arm, tree-filtered, is **byte-identical** to it on 439/439 tiles (71,606 dets; scores, boxes and RLE all equal). The same holds for rpn1000 (79,201 dets, all classes, vs `preds_restor_rpn1000.json`) and rpn512 (61,182 vs `preds_restor_allcls.json`; tree filter = `preds_restor.json`, 42,987). The native canopy count for rpn2000 (27,939) equals the original's `n_canopy_dropped`.

(d) **Anything other than topk?** None found. Score floor, dets/image, resize, format, class handling (tree = class 1), mask paste (2048² native in all three r2048 files) and dtype/runtime are all the same. Within each arm, r2048 and r512_incontainer have identical scores, boxes and classes (PASS ×3). Stability check: **84% of all rpn1000 tree dets (91% of those ≥0.5) reappear in rpn2000 with an exactly identical box and 4-dp score.** Any preprocessing or threshold difference would push that to ~0. The residual is the expected NMS/regression churn from extra proposals; 512→1000 gives 88% / 94%.

(e) **Counts and score distributions** (tree class unless stated):

| arm | tree dets | /tile | max tree/tile | max total/tile | tiles at 600 total cap | tree ≥0.5 | tree ≥0.7 | median score | global max |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 42,987 | 97.9 | 309 | 347 | 0 | 19,039 | 12,060 | 0.432 | 0.9964 |
| 1000 | 56,438 | 128.6 | 377 | 522 | 0 | 20,073 | 12,340 | 0.330 | 0.9964 |
| 2000 | 71,606 | 163.1 | 483 | **600** | **10** | 20,601 | 12,549 | 0.249 | 0.9964 |

Tree dets rise 1000→2000 on 404 tiles, never fall. **Correction to the previous instance: "cap 600 not binding" holds only for the scorer's tree-only count (max 483). In-model, `DETECTIONS_PER_IMAGE 600` applies to tree+canopy jointly and truncates 10 rpn2000 tiles.** This can only lower rpn2000. rpn1000 is never truncated.

(f) **Plausibility of +0.037 AP50.** It is plausible: proposal starvation on dense tiles. Box recall@0.5 by GT-density bin (tree, greedy, my own matcher):

| GT/tile | tiles | GT | 512 | 1000 | 2000 |
|---|---|---|---|---|---|
| 1-10 | 81 | 432 | 0.731 | 0.734 | 0.748 |
| 11-30 | 72 | 1,475 | 0.805 | 0.828 | 0.839 |
| 31-60 | 64 | 2,781 | 0.694 | 0.744 | 0.774 |
| 61-100 | 54 | 4,279 | 0.680 | 0.746 | 0.789 |
| 101-200 | 67 | 9,326 | 0.658 | 0.746 | 0.809 |
| 201-422 | 28 | 7,412 | 0.577 | **0.693** | 0.789 |

At 1000, recall still falls on the densest tiles; at 2000 it does not. The 88 densest tiles (62% of GT) take 80% of the 1000→2000 TP gain. Mask recall is 0.744→0.806 (+0.062), and the AP50/recall gain ratio is 0.60 (512→1000: 0.75). The same jump appears independently at the 512 raster from the Aug-27 runs (0.6255→0.6605, +0.035). So 1000 is still a binding proposal budget for the RPN on this data.

## 4. paper.tex vs code/files

| claim (paper.tex line) | check |
|---|---|
| L636 / L651: 0.648 / 0.333 / box 0.665; pooled 0.634/0.328; rpn512 0.585/0.308 | match the scored files (0.6480/0.3332/0.6646; 0.6342/0.3279; 0.5847/0.3076) |
| L651 "3 configs … all beneficial" | true (thresh, dets, RPN all helping). Tree-only is a 4th, beneficial choice; it is disclosed in the same note |
| L1185 whole 2048² "their own test configuration" | true for the OAM-TCD holdout protocol (OAM L360-361), not for the `tcd-predict` tiled/TTA tool |
| L1196 512 = "half Detectron2's FPN default"; 1000 = `Base-RCNN-FPN.yaml` | true |
| L1196 "max score identical … entirely low-confidence tail" | overstated (see §1) |
| L1204-1207 tab:rpn 97.9 / 0.6596 / 0.5847 / 0.3076; 128.6 / 0.7441 / 0.6480 / 0.3332 | match `logs/rpn_recall_r2048.log` and the scored files |
| L1215 "Restor … never reach the cap, at most 377 per tile" | true for rpn1000 (377 tree; 522 total < 600) |
| L1133 0.432 vs 0.648 "difference of protocol rather than of model" | partly false: +0.063 is our RPN override. Not verified by re-scoring |
| L555 densest tile 422 crowns | true (GT max 422) |
| L636/L940 Imgs 4169 | likely 4406 (see §1) |

**Disclosure risk (factual, for Tom's call).** L1196 justifies leaving the released 512 on starvation grounds. Our own data show the same starvation at 1000: recall +0.062 and AP50 +0.037 at 2000, which would put Restor at 0.685 vs LACE 0.647. `PROTOCOL_439.md:166-167` (decided earlier) says "Report 512 and 2000 as documented sensitivities". `NATIVE_RASTER.md:499` (2026-09-26) reverses that and excludes 2000 entirely. The paper currently carries no rpn2000 row and no mention of it.
