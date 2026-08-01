# SavannaTree — the settled TCD phase4 pipeline, transferred (Modal A100)

Runs the **4-phase real-8px L24 → Detector4Phase → β=0.5 geometry-fixed commonality-EM
masker @ mask_thr 0.25** pipeline, unchanged, on a second dataset: **SavannaTree**
(Zenodo [7094916](https://zenodo.org/records/7094916), paper *Data* 2023, 8(2), 44).

Sister folder to `../../modal_tcd_multiseed/phase4/`, which stays the canonical TCD run.
Everything method-side is **imported** from there — `phase4_features_tcd.feat_4phase`,
`phase4_lib_tcd.train_4p` / `eval_4p_selfmask`, `phase4_fit_tcd.fit_masker_4p`,
`em.TCDMasker`. This folder only adds *data* (loaders, GT, split) and its own Modal app.
No recipe hyperparameter is re-tuned, so this is a clean transfer test, not a new fit.

> ## ⚠️ READ [SESSION 3](#session-3-2026-07-31--the-benchmark-is-the-experiment) FIRST
> Two things below are **withdrawn**: the headline `mAP50 0.1206` is one draw from a
> distribution whose seed spread is ±0.049 (identical config, seed 1 → 0.0516), and the
> single-seed scale sweep that picked the 512 canvas is inside that noise. The diagnosis
> that the deficit is *confidence ranking* is also withdrawn — it is **annotation
> sparsity**, measured four independent ways in [Session 3](#session-3-2026-07-31--the-benchmark-is-the-experiment).

---

## 🧭 HANDOFF — read this first

**What is different from TCD, and why it matters more than it looks:**

| | OAM-TCD (settled) | SavannaTree |
|---|---|---|
| imagery | aerial ortho, global | RPAS/drone ortho, N. Australia savanna |
| source tile | 2048 px | **1024 px, 50%-overlap tiling** |
| classes | `tree` + `canopy` (**ignore**) | **`tree` only — no ignore class** |
| test set | 439 tiles, 25 705 trees | **449 tiles (site S4), 1 911 trees** |
| trees / tile | 58.6 | **4.3** (≈14× sparser) |
| median box (2048 canvas) | 36 px | **304 px** (≈8× larger) |
| train supervision | 61 214 boxes / 900 tiles | **4 701 boxes / 900 tiles** (≈13× less) |

**The load-bearing caveat: the ground truth is deliberately incomplete.** The paper labels
2 547 polygons for **36 identified tree species** — it is a *species* reference dataset, not
an exhaustive crown delineation. The authors say so themselves: the dataset "was generated
using RPAS **and on-ground surveys to confirm species labels**" (Jansen et al., *Deep
Learning with Northern Australian Savanna Tree Species: A Novel Dataset*, Data 2023, 8(2),
44). A crown is labelled only if a ground crew identified its species — so unsurveyed trees
are unlabelled by construction, not by oversight. Many perfectly real, clearly visible crowns carry no
polygon (see `claude_outputs/savannatree_gt_check.png`). Because SavannaTree has **no canopy
/ ignore class**, the TCD scorer's ignore path is a no-op here: **every correct detection of
an unlabelled tree is scored as a false positive.** Absolute mAP on this benchmark is
therefore a floor bounded by label sparsity, not a measure of delineation quality — and it
must never be compared to our TCD numbers.

**The paper DOES report an instance-segmentation baseline** (§3.5.1) — an earlier version of
this README claimed it did not, which was wrong; that claim came from a web-search summary
because MDPI blocks automated access, and it was stated far more confidently than the
evidence supported. The actual baseline:

> Mask R-CNN, ResNet50-FPN, 2885 train tiles, one class, lr 0.005, batch 2, early stopping.
> Validated on the 449 tiles. **Best precision 25.5%, recall 61.4%, mAP 34.5% at epoch 4**
> (early stopping fired at epoch 10). "As model training progressed, precision increased,
> while recall decreased."

Two of their observations independently corroborate this folder's diagnosis:
1. They attribute their low precision to *"predictions made on the unannotated trees present
   in the validation dataset … recorded as False Positive (FP) predictions"* — exactly the
   1.77:1 unlabelled-crown problem measured here.
2. Their recall/precision trade across epochs, and their choice of epoch 4, matches the
   oscillation seen in `LONG_ign`.

Unknowns that limit comparability: their IoU threshold, whether "mAP" is AP50 or AP@[.5:.95],
and the score threshold behind their P/R.

### How much is labelled? (39 random S4 test tiles)

**Exact counts, no estimation:**

| quantity | value |
|---|---|
| annotations in the S4 test split | **1 911** over 449 tiles = 4.26 / tile |
| tile-level annotations, whole dataset | 14 829 |
| **unique** polygons, whole dataset (paper) | **2 547** across all 7 mosaics |
| ⇒ tile-overlap duplication factor | ≈ **5.8×** (50%-overlap tiling) |
| ⇒ **unique labelled crowns behind the S4 benchmark** | **≈ 328** |

That last line is the one to sit with: the entire 449-tile test benchmark rests on roughly
**330 distinct labelled trees**, each re-scored ~6× through overlapping tiles.

**Two independent estimates of the labelled share, and they agree at ~50%:**

| method | labelled share | per-tile spread |
|---|---|---|
| **area**: labelled crown area ÷ woody area (green canopy ∪ labelled crowns) | **54%** | median 54%, p10 **10%**, p90 86% |
| **count**: crown-sized green blobs overlapping a label ÷ all such blobs | **44–60%** | depends on the size floor (see below) |

Area accounting over those tiles: labelled crowns cover 21.0% of image area, ExG green
canopy 31.0%, their union ("woody") 38.9%.

**Why the count method is a range, not a number.** The blob floor was swept over the
labelled-crown area distribution: p10 floor → 44%, p25 → 53%, median → 60%. No floor
separates cleanly, because **small labelled crowns and shrubs are the same size in nadir
view** (labelled crown area spans p10 = 2 399 px² ≈ 55 px across to median 27 487 px² ≈
187 px across). A permissive floor counts shrubs as unlabelled trees (pessimistic); a strict
floor excludes most genuinely labelled crowns — at the median floor only 62 of 194 labelled
crowns are even represented — so it flatters completeness. Visual check:
`claude_outputs/savannatree_unlabelled_check.png` shows the permissive floor's "unlabelled"
blobs are largely shrubs and low understory, not trees.

**The honest bottom line:** roughly **half** the woody vegetation in the test site carries a
label, with enormous tile-to-tile variance (p10 ≈ 10%: a substantial minority of tiles have
almost nothing labelled, so per-tile precision is unrecoverable there however good the
detector is). A precise tree-level count is *not* obtainable from nadir RGB by us or by any
proxy — distinguishing tree from tall shrub from above is exactly the problem the authors
solved with on-ground survey, and is why the labels are species-confirmed rather than
exhaustive.

**Modal Volumes** (isolated; `rm -rf` this folder + both Volumes undoes everything):

| volume | contents |
|---|---|
| `savanna-raw-vol` | `tiles/<tid>.tif` — the 1464 source tiles we use (1024 px) |
| `savanna-phase4-vol` | `feat_4p_train/` (1015), `feat_4p_test/` (449), `out/` (ckpt, masker, preds, results) |

`tcd04-phase4-vol` and `modal_tcd_multiseed/` are never written.

## Dataset → phase4 contract

SavannaTree ships 2887 train tiles (sites S1/S5/S6/S7/S9/S10) + 449 validation tiles
(**site S4 only**) as 1024×1024 RGB TIFs with COCO polygons, single `tree` class.

**Split — site-disjoint, always.** The source tiling uses a 512 px step, so tiles overlap
50% and a *random tile split leaks*. Splits are therefore by site (the same lesson as the
Restor `validation_fold`):

    test  = S4                449 tiles  1 911 trees   (the released validation site)
    val   = S10               115 tiles    789 boxes   (detector early-stopping only)
    train = S1,S5,S6,S7,S9    900 tiles  4 701 boxes   (stratified subsample, seed 0)

`verify` asserts the three site sets are disjoint before any GPU spend.

**Two adaptations** (both in `savanna_data.py`, nothing else differs):

1. **1024 → 2048 upscale (×2 bicubic), all coordinates ×2.** phase4's geometry is pinned to
   a 2048 canvas at *every* layer — `phase4_features_tcd.CANVAS/GRID8`,
   `TargetConfig(grid=256, stride=8)`, `evaluate.RES=512`/`SCALE=4`, the masker's `s_px=8`
   and `cell_origin=s`, and the `+1px@512` raster shift. Upscaling the tile is the only
   adaptation that leaves all of them exactly valid. Re-deriving them for a 1024 canvas
   would re-open the half-cell registration surface that the grid fix closed — the bug that
   cost a full run on TCD. Cost: the frozen backbone sees each crown at 2×, and effective
   sampling is 4 px in native pixels.
2. **`canopy` is always `[]`** — no ignore class exists (see the caveat above).

**Train subsample = 900 tiles** (`--n-train`) — **a mistake, kept only because run 1 used
it.** It was chosen for storage/cost parity with TCD and justified by the dataset-wide
duplication factor (5.8×), on the claim that 900 tiles would still cover nearly every unique
crown. That inference is invalid — a dataset-wide duplication factor does not survive
subsampling. Measured by centroid-canonicalised polygon hashing (a crown re-tiled into a
neighbour matches itself), the true exact-shape duplication is only **~1.5×**:

| pool | tiles | annotations | distinct shapes | dup |
|---|---|---|---|---|
| train sites S1,S5,S6,S7,S9 | 2 772 | 12 129 | **7 919** | 1.53× |
| the 900-tile subset we extracted | 900 | 4 701 | **3 244 (41%)** | 1.45× |
| val S10 | 115 | 789 | 397 | 1.99× |
| test S4 | 449 | 1 911 | 1 400 | 1.36× |

So `n_train=900` threw away **~59% of the distinct training crowns** on an
already-supervision-poor dataset. Use `--n-train 0` (all 2 772) for any serious run; the
extra 1 872 tiles cost ~3.2 h of A100 and ~250 GB of Volume. *(Shape-hashing counts
edge-clipped variants of one tree separately — 9 715 shapes dataset-wide vs the authors'
2 547 trees — so it over-counts distinct trees, but is the right quantity for comparing
subsets.)*

## Commands

Built GT is committed (`savanna_*.json`); rebuild only to change the split:

```bash
python -m boxinst_commonality_tcd_04.modal_savanna_multiseed.phase4.savanna_data \
    --n-train 900 --seed 0
```

From this folder (`modal_savanna_multiseed/phase4/`):

```bash
modal run savanna_modal.py::fetch_data      # Zenodo zip -> raw tiles volume (CPU, ~10 min)
modal run savanna_modal.py::verify          # GT<->tile join + site disjointness (free)
modal run savanna_modal.py::extract_4p      # 4-phase L24 feats (A100)
modal run savanna_modal.py::train_4p --seed 0
modal run savanna_modal.py::fit_masker_4p --beta 0.5 --fix     # CPU
modal run savanna_modal.py::eval_selfmask --beta 0.5 --fix --save-preds
modal run savanna_modal.py                  # or the whole chain
```

Defaults are the settled TCD configuration: `beta=0.5`, `--fix` (geometry-corrected grid,
`cell_origin=s`), `mask_thr=0.25`, and the **masker's own stored knobs** — `em.fit` writes
`prior_weight`=α=0.3 and `kappa_scale`=κ=1.6 into the npz, and the eval's `-1.0` sentinel
defers to them. Pass `--prior-weight 1.0 --kappa-scale 1.0` for the vanilla ablation;
effective knobs tag the output filenames (`_pw030_ks160`) so runs never collide.
Multi-seed band: `::band_selfmask --seeds 0,1,2`.

**Ops:** cancel with `modal app stop <ap-id> --yes` — `pkill` only kills the local client and
leaves the A100 billing. The image pins `transformers 4.57` (matches the TCD run; never
bit-compare Modal features to the local 5.12 cache).

## Local experiment (2026-07-31) — recovering a training signal

Modal hit its spend limit mid-run, so everything below is **local** on an M4 Max (MPS for
training, CPU for the masker); only Modal Volume *reads* were used, which still work under
the cap. Scripts: `local_fetch_feats.py`, `build_ignore.py`, `local_train.py`,
`local_eval_test.py`, `rescore_val.py`, `zeroshot_tcd_local.py`.

### The two diagnosed causes of run 1's failure

**(a) Unlabelled crowns were trained as negatives — the dominant effect.** Measured over 30
train tiles: **10.7 crown-sized unlabelled green blobs per tile vs 6.0 labelled crowns
(1.77:1)**. `focal_heatmap_loss` pushes the heatmap down at every non-peak cell, so the model
was being trained to suppress the very pattern it must detect. A gradient unit-test makes the
magnitude concrete — at a confidently-detected unlabelled crown the negative gradient is
**+1.127 vs +0.000315 at a plain background cell, a 3 577× ratio** (focal loss deliberately
up-weights confident "errors"). With 10.7 such crowns per tile this term *dominated* the loss,
which is exactly why run 1 drove itself to maximum under-confidence (best-F1 pinned to the
bottom of the threshold sweep).

**(b) A mis-scaled size prior — real, but NOT the whole story.** The trained OAM-TCD detector
applied zero-shot to SavannaTree at the x2-upscaled 2048 canvas predicted median **48 px**
boxes against **427 px** GT (ratio 0.113), capping IoU at ~0.013. An earlier version of this
README concluded from that "a SCALE failure, not features or localisation" — **that was
wrong**, and re-running zero-shot at corrected scales disproves it:

| canvas | GSD | pred/GT width | box mAP50 | maxR | tiles |
|---|---|---|---|---|---|
| 2048 (x2 up) | 1 cm | 0.113 | 0.0004 | 0.053 | 12 |
| 512 | 4 cm | 0.180 | 0.0061 | 0.082 | 449 |
| **208 (matched to TCD)** | **10 cm** | **0.529** | **0.0025** | **0.049** | 449 |

Matching the GSD fixes the box SIZE (ratio 0.113 -> 0.529) but not the detection:
**zero-shot still finds <5% of crowns at TCD's own ground sampling.** So TCD->SavannaTree is
a genuine DOMAIN-recognition failure, not merely a scale artefact — consistent with the
asymmetric TCD<->NEON transfer, with savanna a much larger domain jump.

Useful consequence: our trained model's 0.483 recall should be read against a **0.049
zero-shot floor** — a ~10x gain. Training on savanna does essentially all the work.

When training FROM SCRATCH the size head self-corrects to predW/gtW ~1.0-1.4 within ~8
epochs even from the exp(3)=20 px init, so the size prior mattered far less than (a).

### The fix, and why it does not collapse

`ignore = vegetation AND NOT labelled-crown-box`, written into the `canopy` slot that
`focal_heatmap_loss` already consumes — i.e. exactly OAM-TCD's canopy-ignore mechanism,
synthesised. **Zero training-code changes.** Measured over all 1015 tiles:

| | fraction of cells |
|---|---|
| labelled crowns (positives) | 21.9% |
| **new ignore region** | **12.6%** |
| **hard negatives retained** | **65.6%** |
| tiles >50% ignored | **0** |

Bare ground, dry grass, burn scar and shadow stay hard negatives, so the degenerate
"predict foreground everywhere" solution is still penalised — the unit test confirms
background gradient is bit-identical with the mask on. Training logs `hm_pos%` and
`pred/tile` as live collapse guards.

**Training only.** `savanna_test_gt.json` keeps `canopy: []`; the released benchmark is never
modified.

### Are the features even good enough? (linear probe, held-out tiles)

| discrimination | AUC |
|---|---|
| labelled crown vs clear background (bare ground / dry grass) | **0.838** |
| labelled crown vs **unlabelled vegetation** | **0.770** |

This rules out the worst case. Label membership was decided by *on-ground species survey*, so
it could have been invisible in imagery; at AUC 0.77 it is not — labelled crowns are
genuinely distinguishable (larger, greener trees vs shrubs). **The frozen DINOv3 features
carry the signal; the failure was the training signal, not the backbone.** It also sets a
realistic ceiling: at 0.77 separability a meaningful share of false positives is unavoidable,
so precision here is bounded by the data.

### Val metric is contaminated too

The val site (S10) has the same ~50% label sparsity, so strict val AP penalises the model for
finding unlabelled trees. Since `eval_4p_selfmask` already applies canopy-ignore via
`Ign_box` on TCD, doing the same on val is the faithful analogue, not a concession.
`local_train.py` reports **both** and selects on the ignore-aware score. Diagnostic value:
early in training AP50-ignore-aware sat only ~17% above strict, showing most false positives
were *not* on unlabelled vegetation — i.e. genuinely under-trained rather than merely
mis-scored.

### Caveat: the 14-epoch ablation was inconclusive

A 2x2 grid (ignore x size-prior) at 14 epochs on 414 tiles put every arm in the same
0.004–0.013 band — noise, because nothing had learned yet (A1's recall was still climbing at
the last epoch). It was abandoned for one long run rather than reported as an ablation.

## Session 3 (2026-07-31) — the benchmark IS the experiment

Everything here is local on an M4 Max (MPS), ~6 min per 60-epoch run. New scripts:
`rank_diag.py`, `fp_decomp.py`, `rescore.py`, `rank_transfer.py`, `label_density.py`,
`clump_check.py`, `clump_viz.py`, `fetch_more_tiles.py`, `collate.py`, `run_border.sh`,
`run_seedband.sh`.

The brief was: our P/R reach ~80% of the paper's but our mAP only 35%, therefore the deficit
is **confidence ranking** — so fix the score. That inference is wrong, and four independent
measurements say the binding constraint is the **annotations**.

### 1. The score ranks fine. It is ranking a contaminated pool.

`rank_diag.py`, held-out S4, 449 tiles, 15 228 detections, 1 911 GT:

| quantity | value |
|---|---|
| AUC of the score separating TP from FP @IoU0.5 | **0.848** |
| precision in the top score decile | **0.207** |
| AP50 as-is / oracle re-rank (sort by true IoU) / random rank | 0.121 / **0.549** / 0.040 |
| corr(score, IoU) among matched detections | 0.117 |
| median IoU of a true positive | 0.631 |

A score with AUC 0.85 is not a broken ranker. The ceiling is that even the **most confident
decile is only 20.7% correct**, and AP integrates precision at low recall, so that ceiling —
not the ordering — is what caps mAP.

### 2. What the confident false positives ARE (`fp_decomp.py`)

Same detections, classified by what lies under the box (ExG vegetation, the same recipe the
training ignore mask uses — diagnostic only, never applied to the test metric):

| subset | TP | duplicate | near-miss | **unlabelled vegetation** | bare ground |
|---|---|---|---|---|---|
| all 15 228 | 7.0% | 0.3% | 8.0% | 42.4% | 42.4% |
| **top 10% by score** | **20.7%** | 0.2% | 10.5% | **59.4%** | **9.3%** |
| score ≥ 0.30 | 19.3% | 0.0% | 9.2% | **66.4%** | 5.2% |

**Three quarters of the confident non-matches are real crowns with no polygon**, and only
9.3% are on bare ground. Forgiving the unlabelled-vegetation hits would put top-decile
precision at **0.509** instead of 0.207. Duplicates are negligible (0.2%), so NMS and peak
suppression are not the problem either.

### 3. Precision tracks label completeness; recall does not (`label_density.py`)

The 449 test tiles split into quartiles by labelled share (labelled crown area ÷ woody area).
Same detections re-scored within each quartile:

| quartile | tiles | labelled share | AP50 | P@IoU.5 | R@IoU.5 |
|---|---|---|---|---|---|
| Q1 | 112 | 0.147 | 0.042 | 0.093 | 0.324 |
| Q2 | 112 | 0.545 | 0.094 | 0.158 | 0.489 |
| Q3 | 112 | 0.790 | 0.172 | 0.274 | 0.528 |
| **Q4** | 113 | **0.950** | **0.210** | **0.338** | 0.479 |

Precision rises **3.6×** and AP50 **5×** with labelled share while **recall stays flat**
(0.49 across Q2–Q4). That is the exact signature of sparse annotation and it excludes the
alternative that well-labelled tiles are merely easier imagery — easier imagery would lift
recall too. On the best-labelled quartile our precision (0.338) already exceeds the paper's
whole-set 0.255, though the paper's number on that same quartile would rise as well.

### 4. "Which crown was surveyed" does not transfer between sites

If unlabelled crowns are what caps precision, the obvious fix is a head that learns to rank
surveyed crowns above unsurveyed ones — the brief's "higher-capacity classification head".
`rescore.py` builds exactly that: the frozen detector's boxes and NMS are untouched, so only
the ORDER changes, and a second head is trained with unlabelled vegetation as NEGATIVES
(the training ignore mask is what removes that gradient from the main heatmap).

**It does not work.** Best achievable test AP50 over the blend sweep is 0.131 vs 0.121
baseline (+0.010); validation selects a blend weight giving **0.1202, i.e. no gain at all**.

`rank_transfer.py` says why, and rules out "not enough capacity":

| epoch | AUC on train sites | AUC on val site S10 | gap |
|---|---|---|---|
| 3 | 0.802 | 0.654 | +0.148 |
| 15 | 0.939 | **0.682** (best) | +0.257 |
| 30 | **0.993** | 0.668 | **+0.326** |

The head memorises the training sites' survey almost perfectly (0.993) while cross-site
separability plateaus at 0.68 by epoch 15 and then *declines*. Being labelled is a property
of **where the ground crew walked**, not of the pixels. The README's earlier linear-probe
figure (AUC 0.770, "the features carry the signal") was measured on held-out *tiles* — which
overlap 50% and share a site — so it is a within-site number and must not be read as
evidence that rescoring can help on a held-out site.

### 5. The GT is mostly crown FRAGMENTS, and the benchmark rewards reproducing that

The 512 px tiling step cuts crowns at every seam:

| | border-clipped share | median \|log(w/h)\| interior → clipped |
|---|---|---|
| train/val boxes (5 472) | **51.8%** | 0.152 → **0.545** |
| **test GT trees (1 911)** | **72.9%** | 0.170 → **0.501** |

3.6× the aspect distortion, and no box extends past the tile — they are truncated, not merely
adjacent. Max recall @IoU0.5 is **0.678 on whole crowns vs 0.510 on clipped ones**.

Truncated boxes are bad supervision (wrong size target, peak at the fragment's centroid, not
the crown's), and the 50% overlap means a crown cut in one tile is whole in its neighbour —
so dropping them should be nearly free. **It is not.** Two seeds each:

| arm | treatment of clipped GT boxes | mAP50 seed 0 / 1 | mean |
|---|---|---|---|
| **keep** | as-is (published recipe) | 0.1206 / 0.0516 | **0.086** |
| noreg | keep the heatmap peak, drop the box extent | 0.0586 / 0.0696 | 0.064 |
| ignore | neither positive nor negative | 0.0282 / 0.0242 | **0.026** |

Cleaner supervision loses, decisively for `ignore` (−0.060, outside any noise band). The
reason is structural: 73% of the *test* GT is fragments too, so a model taught to emit whole
crowns cannot match them. **Optimising this benchmark and optimising crown delineation point
in different directions** — worth stating plainly in any write-up.

### 6. Some polygons outline GROUPS of trees, not individuals (`clump_check.py`)

Two independent signals, one model-free (polygon shape) and one model-based (how many
well-separated confident detections fall inside one polygon), on interior (non-truncated)
test GT:

| group | n | solidity | median width | median best IoU |
|---|---|---|---|---|
| interior GT, 0–1 detection inside | 460 | 0.955 | 86.5 px | 0.588 |
| **interior GT, 2+ separated detections** | **58 (11.2%)** | **0.891** | **171.0 px** | 0.577 |

The flagged polygons are **2× wider and measurably less convex**, and the two signals agree.
Visual confirmation in
[`claude_outputs/savannatree_clump_candidates_s512.png`](../../../claude_outputs/savannatree_clump_candidates_s512.png):
the low-solidity cases (0.67–0.78, 7–13 m across) clearly wrap several separate canopies with
bare ground — and in one case a road — between them. Interior polygons also fill only **62%**
of their bounding box (a single ellipse gives ~79%), so the outlines are loose as well.

Scale, honestly: ~11% of interior and ~15% of clipped GT are clump-flagged, contributing on
the order of a few hundred of the 14 166 false positives. **Real, corroborated, and much
smaller than the unlabelled-crown effect** — it is a correctness problem for the dataset's
ITC claim more than an explanation of our metric gap.

### 7. The number that invalidates single-seed comparison

Two seeds of the **identical** published config: **mAP50 0.1206 and 0.0516** (P 0.209/0.111,
R 0.483/0.265). Worse, the seed that scored *higher* on validation (0.0711 vs 0.0627 val
AP50-ignore-aware) is the one that scored **2.3× worse on test**. The val site S10 does not
rank checkpoints for the test site S4, so the published 0.1206 is a lucky draw, not a
selected optimum.

Consequences: the scale sweep (208/336/512/1024 → 0.127/0.105/0.121/0.086) is single-seed and
its whole spread sits inside the ±0.049 seed band, so **"inverted U peaking at 512" is
withdrawn** — unsupported, not disproven. Nothing on this dataset should be quoted from one
seed again; `collate.py` reports mean ± sd by config.

### 8. Fix the selection rule and most of the "variance" disappears

`run_seedband.sh` retrains the published config at five seeds and additionally checkpoints at
fixed epochs 6/10/16, so *epoch choice* and *training run* can be separated. Held-out S4:

| selection rule (5 seeds each) | mAP50 mean ± sd | min – max | mean R@IoU.5 |
|---|---|---|---|
| **val-selected** (`ap50_ign` on site S10, the published rule) | 0.067 ± **0.051** | 0.019 – 0.123 | 0.282 |
| fixed epoch 6 | 0.044 ± 0.021 | 0.024 – 0.075 | 0.254 |
| fixed epoch 10 | 0.093 ± 0.034 | 0.037 – 0.123 | 0.447 |
| **fixed epoch 16** | **0.103 ± 0.025** | 0.060 – 0.124 | **0.459** |

**Just training to a fixed epoch and stopping beats selecting on validation** — epoch 16
gives a +0.036 higher mean and half the spread. Validation is not merely uninformative, it is
actively harmful: it repeatedly lands on epochs 2–6, which are reliably terrible (two seeds
selected epoch 2 and scored 0.019 and 0.022).

Two honest caveats. First, an intermediate 3-seed snapshot of this same table showed epoch 10
at 0.102 ± 0.013 and I nearly wrote it up as a ~70× variance reduction; the 4th seed (0.037)
destroyed that. **On this dataset even a 3-seed conclusion is not safe** — which is itself
the most transferable result here. Second, epoch 16 was not chosen a priori; with 5 seeds and
3 candidate epochs it is a selected maximum and its +0.036 should be treated as an upper
estimate.

The robust claim is the weaker one: **the residual spread (sd ~0.025–0.05 on a mean of
~0.07–0.10) is comparable to every effect this folder has ever reported**, so no single-seed
A/B on SavannaTree means anything, whatever the selection rule.

### 9. The lever that actually worked: use all the training data

`n_train=900` discarded ~59% of the distinct labelled crowns (see the note above). All 1 860
missing tiles were pulled from Zenodo by zip range-request (`fetch_more_tiles.py`, ~35 min, no
Modal — the raw Volume never held them and the staged zip is gone), features extracted at
0.26 s/tile, and the recipe retrained unchanged on all **2 772** train tiles:

| | train tiles | boxes | mAP50 | P@IoU.5 | R@IoU.5 | F1@IoU.5 |
|---|---|---|---|---|---|---|
| published recipe, **5 seeds** | 900 | 4 701 | 0.067 ± 0.051 (max 0.123) | 0.138 | 0.282 | — |
| **all train tiles, 2 seeds** | **2 772** | **12 100** | **0.159 ± 0.011** (0.1668, 0.1513) | **0.250** | 0.484 | **0.329** |
| + label-density filter (§10, 1 seed, 4× cheaper) | 1 805 | 10 104 | 0.167 | 0.254 | 0.513 | 0.340 |
| paper (Mask R-CNN, full mask sup.) | 2 885 | — | 0.345 | 0.255 | 0.614 | 0.360 |

This is the only change all session that clears the noise band by a wide margin: **both
full-data seeds (0.1668, 0.1513) exceed the maximum of all five 900-tile seeds (0.1226)**, so
it does not depend on a selection rule or a lucky draw. Precision now **matches the paper's**
(0.250 vs 0.255) at the same IoU, F1 reaches 91% of it, and mAP 46% (was 35%). Recall remains
the gap (0.484 vs 0.614). At IoU 0.4, seed 0 gives P 0.282 / R 0.611.

Cost: ~35 min of Zenodo range-fetch, 8 min of extraction, 29 min of training per seed. No
Modal, no method change, no new hyperparameter.

Read alongside §3 this is not a contradiction but the other half of the same story: label
sparsity caps *precision*, and we were simultaneously nowhere near even that cap because most
of the available labels were unused. Sparse annotation is the ceiling; throwing away 59% of
what exists was the self-inflicted part.

Note the interaction with §8 — with 3× the data, val-selection stopped being catastrophic
(it picked ep8/ep4 and beat both fixed epochs, 0.167/0.151 vs 0.135/0.137). More data
stabilises training, which is consistent with the instability being a small-sample effect.

### 10. Train where the ANNOTATION IS GOOD (user's idea) — precision + efficiency, not headline mAP

Rather than neutralising bad annotation with an ExG proxy at cell level, drop the tiles whose
annotation is bad. Per-tile label density = labelled cells ÷ (labelled + unlabelled-vegetation
cells) is already inside the cached ignore masks, so filtering on it costs nothing. The trade
is cheaper than it looks because **well-annotated tiles carry more trees** — D ≥ 0.5 drops 34%
of tiles but only 16% of the boxes (1 805 tiles / 10 104 boxes, 5.60 per tile vs 4.38).

Arms at **matched tile-visits** (44 000 ≈ 16 epochs of the full set), not matched epochs — a
subset takes fewer optimiser steps per epoch, and equal-epochs would confound annotation
quality with training length. Scored at the **end of the run**, the only rule that involves no
selection at all (see the warning below):

| training tiles | veg ignore mask | mAP50 | P@IoU.5 | R@IoU.5 | F1 |
|---|---|---|---|---|---|
| all 2 772 | ON | 0.147 | 0.214 | 0.514 | 0.302 |
| **density ≥ 0.5** (1 805) | ON | **0.167** | 0.254 | 0.513 | **0.340** |
| density ≥ 0.7 (1 328) | ON | 0.165 | 0.262 | 0.459 | 0.333 |
| **density ≥ 0.7** (1 328) | **OFF** | **0.167** | **0.296** | 0.401 | **0.340** |
| density ≥ 0.9 (650) | OFF | 0.133 | 0.256 | 0.368 | 0.302 |
| *§9 best (all tiles, 60 epochs = 166k visits)* | ON | *0.167* | *0.259* | *0.463* | *0.332* |
| *paper, full mask supervision* | — | *0.345* | *0.255* | *0.614* | *0.360* |

**What holds.**

1. **A monotone precision/recall trade across four arms**: filtering harder and dropping the
   mask moves precision 0.214 → 0.254 → 0.262 → **0.296** while recall falls 0.514 → 0.401.
   Mechanistically coherent — cleaner tiles plus restored hard negatives suppress the
   false positives that §2 showed are 59% unlabelled crowns — and it is the one trend that
   survives a fixed checkpoint rule.
2. **~4× cheaper for the same result.** The filtered arm matches §9's best (0.167) using 66% of
   the tiles and **44k visits against 166k** (8 min vs 28 min).
3. **Over-filtering is real.** D ≥ 0.9 collapses to 0.133 — 650 tiles / 4 745 boxes is back
   below the 900-tile regime. Optimum is D ≈ 0.5–0.7, not "cleaner is always better".

**What does NOT hold, and a warning.** Density filtering does **not** beat §9 on mAP50 — both
land at 0.167. An earlier version of this section reported **0.199** and "58% of the paper's
mAP"; that was the *val-selected* checkpoint of the D ≥ 0.5 arm, and across these five arms the
val-selected value swings **0.060 – 0.199 with no relation to arm quality** (the D ≥ 0.7 arm's
val pick, 0.097, is its own worst checkpoint). It is withdrawn. This is the third time in this
folder that a selected maximum masqueraded as an effect — §8 is the general statement of the
problem, and the rule it implies is: **compare arms only at a fixed, unselected checkpoint.**

Single seed throughout, so even the trends above are indicative. Untested and suggested by the
trend: D ≈ 0.5–0.6 with the ignore mask OFF (the better threshold combined with the better
negative handling), which no arm here covers.

### 11. Recall: +0.080 for free at inference (`recall_levers.py`)

Recall is the only metric sparse annotation cannot contaminate, so it is the only place a real
improvement is measurable. It decomposes into two separate problems, and §5 says where the
bigger one lives:

    R at the operating point   0.463      <- ~0.17 lost to thresholding/ranking
    maxR @IoU0.5               0.631      <- ~0.37 never detected well enough at all
    maxR on whole crowns       0.687   vs   maxR on border-clipped GT   0.609

Two inference-only corrections follow directly, neither touching a weight:

1. **Clip predictions to the tile.** 73% of test GT is truncated at the seam, but the decoder
   happily emits boxes running past the tile edge, so a correctly-found edge crown is scored
   against a fragment it cannot match. The GT is already clipped; the prediction should be too.
2. **Global box-size scale.** The size head was measured drifting ~30% small and AP40 ≫ AP50 is
   a size signature, so sweep one scalar on (w,h).

Scale and clip chosen on the **val site by recall**, then read off on test (full sweep in the
json so the cost of that choice is visible; val preferred 1.10 + clip, and test's own optimum
is 1.10–1.20 + clip, so the selection was not lucky):

| variant | AP50 | **R@IoU.5** | maxR@.5 | P@IoU.5 | AP40 | R@IoU.4 |
|---|---|---|---|---|---|---|
| baseline (§9 best) | 0.1668 | 0.463 | 0.631 | 0.259 | 0.2332 | 0.611 |
| clip only | 0.1707 | 0.469 | 0.640 | 0.262 | 0.2348 | 0.613 |
| scale 1.10 only | 0.1710 | 0.523 | 0.654 | 0.245 | 0.2412 | 0.626 |
| **scale 1.10 + clip** (val-selected) | **0.1845** | **0.543** | **0.681** | 0.254 | **0.2499** | 0.636 |

**+0.080 recall and +0.018 AP50 with precision unchanged (0.259 → 0.254), for zero training.**
Both parts are real and additive: the size correction contributes +0.060, clipping +0.020.

The mechanism check confirms the diagnosis rather than just the outcome — clipping was supposed
to help *fragments specifically*, and it does: maxR on border-clipped GT rises **0.609 → 0.673**
while whole crowns barely move (0.687 → 0.703), closing most of the fragment/whole gap that §5
identified.

Against the paper this puts recall at **88%** of theirs (0.543 vs 0.614, was 75%) with precision
level (0.254 vs 0.255) and F1 at ~96% (0.346 vs 0.360). mAP50 remains the outlier at 53%.

Caveat: the val site under-reported the size of this effect badly (+0.011 there vs +0.080 on
test), so val is usable for *choosing* the constant but not for estimating what it buys. Single
seed. And a global ×1.10 is a symptom fix — the size head is biased and should be corrected in
training (`w_size`, size-head bias), which would likely subsume it.

### 12. The SOTA campaign — objective, and why nothing may be tuned on val

**Objective.** "Beat the paper" was never well posed as AP. The target is a single operating
point with **recall > 0.614 AND precision >= 0.255**, so the scored quantity is

    R@P255 = the highest recall reachable at any score threshold whose precision is still >= 0.255

Best-F1 answers a different question (it trades recall away happily) and AP is contaminated by
label sparsity (§2–§3). `sota.py` scores R@P255 first and reports AP/F1 alongside.

**Val cannot tune anything, and this is now measured twice.** §8 showed S10 does not rank
checkpoints for S4. It also does not rank *inference knobs*: sweeping the box-size scale, the
val-side objective is **anti-correlated** with test —

| box-size scale | val R@anchor | test R@P255 |
|---|---|---|
| 1.00 | **0.2752** | 0.4741 |
| 1.10 | 0.1847 | **0.5369** |
| 1.20 | 0.1134 | **0.5395** |
| 1.30 | 0.0815 | 0.5055 |

val prefers exactly the setting test likes least. (Mechanism: S10's crowns are ~2× smaller than
S4's, so a size knob fitted there transfers backwards. Note also that S10 **cannot reach
precision 0.255 at all** — its R@P255 is ~0.008 for every setting — so even the objective is
degenerate there.)

**Consequence — the campaign selects nothing.** Every arm is read at a **fixed end-of-run**
checkpoint (`det_<tag>_final.pt`, always written now), at matched tile-visits, and the only two
inference adjustments are ones that require no held-out selection at all:

1. **Size-bias correction fitted on the TRAINING sites** — median(gtW/predW) over matched pairs,
   12 100 boxes rather than S10's 785. This is a *fit* of a known bias (the size head
   under-predicts; AP40 ≫ AP50), not a search over a metric. It returns **1.082**, which happens
   to sit right at the test optimum (1.1–1.2) — arrived at without touching test.
2. **Clipping predictions to the tile** — a correctness fix, not a knob: 73% of test GT is
   seam-truncated and the decoder emits boxes past the tile edge, which cannot match.

Baseline under this discipline: **R@P255 = 0.522** (full2772, scale 1.082, clip).

**The main lever is ensembling, for a specific reason.** maxR is already 0.677 while R@P255 is
0.522 — ~0.15 of recall is sitting in the candidate pool *below the precision-feasible
threshold*. So the deficit is a RANKING problem at the operating point, not a missing-candidate
problem, and weighted box fusion attacks exactly that: it scores a box by how many independent
models agree on it, pushing real crowns up the ranking so the threshold can drop without
precision collapsing. Early evidence — fusing full2772 with a deliberately under-trained
1-epoch model still moved R@P255 0.522 → 0.568, maxR 0.677 → 0.724, F1 0.345 → 0.367.

`run_overnight_sota.sh` runs: w_size arms (fixing the size head at source rather than
post-hoc), the D ≈ 0.5–0.6 × mask-off cells §10 left untested, extra seeds for ensemble
diversity, and 336/1024-canvas arms for scale diversity — then fuses them.

**Reading the results table:** it is an exploratory sweep, so its best row is a *selected
maximum* over ~20 configs against a ±0.05 seed band. That is a candidate to re-run at fresh
seeds, not an unbiased estimate — the summary printer says so too.

### 13. RESULT — the PR curve now passes above the paper's operating point

Overnight campaign (`run_overnight_sota.sh`, ~3 h local on MPS), everything at a fixed
end-of-run checkpoint, matched tile-visits, train-fitted size scale + tile clipping, **no
val or test selection of any knob**.

**Single models** (R@P255 = recall at the deepest threshold still holding precision ≥ 0.255):

| arm | R@P255 | maxR | bestF1 (P / R) | AP50 |
|---|---|---|---|---|
| all tiles, ignore, w_size 0.1 | 0.002 | 0.676 | 0.309 (0.219 / 0.525) | 0.153 |
| all tiles, ignore, w_size 0.3 | 0.463 | 0.672 | 0.330 (0.254 / 0.473) | 0.168 |
| all tiles, ignore, **w_size 1.0** | 0.513 | 0.694 | 0.341 (0.258 / 0.504) | 0.176 |
| D ≥ 0.5, **mask off** | 0.509 | 0.618 | 0.354 (0.297 / 0.439) | 0.182 |
| D ≥ 0.6, **mask off** | 0.499 | 0.621 | 0.353 (0.285 / 0.466) | 0.189 |
| D ≥ 0.5, ignore | 0.529 | 0.650 | 0.346 (0.259 / 0.523) | 0.173 |
| all tiles, 2× visits | 0.524 | 0.649 | 0.346 (0.261 / 0.515) | 0.170 |
| canvas 1024, ignore, w_size 0.3 | 0.011 | 0.741 | 0.310 (0.218 / 0.539) | 0.164 |
| **canvas 336**, ignore, w_size 0.3 | **0.589** | 0.683 | **0.375** (0.282 / 0.560) | **0.205** |
| *§9 baseline (full2772)* | *0.522* | *0.677* | *0.345 (0.253 / 0.540)* | *0.182* |

The D ≈ 0.5–0.6 mask-off cells §10 left untested behave exactly as §10 predicted — best
precision of any single 512 model (0.297 / 0.285) but the lowest maxR, so they cap out on this
objective. And **canvas 336 beats every 512 arm**, which further undermines the withdrawn
"inverted U peaking at 512" (§7).

**Ensembles (weighted box fusion):**

| ensemble | models | R@P255 | maxR | bestF1 (P / R) | AP50 |
|---|---|---|---|---|---|
| 3 seeds of one config | 3 | 0.571 | 0.743 | 0.367 (0.288 / 0.506) | 0.210 |
| 3 seeds of another | 3 | 0.592 | 0.735 | 0.389 (0.329 / 0.476) | 0.230 |
| **512 + 336 (scale diversity)** | **2** | **0.625** ✅ | 0.746 | **0.406** (0.331 / 0.526) | 0.248 |
| **512 + 336 + 1024 (3 scales)** | **3** | **0.639** ✅ | 0.793 | 0.401 (0.326 / 0.522) | 0.252 |
| 6 seeds | 6 | 0.606 | 0.787 | 0.394 (0.321 / 0.512) | 0.238 |
| **6 seeds + 336** | 7 | **0.626** ✅ | 0.812 | 0.403 (0.323 / 0.534) | 0.251 |
| 9 models | 9 | 0.617 ✅ | 0.812 | 0.399 (0.329 / 0.506) | 0.252 |
| 6 seeds + 336 + 1024 | 8 | 0.631 ✅ | **0.830** | 0.402 (0.334 / 0.504) | 0.251 |
| *paper* | 1 | *0.614* | — | *0.360 (0.255 / 0.614)* | *0.345* |

**Our precision/recall curve now passes above the paper's reported point**: at precision 0.255
we reach recall **0.639** (best: 512+336+1024) against their 0.614, and best-F1 is **0.406 vs 0.360**. AP50 (0.252)
remains well below theirs (0.345) — consistent with §2–§3, since AP integrates the
sparsity-contaminated low-recall region hardest.

**Scale diversity ≫ seed diversity, and it is much cheaper.** Two models at different canvases
(0.625) beat six seeds at one canvas (0.606), and three canvases (0.639) beat all eight-model
mixtures. Strikingly, the 1024 model is USELESS alone — R@P255 0.011, because its precision
never clears 0.255 — yet it is the best thing to add to the ensemble (0.625 → 0.639): it
contributes candidates at crown sizes the coarser grids miss (maxR 0.741 solo) and fusion
supplies the ranking it lacks. Seeds average away initialisation noise; different
canvases resolve different crown sizes, which is the diversity that actually matters when crown
size spans an order of magnitude. Practical consequence: two models, not six.

**Why ensembling was the right lever** — predicted in §12 and confirmed: maxR was already 0.677
while R@P255 was 0.522, so the recall was sitting in the pool below the precision-feasible
threshold. Fusion is a *ranking* fix (agreement across models raises confidence), and indeed
both maxR (0.677 → 0.812) and the usable operating point moved.

### ⚠️ Two limits on the headline claim

1. **The threshold is chosen using test labels.** R@P255 asks "does a feasible operating point
   exist", and the answer is yes — but *finding* it needs the test PR curve. Calibrating the
   threshold on the training sites instead (no test knowledge) gives **P 0.212 / R 0.660** for
   the 512+336 ensemble: the precision constraint is missed. So the defensible claim is **"our
   PR curve dominates their reported point"**, not "we have a deployable model at P 0.255".
   Closing that needs a calibration set matched to S4's label density, which §8/§12 show S10 is
   not. This is the single most important open item.
2. **Selected maximum.** ~20 configs were scored against a ±0.05 seed band. Three ensembles
   clear 0.614 (0.625 / 0.626 / 0.617), which is more robust than one, but the winner should be
   re-run at fresh seeds before being quoted as an unbiased number.
3. **R@P255 is a cliff-edge metric.** Two arms score ~0.00 not because they detect nothing but
   because their PR curve never clears precision 0.255 at all (the 1024 arm has maxR 0.741 and
   R 0.539 at best-F1, yet R@P255 = 0.011). Never read a low R@P255 as "this model is bad" —
   read the maxR and best-F1 columns beside it.

Also still outstanding: these are **box** metrics against the paper's **instance-segmentation**
numbers (§ "Where this leaves the comparison"). The masker has not been run on any of these.

### 14. Box → mask: the like-for-like comparison, and it is WORSE than the box one

§13's headline is a **box** number; the paper's 0.255 / 0.614 / 0.345 are **instance
segmentation**. Running the settled masker (β = 0.5, `mask_thr` 0.25, the masker's own stored
α = 0.3 / κ = 1.6 — all defaults, nothing re-tuned) closes that gap in the reporting, and the
answer is not favourable.

**The box → mask step costs performance everywhere it was tried:**

| model | grid | box mAP50 | mask mAP50 |
|---|---|---|---|
| single, canvas 512 | 6.7 cells/crown | 0.164 | **0.134** |
| single, canvas 1024 | 13.4 cells/crown | 0.151 | **0.122** |
| ensemble, masks @512 | — | 0.252 | **0.202** |
| ensemble, masks @1024 (hybrid) | — | 0.252 | **0.192** |

**Ensemble, masks at 512 — the best mask configuration — against the paper:**

| metric @ IoU 0.5 | paper (Mask R-CNN, full mask sup.) | ours (mask) | ours (box, §13) |
|---|---|---|---|
| best-F1 | 0.360 | **0.363** (P 0.289 / R 0.489) | 0.401 |
| recall at precision 0.255 | 0.614 | **0.537** ❌ | 0.639 ✅ |
| mAP50 | 0.345 | 0.202 | 0.252 |

**So the correct like-for-like verdict is: level on F1 (0.363 vs 0.360), BEHIND on recall at
their precision (0.537 vs 0.614) and on mAP (0.202 vs 0.345).** §13's "our PR curve passes
above their operating point" holds for boxes and **does not hold for masks** — which is the
comparison that matches what they actually reported. The box result is still meaningful (it
says the detector finds and localises crowns competitively) but it is not the headline against
this paper.

**Two earlier claims die here.**

1. **"The masker wants a finer grid than the detector"** (README above, and the motivation for
   the whole detect@512 / mask@1024 hybrid) **does not replicate.** It was measured against a
   900-tile detector whose boxes were weak enough for the EM masks to beat them. With the
   full-data detector, masking at 1024 is *worse* than at 512 (0.192 vs 0.202), and the hybrid
   is the worst mask configuration tried. The masker never beat the box on any current model.
2. The masker's value on OAM-TCD (where mask beat box) does **not** transfer here. Plausible
   reason, untested: TCD crowns fill their boxes in closed canopy, whereas savanna GT polygons
   fill only **62% of their bounding box** (§6) and are loose, irregular hand-drawn outlines —
   so a tight, well-carved mask is *penalised* against a slack polygon, while a plain box is not.

Cheapest untested lever if masks matter: `mask_thr` and the α/κ box-robustness knobs were left
at their TCD-calibrated defaults, and §6 says this dataset's polygons are systematically looser
than TCD's — a lower `mask_thr` (fatter masks) is the obvious first thing to sweep.

### 15. Overnight 2 — six scales, mask knobs, and an unresolvable metric ambiguity

**The budget constraint that shaped the run.** Every mask inherits its box's match, so box AP50
is a hard ceiling on mask AP50. At the start: box 0.252, mask 0.205 (81% of ceiling) — only
**+0.047** was available from the masker against a **+0.140** gap to the paper. Two thirds of
the gap therefore had to come from *detection*, so most of the night went there.

**Detection: more canvases, again.** Added 208 / 672 / 768 to 336 / 512 / 1024 and fused all six
(weighted box fusion, fixed end-of-run checkpoints, matched visits, train-fitted size scale):

| detector | R@P255 | best-F1 (P / R) | maxR | box AP50 |
|---|---|---|---|---|
| 3 canvases (§13) | 0.639 | 0.401 (0.326 / 0.522) | 0.793 | 0.252 |
| 4 canvases | 0.655 | 0.405 | 0.807 | 0.257 |
| 5 canvases | 0.659 | 0.399 | 0.813 | 0.264 |
| **6 canvases** | **0.664** | **0.427** (0.359 / 0.525) | **0.824** | **0.289** |
| 6 canvases + a seed | 0.658 | 0.419 | 0.825 | 0.275 |

Scale diversity holds up a third time, and **adding a seed to the six-scale mixture makes it
worse** (0.275 vs 0.289) — seeds are not just weaker than scales, they dilute. Note also that
the coarsest single model (canvas 208) has the best solo F1 (0.387) and solo AP (0.233) while
having the *worst* maxR (0.583): coarse grids give few but precise detections.

**Masks: knobs re-derived on the calibration set** (α, κ, `mask_thr` selected on train-site
tiles, never val/test; winner α 0.3, κ 1.0, thr 0.40):

| metric @ IoU 0.5 | paper | ours (mask) | ours (box) |
|---|---|---|---|
| best-F1 | 0.360 | **0.381** ✅ (P 0.321 / R 0.471) | **0.427** ✅ |
| recall at precision 0.255 | 0.614 | 0.560 ❌ | **0.664** ✅ |
| AP50 (pooled) | 0.345 | 0.230 ❌ | 0.289 ❌ |
| AP50 (per-image mean) | 0.345 | **0.364** ✅ | **0.429** ✅ |

The mask knobs were worth +0.025 AP (0.205 → 0.230) — close to the +0.047 ceiling estimate,
so that lever is now essentially exhausted. `mask_thr` moved only 0.25 → 0.40 (+0.003); the
hypothesis in §14 that these loose polygons wanted *fatter* masks was **wrong, and backwards** —
performance falls monotonically below 0.25 and peaks at 0.40–0.50.

### ⚠️ ~~The mAP comparison cannot be settled from the published paper~~ — RESOLVED in §16, see there first

On **identical detections**, the two standard AP aggregations differ by **1.5–1.6×**:

| | pooled (one PR curve over all tiles) | per-image mean (VOC/AutoML alternative) |
|---|---|---|
| box AP50 | 0.289 | **0.429** |
| mask AP50 | 0.230 | **0.364** |
| *paper reports* | *0.345* | *0.345* |

**The paper's number sits between our two conventions**, so which side of it we land on is
decided entirely by an aggregation choice we cannot read: their notebook never sets
`validation_metric_type`, so it is an AutoML default. This is not a rounding-level caveat — it
is larger than every modelling gain in this folder combined. On 4.26 GT/tile, per-image
averaging is hugely more forgiving (a tile with one GT and one correct top-ranked detection
scores AP 1.0 on its own).

**So: we do NOT categorically exceed the paper.** **§16 resolves the metric question from their
source code and the verdict gets worse, not better** — the per-image reading is refuted, and the
F1 comparison above is not apples-to-apples. Read §16.

### 16. AMBIGUITY RESOLVED from their code — and it goes against us

Cloned `ajansenn/SavannaTreeAI` and read `azureml-automl-dnn-vision` (v1.62.0) rather than
guessing. Two things had to be established: which run produced the paper's numbers, and what
AutoML's default metric actually computes.

**Which run.** The repo's `AutoML_Instance_Segmentation.ipynb` contains two configs: a
default-hyperparameter run (`iterations=1`) and a 10-iteration sweep over
`learning_rate ~ U(1e-4, 1e-3)`, `optimizer`, `min_size`. The paper reports **lr 0.005, batch 2**
— which matches neither the sweep nor the Detectron2 notebooks (`BASE_LR 0.00025`,
`IMS_PER_BATCH 2`, `NUM_CLASSES 38`). It matches **AutoML's built-in defaults for
`maskrcnn_resnet50_fpn`**, so the reported baseline is the default run, and every metric
setting is an AutoML default.

**What that metric is** (read from source, not documentation):

| setting | value | source |
|---|---|---|
| `validation_metric_type` | **VOC** | `constants.py` `settings_defaults` |
| `validation_iou_threshold` | **0.5** | `DEFAULT_VALIDATION_IOU_THRESHOLD` |
| `box_score_thresh` / `nms_iou_thresh` | 0.3 / 0.5 | `DEFAULT_BOX_SCORE_THRESH` |
| 11-point AP? | **No** — `_use_voc_11_point_metric = False` | `incremental_voc_evaluator.__init__` |
| aggregation | **POOLED over all images** (`np.concatenate` of per-image tp/fp labels and scores per class; `image_indexes=None`) | `incremental_voc_evaluator.compute_metrics` |
| AP formula | rectangular AUC under the precision **envelope** (`np.maximum.accumulate` reversed), precision 1 prepended at recall 0 | `_map_score_voc_auc` |
| reported P / R | **`precisions[-1]` / `recalls[-1]`** — the LAST point of the PR curve, i.e. every retained detection | `calculate_pr_metrics` |

**Two of our earlier readings are now withdrawn.**

1. **The per-image-averaging hypothesis (§15) is REFUTED.** AutoML pools exactly as we do, so
   our pooled figures are the comparable ones and the flattering 1.5× per-image numbers
   (box 0.429 / mask 0.364) are **not** a valid comparison and must not be quoted.
2. **Their "P 0.255 / R 0.614" is a curve ENDPOINT, not a tuned operating point** — precision
   and recall with *all* detections above the 0.3 score threshold kept. Comparing our
   threshold-optimised best-F1 (0.381 mask / 0.427 box) against their endpoint F1 (0.360)
   flattered us and is withdrawn.

**Scored with AutoML's exact formula** (`automl_metric.py`, our 6-scale ensemble):

| | AutoML AP50 | recall at precision 0.255 | curve passes above their point? |
|---|---|---|---|
| ours, **mask** | **0.225** | 0.560 | **No** ❌ |
| ours, box | 0.285 | 0.664 | Yes ✅ |
| *paper (instance seg)* | *0.345* | *0.614* | — |

(Endpoint P/R is not comparable directly: our fused scores run down to 0.03 against their 0.3
cut, so our endpoint precision is 0.026 at recall 0.824. The curve-versus-point test above is
the scale-free comparison.)

**FINAL VERDICT.** On instance segmentation — the like-for-like comparison — **we lose on every
properly matched measure**: AP 0.225 vs 0.345, and their (recall 0.614, precision 0.255) point
lies *above* our mask PR curve. The one real win is at **box** level, where our curve does pass
above their point (recall 0.664 at precision 0.255 vs their 0.614), though our box AP (0.285)
is still short of their mask AP.

The honest summary of this whole line of work: a frozen-backbone, box-supervised, laptop-trained
pipeline gets **detection** competitive with — on one measure better than — a fully
mask-supervised fine-tuned Mask R-CNN, and the box→mask stage is where it loses. That, plus the
annotation findings in §1–§7, is the result. It is not a SOTA claim.

### 17. Budget-matched recall — the fairest test available, and it still says no

Precision is close to meaningless here (§2–§3: 84% of unmatched boxes are on real vegetation),
so recall is the quantity to compare. But raw recall is bought by predicting more, so it needs
a matched cost — and the paper's own numbers supply one exactly:

    TP     = R x n_gt = 0.614 x 1911 = 1173
    n_pred = TP / P   = 1173 / 0.255 = 4601      (~10.2 predictions per tile)

Their P/R is the *last* PR-curve point (§16), so this is their whole prediction set. Taking our
top-4601 detections by score makes the denominator identical, so **TP alone then decides both
precision and recall** — there is no way to game it by predicting more.

| at 4601 predictions | our TP | our recall | our precision | vs paper recall |
|---|---|---|---|---|
| **box** | **1240** | **0.649** | **0.270** | **+0.035** |
| **mask** | 1095 | 0.573 | 0.238 | **−0.041** |

Stable across the rounding interval implied by their 3-s.f. figures (4589–4614): box +0.034 to
+0.036, mask −0.041 to −0.040. Figure:
[`claude_outputs/savannatree_recall_budget.png`](../../../claude_outputs/savannatree_recall_budget.png).

**The answer is still no, and the reason is the matching criterion.** Their 0.614 is **mask**
recall (AutoML instance segmentation matches on masks — §16), so only our mask curve is
like-for-like, and it loses by 78 trees at equal budget. Our box curve wins by 67 trees, but
box IoU ≥ 0.5 is a strictly easier bar than mask IoU ≥ 0.5, so that is **not** a claim of
superiority on their task and must not be presented as one.

A second reason not to reach for a looser comparison: their 0.614 is measured at AutoML's 0.3
score threshold, i.e. it is *not* their maximum recall. Comparing our maxR (mask 0.728) against
it would be doubly unfair.

**What this localises, though, is useful.** At an identical budget the box stage finds 1240
trees and the mask stage keeps only 1095 — **145 trees lost purely in box → mask**, which is the
same conclusion as §14/§16 but now in units of trees rather than AP. Our mask recall reaches
their 0.614 at ~6000 predictions, i.e. we need ~30% more guesses for the same find rate.

**The one clearly untested lever left** was that the EM masker (`em_c512_b05.npz`) had been
fitted on 300 tiles of the *old 900-tile subset*, never on the full 2772-tile training set.
**Tested (§18): it is a dead end — refitting makes it slightly WORSE.**

### 18. Refitting the masker on the full data does NOT help — the box→mask gap is intrinsic

The reasoning was that the masker learns crown-vs-background *appearance* from the training
boxes, so a fit built on 300 tiles of the discarded-59%-of-crowns subsample should be
impoverished; and that on half-labelled tiles the "background" cells used by the contrastive
term are partly unlabelled crowns, so a density filter should clean it. Both were wrong.

Fits are cheap (~13 s each), so all variants were built and scored (α 0.3, κ 1.0; threshold
selected per masker on the calibration set, read off on test):

| masker fit | cal AP50 | **test AP50** | test R@P255 |
|---|---|---|---|
| **stale — 300 tiles of the 900-subset** | **0.409** | **0.2315** | **0.560** |
| full data, 600 tiles | 0.399 | 0.2258 | 0.556 |
| full data, 1200 tiles | 0.397 | 0.2205 | 0.547 |
| full data, 600 tiles, label-density ≥ 0.5 | 0.392 | 0.2133 | 0.530 |

More fit data is neutral-to-harmful and the density filter is clearly harmful. Cal and test
rank the variants identically, which is worth something given cal could not arbitrate cleanly
here (every masker's fit tiles overlap the cal tiles — a confound worth naming, and the reason
the test column is the one to read).

**Why this is unsurprising in hindsight:** the masker is a *low-capacity* appearance model —
PCA-128 with K = 6–8 vMF components after pruning — so 300 tiles already saturate its
statistics; extra tiles add no parameters to estimate. And the density filter removes exactly
the messy tiles that give the background model its variety.

**Conclusion: the 145 trees lost in box → mask (§17) are not a stale-fit artefact.** They are
intrinsic to carving masks on a 16 px cell grid against loose, hand-drawn polygons that fill
only 62% of their bounding box (§6). Improving it needs a different mask representation — a
finer carving grid than the feature stride, or a learned mask head — not more data for this one.

Best mask configuration therefore remains the original fit at `mask_thr` 0.50:
**test mask AP50 0.2315, R@P255 0.560, best-F1 0.383**; budget-matched (§17) mask TP 1096 /
recall 0.574 against the paper's 1173 / 0.614.

### Where this leaves the comparison

| metric @ IoU 0.5 | paper | ours (all data, 2 seeds) | ratio |
|---|---|---|---|
| precision | 0.255 | **0.250** | **98%** |
| recall | 0.614 | 0.484 | 79% |
| F1 | 0.360 | 0.329 | 91% |
| mAP50 | 0.345 | 0.159 | 46% |

Against a fully-mask-supervised, end-to-end fine-tuned Mask R-CNN, using **box-only
supervision, a frozen backbone, a 3.2 M-param head and 29 min on a laptop**. Precision is
now level; recall and AP are not. And per §2–§3 the *absolute* values of precision and AP on
this benchmark are lower bounds set by the annotation, not by delineation quality — which is
the reason to stop treating this dataset's mAP as a target to close.

### Not attempted, and why

- **Mask-R-CNN-style bounded negative sampling** (the brief's third lever) targets the same
  gradient the ignore mask already neutralises; §2 shows the residual bare-ground FP rate is
  only 9.3% at the top of the ranking, so the headroom is small.
- **Hybrid detect@512 + mask@1024** was left untouched: with a ±0.049 seed band on the
  detector it cannot be measured at one seed, and mask AP inherits detector variance.

---

## Results

### THE FIX THAT MATTERED: ground sample distance (2026-07-31)

The paper gives ~2 cm GSD (80 m AGL, Zenmuse X5S) against OAM-TCD's ~10 cm. **In metres the
crowns are nearly identical — savanna 4.28 m median vs TCD 3.63 m** — so the "8-12x larger
crowns" that wrecked runs 1-2 was pure GSD mismatch. Worse, upscaling 1024->2048 to preserve
phase4's canvas constants moved us from 5x off to 10x off: one 16 px DINOv3 patch covered
**0.16 m** of ground versus **1.60 m** on TCD.

Fix = downsample instead of upsample, and drop the 4-phase interleave entirely (it exists
only to lift TCD from 2.2 to 4.5 feature cells/crown; at a 512 canvas a single pass already
gives 6.7). Back to the original `Detector8`, plain 16 px grid, `cell_origin=s/2` default —
**the whole registration-bug surface disappears**. 2.1 MB/tile instead of 134 MB, so the
dataset fits in RAM and a 60-epoch run takes **5.6 minutes on a laptop**. See
`scaled_pipeline.py`.

### Scale sweep — held-out S4, all 449 tiles / 1911 trees, single pass, box-supervised

| canvas | GSD | cells/crown | mAP50 | mAP40 | P@IoU.5 | R@IoU.5 | P@IoU.4 | R@IoU.4 | maxR@.4 | mask mAP50 |
|---|---|---|---|---|---|---|---|---|---|---|
| 208 | 10 cm | 2.7 | **0.1269** | **0.2132** | 0.251 | 0.337 | **0.324** | 0.434 | 0.598 | — |
| 336 | 6 cm | 4.4 | 0.1045 | 0.1743 | 0.199 | 0.320 | 0.235 | 0.504 | 0.644 | — |
| **512** | **4 cm** | **6.7** | 0.1206 | 0.1961 | 0.209 | **0.483** | 0.268 | **0.618** | **0.730** | 0.1157 |
| 1024 | 2 cm | 13.4 | 0.0855 | 0.1515 | 0.142 | 0.355 | 0.191 | 0.530 | 0.724 | 0.0857 |
| **paper** (Mask R-CNN, FULL mask sup., 2885 tiles) | 2 cm | — | **0.345** | — | — | — | **0.255** | **0.614** | — | — |

**Inverted U with a peak at 512 — NOT monotonic.** The finest arm (1024) is worst on mAP;
the coarsest (208) has the best precision/mAP but poor recall. Coarse grids give few
confident detections, fine grids give coverage.

**RESOLVED: the paper's IoU is 0.5, so we LOSE on every metric.** Their notebook
(`ajansenn/SavannaTreeAI`, `AutoML_Instance_Segmentation.ipynb`) uses Azure AutoML for Images
(`ImageTask.IMAGE_INSTANCE_SEGMENTATION`, `maskrcnn_resnet50_fpn`, BanditPolicy early
termination — matching the paper's description) and sets NEITHER `validation_iou_threshold`
NOR `validation_metric_type`, so both are AutoML defaults: `voc` metrics evaluated at
**IoU 0.5**. Their "mAP 34.5%" is therefore VOC-style **AP50**, not COCO AP@[.5:.95], and
their P/R are at IoU 0.5. The correct comparison is our IoU-0.5 column:

| metric @ IoU 0.5 | paper | ours (512) | ratio |
|---|---|---|---|
| precision | 0.255 | 0.209 | 82% |
| recall | 0.614 | 0.483 | 79% |
| **F1** | **0.360** (implied) | **0.292** | **81%** |
| mAP50 | 0.345 | 0.121 | 35% |

An earlier version of this section claimed parity "at IoU 0.4" — that was an artifact of
comparing at a looser threshold than the paper used, and is withdrawn. We reach ~80% of
their P/R/F1 and ~35% of their mAP, with **box-only supervision, a frozen backbone, 31% of
the training tiles and 5.6 min of training** against their fully-mask-supervised fine-tuned
Mask R-CNN. Respectable for a weakly-supervised method; not parity.

*(Minor caveat in our favour, insufficient to close the gap: VOC AP uses interpolated
precision and reads slightly higher than our COCO-style greedy AP at the same IoU.)*

The mAP gap (35%) is much larger than the P/R gap (~80%), which localises the remaining
deficit to **confidence ranking** rather than detection ability.

### The masker wants a finer grid than the detector

| canvas | box R@IoU.5 | mask R@IoU.5 |
|---|---|---|
| 512 (6.7 cells/crown) | 0.483 | 0.478 (hurts) |
| 1024 (13.4 cells/crown) | 0.355 | **0.391 (helps)** |

The EM masker carves on the 16 px cell grid, so it needs enough cells per crown to beat a
plain box. At 512 it has 6.7 and cannot; at 1024 it has 13.4 and does. On the old 2048
4-phase setup it had ~53 and helped there too. **Follow-up: detect at 512, mask at 1024** —
the two components genuinely want different resolutions.

### Progression

| run | config | box mAP50 | R@IoU.5 |
|---|---|---|---|
| zero-shot TCD detector | 2048, 4-phase | 0.0004 | 0.053 (maxR) |
| run 1 (Modal, faithful recipe) | 2048, 4-phase | — | never learned (val AP50 0.024) |
| run 2 (+ ignore mask) | 2048, 4-phase | 0.0716 | 0.248 |
| **run 3 (+ GSD fix, single pass)** | **512, 1-pass** | **0.1206** | **0.483** |

### Versus the paper's Mask R-CNN baseline — WE LOSE

| | precision | recall | mAP |
|---|---|---|---|
| **paper, Mask R-CNN R50-FPN @ep4, full mask supervision, 2885 tiles** | **25.5%** | **61.4%** | **34.5%** |
| ours, mask @IoU0.5 best-F1 | 14.8% | 26.8% | 7.4% (mAP50) |
| ours, box @IoU0.4 best-F1 | 19.7% | 35.6% | 12.5% (mAP40) |
| ours, box @IoU0.4 max-recall | — | 51.2% | — |

We reach ~44% of their recall at matched IoU (26.8 vs 61.4) and ~1/5 of their mAP. **The
weakly-supervised recipe that BEAT fully-supervised DetecTree2 on OAM-TCD (0.630 vs 0.545)
loses to a fully-supervised Mask R-CNN here** — that contrast is the real finding, and it is
negative for the transfer claim.

Differences that are ours to own: they fine-tune a COCO-pretrained detector end-to-end with
FULL MASK supervision; we freeze DINOv3 and train a 3.2M-param head on BOXES ONLY with a
training-free masker. They used 2885 train tiles, we used 775 (27%). Their pretrained
detector converged by epoch 4; ours trains from scratch.

Cheapest untested lever for the recall gap specifically: **they selected epoch 4 to preserve
recall** (their recall FELL with further training), whereas we selected on an ignore-aware AP,
which leans precision. Re-selecting our checkpoint on recall costs nothing.

**Against the zero-shot baseline:**

| | box mAP50 | box maxR |
|---|---|---|
| zero-shot OAM-TCD detector | 0.0004 | 0.053 |
| **this run** | **0.0716** | **0.365** |
| | **×179** | **×6.9** |

Run 1 (faithful TCD recipe on Modal) reached val box AP50 0.024 and never learned; this
detector reaches 0.0716 box / 0.0739 mask mAP50 on held-out data.

### What actually fixed it — ablation on one common metric

All checkpoints re-scored on identical val tiles via `rescore_val.py` (the per-run
best-epoch numbers were selected under different criteria and are not comparable):

| run | ignore | size bias | epochs | AP50 | AP50i | AP40i | R | **maxR** |
|---|---|---|---|---|---|---|---|---|
| A1_base (= run-1 recipe) | OFF | 3.00 | 14 | 0.0137 | 0.0196 | 0.0381 | 0.057 | 0.134 |
| A2_size | OFF | 5.19 | 14 | 0.0090 | 0.0132 | 0.0313 | 0.085 | 0.194 |
| **A3_ignore** | **ON** | 3.00 | **10** | 0.0174 | 0.0233 | 0.0491 | 0.089 | **0.327** |
| **LONG_ign** | **ON** | 5.43 | 44 | **0.0250** | **0.0290** | **0.0600** | **0.148** | **0.345** |

**A1 vs A3 is the clean comparison** — same frozen tiles, same everything but the ignore
mask — and it gives **maxR 0.134 → 0.327, a 2.4× recall gain with 30% FEWER epochs** (A3 was
cut short at 10). FIX 1 is the load-bearing change, and its effect lands on recall.

FIX 2 (size prior) alone *hurts* AP (A2 < A1) because the size head self-corrects anyway when
trained from scratch; it only pays off combined with the ignore mask.

### Caveats

- **250 of 449 test tiles.** Modal Volume reads degraded overnight (10–17 s/tile, several
  hangs); a full-449 attempt hung at tile 300 because `read_file` has no socket timeout.
  `local_eval_test.py` now bounds each read and skips (7 tiles skipped here). The remaining
  199 tiles are un-run, not excluded for cause.
- **Precision is a lower bound**, not an estimate: ~50% label completeness with no ignore
  class at test. Recall is the trustworthy figure.
- **val (0.0194 AP50) ≪ test (0.0716)** — the val site S10 is markedly harder than test S4
  (S4 crowns are larger: ~427 px vs ~179–228 px on the train/val sites). Model selection on
  S10 is therefore conservative rather than optimistic.
- ~~Single seed, no variance band.~~ **MEASURED (session 3): the band is ±0.051 on a mean of
  0.067 over 5 seeds, i.e. wider than every effect this folder has reported.** Any conclusion
  in the sections below that rests on one seed is unsupported until re-run. Use `collate.py`.

### Next levers (untested)

> **Superseded by [session 3](#session-3-2026-07-31--the-benchmark-is-the-experiment).**
> Lever 2 ("more data") was executed and is the only one that cleared the noise band
> (+0.09 mAP50, both seeds above all five 900-tile seeds). Lever 1 (`w_size`) is untested but
> is a ~0.01-scale effect against a ±0.05 band, so it is not measurable at fewer than ~5
> seeds per arm. The list below is kept for provenance.

1. **`w_size`** — `predW/gtW` drifted to 0.61–0.77 during training, i.e. boxes ~30% too
   small, which directly caps IoU 0.5. `w_size=0.1` down-weights exactly the term that must
   move. Raising it is the clearest remaining gain, and AP40 ≫ AP50 (0.125 vs 0.072) is the
   signature of a size problem, not a localisation one.
2. **More data** — only 41% of the available train crowns were used (see the n_train note).
3. **Species-conditioned size priors** — the recovered `Class` attribute (below).


### Run 1 (from scratch) FAILED — diagnosed: the size prior is ~9–12× too small

Seed-0 `train_4p` stalled at **val box AP50 0.024**, flat from epoch 5 to 10 while hm/giou
losses fell. It was then preempted at ep10 and the workspace hit its spend limit.

`zeroshot_tcd_local.py` isolated the cause for free (no GPU): applying the **trained OAM-TCD
detector** to SavannaTree test tiles gives box mAP50 **0.0004**, maxR 0.053 — and the box
widths say why:

| box widths @2048 | p5 | p25 | **p50** | p75 | p95 |
|---|---|---|---|---|---|
| TCD detector predictions (all) | 16 | 28 | **48** | 85 | 166 |
| TCD predictions @ op thr 0.40 | 22 | 34 | **47** | 63 | 95 |
| **SavannaTree GT** | 212 | 327 | **427** | 634 | 1137 |

Median pred/GT ratio **0.113**. A 48 px box against a 427 px crown caps at IoU ≈ 0.013 even
perfectly centred, so mAP50 ≈ 0 is forced arithmetically — this is a *scale* failure, not a
localisation or feature failure.

The same mechanism explains run 1: `Detector8.reg.bias` initialises log-size at
**`exp(3)` ≈ 20 px** (a value chosen for TCD's 36 px crowns — see the comment in
`detector.py`), so the head must travel ~3 log-units to reach savanna's ~427 px, while
`det_loss`'s `w_size=0.1` down-weights precisely the term that has to move. The detector
learns *where* crowns are and not *how big*.

**Fix (not tuning — re-deriving the same design decision for a new data scale):** set the
size-head bias to the dataset's log-median crown width (≈ `log(304)` = 5.7 on the train
split) and consider raising `w_size`. Untested — the spend limit blocked the re-run.

### Aside: the ×2 upscale made this worse

Upscaling 1024 → 2048 (needed to keep every phase4 geometry constant valid) doubles crown
pixels and therefore doubles the distance the size prior must travel. Running at the native
1024 canvas would halve the mismatch — but costs a re-derivation of `RES`/`SCALE`/`s_px`/
`cell_origin`, i.e. re-opening the registration surface. Fix the prior, not the canvas.

**How to read them.** Under sparse labelling the two halves of AP are contaminated
asymmetrically, so report both and trust them differently:

- **Recall is clean.** An unlabelled tree cannot lower recall — it is simply absent from the
  denominator. `instance_seg.iou0.50.best.R` and `.maxR` therefore measure what the pipeline
  actually finds, and are the headline numbers to quote.
- **Precision is contaminated.** Every correct detection of an unlabelled tree lands in the
  FP count, so precision (and hence AP and F1, which AP integrates) is a *lower bound*, not
  an estimate. A low mAP50 here is consistent with a perfectly good detector.

`eval_4p_selfmask` already emits P/R/F1 at both the val-picked operating point and the
best-F1 point, plus `maxR`, for box IoU 0.4/0.5 and mask IoU 0.5 — no extra run needed.
Do **not** invent a vegetation-index ignore mask to "fix" precision: it would be a bespoke
protocol, non-comparable to every published number on this dataset.

## Files

| file | role |
|---|---|
| `scaled_pipeline.py` | **the working recipe** — extract / train / evaltest at a downsampled canvas, single-pass L24, `Detector8`; `--border`, `--train-gt`, `--snap-epochs` |
| `fetch_more_tiles.py` | pull the 1 860 unused train tiles from Zenodo by zip range-request → `savanna_train_tiles_gt_full.json` (2 772 train tiles) |
| `rank_diag.py` · `fp_decomp.py` | where the AP goes: oracle-rerank ceiling, score AUC; and what the confident FPs are (unlabelled veg vs bare ground) |
| `rescore.py` · `rank_transfer.py` | the ranking-head lever (negative) and the cross-site AUC that explains why |
| `label_density.py` | test tiles stratified by labelled share — precision tracks it, recall does not |
| `clump_check.py` · `clump_viz.py` | are GT polygons individual crowns or groups? shape + detection evidence, and the figure |
| `collate.py` | every `test_*.json` as one table, grouped mean ± sd (use this, not single runs) |
| `run_border.sh` · `run_seedband.sh` | the truncated-box ablation and the 5-seed / fixed-epoch band |
| `savanna_data.py` | Zenodo range-fetch (no 9.6 GB download needed for the GT), COCO → phase4 GT, site split, ×2 upscale |
| `savanna_modal.py` | Modal app `savanna-phase4-l24`: `fetch_data` · `verify` · `extract_4p` · `train_4p` · `fit_masker_4p` · `eval_selfmask` · `band_selfmask` |
| `savanna_test_gt.json` | 449 S4 tiles — `{trees, canopy:[], W:2048, H:2048}` |
| `savanna_train_tiles_gt.json` | 1015 tiles — `{boxes, canopy:[], partition}` |
| `savanna_split.json` | the split's provenance (sites, seed, counts) |
| `stubs/`, `ref_feat_tcd.npz` | Modal masker deps + the layers-trap parity reference (copied from the TCD folder) |

## Caveat log

- **Incomplete GT (the big one).** Species-reference labelling + no ignore class ⇒ correct
  detections of unlabelled trees are counted as false positives. Bounds absolute mAP.
- **Crown scale vs the backbone window.** 95th-percentile crowns reach ~1070 px on the 2048
  canvas, and `_extract_2048` runs the frozen backbone on 2×2 *independent* 1024 px windows —
  so the largest crowns straddle a boundary with no cross-window attention. The detector's
  size head also initialises at `exp(3)`≈20 px (tuned for TCD's 36 px crowns), so early
  training has ~2.7 log-units to travel. Neither is fatal; both are named follow-ups.
- **No native-4096 features.** The self-mask path reads the same 4-phase L24 cells as the
  detector, so the 4096-d native grid the *vaulted* TCD masker needed is never extracted
  (saves ~¼ of extraction and ~60 GB). `eval_4p` (vault-masker path) is consequently not
  available here — only `eval_selfmask`.
- **Two Modal failures hit this run back-to-back (2026-07-30), diagnose them separately.**
  (1) `train_4p` was **preempted at epoch 10** and then could not be rescheduled: *"waiting to
  be scheduled on a GPU_A100 worker … Relaxing requirements (memory=64.8GiB) may lead to
  faster scheduling"*. The 64 GB memory request was inherited from the TCD app and is ~20×
  the real need (~3 GB: encoded targets + one 3-tile batch) — it both blocked scheduling and
  billed ~48 GiB/h for nothing. Now 16 GB. (2) Then the workspace hit **"has exceeded its
  spend limit"**, which blocks every new run (raise it in Modal settings → Billing). Volume
  *reads* keep working under the cap, so cached features remain retrievable.
- **Preemption left a poisoned checkpoint.** `train_4p` skips training when the ckpt exists,
  so the ep10/AP50-0.024 partial would have been silently reused as a finished model — the
  exact trap the TCD README flagged as unaddressed. Fixed here with a **`.done` sentinel**
  (a ckpt without one is discarded), an `on_best_save` → `vol.commit()` hook so a preemption
  loses at most one eval interval, `retries=2`, and a `--force` flag.
- **Train/eval do not need an A100.** They stream ~121 GB of cached features per epoch from
  the Volume (~8 min/epoch ⇒ ~250 MB/s) and are I/O-bound, not compute-bound. `TRAIN_GPU`
  is now `L4` (~1/3 the price, schedules immediately); only `extract_4p` (16 ViT-L passes per
  tile) is genuinely GPU-bound and stays on A100.
- **Always `modal run --detach` for long stages.** Without it the remote app's life is tied
  to the local client heartbeat: a network blip raises `ConnectionError: Deadline exceeded`,
  and Modal then *stops the app* ("local client disconnected") mid-stage. This killed a
  30-minute Zenodo download 3.2 GB in. It reads like a local crash, but the remote work was
  cancelled. `run_savanna.sh` passes `--detach` on every stage; reattach with
  `modal app logs <ap-id>`.
- **`fetch_data` is resumable.** The zip is staged on the Volume and the download resumes
  with an HTTP `Range` header, committing every ~512 MB — Zenodo serves this file at only
  ~3 MB/s (~55 min), so throwing a partial transfer away is expensive.
- **Zenodo range requests are flaky** (`IncompleteRead`); `savanna_data._fetch` retries.
- **`train_4p` used to drop `gt_path`.** It accepted the argument and never forwarded it to
  `TileData`, so a non-TCD dataset would have silently trained against TCD's GT. Fixed in
  `../../modal_tcd_multiseed/phase4/phase4_lib_tcd.py` (backward-compatible: `None` keeps
  the TCD default).
