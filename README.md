# CanopyAI — weakly-supervised tree-crown instance segmentation on frozen DINOv3

Individual-tree-crown (ITC) **instance segmentation from bounding boxes only**, on a
**frozen** DINOv3 backbone, scored on the official Restor **OAM-TCD** 439-tile test
set. The through-line across the repo is one question: *how far can a frozen DINOv3
backbone be pushed on canopy tasks under weak (box-only) supervision, and does
domain-adaptive pretraining (DAPT) of that backbone help?*

**Headline result: mask mAP50 = 0.504** on the OAM-TCD 439 test — **beating 0.432**,
the fully-supervised Restor Mask R-CNN benchmark (mask labels + a trained backbone) —
using **only ITC boxes** and a **frozen** backbone. Training never sees a polygon;
canopy is COCO-ignore throughout; polygons touch the pipeline only at evaluation.

The flagship thread is **[`boxinst_commonality_tcd_04/`](boxinst_commonality_tcd_04/README.md)** —
read its README for the full method, results table, and replication commands. This
root README maps how the surrounding threads relate to it.

## Why this is interesting

Fully-supervised crown segmentation needs polygon masks (expensive to annotate) and a
trained backbone. This project shows a competitive result from **cheap supervision**
(bounding boxes, which are far faster to draw) on a **frozen** self-supervised
backbone, by decoupling the problem into two modules:

```
frozen DINOv3 (patch-16, cached)  ─►  8px DETECTION DECODER  ─►  ITC boxes
                                        (trained, boxes only)        │
                                                                     ▼
                                                training-free COMMONALITY EM ─► masks
```

The **detector** is the only trained part and the sole lever on mask mAP50, because
on TCD crowns the mask-given-a-box problem is near-saturated — a learned mask head
*underperformed* the training-free EM masks. So masks stay training-free (an EM
"commonality" model fit on box interiors, no polygon, no gradient) and all effort
goes into detection. See the flagship README for the architecture rationale and the
full-tile-training unlock that drove box AP from 0.275 → 0.555.

## Repository threads

Each directory is a self-contained thread with its own README. They build on each
other roughly in the order listed.

| thread | question | status |
|--------|----------|--------|
| **[`boxinst_commonality_tcd_04/`](boxinst_commonality_tcd_04/README.md)** | box-only ITC instance seg on OAM-TCD, frozen DINOv3 | **flagship — mask mAP50 0.504 > supervised 0.432** |
| [`boxinst/`](boxinst/README.md) | one-shot head: detection + CondInst masks from boxes only | prototype (dryland test mAP50 0.589) |
| [`boxinst_commonality/`](boxinst_commonality/README.md) | masks from embedding-space commonality instead of a CondInst branch | training-free EM mask route (feeds the flagship) |
| [`boxinst_tcd/`](boxinst_tcd/README.md) | does the commonality objective help on data that HAS polygons? | box-only training, mask-scored on OAM-TCD |
| [`tcd04_semseg/`](tcd04_semseg/README.md) | add a canopy tree-cover head without hurting ITC mAP50 | isolated dual-decoder experiment |
| [`dino/`](dino/README.md) | frozen-DINOv3 backbone benchmark (web vs satellite pretraining) on OAM-TCD | backbone-selection thread |
| [`dapt/`](dapt/PLAN.md) | domain-adaptive SSL pretraining of the backbone; is DAPT > web? | see `dapt/REPORT.md` |
| [`experiments/`](experiments/) | directional shadow-prior signal checks and geometry filters | superseded (see git history / memory) |

The `dapt/` thread produces the alternative backbone checkpoint that `--arm` swaps in:
the frozen backbone can be the public **web** DINOv3 or a **DAPT** variant, and the
DAPT-vs-web comparison is the outstanding thematic question this pipeline is built to
answer.

## Quick start (flagship result)

```bash
V=.venv/bin/python
# 1. test GT + features (once)
$V -m boxinst_commonality_tcd_04.prepare_test
$V -m boxinst_commonality_tcd_04.cache_test                # 439 tiles → feat_test/
# 2. train-tile GT + features (sample 900 of ~3611 ITC tiles)
$V -m boxinst_commonality_tcd_04.cache_train_tiles --n 900
# 3. fit the training-free EM masker (boxes only)
$V -m boxinst_commonality_tcd_04.em --seed 0               # → em_model.npz
# 4. train the 8px detector on full tiles (best-ckpt saved each eval; ~ep20 peak)
$V -m boxinst_commonality_tcd_04.train_detector_tiles --tag t8 --epochs 40 --bs 3 --eval_every 5
# 5. evaluate on the 439 (mask mAP50 + box + semantic F1)
$V -m boxinst_commonality_tcd_04.evaluate --det det_t8     # → eval_t8.json
```

Shared infrastructure lives in `dapt/` (frozen DINOv3 backbone, target encoding, peak
decode, AP eval) and `boxinst_commonality/` (the EM mask math), imported by the
flagship package. `requirements.txt` covers the Python environment; datasets are
stored locally (OAM-TCD train/test + an Australian arid-rangeland set).

## Caveats

- Backbone is frozen (web); the DAPT-vs-web comparison is the next step (swap `--arm`).
- Masks never see a polygon — the EM masker is fit on box interiors, canopy ignored.
- The headline `det_t8` detector overfits: 0.504 is the early-peak, overfit-limited
  number. Regularization + more of the ~3,611 available train tiles is the open lever.
- The directional shadow-prior thread (the repo's original objective) was found to be
  real on raw features but subsumed by DINOv3, and is retained only for history.
