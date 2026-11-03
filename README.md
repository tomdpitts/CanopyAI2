# CanopyAI — weakly-supervised tree-crown instance segmentation on frozen DINOv3

Individual-tree-crown (ITC) **instance segmentation from bounding boxes only**, on a
**frozen** DINOv3 backbone, scored on the official Restor **OAM-TCD** 439-tile test
set. The through-line across the repo is one question: *how far can a frozen DINOv3
backbone be pushed on canopy tasks under weak (box-only) supervision, and does
domain-adaptive pretraining (DAPT) of that backbone help?*

**Current headline (2026-11-03; single source of truth:
[`results_439/TABLE1_CURRENT.md`](boxinst_commonality_tcd_04/results_439/TABLE1_CURRENT.md),
canopy-neutral, 2048², maxDets 600):** on the OAM-TCD 439 test, box supervision only,
- **LACE** (D1 variant: DINO-Swin-L detector trained on OAM-TCD boxes only + the
  training-free 4 px LACE masker; [`dino_det/`](boxinst_commonality_tcd_04/dino_det/README.md)):
  mask AP50 **0.704**, box AP50 0.739 (single seed), +0.039 [0.032, 0.049] over BoxTeacher
  (0.664), the strongest box-supervised baseline run;
- LACE (CenterNet variant: frozen-DINOv3 CenterNet-style detector + the same masker):
  mask AP50 0.655 ± 0.004 (3 seeds), level with the mask-supervised Restor Mask R-CNN (0.648).

Externally (paper, talks) the D1 variant is simply **LACE**; the variant labels are
repo-internal only.

Training never sees a polygon; canopy is ignored, never background; polygons touch the
pipeline only at evaluation.

*(Historical: the earlier headline "mask mAP50 = 0.504, beating 0.432" was on the
superseded 512-px raster protocol with the 8 px masker; do not quote it.)*

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
| **[`boxinst_commonality_tcd_04/`](boxinst_commonality_tcd_04/README.md)** | box-only ITC instance seg on OAM-TCD, frozen DINOv3 | **flagship — 439 test mask AP50 0.704 (LACE, D1 variant) / 0.655 (LACE, CenterNet variant); see `results_439/TABLE1_CURRENT.md`** |
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

## Environment

Two isolated environments, both CPython 3.10, both rebuildable from pinned manifests:

```bash
# interpreter: 3.10.20 exactly (brew's python@3.10 formula only ships 3.10.21).
# openssl is not optional — without it CPython's _ssl fails to build and pip
# cannot reach PyPI.
brew install pyenv openssl@3 readline xz zlib
pyenv install 3.10.20

# main env — local analysis/eval, MPS training, launching Modal jobs
~/.pyenv/versions/3.10.20/bin/python -m venv .venv
.venv/bin/pip install -r requirements.txt

# DeepForest env — ONLY for scoring NEON with the benchmark authors' own
# evaluate_boxes (df_scorer.py). CPU, runs on saved prediction JSONs.
cd boxinst_commonality_tcd_04/modal_neon_multiseed
~/.pyenv/versions/3.10.20/bin/python -m venv .venv_df
.venv_df/bin/pip install -r requirements-deepforest.txt
```

The split keeps DeepForest's ~50-package surface (lightning, kornia, timm, geopandas,
rasterio) out of the main env. At these pinned versions the two do not actually clash —
it is deliberate insurance against future upgrades, not a live conflict.

Both paths are load-bearing: roughly fourteen docs and `run_*.sh` scripts hardcode
`.venv/bin/python`, `.venv/bin/modal` and `.venv_df/bin/python`.

**Three things to know before running anything locally:**

- **Local and Modal are different environments, on purpose.** Modal images are pinned
  in code (`phase4_modal.py`: python 3.11, torch 2.12.1, transformers 4.57.1); local
  runs transformers 5.12.1. Frozen-DINOv3 features from the two agree only at cos 0.86,
  so **never bit-compare local features to Modal features** — this has bitten the
  project before (see the phase4 "layers-trap" caveat).
- **Modal entrypoints need only the `modal` client locally.** `torch`, `datasets` and
  `detectron2` are imported inside the remote functions, never at module scope, so they
  are deliberately not in `requirements.txt`.
- **The Hugging Face cache does not survive a machine move.** DINOv3 ViT-L/16 weights
  (~1.1 GB) re-download on the first local backbone call.

## Quick start (original 2025 pipeline; historical)

*Reproduces the original 8 px detector + EM masker on the 512 raster. Current
configurations are documented in the flagship README (`lace4px_simple` masker), in
`dethead/` (frozen-feature detector heads) and in `dino_det/` (LACE, D1 variant).*

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
- (Historical) The original `det_t8` detector overfits: 0.504 (512 raster) was the
  early-peak, overfit-limited number.
- The frozen-DINOv3 detector heads plateau near test box AP50 0.69 (`dethead/`); the
  0.704 LACE row (D1 variant) uses a fine-tuned box detector (Swin-L, COCO-pretrained), so its gain
  over BoxTeacher is mostly detection, and it is a single seed (`dino_det/README.md`
  lists the open caveats).
- The directional shadow-prior thread (the repo's original objective) was found to be
  real on raw features but subsumed by DINOv3, and is retained only for history.
