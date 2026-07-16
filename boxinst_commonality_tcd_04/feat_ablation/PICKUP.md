# PICKUP — feature compression to unlock full-4k detector training (parked)

Optional future branch. Parked 2026-07-16 after stage 1 (a linear-probe gate);
detector training was never run. **Primary path to resume: block-1024.** Whole
experiment is self-contained and deletable (`rm -rf feat_ablation/`); the parent
pipeline was never modified.

## Why this exists

`det_t8` (mask mAP50 0.504) is **overfit-limited** — val boxAP50 peaks ~ep20 on
only 792 train tiles then declines. The obvious fix is more data: ~3,611 ITC train
tiles are available (we use 900). But full-tile DINOv3-web features are 4096-dim,
134 MB/tile → caching all 3,611 ≈ **484 GB**, which does not fit local disk
(~234 GB free). So: can we **compress the cached features** enough to fit full-4k
locally, without losing detection accuracy? `Detector8`'s first layer is a 1×1 conv
4096→256, so it discards most of the width immediately — compression should be
near-lossless.

## The main thrust: block-1024

Use **only the last DINOv3 block (channels 3072:4096 of the 4096-dim feature)** —
1024-dim. Why this over PCA-256:

- **Best signal** — stage-1 probe AUC 0.9491, actually edging full-4096 (0.9480).
- **Zero machinery** — it's a *contiguous mmap slice* of the parent `.npy` files
  (the last block is the last 1024 channels on disk). No PCA basis to fit, store,
  or version; no separate cache to build or keep in sync.
- **Fits full-4k locally** — 33.6 MB/tile → **~121 GB** for all 3,611 tiles
  (< 234 GB free). Comfortably enough headroom to actually run the data-scaling.

PCA-256 (33 GB) is the *fallback* only if disk gets tight — it's smaller but AUC
is −0.005, needs a fitted basis + a built cache, and its variance gate "failed"
(84% < 95%; a red herring — 95% is a reconstruction threshold, irrelevant when the
features feed a task, and DINO's spectrum is heavy-tailed so it always looks low).

## What's settled vs open

- **Settled (stage 1):** linear in-box-vs-background separability is preserved by
  block-1024 (AUC ≈ full). The box-only evaluator here also reproduces det_t8's
  single-scale box mAP50 **0.555** exactly, so the eval path is proven.
- **Open (the real question):** linear presence-vs-background AUC is *not*
  localization or box regression. Whether block-1024 holds **box mAP50** is
  unanswered and can only be closed by training. Expectation is positive (nothing
  flagged), but it must be measured.

## Resume — the one experiment that decides it

Train `Detector8` on block-1024 and compare box mAP50 to the 0.555 baseline.
Sequential MPS only; treat gaps < ~0.02 as noise (MPS not bitwise-reproducible).

```bash
# block-1024 (PRIMARY) — no cache build needed, it's a slice of the parent .npy
.venv/bin/python -u -m boxinst_commonality_tcd_04.feat_ablation.train_variant \
    --variant block1024 --epochs 40 --bs 3 --eval_every 5 --seed 0
.venv/bin/python -u -m boxinst_commonality_tcd_04.feat_ablation.eval_box --variant block1024

# PCA-256 (FALLBACK, only if footprint matters) — rebuild cache if deleted (~5 min)
.venv/bin/python -u -m boxinst_commonality_tcd_04.feat_ablation.train_variant \
    --variant pca256 --epochs 40 --bs 3 --eval_every 5 --seed 0
.venv/bin/python -u -m boxinst_commonality_tcd_04.feat_ablation.eval_box --variant pca256
```

Checkpoints → `artifacts/det_fa_<variant>.pt`; eval jsons → `results/`.

**Decision rule:** if block-1024 box mAP50 is within ~0.02 of 0.555, it's the
green light — proceed to cache the full ~3,611 tiles as block-1024 (~121 GB
local) and run the data-scaling + multiseed program (the real fix for the
overfit). If it drops materially, detection genuinely needs the full width →
full-fat features on Modal (A100/H100; DAPT SSL infra already exists) is the path.

## Full-4k multiseed detector training on Modal (A100/H100)

Runs *after* block-1024 is validated (box mAP50 within ~0.02 of 0.555). This is the
real fix for the overfit: 4× data + regularization + a proper LR/stop schedule, ×5
seeds for an error-barred number. Modal because 5 seeds is too much sequential MPS —
and on Modal the seeds run **in parallel** (one A100/H100 container each), so the
whole program is wall-clock ~one training run, not five.

### Step 1 — cache full ~3,611 tiles as block-1024 on the GPU

- Download OAM-TCD train on the Modal side (HF), extract features per tile (2×2 of
  1024 windows, as `cache_train_tiles.py`), keep only the last-block slice
  (channels 3072:4096) → ~121 GB into a Modal **Volume**. ~1 hr on one A100
  (backbone-bound; compression cuts storage, not extraction).
- Sample the train/val cohort once (fixed seed) so all 5 detector seeds share the
  same tiles — vary only the detector init/shuffle seed, not the data.

### Step 2 — multiseed ×5 detector training (seeds 0–4, parallel containers)

Per seed, `Detector8` on block-1024 (in_dim 1024, frozen DINOv3-web), with the
regularization + early-stopping recipe below. Report **box mAP50 on the 439 test**
per seed, then mean ± std vs the 0.555 single-seed baseline; flag any seed > 0.02
from the mean instead of silently averaging.

**Hyperparameters (recommended — the 0.504 run's defaults were un-regularized):**

| group | setting | note |
|---|---|---|
| model | width 256, **tower 2** | tower 2 (down from 3) trims capacity to fight overfit |
| backbone | frozen web, block-1024 | in_dim 1024 |
| optimizer | Adam, lr 1e-3, **wd 5e-4** | wd up from 1e-4 (regularization); 4× data also helps |
| batch size | **8–16** (A100/H100) | scale lr linearly if bs > ~4 (e.g. bs 12 → lr ~3e-3) |
| **LR schedule** | **ReduceLROnPlateau** on val boxAP50, factor 0.3, patience 2 evals | dropping LR on plateau often *lifts* the peak before overfit — cosine `T_max=epochs` never anneals if you stop early, the 0.504 run's flaw |
| **early stop** | eval_every 3 ep; min_epochs 12; patience 4 evals no-improve; min_delta 0.005; max_epochs 60 | patience absorbs the ±0.01 MPS/val noise; more data may push the peak later than ep20, hence max 60 |
| selection | best-on-val checkpoint | the actual stop; track val boxAP50, **not** loss (loss falls monotonically, useless as a signal) |
| seeds | 0,1,2,3,4 | parallel on Modal; sequential-only on MPS |

**Caveat to keep honest:** early stopping + ReduceLROnPlateau reliably *capture and
slightly lift* the peak; the big mover is the 4× data + wd/tower regularization. Don't
attribute a jump to the schedule alone.

### Cost

- Extraction: ~1 hr (one A100). Training: ~4–6 min/epoch on 4× data at A100 speed,
  ~30 epochs to the early-stop → ~2–3 hr/seed, **all 5 in parallel ≈ 2–3 hr total**.
- H100 ~1.5–2× faster. A100-40GB is plenty (decoder is ~4 M params, features 1024-dim).
- Rough spend: ~$15–40 for the full parallel program incl. extraction.

### Local fallback (no Modal)

Same recipe, seeds sequential (never two MPS jobs at once): ~2–3 hr/seed on 4× data
→ ~12–15 hr for 5 seeds. Feasible overnight but Modal-parallel is the sane path.

## Stage-1 numbers (reference)

| variant | dim | probe val AUC | Δ vs full | MB/tile | full-4,169 cache |
|---|---|---|---|---|---|
| full-4096 | 4096 | 0.9480 | — | 134 | ~484 GB (does not fit) |
| **block-1024** | 1024 | **0.9491** | +0.001 | 33.6 | **~121 GB (fits)** |
| PCA-256 | 256 | 0.9426 | −0.005 | 8.4 | ~33 GB (fits) |

Full details, probe methodology, and baseline validation: `README.md` (same folder).
