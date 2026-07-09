# boxinst_commonality — masks from embedding-space commonality (boxes only)

A different route to instance masks than [`boxinst/`](../boxinst): instead of the
per-instance CondInst mask branch trained with BoxInst projection + pairwise
losses, this infers **what the annotated boxes have in common in DINO embedding
space** and turns that into a figure/ground posterior. Detection is unchanged —
we reuse a trained `boxinst` CenterNet checkpoint; only the mask generator is new.

`boxinst/` is untouched; this package imports read-only helpers from it and from
`dapt/`.

## The idea

> "If I see 100 photos of ducks — even having never seen a duck — I quickly learn
> to ignore the background, because only the duck recurs."

Made precise, patch by patch. Every 16 px DINO patch inside every GT box is an
embedding. Across the ~1100 training boxes:

- **crown = embeddings that recur across boxes AND are rare outside them**;
- **background = embeddings common inside boxes and outside alike.**

The only genuine label anywhere is "everything outside every box is background",
which anchors the decomposition. No mask labels, and — as in `boxinst` — **no
hyperparameter is ever tuned on masks** (there is no mask GT to tune on).

## Model (`em.py`)

Centred, PCA-128, **whitened**, L2-normed DINO patch space. vMF likelihoods
(`kappa * cos`).

- **Features (default): 8px half-stride** (`cache_s8.py`) — DINO stays at
  patch-16 on the 512 tile; 4 passes with ±4px-shifted canvases interleave to a
  64×64 grid, so every 8px cell gets a 16px-window feature centred on it. No
  new pixel information (source imagery is native 10cm GSD) — purely finer
  feature localisation, and the single biggest mask-quality win measured
  (corner 0.36→0.27, centre 0.91→0.96 vs the 16px cache).
- **Instance-mask rules**: a mask is the **largest 4-connected component** of
  the posterior's 0.5 superlevel set (one detection = one crown). Without this,
  a box straddling two trees yields a bimodal posterior whose 0.5 level line
  pinches into a figure-8 at the saddle (measured: 2.7% of test detections).
  In the detection pipeline, masks are additionally **mutually exclusive**:
  contested pixels go to the instance with the highest posterior (score-order
  tie-break), so adjacent crowns share a boundary instead of crossing.

- **Background**: a `K_bg=12`-component mixture fit by spherical k-means on
  clear-background cells **plus a one-cell ring around every box** (context-
  matched background — a ViT patch of soil-next-to-a-crown embeds differently
  from open-ground soil; without the ring, tree-adjacent soil leaks to
  foreground). **Frozen** after init so EM can't quietly relabel it as crown.
- **Foreground**: `K=8` crown prototypes `C_k`, each with a **spatial prior**
  `pi[s,k,u,v]` over box-normalized coordinates, learned **per box-size tercile
  `s`**. This is where "crown centre" vs "crown edge" gets discovered — from
  where in the box each embedding tends to sit. `P(bg | u,v) = 1 - sum_k pi`.
- **Contrastive prototype update** (`contrastive_beta=0.5`): each M-step pulls
  every prototype toward its in-box responsibility-weighted mean AND **repels it
  from the outside-box signature it most resembles** —
  `C_k = normalise(pos_k - beta * neg_k)`, negatives being the near-box ring +
  clear background. This is the load-bearing definition: **commonality is
  agreement-across-boxes MINUS out-of-box signature**, not "whatever fills the
  box". Without it, a prototype can quietly converge to in-box *soil* (soil
  recurs across boxes too); with it, all 8 prototypes stay vegetation (see
  `prototype_montage.png`). `beta=0` recovers the purely generative M-step.
- **EM** over all in-box cells (E: responsibilities; M: re-estimate `C_k` and
  `pi`, background fixed). Two constraints stop the classic box-filler collapse:
  the prior's **total fg mass is pinned to pi/4** each M-step (a tight box of a
  convex crown is ~pi/4 crown — the inscribed-ellipse fraction), so EM only
  decides *where* the mass goes, not how much; and no `(u,v)` bin may exceed
  `BIN_CAP=0.95`.

**Mask = posterior `P(fg | embedding, position, size)`** for the cells covering a
box (GT at eval, predicted at inference), upsampled to the 128×128 grid.

### Two design choices, each pinned by an ablation (not by mask proxies)

1. **Whiten** the PCA components to unit variance. Raw-PCA cosine is dominated by
   a few high-variance "context" directions shared by all vegetation; whitening
   lets the low-variance *local* directions (crown-vs-soil) count. Chosen on a
   feature-space diagnostic: corr(appearance score, crown brightness) inside WON
   boxes — crowns are dark, so more-negative is better — goes −0.35 → **−0.50**.
2. **Contrast** the appearance log-ratio (recentre it within each box) before
   adding the spatial-prior logit. The *absolute* fg/bg ratio is confounded by
   ViT context (nearly all in-box cells read "foreground"); the *within-box
   ordering* is where the crown-vs-soil signal lives (confirmed against a
   supervised within-box brightness ceiling of CV corr ≈ 0.92 — the space
   contains the info, EM just has to use the ranking). Heterogeneous boxes
   (crown + soil) have a wide spread → carving; homogeneous dense-canopy boxes a
   narrow spread → they stay filled, correctly.

## Results (seed 0)

GT-box-prompted proxy metrics on the 33 test tiles (n=359 boxes), **identical
protocol/formulas to `boxinst/eval_masks.py`** so numbers compare directly. No
mask GT exists; these are proxies (`fill`/`corner`/`centre` — a crown blob wants
high centre, moderate fill, low corner), read alongside the renders.

| method | fill | corner | centre | leak |
|---|---|---|---|---|
| `boxinst_s0` head (projection + pairwise) | 0.58 | 0.36 | 0.87 | 0.084 |
| **EM commonality, 8px stride (primary)** | **0.70** | **0.27** | **0.96** | **0.017** |

Higher centre with lower fill than a box-filler (fill 0.95, corner 0.89) = the
posterior carves genuine blobs. Ablations, same protocol (16px unless noted):

| variant | fill | corner | centre | reading |
|---|---|---|---|---|
| **8px half-stride (primary)** | 0.70 | 0.27 | 0.96 | tightest |
| 16px, whiten + contrast + repel β=0.5 | 0.72 | 0.37 | 0.91 | carves crowns |
| − contrastive repel (β=0) | 0.73 | 0.41 | 0.91 | prototypes can drift to soil |
| − within-box contrast | 0.89 | 0.61 | 0.86 | box-filler |
| − whiten | 0.75 | 0.44 | 0.91 | weaker (see diagnostic) |

Auto-K (start K=16, prune starved prototypes): effective K=12 at 8px, 15 at
16px — denser sampling starves redundant prototypes; K is picked by the data.

`leak` is omitted: it is structurally ≈0 here (the EM mask only exists on
box+half-cell cells), so it is not comparable to the head's and would mislead.

Qualitative: `claude_outputs/boxinst_commonality/`. WON/BRU dryland crowns are
pulled cleanly out of their boxes; dense NEON canopy still under-segments (the
patch-16 evidence limit inherited from the features, same as `boxinst`). The
`detector test mAP50=0.589` and `pixel area-F1: N/A (no mask GT on dryland)`
annotations are on every render per the viz-metrics convention.

## Usage

Requires the `boxinst` feature cache (`dapt/cache/web_last4/`, made by
`python -m boxinst.cache_feats`) and a trained detector checkpoint.

```bash
.venv/bin/python -m boxinst_commonality.cache_s8               # once: 8px half-stride features
.venv/bin/python -m boxinst_commonality.em --seed 0            # fit the EM
.venv/bin/python -m boxinst_commonality.evaluate \
    --baseline boxinst/artifacts/boxinst_s0.pt                 # proxy table + GT-prompt renders
.venv/bin/python -m boxinst_commonality.infer_viz \
    --ckpt boxinst/artifacts/boxinst_s0.pt                     # full detect->mask pipeline PNGs
```

Renders land in numbered subfolders of `claude_outputs/boxinst_commonality/`
(`01_primary`, `02_beta1`, ... — one per model tag). Pipeline renders use
green=TP / red=FP / teal dashed=missed GT, boundary-only, with the patch grid
and purple active-cell outlines.

Diagnostics: `spatial_priors.png` (learned pi per size bin) and
`prototype_montage.png` (highest-responsibility 32 px patches per `C_k` — a
direct look at what each prototype learned to be).

Ablation flags on `em.py`: `--no_contrast`, `--no_whiten`, `--feat_dir
dapt/cache/web` (mid-layer blocks 3/6/9/12 instead of last-4), `--tag NAME`
(suffixes all outputs so a run doesn't overwrite the primary model).

## Limits / honesty

- **Proxy-only evaluation.** There is no mask GT anywhere; a real mask IoU would
  need hand-labelled crowns (deferred). The proxies + renders are the evidence.
- **Patch-16 granularity.** Boundaries follow the 16 px feature grid; masks are
  soft blobs, not crisp outlines. Sub-patch sharpening is not implemented here.
- **Dense canopy under-segments** — a box in continuous forest is mostly crown,
  so contrast correctly leaves it filled; separating touching crowns is a
  detection-grid problem, not a mask problem.
- MPS/BLAS reductions are not bitwise reproducible; expect ~±0.01 on the proxies.
