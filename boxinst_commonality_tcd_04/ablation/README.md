# LACE ablation section — Tier 0

Everything here **re-scores predictions that already exist on disk**. No Modal spend, no
retraining, no feature extraction. Every byte written by this work lands in this folder;
everything outside it is a read-only input.

Run order (each is independent except T0.6, which needs T0.4's per-instance table):

```bash
.venv/bin/python -m boxinst_commonality_tcd_04.ablation.scripts.t01_boundary
.venv/bin/python -m boxinst_commonality_tcd_04.ablation.scripts.t02_curves
.venv/bin/python -m boxinst_commonality_tcd_04.ablation.scripts.t03_strata_tile
.venv/bin/python -m boxinst_commonality_tcd_04.ablation.scripts.t04_strata_instance
.venv/bin/python -m boxinst_commonality_tcd_04.ablation.scripts.t05_convergence
.venv/bin/python -m boxinst_commonality_tcd_04.ablation.scripts.t06_figures
.venv/bin/python -m boxinst_commonality_tcd_04.ablation.scripts.t07_grid_8v16
.venv/bin/python -m boxinst_commonality_tcd_04.ablation.scripts.t08_table_notes
.venv/bin/python -m boxinst_commonality_tcd_04.ablation.scripts.t09_proxy_calibration
```

Every re-scoring script runs a **reproduction gate** first (`lib/gate.py`): it re-scores the
same predictions and asserts it reproduces the published per-seed numbers
(0.6250 / 0.6358 / 0.6294 mask AP50 for knobbed seeds 0/1/2) before reporting anything new.
All gates pass. Bands are sample standard deviation (`ddof=1`) throughout.

---

## Findings

### 1. The recorded boundary number does not describe LACE — and the real one is fine

`phase4_research/figures/boundary_f1.json` reports **ours BF@3px = 0.286** against SAM 0.639
and a filled box 0.546. That "ours" is **`crown_mask`**, the RGB-guided-filter research
variant that `phase4/README.md` records as *failed*, scored on 13 tiles of size-filtered
crowns with GT boxes. It is not the EM masker and must not be quoted as LACE's boundary
quality.

Measured on the deployed masker (all 439 tiles, predicted boxes, matched TPs at IoU≥0.5,
512 raster) — `results/boundary_band.json`:

| arm | BF@1px | BF@2px | BF@3px | BF@5px |
|---|---|---|---|---|
| **LACE (3 seeds)** | **0.7611 ± 0.0020** | 0.9019 | 0.9588 | 0.9888 |
| filled predicted box | 0.6576 ± 0.0220 | 0.8306 | 0.9219 | 0.9793 |
| DetecTree2 (mask-supervised, 1 seed) | 0.7666 | 0.9082 | 0.9639 | 0.9913 |

LACE beats a filled box by **+0.104** at the tightest tolerance (≈50 seed-σ) and is within
0.006 of mask-supervised DetecTree2. Tolerances ≥2px saturate; BF@1px is the informative
column. `ours` vs `boxfill` is exactly paired; DetecTree2 has its own matched set (19,217
TPs vs ~20,750) so that comparison is approximate.

### 2. Where LACE does pay: high IoU

`results/per_iou.json`, `figures/per_iou.pdf`. LACE leads DetecTree2 by +0.096 at IoU 0.50,
the lead decays monotonically, and **DetecTree2 overtakes at IoU 0.75**. Consistent with
(1): the masks are well-placed but not tight enough for strict IoU. This is the honest
statement of what box supervision costs here.

### 3. The masker is not what degrades across strata — detection is

`results/strata_tile.json`. Mask AP50 varies hugely by stratum (0.217 for <10 crowns/tile
up to 0.684 for ≥80), but **box AP50 tracks it almost exactly** and the mask−box gap stays
in a narrow **+0.012…+0.042** band, positive in every stratum. On the canopy-closure axis
the gap *rises* slightly with closure (+0.0247 open → +0.0326 closed). The commonality
assumption does not visibly break where a reviewer would predict. Weakest strata (sparse
tiles; Lower Gangetic Plains biome at 0.260) are detection-limited.

Join verified: all 439 test tids resolve in `tile_index.json`, all `split=test`/`seen=False`,
no nulls, `sum(n_crowns) == 25705 ==` the GT tree count.

### 3b. Crown overlap does not break the commonality assumption

`results/strata_instance.json`. The premise "the box is mostly one crown" should fail where
crowns interlock. It does not. Among **box-matched** TPs — the arm that can actually observe a
mask failure, matching on box IoU then measuring mask IoU, as `failure_analysis.py` does:

| group | n (s0) | mean mask IoU | mask fail (<0.5) | mean GT size |
|---|---|---|---|---|
| isolated | 12,073 | 0.7148 ± 0.0019 | 0.0252 ± 0.0008 | 44.1 px |
| touching | 8,147 | 0.7093 ± 0.0015 | 0.0250 ± 0.0031 | 70.2 px |

Identical failure rates, and 0.0055 mask IoU between them. **39.3% of the 25,705 GT crowns
touch another**, so this is not a small-sample corner. Honest confound: touching crowns are
much larger, and size helps mask quality, so neighbour interference may be partly offset.
Either way the effect is ~0.005 IoU.

Note the two matching arms. Matching on *mask* IoU (also reported) defines mask-IoU≥0.5 into
existence and returns a fail rate of exactly zero — true but vacuous. Only the box-matched arm
answers the question.

### 4. The deployed prototypes have collapsed

`results/convergence.json`. Measured directly on `em_model_4p_fix.npz`:

| model | K | mean pairwise cos | pairs >0.9 | effective rank |
|---|---|---|---|---|
| `em_model_4p_fix.npz` (**deployed**, β=0.5) | 16 | **0.9935** | **100%** | **1.66** |
| `em_model_4p.npz` (β=0.5, pre-fix) | 16 | 0.9915 | 100% | 1.78 |
| `em_model_4p_b0_fix.npz` (β=0) | 14 | 0.2178 | 3% | 10.83 |
| `em_model_4p_b0.npz` (β=0) | 16 | 0.2225 | 1% | 12.33 |

The winning configuration's 16-component mixture spans an **effective rank of 1.66** — it
behaves as roughly one foreground prototype. The mechanism is in the method itself:
`contrastive_update` computes `C = normalise(pos − β·neg)` (`boxinst_commonality/em.py:171`),
and subtracting a near-common background centroid from every prototype drives them together.
Pruning keys on responsibility *share* and is blind to redundancy, so K never falls.

This does not mean the masks are bad — the posterior comes from the fg-vs-bg log-ratio, so a
degenerate fg mixture still works. It means **K is nominal, the mixture is not doing the work
the method description implies, and a k_init sweep would be nearly flat.**

Also recorded: there is **no convergence criterion** in the code (`for it in range(args.iters)`,
30 fixed, no tolerance, no break). Post hoc, the β=0.5 trace is flat from ~iteration 15
(0.677 → 0.656). The β=0 trace settles at fg mass **0.934** — quantitatively the box-filler.

#### Follow-up (`phase4_collapse/` Stage 1) — two corrections and a verdict

Six fits, K ∈ {2,16} × EM seed ∈ {0,1,2}, settle what Tier 0 could only predict:

- **K=16 is decoration, not scaffolding.** Fitting with k=2 reaches the same foreground
  direction as k=16: cross agreement `cos(C̄)` = **0.9942**, *above* the within-k=16
  between-seed floor [0.9915, 0.9922]. The k=2-vs-k=16 difference is smaller than the gap
  between two random seeds of k=16.
- **EM init is stable** — k=16 seeds agree at cos 0.9922, k=2 at 0.9912. First measurement of
  masker-init variance here; it bounds *direction*, not AP.
- **Correction: pruning does not "never fire".** k=16 seed 1 pruned to K_eff=15; seeds 0 and 2
  did not. The earlier phrasing came from the deployed model plus two archived logs that
  happened to be non-pruning runs. The conclusion stands — pruning removes at most one
  prototype and never approaches the effective rank (~1.3–2.0), because it tests starvation
  rather than redundancy.
- **Sanity gate passed:** the k=16 seed-0 refit reproduces the deployed model (rank 1.6562 vs
  1.66; cosine 0.9935 vs 0.9935).

### 5. The whitening evidence is weaker than it looks

`results/proxy_calibration.json`. Whitening is ablated only on dryland, in shape proxies
(`fill`/`corner`/`centre`), because dryland has no mask GT. Whether that ablation means
anything depends on whether `corner` tracks accuracy. Measured on OAM-TCD across five masker
arms with known AP:

- Pearson(corner, AP50) = **−0.865** — but carried entirely by the single β=0 outlier.
- Spearman = **−0.100** — no rank relationship.
- Within the four β=0.5 arms, corner *rises* 0.232 → 0.424 while AP50 also *rises*.

So `corner` is a valid **degeneracy detector** (β=0's fill 0.98 / corner 0.94 / centre 0.999
is an unmistakable box-filler signature) but **does not rank working configurations**. The
dryland −whiten shift (corner 0.37 → 0.44) sits in the working range, not the failure range,
so it **cannot be read as an accuracy loss**. Whitening should be described as a design
choice whose effect on accuracy is unmeasured; settling it needs the TCD refit (~$8).

### 6. The 8px-vs-16px comparison is confounded

`results/grid_8v16.json`. The +0.123 AP50 difference spans **five simultaneous changes**:
feature grid, detector recipe, masker refit, multiscale-vs-single-scale eval, and local-MPS
vs Modal (transformers 5.12.1 vs 4.57.1, cos 0.86). It is not a feature-grid ablation and
should not be presented as one.

Also found: `mps_multiseed/README.md` quotes that band as ±0.0231, which is `ddof=0`; the
headline 0.630 ± 0.005 band is `ddof=1`. The same seeds give ±0.0259 under `ddof=1`.
Bands quoted across this project's READMEs are not directly comparable as written.

### 7. Figures

- `figures/per_iou.pdf` / `.svg` — AP vs IoU threshold, LACE band vs DetecTree2, with the
  no-canopy-ignore variants dashed in the same frame.
- `figures/posteriors.pdf` / `.svg` — self-masks on three tiles chosen *by measurement* from
  `results/strata_instance_rows_s0.json` (best / most-crowded / worst), rendered from the
  deployed `preds/knobbed_s0.json` at the operating point (score ≥ 0.40), green TP / red FP /
  teal missed GT. The selection is itself evidence for finding 3b: the most-crowded tile
  (100% touching crowns) scores 0.694 mean mask IoU, while the *worst* tile has only 12%
  touching — a mangrove/water edge where 25 of 43 GT crowns are missed outright.

A soft EM posterior heatmap is **not** included: the saved preds carry only RLE masks already
thresholded at 0.25, and recomputing the posterior locally would use DINOv3 features that
differ from the Modal ones (cos 0.86), so it would not depict the deployed model.

---

## Provenance notes found along the way

- `preds/knobbed_s*.json` and `preds_b05_fix_thr025_439.json` all carry
  `meta.model = "...β=0 self-mask masker"`. That string is stale for every one of them; the
  real discriminator is `meta.em` (`em_model_4p_fix.npz` = β=0.5). Cosmetic, but it will
  mislead anyone reading the preds files directly.
- No fit log or fit report survives for the deployed `em_model_4p_fix.npz`.
  `em_fit_report_4p.json` and `fit.log` describe `em_model_4p.npz`, a genuinely different
  fit (every learned array differs; the fix model is the *older* file).
  `results/em_fit_report_4p_fix.json` is a **reconstruction** from the npz — static fields
  only, no per-iteration trace, because none exists.
- The qualitative panels in `claude_outputs/mask_compare_tpfpfn/` and `beta_disagree_*/` all
  depict the **pre-registration-fix** maskers. `t06_figures.py` renders fresh panels from the
  deployed predictions instead.

## What Tier 0 cannot answer

Deferred to the Modal items: whitening on/off on TCD (~$8); EM-init seed sweep with the
detector held fixed (~$14 — the published ±0.005 band contains **no** masker variance, since
the masker is fit once at seed 0 and shared); `k_init` sensitivity (~$8, and finding 4
predicts it will be flat); whitening-stat scope and cross-site transfer (~$6).
