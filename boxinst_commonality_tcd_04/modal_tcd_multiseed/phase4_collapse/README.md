# `phase4_collapse` — prototype-collapse ablation

> ## Answers (runs complete)
> **Q1 — is K=16 scaffolding?** No. K=2 reaches the same direction as K=16
> (cross-agreement 0.9942, above the within-K=16 seed floor 0.9915–0.9922). The mixture is
> decoration at β=0.5. EM init is stable across seeds.
>
> **Q2 — does collapse buy box-robustness?** **No — hypothesis rejected.** At one
> configuration, β=0 → β=0.5 raises mean crown IoU by **+0.0483** with oracle boxes and
> **+0.0476** with predicted boxes, agreeing to 0.0007. The gain is independent of box
> precision. The earlier apparent sign flip was a stride/stack artefact; see the retraction in
> `ablation/results/collapse_beta_kappa.json`.
>
> **Q3 (added) — which component earns the gain?** The 2×2 shows E-step recentring dominates
> (+0.0493 / +0.0471 alone) over M-step repulsion (+0.0356 / +0.0346), with a strongly negative
> interaction (−0.0366 / −0.0341): they are substitutes. **β=0 with recentring ties the deployed
> β=0.5 configuration** (0.7895 vs 0.7885 oracle; 0.7188 vs 0.7193 predicted) at effective rank
> **11.43 instead of 1.66** — the same accuracy without the degeneracy. Collapse is governed
> entirely by β; the E-step moves rank by <0.7.
>
> Results: `results/stage1_fits.json`, `results/stage2_gate.json`,
> `results/factorial_summary.json`. Total spend ≈ $10.

Sibling of `phase4/`, same pattern as `phase4_sam/`: reuses phase4's Volume, features,
detector checkpoints and fit recipe, but **writes only under `/vol/collapse/`**.

```bash
.venv/bin/modal run collapse_modal.py::fit_grid     # Stage 1  CPU   ~$2
.venv/bin/modal run collapse_modal.py::eval_grid    # Stage 2  A100  ~$4
```

## Why this exists

Tier 0 (`boxinst_commonality_tcd_04/ablation/`) found that the deployed β=0.5 masker's 16
foreground prototypes have collapsed — mean pairwise cosine **0.9935**, **effective rank
1.66** — and that the whole mixture is reproducible by one prototype plus the summed spatial
prior to within 0.03 nats. Pruning never fires (zero prune lines in `fit.log`/`fit_b0.log`),
so `K=16` is nominal and the "auto-K adapts to the data" claim in `methods_box_to_mask.md:41`
and `vault/README.md:102` is false for every deployed β=0.5 model.

Two questions follow, and this folder answers them in cost order.

**Q1 — is K=16 scaffolding or decoration?** Does the fit *need* 16 prototypes to reach a
rank-1 solution, or is the mixture simply removable? `contrastive_update` computes `neg` as a
*per-prototype* softmax centroid, so with 16 spread prototypes the negatives partition and
each subtraction is locally targeted. At K=2 they don't. The endpoint might differ even though
the destination is one-dimensional.

**Q2 — does collapse buy box-robustness?** The existing evidence for a sign flip mixes two
incompatible arms:

| regime | β 0 → 0.5 | eff. rank | metric |
|---|---|---|---|
| GT boxes, 16 px local (`masker_lab/sweep_results.json`) | 13.02 → 1.73 | ↓ | AP50 0.969 → **0.771** |
| predicted boxes, 8 px Modal (439) | 10.83 → 1.66 | ↓ | AP50 0.579 → **0.620** |

Box precision is confounded with stride and stack, so the flip is unproven. Stage 2 holds
everything but box precision fixed.

## Isolation — read before running

`phase4_modal._selfmask_npz(beta, fix)` keys on **beta and fix, not on k or EM seed**, so
`(beta=0.5, fix=True)` resolves to `/vol/out/em_model_4p_fix.npz` — **the deployed masker**.
A K=2 fit would target that exact path and `fit_masker_4p`'s `if os.path.exists(npz): skip`
would **silently return the K=16 model labelled as K=2**. Hence:

- own `_npz(k, em_seed, beta)` carrying k and seed: `/vol/collapse/k{K}_s{S}_b{B}/em_k{K}_s{S}_b{B}_fix.npz`
- `_assert_isolated()` raises if any output path escapes `/vol/collapse/` (verified: no
  `(k, seed)` in `k∈1..32 × s∈0..7` resolves to the deployed npz)
- per-cell output dir, because `em.py:292` writes `em_fit_report.json` under a **fixed** name
- `/vol/out/` is read-only here: detector checkpoint and the two existing maskers are inputs

## Stage 1 — K × EM-seed fits (CPU, no eval, ~$2)

`K ∈ {2,16} × EM seed ∈ {0,1,2}`, β=0.5, 120 fit tiles — the deployed recipe. The fit is
CPU-only numpy (no `gpu=`), so six fits are cheap. **No eval**: every output is read off the
npz.

Runs first because it produces the noise floor Stage 2 needs. The published ±0.005 band is
**detector** variance only — the masker is fit once at seed 0 and shared — so the spread of
`cos(C̄ᵢ, C̄ⱼ)` between K=16 seeds is this project's first measurement of masker-init variance.

**K ∈ {2,16} and not more:** two points establish "inert between 2 and 16", which is the
claim; given effective rank 1.66 a third K adds almost nothing. Spend the third dimension on
seeds. K=2 rather than K=1 because `em.py:264` guards pruning at `C.shape[0] > 2` and the
k-means / contrastive paths are untested at K=1.

### Gate G1

Compares **between-K** agreement against **within-K=16** agreement, mean-vs-min (with 3 seeds
there are only 3 pairs, and a min-vs-min test flips on sampling noise alone).

| Condition | Verdict | Effect |
|---|---|---|
| within-K=16 mean cos < 0.80 | **EM init unstable** | **Stop.** Bigger finding than either question; the shared-masker design is fragile and ±0.005 understates variance |
| cross mean ≥ within-K=16 min | **Mixture removable** | K=16 is decoration; Stage 2 runs β arms only |
| cross mean < within-K=16 min | **Scaffolding candidate** | Stage 2 adds `--include-k2 2,0` |

## Stage 2 — GT vs predicted boxes (A100, ~$4)

No refits — both maskers already exist: `em_model_4p_b0_fix.npz` (rank 10.83) and
`em_model_4p_fix.npz` (rank 1.66). Grid: β ∈ {0, 0.5} × box source ∈ {GT, predicted}, detector
seed 0, on the **108-tile val** split.

**No jitter.** The real detector's median matched box IoU is **0.731** (measured over 20,220
box-TPs in `ablation/`), so GT (1.0) and predicted (0.731) already bracket the operating
regime. A jitter ladder only interpolates between two points already in hand.

**Primary metric is `mean_crown_iou`, not AP.** Under GT boxes recall is perfect by
construction, so AP mostly measures score ordering and is not commensurable with predicted-box
AP. Mean per-crown mask IoU is defined identically in both regimes (GT: boxes are 1:1 with
crowns; predicted: greedy box-IoU≥0.5 match, then mask IoU). AP50 and mAP50-95 are still
reported for continuity.

### Val really is held out — checked, not assumed

`make_load_train_4p` (`phase4_fit_tcd.py`) filters `v["partition"] == "train"` **before**
taking the first 120 tiles, and `train_tiles_gt.json` marks all 108 val tiles
`partition: "val"`. **The fit/val overlap is zero.** `eval_grid` asserts this before spending.

(A naive `sorted(all 900)[:120]` suggests an 11-tile overlap. That ignores the partition
filter and is wrong — I made exactly that error before reading the loader.)

Stronger still: those records carry only `['boxes', 'canopy', 'partition']` — **no crown
polygons** — so the masker fit is structurally incapable of reading mask labels. The
"no mask labels at any stage" claim in `table_439.tex` holds, and selecting α/κ on the 108 val
tiles is ordinary fit-on-train / select-on-val / report-on-test. The only caveat worth stating
is the one already in `phase4/README.md`: α/κ are two scalars fitted against 108 images of
mask labels, so call them "two inference scalars selected on 108 held-out images" rather than
fully mask-label-free.

### Gate G2

| Condition | Verdict |
|---|---|
| GT-box Δ negative, predicted-box Δ positive, both beyond the Stage-1 floor | **Sign flip confirmed at one config** — claim collapse-as-box-robustness |
| both Δ same sign | The earlier flip was a stride/stack artefact — **abandon the robustness framing** |
| Δs differ but inside the floor | Underpowered — add detector seeds before adding β points |

## Not worth spending on

α/κ. `ablation/figures/collapse_beta_kappa.pdf` panel (c) shows that val surface spans
0.630–0.639 — a broad plateau, already swept under a pre-registered rule.

## Outputs

`results/stage1_fits.json`, `results/stage2_boxsource.json` (local), and per-cell JSONs under
`/vol/collapse/` on the Volume.
