# `phase4_collapse` — results

Prototype-collapse investigation for the LACE box-to-mask module. Every number below is measured on the **108-tile validation split** with **detector seed 0** unless stated otherwise; the 439-tile test split is untouched by this folder. Masker fits use 120 training tiles (boxes only, no crown polygons). Bands are sample standard deviation (`ddof=1`).

Two mechanisms are varied throughout and must not be conflated:

| knob | stage | effect |
|---|---|---|
| `contrastive_beta` (β) | M-step | prototype repulsion, `C = norm(pos − β·neg)` |
| `contrast` | E-step | within-box recentring, `Ā = A − mean_box A` |

`phase4` couples them (`no_contrast = beta == 0`), so its recorded β ablation varies both at once. This folder decouples them.

## 1. Component ablation (2×2), K=16, EM seed 0

The published β ablation is the *diagonal* of this table, so its +0.041 AP50 was never a β measurement.

| M-step | E-step | eff. rank | cos | oracle IoU | pred IoU | pred AP50 |
|---|---|---|---|---|---|---|
| generative (β=0) | absolute | 10.83 | 0.218 | 0.7402 | 0.6717 | 0.5941 |
| generative (β=0) | recentred | 11.43 | 0.234 | 0.7895 | 0.7188 | 0.6406 |
| contrastive (β=0.5) | absolute | 1.75 | 0.992 | 0.7758 | 0.7063 | 0.6224 |
| contrastive (β=0.5) **(deployed)** | recentred | 1.66 | 0.994 | 0.7885 | 0.7193 | 0.6392 |

**Effects on mean crown IoU** (β=0/absolute is the reference cell):

| | oracle | predicted |
|---|---|---|
| recentring alone | +0.0493 | +0.0471 |
| repulsion alone | +0.0356 | +0.0346 |
| joint (the published +0.041 AP50) | +0.0483 | +0.0476 |
| **interaction** | **−0.0366** | **−0.0341** |

The two are **substitutes, not complements**: either alone recovers ~+0.04, both together recover the same ~+0.048. Recentring is the larger term in both box regimes, matching the dryland proxy ablation (`boxinst_commonality/README.md:107-119`), where removing recentring moves `corner` 0.37→0.61 against 0.37→0.41 for repulsion.

## 2. β sweep at K=16, recentring held on, EM seed 0

Single-axis: no previous sweep held the E-step fixed. `masker_lab/sweep.py` (16 px) also sets `no_contrast=(beta==0)`, so its β=0 row differs in two components.

| β | K_eff | eff. rank | cos | oracle IoU | pred IoU | pred AP50 | pred AP50:95 |
|---|---|---|---|---|---|---|---|
| 0 | 16 | 11.43 | 0.234 | 0.7895 | 0.7188 | 0.6406 | 0.2680 |
| 0.1 | -- | -- | -- | -- | -- | -- | -- |
| 0.25 | -- | -- | -- | -- | -- | -- | -- |
| 0.5 | -- | -- | -- | -- | -- | -- | -- |

Effective rank falls **11.4 → 1.7** (a ~7× reduction in distinct foreground directions) while predicted-box crown IoU spans **0.0000**. β is a geometry knob with no accuracy consequence.

## 3. Component count

| β | K init | K_eff | eff. rank | cos | oracle IoU | pred IoU | pred AP50 |
|---|---|---|---|---|---|---|---|
| 0 | 2 | 2 | 2.00 | 0.053 | 0.7888 | 0.7181 | 0.6369 |
| 0 | 16 | 16 | 11.43 | 0.234 | 0.7895 | 0.7188 | 0.6406 |
| 0.5 | 2 | 2 | 1.30 | 0.988 | -- | -- | -- |
| 0.5 | 16 | -- | -- | -- | -- | -- | -- |

At β=0 the K=2 fit is a genuine two-component mixture (effective rank 2.00 of a possible 2.00, cos 0.05) and still matches K=16's rank-11.4 mixture. The extra fourteen components buy ≈0.0007. At β=0.5 both collapse, and K=2 was never evaluated for accuracy — the equivalence there rests on direction agreement, `cos(C̄_K2, C̄_K16) = 0.9942`, which exceeds the between-seed agreement within K=16 itself (0.9915–0.9922).

## 4. Box precision — is collapse a robustness mechanism?

| box source | β=0 (rank 10.8) | β=0.5 (rank 1.7) | Δ |
|---|---|---|---|
| oracle (ground-truth boxes) | 0.7402 | 0.7885 | +0.0483 |
| predicted boxes | 0.6717 | 0.7193 | +0.0476 |

**No.** The two deltas agree to 0.0007, so the benefit is independent of box precision. The apparent sign flip in the prior evidence — GT boxes favouring β=0 at 16 px (`masker_lab`: 0.7412 → 0.6590) while predicted boxes favour β=0.5 at 8 px — was a stride/stack artefact, not box precision. Note β=0 scores almost identically at both strides (0.7402 here vs 0.7412 in the lab); it is β=0.5 that differs (0.7885 vs 0.6590). *(Both arms here use the absolute E-step for β=0, so these deltas are two-component; section 1 decomposes them.)*

## 5. Masker-seed variance

The published 0.630 ± 0.005 varies **detector** seeds and holds one masker fixed (`em_model_4p_fix.npz`, EM seed 0), so it contains no masker variance. This section isolates the other axis: detector fixed at seed 0, masker refitted at three EM seeds.

| β | seed | K_eff | eff. rank | cos | pred IoU | pred AP50 |
|---|---|---|---|---|---|---|
| 0 | 0 | 16 | 11.43 | 0.234 | 0.7188 | 0.6406 |
| 0 | 1 | 15 | 11.35 | 0.252 | 0.7188 | 0.6401 |
| 0 | 2 | 16 | 11.88 | 0.245 | 0.7195 | 0.6409 |
| 0.5 | 0 | 16 | 1.66 | 0.994 | 0.7193 | 0.6392 |
| 0.5 | 1 | 15 | 1.89 | 0.987 | 0.7214 | 0.6389 |
| 0.5 | 2 | 16 | 2.01 | 0.985 | 0.7205 | 0.6390 |

- **β=0** — crown IoU 0.7190 ± 0.0004, AP50 0.6405 ± 0.0004 (3 EM seeds)
- **β=0.5** — crown IoU 0.7204 ± 0.0011, AP50 0.6390 ± 0.0002 (3 EM seeds)

**Masker-seed variance is small — an order of magnitude below detector variance.** Refitting the masker moves crown IoU by ±0.0011 and AP50 by ±0.0004, against the published detector band of ±0.005. Fixing the masker at one EM seed therefore hides nothing, and the reported band is genuinely dominated by detector variance.

**β is inert, and the two metrics disagree on its sign.** β=0.5 leads on crown IoU by +0.0014; β=0 leads on AP50 by +0.0015. A real effect would move both metrics the same way. Combined with the single-seed β sweep — effective rank 11.4 → 1.7 for a 0.0011 spread — the repulsion term changes the model's geometry substantially and its accuracy not at all.

Note the diverse regime is also the more seed-stable one: effective rank spans 11.35–11.88 at β=0 against 1.66–2.01 at β=0.5, and one seed prunes to K=15 in both.

## 6. Isolation and provenance

- Every output is written under `/vol/collapse/`; `_assert_isolated()` refuses any path outside it. `/vol/out/` (phase4's outputs, including the deployed masker) is read-only here and was verified byte-identical before and after every run.

- Cell names carry every axis that changes the fit — `k{K}_s{seed}_b{beta}_c{contrast}`. Cells fitted before the `contrast` axis existed carry no `_c` tag and are implicitly `contrast == (beta != 0)`, matching phase4's coupling.

- `phase4_modal._selfmask_npz()` keys only on `(beta, fix)`, so a K=2 fit at β=0.5 would resolve to the deployed `em_model_4p_fix.npz` and be silently skipped by its `if os.path.exists(npz)` guard — returning the K=16 model labelled as K=2. This folder never uses that function.

- Masker fits draw 120 tiles from the **792 train-partition** tiles only; `make_load_train_4p` filters `partition == "train"` before slicing, and all 108 validation tiles are marked `partition: "val"`. Fit/validation overlap is **zero**, and those records carry only `boxes`, `canopy`, `partition` — no crown polygons.

- The new box-source eval reproduces the recorded validation cell exactly: the deployed masker on predicted boxes gives AP50 **0.6392** and AP50:95 **0.2680**, matching `sweep_val_knobs_s0.json` at α=0.3, κ×1.6, τ=0.25.

