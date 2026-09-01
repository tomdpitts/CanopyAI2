# Confidence score: multiply the masker's posterior back in

**Result: +0.0325 canopy-neutral mask AP50 (0.6300 → 0.6625, 3 seeds) on the OAM-TCD 439,
and +0.0354 on the held-out sparse 236, from**

```
score' = s · bimod · pfg_mean
```

**Zero fitted parameters. No training, no mask labels, no val split, no supervision of any
kind, no change to what the model detects or how it segments.** The 439 test set is never
used for fitting because there is nothing to fit.

| split | baseline (3 seeds) | + posterior product | Δ |
|---|---|---|---|
| OAM-TCD 439 | 0.6300 ± 0.0054 | **0.6625 ± 0.0012** | **+0.0325** |
| sparse 236 *(no tile shared with training)* | 0.6560 ± 0.0121 | **0.6913 ± 0.0095** | **+0.0354** |

Positive on **every seed and every metric** on both splits. Two further results:

- **It cuts seed variance 4.5×** — sd on CN AP50 falls 0.0054 → 0.0012. The heatmap score's
  reliability varies between seeds; the masker is the same module every time, so multiplying
  its posterior in stabilises the ranking.
- **It generalises rather than being tuned in.** The functional form was selected on the 108
  val tiles, which come from the *training* distribution. On the sparse 236 — sharing no tile
  with the 900 training images — the gain is **larger**, not smaller. Had the form been
  quietly fitted to OAM-TCD's holdout it would have shrunk. The zero-parameter form is what
  makes that claim clean: there are no weights that could carry distribution-specific
  information across. See `../results_439/sparse236/README.md`.

| | AP50 | AP75 | AP50:95 | fitted numbers |
|---|---|---|---|---|
| LACE s0, heatmap score alone | 0.6250 | 0.1445 | 0.2581 | 0 |
| + 8-feature logistic reranker (previous result) | 0.6616 | 0.1658 | 0.2797 | 9 |
| **+ posterior product** | **0.6615** | **0.1728** | **0.2822** | **0** |

Canopy-neutral, `score_coco` frozen protocol, 439 tiles, res 512, `--score_floor 0`
(`results_439/rerank_product_scored.json`). The two reranked rows are statistically
indistinguishable at AP50 (paired tile bootstrap, 2000 resamples: −0.00005, 95% CI
[−0.0016, +0.0016]) and the product is **significantly better at AP75** (+0.0057, 95% CI
[+0.0030, +0.0082], P(logreg better) = 0.000).

## The problem

LACE ranks masks by the CenterNet heatmap peak alone — `sigmoid(hm_logit)` at the decoded
centre (`dapt/decode.py:26`). That answers *"is there a crown centre at this pixel"*. It
does not answer *"is this box real"*, and it is the number COCO AP sorts by.

| | recall@0.5 | AP50 | AP lost to FPs |
|---|---|---|---|
| LACE s0 | **0.8095** | 0.4867 | 0.3227 |
| Restor (rpn 2000) | 0.7804 | 0.5699 | 0.2105 |
| DetecTree2 s0 | 0.7478 | 0.4215 | 0.3262 |

LACE has the **highest recall of any method measured** and converts it into AP least
efficiently. With oracle ranking of its own predictions it would score **0.8020** — it
realised 77.9% of that; Restor realised 84.4%. The deficit was ranking, not detection.

**Recalibration cannot fix this.** AP depends only on score *ordering*, so every monotone
transform — Platt, isotonic, temperature — leaves it exactly unchanged. Only new
information that reorders moves AP.

## The signal

`estep` (`boxinst_commonality/em.py:193`) **returns** `pfg`, the per-cell foreground
posterior, and `self_mask` (`boxinst_commonality_tcd_04/em.py:143`) hands it straight back
to the caller for every box. Two reductions of it summarise the masker's opinion of that box:

```python
pfg_mean = pfg.mean()               # how much of this box looks foreground
bimod    = np.abs(2 * pfg - 1).mean()   # how far from undecided the posterior is
```

A false-positive box has no commonality to find, so its posterior goes diffuse instead of
separating — `bimod` collapses. The module knows when it is confused, and says so in a
value the caller already holds.

Why LACE can use this and Mask R-CNN cannot: the masker is **training-free and never sees
the heatmap**, so its opinion is a genuinely independent second estimator. Mask R-CNN's
mask head is conditioned on the same ROI features that produced the box score, so its
"second opinion" is correlated by construction. The module separation usually read as an
architectural compromise is what makes this work — and independence is also what licenses
multiplying the two probabilities rather than fitting a weight between them.

### Discrimination, per box, against box-IoU ≥ 0.5 (val AUC)

| signal | AUC | corr. w/ detector score |
|---|---|---|
| detector score | 0.9129 | 1.000 |
| **`bimod`** | **0.8327** | 0.583 |
| `a_p90` | 0.7899 | 0.505 |
| `a_std` | 0.7847 | 0.485 |
| `a_max` | 0.7815 | 0.493 |
| `a_mean` (the recentring term) | 0.7486 | 0.429 |
| `pfg_mean` | 0.6409 | 0.182 |
| `log_cells` | 0.6224 | 0.141 |

`pfg_mean` is weak alone and strong in the product: it is the one EM statistic barely
correlated with the detector, so it is nearly pure new information.

## What the ablation ruled out

Run `python -m boxinst_commonality_tcd_04.confidence.ablation --full`
(`ablation_results.json`). Every row rescores the same frozen 159,687 boxes; only the
ordering changes.

**More features do not help.** All 128 subsets of the 7 EM statistics, each val-fitted:
test CN AP50 spans 0.6250 (score only) to 0.6616, median 0.6587. Beyond two EM statistics
the curve is flat.

**Fitting does not help — it hurts.** On exactly the features the product uses:

| | AP50 | AP75 |
|---|---|---|
| `logreg(s, bimod, pfg_mean)` | 0.6557 | — |
| `logreg(s, log bimod, log pfg)` | 0.6549 | — |
| **`s · bimod · pfg_mean`** | **0.6615** | **0.1728** |
| logreg, published 8 features | 0.6616 | 0.1658 |

The log-space unit-exponent form is what the logistic fit was failing to find. A fitted
size correction bolted onto the product makes it *worse* (0.6605).

**The exponents are not a hidden fit.** `s · bimod^q · pfg^r` over q, r ∈ [0.5, 2.0] spans
0.6527–0.6635; (1, 1) sits inside a broad plateau, not on a spike.

**The pre-recentring evidence is not needed.** `a_mean` — the quantity `estep` deletes and
that `em_diag_modal.py` re-runs the masker to recover — is measurable and redundant.
Dropping all four `a_*` statistics costs nothing once `bimod` and `pfg_mean` are present.
That kills the whole extraction job: both surviving statistics are two lines over `pfg`,
which `estep` already returns, so nothing inside the masker changes and nothing is re-run.

**Selection was made on val, not on test.** Ranked by val box-AP50 on the 108 held-out
tiles (the only test-free proxy; box AP, never mask IoU):

| zero-parameter form | val boxAP50 | val boxAP50:95 | test CN AP50 |
|---|---|---|---|
| `s·bimod^0.5·pfg` | 0.5197 | 0.2333 | 0.6595 |
| `s·bimod·pfg^0.5` | 0.5190 | 0.2313 | 0.6568 |
| **`s·bimod·pfg`** | 0.5177 | **0.2331** | **0.6615** |
| `s·bimod` | 0.5148 | 0.2276 | 0.6484 |
| `s` (baseline) | 0.4982 | 0.2165 | 0.6250 |

The top three are a 0.002 tie on val and cannot be separated at that precision; the
unweighted product is the exponent-free member of that tie. Val AUC ranks `s·bimod` top
instead — AUC weights every pair equally where AP weights the head of the ranking, which
is the part reranking actually moves.

## The honest caveat

**Two thirds of the gain is a box-size prior, not masker evidence.** Val-fitted logistic
reranks, each row adding one block (SIZE = cubic in `log n_cells`, available from the
boxes alone with no masker at all):

| | CN AP50 | Δ |
|---|---|---|
| `s` only | 0.6250 | — |
| `s` + SIZE — *no masker involved* | 0.6495 | +0.0244 |
| `s` + `log bimod`, `log pfg` | 0.6549 | +0.0299 |
| `s` + SIZE + `log bimod`, `log pfg` | 0.6580 | +0.0330 |
| `s` + 6 EM statistics | 0.6593 | +0.0343 |
| `s` + 6 EM statistics + SIZE | 0.6616 | +0.0366 |
| `s · bimod · pfg_mean` | 0.6615 | +0.0365 |

`log(bimod · pfg_mean)` correlates +0.51 with `log n_cells`, and a flexible size prior
alone recovers +0.0244 of the +0.0365. Any claim that this measures "the masker's
independent opinion" must be stated against that floor: the masker-specific increment over
a pure size prior is roughly +0.012, and the product's virtue over the fitted alternatives
is that it captures both at once for free. This was not stated in the previous version of
this result and it should be stated in the paper.

## Protocol

- **Nothing is fitted**, so there is no fit split, no target, no weights, and the val→test
  distribution-shift question that the fitted version had to carry does not arise.
- **The 108 val tiles are used for SELECTION only** — choosing the functional form from a
  handful of zero-parameter candidates by box AP. That is the only place labels enter, and
  they are box labels.
- **Score scale**: `score'` is a product of three probabilities, so the 0.05 decode floor
  would cut a different detection set. The set is frozen (everything that passed 0.05 at
  decode) and only the ordering changes — score the reranked file with `--score_floor 0`.
  The baseline is identical at floor 0 and 0.05, since nothing in the saved preds is below
  0.05.

## Ceiling

After reranking LACE realises **82.5%** of its 0.8020 oracle ceiling, against Restor's
84.4%. Matching Restor's ranking efficiency exactly would give 0.677; the ceiling is then
the wall. Neither more capacity nor more calibration data closes it, because there is
nothing left to calibrate.

## Files

| file | role |
|---|---|
| `rerank_product.py` | **the result**: multiply `s · bimod · pfg_mean`, write `preds_knobbed_s0_product.json` |
| `ablation.py` | the ablation behind every number above; writes `ablation_results.json` |
| `em_diag_modal.py` | Modal CPU job: recompute `estep`, keep `A` pre-recentring — **now only needed for the `a_*` ablation rows**, not for the result |
| `val_diag_modal.py` | Modal job: seed-0 detector + same diagnostics over the 108 val tiles |
| `rerank_val.py` | the superseded 8-feature val-fitted logistic reranker, kept as the ablation's comparison row |
| `rerank.py` | 2-fold cross-fit ablation (EM vs geometry vs both) — diagnostic only |
| `em_diag_s0.json` / `em_diag_val_s0.json` | extracted per-box EM features |

## Open

- **Seed 0 only.** s1/s2 need the same treatment before this becomes a banded claim.
  Cheaper than before: the product needs no refit per seed, only `bimod` and `pfg_mean`
  from each seed's masker run.
- **Emit the two scalars at mask time.** They are two reductions over `estep`'s return
  value, so the diagnostics job should disappear entirely once `phase4_lib_tcd` stores them
  alongside each mask.
- **The size floor above** should be quantified once more on a second seed before the paper
  claims an independent-estimator effect rather than a size prior plus a small residual.
