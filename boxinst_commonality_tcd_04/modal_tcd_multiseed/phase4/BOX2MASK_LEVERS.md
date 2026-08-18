# Box→mask opportunity — local lever search (commonality-EM, β=0.5-fixed, seed-0)

Goal: raise mask mAP50 above the settled **0.620** *without touching detection* and while
**preserving the training-free commonality-EM novelty** (prefer single-knob, inference-time
levers — "don't rip up the floorboards"). All work is LOCAL: the 4-phase test features +
fixed masker are pulled, the soft P(fg) is regenerated (verified to reproduce Modal 0.620
exactly), and every lever is scored with the project's own COCO-101pt mask-AP core.

Pilot = 40 tiles (happens to reproduce the full-439 number, 0.6203, so it's well calibrated).

## Ceiling / target context (same seed-0 boxes, 439)
- **Oracle-perfect masks, same detections: mask AP50 = 0.737** (mAP50-95 = 0.737). This is the
  box→mask headroom with detection frozen — *maxR (recall ceiling) 0.818, so 0.70+ ultimately
  needs better DETECTION, not just masks.*
- **SAM 3 (box-prompt, same boxes) = 0.633 / mAP50-95 0.268** — a FLOOR showing >0.62 is
  extractable; its edge is at HIGH IoU (boundary precision), and it recovers no extra crowns.
- **Our EM BEATS SAM on GT boxes** (mean crown IoU 0.747 vs 0.724; IoU≥0.5 0.987 vs 0.947,
  masker_lab 60-tile). ⇒ the masker is not worse than SAM — it is **more sensitive to box
  imprecision** (it normalizes its spatial prior by the box corners and clips to the box).
  The whole predicted-box gap is **box-robustness**.

## Levers that DON'T help (exhausted — all ≤ +0.003 or negative on the pilot)
| lever family | best | verdict |
|---|---|---|
| content-aware thresholding (global/size-gated/relative/Otsu/hysteresis) | +0.003 | current mask_thr 0.25 ~optimal |
| binary-mask shape post-proc (largest-CC/center-CC/fill-holes/open/close/hull) | +0.001 | masks already clean (0.8% multi-CC, 0.8% holes) |
| RGB / vegetation (ExG) guided boundary refinement (guided filter, ExG-gating) | **negative** | TCD boundaries are crown-to-**crown** (image-invisible); weak guides can't beat it |
| EM-internal (center-weight / confidence-sharpen / prototype-consistency) | +0.0016 | prototypes don't separate touching crowns at the boundary |
| instance-competition (boxes compete for boundary pixels) | **negative** | removes correct boundary pixels too |
| posterior-driven iterative box refinement | **negative** (collapse) | re-boxing from the posterior is a positive-feedback loop |

Takeaway: **nothing that post-processes the *current* posterior beats it by >0.003.** Gains
require changing how the posterior is *computed* from the (imprecise) box.

## Lever that DOES help — prior-weight α (the box-robustness knob) ✅
The E-step posterior is `sigmoid( A + α·logit(psum) )`, where `A` is the box-INDEPENDENT
appearance log-ratio (crown prototypes vs frozen bg mixture) and `psum` is the box-NORMALIZED
spatial-prior fg mass. Current code fixes **α=1**. An imprecise predicted box misplaces
`psum`, so relaxing α lets the reliable appearance term drive:

| α (prior weight) | mask AP50 | mAP50-95 |
|---|---|---|
| 1.0 (current) | 0.6203 | 0.2450 |
| 0.7 | 0.6216 | 0.2504 |
| **0.5** | **0.6292** | 0.2544 |
| **0.3** | **0.6290** | **0.2592** |
| 0.0 (appearance only) | 0.6110 | 0.2395 |

**α≈0.3–0.5 → +0.009 AP50 / +0.014 mAP50-95 (pilot).** One scalar in `estep`, no re-fit, no
re-extraction, novelty intact. mAP50-95 gain (+0.014) exceeds the AP50 gain → it is closing
the SAME boundary-precision gap SAM wins on. Dilating the mask region does NOT help (−).

*Note α<1 is a pure inference change to `boxinst_commonality.em.estep` (add a prior-weight
arg; default 1.0 keeps every other consumer identical). Confirm on all 439 via a Modal eval
before committing to the code default.*

## Second knob — appearance sharpness κ, and the COMBO ✅✅
Sharpening the vMF concentration at inference (`k = ks·κ`, so `zc` and `bgll` both scale)
makes the appearance term more decisive → tighter boundaries. It is ORTHOGONAL to α, and
they stack. Joint pilot sweep (α × ks × mask_thr):

| config | mask AP50 | mAP50-95 |
|---|---|---|
| baseline (α=1, ks=1, thr=0.25) | 0.6203 | 0.245 |
| **α=0.4, ks=1.6, thr=0.30** | **0.636** | 0.255 |
| α=0.3, ks=1.6, thr=0.25 (best joint) | 0.635 | **0.262** |
| _SAM 3 floor (ref)_ | _0.633_ | _0.268_ |

**+0.016 AP50 / +0.017 mAP50-95 (pilot) — BEATS the SAM floor with the training-free masker.**
Broad optimum (0.634–0.636 across α∈[0.3,0.4], ks∈[1.3,2.0], thr∈[0.25,0.30]) → robust, not a
knife-edge. Both are pure inference scalars in `estep` (add `prior_weight` and a `kappa` scale;
defaults 1.0 keep every other consumer byte-identical). bg-weight, contrast-temp, size-adaptive
α, dilation, box-refinement: all neutral/negative — the two winners are α and κ.

## Confirmed on all 439 (seed 0) + a THIRD lever
- **α=0.3, κ×1.6, thr=0.25 on all 439:** mask AP50 **0.625** (+0.005) / mAP50-95 **0.257** (+0.013).
  AP50 gain shrank pilot→130→439 (+0.016→+0.009→+0.005, regression to mean); the mAP50-95 gain (+0.013)
  held exactly. AP50 gain < 5-seed σ (0.011) → marginal on the headline; mAP50-95 is the robust win.
- **Box-jitter TTA** (average posterior over ±10% box translations, 9-pt): **+0.009 AP50 / +0.008
  mAP50-95 on top of α/κ** (130 tiles: 0.6106→0.6199). Marginalizes box imprecision without the
  feedback-collapse of hard refinement — the box-robustness theme, working. **Cost: ~9× masker
  compute at inference** (9 box_masks/box) — floorboard-safe in code but a real deployment cost; needs
  439 confirmation (may also regress). β=0⊕β=0.5 blend: neutral/negative.

## 3-seed knobbed band (α=0.3, κ×1.6, thr=0.25) — DEPLOYABLE, paired
| seed | vanilla AP50 → knobbed | vanilla 50-95 → knobbed |
|---|---|---|
| 0 | 0.6203 → 0.6250 | 0.2440 → 0.2574 |
| 1 | 0.6253 → **0.6358** | 0.2362 → 0.2567 |
| 2 | 0.6241 → 0.6294 | 0.2411 → 0.2564 |
| **mean** | **0.623 → 0.630 ± 0.005** (+0.007) | **0.240 → 0.257 ± 0.001** (+0.016) |

EVERY seed improves on BOTH metrics → robust (not within-noise). Box AP50 unchanged (0.603,
masker-invariant). SAM-level (0.633) with a training-free masker, same comparison basis.
Seeds 3,4 not run (user chose 3 seeds). Defaults NOT yet promoted (still 1.0).

## Status / next
- Confirming the +0.016 against TCD noise by scaling the local pilot 40→130 tiles.
- Then ONE Modal eval on all 439 with (α, ks, thr) to lock the deployable number + implement
  the two-scalar change in `boxinst_commonality.em.estep` (novelty intact).
- Deferred (rips up floorboards): finer real-4px features (Modal re-extraction) for more
  boundary precision; better DETECTION is the only route past ~0.633→0.70 (SAM maxR 0.818).

## Selection protocol for α/κ — PRE-REGISTERED 2026-08-18, before the val grid landed

Everything above chose α/κ on TEST tiles (40-tile pilot of the 439 → 130 → all 439), so the
settled 0.630 is a tuned-on-test number. `sweep_val_knobs` re-runs the same neighbourhood on
the held-out 108-tile VAL split (7041 crowns; `val_gt.json`, prepare_test convention, scored
by the identical AP core). This rule was written to disk while that sweep was at 60/108 tiles
and NO val cell had been read:

> Select the (α, κ) maximising val mask **mAP50-95** — the metric that held stable across the
> 40→130→439 scaling, where the AP50 gain decayed (+0.016→+0.009→+0.005). If any grid point
> within **0.002 mAP50-95** of the val maximum has already been evaluated on test, report that
> point: ties inside the noise band break toward the pre-existing evaluation.

Rationale for the tie band: the test sweep found a BROAD optimum (0.634–0.636 across
α∈[0.3,0.4], κ∈[1.3,2.0], thr∈[0.25,0.30]), so the grid argmax is a near-coin-toss among
~6 plateau points, and val carries ~1.9× the test's sampling noise (7041 vs 25705 crowns).
The band is fixed HERE at 0.002 so it cannot be widened afterwards to capture a preferred
answer.

**If the val optimum falls outside the band**, the honest cost is 3 test evals at the val-chosen
setting (~$6, ~1 h as parallel single-seed containers) — NOT retaining (0.3, 1.6) because its
test number already exists. Retaining it in that case would be the same tuned-on-test problem
in val-shaped disguise.

Report the FULL val surface alongside whichever point is selected, so the plateau is visible.

### Val-grid RESULT (2026-08-18) — the pre-registered rule selects (α=0.3, κ×1.6, thr=0.25)

`sweep_val_knobs`, seed-0 detector, β=0.5-fixed masker, 108 val tiles / 7041 crowns
(val box mAP50 0.6065). Full surface, mask mAP50-95 (mask AP50 in brackets):

| α \ κ | 1.0 | 1.3 | **1.6** | 2.0 |
|---|---|---|---|---|
| **0.3** | 0.2647 (0.6348) | 0.2666 (0.6366) | **0.2680 (0.6392)** | 0.2677 (0.6382) |
| 0.4 | 0.2624 (0.6350) | 0.2649 (0.6361) | 0.2661 (0.6382) | 0.2673 (0.6383) |
| 0.5 | 0.2585 (0.6346) | 0.2623 (0.6353) | 0.2635 (0.6367) | 0.2651 (0.6375) |
| 0.7 | 0.2537 (0.6338) | 0.2566 (0.6348) | 0.2589 (0.6347) | 0.2622 (0.6367) |
| 1.0 (vanilla) | 0.2490 (0.6304) | 0.2512 (0.6320) | 0.2534 (0.6326) | 0.2560 (0.6329) |

thr=0.30 variants: (0.3, 1.6) 0.2639 (0.6386); (0.4, 1.6) 0.2610 (0.6366) — both below their
thr=0.25 counterparts, so thr 0.25 survives on val too.

**(0.3, 1.6, 0.25) is the outright val argmax on BOTH metrics** — the tie-break clause was never
needed. The test-side choice is therefore reproduced by a selection that never touches test data,
and the existing 3-seed 439 band (0.630 ± 0.005 / 0.257 ± 0.001) stands as select-on-val /
report-on-test. **No test re-evaluation was required.**

**What this does and does NOT establish.** It does establish that α/κ were not chosen on the
evaluation set — the tuned-on-test objection is answered. It does NOT establish that (0.3, 1.6) is
uniquely optimal: the margin over the runner-up (0.3, 2.0) is **0.0003 mAP50-95**, far inside
noise, and the whole α≤0.4 / κ≥1.3 corner sits within 0.002. The honest reading is a **broad
plateau whose corner the rule happens to land in**; any point in it would give materially the same
test number. What IS robust across the surface: α<1 beats α=1 at every κ (+0.012–0.016 mAP50-95),
and κ>1 beats κ=1 at every α — i.e. both knobs help monotonically in their useful range, which is
the claim the paper should make.

**Residual caveats (disclose in the paper).** (1) Val tiles come from the same OAM-TCD train pool,
not an independent site. (2) The seed-0 detector early-stopped on these val tiles, so val is held
out for the MASKER knobs but not for detection — box AP is masker-invariant so the selection is
uncontaminated, though val boxes may be marginally better-fit than test boxes. (3) Selection used
val CROWN POLYGONS: 2 scalars are fitted against 108 tiles of mask labels, so "box+canopy-weak" is
exact for training and near-exact overall — state it as "two inference scalars selected on 108
held-out images". (4) `mask_thr=0.25` was originally chosen test-side; the two thr=0.30 cells above
re-confirm 0.25 on val, but a full thr sweep on val was not run.
