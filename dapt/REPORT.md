# DAPT for dryland crown detection — results (web vs sat vs DAPT)

> **STATUS 2026-07-14:** the results below are **v1/v2** (old pool, 33-tile fixed
> split, NEON included). The **mechanism findings stand** (web≫sat; DAPT gain is
> small, arid-specific, ~60% low-level appearance / ~40% structure; instance-level not
> occupancy). The **significance question** (effect ~+0.018 vs 33-tile CI ±0.02) is
> re-run in **v3** — arid-only, re-annotated, repeated-k-fold over 81 leakage-safe
> tiles. **v3 verdict (see `# v3` section at the end): the DAPT effect does NOT
> replicate (L2 gap −0.000 ± 0.003); web ≫ sat does (+0.120).** Plan + cost:
> `dapt/ssl/SPEC.md` `## v3 study`; code isolated in `dapt/v3/`.

---

**Setup (v1/v2).** Frozen DINOv3 ViT-L backbones; one fixed CenterNet-style probe, identical
targets/NMS/seeds across arms; only the backbone differs. Test = 33 tiles (18 arid
WON+BRU / 15 non-arid NEON), ~360 crowns. Primary instrument = L2 (MLP) probe,
test mAP50, paired-gap bootstrap over tiles (5 probe seeds). DAPT = continue DINOv3
SSL (DINO+iBOT+Koleo, ImageNet norm, Gram off) on 1,084 unlabelled arid tiles;
checkpoint selected on VAL; reported once on TEST. Two SSL seeds (42, 123). Total
GPU ≈ $6 (3 Modal A100 runs).

## Backbone baselines (test mAP50, L2, 5 seeds)

| arm | mAP50 | vs web (paired) |
|---|---|---|
| **DAPT** (seed42 winner i999) | **0.483 ± 0.009** | **+0.033 CI[+0.009,+0.058] RESOLVED** |
| web (lvd1689m) | 0.450 ± 0.015 | — |
| sat (sat493m) | 0.354 ± 0.027 | web−sat +0.094 CI[+0.038,+0.149] RESOLVED |

web ≫ sat is the robust, large effect (detection AND area-F1). DAPT > web is real but
small (below).

## Pooled 5-SSL-seed estimate (P1K protocol) — the definitive effect size

Five independent SSL seeds (43, 124, 317, 588, 902), short 1000-iter protocol
(validated: seed-43 gate +0.022 matched the long-run family at 1/5 the cost), each
val-selected, pooled paired-bootstrap vs web:

| subset | pooled gap | 95% CI | per-SSL-seed spread |
|---|---|---|---|
| FULL | +0.018 | [−0.001, +0.036] | **+0.018 ± 0.003** |
| arid | +0.020 | [−0.004, +0.043] | **+0.020 ± 0.004** |
| NEON | +0.002 | [−0.016, +0.014] | +0.002 ± 0.003 |

**The headline finding: the DAPT effect is real and extraordinarily reproducible
(±0.003 across 5 independent SSL runs) but its magnitude (~+0.018–0.020 arid, ≈0
NEON) sits just below what a 33-tile test can resolve.** Pooling SSL seeds did NOT
push it over the CI — because the residual uncertainty is the TEST SET, not SSL
variability (per-seed spread ±0.003 ≪ the ±0.02 bootstrap CI). Actionable conclusion:
**more training/seeds won't resolve it; only a larger annotated test set will.** The
pre-registered single run (+0.033, resolved) was on the high side of this
distribution; the pooled ~+0.018 is the honest best estimate.

## Five findings

**1. DAPT > web: small, arid-specific, directionally reproducible — not a slam dunk.**
Pre-registered seed-42 endpoint RESOLVED (+0.033). Independent seed-123 replication
same sign + same arid-concentration but magnitude smaller and NOT independently
resolved:

| | full-test gap | arid gap | NEON gap |
|---|---|---|---|
| seed 42 (i999) | +0.033 RESOLVED | +0.033 RESOLVED | +0.017 n.r. |
| seed 123 (i499) | +0.017 (P=0.94) n.r. | +0.022 (P=0.93) n.r. | −0.002 n.r. |

Both seeds: gain concentrated in arid, ≈0 on NEON (the arid-specificity signature).
Honest read: a **real but small (~+0.02–0.03) arid-specific gain sitting near the
33-tile resolution floor**; the pre-registered run landed on the high side. Pooling
seeds would likely resolve it — worth doing.

**2. Mechanism (shuffled-pool control) — the most interesting result.** SSL on a
64px-block-SHUFFLED arid pool (arid color/texture kept, scene layout + inter-crown
structure destroyed) still gains **+0.021 (P=0.97), arid-concentrated**. Real vs
shuffled is a small but RESOLVED **+0.013 CI[+0.003,+0.021]**. So the total DAPT gain
decomposes cleanly:

```
web 0.450  --(+0.021 low-level arid appearance)-->  shuffled 0.471
           --(+0.013 coherent scene/structure, RESOLVED)-->  real 0.483   (≈ +0.033 total)
```

**~60% of the DAPT benefit is low-level arid appearance adaptation** (photometric /
crown-scale texture, invariant to layout); **~40% is coherent arid structure.** Both
positive; the cheap low-level part dominates. Caveat: 64px blocks keep individual
crowns largely intact, so "structure destroyed" = global layout + long-range context,
not crown-scale texture.

**3. Ceiling gain, NOT label efficiency** (contra the usual DAPT narrative). At
25%/50%/100% labels DAPT vs web = −0.001 / −0.009 / **+0.033**: the advantage appears
only at full labels; at 25/50% DAPT ≈ web (25% is the degenerate near-zero regime).
DAPT raises the achievable ceiling rather than reducing label need here.

**4. Instance-level, not occupancy-level.** Area-F1 (semantic seg, box-fill
pseudo-GT): DAPT ≈ web (0.761 vs 0.764 MLP; 0.640 vs 0.643 linear), both > sat.
DAPT beats web on detection but ties on pixel occupancy → the gain is in centre/box
discrimination (the counting task), not bulk canopy coverage.

**5. Robustness of the SSL itself.** No catastrophic forgetting at any dose despite
features drifting hard from web (cosine 0.66→0.39 by iter 3000); val trail FLAT
0.40–0.41 across the whole 0–5k range, web = 0.400 identical protocol. Drift
trajectory + flat trail reproduced across both SSL seeds. Winning dose ≈ 500–1000
iters (≈ $0.40 GPU); iters 1000–5000 added nothing.

## L1 linear (secondary; known-insensitive here)

web 0.181 / sat 0.163 / DAPT 0.183; DAPT−web +0.005 n.r. L1 could not resolve even
web−sat (which is +0.094 at L2), so its silence is expected insensitivity — crown
info is mostly non-linearly decodable from these features.

## Verdict vs GOAL decision rule

**DAPT > web (and ≫ sat) — arid adaptation helps, but the effect is small and its
mechanism is mostly low-level.** Extends the "web wins at 0.1 m" story: web is the
right init; ~$0.40 of arid SSL adds a real arid-specific gain, ~60% of which is cheap
photometric/texture adaptation and ~40% coherent structure. Not the label-efficiency
win the prior predicted — a modest ceiling lift on the counting task.

## Caveats

- 33-tile test: gains < ~±0.02 unresolvable; the pooled DAPT−web effect (+0.018) sits
  right at that floor — RESOLVING it requires a bigger test set, not more SSL (the
  per-seed spread is only ±0.003).
- Shuffled-arid still contains arid statistics, so it does not exclude a "generic
  extra-SSL on any tree imagery" component — but the NEON-neutral pattern argues
  against pure generic SSL. A non-arid SSL control would close this fully.
- Single test cohort; area-F1 uses box-fill pseudo-GT (fair for ranking, not true
  canopy F1); checkpoint selection adds mild optimism; LR ran hot (7e-5, documented).

## Artifacts

Organized under `dapt/artifacts/{detection,selection,labeff,semseg,probes,logs}/`
(see `artifacts/README.md` for layout + naming). Winner ckpts in `dapt/ckpt/`
(non-winners archived on the Modal volume). SSL runs: out_s42 / out_s123 /
out_s42_shuf on `dinov3-dapt-arid-vol`.

---

# v3 (2026-07-14) — repeated k-fold, leakage-safe, arid-only: THE DAPT EFFECT DOES NOT SURVIVE

**One-line verdict: on the re-annotated, leakage-safe, arid-only v3 design, arid DAPT
provides NO gain over off-the-shelf web DINOv3 (L2 gap −0.000, CI ±0.003) — the v1/v2
~+0.018–0.020 effect does not replicate. web ≫ sat replicates emphatically (+0.120).**

**Setup.** 97 re-annotated arid tiles (WON+BRU, NEON dropped): 81 leakage-safe eval
tiles (53 BRU + 28 WON, 1,375 boxes; incl. the 2026-07-14 correction moving 19
strip-sourced BRU tiles from train/ to test/) + 16 WON pool-overlap tiles (excluded
from the headline entirely). Fresh SSL pool: 1,051 tiles from BRU162-center80 /
CAN091/095/117 / WON003-right50 (below the ~1,300–1,500 estimate; ≥95%-valid white
filter dropped more than projected). 5 SSL seeds (101–501), P1K protocol, ImageNet
norm, on `dinov3-dapt-v3-vol`; ~$2.6 Modal GPU all-in. Per-seed val-selection
(inner-val only, i499 vs i999: mlp picked i499 for s101/s501, i999 otherwise; cosine
gates all healthy, smooth 0.93→0.74→0.65 decay). Evaluation: 5-fold × 3-repeat
k-fold over the 81 tiles, identical folds across arms, 5 probe seeds, OOF pooled →
paired-gap bootstrap; pooled-seed estimator = 25 runs (5 SSL × 5 probe seeds).
Selection val-only; the paired test gap was read once, at the end.

## Headline (L2/MLP, OOF mAP50 over 81 tiles)

| arm | mAP50 |
|---|---|
| DAPT (pooled 5-seed winners) | 0.2556 |
| web (lvd1689m) | 0.2560 |
| sat (sat493m) | 0.1366 |

| paired gap (L2) | FULL 81t | WON 28t | BRU 53t |
|---|---|---|---|
| **DAPT − web** | **−0.000 CI[−0.003,+0.003] n.r.** | −0.000 CI[−0.004,+0.003] n.r. | −0.002 CI[−0.004,−0.000] RESOLVED (negative) |
| web − sat | +0.120 CI[+0.100,+0.140] RESOLVED | +0.128 RESOLVED | +0.035 RESOLVED |

The design delivered the power it promised (paired CI half-width ±0.003 ≪ the ±0.02
v1/v2 floor) — and at that resolution the effect is **zero**, with a tiny *negative*
resolved gap on BRU. The DAPT arm is functional (matches web exactly; no collapse;
adaptation confirmed by cosine drift), so this is a genuine null, not a broken arm.

**L1/linear is at floor (all arms 0.06–0.08 mAP50)** — as in v1/v2 it is not a usable
instrument; its tiny resolved gaps (DAPT−web +0.007; sat above web) are floor
artifacts and carry no weight against the L2 readout.

## Interpretation

- The v1/v2 pooled +0.018–0.020 arid effect (±0.003 across seeds) came from a design
  with WON pool/test adjacency-leakage risk, first-pass annotations, NEON-mixed
  probes, and a starved fixed split. Removing all four removes the effect. Which one
  carried it is not isolated here; the leakage + annotation-quality pair is the prime
  suspect.
- web ≫ sat is design-robust (replicated at +0.120 on a clean 81-tile instrument) —
  backbone provenance (curated web vs satellite SSL) remains the dominant lever.
- Actionable: **do not spend further GPU on arid DAPT for this detector**; the
  planned Restor OAM-TCD / NEON DAPT follow-ups should carry pre-registered nulls as
  the default expectation.

## v3 artifacts

`dapt/v3/artifacts/{kfold_mlp,kfold_linear,val_select,winners}.json`; OOF runs in
`dapt/v3/cache/oof/`; ckpts `dapt/v3/ckpt/` (registered as `dapt_v3_*`); SSL outputs
on `dinov3-dapt-v3-vol` (out_v3_s101–501); pool `dapt/v3/ssl/pool/`. Runbook + dated
corrections: `dapt/ssl/SPEC.md` `## v3 study`.
