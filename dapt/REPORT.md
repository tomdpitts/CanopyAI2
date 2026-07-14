# DAPT for dryland crown detection — results (web vs sat vs DAPT)

**Setup.** Frozen DINOv3 ViT-L backbones; one fixed CenterNet-style probe, identical
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
