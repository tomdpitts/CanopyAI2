# Fusion: LACE + SAM 3 on identical D1 boxes

**Question.** Does LACE add anything on top of SAM 3? Both maskers' masks already exist on the same frozen D1 boxes (test 439, sparse 236). If a fixed, untuned fusion beats SAM 3 alone, LACE has a role alongside SAM 3.

## Pre-registration (2026-11-14, written BEFORE any fusion number)

**Inputs (all stored, nothing re-rendered):**
- LACE: `dino_det/results/preds_dino_swinl_s0_{split}_R3_ef_r2048.json` (TABLE1_CURRENT row "LACE (D1 variant) + edge fix + R3": test 0.714 / 0.342, sparse 0.743 / 0.403). Its scores are the learned confidence ŝ (R3).
- SAM 3 zero-shot: `feat_cache/sam3_d1_{split}/<tid>.json` (masks + SAM scores, same boxes) with detector s from `preds_dino_swinl_s0_{split}_detscore_r2048.json`. Reference row √(s·SAM): test 0.7013 / 0.3553, sparse 0.7399 / 0.4252.

**Arms (fixed rules, no tunable, no selection):**

| arm | masks | ranking |
|---|---|---|
| `lace` (reference) | LACE | ŝ |
| `sam` (reference) | SAM 3 | √(s·SAM) |
| `F1_sam_at_shat` | SAM 3 | ŝ |
| `F2_sam_at_gmean_shat` | SAM 3 | √(ŝ·SAM) |
| `F3_sam_at_agree` | SAM 3 | ŝ · IoU(LACE mask, SAM mask) |
| `F4_intersection` | LACE ∩ SAM 3 | ŝ |
| `F5_union` | LACE ∪ SAM 3 | ŝ |

**Scoring.** `sam3_on_d1.per_tile` / `ap_at` (frozen COCOeval path: crowd arm, maxDets 600, floor 0), AP50 / AP75 / AP50:95. Gate: the `lace` and `sam` arms must reproduce their published scorer numbers to 4 dp. Paired tile bootstrap (2,000, seed 0) of each fusion arm − `sam` and − `lace`.

**Reading rule (fixed now).**
- A fusion "beats SAM 3" only if its AP50 and AP50:95 differences vs `sam` both have 95 % CIs above 0 on test AND the signs replicate on sparse.
- Five fusion arms are looked at once on test, so a single marginal CI is not a finding; sparse is the replication.
- Known prior (so not a surprise): SAM masks at LACE's older s′ ranking scored 0.7094 / 0.3532 on test (`mask_swap_diag`, 11-04), i.e. above SAM at AP50, slightly below at AP50:95.
- Caveats that hold whatever the outcome: ŝ was trained (on val) for LACE masks and is applied unchanged; single detector seed; test-set sampling only; any fusion costs BOTH maskers' compute (LACE 66 s + SAM 3 10 s per tile, `masker_cost/`).
- Earlier ruling stands (Tom, 10-16): SAM masks at a LACE ranking are not a *SAM baseline*. Here they are reported as a fusion, never as SAM's row.

Reported whatever the outcome.

## RESULTS (2026-11-14; `results/fusion_{test,sparse}.json`, logs `logs/fuse_*.log`)

Gate PASS on both splits: `lace` and `sam` arms reproduce their published rows to 4 dp. Mean IoU(LACE mask, SAM mask) 0.761 test / 0.764 sparse. Mask AP50 / AP75 / AP50:95:

| arm | test 439 | sparse 236 |
|---|---|---|
| `lace` (LACE masks, ŝ) | 0.7141 / 0.2713 / 0.3424 | 0.7434 / 0.4006 / 0.4031 |
| `sam` (SAM 3, √(s·SAM)) | 0.7013 / 0.3107 / 0.3553 | 0.7399 / 0.4505 / 0.4252 |
| `F1_sam_at_shat` | 0.7189 / 0.3058 / 0.3590 | 0.7495 / 0.4488 / 0.4268 |
| **`F2_sam_at_gmean_shat`** | **0.7203 / 0.3259 / 0.3678** | **0.7516 / 0.4609 / 0.4340** |
| `F3_sam_at_agree` | 0.7189 / 0.3202 / 0.3655 | 0.7478 / 0.4546 / 0.4295 |
| `F4_intersection` | 0.7185 / 0.2747 / 0.3466 | 0.7445 / 0.3964 / 0.4019 |
| `F5_union` | 0.7135 / 0.3034 / 0.3545 | 0.7486 / 0.4471 / 0.4263 |

Paired tile bootstrap, fusion − `sam` (diff [95 % CI]):

| arm | split | ΔAP50 | ΔAP75 | ΔAP50:95 |
|---|---|---|---|---|
| F1 | test | +0.0176 [0.0118, 0.0235] | −0.0049 [−0.0087, −0.0009] | +0.0037 [0.0006, 0.0068] |
| F1 | sparse | +0.0096 [0.0052, 0.0158] | −0.0017 [−0.0081, 0.0065] | +0.0016 [−0.0025, 0.0071] |
| **F2** | test | +0.0190 [0.0135, 0.0247] | +0.0152 [0.0118, 0.0182] | +0.0125 [0.0097, 0.0154] |
| **F2** | sparse | +0.0117 [0.0063, 0.0195] | +0.0104 [0.0042, 0.0169] | +0.0088 [0.0050, 0.0140] |
| F3 | test | +0.0176 [0.0121, 0.0233] | +0.0095 [0.0058, 0.0132] | +0.0102 [0.0070, 0.0129] |
| F3 | sparse | +0.0079 [0.0034, 0.0148] | +0.0041 [−0.0007, 0.0101] | +0.0043 [0.0011, 0.0088] |
| F4 | test | +0.0172 [0.0110, 0.0238] | −0.0360 [−0.0446, −0.0284] | −0.0087 [−0.0134, −0.0047] |
| F4 | sparse | +0.0046 [−0.0018, 0.0122] | −0.0541 [−0.0721, −0.0365] | −0.0233 [−0.0312, −0.0135] |
| F5 | test | +0.0122 [0.0060, 0.0197] | −0.0073 [−0.0175, 0.0030] | −0.0008 [−0.0057, 0.0043] |
| F5 | sparse | +0.0087 [0.0034, 0.0151] | −0.0034 [−0.0149, 0.0115] | +0.0011 [−0.0044, 0.0085] |

F2 − `lace`: test +0.0062 [0.0011, 0.0090] / +0.0546 / +0.0254 [0.0212, 0.0296]; sparse +0.0082 [0.0022, 0.0126] / +0.0603 / +0.0309 [0.0233, 0.0378]. All fusion − `lace` rows are in the JSON.

**By the pre-registered rule: F2 and F3 beat SAM 3** (AP50 and AP50:95 CIs above 0 on test, signs replicate on sparse; for F2 the sparse CIs are above 0 too). F1 passes on test, sparse AP50:95 CI spans 0. Mask-level fusions (F4, F5) do not beat SAM 3 at strict IoU: the gain is ranking, not mask shape.

### POST-HOC control (`control.py`, `results/control_{test,sparse}.json`): does the gain need LACE's mask?

F2c = SAM 3 masks ranked by √(ŝ₃·SAM), ŝ₃ = the learned confidence WITHOUT its LACE-mask input (T3 `no_logpm` scores: s, log h, IoU↑ only).

| | test AP50 / AP75 / AP50:95 | sparse |
|---|---|---|
| `sam` | 0.7013 / 0.3107 / 0.3553 | 0.7399 / 0.4505 / 0.4252 |
| F2c (no LACE mask anywhere) | 0.7121 / 0.3225 / 0.3638 | 0.7476 / 0.4588 / 0.4318 |
| F2 (ŝ with LACE mask input) | 0.7203 / 0.3259 / 0.3678 | 0.7516 / 0.4609 / 0.4340 |

| | split | ΔAP50 | ΔAP75 | ΔAP50:95 |
|---|---|---|---|---|
| F2c − sam (box-only learned confidence) | test | +0.0108 [0.0063, 0.0154] | +0.0118 [0.0092, 0.0142] | +0.0085 [0.0061, 0.0108] |
| | sparse | +0.0077 [0.0029, 0.0146] | +0.0083 [0.0027, 0.0146] | +0.0066 [0.0032, 0.0113] |
| **F2 − F2c (LACE masker's contribution)** | test | **+0.0082 [0.0062, 0.0100]** | +0.0034 [0.0020, 0.0045] | +0.0040 [0.0030, 0.0051] |
| | sparse | **+0.0040 [0.0018, 0.0064]** | +0.0021 [−0.0000, 0.0044] | +0.0022 [0.0006, 0.0038] |

## Reading

- **LACE + SAM 3 beats SAM 3 alone and LACE alone** on both splits, at every metric: 0.720 / 0.368 test, 0.752 / 0.434 sparse.
- **Decomposition of the +0.019 test AP50 over SAM 3:** +0.011 from the learned confidence's box-only inputs (needs no LACE masker), +0.008 from the LACE masker's posterior as a confidence signal. The masker's share is significant on both splits and equals its value inside LACE itself (T3: mask input +0.008 test / +0.004 sparse).
- **So the LACE masker's role in a SAM 3 world is as a confidence signal, not as the mask.** SAM 3 draws better outlines; LACE's box-supervised foreground posterior tells you which detections are real trees, information SAM 3's own score does not carry.
- **Caveats:** ŝ is post-hoc (trained on val for LACE masks, applied unchanged); five arms looked at once on test (F2's CIs are far from 0 and replicate on sparse); the control is post-hoc; single detector seed; test-set sampling only; the fusion costs both maskers (LACE 66 s deployed or 25 s at 4 shifts, + SAM 3 10 s per tile, `masker_cost/`). The masker's +0.008 costs 2.5–6.6× SAM 3's compute.

## Commands

```
cd boxinst_commonality_tcd_04/fusion_lace_sam3 && PYTHONPATH=/Users/tompitts/dphil/CanopyAI2 ../../.venv/bin/python -u fuse.py test    # then: sparse; then control.py test|sparse
```
