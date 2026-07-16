# tcd04_semseg — two-head model: ITC detection + semantic tree-cover (canopy)

ISOLATED, DELETABLE experiment on top of `boxinst_commonality_tcd_04`. Tests whether
one model with **two heads sharing a trunk** can add canopy tree-cover (area-F1
lever) **without compromising ITC instance mAP50**. `rm -rf tcd04_semseg/` to discard
(its only artifact is a ~16MB checkpoint; it reuses the main package's caches).

## Design

`MultiHead8` = the exact `Detector8` trunk (frozen-feat stem + tower + 16→8px up) with
two heads:
- **det head** (5ch CenterNet): ITC instances — the `det_t8` head, unchanged.
- **sem head** (1ch): semantic **tree-cover** logit = P(pixel is tree, incl. canopy).

**Zero-risk training:** warm-start from `det_t8`, **freeze trunk + det head**, train
**only** the sem head (one conv on frozen features) on the (ITC-box ∪ canopy)
foreground. So the det output is byte-identical to `det_t8` → ITC mAP50 unchanged by
construction. The sem operating threshold is picked on held-out val train-tiles.

Why this answers "target canopy without hurting ITC": canopy has no instance labels,
so it can't supervise the ITC head; instead it supervises a *separate semantic* head.
Area F1 is pixel-level, so a tree-cover mask (not instances) is all it needs.

## Result (full 439 test, web backbone)

| metric | value | vs baseline |
|--------|-------|-------------|
| **ITC mask mAP50** | **0.4986** | == `det_t8` 0.499 (freeze holds) |
| area F1, canopy-EXCLUDED (old metric) | 0.587 | unchanged (sem head doesn't touch it) |
| area F1, canopy-INCLUDED, instances only | 0.196 | instances can't cover canopy (R 0.11) |
| **area F1, canopy-INCLUDED, + sem head** | **0.632** | **+0.44 from the semantic head** |

**Read:** if the metric counts canopy as tree-cover (the meaningful one for tree-cover
mapping), the semantic head lifts area F1 from 0.196 → 0.632 (recall 0.11 → 0.57),
while ITC instance mAP50 is untouched at 0.499. The two objectives are cleanly
decoupled: one model, one forward, two heads.

Caveats: canopy-included (0.632) and canopy-excluded (0.587) are *different* metrics
(different denominators) — not a like-for-like +0.045; the honest gain is the +0.44
the sem head adds *under the canopy-inclusive metric*. sem head is frozen-trunk only
(a probe); joint fine-tuning could raise the sem F1 further but would perturb ITC
mAP50 (untested). `sem_thr` picked on val tree-cover F1.

## Usage

```bash
V=.venv/bin/python
$V -m tcd04_semseg.train --tag sem --epochs 20     # trains sem head only (~10min)
$V -m tcd04_semseg.evaluate --mh mh_sem            # full 439: mAP50 + area F1 (both)
```

Files: `model.py` (MultiHead8), `train.py` (sem-head-only trainer + val-thr pick),
`evaluate.py` (439 eval, 4 metrics). Reuses `boxinst_commonality_tcd_04` caches,
`TCDMasker`, and eval helpers.
