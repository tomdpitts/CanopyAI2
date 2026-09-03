# Sparse OAM-TCD 236 — LACE + posterior product vs DetecTree2

Same frozen scorer as the 439 (`score_coco.py`, pycocotools COCOeval, masks 512², maxDets
600, `--score_floor 0`, canopy as `iscrowd`), against `data/tcd_sparse/sparse_gt.json`
(236 tiles, 14,937 GT crowns). **The sparse slice shares no tile with the 900 images used
for training**, so this is a held-out generalisation test, not a second read of the same
distribution. No retraining, no new detector — the saved seed predictions are re-ranked.

| Method | CN AP50 | CN AP75 | CN AP50:95 | FP AP50 | Box AP50 | dets |
|---|---|---|---|---|---|---|
| LACE seed 0 | 0.6699 | 0.2055 | 0.2969 | 0.6049 | 0.6197 | 79,562 |
| LACE seed 1 | 0.6476 | 0.1879 | 0.2837 | 0.5941 | 0.5744 | 86,784 |
| LACE seed 2 | 0.6504 | 0.1978 | 0.2880 | 0.5950 | 0.6252 | 90,526 |
| **LACE 3-seed** | **0.6560 ±0.0121** | **0.1971 ±0.0088** | **0.2895 ±0.0067** | **0.5980 ±0.0060** | **0.6064 ±0.0279** | 85,624 |
| LACE + product, seed 0 | 0.7018 | 0.2427 | 0.3218 | 0.6280 | 0.6543 | 79,562 |
| LACE + product, seed 1 | 0.6831 | 0.2214 | 0.3081 | 0.6217 | 0.6134 | 86,784 |
| LACE + product, seed 2 | 0.6891 | 0.2336 | 0.3146 | 0.6220 | 0.6650 | 90,526 |
| **LACE + product, 3-seed** | **0.6913 ±0.0095** | **0.2326 ±0.0107** | **0.3148 ±0.0069** | **0.6239 ±0.0036** | **0.6442 ±0.0272** | 85,624 |
| **DetecTree2 (fine-tuned, converged 2026-09-03)** | 0.6118 | 0.2155 | 0.2800 | 0.5666 | 0.5897 | 99,558 |

## What it shows

**1. LACE + product beats DetecTree2 on every metric** — CN AP50 0.6913 vs 0.6118 (**+0.080**),
AP75 +0.017, AP50:95 +0.035, box AP50 +0.055 — from boxes only, against a full-mask-supervised
Mask R-CNN. Note this is against the **converged** DetecTree2 (retrained 2026-09-03); the earlier
+0.138 was measured against an under-trained checkpoint and must not be quoted.

**2. The confidence fix generalises, slightly better than on the 439** — +0.0354 CN AP50 here vs +0.0325 on the test set, positive on all three seeds and all five metrics. This matters: the
functional form was selected on the 108 val tiles, which come from the *training* distribution.
Had `s · bimod · pfg_mean` been quietly fitted to OAM-TCD's holdout the gain would have shrunk
here; it grew. The zero-parameter form is what makes that claim clean — there are no weights
that could have carried distribution-specific information across.

**3. Both methods score higher on sparse than on the 439** (LACE 0.69 vs 0.66; DetecTree2 0.61 vs
0.60), consistent with open-canopy crowns being easier to separate. The regime is easier for
both, so the margin is not an artifact of one model liking it more.

**4. On-thesis, and it survives the retrain**: the sparse/open-canopy regime is the paper's
motivation, and LACE's margin over DetecTree2 is larger there (**+0.080**) than on the 439
(**+0.066**). The stronger point is at strict IoU: on the 439 the converged DetecTree2 *beats*
LACE at AP75 (0.1748 vs 0.1649), whereas on the sparse slice LACE still leads it (0.2326 vs
0.2155). The masker's disadvantage against mask supervision is a dense-canopy phenomenon; in the
open-canopy regime the paper is about, it does not appear.

Artifacts: `sparse236/*.json` (DetecTree2: `dt2_s0v2.json`; the superseded `dt2_s0.json` was deleted, see git history and `../../detectree2_baseline/RETRAIN_V2.md`). Posterior stats: `../confidence/em_post_sparse_s*.json`.
Rebuild: `confidence/apply_product.py` then `score_coco.py --gt data/tcd_sparse/sparse_gt.json`.
