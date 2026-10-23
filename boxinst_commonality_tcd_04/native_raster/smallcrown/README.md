# smallcrown — LACE small-crown diagnosis (2026-10-15)

Written 2026-10-19 from the scripts' own docstrings and outputs. The scripts (15 Oct, 18:29–18:42) predate the
`boxfix/` and `hunt/` sessions; that session only resumed `diag_val.py` to complete `val_crowns/` (38 → 108 tiles).
Small = GT area < 622 px at 2048 (test tercile edge, `tierA/a5_strata.json`).

| file | what | result |
|---|---|---|
| `figure_small_failures.py` → `small_crown_failures_labelled.png`, `worst_small_crowns_test_gtbox.png` | test GT boxes: 4 crowns with an EMPTY LACE mask + 4 with IoU < 0.5 (SAM 3 zs ≥ 0.8); image / GT / posterior / LACE mask / SAM mask | posterior often strong in the box (peak 0.64–0.87); the r12 colour-guided render erases it |
| `diag_val.py` → `val_crowns/<tile>.json` (108 val tiles, log `../logs/smallcrown_diag_val.log` + `_resume.log`) | per GT crown on VAL: area, LACE deployed IoU, posterior max/mean in box, ellipse and box-fill IoU, SAM 3 zs per-subtile IoU (local MPS); RLE masks for small crowns | val LACE 0.748 vs SAM 0.795 mean IoU; used as the SAM val reference by `../boxfix/` |
| `fallback_rescore.py` → `preds_/scored_lace_s{0,1,2}_r2048_cgf_emptyfallback.json` (log `../logs/smallcrown_fallback.log`) | every test detection with an EMPTY deployed mask gets the box-inscribed ellipse; boxes and s′ unchanged; frozen protocol | AP50 0.6475 / 0.6446 / 0.6496 vs deployed 0.6472 / 0.6444 / 0.6493 → **+0.0002: useless** |
| `hybrid_small.py` → `preds_hybrid_small_s{0,1,2}_{ranks,ranksprime}.json` (log `../logs/smallcrown_hybrid.log`) | upper bound: SAM 3 zs masks for small boxes (w·h < 800 px), LACE masks otherwise, rankings s and s′ | **INCOMPLETE**: preds written, never scored (log ends mid-scoring at 18:54, no `scored_*` files) |

Follow-ups: `../boxfix/README.md` (render fix on GT boxes), `../hunt/README.md` (4 px LACE masker, AP50 vs SAM 3 on detections).
