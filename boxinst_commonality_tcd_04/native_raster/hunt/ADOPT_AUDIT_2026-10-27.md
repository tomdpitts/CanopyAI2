# lace4px_simple adoption — paper.tex audit (2026-10-27)

<!-- TABLE1_CURRENT:begin (auto, results_439/mark_superseded.py) -->
> ⚠️ **SUPERSEDED 439 / sparse236 NUMBERS (marked 2026-10-27).** Current headline table: [`TABLE1_CURRENT.md`](../../results_439/TABLE1_CURRENT.md). Test-set numbers in this file are a historical experiment record (512 raster, pre-seam-fix baselines, 8 px masker or earlier models) and must not be quoted unless they match that table.
<!-- TABLE1_CURRENT:end -->

This audit covers paper.tex as of HEAD `d564b8b`, which includes `967f09c` "Adopt lace4px_simple masker". It lists every place that still describes the **8 px masker** or quotes its numbers. Nothing in paper.tex was edited for this audit. Line numbers are as of 2026-10-27.

Current numbers come from `results_439/TABLE1_CURRENT.md`; 3 dp, round half up.

## Already done in 967f09c (verified by `git diff bb46316 HEAD`)

- **Tables:**
  - Table 1 (LACE row; A1 row with note e; note c FT) and tab:sparse236 (with the Box2Mask row).
  - tab:gtbox LACE row: 0.786 / 0.991 / 0.720 / 0.050.
  - tab:masker, its note, tab:scorer and tab:factors.
  - tab:strata, the touching-crown numbers, and the appendix mask-IoU strata.
- **Prose:** the L924 discussion ("4 px … four times", guided-filter gain, AP75) and L1082.

## Still describing the 8 px masker

| # | Line | Current | Like-for-like fix | Status |
|---|---|---|---|---|
| 1 | 126 abstract | "mask AP$_{50}$ of $0.647 \pm 0.003$" | $0.655 \pm 0.004$ | mechanical |
| 2 | 126 abstract | "level with the $0.648$ of Restor's…" | LACE now leads (0.655 vs 0.648, Restor single seed; Restor leads AP50:95 0.333 vs 0.280) | **needs Tom** (wording) |
| 3 | 126 abstract | "applied at four spatial offsets and interlaced into a denser feature grid" | true for the detector; the masker posterior is now evaluated on 16 offsets ({0,4,8,12}², 4 px) | **needs Tom** (depends on §3.1 framing) |
| 4 | 126 abstract | sparse "$0.678$ versus $0.636$" | unchanged: 4 px = 0.678 (0.67847) | OK |
| 5 | 375 Fig. 1 caption | "run at four $8$~px offsets and interlaced into a stride-$8$ feature grid" | detector true; masker posterior is 4 px | **needs Tom**; the graphic `figs/lace_architecture.pdf` is also 8 px |
| 6 | 502 | "The posterior lives on the $8$~px feature grid" | "$4$~px" — but the 4 px lattice is introduced nowhere in Methods (§3.1 L384 describes only the four-offset 8 px lattice) | **needs Tom** (§3.1 single-mechanism draft never written; ADOPT Phase C) |
| 7 | 502 | "radius $12$~px, i.e.\ $1.5$ feature cells" | "$3$ feature cells" | mechanical (follows #6) |
| 8 | 504 | "The mask threshold $\tau$ is lowered from an initial $0.5$ so that small crowns … are recovered" | τ = 0.33 × the maximum of the filtered posterior inside the box | mechanical wording, but see #9 |
| 9 | 504 | "chosen on the $108$ validation images … under a rule fixed in advance" | the 4 px lattice and c = 0.33 came from ~170 val arms explored after the deployed masker's test numbers were known. Only the final variant was frozen by a pre-written rule (hunt README L31) | **needs Tom** (selection disclosure) |
| 10 | 567 tab:config "Feature grid" | "$4$ offsets $\{0,8\}^2$, interlaced to $8$~px; $256\times256$ cells" | add: masker posterior at inference: 16 offsets $\{0,4,8,12\}^2$, $4$~px, $512\times512$ cells | mechanical (follows #6) |
| 11 | 585 tab:config "Inference" | "mask threshold $\tau = 0.35$" | "$\tau = 0.33\times$ the in-box maximum of the filtered posterior" | mechanical |
| 12 | 662 | "against LACE's $0.647$" | $0.655$ | mechanical |
| 13 | 662 | "are level at AP$_{50}$ ($0.647 \pm 0.003$ against $0.648$) and at box AP$_{50}$" | $0.655 \pm 0.004$; "level" no longer true at AP50 (box AP50 0.666 vs 0.665 still level) | number mechanical; **wording needs Tom** |
| 14 | 1215 tab:compute | "LACE mask rendering (test, sparse slice, ablations) & CPU, Modal" | the 4 px features (16 offsets, test/val/sparse) and the 4 px renders were local: Apple M4 Max GPU (features, pinned `venv_feat_t457`) and local CPU (render/score) | mechanical |
| 15 | 1215 tab:compute | no row for Table 1's A1 row | LACE boxes → SAM 3 zs ran locally on MPS (`tierA/a1_lace_sam3_persub.py`) | mechanical |

## Figures

| # | Figure | State |
|---|---|---|
| 16 | fig:qual (L651) `figs/qualitative_tcd439.pdf` | still 8 px (2026-10-14). A 4 px version exists: `figs/qualitative_tcd439_lace4px_simple.pdf` (2026-10-26). Swap after Tom has looked. |
| 17 | fig:qual-sparse (L194) `figs/qualitative_sparse.pdf` | still 8 px; no 4 px version generated |
| 18 | fig:architecture (L373) | graphic shows the 8 px masker (see #5) |

## Pre-existing, unrelated to the masker change (listed, not touched)

| # | Item |
|---|---|
| 19 | tab:gtbox note a: SAM 3 zs is box-prompted whole-tile (0.741). Per-subtile gives 0.789 (A4), and Table 1's new A1 row IS per-subtile, so the two SAM 3 zs placements now sit in the same paper. |
| 20 | Table 1 SelvaBox → SAM 3 (FT) is whole-tile 0.598; per-subtile 0.712–0.724 is provisional, pending the A100 re-run (tab:compute already says PENDING). |

## Counts

- **Mechanical:** 1, 7, 8, 10, 11, 12, 13 (number), 14, 15.
- **Needs Tom:** 2, 3, 5, 6, 9, 13 (wording), 16–18.
- 7 and 10 depend on 6.
