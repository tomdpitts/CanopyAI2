# Handover prompt: adopt the 4 px masker `lace4px_simple` into paper.tex (written 2026-10-23, re-pointed 2026-10-26)

<!-- TABLE1_CURRENT:begin (auto, results_439/mark_superseded.py) -->
> ⚠️ **SUPERSEDED 439 / sparse236 NUMBERS (marked 2026-10-27).** Current headline table: [`TABLE1_CURRENT.md`](../../results_439/TABLE1_CURRENT.md). Test-set numbers in this file are a historical experiment record (512 raster, pre-seam-fix baselines, 8 px masker or earlier models) and must not be quoted unless they match that table.
<!-- TABLE1_CURRENT:end -->

You are taking over a self-contained task in `/Users/tompitts/dphil/CanopyAI2`. Read `CLAUDE.md` and the memory index first; they bind you. Work in two phases: **A — compute** every number the paper needs for the adopted masker; **B — edit `paper.tex`** under Tom's editing rule below. Then **C — draft (do not apply)** one rewrite. Do not commit; Tom commits.

(2026-10-26: this prompt originally targeted candidate 1 `lace4px_g30`. Tom chose the simpler `lace4px_simple` instead — no per-pixel E-step, no size gate, ranking from its own posterior. Do not use cand1 anywhere in the paper.)

## Tom's instruction for the paper edits (verbatim — this governs Phase B)

> ok. I want you to make these changes but! very important: zero editorializing. Don't add any text or change any text beyond extremely dry like for like - simply update as minimally as possible to make it true. The only exception is presenting 3.1 as a single mechanism - don't make this change, but suggest a draft for me

## What is being adopted

- **`lace4px_simple`**, frozen spec `boxinst_commonality_tcd_04/native_raster/hunt/FROZEN_simple_2026-10-26.json`: `{"kind": "post4", "c_peak": 0.33, "rank": "own4"}`.
  - Same masker npz (`feat_cache/em_model_4p_fix.npz`) and knobs (α 0.3, κ 16) as deployed.
  - The masker's posterior is evaluated on REAL 4 px DINOv3-L24 features: 16 shifted passes {0,4,8,12}², `hunt/feat16p.py`, a 512 × 512 lattice per 2048 tile.
  - Rendering as deployed: posterior bilinear to 2048, colour-guided filter r 12 / ε 1e-4, mask clipped to the box.
  - Threshold **τ = 0.33 × the maximum of the filtered posterior inside the box** (replaces the fixed τ 0.35).
  - Ranking s′ = s · mean|2r−1| · mean(r) with r = the box's **4 px** cell posterior (the same posterior that makes the mask).
  - **Unchanged:** the detector (8 px interlaced grid, 4 passes; the 8 px grid is the even sub-lattice of the 4 px lattice) and masker *fitting* (8 px features).
- **Test result (439 tiles × 3 detector seeds):** AP50 0.6556 / 0.6503 / 0.6576 = **0.6545 ± 0.0038**; AP50:95 0.2844 / 0.2729 / 0.2837 = **0.2803 ± 0.0064**; box AP50 0.6748 / 0.6535 / 0.6708 = **0.6664 ± 0.0113**. Files: `hunt/test/{preds,scored}_lace_s{0,1,2}_r2048_lace4px_simple.json`, `hunt/test/compare_lace4px_simple.json`, log `native_raster/logs/hunt_test_simple.log`. In the paper now (deployed): 0.6470 ± 0.0025 / 0.2778 ± 0.0051 / 0.6666 ± 0.0108.
- Lab notebook `native_raster/hunt/README.md` (L1–L32; L31–L32 for this variant). Summary `native_raster/NATIVE_RASTER.md` § "MASKER vs SAM 3 ZERO-SHOT". Memory `project_hunt_4px_masker.md`.
- **Selection history** (needed for Phase B honesty): the two ingredients, the 4 px lattice and the relative threshold c 0.33, came out of a ~170-arm exploration on the 108 val tiles (det seed 0), after the deployed masker's test numbers were known. The simple variant itself was frozen on val by a rule written beforehand (README L31) and tested once.

## Hard rules (from CLAUDE.md / memory — do not skip)

- **Never lose data.**
  - No file is overwritten; the scripts already refuse, and new outputs get new names.
  - Persist RLE masks at native resolution and raw scores unrounded.
  - Copy anything off the scratchpad in the same session.
  - Document every run in `hunt/README.md` (new L-entries after L32) and in `NATIVE_RASTER.md`.
- **Long runs:** log to `native_raster/logs/<name>.log` with `python -u ... >> log 2>&1` and run them in the background. **Never pipe through `tail`.** Checkpoint and resume per tile; the existing scripts do this via per-tile files in `feat_cache/<run>/`.
- **4 px feature extraction must use the pinned venv.**
  - Use `/Users/tompitts/dphil/feat_cache/venv_feat_t457/bin/python` (transformers 4.57.1). The main `.venv` (transformers 5.12.1) gives cos 0.59 against the Modal features and is wrong.
  - Gate per tile: cos against the Modal 8 px features on the {0,8}² sub-lattice must be 1.0000 (`_gate.jsonl` in the output dir).
  - Rendering and scoring use the main `.venv`.
- **No premature findings.** A number is provisional until it covers the full set and 3 seeds and has passed its parity gates. Never report something and then retract it.
- **Local CPU/MPS only.** Modal is not needed; if you think it is, stop and ask.
- **Like-for-like LACE rows.** Every LACE row keeps the published boxes and raw detector scores; only the masks and s′ change. Use `hunt/test_eval.py`:
  - Its spec key `"rank": "own4"` recomputes s′ from the 4 px posterior, using the same-box raw scores in `preds_lace*_s{s}_r2048_cgf_r12_tau035_detscore.json`.
  - Rank gate: local 8 px s′ must match the published s′ (max |diff| 1.53e-6 on the gate tiles, tolerance 1e-5).
  - The precision control `hunt/rank_precision_control.py` showed local s′ reproduces the published AP to 4 dp.
- **Generalise, don't copy-paste.** Extend `test_eval.py` / `arms.py` (e.g. a `--split sparse`) rather than writing parallel scripts.

## Phase A — compute (each item: gate first, then full run, then record)

1. **Sparse-236 slice, 3 seeds.** This is the main new run.
   - **4 px features for the 236 sparse tiles.** Add `"sparse"` to `SPLITS` in `hunt/feat16p.py`: RGB `data/tcd_sparse/test`, 8 px reference features `feat_cache/feat_4p_sparse_dl/feat_4p_sparse`, output `feat_cache/hunt_z4p_sparse/`. Expect about 64 s per tile on MPS, roughly 4.2 h. Gate cos 1.0000 per tile.
   - **Render and rank.** Render `lace4px_simple` with `rank own4` on the published sparse detections: `native_raster/preds_lace_sparse_s{0,1,2}_r2048_cgf_r12_tau035_guidedrun.json` for boxes and `..._detscore.json` for raw s. GT is `data/tcd_sparse/sparse_gt.json`; the scorer is the same (`score_coco`, 2048, maxDets 600, floor 0, canopy as crowd). Gate: spec `deployed` reproduces the published sparse masks byte-identically on 3 tiles.
   - **Outputs:** AP50 / AP50:95 / box AP50 per seed and mean ± sd (ddof=1), under s′ and under detector score s. These feed tab:sparse236 and tab:scorer.
2. **439 test, detector-score ranking s** with `lace4px_simple` masks, 3 seeds (the tab:scorer "s" row). This is a rescore only.
3. **tab:factors (seed 0):** s, s·m, s·p̄, s·p̄·m with p̄ and m from the 4 px posterior, on the same 159,687 predictions, including the val box AP50 column as the existing table does.
   - Find the script that made the current table: grep `ablation/` and `confidence/` for 0.6338 / 0.6329. `confidence/ablation.py` is the likely source.
   - Val features are already on disk in `feat_cache/hunt_z4p_val`.
4. **tab:masker (α/κ):** re-render `lace4px_simple` at (α 0, κ 16) and (α 1.0, κ 10), 3 seeds, each ranked by s′ from its own posterior. In `test_eval`/`arms` spec, `"alpha"` / `"kappa"` (scale) set the posterior used for both the mask and the rank. The (α 0.3, κ 16) row is the adopted result itself. Units: the spec `"kappa"` is the masker's kappa SCALE — paper κ 16 ↔ `"kappa": 1.6`, κ 10 ↔ `"kappa": 1.0`; `"alpha"` is the paper α directly.
5. **tab:strata and the touching-crown numbers:**
   - tab:strata: canopy-fraction bins with mask AP50, box AP50 and the gap.
   - The "Dense canopy" paragraph's touching vs isolated numbers: mean mask IoU ± sd, failure rate ± sd, mean box side.
   - Find the generating scripts (grep for 0.5667 and 0.7078) and rerun them on the `lace4px_simple` preds.
6. **tab:gtbox LACE row:** `lace4px_simple` mask render on GT boxes, 439 tiles, 25,692 valid crowns, using the same `gtbox_lib` validity rule (code in `native_raster/audit_2026-10-14/selvabox_evidence/`). Report mean IoU and the ≥ 0.5 / 0.75 / 0.9 fractions. The 4 px test features already exist (`feat_cache/hunt_z4p_test`). `boxfix/` is a *different* GT-box render; do not use it.
7. **Appendix mask-IoU strata** (L1051–1075, `box_iou_strata.py`): rerun the LACE row. The common crown subset may change because LACE's true positives change, so report the new n and whether other rows' values moved.
8. **Discussion L924 numbers:**
   - (a) `lace4px_simple` with the guided filter replaced by bilinear upsampling (`post4` with `"r": 0` = the bilinear 4 px posterior, no filter; keep `c_peak` and `rank own4`), 3 seeds on test, to restate "adds X AP50 over bilinear (Y ± Z)".
   - (b) The AP75 3-seed mean, from the scored files.
   - (c) The same sentence makes two val claims measured on the 8 px masker: "neither a blur with no image guide nor the RGB of a different tile matches bilinear". Either rerun those two val controls with `lace4px_simple`, or list the sentence under "needs Tom". Do not keep a claim that was not measured for the adopted masker.
9. **Qualitative figures** `figs/qualitative_tcd439.pdf` and `figs/qualitative_sparse.pdf`: find their generating scripts and re-render with `lace4px_simple` masks to NEW filenames. Show Tom, then swap.

Record every result in `hunt/README.md` (new L-entries), `NATIVE_RASTER.md`, and the `box_supervised_baselines/PLAN.md` status line. Update memory `project_hunt_4px_masker.md` at the end.

## Phase B — paper.tex edits (`boxinst_commonality_tcd_04/ablation/results/paper.tex`)

Apply Tom's verbatim rule literally: replace numbers and the specific words that become false. No new sentences, no new rows, no rephrasing, no adjectives.
- **Format:** prose stays on one line per paragraph (CLAUDE.md LaTeX rule).
- **CVPR build:** saving triggers it (`cvpr/on_save.sh`). That folder is gitignored and must never be committed; leave it alone.
- **Build check:** after editing, build with `tectonic` (see memory `project_latex_build.md`): 0 errors, 0 undefined refs.
- **Line numbers** are as of 2026-10-23; re-grep before editing.

**Change (numbers / words that become false):**
- **L126 abstract:** 0.647 ± 0.003 → 0.655 ± 0.004; sparse 0.678 → new value. "applied at four spatial offsets and interlaced into a denser feature grid" is now only true for the detector, so make a minimal factual fix.
- **L375 Fig. 1 caption** ("four 8 px offsets … stride-8 feature grid"): minimal fix. Do not redraw the graphic; list it for Tom.
- **L502:**
  - "The posterior lives on the 8 px feature grid" → 4 px.
  - "radius 12 px, i.e. 1.5 feature cells" → 3 feature cells.
  - The fixed-threshold description → τ = 0.33 × the maximum of the filtered posterior in the box.
- **L504** ("Three scalars are set at inference … τ is lowered from an initial 0.5 …" and "under a rule fixed in advance"): τ is now a relative threshold chosen on val within the ~170-arm exploration. Use minimal true wording, or list it under "needs Tom" if no dry like-for-like fix exists.
- **Config table:** L560 tab:config caption; the L567 "Feature grid" row (state the 4 px masker grid only as the minimal true statement); the L585 "Inference" row (τ).
- **L627 Table 1 LACE row:** 0.647 → 0.655, 0.278 → 0.280, ±0.003 → ±0.004, ±0.005 → ±0.006; box 0.667 → 0.666 (0.6664 ± 0.0113; band ±0.011 unchanged).
- **L656:** "against LACE's 0.647" → 0.655; "(0.647 ± 0.003 against 0.648)" → 0.655 ± 0.004. Leave the word "level"; that's on Tom's list.
- **Tables and prose fed by Phase A:**
  - L701 / L707 tab:sparse236 and its note
  - L753 tab:gtbox LACE row
  - L806 tab:masker
  - L849 / L851–853 tab:scorer
  - L870 tab:factors
  - tab:strata (the L893 table) and the L908 prose
  - L888 touching numbers
  - L924 discussion: "8 px … eight times coarser", the guided-filter gain, AP75
  - L1071 appendix strata LACE row
- **L1213 tab:compute:** add 4 px feature extraction (Apple M4 Max GPU, local) to the existing local row, minimally.
- **Any other occurrence:** grep `0\.647|0\.278|0\.678|0\.6470|0\.2778|0\.6775|0\.3215` and fix whatever else carries a LACE number.

**Leave unchanged (8 px is still true there):**
- L367 overview
- L379–401 §3.1 (detector grid)
- L407, L569, L571: detector stride 8 / grid 256
- L547 NEON features
- L769–791 interlacing ablation (NEON detection)
- tab:components (L815) and the collapse appendix (masker *fitting* at 8 px)
- NEON tables
- the Restor / Detectree2 / Box2Mask / SelvaBox rows

**Do NOT do these — list them for Tom instead** (a "needs Tom" section of your report):
- **The A1 row.** Whether to add the pre-registered A1 row (LACE boxes + SAM 3 zero-shot: 0.648 / 0.301 under √(s·SAM), 0.633 / 0.287 under s) to Table 1. `box_supervised_baselines/PLAN.md` §13 says approved rows are reported whatever the result.
- **"Level with Restor."** Whether it still holds (abstract, L656) at 0.655 vs 0.648. Restor is single seed, and it leads AP50:95 at 0.333.
- **Selection disclosure** beyond the minimal truth fix at L504 / L560.
- **Two pre-existing problems, unrelated to this change; touch neither:**
  - tab:gtbox shows SAM 3 zero-shot whole-tile (0.741); per-subtile gives 0.789 (A4).
  - Table 1 SelvaBox is whole-tile 0.598; per-subtile is 0.712–0.724, provisional pending an A100 re-run.
- **The architecture figure graphic** and the cost-Pareto draft figure (not in paper.tex).

## Phase C — suggestion only (do NOT apply)

Write `ablation/results/draft_sec31_single_mechanism_2026-10-26.md`, not paper.tex: a draft of §3.1 "Dense features by phase interlacing" that presents one mechanism.
- One interlaced lattice at stride p/4: 16 passes give the 4 px lattice. The detector and masker fitting read its stride-p/2 sub-lattice (the 4-pass 8 px grid); the masker posterior is evaluated on the full 4 px lattice.
- Reuse the existing equation and registration check.
- Write in Tom's voice (memory `feedback_preserve_author_voice.md`), one line per paragraph.
- Show the diff against the current §3.1.

## Report back to Tom (briefest possible, per CLAUDE.md)

1. Table of every new number vs the old one, with file paths.
2. List of paper.tex edits (line → old → new).
3. "Needs Tom" list.
4. Path of the §3.1 draft.
5. Build status. Nothing committed.
