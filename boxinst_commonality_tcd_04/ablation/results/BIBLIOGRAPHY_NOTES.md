# Bibliography notes — LACE manuscript (MDPI format)

Set up 2026-09-04. Bibliography moved from the untouched MDPI dummy template to an
external BibTeX file.

## What changed

- **`lace.bib`** (new, 57 entries). Every entry checked against Crossref, arXiv, or the
  publisher page — authors, journal, year, volume, issue, pages, DOI. Journal names use
  ISSN LTWA abbreviations as MDPI requires. arXiv preprints follow the MDPI convention
  (`journal = arXiv`, `note = arXiv:NNNN.NNNNN`).
- **`paper.tex`**:
  - Deleted all three dummy `thebibliography` blocks (ACS / APA / Chicago, ~90 lines) and
    replaced them with MDPI variant A: `\bibliography{lace}`.
  - Converted 57 draft author–year prose mentions to `\cite{}`. Citation commands went
    from 14 to 84; 56 distinct keys are now cited.
  - Every replaced author-year mention is preserved verbatim as a `% Original citation
    notes:` comment on the line(s) directly above the paragraph it came from — 49 comment
    lines over 28 paragraphs. Mentions that sat together in one sentence share a comment
    line; a later mention in the same paragraph gets its own. Excluding those comments the
    file is byte-identical to the pre-annotation version.
  - Backup of the pre-change file: `paper.tex.prebib`.

## Build

Set up 2026-09-14. `Definitions/` now holds the official MDPI class (mdpi.cls dated
20 March 2024, from github.com/xiuliyouran/MDPI_Template_LaTex, mirroring the mdpi.com zip)
plus mdpi.bst, chicago2.bst, journalnames.tex and the logos. Two local edits, both needed
because we compile with tectonic (XeTeX), not pdflatex:

- Logo EPS files were converted to PDF with ghostscript and the class's
  `Definitions/logo-*.eps` references repointed to `.pdf` (XeTeX cannot embed EPS).
- `pdftex` was dropped from `\documentclass[...]` in paper.tex (it forced pdftex.def under XeTeX).
- `\hreflink{https://doi.org/}` was added to paper.tex front matter; the 2024 class has no
  default and errors at `\begin{document}` without it.

Build: `tectonic --synctex --keep-logs paper.tex` (runs bibtex itself; 32 pages, 0 undefined
citations as of 2026-09-14). VSCode: LaTeX Workshop with a tectonic recipe, build on save,
PDF tab with SyncTeX. Byproducts (.aux/.log/.bbl/.synctex.gz) are gitignored; paper.pdf is not.

## Open items

1. **`\cite{bare}` is undefined** — [paper.tex:299](paper.tex#L299), the "BARE" row of
   Table `tab:gap` (polygon-supervised, RGB, 10 cm, semantic segmentation only). No paper
   or model by that name could be found. It will render as `[?]`. Needs the actual reference.

2. **`dersch2023` is mislabelled in Table `tab:gap`** — [paper.tex:297](paper.tex#L297)
   labels the row "Mask2Former", but Dersch et al. 2023 (ISPRS Open J. Photogramm. Remote
   Sens. **8**, 100037) is Mask R-CNN and DETR, not Mask2Former. Either the label or the
   citation is wrong.

3. **Unresolved draft mentions left as plain text** (ambiguous — need the author's source):
   - `(Zhao 2023)` — [paper.tex:230](paper.tex#L230), twice; too generic to identify.
   - `Brandt 2016` — [paper.tex:184](paper.tex#L184); cited alongside Brandt 2020, which is
     resolved. 2016 could be several papers.
   - `(Reddy 2011, Raty 2020)`, `Mayamanikandan when?` — [paper.tex:186](paper.tex#L186).
   - `(Yang et al. 2022b, Fu et al. 2024)` — [paper.tex:234](paper.tex#L234).
   - `MP-PolarMask (2024 cite)` — [paper.tex:252](paper.tex#L252).
   - `Yi et al. 2023` — [paper.tex:885](paper.tex#L885).
   - The scratch line at [paper.tex:278](paper.tex#L278): FM-SAM (Que 2026), Tree-SAM
     (Huang 2026), Zeroshot (Chen 2025), TreePseCo (Vaschetti 2025), Ginkgo (Mengyuan 2025).
   - `(cite 5 models here and/or literature reviews)` — [paper.tex:190](paper.tex#L190).
   - `(cite Nature 615 p.85)` — [paper.tex:198](paper.tex#L198); that *is* `tucker2023`, but
     it is a page-specific quote citation, so left for the author to phrase.
   - `von Mises--Fisher components (cite)` — [paper.tex:396](paper.tex#L396).

4. **Uncited entries in `lace.bib`**: `boxsnake`, `sam2`. Both are named in the draft
   ([paper.tex:318](paper.tex#L318) and the Table `tab:gap` "2nd net" column) but not yet
   cited. BibTeX ignores uncited entries, so they cost nothing; wire them up if wanted.

5. **`\citep`/`\citeauthor`/`\citeyear` occurrences remain** at
   [paper.tex:180](paper.tex#L180) and lines 1149–1150 — all inside comments (MDPI template
   boilerplate), so they are inert. Nothing to convert.

## Substitutions made where the draft was imprecise

- "DeepForest (Weinstein et al., 2019, 2020)" → `weinstein2020` (Methods Ecol. Evol. **11**,
  1743–1751). The 2019 arXiv preprint was dropped as redundant.
- "NEONTreeEvaluation dataset (Weinstein et al., 2020)" at
  [paper.tex:349](paper.tex#L349) → `neontreeeval` (Weinstein et al. 2021, PLoS Comput.
  Biol. **17**, e1009180). The benchmark paper is 2021, not 2020.
- "(Jucker et al., 2022)" → `jucker2022` = Tallo (Glob. Change Biol. **28**, 5254–5268).
  Confirm this is the intended allometry reference rather than Jucker et al. 2017.
- "(Tian et al., 2021; Li et al., 2024; Cheng et al., 2023)" → `boxinst`, `box2mask`,
  `boxteacher`. Cheng et al. 2023 was read as BoxTeacher (CVPR 2023) — confirm.
- "(Radford et al., 2021; Dehghani et al., 2023; Bolya et al., 2025)" → `clip`, `vit22b`
  (Scaling ViT to 22B), `perceptionencoder`. Confirm the latter two.
- "(Zhu 2017, Zhao 2023)" → `zhu2017` only (IEEE Geosci. Remote Sens. Mag. **5**, 8–36).
- "(RSPrompter, Chen et al., 2025 … )" → `rsprompter` (IEEE TGRS **62**, 1–17, 2024). The
  draft gives 2023, 2024 and 2025 for this paper in three different places; 2024 is the
  journal version, arXiv is 2023.
