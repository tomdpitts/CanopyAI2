# Citation audit — LACE manuscript

Generated 2026-09-08 from `paper.tex` (1216 lines, 85 `\cite` commands, 61 distinct keys) and `lace.bib`. For every `\cite`, the source was fetched, the backing passage located, and a verdict given. **Use:** open the PDF named under each entry, go to the page/section given, and check the quoted text against the claim on the given `paper.tex` line.

Source PDFs and `pdftotext` extracts: `/Users/tompitts/dphil/refs/lace_bib_pdfs/<key>.pdf` / `.txt` (73 PDFs; CVF camera-ready copies used where available, arXiv copies kept as `<key>_arxiv.pdf`). Page numbers are given as PDF page and printed page where the two differ.

**Verdicts:** SUPPORTED = source says what is attributed to it. PARTIAL = topic matches but a number, wording or attribution is off. NOT SUPPORTED = searched, source does not say this. WRONG SOURCE = claim is true but belongs to another paper (named). INACCESSIBLE = full text unobtainable; checked from abstract.

Tally over 102 claim instances: SUPPORTED 67, PARTIAL 29, NOT SUPPORTED 3, WRONG SOURCE 3.

## 1. Summary table (ordered by paper.tex line)

| paper.tex | key | verdict | where to look |
|---|---|---|---|
| L187 | `hiernaux2023` | SUPPORTED | PDF p2 (ms lines 21–26), Abstract; PDF p4 (ms lines 99–105), Introduction, objectives para |
| L187 | `tucker2023` | SUPPORTED | PDF p1 (printed p80), Abstract; also PDF p3 (printed p82), L col, para beginning "9.9 billion trees (Fig. 1... |
| L187 | `tucker2023` | SUPPORTED | PDF p1 (printed p80), Abstract, first sentence |
| L187 | `selvabox` | PARTIAL | PDF p.2, §1 Introduction, para 2 |
| L187 | `tucker2023` | PARTIAL | PDF p6 (printed p85), Discussion, R col top (continuation of first Discussion para); counter-evidence PDF p... |
| L187 | `brandt2020` | PARTIAL | PDF p4 (printed p80), L col, para beginning "Canopies larger than 200 m2"; also PDF p2 (printed p78), Abstract |
| L187 | `crowther2015` | NOT SUPPORTED | PDF p1 (printed ~201), R col, §"Mapping tree density", first para; PDF p8 (Methods), "Model validation and ... |
| L187 | `bastin2017` | PARTIAL | PDF p10 (ms lines 192–194), final Discussion para; PDF p9 (ms lines 162–165) |
| L187 | `skole2021` | SUPPORTED | PDF p1, Abstract; PDF p18–19, §4.4 "Comparison with Prominent Medium and Coarse Resolution Biomass Datasets... |
| L187 | `crowther2015` | NOT SUPPORTED | searched whole text ("outside", "third", "10%", "non-forest", "canopy cover"); nearest: PDF p1 Abstract and... |
| L192 | `hiernaux2009` | SUPPORTED | PDF p3 (ms lines 17–24), Abstract; PDF p8 (ms lines 3–25), §"Woody plant population record methods" (Crown ... |
| L192 | `raty2020` | PARTIAL | PDF p1, Abstract (Background); PDF p3, §Study region, L col |
| L192 | `reddy2024` | PARTIAL | Abstract (Springer page; page unknown) |
| L192 | `mayamanikandan2022` | PARTIAL | Abstract (from Semantic Scholar; page unknown) |
| L192 | `jucker2018` | PARTIAL | PDF p.1 (printed p.3811), Abstract, L col, sentences 4–5; §2.2 "Field data" PDF p.3 (printed 3813) R col ("... |
| L192 | `brandt2020` | SUPPORTED | PDF p2 (printed p78), Abstract; 0.5 m: PDF p3 (printed p79), L col, para "of 600–1,000 mm yr−1) areas" |
| L192 | `tucker2023` | SUPPORTED | PDF p1 (printed p80), Abstract |
| L192 | `reiner2023` | SUPPORTED | PDF p1, Abstract; also PDF p2, §Introduction, R col, first para |
| L192 | `mugabowindekwe2023` | SUPPORTED | PDF p1 (printed p91), Abstract |
| L198 | `neontreeeval` | SUPPORTED | PDF p.14 (printed 14/18), §"Benchmark evaluation scores"/Table 3 and following para |
| L198 | `maskrcnn` | SUPPORTED | PDF p.1 (printed p.2961), Abstract, L col, para 1 |
| L198 | `weinstein2020` | SUPPORTED | PDF p.26, Appendix "Model architecture", lines 445-446; PDF p.9 §"Prebuilt model evaluation", lines 166-168 |
| L198 | `ball2023` | SUPPORTED | PDF p.1 (printed 1), Abstract |
| L198 | `tong2025` | SUPPORTED | Abstract (first page of article), sentences 3–5 |
| L198 | `stardist` | SUPPORTED | PDF p.2, §1 Introduction, para 2 (and Fig. 1 caption (c)); PDF p.2–3, §2 Method, para 1 |
| L198 | `unet` | SUPPORTED | PDF p.1, title + Abstract |
| L198 | `selvamask` | PARTIAL | PDF p.5, §4.2 (left col, para "We will denote this pipeline as 'detection-prompter → SAM'") |
| L198 | `dino` | SUPPORTED | PDF p.1, Abstract, L col, last sentences; also PDF p.2 Figure 2 caption ("Self-distillation with no labels"... |
| L198 | `sam` | SUPPORTED | PDF p.1, title and Abstract, L col, first sentence |
| L198 | `rsprompter` | PARTIAL | PDF p.1, Fig. 1 caption and Abstract; PDF p.9 (printed 9), L col, §IV.C.1 last para; Tables (p.9) rows "SAM... |
| L198 | `teng2025` | SUPPORTED | PDF p.1, Abstract, last 3 sentences; also PDF p.7 §5 para "Using SAM out-of-the-box is suboptimal..." and P... |
| L201 | `weinstein2020` | SUPPORTED | PDF p.10, §"Prebuilt model evaluation" (Results), lines 198-200 |
| L209 | `bastin2017` | PARTIAL | PDF p6 (ms lines 94–103), Introduction, 3rd para; PDF p9 (ms lines 169–171) |
| L211 | `tucker2023` | SUPPORTED | PDF p6 (printed p85), §Discussion. Quote 2: L col, first Discussion para, last 3 lines. Quote 1: R col, fir... |
| L216 | `selvabox` | SUPPORTED | PDF p.1, Abstract |
| L216 | `selvamask` | SUPPORTED | PDF p.1, Abstract |
| L216 | `oamtcd` | SUPPORTED | PDF p.1, Abstract; PDF p.5, §3.1 (para "We also consider biome distribution"); PDF p.23, Appendix Table 2 |
| L226 | `teng2025` | SUPPORTED | PDF p.1, §1 Introduction, para 1, last sentence (immediately above the NeurIPS footer); restated PDF p.2, §... |
| L226 | `boxinst` | SUPPORTED | PDF p.1, Abstract, L col, first sentence |
| L226 | `box2mask` | SUPPORTED | PDF p.2, §1 Introduction, para 2 (single column) |
| L226 | `boxteacher` | SUPPORTED | PDF p.1, Abstract, L col, first sentence |
| L226 | `dinov2` | SUPPORTED | PDF p.1 (printed p.1), Abstract |
| L226 | `dinov3` | PARTIAL | (1.7B) PDF p.8, §3.1 Data Preparation, para "Data Collection and Curation", single column, mid-page; (web v... |
| L226 | `geobench` | WRONG SOURCE | PDF p.1 (printed p.1) Abstract (benchmark definition); PDF p.8 (printed p.8) §6.2.2 Comparing Baselines on ... |
| L229 | `dinov3` | SUPPORTED | PDF p.15, §5.2 Model Distillation, paras 1–2; PDF p.9 Table 2 (Patch Size 16) and p.9 line 2; PDF p.30 Figu... |
| L229 | `centernet` | SUPPORTED | PDF p.1, Abstract, L col para 1 |
| L242 | `dinov3` | SUPPORTED | as L229: PDF p.15 §5.2 paras 1–2; p.9 Table 2; p.30 Fig. 16(a) |
| L242 | `centernet` | SUPPORTED | PDF p.1, §1 Introduction, R col para 1 |
| L249 | `emadapt` | PARTIAL | PDF p.3, §3.2 Image-level annotations, L col (Eq. 4–9, Algorithm 1) and R col para "EM-Adapt"; box variants... |
| L249 | `discobox` | SUPPORTED | PDF p.2, §1 Introduction, R col, para 2 ("Teacher model."); formalised PDF p.4, §3.2 Structured teacher, R ... |
| L249 | `ddt` | SUPPORTED | PDF p.3 (printed p.3050), §3.3 Deep Descriptor Transforming (DDT), L col para 1 and R col para 2 (after Eq.... |
| L249 | `stego` | SUPPORTED | PDF p.3, §3.1 Feature Correspondences Predict Class Co-occurrence, para 1, Eq. (1); also Abstract p.1 |
| L249 | `jepa` | SUPPORTED | PDF p.24 (printed 24), §4.4 Joint Embedding Predictive Architecture (JEPA), para after Eq. (13); repeated i... |
| L249 | `dinov3` | PARTIAL | PDF p.8, §3.1, para "Data Collection and Curation"; PDF p.9 Table 2 (ViT-7B backbone); PDF p.1 Abstract |
| L255 | `zhu2017` | PARTIAL | PDF p.18 (printed p.18), §III-C "Scene Classification" (paragraph "Using pre-trained networks", first bulle... |
| L255 | `vit` | SUPPORTED | PDF p.1, Title + Abstract, single column |
| L261 | `ball2023` | PARTIAL | PDF p.4 (printed 4), §"Model architecture and parameterization", L col para 1 continuing R col |
| L261 | `yolo` | SUPPORTED | PDF p.1 (printed p.779), Title + Abstract, L col para 1 |
| L270 | `goldblum2023` | NOT SUPPORTED | PDF p.1, Abstract, para 1 (only the "backbone = feature extractor" definition) |
| L273 | `soviany2018` | SUPPORTED | PDF p.1, Abstract (L col, sentence 2); repeated verbatim in §I Introduction, L col para 2 |
| L279 | `rcnn` | PARTIAL | PDF p.1 (printed p.580), Abstract L col + Fig. 1 caption R col; also PDF p.2 (printed p.581), §2 "Object de... |
| L279 | `yolo` | SUPPORTED | PDF p.1 (printed p.779), Abstract, L col para 1 |
| L279 | `yolov4` | SUPPORTED | PDF p.1, Title + Abstract L col, and Fig. 1 caption R col |
| L279 | `retinanet` | SUPPORTED | PDF p.1 (printed p.2980), Fig. 2 caption, R col; also Abstract L col |
| L279 | `efficientdet` | SUPPORTED | PDF p.2 (printed p.10782), §2 Related Work "One-Stage Detectors", L col para 1; also PDF p.1 §1 L col ("the... |
| L285 | `tong2025` | PARTIAL | Abstract, sentences 5–6 and highlights |
| L285 | `stardist` | SUPPORTED | PDF p.2, §1 Introduction, para 2; PDF p.4, §2 "Non-maximum suppression" para |
| L285 | `fgtreeseg` | SUPPORTED | PDF p.1, Abstract; PDF p.2, R col, §II.B "Flow-guided Instance Segmentation Framework", para 1 |
| L290 | `weinstein2020` | SUPPORTED | PDF p.1, Abstract, points 2-3 |
| L295 | `detectron2` | SUPPORTED | README para 1 and §"Model Zoo and Baselines"; MODEL_ZOO.md §"COCO Object Detection Baselines" › "Faster R-C... |
| L299 | `sam` | SUPPORTED | PDF p.2, §1 Introduction, L col, para "Model (§3)"; PDF p.5, §3 Segment Anything Model, L col, "Prompt enco... |
| L299 | `sam3` | SUPPORTED | PDF p.1, title and abstract ("We present Segment Anything Model (SAM) 3 …"); box prompts: PDF p.3, §3 Model... |
| L306 | `dinov3` | PARTIAL | PDF p.8, §3.1 para "Data Collection and Curation"; PDF p.1 Abstract |
| L306 | `clip` | SUPPORTED | PDF p.1, Abstract, L col; PDF p.3, §2.2 Creating a Sufficiently Large Dataset, R col |
| L306 | `vit22b` | SUPPORTED | PDF p.5 (printed p.5), §4.1 Training details, para "Dataset." |
| L306 | `perceptionencoder` | SUPPORTED | PDF p.1 (printed p.1), Abstract; PDF p.3 (printed p.3), §2.1 Robust Image Pretraining, para 1 and "Setup." ... |
| L306 | `dinov3` | SUPPORTED | PDF p.2, §1 Introduction, para 4 ("Addressing the problems above leads to this work, DINOv3 ..."), single c... |
| L306 | `dinov3` | SUPPORTED | PDF p.9, §3.2 Large-Scale Training with Self-Supervision, para "Learning Objective"; Eq. (1) PDF p.10 |
| L306 | `ibot` | SUPPORTED | PDF p.1 (printed p.1) Abstract; PDF p.3 (printed p.3) §3.1 Framework, Eq. (3) and following para; PDF p.4 F... |
| L313 | `selvabox` | SUPPORTED | PDF p.1 Abstract and Fig. 1 caption; PDF p.4 §3 "Locations" |
| L313 | `selvamask` | SUPPORTED | PDF p.5 §4.2 (detection-prompter para); PDF p.7 Table 4, "Urban / OAM-TCD" rows and caption; PDF p.7 §5.3 r... |
| L319 | `rsprompter` | SUPPORTED | PDF p.2, L col, §I Introduction, para 3 ("Our research is primarily centered..."); PDF p.4, §III.B para 1 a... |
| L338 | `oamtcd` | SUPPORTED | PDF p.1 Abstract (10 cm/px, instance masks); PDF p.7, §4 "Instance segmentation" para; PDF p.5 §3.1 (biome-... |
| L339 | `ball2023` | SUPPORTED | PDF p.1 Abstract (polygon masks, RGB); PDF p.3 (printed 3) Table 1 and §"Remote sensing data" (GSD); PDF p.... |
| L340 | `mask2former` | WRONG SOURCE | PDF p.1 (printed p.1290), Abstract; datasets PDF p.4 (printed 1293) §4 para 1 |
| L341 | `crownvim` | SUPPORTED | PDF p.6 (printed p.6 of 30), §3.1 "Dataset" (item (1) OAM-TCD, item (2) SSD), single column, first para; ev... |
| L344 | `weinstein2020` | PARTIAL | Detection/RGB: PDF p.1 abstract; GSD: PDF p.6 line 116 and p.12 line 223; sites: PDF p.10 lines 192-197 and... |
| L345 | `selvabox` | PARTIAL | GSD: PDF p.4 §3 para 1 and PDF p.20 Table 6; detection-only/RGB: p.1 title/abstract; biomes: PDF p.9 Table ... |
| L348 | `teng2025` | PARTIAL | Method: PDF p.5, §4.1 "Faster/Mask R-CNN+SAM and variations"; GSD: PDF p.3–4 §3 dataset paras; Biomes: PDF ... |
| L349 | `sam2lidar` | PARTIAL | Abstract, sentences 4–6 |
| L350 | `selvamask` | SUPPORTED | GSD: PDF p.2 Table 1 last row and PDF p.4 §3 (left col); VFM: p.5 §4.2; biomes: PDF p.6 §5.1 "External data... |
| L353 | `alspseudo` | PARTIAL | Supervision: PDF p.1 Abstract; Input/GSD: PDF p.3 (printed p.3) §3.1 "Data", para 1; SAM 2 role: PDF p.5 §3... |
| L354 | `fgtreeseg` | PARTIAL | Supervision: PDF p.1 Abstract + p.1 R col §II.A; 2nd net: PDF p.1 R col §II (last para) and p.2 §II.A/B; In... |
| L382 | `oamtcd` | PARTIAL | PDF p.1 Abstract; PDF p.5, §3.2 "Image and label characteristics", para 2 |
| L393 | `neontreeeval` | PARTIAL | PDF p.1 Abstract; PDF p.7 §"Image-annotated crowns" para 1; PDF p.11 §"Training annotations"; PDF p.14 para... |
| L393 | `silva2016` | SUPPORTED | PDF p.1 (printed 554) Abstract; PDF p.5 (printed 558) R col, §"Individual Tree Detection and HMAX Extractio... |
| L425 | `centernet` | SUPPORTED | PDF p.3, §3 Preliminary, R col para 1 (heatmap Ŷ ∈ [0,1]^{W/R×H/R×C}) and §4 "Objects as Points", R col |
| L425 | `cornernet` | SUPPORTED | PDF p.2 (printed p.766), §1 Introduction, para 3 (single column); also PDF p.1 Abstract |
| L618 | `selvamask` | WRONG SOURCE | Searched all of selvamask.txt for canopy/group/cat/class/crowd/OAM-TCD (OAM-TCD appears only at p.2 Table 1... |
| L652 | `selvabox` | SUPPORTED | PDF p.32, Appendix F.1, Table 19, row "OAM-TCD" |
| L653 | `box2mask` | PARTIAL | PDF p.14, §4.3.1, Table 2 "Performance comparison on COCO test-dev", last rows; text on PDF p.16 top para; ... |
| L934 | `ball2023` | PARTIAL | PDF p.4 (printed 4), §"Manual tree crown data", L col para 1 (continues from p.3) |

## 2. Highest-priority problems

- **L187 `crowther2015`** — "~1/3 of all trees are outside forests at <10% canopy cover": not in Crowther at all (forest tree density only). Also does not support the carbon-sink multi-cite. Needs a different source (trees-outside-forests inventory, e.g. Schnell et al. 2015 / FAO TOF).
- **L187 multi-cite** (carbon sink underestimated): only `skole2021` backs it cleanly; `tucker2023` largely says the opposite (11 of 14 models *over*estimate dryland carbon; only the zero-value effect supports underestimation); `brandt2020`/`bastin2017` back count/area underestimation, not carbon.
- **L198/L618 `selvamask`** — the "DINO" detector in SelvaMask/SelvaBox is the DETR-based DINO (Zhang et al. 2023), not Caron et al. self-distillation; `\cite{dino}` and the "DIstillation with NO labels" gloss are the wrong DINO. The "ignore canopy annotations" practice (L618) is SelvaBox App. F.1, not SelvaMask.
- **L226 `geobench`** — the DINOv3-web vs DINOv3-sat comparison is in the DINOv3 paper (§8.3, Table 18, p.34–35), not GEO-Bench; and DINOv3 attributes the sat model's edge to metric/physically-grounded tasks, not multispectral/multitemporal signal (both models are RGB-only).
- **L270 `goldblum2023`** — the early-layers-simple / deeper-layers-composite claim is not in Battle of the Backbones. Use Zeiler & Fergus 2014 or Yosinski et al. 2014.
- **L340 `mask2former`** — the CV paper cannot back a tree-crown MS+LiDAR row; Dersch et al. 2023 is Mask R-CNN + DETR at 5 cm (not <5), temperate. Relabel row and cite Dersch.
- **L348 `teng2025`** — tropical BCI site is evaluated separately (Table 3), so Tr should be ●, not ×.
- **L353 `alspseudo`** — SAM 2 is used only to build training pseudo-labels; inference is plain Mask R-CNN on RGB/MS. GSD is stated (5 cm). Site is boreal Finland.
- **L653 `box2mask`** — 42.5 / 35.9 are the BoxInstSeg repo checkpoint numbers (mixed test-dev/val); the paper reports 42.4 (Swin-L, test-dev) and 36.7/36.1 (R-50 test-dev/val).
- **L209 `bastin2017`** — defines VHR as ≤1 m and calls 10–30 m "high"; never mentions ≤0.1 m or says any resolution is "required".
- **L393 `neontreeeval`** — the 22-site LiDAR pretraining with "millions" of Silva-derived crowns is in `weinstein2020` (DeepForest), not the benchmark paper; move that clause's cite.
- **L192 `raty2020`** — boreal Finland NFI sampling-design simulation; nothing on drylands or tree cover.
- **L192 `jucker2018`** (added 2026-09-08) — backs plots + ALS, but Borneo rainforest carbon, not dryland tree cover; `mayamanikandan2022` is Sentinel-1 SAR + Sentinel-2, no LiDAR; `reddy2024` is a census, not plots. The rewritten sentence's "arid and semi-arid dryland" framing is backed only by `hiernaux2009` and (loosely) `reddy2024`.

## 3. Bib metadata discrepancies

- `crownvim` — author list wrong. Journal PDF p.1: Erkang Shi, Ziyang Shi, Fulin Su, Lin Li, Ruifeng Liu, Fangying Wan, Kai Zhou (7). Bib names Xiong, Zhou Guoxiong, Yan, Zhao, which do not exist. Title/journal/vol/DOI OK.
- `sam3` — author list wrong. Co-first authors are Carion, Gustafson, Hu, Debnath, Hu, Suris, Ryali, Alwala, Khedr; Lo, Mintun, Rolland, Girshick (in bib) are SAM 1 authors, not SAM 3. Title/arXiv/year OK.
- `ddt` — bib omits two authors (Yao Li, Chen-Wei Xie); IJCAI proceedings list 7. Pages 3048–3054 OK.
- `reddy2024` — publisher gives Volume 2, pp. 197–211 (2023), online 18 April 2024; bib year 2024 with vol. 2 is inconsistent.
- `fgtreeseg` — venue "IEEE Geosci. Remote Sens. Lett." not stated anywhere in the arXiv PDF; unverified. arXiv v1 was titled ZS-TreeSeg.
- `maskrcnn` — bib cites TPAMI 2020 extension; audited against ICCV 2017 camera-ready (claim content identical).
- `sam2lidar` — Crossref gives last author "Lü Liang" vs bib "Lu Liang".
- `box2mask` — arXiv prints "Xiansheng Hua" vs bib "Xian-Sheng Hua" (cosmetic).
- `weinstein2020` — audited against bioRxiv v1; MEE version behind Cloudflare. ONAQ quote verified in the preprint.

**Full text inaccessible (checked from abstract/metadata only):** `tong2025` (RSE, ScienceDirect 403), `sam2lidar` (Information Geography, ScienceDirect 403), `mayamanikandan2022` (Geocarto Int., T&F), `reddy2024` (Anthropocene Sci., Springer). Also `dersch2023` (not cited, but needed for the Mask2Former row) — ScienceDirect 403. You will need institutional access for these four.

## 4. Plain-text mentions still lacking a `\cite` (not audited)

Lines: 186 (Brandt 2016), 195 ×2, 197 (ICLR workshop paper, BalSAM), 201 ("cite 5 models"), 211 (Nature 615 p.85 = `tucker2023`, confirmed p.85), 228 ×2, 241, 244, 246 ×3, 247, 248, 255 (Zhao 2023), 260, 261 (Yang et al. 2022b, Fu et al. 2024), 278, 285 (MP-PolarMask 2024), 298, 426 (CenterNet/CornerNet — now cited at L425), 444, 619 (SELVAMASK — should be SelvaBox per §2), 934–935 (Yi et al. 2023). Uncited bib entries: `boxsnake`, `sam2`, `dersch2023`.

## 5. Full audit, per key (ordered by first appearance in paper.tex)

## hiernaux2023 — Allometric equations to estimate the dry mass of Sahel woody plants mapped with very-high resolution satellite imagery (Hiernaux 2023)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/hiernaux2023.pdf` — from https://curis.ku.dk/ws/files/412465319/Allometric_equations_to_estimate_the_dry_mass_of_Sahel_woody_plants.pdf (accepted author manuscript, line-numbered, 51 pp; page numbers are manuscript pages)
**Bib check:** OK on title and 8 authors (Hiernaux, Issoufou, Igel, Kariryaa, Kourouma, Chave, Mougin, Savadogo); journal/vol 529/120653/2023 not printed on the manuscript, consistent with DOI metadata.

### paper.tex L187
> Accurate measurement of individual tree crowns can serve as a proxy for woody biomass estimation via allometries~\cite{hiernaux2023,tucker2023} and thereby enable monitoring of carbon sequestration within areas containing trees since carbon content can be derived from woody biomass.
- **Location:** PDF p2 (ms lines 21–26), Abstract; PDF p4 (ms lines 99–105), Introduction, objectives para
- **Quote:** "Each woody plant is precisely georeferenced and defined by its crown area and, sometimes, its height. The challenge is to build allometric equations for foliage, wood and root dry masses based either on crown area alone, or the product crown area x tree height as independent variables regardless of species."
- **Verdict:** SUPPORTED

## tucker2023 — Sub-continental-scale carbon stocks of individual trees in African drylands (Tucker 2023)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/tucker2023.pdf` — from https://www.nature.com/articles/s41586-022-05653-6.pdf; journal version (PDF p1 = printed p80; p3 = 82; p5 = 84; p6 = 85; p7 = 86)
**Bib check:** OK (Nature 615, 80–86, 2 March 2023; 30 authors match; issue 7950 not printed on PDF)

### paper.tex L187
> Accurate measurement of individual tree crowns can serve as a proxy for woody biomass estimation via allometries~\cite{hiernaux2023,tucker2023} and thereby enable monitoring of carbon sequestration within areas containing trees since carbon content can be derived from woody biomass.
- **Location:** PDF p1 (printed p80), Abstract; also PDF p3 (printed p82), L col, para beginning "9.9 billion trees (Fig. 1 and Methods)"
- **Quote:** "We take this step to assess the woody carbon pool by adding up tree-by-tree values, calculated using allometric equations to predict foliage, wood and root dry masses from crown area multiplied by the average carbon concentration (0.47)."
- **Verdict:** SUPPORTED

### paper.tex L187
> Measurement techniques in this domain have traditionally focused on dense canopy areas and forests, but there is an increasing body of literature that highlights the limitations of this approach as distribution, density, cover, and carbon content of trees in sparse-canopy biomes such as savannah and dryland are not well understood at sub-continental to continental scales~\cite{tucker2023}, and in the Global South as a whole \cite{selvabox}.
- **Location:** PDF p1 (printed p80), Abstract, first sentence
- **Quote:** "The distribution of dryland trees and their density, cover, size, mass and carbon content are not well known at sub-continental to continental scales1–14."
- **Verdict:** SUPPORTED

### paper.tex L187
> Recent studies highlight that the global contribution of sparse-canopy biomes as a carbon sink is underestimated due to the underestimation of tree mass in sparse-canopy areas~\cite{tucker2023,brandt2020,crowther2015,bastin2017,skole2021}.
- **Location:** PDF p6 (printed p85), Discussion, R col top (continuation of first Discussion para); counter-evidence PDF p5 (printed p84), R col, para "herbaceous, wood, foliage and root carbon..."
- **Quote:** "Consequently, areas with scattered trees are often represented by zero values (Fig. 6), whereas the carbon density of larger groups of trees may be overestimated in previous assessments, as these areas are wrongly considered as dense forests."
- **Verdict:** PARTIAL
- **Note:** Tucker's headline finding is the opposite direction: "whereas previous studies assumed that ecosystem models underestimated dryland carbon stocks, our results show overall higher values from the model outputs" (p84) and "the density and carbon stocks of scattered trees have been underestimated by three models and overestimated by 11 models" (Abstract). Only the local zero-value effect for scattered trees supports "underestimation".

### paper.tex L192
> Machine learning and deep learning applied to satellite and UAV data have since shifted the field from aggregate cover fractions toward mapping every individual tree: Brandt et al.~\cite{brandt2020} mapped over 1.8 billion crowns across 1.3 million $\mathrm{km}^2$ of the Sahara and Sahel from 0.5 m imagery; Tucker et al.~\cite{tucker2023} extended this to carbon over 9.9 billion trees.
- **Location:** PDF p1 (printed p80), Abstract
- **Quote:** "We assessed more than 9.9 billion trees derived from more than 300,000 satellite images, covering semi-arid sub-Saharan Africa north of the Equator. We attributed wood, foliage and root carbon to every tree in the 0–1,000 mm year−1 rainfall zone"
- **Verdict:** SUPPORTED

### paper.tex L211
> In a study of 9.9 billion individual tree crowns in the Sahara/Sahel/Sudan region, Tucker et al.~\cite{tucker2023} found that ``areas with scattered trees are often represented by zero values'' (cite Nature 615 p.85), attributing this to ``the fact that previous models are rarely developed, trained and validated with plots of very sparse tree cover''.
- **Location:** PDF p6 (printed p85), §Discussion. Quote 2: L col, first Discussion para, last 3 lines. Quote 1: R col, first text lines below the Fig. 6 caption (same sentence continues across the column break).
- **Quote:** "The explanation for this apparent paradox—higher tree cover but less carbon—is related to the fact that previous models are rarely developed, trained and validated with plots of very sparse tree cover, thus leaving high uncertainty for drylands with scattered trees. Consequently, areas with scattered trees are often represented by zero values (Fig. 6)"
- **Verdict:** SUPPORTED
- **Note:** Page 85 confirmed. Both quotes verbatim. Source describes the region as "semi-arid sub-Saharan Africa north of the Equator" / 0–1,000 mm rainfall zone (Sahara, Sahel, Sudanian zone in Fig. 1), so "Sahara/Sahel/Sudan" is fine.

## selvabox — SelvaBox: A high-resolution dataset for tropical tree crown detection (Baudchon 2026)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/selvabox.pdf` — from https://arxiv.org/pdf/2507.00170; arXiv v2 (27 Feb 2026), header "Published as a conference paper at ICLR 2026"; page numbers are arXiv pages (printed = PDF page)
**Bib check:** OK (title, 8 authors, Baudchon first, ICLR 2026, 2026)

### paper.tex L187
> ... distribution, density, cover, and carbon content of trees in sparse-canopy biomes such as savannah and dryland are not well understood at sub-continental to continental scales~\cite{tucker2023}, and in the Global South as a whole \cite{selvabox}.
- **Location:** PDF p.2, §1 Introduction, para 2
- **Quote:** "Most open access high-quality, high-resolution tree detection RGB datasets represent temperate forests of the global North (Tab. 1). Tropical forests remain severely underrepresented and have relatively modest annotation counts"
- **Verdict:** PARTIAL
- **Note:** Source says tropical-forest *datasets* are under-represented vs the global North; it says nothing about savannah/dryland or tree distribution/carbon being poorly understood in the Global South "as a whole".

### paper.tex L216
> Recent annotated datasets such as SelvaBox~\cite{selvabox} \& SelvaMask~\cite{selvamask} have broadened geographic coverage ...
- **Location:** PDF p.1, Abstract
- **Quote:** "We introduce SELVABOX, the largest open-access dataset for tropical tree crown detection in high-resolution drone imagery. It spans three countries and contains more than 83 000 manually labeled crowns"
- **Verdict:** SUPPORTED

### paper.tex L313
> SELVABOX~\cite{selvabox} is a UAV-captured RGB dataset of tropical dense-canopy forest in Panama, Brazil, and Ecuador, composed of 83,000 ITC manual bounding box annotations.
- **Location:** PDF p.1 Abstract and Fig. 1 caption; PDF p.4 §3 "Locations"
- **Quote:** "It spans three countries and contains more than 83 000 manually labeled crowns" / "The RGB imagery was acquired in three countries: Brazil, Ecuador, and Panama"
- **Verdict:** SUPPORTED

### paper.tex L345
> SelvaBox \cite{selvabox} & -- & RGB & -- & 1--3 & \cmark & \cmark & \xmark \\
- **Location:** GSD: PDF p.4 §3 para 1 and PDF p.20 Table 6; detection-only/RGB: p.1 title/abstract; biomes: PDF p.9 Table 4 (SelvaBox, Detectree2, BCI50ha) and Table 5 (NeonTreeEvaluation, QuebecTrees, OAM-TCD)
- **Quote:** "(GSD) between 1.2–5.1 cm per pixel (Tab. 6 in App. A.1)" / "Table 5: Non-tropical datasets evaluation ... temperate (NeonTreeEvaluation and QuebecTrees) and urban (OAM-TCD) datasets"
- **Verdict:** PARTIAL
- **Note:** GSD is 1.2–5.1 cm (Table 6 per-raster values 1.2–5.1), not 1–3. Sup/Input/2nd-net/Tr●/Te●/Sv× all verified (tropical and temperate datasets reported separately in Tables 4 and 5; no savannah dataset).

### paper.tex L652
> The image count is the detector's, taken from SelvaBox~\cite{selvabox} Table~19: $3024$ OAM-TCD $2048^2$ training tiles.
- **Location:** PDF p.32, Appendix F.1, Table 19, row "OAM-TCD"
- **Quote:** "OAM-TCD 10 3024 2048 [666, 2666] [1024, 1777] [66.6, 204.8] [3.8, 20] 2527 1024 102.4" (columns: GSD, # Train Images, Train size (px), ..., # Test Images, Test size, Test extent)
- **Verdict:** SUPPORTED
- **Note:** Same footnote's "maxDets 400 per subtile" is only PARTIALLY backed: PDF p.6 §4 "Evaluation metrics" says "we increase the maxDets parameter of COCOEval from 100 to 400 for those datasets" naming SelvaBox, QuebecTrees and BCI50ha only — OAM-TCD is not listed. The 0.577 / 0.205 / 0.748 figures are not in SelvaBox (its OAM-TCD numbers are mAP50:95 44.29±0.33 / mAR 55.57, Table 5 p.9); they read as the manuscript's own re-run.

## brandt2020 — An unexpectedly large count of trees in the West African Sahara and Sahel (Brandt 2020)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/brandt2020.pdf` — from https://hal.science/hal-02997632/file/s41586-020-2824-5.pdf (publisher PDF, CC BY, HAL cover sheet as PDF p1); journal version (PDF p2 = printed p78; p3 = 79; p4 = 80; p6 = 82)
**Bib check:** OK (Nature 587(7832), 78–82, 5 Nov 2020; 24 authors match)

### paper.tex L187
> Recent studies highlight that the global contribution of sparse-canopy biomes as a carbon sink is underestimated due to the underestimation of tree mass in sparse-canopy areas~\cite{tucker2023,brandt2020,crowther2015,bastin2017,skole2021}.
- **Location:** PDF p4 (printed p80), L col, para beginning "Canopies larger than 200 m2"; also PDF p2 (printed p78), Abstract
- **Quote:** "accounting only for large patches of vegetation in canopy-cover assessments severely underestimates the total density and canopy cover." / "trees outside of forests are not well-documented3."
- **Verdict:** PARTIAL
- **Note:** Brandt shows tree count/cover is underestimated by canopy-cover products; it does not quantify carbon or claim the carbon sink is underestimated (carbon storage mentioned only as an ecosystem service in the Abstract).

### paper.tex L192
> Machine learning and deep learning applied to satellite and UAV data have since shifted the field from aggregate cover fractions toward mapping every individual tree: Brandt et al.~\cite{brandt2020} mapped over 1.8 billion crowns across 1.3 million $\mathrm{km}^2$ of the Sahara and Sahel from 0.5 m imagery; Tucker et al.~\cite{tucker2023} extended this to carbon over 9.9 billion trees.
- **Location:** PDF p2 (printed p78), Abstract; 0.5 m: PDF p3 (printed p79), L col, para "of 600–1,000 mm yr−1) areas"
- **Quote:** "Here we map the crown size of each tree more than 3 m2 in size over a land area that spans 1.3 million km2 in the West African Sahara, Sahel and sub-humid zone, using submetre-resolution satellite imagery and deep learning4. We detected over 1.8 billion individual trees" / "satellite data at very high spatial resolution (0.5 m) from DigitalGlobe satellites"
- **Verdict:** SUPPORTED

## crowther2015 — Mapping tree density at a global scale (Crowther 2015)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/crowther2015.pdf` — from https://cdanfort.w3.uvm.edu/csc-reading-group/crowther-nature-2015.pdf (publisher advance-online-publication PDF, "VOL 000" placeholders; PDF p1–5 = printed 201–205 approximately, p6+ = Methods/Extended Data; Nature refuses non-subscriber download)
**Bib check:** OK (Nature 525, 201–205, 2015, doi:10.1038/nature14967; 38 authors match)

### paper.tex L187
> Recent studies highlight that the global contribution of sparse-canopy biomes as a carbon sink is underestimated due to the underestimation of tree mass in sparse-canopy areas~\cite{tucker2023,brandt2020,crowther2015,bastin2017,skole2021}.
- **Location:** PDF p1 (printed ~201), R col, §"Mapping tree density", first para; PDF p8 (Methods), "Model validation and testing" preceding para
- **Quote:** "Forested areas are found in most of Earth’s biomes, even those as counterintuitive as desert, tundra, and grassland" / "tree density estimates were only collected in forested ecosystems and non-forested regions are under-represented"
- **Verdict:** NOT SUPPORTED
- **Note:** Crowther maps *forest* tree density (3.04 trillion trees) scaled by forested area only; it contains no carbon, biomass or sink estimate and no statement that sparse-canopy carbon is underestimated. It explicitly excludes non-forested sites ("we ensured that we did not overestimate tree densities in non-forested sites").

### paper.tex L187
> Moreover, approximately 1/3 of all trees globally are outside forests, i.e., individual \& sparsely distributed at $<10\%$ tree canopy cover~\cite{crowther2015}.
- **Location:** searched whole text ("outside", "third", "10%", "non-forest", "canopy cover"); nearest: PDF p1 Abstract and p3 (printed ~203) L col para "At the biome-level"
- **Quote:** "This map reveals that the global number of trees is approximately 3.04 trillion ... approximately 1.39 trillion exist in tropical and subtropical forests, with 0.74 trillion in boreal regions and 0.61 trillion in temperate regions." / "A total of 42.8% of the planet’s trees exist in tropical and subtropical regions, with another 24.2% and 21.8% in boreal and temperate biomes"
- **Verdict:** NOT SUPPORTED
- **Note:** Crowther gives no "outside forests" fraction and no <10% canopy-cover statistic; its counts are forest trees only (trees ≥10 cm DBH, scaled by forested land area per pixel). The 1/3 figure is not in this paper and needs a different source (a trees-outside-forests inventory study, e.g. Schnell et al. 2015 Environ. Monit. Assess., or an FAO TOF assessment) — verify before re-citing.

## bastin2017 — The extent of forest in dryland biomes (Bastin 2017)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/bastin2017.pdf` — from https://core.ac.uk/download/83940896.pdf (White Rose eprints 115833, accepted author manuscript, line-numbered; PDF pages are manuscript pages, no journal pagination)
**Bib check:** OK (Science 356(6338), 635–638, 2017; cover sheet: Bastin, Berrahmouni, Grainger + 28 more = 31 authors)

### paper.tex L187
> Recent studies highlight that the global contribution of sparse-canopy biomes as a carbon sink is underestimated due to the underestimation of tree mass in sparse-canopy areas~\cite{tucker2023,brandt2020,crowther2015,bastin2017,skole2021}.
- **Location:** PDF p10 (ms lines 192–194), final Discussion para; PDF p9 (ms lines 162–165)
- **Quote:** "Using numbers on the carbon pools of woody savannas (28), further research could use our publicly available data to increase estimates of global forest carbon stocks by 15 to 158.3 GtC, or by 2 to 20 % (29)" / "the latter maps were missing significant areas of tree cover and forest in dryland biomes"
- **Verdict:** PARTIAL
- **Note:** Bastin shows dryland forest *area* is underestimated by coarser-resolution maps (by 40–47%); the carbon-stock increase is a speculative extrapolation ("could"), and nothing is said about tree mass or sink strength.

### paper.tex L209
> High ($<1$~m/px) or ultra-high ($\leq 0.1$~m/px) resolution is required~\cite{bastin2017}.
- **Location:** PDF p6 (ms lines 94–103), Introduction, 3rd para; PDF p9 (ms lines 169–171)
- **Quote:** "Mapping forests in the drylands using satellite data is challenging, even with high spatial resolution imagery (10-30 m). ... Very High spatial Resolution (VHR) images (with a pixel width ≤1 m). VHR images allow scientists to visually identify individual tree crowns in dry areas" / "illustrates the limitations of using medium-to-high resolution satellite images to identify low tree cover"
- **Verdict:** PARTIAL
- **Note:** Bastin used ≤1 m ("very high resolution") Google Earth imagery for photo-interpretation of 210,000 0.5-ha plots and argues 10–30 m imagery misses low tree cover. It calls 10–30 m "high" resolution, defines VHR as ≤1 m, and never mentions an "ultra-high ≤0.1 m/px" class or that such resolution is "required".

## skole2021 — The contribution of trees outside of forests to landscape carbon and climate change mitigation in West Africa (Skole 2021)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/skole2021.pdf` — from https://mdpi-res.com/d_attachment/forests/forests-12-01652/article_deploy/forests-12-01652.pdf; journal version (PDF p = printed "p of 25")
**Bib check:** OK (Forests 2021, 12(12), 1652; 4 authors match)

### paper.tex L187
> Recent studies highlight that the global contribution of sparse-canopy biomes as a carbon sink is underestimated due to the underestimation of tree mass in sparse-canopy areas~\cite{tucker2023,brandt2020,crowther2015,bastin2017,skole2021}.
- **Location:** PDF p1, Abstract; PDF p18–19, §4.4 "Comparison with Prominent Medium and Coarse Resolution Biomass Datasets"; PDF p20, para "We did not test other similar coarse resolution data"
- **Quote:** "Prominent biomass and carbon maps from global-scale remote sensing greatly underestimate the “invisible” carbon in these sparse tree-based systems." / "The coarse resolution mapping underestimated our total carbon in the study area ... Our estimate of 44.28 × 103 MgC was 30% higher than their estimate of 33.6 × 103 MgC." / "coarse resolution data omit the “invisible” TOF cover and associated carbon stocks in landscapes where TOFs are significant such as these African drylands"
- **Verdict:** SUPPORTED
- **Note:** Evidence is a Senegal (Fatick) case study extrapolated to TOF landscapes generally (p22: "overlooked in favor of closed forest systems"); best single backing of the five citations.

## hiernaux2009 — Woody plant population dynamics in response to climate changes from 1984 to 2006 in Sahel (Gourma, Mali) (Hiernaux 2009)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/hiernaux2009.pdf` — from https://core.ac.uk/download/52742718.pdf (HAL-IRD ird-00406590 accepted manuscript, line-numbered, 38 pp; page numbers are manuscript pages, no journal pagination)
**Bib check:** OK on title/authors (6)/journal/year/DOI; vol. 375, pp. 103–113 not printed on the manuscript (HAL cover cites in-press "HYDROL 16508"), consistent with Crossref.

### paper.tex L192
> Several conventional methodologies have been applied to the measurement of tree cover in arid and semi-arid dryland areas, including local field surveys~\cite{hiernaux2009}, area sampling~\cite{raty2020}, and hybrid approaches combining field plots and Remote Sensing (RS) techniques using optical, multispectral, and LiDAR~\cite{reddy2024,mayamanikandan2022,jucker2018}.
- **Location:** PDF p3 (ms lines 17–24), Abstract; PDF p8 (ms lines 3–25), §"Woody plant population record methods" (Crown Linear Intercept, Circular Plots Census)
- **Quote:** "documented and discussed for 24 rangeland sites monitored from 1984 to 2006 in Gourma (Mali). ... Three different methods contributed to assess and monitor woody plant density and canopy cover." / "woody plants which crown overcast the transect line are recorded with the exact position and length of the crown intercept"
- **Verdict:** SUPPORTED
- **Re-check 2026-09-08 (rewritten sentence):** unchanged, SUPPORTED.

## raty2020 — Comparison of the local pivotal method and systematic sampling for national forest inventories (Räty 2020)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/raty2020.pdf` — from https://www.sciopen.com/local/article_pdf/10.1186/s40663-020-00266-9.pdf (Springer/MDPI blocked curl); journal version (article 54, PDF p = printed p)
**Bib check:** OK (For. Ecosyst. 7:54, 2020; 6 authors match)

### paper.tex L192
> Several conventional methodologies have been applied to the measurement of tree cover in arid and semi-arid dryland areas, including local field surveys~\cite{hiernaux2009}, area sampling~\cite{raty2020}, and hybrid approaches combining field plots and Remote Sensing (RS) techniques using optical, multispectral, and LiDAR~\cite{reddy2024,mayamanikandan2022,jucker2018}.
- **Location:** PDF p1, Abstract (Background); PDF p3, §Study region, L col
- **Quote:** "In this simulation study we compared all these sampling methods to systematic sampling. The LPM samples were selected solely using the coordinates (LPMxy) or, in addition to that, auxiliary remote sensing-based forest variables" / "The study region covers a 60,000 km2 land area in Southern Finland"
- **Verdict:** PARTIAL
- **Note:** A simulation study of NFI plot-sampling designs (systematic vs local pivotal) in boreal Finland; target variables are growing-stock volume, basal area etc., not tree cover, and nothing on arid/semi-arid drylands. Supports "area/systematic sampling exists as an inventory method" only.
- **Re-check 2026-09-08 (rewritten sentence):** unchanged, PARTIAL. Sentence still frames it as a dryland tree-cover method; source is a boreal Finland NFI sampling-design simulation.

## reddy2024 — Assessment of tree density, tree cover, species diversity and biomass in semi-arid human dominated landscape using large area inventory and remote sensing data (Reddy 2024)
**PDF:** INACCESSIBLE — Springer closed (no OA copy via S2/CORE/Unpaywall). Used Springer landing-page abstract (https://link.springer.com/article/10.1007/s44177-024-00066-8).
**Bib check:** Discrepancy: publisher lists "Volume 2, pages 197–211 (2023)", published online 18 April 2024; bib year 2024 with vol. 2 — use 2023 for the volume year or keep 2024 as online date (check house style). Authors (2), title, journal, issue 3–4, pages OK.

### paper.tex L192
> Several conventional methodologies have been applied to the measurement of tree cover in arid and semi-arid dryland areas, including local field surveys~\cite{hiernaux2009}, area sampling~\cite{raty2020}, and hybrid approaches combining field plots and Remote Sensing (RS) techniques using optical, multispectral, and LiDAR~\cite{reddy2024,mayamanikandan2022,jucker2018}.
- **Location:** Abstract (Springer page; page unknown)
- **Quote:** "This work is the first of its kind and attempts to estimate tree density, tree cover, species diversity, and biomass from a comprehensive survey and very high-resolution remote-sensing data. This research compared the census of the entire tree population over a 900-ha site (local landscape) and a 15,142-ha site (regional landscape)"
- **Verdict:** PARTIAL
- **Note:** Semi-arid, field + VHR RS hybrid confirmed; but the field component is a complete census ("Since no sampling is involved"), not field plots. Abstract only.
- **Re-check 2026-09-08 (rewritten sentence):** still PARTIAL. Semi-arid ✓, field + very-high-resolution optical RS ✓; but the field component is a complete census ("no sampling is involved"), not plots, and no multispectral/LiDAR beyond VHR optical is stated in the abstract.

## mayamanikandan2022 — Quantifying the influence of plot-level uncertainty in AGB upscaling using remote sensing data in central Indian dry deciduous forest (Mayamanikandan 2022)
**PDF:** INACCESSIBLE — T&F closed; no OA copy on Semantic Scholar/CORE/Unpaywall; tandfonline returns 403. Used Semantic Scholar abstract + Crossref metadata.
**Bib check:** OK (Geocarto Int. 37(12), 3489–3503; online 29 Dec 2020, issue 2022; 9 authors match Crossref)

### paper.tex L192
> Several conventional methodologies have been applied to the measurement of tree cover in arid and semi-arid dryland areas, including local field surveys~\cite{hiernaux2009}, area sampling~\cite{raty2020}, and hybrid approaches combining field plots and Remote Sensing (RS) techniques using optical, multispectral, and LiDAR~\cite{reddy2024,mayamanikandan2022,jucker2018}.
- **Location:** Abstract (from Semantic Scholar; page unknown)
- **Quote:** "Detailed tree measurements and location mapping are performed in 13 (1 ha) plots and 1 a very large permanent plot of 32 ha and AGB is estimated using local volume equations. Remote sensing-based AGB estimated using a multiple linear regression model between the reflectance (Sentinel-2) and backscatter (Sentinel-1) with field AGB."
- **Verdict:** PARTIAL
- **Note:** Hybrid field-plot + RS approach confirmed, but the target is above-ground biomass in tropical dry deciduous forest (central India), not tree cover in arid/semi-arid drylands. Abstract only.
- **Re-check 2026-09-08 (rewritten sentence):** still PARTIAL. Field plots (13×1 ha + 32 ha) + Sentinel-2 optical/multispectral ✓; the second sensor is Sentinel-1 SAR, not LiDAR; site is tropical dry deciduous forest (central India), not arid/semi-arid dryland; target is AGB, not tree cover.

## jucker2018 — Estimating aboveground carbon density and its uncertainty in Borneo's structurally complex tropical forests using airborne laser scanning (Jucker 2018)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/jucker2018.pdf` — from https://bg.copernicus.org/articles/15/3811/2018/bg-15-3811-2018.pdf; journal version, CC BY (PDF p.N = printed p.3810+N)
**Bib check:** OK — entry added 2026-09-08 from Crossref/PDF p.1: Biogeosciences 15(12), 3811–3830, 22 June 2018, 26 authors, Jucker first, Coomes last. **Assumption:** the key `jucker2018` was undefined in lace.bib; this paper was chosen because the sentence pairs field plots with LiDAR. If the intended reference was Jucker et al. 2017 (Glob. Change Biol. 23, 177–190, "Allometric equations for integrating remote sensing imagery into forest monitoring programmes") or Jucker et al. 2022 (Tallo), replace the entry.

### paper.tex L192
> Several conventional methodologies have been applied to the measurement of tree cover in arid and semi-arid dryland areas, including local field surveys~\cite{hiernaux2009}, area sampling~\cite{raty2020}, and hybrid approaches combining field plots and Remote Sensing (RS) techniques using optical, multispectral, and LiDAR~\cite{reddy2024,mayamanikandan2022,jucker2018}.
- **Location:** PDF p.1 (printed p.3811), Abstract, L col, sentences 4–5; §2.2 "Field data" PDF p.3 (printed 3813) R col ("Across the five study sites we compiled a total of 173 plots"); §2.3 ALS metrics PDF p.4 (printed 3814)
- **Quote:** "By combining ALS imagery with data from 173 permanent forest plots spanning the lowland rainforests of Sabah on the island of Borneo, we develop a simple yet general model for estimating forest carbon stocks using ALS-derived canopy height and canopy cover as input metrics."
- **Verdict:** PARTIAL
- **Note:** Cleanly backs "hybrid field plots + LiDAR (ALS)". Does not back the sentence's framing: the study is lowland tropical rainforest in Borneo, not arid/semi-arid dryland, and its target is aboveground carbon density, not tree cover. Either cite it for the method class only (reword "including in other biomes") or swap for a dryland plots+LiDAR study.

## reiner2023 — More than one quarter of Africa's tree cover is found outside areas previously classified as forest (Reiner 2023)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/reiner2023.pdf` — from https://www.nature.com/articles/s41467-023-37880-4.pdf; journal version (article 2258, PDF p = printed p)
**Bib check:** OK (Nat. Commun. 14:2258, 2023; 24 authors match)

### paper.tex L192
> Reiner et al.~\cite{reiner2023} found that 29\% of Africa's tree cover lies outside previously classified forest.
- **Location:** PDF p1, Abstract; also PDF p2, §Introduction, R col, first para
- **Quote:** "reveals that 29% of tree cover is found outside areas previously classified as tree cover in state-of-the-art maps, such as in croplands and grassland." / "at the continental scale, 29% of all tree cover is found outside areas classified as forest in a current state-of-the-art map based on Sentinel-2 10 m"
- **Verdict:** SUPPORTED

## mugabowindekwe2023 — Nation-wide mapping of tree-level aboveground carbon stocks in Rwanda (Mugabowindekwe 2023)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/mugabowindekwe2023.pdf` — from https://www.nature.com/articles/s41558-022-01544-w.pdf; journal version (PDF p1 = printed p91)
**Bib check:** OK (Nat. Clim. Change 13, Jan 2023, 91–97; 21 authors match; issue "1" not printed but January issue)

### paper.tex L192
> Mugabowindekwe et al.~\cite{mugabowindekwe2023} produced nation-wide tree-level carbon estimates for Rwanda.
- **Location:** PDF p1 (printed p91), Abstract
- **Quote:** "Here, we propose an approach to map the carbon stock of each individual overstory tree at the national scale of Rwanda using aerial imagery from 2008 and deep learning."
- **Verdict:** SUPPORTED

## weinstein2020 — DeepForest: A Python package for RGB deep learning tree crown delineation (Weinstein 2020)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/weinstein2020.pdf` — from https://www.biorxiv.org/content/10.1101/2020.07.07.191551v1.full.pdf; bioRxiv v1 (8 Jul 2020); page numbers are bioRxiv pages (printed = PDF page). Wiley pdfdirect/pdf/am-pdf all returned a Cloudflare challenge page (bronze OA only); MEE journal PDF not obtained.
**Bib check:** OK on title, 6 authors, Weinstein first, 2020. Journal volume/pages (MEE 11(12):1743–1751) not verifiable from preprint; Crossref confirms DOI 10.1111/2041-210X.13472 published 2020-10-16.

### paper.tex L198
> ... DeepForest~\cite{weinstein2020} established a RetinaNet-based baseline evaluated on the NeonTreeEvaluation benchmark~\cite{neontreeeval} ...
- **Location:** PDF p.26, Appendix "Model architecture", lines 445-446; PDF p.9 §"Prebuilt model evaluation", lines 166-168
- **Quote:** "Deepforest uses the keras-retinanet implementation (Gaiser et al 2017) of the Retinanet one-stage object detector with a Resnet-50 classification backbone" / "This dataset consists of 212 images containing 5852 trees from 22 sites that is part of an upcoming tree crown benchmark data package (Weinstein et al. 2020)."
- **Verdict:** SUPPORTED

### paper.tex L201
> In the DeepForest paper, Weinstein et al. highlight Onaqui, Utah (ONAQ) as the model's worst-performing site~\cite{weinstein2020} since it is a ``desert scrub site with a different vegetation structure from any of the training data''.
- **Location:** PDF p.10, §"Prebuilt model evaluation" (Results), lines 198-200
- **Quote:** "The site with the worst performance is Onaqui, Utah (ONAQ), which is a desert scrub site with a different vegetation structure from any of the training data. The site is almost treeless and includes trees with short and gnarled stature."
- **Verdict:** SUPPORTED
- **Note:** Quote is verbatim in the DeepForest preprint (correct source, not the benchmark paper). Verified against bioRxiv v1; MEE version not checked.

### paper.tex L290
> DeepForest~\cite{weinstein2020} is a significant baseline model in Tree Crown Detection (TCD).
- **Location:** PDF p.1, Abstract, points 2-3
- **Quote:** "DeepForest overcomes this limitation by including a model pre-trained on over 30 million algorithmically generated crowns from 22 forests and fine-tuned using 10,000 hand-labeled crowns from 6 forests."
- **Verdict:** SUPPORTED

### paper.tex L344
> DeepForest \cite{weinstein2020} & -- & RGB & -- & 10 & \xmark & \cmark & \xmark \\
- **Location:** Detection/RGB: PDF p.1 abstract; GSD: PDF p.6 line 116 and p.12 line 223; sites: PDF p.10 lines 192-197 and PDF p.25 Supplementary 1; tropical: PDF p.12 §"French Guiana Tropical Forest"
- **Quote:** "prebuilt model was trained on square crops of length 400px (40m at 0.1m resolution)" / "make reasonable predictions in forests ranging from deciduous forests of the Northeast, to southern pinelands, to coniferous forests of the mountain west" / "Figure 6. Predictions made on a tropical forest in French Guiana using the prebuilt model"
- **Verdict:** PARTIAL
- **Note:** Sup/RGB/no 2nd net/10 cm/Te● verified. Tr× is arguable: the paper includes a French Guiana tropical-forest case study (p.12-13, qualitative plus a retrained model) — arguably ○ or ●. Sv×: NEON sites include desert scrub (ONAQ, SRER) and prairie (KONZ) but no savannah as such.

## neontreeeval — A benchmark dataset for canopy crown detection and delineation ... NEON (Weinstein 2021)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/neontreeeval.pdf` — from https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1009180&type=printable; journal version (footer "N / 18")
**Bib check:** OK (title, 8 authors, Weinstein first, PLoS Comput Biol 17(7): e1009180, 2021)

### paper.tex L198
> Modern deep learning approaches focus on instance segmentation on RGB imagery: DeepForest~\cite{weinstein2020} established a RetinaNet-based baseline evaluated on the NeonTreeEvaluation benchmark~\cite{neontreeeval}, while Detectree2~\cite{ball2023} applied Mask R-CNN~\cite{maskrcnn} to tropical crown polygons.
- **Location:** PDF p.14 (printed 14/18), §"Benchmark evaluation scores"/Table 3 and following para
- **Quote:** "we used the DeepForest Python package to generate crown detections in the benchmark sensor data [34]. DeepForest is a RGB deep learning model that predicts canopy crown bounding boxes [11, 23, 35]." Table 3: "Recall 79.0 Precision 65.9 ... 72.2 ... 74.0"
- **Verdict:** SUPPORTED
- **Note:** "RetinaNet" is not named in this paper (it is in weinstein2020); DeepForest is a box detector, not instance segmentation.

### paper.tex L393
> The DeepForest model released alongside the NEONTreeEvaluation benchmark~\cite{neontreeeval} was pretrained on all 22 sites, using an unsupervised LiDAR-based algorithm~\cite{silva2016} to generate millions of moderate-quality annotations, and was then fine-tuned on a further 10,000 manual annotations of RGB imagery from six sites.
- **Location:** PDF p.1 Abstract; PDF p.7 §"Image-annotated crowns" para 1; PDF p.11 §"Training annotations"; PDF p.14 para after Table 3
- **Quote:** "In addition, we include over 10,000 training crowns for optional use." / "We selected airborne imagery from 22 sites surveyed by the NEON AOP." / "oak woodland (NEON site: SJER), mixed pine (TEAK), alpine forest (NIWO), riparian woodlands (LENO), southern pinelands (OSBS), and eastern deciduous forest (MLBS)" / "The prebuilt model in DeepForest was trained with the training data described above"
- **Verdict:** PARTIAL
- **Note:** 22 sites, >10,000 hand-annotated training crowns and the six sites are here; the LiDAR (Silva) pretraining on all 22 sites with "millions of moderate quality annotations" is NOT in this paper (Silva [6] is cited here only for stem-location definition, p.15). Those specifics are in weinstein2020 (DeepForest, bioRxiv p.4 lines 72-76 and p.26 lines 455-456), which should carry that part of the sentence.

## ball2023 — Accurate delineation of individual tree crowns in tropical forests from aerial RGB imagery using Mask R-CNN (Ball 2023)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/ball2023.pdf` — from https://plymsea.ac.uk/id/eprint/9929/1/ (author-deposited Wiley early-view PDF); journal version, early-view pagination 1–14 (footer "20563485, 0, Downloaded ... [14/05/2023]"); issue pages 641–655 not printed
**Bib check:** OK on title, 10 authors, Ball first, RSEC, 2023, DOI 10.1002/rse2.332. Volume/issue/pages 9(5):641–655 not printed in early-view copy (consistent with Semantic Scholar/OAM-TCD ref list "9(5):641–655").

### paper.tex L198
> ... while Detectree2~\cite{ball2023} applied Mask R-CNN~\cite{maskrcnn} to tropical crown polygons.
- **Location:** PDF p.1 (printed 1), Abstract
- **Quote:** "Here we describe a new deep convolutional neural network method, Detectree2, which builds on the Mask R-CNN computer vision framework to recognize the irregular edges of individual tree crowns from airborne RGB imagery. We trained and evaluated this model with 3797 manually delineated tree crowns at three sites in Malaysian Borneo and one site in French Guiana."
- **Verdict:** SUPPORTED

### paper.tex L261
> We follow Ball et al.~\cite{ball2023} in defining Instance Segmentation as the precise delineation of individual tree crowns.
- **Location:** PDF p.4 (printed 4), §"Model architecture and parameterization", L col para 1 continuing R col
- **Quote:** "Instance segmentation combines object detection with object segmentation. Once an object has been detected in a scene, a region of interest (as a bounding box) is established around the object. Then a 'segmentation' is then carried out to identify which pixels within the region of interest make up the object of interest"
- **Verdict:** PARTIAL
- **Note:** Ball et al. define instance segmentation generically (detection + per-pixel segmentation within the box) and apply it to crowns ("perform the delineation of individual tree crowns", p.4 R col); they do not define it as "precise delineation of individual tree crowns".

### paper.tex L339
> detectree2 \cite{ball2023} & Polygon & RGB & -- & 10 & \cmark & \xmark & \xmark \\
- **Location:** PDF p.1 Abstract (polygon masks, RGB); PDF p.3 (printed 3) Table 1 and §"Remote sensing data" (GSD); PDF p.6 (printed 6) §"Performance by site and tree height" (per-site tropical evaluation)
- **Quote:** "Danum RGB 10 cm ... Sepilok RGB 10 cm ... Paracou RGB 8 cm" / "Malaysian forest, with a ground resolution of 10 cm. In Paracou, we sampled 10.2 km2 of imagery, with an 8 cm ground resolution."
- **Verdict:** SUPPORTED
- **Note:** GSD is 8–10 cm (Paracou 8 cm); only tropical sites (Malaysia, French Guiana), reported per site.

### paper.tex L934
> Existing datasets that try to provide instance segmentation labels very often will tend to attempt to draw definitive `ground truth' on dense canopy where there arguably is simply not enough information in the image to make that determination definitively (Yi et al. 2023, Ball et al.~\cite{ball2023}).
- **Location:** PDF p.4 (printed 4), §"Manual tree crown data", L col para 1 (continues from p.3)
- **Quote:** "These techniques meant that the vast majority of tree crowns were separable by eye but, it should be noted, that in rare cases, tree crowns were near impossible to delineate with certainty and the labeller's best estimate was used."
- **Verdict:** PARTIAL
- **Note:** Ball et al. concede uncertainty only "in rare cases" and say the "vast majority" were separable by eye; they used lidar CHM masks and contrast tweaks to aid labelling. Source does not say dense canopy "very often" lacks enough information.

## maskrcnn — Mask R-CNN (He 2017/2020)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/maskrcnn.pdf` — from https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf; ICCV 2017 camera-ready (printed pp.2961–2969). arXiv 1703.06870 also saved as `maskrcnn_arxiv.pdf`. The TPAMI 2020 version cited in the bib is paywalled.
**Bib check:** Authors (He, Gkioxari, Dollár, Girshick; 4) and title OK. Bib cites the TPAMI 2020 journal extension (42(2):386–397, doi 10.1109/TPAMI.2018.2844175), which exists; the audited PDF is the ICCV 2017 version — content of the cited claim is identical in both.

### paper.tex L198
> Modern deep learning approaches focus on instance segmentation on RGB imagery: DeepForest~\cite{weinstein2020} established a RetinaNet-based baseline evaluated on the NeonTreeEvaluation benchmark~\cite{neontreeeval}, while Detectree2~\cite{ball2023} applied Mask R-CNN~\cite{maskrcnn} to tropical crown polygons.
- **Location:** PDF p.1 (printed p.2961), Abstract, L col, para 1
- **Quote:** "We present a conceptually simple, flexible, and general framework for object instance segmentation. ... The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition."
- **Verdict:** SUPPORTED

## tong2025 — ITC delineation with StarDist-based model (Tong 2025)
**PDF:** INACCESSIBLE — ScienceDirect PII S0034425725000227 (pdfft, cookie session, r.jina.ai proxy, WebFetch all return 403/captcha); Semantic Scholar lists openAccessPdf only as the DOI (hybrid CC-BY); no arXiv/repository copy. Verified from the OpenAlex/Crossref abstract + highlights.
**Bib check:** OK against Crossref (Tong, Zhang; RSE vol 319, art. 114618, 2025, DOI OK).

### paper.tex L198
> Tong \& Zhang~\cite{tong2025} applied StarDist~\cite{stardist}, a U-Net~\cite{unet} backbone model applying star-convex polygons as proposals during the detection stage rather than the more commonly used axis-aligned bounding boxes, allowing for improved Non-Maximum Suppression (NMS) performance and more accurate detections.
- **Location:** Abstract (first page of article), sentences 3–5
- **Quote:** "The StarDist model captures tree crown shapes uniquely through star-convex polygons, which are predicted by the U-Net architecture. The final tree crowns are determined by applying non-maximum suppression (NMS) to all identified star-convex polygons."
- **Verdict:** SUPPORTED (from abstract; INACCESSIBLE for body text)
- **Note:** The "rather than axis-aligned bounding boxes / improved NMS" rationale is StarDist's (see stardist below), not spelled out in Tong & Zhang's abstract.

### paper.tex L285
> Other significant contributions in the field include Tong \& Zhang~\cite{tong2025}, based on StarDist~\cite{stardist}, which demonstrated the TCIS improvements provided by the usage of star-convex polygons during the Non-Maximum Suppression (NMS) stage whereby initial predictions with significant overlap are pruned, eliminating lower confidence predictions through a variety of methods and strategies.
- **Location:** Abstract, sentences 5–6 and highlights
- **Quote:** "Performance evaluation on two mixed forest areas reveals a delineation accuracy exceeding 92%, notably outperforming the widely used deep learning model MASK R-CNN by over 6%."
- **Verdict:** PARTIAL
- **Note:** The paper shows StarDist beats Mask R-CNN by >6% delineation accuracy on two mixed-forest test areas; it does not isolate the NMS stage as the cause of the improvement, and "a variety of methods and strategies" for NMS is not in the abstract. Body text unverified.

## stardist — Cell Detection with Star-Convex Polygons (Schmidt 2018)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/stardist.pdf` — from https://arxiv.org/pdf/1806.03535; arXiv v2 (8 Nov 2018); page numbers are arXiv pages (no printed LNCS pagination)
**Bib check:** OK (title, 4 authors Schmidt/Weigert/Broaddus/Myers match p.1; MICCAI 2018 / LNCS 11071 pp.265–273 not printed in arXiv version, consistent with DOI)

### paper.tex L198
> Tong \& Zhang~\cite{tong2025} applied StarDist~\cite{stardist}, a U-Net~\cite{unet} backbone model applying star-convex polygons as proposals during the detection stage rather than the more commonly used axis-aligned bounding boxes, allowing for improved Non-Maximum Suppression (NMS) performance and more accurate detections.
- **Location:** PDF p.2, §1 Introduction, para 2 (and Fig. 1 caption (c)); PDF p.2–3, §2 Method, para 1
- **Quote:** "NMS can be problematic if the objects of interest are poorly represented by their axis-aligned bounding boxes" (p.2); "StarDist uses a light-weight neural network based on U-Net [15]" (p.2); "Unlike most of them, we do not use axis-aligned bounding boxes as the shape representation ... Instead, our model predicts a star-convex polygon for every pixel ... we perform non-maximum suppression (NMS) to arrive at the final set of polygons" (p.2–3)
- **Verdict:** SUPPORTED

### paper.tex L285
> Other significant contributions in the field include Tong \& Zhang~\cite{tong2025}, based on StarDist~\cite{stardist}, which demonstrated the TCIS improvements provided by the usage of star-convex polygons during the Non-Maximum Suppression (NMS) stage whereby initial predictions with significant overlap are pruned, eliminating lower confidence predictions through a variety of methods and strategies.
- **Location:** PDF p.2, §1 Introduction, para 2; PDF p.4, §2 "Non-maximum suppression" para
- **Quote:** "performing a non-maximum suppression (NMS) step where boxes with lower confidence are suppressed by boxes with higher confidence if they substantially overlap" (p.2); "We perform common, greedy non-maximum suppression (NMS, cf. [14,9,12]) to only retain those polygons in a certain region" (p.4)
- **Verdict:** SUPPORTED (for the NMS mechanism); the TCIS result is Tong & Zhang's, see above

## unet — U-Net (Ronneberger 2015)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/unet.pdf` — from https://arxiv.org/pdf/1505.04597; arXiv v1; page numbers are arXiv pages
**Bib check:** OK (title, Ronneberger/Fischer/Brox, 2015; MICCAI 2015 / LNCS 9351 pp.234–241 not printed, consistent with DOI)

### paper.tex L198
> Tong \& Zhang~\cite{tong2025} applied StarDist~\cite{stardist}, a U-Net~\cite{unet} backbone model applying star-convex polygons ...
- **Location:** PDF p.1, title + Abstract
- **Quote:** "U-Net: Convolutional Networks for Biomedical Image Segmentation ... The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization."
- **Verdict:** SUPPORTED (generic "this is U-Net" citation; StarDist p.2 confirms "based on U-Net [15]")

## selvamask — SelvaMask: Segmenting Trees in Tropical Forests and Beyond (Duguay 2026)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/selvamask.pdf` — from https://arxiv.org/pdf/2602.02426; arXiv v1 (2 Feb 2026, "Preprint. February 3, 2026"); page numbers are arXiv pages (printed = PDF page)
**Bib check:** OK (title, 6 authors, Duguay first, 2026, arXiv:2602.02426)

### paper.tex L198
> Similarly, the authors of the tropical rainforest SelvaMask dataset~\cite{selvamask} utilised a DINO (DIstillation with NO labels)~\cite{dino} based tree crown detection model coupled with a Segment Anything Model (SAM)~\cite{sam} for prediction mask generation, applied to instance segmentation in dense canopy rainforests.
- **Location:** PDF p.5, §4.2 (left col, para "We will denote this pipeline as 'detection-prompter → SAM'")
- **Quote:** "We evaluate three detection-prompter variants: DeepForest (Zhou & Feng, 2020), a tree detection model; DINO (Caron et al., 2021), a DINO-Swin-L backbone pre-trained on COCO; and SelvaBox (Baudchon et al., 2026), a DINO-Swin-L detection model trained on tropical forest data."
- **Verdict:** PARTIAL
- **Note:** SelvaMask does cite Caron et al. 2021 for "DINO", but the DINO-Swin-L detector is the DETR-based DINO (Zhang et al. 2023); SelvaBox p.6 states explicitly "DINO (the DETR-based detector) and DINO (the self-supervised embedding model) are unrelated despite sharing the same name." The "DIstillation with NO labels" expansion and \cite{dino} are therefore the wrong DINO.

### paper.tex L216
> Recent annotated datasets such as SelvaBox~\cite{selvabox} \& SelvaMask~\cite{selvamask} have broadened geographic coverage ...
- **Location:** PDF p.1, Abstract
- **Quote:** "we introduce SELVAMASK, a new tropical dataset containing over 8 800 manually delineated tree crowns across three Neotropical forest sites in Panama, Brazil, and Ecuador."
- **Verdict:** SUPPORTED

### paper.tex L313
> A second related dataset, SELVAMASK~\cite{selvamask}, provides ITC instance segmentation masks and assesses the SelvaBox detection model (also referred to by the authors as `SelvaBox') in combination with both frozen and fine-tuned SAM3 decoder modules on the OAM-TCD dataset for the TCIS task.
- **Location:** PDF p.5 §4.2 (detection-prompter para); PDF p.7 Table 4, "Urban / OAM-TCD" rows and caption; PDF p.7 §5.3 right col, para "For urban OAM-TCD"
- **Quote:** "SelvaBox (Baudchon et al., 2026), a DINO-Swin-L detection model trained on tropical forest data" / "For urban OAM-TCD, Detectree2 (flexi) reaches 12.3 mAP. While the frozen pipeline struggles, fine-tuning on SELVAMASK achieves 18.5 mAP" / Table 4 caption: "We note [fire] models that are fine-tuned on SELVAMASK and [snowflake] models whose weights are kept frozen."
- **Verdict:** SUPPORTED
- **Note:** Fine-tuning updates "the image encoder, prompt encoder, and mask decoder jointly" (p.5), i.e. whole SAM3, not only the decoder.

### paper.tex L350
> SelvaMask \cite{selvamask} & Polygon & RGB & VFM & 1--4 & \cmark & \cmark & \xmark \\
- **Location:** GSD: PDF p.2 Table 1 last row and PDF p.4 §3 (left col); VFM: p.5 §4.2; biomes: PDF p.6 §5.1 "External datasets" and PDF p.7 Table 4 (Tropical/Temperate/Urban groups)
- **Quote:** "SELVAMASK (ours) tropical natural 8.9k 1.3–3.5 All Crowns" / "we conduct an ablation study on five additional datasets (Tab. 4): two tropical (Detectree2, BCI50ha), two temperate (BAMForest, QuebecTrees), and one mixed-biome urban (OAM-TCD)"
- **Verdict:** SUPPORTED
- **Note:** Dataset GSD is 1.3–3.5 cm (1–4 is a loose rounding). Tropical and temperate reported per dataset; no savannah.

### paper.tex L618
> One approach in the literature~\cite{selvamask}, which we support, is to ignore all canopy annotations (cat=2) altogether.
- **Location:** Searched all of selvamask.txt for canopy/group/cat/class/crowd/OAM-TCD (OAM-TCD appears only at p.2 Table 1, p.6 §5.1, p.7 Table 4/§5.3): no preprocessing statement.
- **Quote:** (none in SelvaMask) — the backing text is in SelvaBox PDF p.32 App. F.1: "OAM-TCD contains two types of annotations: individual trees and tree groups ... we only consider individual trees annotations and we mask the pixels associated to tree groups from the training data to ensure consistency."
- **Verdict:** WRONG SOURCE
- **Note:** Belongs to SelvaBox (Baudchon et al. 2026) App. F.1, which masks group pixels in *training* data; it does not state that groups are ignored at evaluation. The same paragraph in paper.tex already attributes the pixel-deletion to SELVABOX, consistent with this.

## dino — Emerging Properties in Self-Supervised Vision Transformers (Caron 2021)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/dino.pdf` — from https://arxiv.org/pdf/2104.14294; arXiv v2 (24 May 2021); no printed page numbers (ICCV pages 9650–9660 not verifiable from arXiv)
**Bib check:** OK (title, first author Mathilde Caron, 7 authors, FAIR/Inria/Sorbonne; ICCV 2021 venue/pages not printed on arXiv copy)

### paper.tex L198
> Similarly, the authors of the tropical rainforest SelvaMask dataset~\cite{selvamask} utilised a DINO (DIstillation with NO labels)~\cite{dino} based tree crown detection model coupled with a Segment Anything Model (SAM)~\cite{sam} for prediction mask generation, applied to instance segmentation in dense canopy rainforests.
- **Location:** PDF p.1, Abstract, L col, last sentences; also PDF p.2 Figure 2 caption ("Self-distillation with no labels") and §3.1 heading "SSL with Knowledge Distillation" (p.3 L col)
- **Quote:** "We implement our findings into a simple self-supervised method, called DINO, which we interpret as a form of self-distillation with no labels."
- **Verdict:** SUPPORTED
- **Note:** Paper writes "self-distillation with no labels" (lowercase; "DIstillation with NO labels" capitalisation is the manuscript's own gloss). The SelvaMask usage itself must be verified against \cite{selvamask}, not this paper.

## sam — Segment Anything (Kirillov 2023)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/sam.pdf` — from https://arxiv.org/pdf/2304.02643; arXiv v1 (5 Apr 2023); page numbers are arXiv pages (30 pp.)
**Bib check:** OK (title, 12 authors Kirillov…Girshick, Meta AI; ICCV 2023 pages not printed on arXiv copy)

### paper.tex L198
> Similarly, the authors of the tropical rainforest SelvaMask dataset~\cite{selvamask} utilised a DINO (DIstillation with NO labels)~\cite{dino} based tree crown detection model coupled with a Segment Anything Model (SAM)~\cite{sam} for prediction mask generation, applied to instance segmentation in dense canopy rainforests.
- **Location:** PDF p.1, title and Abstract, L col, first sentence
- **Quote:** "We introduce the Segment Anything (SA) project: a new task, model, and dataset for image segmentation."
- **Verdict:** SUPPORTED
- **Note:** Generic "this is SAM" citation; the SelvaMask usage claim itself must be checked against the selvamask source (not in this group).

### paper.tex L299
> Meta's Segment Anything Model (SAM)~\cite{sam}, now in its third iteration~\cite{sam3}, is a promptable class-agnostic segmentation foundation model: given a point, box or mask prompt it emits an instance mask, which makes it a natural box-to-mask module for detectors that emit only rectangles, and it is used in that role by several of the baselines compared here. %TBC expand
- **Location:** PDF p.2, §1 Introduction, L col, para "Model (§3)"; PDF p.5, §3 Segment Anything Model, L col, "Prompt encoder"; class-agnosticism PDF p.5 R col §4 Data Engine ("Assisted-manual stage") and p.6 L col
- **Quote:** p.2: "We focus on point, box, and mask prompts, and also present initial results with free-form text prompts." p.5: "Prompt encoder. We consider two sets of prompts: sparse (points, boxes, text) and dense (masks)." p.5–6: "We did not impose semantic constraints for labeling objects, and annotators freely labeled both 'stuff' and 'things'"; "generic 'object' category".
- **Verdict:** SUPPORTED
- **Note:** The paper never uses the phrase "class-agnostic"; masks carry no class labels (generic "object" category), so the paraphrase is fair.

## rsprompter — RSPrompter (Chen 2024)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/rsprompter.pdf` — from https://arxiv.org/pdf/2306.16269; arXiv v2 (29 Nov 2023); page numbers are arXiv pages (printed 1–18)
**Bib check:** OK (title, 7 authors Chen/Liu/Chen/Zhang/Li/Zou/Shi match p.1; TGRS vol 62 pp.1–17 (2024) not printed on arXiv version)

### paper.tex L198
> This approach using SAM has also been explored in a number of works, including RSPrompter~\cite{rsprompter} and Teng et al.~\cite{teng2025} ... these works report suboptimal results for out-of-the-box SAM but find further fine-tuning promising.
- **Location:** PDF p.1, Fig. 1 caption and Abstract; PDF p.9 (printed 9), L col, §IV.C.1 last para; Tables (p.9) rows "SAM-cls"
- **Quote:** "It is evident that the type, location, and number of prompts significantly impact the results produced by SAM." (p.1); "the SAM backbone, trained on an extensive dataset, can offer invaluable instance segmentation guidance even when it is completely frozen (as observed in SAM-seg)" (p.9); SAM-cls (SAM "everything" mode + classifier) scores 46.8 APbox vs 71.9 for RSPrompter-anchor on WHU (Table, p.9)
- **Verdict:** PARTIAL
- **Note:** Suboptimal out-of-the-box SAM ✓ (SAM-cls zero-shot masks score far below learned variants). But RSPrompter does NOT fine-tune SAM — the encoder is kept frozen and only a prompter is trained ("keeping the cumbersome encoder frozen", p.7). "Further fine-tuning promising" fits Teng et al., not RSPrompter; say "learning to prompt / lightweight adaptation" instead.

### paper.tex L319
> Leveraging SAM as an out-of-the-box instance segmenter for remote sensing has been explored in RSPrompter~\cite{rsprompter}, which overcomes SAM's reliance on prompts such as points/boxes/masks by training a prompter module that learns the intermediate layer features of the SAM encoder on labelled training images, and then using those embeddings as input to SAM's mask decoder in order to produce category-labelled instance masks.
- **Location:** PDF p.2, L col, §I Introduction, para 3 ("Our research is primarily centered..."); PDF p.4, §III.B para 1 and Eq. block
- **Quote:** "We propose a lightweight feature enhancer to collect features from the SAM encoder's intermediate layers for the subsequent prompter. The prompter can generate prompts with semantic categories ... we propose a more flexible representation of prompts, i.e., prompt embeddings, rather than the original coordinates." (p.2); "the image ... is processed by the frozen SAM image encoder to generate Fimg ... and multiple intermediate feature maps {Fi}" (p.4)
- **Verdict:** SUPPORTED
- **Note:** Minor wording: RSPrompter is not "out-of-the-box" use of SAM — it adds a trained prompter; the sentence's own "training a prompter module" makes that clear.

## teng2025 — Bringing SAM to New Heights (Teng 2025)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/teng2025.pdf` — from https://arxiv.org/pdf/2506.04970; arXiv camera-ready (footer "39th Conference on Neural Information Processing Systems (NeurIPS 2025)"); page numbers are arXiv pages (printed 1–33)
**Bib check:** OK (title, 6 authors Teng/Ouaknine/Laliberté/Bengio/Rolnick/Larochelle, NeurIPS 2025 confirmed on p.1 footer)

### paper.tex L198
> This approach using SAM has also been explored in a number of works, including RSPrompter~\cite{rsprompter} and Teng et al.~\cite{teng2025}, the latter both with and without the integration of a Digital Surface Model (DSM) depth map; these works report suboptimal results for out-of-the-box SAM but find further fine-tuning promising.
- **Location:** PDF p.1, Abstract, last 3 sentences; also PDF p.7 §5 para "Using SAM out-of-the-box is suboptimal..." and PDF p.10 §7 Conclusion
- **Quote:** "We also study the integration of elevation data into models, in the form of Digital Surface Model (DSM) information ... We find that methods using SAM out-of-the-box do not outperform a custom Mask R-CNN, even with well-designed prompts. However, efficiently tuning SAM end-to-end and integrating DSM information are both promising avenues"
- **Verdict:** SUPPORTED

### paper.tex L226
> Besides domain coverage and distribution, tree crown instance segmentation also remains understudied due to the lack of individual tree crown mask annotations in datasets~\cite{teng2025}.
- **Location:** PDF p.1, §1 Introduction, para 1, last sentence (immediately above the NeurIPS footer); restated PDF p.2, §2 Related work, para 1
- **Quote:** "Despite the success of deep learning methods for tree mapping at scale using remote sensing imagery [12, 13], instance segmentation of tree crowns remains understudied, in large part because of the lack of annotated data at the individual tree level." (p.1); "This task has remained understudied due to the limited availability of labelled high-resolution datasets." (p.2)
- **Verdict:** SUPPORTED
- **Note:** Source says "annotated data at the individual tree level"; manuscript's "mask annotations" is a slight narrowing but consistent.

### paper.tex L348
> Box-prompt SAM \cite{teng2025} & Box & RGB+DSM & SAM & $<$5 & \xmark & \cmark & \xmark \\
- **Location:** Method: PDF p.5, §4.1 "Faster/Mask R-CNN+SAM and variations"; GSD: PDF p.3–4 §3 dataset paras; Biomes: PDF p.1 Abstract, and per-dataset Tables 1 (p.6, Quebec Plantations), 2 (p.7, SBL), 3 (p.7, BCI)
- **Quote:** "We train a Faster R-CNN for tree crown detection on each dataset to provide box prompts to SAM (Faster R-CNN+SAM) ... we also consider stacking the DSM modality to its corresponding RGB image as a fourth channel" (p.5); "resolution of 5 mm/pixel" (p.3), "1.9 cm/pixel" (p.4), "50-ha rectangular plot of tropical forest at a resolution of 4 cm/pixel" (p.4); "three use cases: 1) boreal plantations, 2) temperate forests and 3) tropical forests" (p.1)
- **Verdict:** PARTIAL
- **Note:** Box ✓ (Faster R-CNN+SAM variant; note the training data are full masks, only the prompt is a box), RGB+DSM ✓, SAM ✓, <5 cm ✓. BIOME WRONG: Teng et al. evaluate a tropical site (BCI, Panama) separately in Table 3 (p.7) and a boreal plantation site (Table 1) — so Tr should be ● not ×; Te ● ✓ (SBL); Sv × ✓.

## oamtcd — OAM-TCD: A globally diverse dataset of high-resolution tree cover maps (Veitch-Michaelis 2024)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/oamtcd.pdf` — from https://arxiv.org/pdf/2407.11743; arXiv v1 (16 Jul 2024); page numbers are arXiv pages (printed = PDF page)
**Bib check:** OK (title, 8 authors, Veitch-Michaelis first, 2024; venue NeurIPS D&B not printed on arXiv PDF but consistent with paper style)

### paper.tex L216
> Recent annotated datasets such as SelvaBox~\cite{selvabox} \& SelvaMask~\cite{selvamask} have broadened geographic coverage, and Restor Foundation's OAM-TCD dataset~\cite{oamtcd} in particular incorporates a variety of biomes across the globe, including urban, tundra, and savannah.
- **Location:** PDF p.1, Abstract; PDF p.5, §3.1 (para "We also consider biome distribution"); PDF p.23, Appendix Table 2
- **Quote:** "By sampling imagery from around the world, we are able to better capture the diversity and morphology of trees in different terrestrial biomes and in both urban and natural environments." / Table 2 rows: "(7) Tropical & Subtropical Grasslands, Savannas, and Shrublands ... 142", "(11) Tundra ... 9"
- **Verdict:** SUPPORTED
- **Note:** Tundra (9 tiles) and Flooded Savannas (10 tiles) are explicitly called "not well-represented" (p.5); "urban" is not a WWF biome class but is claimed in the abstract.

### paper.tex L338
> Mask R-CNN \cite{oamtcd} & Polygon & RGB & -- & 10 & \omark & \omark & \omark \\
- **Location:** PDF p.1 Abstract (10 cm/px, instance masks); PDF p.7, §4 "Instance segmentation" para; PDF p.5 §3.1 (biome-stratified k-fold); PDF p.23 Table 2
- **Quote:** "OAM-TCD, comprises 5072 2048x2048 px images at 10 cm/px resolution with associated human-labeled instance masks" / "we trained Mask-RCNN models with a ResNet50 backbone, using the Detectron2 framework ... Cross-validation performance of the models is mAP50=41.79±1.38, and mAP50=43.22 on the holdout set" / "results using a k-fold biome-stratified cross-validation and a test (holdout) split"
- **Verdict:** SUPPORTED
- **Note:** All three ○ correct: tropical, temperate and savannah tiles exist (Table 2) but only aggregate mAP is reported, no per-biome breakdown.

### paper.tex L382
> In our work, the OAM-TCD dataset has been an invaluable benchmark ... due to the inclusion of not only bounding boxes but also precise polygon masks, as well as its broad range of ecological biomes to capture the diversity and morphology of trees in different terrestrial biomes including both urban and natural environments~\cite{oamtcd}.
- **Location:** PDF p.1 Abstract; PDF p.5, §3.2 "Image and label characteristics", para 2
- **Quote:** "Labels are provided as semantic masks and in MS-COCO instance segmentation (polygon) format with two classes: tree and canopy (group of trees)." / "capture the diversity and morphology of trees in different terrestrial biomes and in both urban and natural environments"
- **Verdict:** PARTIAL
- **Note:** Source describes labels as semantic masks + COCO polygons only; bounding boxes are not listed as a provided annotation (they are derivable from COCO polygons). Biome/urban wording is near-verbatim from the abstract.

## boxinst — BoxInst: High-Performance Instance Segmentation with Box Annotations (Tian 2021)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/boxinst.pdf` — from https://arxiv.org/pdf/2012.02310; arXiv v1 (3 Dec 2020); page numbers are arXiv pages (10 pp.)
**Bib check:** OK (title, 4 authors Tian/Shen/Wang/Chen match; CVPR 2021 venue/pages not printed on arXiv copy)

### paper.tex L226
> Since crown polygons are slower and typically more expensive to annotate than boxes, box-supervised instance segmentation~\cite{boxinst,box2mask,boxteacher} offers a practical route to instance-level delineation without polygon labels.
- **Location:** PDF p.1, Abstract, L col, first sentence
- **Quote:** "We present a high-performance method that can achieve mask-level instance segmentation with only bounding-box annotations for training."
- **Verdict:** SUPPORTED
- **Note:** BoxInst backs "instance segmentation from boxes only"; the annotation-cost comparison is backed by box2mask/boxteacher below, not by BoxInst.

## box2mask — Box2Mask: Box-Supervised Instance Segmentation via Level-Set Evolution (Li 2024)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/box2mask.pdf` — from https://arxiv.org/pdf/2212.01579; arXiv v1 (3 Dec 2022, only version); page numbers are arXiv pages (29 pp.). TPAMI version not OA; journal abstract checked via PubMed 38319771 and Crossref.
**Bib check:** OK — Crossref confirms TPAMI vol 46 no 7 pp 5157–5173 (Jul 2024), 7 authors. arXiv prints "Xiansheng Hua" (bib "Xian-Sheng Hua") — cosmetic.

### paper.tex L226
> Since crown polygons are slower and typically more expensive to annotate than boxes, box-supervised instance segmentation~\cite{boxinst,box2mask,boxteacher} offers a practical route to instance-level delineation without polygon labels.
- **Location:** PDF p.2, §1 Introduction, para 2 (single column)
- **Quote:** "On average, it costs about 79.2 seconds to generate the polygon-based mask of an object on COCO [45], whereas it costs only 7 seconds [56] to annotate the bounding box of an object."
- **Verdict:** SUPPORTED

### paper.tex L653
> \item[d] Box2Mask~\cite{box2mask}, in its strongest released backbone (Swin-L, $42.5$ COCO mask AP) and R-50 ($35.9$) as a backbone sensitivity.
- **Location:** PDF p.14, §4.3.1, Table 2 "Performance comparison on COCO test-dev", last rows; text on PDF p.16 top para; Abstract p.1
- **Quote:** Table 2: "Box2Mask-T – Swin-L/ 42.4 70.2 43.3 22.1 45.9 62.9" and "Box2Mask-T – ResNet-50 36.7 61.9 37.2 18.2 39.6 53.2" (test-dev); "Box2Mask-T† – ResNet-50 36.1 ..." (val2017). p.16: "With Swin-Transformer base model (Swin-B) and Swin-Transformer large (Swin-L) model as the backbone, Box2Mask-T obtains 41.5% and 42.4% mask AP, respectively. It is on par with the recent fully-supervised approaches with ResNet-101"
- **Verdict:** PARTIAL
- **Note:** Neither 42.5 nor 35.9 appears in the paper (arXiv or TPAMI; the TPAMI abstract also says "42.4% mask AP on COCO"). Both numbers come from the BoxInstSeg GitHub README (https://github.com/LiWentomng/BoxInstSeg, "COCO (val)" table, column "AP (this rep)"): Swin-L "41.9/42.5" (val/test-dev), R-50 "35.9" (val); the README's "AP (original rep/paper)" column gives 41.3/42.4 and 36.1. So 42.5 is the repo's re-trained checkpoint on test-dev and 35.9 is the repo checkpoint on val — mixed splits. The scratch line "42.4 mask AP COCO with Swin-L, on par with fully mask-supervised" is the paper's own number (abstract, identical in arXiv v1 and TPAMI). Either cite 42.4 (paper, test-dev) and 36.7/36.1 for R-50, or keep 42.5/35.9 and attribute them to the released checkpoints in the BoxInstSeg repo.

## boxteacher — BoxTeacher: Exploring High-Quality Pseudo Labels for Weakly Supervised Instance Segmentation (Cheng 2023)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/boxteacher.pdf` — from https://arxiv.org/pdf/2210.05174; arXiv v2 (17 Mar 2023); page numbers are arXiv pages (11 pp.)
**Bib check:** OK (title, 5 authors Cheng/Wang/Chen/Zhang/Liu; CVPR 2023 pages not printed on arXiv copy)

### paper.tex L226
> Since crown polygons are slower and typically more expensive to annotate than boxes, box-supervised instance segmentation~\cite{boxinst,box2mask,boxteacher} offers a practical route to instance-level delineation without polygon labels.
- **Location:** PDF p.1, Abstract, L col, first sentence
- **Quote:** "Labeling objects with pixel-wise segmentation requires a huge amount of human labor compared to bounding boxes."
- **Verdict:** SUPPORTED

## dinov2 — DINOv2: Learning Robust Visual Features without Supervision (Oquab 2024)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/dinov2.pdf` — from https://arxiv.org/pdf/2304.07193; arXiv v2 (2 Feb 2024) = TMLR camera-ready ("Published in Transactions on Machine Learning Research (01/2024)"); printed page numbers = PDF pages
**Bib check:** OK (title, first author Maxime Oquab, 26 authors, TMLR 01/2024)

### paper.tex L226
> The recent rise of self-supervised backbones such as DINOv2~\cite{dinov2} and DINOv3~\cite{dinov3} has demonstrated that models pretrained on supersized sets of diverse web imagery (1.7 billion images in the case of DINOv3-web) are capable of matching or even exceeding Earth-observation-specific foundation models (e.g. DINOv3-sat) on high-resolution RGB RS tasks, with the latter retaining an advantage chiefly where multispectral or multitemporal signal is essential~\cite{geobench}.
- **Location:** PDF p.1 (printed p.1), Abstract
- **Quote:** "This work shows that existing pretraining methods, especially self-supervised methods, can produce such features if trained on enough curated data from diverse sources. ... we propose an automatic pipeline to build a dedicated, diverse, and curated image dataset"
- **Verdict:** SUPPORTED (for the "self-supervised backbone pretrained on curated diverse web imagery" part only)
- **Note:** DINOv2 contains no Earth-observation / DINOv3-sat comparison; that part of the sentence rests on dinov3 (see above), not dinov2 or geobench.

## dinov3 — DINOv3 (Siméoni 2025)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/dinov3.pdf` — from https://arxiv.org/pdf/2508.10104; arXiv v1 (13 Aug 2025); printed page numbers = PDF pages
**Bib check:** OK (title "DINOv3", first author Oriane Siméoni, 26 authors, Meta AI Research, 2025, arXiv:2508.10104)

### paper.tex L226
> The recent rise of self-supervised backbones such as DINOv2~\cite{dinov2} and DINOv3~\cite{dinov3} has demonstrated that models pretrained on supersized sets of diverse web imagery (1.7 billion images in the case of DINOv3-web) are capable of matching or even exceeding Earth-observation-specific foundation models (e.g. DINOv3-sat) on high-resolution RGB RS tasks, with the latter retaining an advantage chiefly where multispectral or multitemporal signal is essential~\cite{geobench}.
- **Location:** (1.7B) PDF p.8, §3.1 Data Preparation, para "Data Collection and Curation", single column, mid-page; (web vs sat) PDF p.34, §8.3 Comparison to the Earth Observation State of the Art, paras 1–2, and PDF p.35 Table 18 + final para of §8.3 (p.35 top, after Fig. 18)
- **Quote:** "This results in a curated subset of 1,689 million images (named LVD-1689M)" (p.8). "Interestingly, the DINOv3 7B web model is very competitve on these benchmarks. It achieves comparable or stronger performance on many GEO-Bench tasks as well as on large-scale, high-resolution remote sensing benchmarks for segmentation and detection." (p.34). "The DINOv3 satellite model excels in metric tasks like depth estimation, leveraging satellite-specific priors. In contrast, the DINOv3 web model achieves state-of-the-art results on semantic geospatial tasks" (p.35).
- **Verdict:** PARTIAL
- **Note:** Web-vs-sat comparison is in this paper (Tab. 18 p.35: DINOv3 Web 7B mean 81.6/75.9 vs DINOv3 Sat 7B 81.1/75.0), not in geobench, so the sentence should cite dinov3 for it. "1.7 billion" ≈ LVD-1689M, but LVD-1689M is only one of three mixed pre-training parts (plus retrieval-curated part and ImageNet1k/22k/Mapillary; ImageNet1k homogeneous batches = 10% of training, p.9). The "latter retaining an advantage chiefly where multispectral or multitemporal signal is essential" clause is NOT what DINOv3 says: both DINOv3 models are RGB-only; the paper attributes the sat model's advantage to "physically grounded"/"metric tasks" (canopy height, p.34 §8.2: "sensor-specific priors and radiometric consistency"). Multitemporal/multisensor fusion is mentioned (p.34) only as heuristics other GeoFMs use.

### paper.tex L229
> In this work we propose a 3-stage model ``LACE'' built on a frozen DINOv3-web (ViT-L/16px)~\cite{dinov3} encoder, with a 4m parameter CenterNet-style~\cite{centernet} detector head, and a novel lightweight expectation-maximization module that leverages latent commonality of the ground truth bounding box annotations.
- **Location:** PDF p.15, §5.2 Model Distillation, paras 1–2; PDF p.9 Table 2 (Patch Size 16) and p.9 line 2; PDF p.30 Figure 16(a) "DINOv3 family of models" (row ViT-L, 300M params)
- **Quote:** "We perform knowledge distillation of the ViT-7B model into smaller Vision Transformer variants (ViT-S, ViT-B, and ViT-L)" (p.15); "Importantly, we use a patch size of 16 pixels" (p.9); "the model trained on LVD-1689M" is denoted "DINOv3 Web" (p.34)
- **Verdict:** SUPPORTED
- **Note:** The literal string "ViT-L/16" does not appear in the paper (only "ViT-7B/16", p.18–19); ViT-L is listed as a distilled model and patch size 16 is stated for the family. "DINOv3-web" naming is the paper's ("DINOv3 Web", §8).

### paper.tex L242
> \item a) our model LACE, built on a DINOv3-web (ViT-L/16) backbone~\cite{dinov3}, with a trained CenterNet-style detector head~\cite{centernet} that identifies crown centers for detection based on the DINO output embeddings.
- **Location:** as L229: PDF p.15 §5.2 paras 1–2; p.9 Table 2; p.30 Fig. 16(a)
- **Quote:** "Our ViT-7B model is distilled into a series of ViT models with sizes covering a broad range of compute budgets ... They include the standard ViT-S (21M params), B (86M), L (0.3B), along with a custom ViT-S+ (29M) and a custom ViT-H+ (0.8B)" (p.15)
- **Verdict:** SUPPORTED

### paper.tex L249
> This approach is made possible by the capabilities of Meta's DINOv3 encoder foundation model~\cite{dinov3}, a Vision-transformer-based model trained on 1.7 Billion images using Self-supervised Learning.
- **Location:** PDF p.8, §3.1, para "Data Collection and Curation"; PDF p.9 Table 2 (ViT-7B backbone); PDF p.1 Abstract
- **Quote:** "This results in a curated subset of 1,689 million images (named LVD-1689M) that guarantees a balanced coverage of all visual concepts appearing on the web." (p.8); "Self-supervised learning holds the promise of eliminating the need for manual data annotation ... This technical report introduces DINOv3" (p.1)
- **Verdict:** PARTIAL
- **Note:** 1.689B is the size of the curated LVD-1689M part; the actual pre-training mix also includes a retrieval-curated part and raw ImageNet1k/22k + Mapillary (p.8–9). Safer wording: "curated from a ~1.7 billion-image web dataset (LVD-1689M)".

### paper.tex L306 (a)
> Meta's latest version of the Self-Distillation with No Labels (DINO) family is DINOv3~\cite{dinov3}: a SOTA encoder trained with fully Self-Supervised Learning on 1.7 billion images pulled from various online sources.
- **Location:** PDF p.8, §3.1 para "Data Collection and Curation"; PDF p.1 Abstract
- **Quote:** "We build our large-scale pre-training dataset by leveraging a large data pool of web images collected from public posts on Instagram. ... we obtain an initial data pool of approximately 17 billions of images. ... This results in a curated subset of 1,689 million images (named LVD-1689M)" (p.8)
- **Verdict:** PARTIAL
- **Note:** Source is Instagram public posts (one platform), not "various online sources"; and 1.7B is one component of the mix (see L249). "Self-Distillation with No Labels" expansion is from Caron et al. 2021 (dino), not stated in DINOv3.

### paper.tex L306 (b)
> Highly context-agnostic, DINOv3-web is an example of a single frozen SSL backbone that can serve as a ``universal visual encoder''~\cite{dinov3} capable of generating rich embedding vectors that capture the semantic information from the raw pixel patches in context.
- **Location:** PDF p.2, §1 Introduction, para 4 ("Addressing the problems above leads to this work, DINOv3 ..."), single column
- **Quote:** "We demonstrate that a single frozen SSL backbone can serve as a universal visual encoder that achieves state-of-the-art performance on challenging downstream tasks, outperforming supervised and metadata-reliant pre-training strategies."
- **Verdict:** SUPPORTED
- **Note:** "universal visual encoder" and "single frozen SSL backbone" are verbatim (p.2).

### paper.tex L306 (c) [ibot sentence, dinov3 loss description — for cross-reference]
> Incorporating both a global image-level objective and a local iBOT-style~\cite{ibot} patch-level latent reconstruction objective in the loss function results in a model that excels at encoding global and local features.
- **Location:** PDF p.9, §3.2 Large-Scale Training with Self-Supervision, para "Learning Objective"; Eq. (1) PDF p.10
- **Quote:** "Following DINOv2 (Oquab et al., 2024), we use an image-level objective (Caron et al., 2021) LDINO, and balance it with a patch-level latent reconstruction objective (Zhou et al., 2021) LiBOT." ; "LPre = LDINO + LiBOT + 0.1 ∗ LDKoleo" (p.10, Eq. 1)
- **Verdict:** SUPPORTED (wording "image-level objective" / "patch-level latent reconstruction objective" is DINOv3's, p.9; consider adding \cite{dinov3} here)

## geobench — GEO-Bench: Toward Foundation Models for Earth Monitoring (Lacoste 2023)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/geobench.pdf` — from https://arxiv.org/pdf/2306.03831; arXiv v2 (23 Dec 2023); printed page numbers = PDF pages
**Bib check:** OK (title, first author Alexandre Lacoste, 17 authors; NeurIPS 2023 D&B venue not printed on arXiv copy)

### paper.tex L226
> The recent rise of self-supervised backbones such as DINOv2~\cite{dinov2} and DINOv3~\cite{dinov3} has demonstrated that models pretrained on supersized sets of diverse web imagery (1.7 billion images in the case of DINOv3-web) are capable of matching or even exceeding Earth-observation-specific foundation models (e.g. DINOv3-sat) on high-resolution RGB RS tasks, with the latter retaining an advantage chiefly where multispectral or multitemporal signal is essential~\cite{geobench}.
- **Location:** PDF p.1 (printed p.1) Abstract (benchmark definition); PDF p.8 (printed p.8) §6.2.2 Comparing Baselines on RGB only, para 2; PDF p.9 (printed p.9) §6.2.4 Leveraging Multispectral Information, para 2
- **Quote:** "we propose a benchmark comprised of six classification and six segmentation tasks" (p.1); "perhaps disappointingly, the existing model pre-trained on remote sensing data does not exhibit any improvement compared to their timm pre-trained weights, i.e., ResNet18-MoCo-S2, ResNet50-MoCo-S2, and ResNet50-SeCo-S2 are all comparable to ResNet18" (p.8); "the ResNet50 pre-trained on Sentinel-2 using DINO or MoCo Wang et al. leads to a modest performance increase on average. When looking at ViT-S (Fig. 7), incorporating multi-spectral only leads to a systematic performance decrease." (p.9)
- **Verdict:** WRONG SOURCE (PARTIAL at best)
- **Note:** GEO-Bench predates DINOv2/DINOv3-sat (2023) and only defines the benchmark; it contains no DINOv2/DINOv3 or DINOv3-sat results. The DINOv3-web vs DINOv3-sat comparison is DINOv3 §8.3 / Tab. 18 (p.34–35) — cite \cite{dinov3} there. GEO-Bench's own finding is weaker and different: ImageNet (timm) weights match RS-pretrained MoCo/SeCo ResNets, and multispectral pretraining gives only a "modest" gain (ResNet50) or a "systematic performance decrease" (ViT-S) — it does not establish that EO models retain an advantage "where multispectral or multitemporal signal is essential". Keep \cite{geobench} only as the benchmark reference (e.g. "on GEO-Bench~\cite{geobench}").

## centernet — Objects as Points (Zhou 2019)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/centernet.pdf` — from https://arxiv.org/pdf/1904.07850; arXiv v2; page numbers are arXiv pages.
**Bib check:** OK (Zhou, Wang, Krähenbühl; 3 authors; arXiv:1904.07850, 2019).

### paper.tex L229
> In this work we propose a 3-stage model ``LACE'' built on a frozen DINOv3-web (ViT-L/16px)~\cite{dinov3} encoder, with a 4m parameter CenterNet-style~\cite{centernet} detector head, and a novel lightweight expectation-maximization module that leverages latent commonality of the ground truth bounding box annotations.
- **Location:** PDF p.1, Abstract, L col para 1
- **Quote:** "We model an object as a single point — the center point of its bounding box. Our detector uses keypoint estimation to find center points and regresses to all other object properties, such as size, 3D location, orientation, and even pose. Our center point based approach, CenterNet, is end-to-end differentiable, simpler, faster, and more accurate than corresponding bounding box based detectors."
- **Verdict:** SUPPORTED

### paper.tex L242
> \item a) our model LACE, built on a DINOv3-web (ViT-L/16) backbone~\cite{dinov3}, with a trained CenterNet-style detector head~\cite{centernet} that identifies crown centers for detection based on the DINO output embeddings.
- **Location:** PDF p.1, §1 Introduction, R col para 1
- **Quote:** "We simply feed the input image to a fully convolutional network [37, 40] that generates a heatmap. Peaks in this heatmap correspond to object centers. Image features at each peak predict the objects bounding box height and weight."
- **Verdict:** SUPPORTED

### paper.tex L425
> This is based on the CenterNet design~\cite{centernet,cornernet}.
- **Location:** PDF p.3, §3 Preliminary, R col para 1 (heatmap Ŷ ∈ [0,1]^{W/R×H/R×C}) and §4 "Objects as Points", R col
- **Quote:** "Our aim is to produce a keypoint heatmap Ŷ ∈ [0, 1]^{W/R × H/R × C}" ... "We use our keypoint estimator Ŷ to predict all center points. In addition, we regress to the object size sk = (x2 − x1, y2 − y1) for each object k."
- **Verdict:** SUPPORTED

## emadapt — Weakly- and Semi-Supervised Learning of a DCNN for Semantic Image Segmentation (Papandreou 2015)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/emadapt.pdf` — from https://arxiv.org/pdf/1502.02734; arXiv v3 (5 Oct 2015); page numbers are arXiv pages (12 pp.)
**Bib check:** OK (title, 4 authors Papandreou/Chen/Murphy/Yuille; ICCV 2015 pages not printed on arXiv copy)

### paper.tex L249
> Unlike EM-Adapt~\cite{emadapt}, which amortises the E-step into a gradient-trained DCNN with implicit cross-box sharing, we fit an explicit closed-form commonality mixture over frozen features.
- **Location:** PDF p.3, §3.2 Image-level annotations, L col (Eq. 4–9, Algorithm 1) and R col para "EM-Adapt"; box variants PDF p.4, §3.3 Bounding Box Annotations, L col
- **Quote:** p.3 L: "where we adopt a hard-EM approximation, estimating in the E-step of the algorithm the latent segmentation by ŷ = argmax_y P(y|x;θ')P(z|y)" … "In the M-step of the algorithm, we optimize Q(θ;θ') ≈ log P(ŷ|x;θ) by mini-batch SGD similarly to (1), treating ŷ as ground truth segmentation." p.3 R: "EM-Adapt … we adaptively set the image- and class-dependent biases b_l so as the prescribed proportion of the image area is assigned to the background or foreground object classes." p.4: "Our third Bbox-EM-Fixed method is an EM algorithm … a variant of the EM-Fixed algorithm in Sec. 3.2, in which we boost the present foreground object scores only within the bounding box area."
- **Verdict:** PARTIAL
- **Note:** EM-Adapt is defined for image-level labels (§3.2), not boxes; the box-supervised EM variant is Bbox-EM-Fixed (§3.3). The E-step is an explicit per-pixel argmax of the DCNN's current scores plus an adaptive cardinality bias (hard-EM), and only the M-step is gradient training; "amortised into a DCNN" is a fair gloss (the network's own scores define the E-step) but "cross-box sharing" is only via the shared weights θ and the paper never frames it that way. Suggest "Unlike the EM approach of Papandreou et al. (EM-Adapt / Bbox-EM-Fixed), whose E-step is a per-pixel argmax over the current network's scores and whose only cross-image coupling is the shared weights…".

## discobox — DiscoBox: Weakly Supervised Instance Segmentation and Semantic Correspondence from Box Supervision (Lan 2021)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/discobox.pdf` — from https://arxiv.org/pdf/2105.06464; arXiv v2 (5 Jun 2021); page numbers are arXiv pages (13 pp.)
**Bib check:** OK (title, 8 authors Lan…Anandkumar; ICCV 2021 pages not printed on arXiv copy)

### paper.tex L249
> We exploit the similarities between signatures of composite embeddings within ground truth bounding boxes, following previous works such as DiscoBox~\cite{discobox}, DDT~\cite{ddt}, and STEGO~\cite{stego}.
- **Location:** PDF p.2, §1 Introduction, R col, para 2 ("Teacher model."); formalised PDF p.4, §3.2 Structured teacher, R col (cross-image potential τc with soft assignment T^{ns} between box proposals r^n, r^s)
- **Quote:** "The teacher is defined by a Gibbs energy which comprises a unary potential, a pairwise potential and a cross-image potential. The unary potential takes the initial output from the student whereas the pairwise and cross-image potentials model the pairwise pixel relationships both within and across bounding boxes."
- **Verdict:** SUPPORTED
- **Note:** In DiscoBox the within-box pairwise term is contrast-sensitive (colour) smoothness; the feature-embedding similarity ("cost volume matrix which models the appearance similarity", p.4) is used across box pairs (RoI features of r^n vs r^s). Fine as a "following" citation.

## ddt — Deep Descriptor Transforming for Image Co-Localization (Wei 2017)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/ddt.pdf` — from https://www.ijcai.org/proceedings/2017/0425.pdf; journal (IJCAI-17 proceedings) version, printed pp. 3048–3054 (PDF p.N = printed p.3047+N). arXiv v1 copy kept as `ddt_arxiv.pdf`.
**Bib check:** DISCREPANCY — proceedings list 7 authors: Xiu-Shen Wei, Chen-Lin Zhang, Yao Li, Chen-Wei Xie, Jianxin Wu, Chunhua Shen, Zhi-Hua Zhou; bib omits Yao Li and Chen-Wei Xie. Title, venue, pages 3048–3054, year OK.

### paper.tex L249
> We exploit the similarities between signatures of composite embeddings within ground truth bounding boxes, following previous works such as DiscoBox~\cite{discobox}, DDT~\cite{ddt}, and STEGO~\cite{stego}.
- **Location:** PDF p.3 (printed p.3050), §3.3 Deep Descriptor Transforming (DDT), L col para 1 and R col para 2 (after Eq. 4); also Abstract p.1 (printed p.3048)
- **Quote:** "What distinguishes DDT from SCDA is that we can leverage the correlations beneath the whole image set, instead of a single image." … "Because ξ1 is obtained through all N images, the positive correlation could indicate the common characteristic through N images. Specifically, in the image co-localization scenario, the corresponding positive correlation indicates indeed the common object inside these images."
- **Verdict:** SUPPORTED
- **Note:** DDT does PCA on descriptors pooled over whole images (no boxes); the "within GT boxes" part is the manuscript's own extension.

## stego — Unsupervised Semantic Segmentation by Distilling Feature Correspondences (Hamilton 2022)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/stego.pdf` — from https://arxiv.org/pdf/2203.08414; arXiv v1 (16 Mar 2022) = ICLR 2022 camera-ready ("Published as a conference paper at ICLR 2022" header); page numbers are arXiv pages (26 pp.)
**Bib check:** OK (title, 5 authors Hamilton/Zhang/Hariharan/Snavely/Freeman, ICLR 2022)

### paper.tex L249
> We exploit the similarities between signatures of composite embeddings within ground truth bounding boxes, following previous works such as DiscoBox~\cite{discobox}, DDT~\cite{ddt}, and STEGO~\cite{stego}.
- **Location:** PDF p.3, §3.1 Feature Correspondences Predict Class Co-occurrence, para 1, Eq. (1); also Abstract p.1
- **Quote:** "We form the feature correspondence tensor: F_hwij := Σ_c f_chw g_cij / (|f_hw||g_ij|), whose entries represent the cosine similarity between the feature at spatial position (h, w) of feature tensor f and position (i, j) of feature tensor g." Abstract: "current unsupervised feature learning frameworks already generate dense features whose correlations are semantically consistent."
- **Verdict:** SUPPORTED
- **Note:** STEGO is unsupervised (no boxes); it distils DINO feature cosine-similarities within and across images.

## jepa — A Path Towards Autonomous Machine Intelligence, Version 0.9.2 (LeCun 2022)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/jepa.pdf` — OpenReview direct download blocked (403/bot check); obtained from Wayback Machine copy of https://openreview.net/pdf?id=BZ5a1r-kVsf (https://web.archive.org/web/2023id_/…); 62 pp., printed page numbers = PDF page numbers
**Bib check:** OK (title "A Path Towards Autonomous Machine Intelligence, Version 0.9.2, 2022-06-27", Yann LeCun, June 27 2022, OpenReview)

### paper.tex L249
> In our model however, we utilise a World Model/JEPA~\cite{jepa} style approach by eschewing pixel-space calculations in favour of strictly latent space analysis.
- **Location:** PDF p.24 (printed 24), §4.4 Joint Embedding Predictive Architecture (JEPA), para after Eq. (13); repeated in Fig. 12 caption p.25; also p.27 §4.5 (para before Fig. 14) and p.7 §3 para 1
- **Quote:** p.24: "The main advantage of JEPA is that it performs predictions in representation space, eschewing the need to predict every detail of y. This is enabled by the fact that the encoder of y may choose to produce an abstract representation from which irrelevant details have been eliminated." p.27: "The ability of the JEPA to predict in representation space makes it considerably preferable to generative models that directly produce a prediction of y. In a video prediction scenario, it is essentially impossible to predict every pixel value of every future frame." p.7: "The predictions are performed within an abstract representation space that contains information relevant to the task at hand."
- **Verdict:** SUPPORTED

## zhu2017 — Deep Learning in Remote Sensing: A Comprehensive Review (Zhu 2017)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/zhu2017.pdf` — from https://arxiv.org/pdf/1710.03959; arXiv v1 (pre-acceptance, titled "Deep Learning in Remote Sensing: A Review"); page numbers are arXiv pages (printed page numbers in header match PDF pages). IEEE GRSM 5(4):8–36 version is paywalled.
**Bib check:** OK for the published version (IEEE GRSM 2017, 5(4):8–36; 7 authors, Zhu first). arXiv title is the shorter pre-acceptance title.

### paper.tex L255
> Consequently, in recent years there has been a focus in the RS community on RGB datasets which can directly leverage Computer Vision deep learning models~\cite{zhu2017} (SelvaBox refs: Weinstein 2021 NeonTreeEval) especially in the use of Convolutional Neural Networks (CNNs) and Vision Transformers (ViT)~\cite{vit}.
- **Location:** PDF p.18 (printed p.18), §III-C "Scene Classification" (paragraph "Using pre-trained networks", first bullet); also PDF p.24 (printed p.24) §III-D, "networks trained on color RGB data (fine tuned from existing architectures)"
- **Quote:** "Using pre-trained networks. The pre-trained deep CNN on the natural image dataset, e.g., OverFeat [88], GoogLeNet [89], etc., have led to impressive results on scene classification of high-resolution satellite images by directly extracting the features from the intermediate layers to form global feature representations [81–83, 87]."
- **Verdict:** PARTIAL
- **Note:** The review documents that natural-image-pretrained CNNs are applied directly to high-resolution (RGB) satellite scenes, but nowhere states a community "focus on RGB datasets"; its framing (PDF p.2, §I bullet 1) is the opposite emphasis — "Remote sensing data are often multi-modal, e.g. from optical (multi- and hyperspectral) and synthetic aperture radar (SAR) sensors". Soften to "natural-image-pretrained CNNs have been applied directly to high-resolution RGB RS imagery" or cite a TCD-specific source (e.g. the SelvaBox/NeonTreeEval refs already noted).

## vit — An Image Is Worth 16x16 Words (Dosovitskiy 2021)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/vit.pdf` — from https://arxiv.org/pdf/2010.11929; arXiv v2 (header "Published as a conference paper at ICLR 2021"); page numbers are arXiv pages.
**Bib check:** OK (Dosovitskiy et al., 12 authors; ICLR 2021; arXiv:2010.11929).

### paper.tex L255
> Consequently, in recent years there has been a focus in the RS community on RGB datasets which can directly leverage Computer Vision deep learning models~\cite{zhu2017} (SelvaBox refs: Weinstein 2021 NeonTreeEval) especially in the use of Convolutional Neural Networks (CNNs) and Vision Transformers (ViT)~\cite{vit}.
- **Location:** PDF p.1, Title + Abstract, single column
- **Quote:** "We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. ... Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks"
- **Verdict:** SUPPORTED
- **Note:** Cited only as the ViT origin paper; it says nothing about remote sensing.

## yolo — You Only Look Once (Redmon 2016)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/yolo.pdf` — from https://openaccess.thecvf.com/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf; CVPR 2016 camera-ready (printed pp.779–788). arXiv also saved as `yolo_arxiv.pdf`.
**Bib check:** OK (Redmon, Divvala, Girshick, Farhadi; 4 authors; CVPR 2016; pp.779–788).

### paper.tex L261
> YOLO family~\cite{yolo}, and Semantic Segmentation (TCS) models which aim to categorise each pixel into a class.
- **Location:** PDF p.1 (printed p.779), Title + Abstract, L col para 1
- **Quote:** "We present YOLO, a new approach to object detection. ... A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation."
- **Verdict:** SUPPORTED

### paper.tex L279
> One-stage detectors include YOLO (`You Only Look Once') series of models~\cite{yolo,yolov4}, RetinaNet~\cite{retinanet} and EfficientDet~\cite{efficientdet}.
- **Location:** PDF p.1 (printed p.779), Abstract, L col para 1
- **Quote:** "A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance."
- **Verdict:** SUPPORTED
- **Note:** The paper says "single network"/"unified", not the phrase "one-stage"; the categorisation is standard (e.g. RetinaNet §1 lists YOLO among one-stage detectors).

## goldblum2023 — Battle of the Backbones (Goldblum 2023)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/goldblum2023.pdf` — from https://arxiv.org/pdf/2310.19909; arXiv v2 (NeurIPS 2023 camera-ready format, footer "37th Conference on Neural Information Processing Systems (NeurIPS 2023)"); page numbers are arXiv pages.
**Bib check:** OK (Goldblum, Souri, ... Goldstein; 13 authors; NeurIPS 2023). Proceedings pages 29343–29371 not printed on PDF.

### paper.tex L270
> \item a feature extractor network (`backbone') which is a neural network that scans with convolutions to pick up on features in the image, with simpler features typically extracted earlier in the network's layers and more complex composite features requiring greater depth~\cite{goldblum2023};
- **Location:** PDF p.1, Abstract, para 1 (only the "backbone = feature extractor" definition)
- **Quote:** "Neural network based computer vision systems are typically built on a backbone, a pretrained or randomly initialized feature extractor."
- **Verdict:** NOT SUPPORTED
- **Note:** Searched full text for "earlier layer", "low-level", "simple(r) features", "hierarch", "edges", "composite", "shallow", "depth" — no passage on simple-to-complex feature hierarchies across depth; the paper is a benchmark of pretrained backbones. It backs only "backbone = feature extractor". The hierarchy claim belongs to Zeiler & Fergus 2014, "Visualizing and Understanding Convolutional Networks" (ECCV; §4.1: layer 2 corners/edges → layer 5 entire objects) and Yosinski et al. 2014, "How transferable are features in deep neural networks?" (NeurIPS; first-layer Gabor/colour-blob features general, later layers specific); LeCun, Bengio & Hinton 2015 (Nature "Deep learning") is the standard review citation. Also "scans with convolutions" is CNN-specific and excludes the ViT backbone the manuscript itself uses.

## soviany2018 — Optimizing the Trade-off between Single-Stage and Two-Stage Detectors (Soviany 2018)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/soviany2018.pdf` — from https://arxiv.org/pdf/1803.08707; arXiv v3; page numbers are arXiv pages (SYNASC pp.209–214 not printed).
**Bib check:** OK (Soviany, Ionescu; 2 authors; title matches). Venue SYNASC 2018 not printed on arXiv version; consistent with IEEE DOI 10.1109/SYNASC.2018.00041.

### paper.tex L273
> Two-stage detectors first utilise a module to create many possible bounding boxes which are region proposals, before the second stage module extracts features from the proposals~\cite{soviany2018}.
- **Location:** PDF p.1, Abstract (L col, sentence 2); repeated verbatim in §I Introduction, L col para 2
- **Quote:** "On one hand, we have two-stage detectors, such as Faster R-CNN (Region-based Convolutional Neural Networks) [1] or Mask R-CNN [2], that (i) use a Region Proposal Network (RPN) to generate regions of interests in the first stage and (ii) send the region proposals down the pipeline for object classification and bounding-box regression."
- **Verdict:** SUPPORTED
- **Note:** Source says the second stage does "object classification and bounding-box regression" on the proposals; the manuscript's "extracts features from the proposals" is a fair paraphrase (RoI feature pooling), not the source's wording.

## rcnn — Rich Feature Hierarchies (R-CNN) (Girshick 2014)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/rcnn.pdf` — from https://openaccess.thecvf.com/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf; CVPR 2014 camera-ready (CVF file has no printed 580–587 numbering; PDF p.1 = printed p.580). arXiv v5 also saved as `rcnn_arxiv.pdf`.
**Bib check:** OK (Girshick, Donahue, Darrell, Malik; 4 authors; CVPR 2014).

### paper.tex L279
> Within TCD, widely used two-stage detector CNNs include the Region-based CNN (R-CNN) family of models~\cite{rcnn} -- current best-in-class models include Fast R-CNN, Faster R-CNN and Mask R-CNN.
- **Location:** PDF p.1 (printed p.580), Abstract L col + Fig. 1 caption R col; also PDF p.2 (printed p.581), §2 "Object detection with R-CNN", L col para 1
- **Quote:** "Our system (1) takes an input image, (2) extracts around 2000 bottom-up region proposals, (3) computes features for each proposal using a large convolutional neural network (CNN), and then (4) classifies each region using class-specific linear SVMs." / "Since our system combines region proposals with CNNs, we dub the method R-CNN: Regions with CNN features."
- **Verdict:** PARTIAL
- **Note:** Source supports R-CNN as the origin of the region-proposal-then-CNN family only. Fast R-CNN (Girshick, ICCV 2015), Faster R-CNN (Ren et al., NeurIPS 2015) and Mask R-CNN (He et al., ICCV 2017) post-date this 2014 paper and are not mentioned in it; the "current best-in-class" clause needs its own citations (Mask R-CNN is already in the bib as `maskrcnn`).

## yolov4 — YOLOv4: Optimal Speed and Accuracy (Bochkovskiy 2020)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/yolov4.pdf` — from https://arxiv.org/pdf/2004.10934; arXiv v1; page numbers are arXiv pages.
**Bib check:** OK (Bochkovskiy, Wang, Liao; 3 authors; arXiv:2004.10934, 2020).

### paper.tex L279
> One-stage detectors include YOLO (`You Only Look Once') series of models~\cite{yolo,yolov4}, RetinaNet~\cite{retinanet} and EfficientDet~\cite{efficientdet}.
- **Location:** PDF p.1, Title + Abstract L col, and Fig. 1 caption R col
- **Quote:** "combine some of them to achieve state-of-the-art results: 43.5% AP (65.7% AP50) for the MS COCO dataset at a real-time speed of ∼65 FPS on Tesla V100." / "Improves YOLOv3's AP and FPS by 10% and 12%, respectively."
- **Verdict:** SUPPORTED

## retinanet — Focal Loss for Dense Object Detection (Lin 2017)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/retinanet.pdf` — from https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf; ICCV 2017 camera-ready (printed pp.2980–2988). arXiv also saved as `retinanet_arxiv.pdf`.
**Bib check:** OK (Lin, Goyal, Girshick, He, Dollár; 5 authors; ICCV 2017; pp.2980–2988).

### paper.tex L279
> One-stage detectors include YOLO (`You Only Look Once') series of models~\cite{yolo,yolov4}, RetinaNet~\cite{retinanet} and EfficientDet~\cite{efficientdet}.
- **Location:** PDF p.1 (printed p.2980), Fig. 2 caption, R col; also Abstract L col
- **Quote:** "Enabled by the focal loss, our simple one-stage RetinaNet detector outperforms all previous one-stage and two-stage detectors, including the best reported Faster R-CNN [27] system from [19]."
- **Verdict:** SUPPORTED

## efficientdet — EfficientDet: Scalable and Efficient Object Detection (Tan 2020)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/efficientdet.pdf` — from https://openaccess.thecvf.com/content_CVPR_2020/papers/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.pdf; CVPR 2020 camera-ready (printed pp.10781–10790). arXiv also saved as `efficientdet_arxiv.pdf`.
**Bib check:** OK (Tan, Pang, Le; 3 authors; CVPR 2020; pp.10781–10790).

### paper.tex L279
> One-stage detectors include YOLO (`You Only Look Once') series of models~\cite{yolo,yolov4}, RetinaNet~\cite{retinanet} and EfficientDet~\cite{efficientdet}.
- **Location:** PDF p.2 (printed p.10782), §2 Related Work "One-Stage Detectors", L col para 1; also PDF p.1 §1 L col ("the one-stage detector design")
- **Quote:** "Existing object detectors are mostly categorized by whether they have a region-of-interest proposal step (two-stage [9, 32, 3, 11]) or not (one-stage [33, 24, 30, 21]). ... In this paper, we mainly follow the one-stage detector design"
- **Verdict:** SUPPORTED

## fgtreeseg — FG-TreeSeg: Flow-Guided Tree Crown Segmentation without Instance Annotations (Chen 2026)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/fgtreeseg.pdf` — from https://arxiv.org/pdf/2602.00470; arXiv v2 (16 May 2026); page numbers are arXiv pages (1–5). arXiv v1 (Feb 2026) was titled "ZS-TreeSeg: A Zero-Shot Framework for Tree Crown Instance Segmentation", hence the dual name.
**Bib check:** Title/4 authors (Chen, Lyu, Wang, Wang)/arXiv id OK. Venue "IEEE Geosci. Remote Sens. Lett." is NOT stated anywhere in the PDF (only in a reference to another paper) — unverified; treat as arXiv preprint unless publication confirmed.

### paper.tex L285
> ZS-TreeSeg/FG-TreeSeg~\cite{fgtreeseg} also models tree crowns as star-convex objects, and introduces a vector field flow-based crown separation mechanism to improve instance segmentation of touching tree crowns and dense canopy.
- **Location:** PDF p.1, Abstract; PDF p.2, R col, §II.B "Flow-guided Instance Segmentation Framework", para 1
- **Quote:** "By modeling tree crowns as star-convex objects within a topological flow field using Cellpose-SAM, the FG-TreeSeg framework forces the separation of touching tree crown instances based on vector convergence." (p.1); "leveraging the geometric property that biological cells and tree crowns share a quasi-star-convex structure [15], [17]. This inductive bias is particularly effective for separating dense, touching crowns" (p.2)
- **Verdict:** SUPPORTED
- **Note:** The flow mechanism is Cellpose-SAM's, adopted training-free ("adapts the Cellpose-SAM architecture"); "introduces" slightly overstates — they transfer it, not invent it.

### paper.tex L354
> FG-TreeSeg \cite{fgtreeseg} & Semantic$^{\ddagger}$ & RGB & Cellpose & 10 & \xmark & \cmark & \xmark \\
- **Location:** Supervision: PDF p.1 Abstract + p.1 R col §II.A; 2nd net: PDF p.1 R col §II (last para) and p.2 §II.A/B; Input/GSD/biome: PDF p.3 R col §III.B and p.4 Table I
- **Quote:** "we propose FG-TreeSeg, a training-free framework" (p.1); "a SegFormer model (MiT-B5 backbone) trained on the OAM-TCD ... dataset [16] is used to perform binary semantic segmentation of tree canopies" (p.1); "we employ a semantic prior (SegFormer) to spatially regularize Cellpose-SAM" (p.1); "benchmark evaluation across two diverse datasets: NEON [18] (aerial RGB) and BAMFORESTS [19] (UAV VHR)" (p.3)
- **Verdict:** PARTIAL
- **Note:** Supervision = semantic only ✓ (SegFormer on OAM-TCD semantic masks; no tree instance labels). Input RGB ✓. 2nd net: the method needs TWO extra pretrained networks at inference — SegFormer (semantic) AND Cellpose-SAM — the row lists only "Cellpose". GSD "10": no GSD is stated anywhere in the paper; NEON is 10 cm but BAMFORESTS is "UAV VHR" (sub-5 cm), so a range would be more accurate. Biomes: NEON (USA) and BAMFORESTS (Bamberg, Germany) reported separately (Table I Panels A/B, p.4) — both temperate, so Te ●, Tr ×, Sv × ✓.

## detectron2 — Detectron2 (Wu 2019)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/detectron2_README.md` and `detectron2_MODEL_ZOO.md` — from https://raw.githubusercontent.com/facebookresearch/detectron2/main/README.md and .../MODEL_ZOO.md (software; HTML/Markdown, no PDF).
**Bib check:** OK — matches the repo's own BibTeX (README §"Citing Detectron2": Wu, Kirillov, Massa, Lo, Girshick; 2019; same URL).

### paper.tex L295
> Detectron2~\cite{detectron2} is the object detection and segmentation library on which several of the tree crown baselines in this work are built, including Detectree2 and Restor's Mask R-CNN; it supplies the Mask R-CNN, RetinaNet and Faster R-CNN reference implementations together with the training, inference and evaluation harness used to fine-tune them. %TBC expand
- **Location:** README para 1 and §"Model Zoo and Baselines"; MODEL_ZOO.md §"COCO Object Detection Baselines" › "Faster R-CNN:", "RetinaNet:", and §"COCO Instance Segmentation Baselines with Mask R-CNN"; MODEL_ZOO.md §"How to Read the Tables" (train/eval harness)
- **Quote:** "Detectron2 is Facebook AI Research's next generation library that provides state-of-the-art detection and segmentation algorithms." / "Models can be reproduced using `tools/train_net.py` with the corresponding yaml config file" / "Inference speed is measured by `tools/train_net.py --eval-only`, or inference_on_dataset()"
- **Verdict:** SUPPORTED
- **Note:** The "Detectree2 and Restor's Mask R-CNN are built on it" half of the sentence is backed by those papers/repos, not by Detectron2's README.

## sam3 — SAM 3: Segment Anything with Concepts (Carion 2025)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/sam3.pdf` — from https://arxiv.org/pdf/2511.16719; arXiv v2 (28 Mar 2026; v1 Nov 2025); page numbers are arXiv pages (78 pp.)
**Bib check:** DISCREPANCY — author list is wrong. PDF p.1 co-first authors: Nicolas Carion*, Laura Gustafson*, Yuan-Ting Hu*, Shoubhik Debnath*, Ronghang Hu*, Didac Suris*, Chaitanya Ryali*, Kalyan Vasudev Alwala*, Haitham Khedr* …; project leads Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, Christoph Feichtenhofer; affiliation "Meta Superintelligence Labs". Wan-Yen Lo, Eric Mintun, Chloe Rolland and Ross Girshick (in bib) are NOT authors of SAM 3 (they are SAM 1 authors). Title, arXiv id, year 2025 OK.

### paper.tex L299
> Meta's Segment Anything Model (SAM)~\cite{sam}, now in its third iteration~\cite{sam3}, is a promptable class-agnostic segmentation foundation model: given a point, box or mask prompt it emits an instance mask, which makes it a natural box-to-mask module for detectors that emit only rectangles, and it is used in that role by several of the baselines compared here. %TBC expand
- **Location:** PDF p.1, title and abstract ("We present Segment Anything Model (SAM) 3 …"); box prompts: PDF p.3, §3 Model, para 1; PDF p.20, §C.1 Model Architecture, "Geometry and Exemplar Encoder"; evaluation PDF p.9 Table 7 and p.50 §F.6
- **Quote:** p.3: "SAM 3 is a generalization of SAM 2, supporting the new PCS task (§2) as well as the PVS task. It takes concept prompts (simple noun phrases, image exemplars) or visual prompts (points, boxes, masks) to define the objects to be (individually) segmented spatio-temporally." p.20: "It is additionally used to encode visual prompts for the PVS task on images as an auxiliary functionality that is primarily used to include pre-training data for the PVS task in stages-2,-3 of training"
- **Verdict:** SUPPORTED
- **Note:** "Third iteration" — title page suffices. Box prompts: yes, SAM 3 still accepts box prompts on images (paper §3 lists points/boxes/masks; GitHub README: "text or visual prompts such as points, boxes, and masks", notebook `sam3_image_predictor_example.ipynb` "prompt SAM 3 with text and visual box prompts on images"). Caveats: the paper calls point/box PVS on images an "auxiliary functionality" (p.20), its image PVS benchmark (Table 7, SA-37) is evaluated with 1/3/5 clicks only, and its interactive refinement text (p.5) describes clicks; box-prompt mask quality on images is not separately reported. Also note the SA-Co/Bronze+Bio GT masks were "generated by using boxes as prompts to SAM 2" (p.7), i.e. the box-to-mask role is precedented with SAM 2, not benchmarked for SAM 3. Safer phrasing: "SAM 3 retains SAM 2's point/box/mask visual prompting alongside the new concept prompts."

## clip — Learning Transferable Visual Models From Natural Language Supervision (Radford 2021)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/clip.pdf` — from https://arxiv.org/pdf/2103.00020; arXiv v1 (26 Feb 2021); page numbers are arXiv pages (no printed PMLR pagination)
**Bib check:** OK (title, first author Alec Radford, 12 authors; ICML 2021 / PMLR 139:8748–8763 not printed on arXiv copy)

### paper.tex L306
> Utilising SSL facilitates training regimes on orders of magnitude more image data since weakly and fully supervised methods~\cite{clip,vit22b,perceptionencoder} require high-quality labels and/or metadata for the training images.
- **Location:** PDF p.1, Abstract, L col; PDF p.3, §2.2 Creating a Sufficiently Large Dataset, R col
- **Quote:** "We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet." (p.1)
- **Verdict:** SUPPORTED
- **Note:** CLIP = weakly supervised via image–text metadata (captions); the grouping "weakly and fully supervised ... require ... labels and/or metadata" mirrors DINOv3 p.1 Intro verbatim ("Unlike weakly and fully supervised pretraining methods (Radford et al., 2021; Dehghani et al., 2023; Bolya et al., 2025) which require images paired with high-quality metadata"). CLIP itself frames its point as needing *less* curated labelling than ImageNet-style datasets, so "high-quality" is a stretch for CLIP.

## vit22b — Scaling Vision Transformers to 22 Billion Parameters (Dehghani 2023)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/vit22b.pdf` — from https://arxiv.org/pdf/2302.05442; arXiv v1 (10 Feb 2023); printed page numbers = PDF pages (ICML/PMLR 7480–7512 not verifiable from arXiv)
**Bib check:** OK (title, first author Mostafa Dehghani, 43 authors, Google Research, 2023)

### paper.tex L306
> Utilising SSL facilitates training regimes on orders of magnitude more image data since weakly and fully supervised methods~\cite{clip,vit22b,perceptionencoder} require high-quality labels and/or metadata for the training images.
- **Location:** PDF p.5 (printed p.5), §4.1 Training details, para "Dataset."
- **Quote:** "ViT-22B is trained on a version of JFT (Sun et al., 2017), extended to around 4B images (Zhai et al., 2022a). These images have been semi-automatically annotated with a class-hierarchy of 30k labels. ... we flatten the hierarchical label structure and use all the assigned labels in a multi-label classification fashion employing the sigmoid cross-entropy loss."
- **Verdict:** SUPPORTED
- **Note:** Fully supervised (multi-label classification on JFT-4B); labels are "semi-automatically annotated", so "high-quality" is the manuscript's characterisation, not the source's.

## perceptionencoder — Perception Encoder (Bolya 2025)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/perceptionencoder.pdf` — from https://arxiv.org/pdf/2504.13181; arXiv v2 (28 Apr 2025); printed page numbers = PDF pages
**Bib check:** OK (title "Perception Encoder: The best visual embeddings are not at the output of the network", first author Daniel Bolya, 18 authors, Meta FAIR, 2025)

### paper.tex L306
> Utilising SSL facilitates training regimes on orders of magnitude more image data since weakly and fully supervised methods~\cite{clip,vit22b,perceptionencoder} require high-quality labels and/or metadata for the training images.
- **Location:** PDF p.1 (printed p.1), Abstract; PDF p.3 (printed p.3), §2.1 Robust Image Pretraining, para 1 and "Setup." para
- **Quote:** "we find that contrastive vision-language training alone can produce strong, general embeddings for all of these downstream tasks" (p.1); "In the first stage of pretraining, we want to learn as much visual information as possible from a large set of image-text data. ... train on a fixed 2.3B image-text dataset curated using the MetaCLIP [152] text-only curation" (p.3)
- **Verdict:** SUPPORTED
- **Note:** Weakly supervised via image–text pairs (CLIP-style contrastive); requires text metadata, not labels.

## ibot — iBOT: Image BERT Pre-Training with Online Tokenizer (Zhou 2022)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/ibot.pdf` — from https://arxiv.org/pdf/2111.07832; arXiv v3 (27 Jan 2022), header "Published as a conference paper at ICLR 2022"; printed page numbers = PDF pages
**Bib check:** OK (title, first author Jinghao Zhou, 7 authors, ICLR 2022)

### paper.tex L306
> Incorporating both a global image-level objective and a local iBOT-style~\cite{ibot} patch-level latent reconstruction objective in the loss function results in a model that excels at encoding global and local features.
- **Location:** PDF p.1 (printed p.1) Abstract; PDF p.3 (printed p.3) §3.1 Framework, Eq. (3) and following para; PDF p.4 Figure 3 caption
- **Quote:** "Specifically, we perform self-distillation on masked patch tokens and take the teacher network as the online tokenizer, along with self-distillation on the class token to acquire visual semantics." (p.1); "The second loss LMIM is self-distillation between in-view patch tokens, with some tokens masked and replaced by e[MASK] for the student network. The objective is to reconstruct the masked tokens with the teacher networks' outputs as supervision." (p.4, Fig. 3 caption)
- **Verdict:** SUPPORTED
- **Note:** iBOT calls it "masked image modeling (MIM)" via self-distillation on patch tokens; "patch-level latent reconstruction objective" is DINOv3's paraphrase (dinov3 p.9), which is fine.

## mask2former — Masked-Attention Mask Transformer (Cheng 2022)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/mask2former.pdf` — from https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.pdf; CVF camera-ready (printed pp.1290–1299)
**Bib check:** OK (title, 5 authors Cheng/Misra/Schwing/Kirillov/Girdhar, CVPR 2022, pp.1290–1299 printed)

### paper.tex L340
> Mask2Former \cite{mask2former} & Polygon & MS+LiD & -- & $<$5 & \xmark & \cmark & \xmark \\
- **Location:** PDF p.1 (printed p.1290), Abstract; datasets PDF p.4 (printed 1293) §4 para 1
- **Quote:** "We present Masked-attention Mask Transformer (Mask2Former), a new architecture capable of addressing any image segmentation task (panoptic, instance or semantic)." (p.1290); evaluated on "COCO [35] ... ADE20K [65] ... Cityscapes [16] ... Mapillary Vistas" (p.1293)
- **Verdict:** WRONG SOURCE
- **Note:** The CVPR paper contains no tree-crown, multispectral, LiDAR, GSD or biome content (grep for tree/crown/LiDAR/multispectral/forest: 0 hits) and cannot support any row attribute. The row attributes match Dersch, Schöttl, Krzystek & Heurich 2023, ISPRS Open J. Photogramm. Remote Sens. 8:100037 (DOI 10.1016/j.ophoto.2023.100037) — but only partly: its abstract (OpenAlex/Crossref; PDF INACCESSIBLE: ScienceDirect 403/captcha, NVA/Brage repository download 403) states the methods are "Mask R–CNN and DETR" (NOT Mask2Former), input "multispectral images and images generated from UAV lidar data" (Micasense RedEdge-MX + Riegl miniVUX-1UAV; MS ✓, LiDAR ✓), "ground resolution of 5 cm" (so "<5" should be "5"), site "in Bavaria, 35 km north of Munich (Germany), comprising a mixed forest stand of around 7 ha ... Norway spruce ... European beeches" (temperate ✓, Tr ×, Sv × ✓), supervision = 1408 visually-interpreted trees (polygon ✓), no second network at inference ✓. Either relabel the row "Mask R-CNN/DETR (Dersch et al. 2023)" with GSD 5 and cite Dersch, or, if Mask2Former on MS+LiDAR at <5 cm is meant, a different source is required (Teng et al. 2025 run Mask2Former but on RGB+DSM, not MS+LiDAR).

## crownvim — CrownViM: Context Clustering Meets Vision Mamba (Shi 2026)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/crownvim.pdf` — from https://mdpi-res.com/d_attachment/remotesensing/remotesensing-18-00860/article_deploy/remotesensing-18-00860.pdf; journal version (30 pp., printed "Remote Sens. 2026, 18, 860, N of 30")
**Bib check:** AUTHOR LIST WRONG. PDF p.1: "Erkang Shi, Ziyang Shi, Fulin Su, Lin Li, Ruifeng Liu, Fangying Wan and Kai Zhou" (7 authors; first author Erkang Shi, not Ziyang Shi; Xiong, Zhou Guoxiong, Yan, Zhao do not exist). Title, journal, year, vol 18, no. 6, art. 860, DOI OK.

### paper.tex L341
> CrownViM \cite{crownvim} & Polygon & RGB & -- & 10 & \omark & \omark & \omark \\
- **Location:** PDF p.6 (printed p.6 of 30), §3.1 "Dataset" (item (1) OAM-TCD, item (2) SSD), single column, first para; evaluation PDF p.18–19 (§4.4, Table 2)
- **Quote:** "It contains 5072 annotated images (2048 × 2048 pixels at 10 cm/pixel resolution) with instance masks delineating >280,000 individual trees and 56,000 tree clusters. Globally sampled across biomes including urban and forest landscapes" (p.6); "To systematically evaluate the proposed model, we conduct comparative experiments on the OAM-TCD and single-tree segmentation datasets." (p.18)
- **Verdict:** SUPPORTED
- **Note:** Polygon (instance masks) ✓, RGB ✓ (title), no 2nd net ✓ (single CCViM encoder + MaskFormer decoder), 10 cm ✓ (OAM-TCD). Biomes: results are aggregate over OAM-TCD (Table 2, p.19) with no per-biome stratum, so ○○○ is fair; the second dataset (SSD, Wuhan Univ., 731 images, GSD not stated, "coniferous, broad-leaved, and mixed forest types") is also not stratified by biome.

## sam2lidar — Leveraging SAM 2 and LiDAR for automated ITC delineation (Zhu 2025)
**PDF:** INACCESSIBLE — ScienceDirect PII S3050520825000259 (pdfft, cookies, r.jina.ai, WebFetch all 403/captcha); no arXiv/repository copy found (Semantic Scholar, OpenAlex, Unpaywall). Verified from the OpenAlex abstract.
**Bib check:** OK against Crossref (Zhu, Locke, Yuan, Zhang, Ma, Liang; Information Geography vol 1 no 2, art. 100025, 2025). Note Crossref gives the last author as "Lü Liang" vs bib "Lu Liang".

### paper.tex L349
> LiDAR-prompt SAM \cite{sam2lidar} & Box & RGB+LiD & SAM\,2 & 10 & \xmark & \cmark & \xmark \\
- **Location:** Abstract, sentences 4–6
- **Quote:** "We explore three methods, one manual and two automatic, of providing bounding box prompts to SAM 2 for ITC delineation. We also introduce a novel method of incorporating LiDAR data as both a bounding box filter and as point prompts to SAM 2 ... We evaluate these methods on a diverse subset of 592 trees from the NeonTreeEvaluation dataset"
- **Verdict:** PARTIAL (INACCESSIBLE for body)
- **Note:** Box prompts ✓, RGB+LiDAR ✓, SAM 2 ✓. GSD 10 cm not stated in abstract (NEON RGB is 10 cm, so plausible). Biome: NeonTreeEvaluation's sites are US NEON sites, which include oak-savanna/woodland (e.g. SJER) as well as temperate forest; whether the 592-tree subset spans these and whether results are stratified by site cannot be checked without the full text — Te ✓ only is unverified.

## alspseudo — Learning Image-Based Tree Crown Segmentation from Enhanced Lidar-Based Pseudo-Labels (Pesonen 2026)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/alspseudo.pdf` — from https://arxiv.org/pdf/2602.13022; arXiv v1 (13 Feb 2026); page numbers are arXiv pages (printed 1–21)
**Bib check:** OK (title, 6 authors Pesonen/Rua/Taher/Koivumäki/Yu/Honkavaara, arXiv:2602.13022, 2026)

### paper.tex L353
> ALS pseudo-labels \cite{alspseudo} & None$^{\dagger}$ & MS+ALS & SAM\,2 & -- & \xmark & \cmark & \xmark \\
- **Location:** Supervision: PDF p.1 Abstract; Input/GSD: PDF p.3 (printed p.3) §3.1 "Data", para 1; SAM 2 role: PDF p.5 §3.3 "SAM 2 enhancement" and §3.4 "Pseudo-supervised model"; Biome: PDF p.4 §3.1 para after Fig. 1; inference-only-images: PDF p.2 §1 last para
- **Quote:** "Our method offers a way to obtain domain-specific training annotations for optical image-based models without any manual annotation cost" (p.1); "an orthophoto with a spatial resolution of 5 cm collected in July 2023 ... only the five multispectral bands were used" (p.3); "We used the CHM and SAM 2-derived labels to train a downstream model which directly predicts individual tree crown masks from RGB or multispectral images" (p.5); "ranging from typical Fennoscandian coastal landscape to managed Finnish boreal forests" (p.4); "training data for tree crown instance segmentation models, which can be deployed in areas where only image data is available" (p.2)
- **Verdict:** PARTIAL
- **Note:** Supervision None† ✓ (no manual labels; ALS needed for pseudo-labels; the 362-tree test area is manually annotated for evaluation only). Input "MS+ALS" is a training-time description; at inference the Mask R-CNN takes RGB or RGB+NIR images only. 2nd net "SAM 2" ✗ under the table's own definition ("additional pretrained network required at inference"): SAM 2 is used only to refine pseudo-labels before training; inference is a plain Mask R-CNN. GSD "--" ✗: stated as 5 cm. Biome: single site in Espoo, Finland — boreal/hemiboreal, not temperate; "Te ✓" is a stretch (no boreal column), Tr ×/Sv × ✓.

## silva2016 — Imputation of individual longleaf pine tree attributes from field and LiDAR data (Silva 2016)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/silva2016.pdf` — from https://www.fs.usda.gov/rm/pubs_journals/2016/rmrs_2016_silva_c002.pdf (USDA RMRS author copy); journal version (printed pages 554–573; PDF p.N = printed p.553+N)
**Bib check:** OK (title, 11 authors, Silva first, Can. J. Remote Sens. 42(5):554–573, 2016, DOI matches)

### paper.tex L393
> The DeepForest model released alongside the NEONTreeEvaluation benchmark~\cite{neontreeeval} was pretrained on all 22 sites, using an unsupervised LiDAR-based algorithm~\cite{silva2016} to generate millions of moderate-quality annotations, and was then fine-tuned on a further 10,000 manual annotations of RGB imagery from six sites.
- **Location:** PDF p.1 (printed 554) Abstract; PDF p.5 (printed 558) R col, §"Individual Tree Detection and HMAX Extraction"; PDF p.6 (printed 559) L col, §"Individual Tree Crown Delineation and Crown Area Computation"
- **Quote:** "(i) individual tree detection, crown delineation, and tree-level-based metrics computation from LiDAR-derived CHM" / "The FindTreesCHM function uses a local maximum algorithm to search for treetops in the CHM through a moving window with a fixed treetop window size" / "Tree-crown delineation was also performed in R, using the ForestCAS function from the rLiDAR package"
- **Verdict:** SUPPORTED
- **Note:** Silva 2016 is correctly the LiDAR local-maximum + crown-delineation algorithm; the word "unsupervised" and the use for generating DeepForest pretraining labels are from weinstein2020 (bioRxiv p.4 line 73; p.26 line 455: "a pool of unsupervised LiDAR-based tree predictions generated using Silva et al (2016)").

## cornernet — CornerNet: Detecting Objects as Paired Keypoints (Law 2018)
**PDF:** `/Users/tompitts/dphil/refs/lace_bib_pdfs/cornernet.pdf` — from https://openaccess.thecvf.com/content_ECCV_2018/papers/Hei_Law_CornerNet_Detecting_Objects_ECCV_2018_paper.pdf; ECCV 2018 camera-ready (17 pp., no printed LNCS page numbers; = LNCS 11218 pp.765–781). arXiv also saved as `cornernet_arxiv.pdf`.
**Bib check:** OK (Law, Deng; 2 authors; ECCV 2018; LNCS 11218).

### paper.tex L425
> This is based on the CenterNet design~\cite{centernet,cornernet}.
- **Location:** PDF p.2 (printed p.766), §1 Introduction, para 3 (single column); also PDF p.1 Abstract
- **Quote:** "We detect an object as a pair of keypoints—the top-left corner and bottom-right corner of the bounding box. We use a single convolutional network to predict a heatmap for the top-left corners of all instances of the same object category, a heatmap for all bottom-right corners, and an embedding vector for each detected corner."
- **Verdict:** SUPPORTED
- **Note:** CornerNet is the keypoint-heatmap predecessor CenterNet builds on (CenterNet §2 "CornerNet [30] detects two bounding box corners as keypoints"); it predicts corner heatmaps, not centre heatmaps — wording "CenterNet design" is accurate for centernet, "keypoint-heatmap design" would cover both.
