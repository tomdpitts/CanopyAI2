# Zero-shot PhenoBench — LACE box-to-mask on ground-truth plant boxes

**Self-contained.** This folder holds its own copy of the LACE port, its data, its parity gates and its results. It imports nothing from other experiment folders; it only reads the repo itself (source-identity checks and the deployed `em_model_4p_fix.npz`), read-only. To remove it, delete the folder.

**Question.** Does LACE's box-to-mask module carry over beyond tree crowns? It is tested on crop and weed plants in PhenoBench v1.1.0 (drone RGB of sugar beet fields), given tight ground-truth boxes. There is no detector, so no AP.

## Runs

| run | images | status |
|---|---|---|
| `sample60` | 60 val images, seed 0, 20 per acquisition date | done 2026-11-13; superseded by `val` |
| `val` | all 772 val images (the labelled split; test labels are withheld) | **done 2026-11-13; the result of record** |

## Result: val, all 772 images (result of record)

Mean per-instance mask IoU on identical tight GT boxes. Headline subset: 8,568 non-partial plants (4,967 crop, 3,601 weed); 10,408 instances incl. partial. GT fill 0.41. Size tercile edges 31.5 / 104.0 px box √area. Source: `results/val/summary.json`, `logs/val/summarise.log`.

| arm | non-partial | crop | weed | 05-15 | 05-26 | 06-05 | T1 | T2 | T3 | incl. partial | fill |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LACE per-image test-time fit (α 0.3) | 0.580 | 0.581 | 0.577 | 0.609 | 0.582 | 0.535 | 0.635 | 0.558 | 0.546 | 0.590 | 0.61 |
| LACE test-time fit, α = 0 (sensitivity) | 0.589 | 0.592 | 0.585 | 0.618 | 0.593 | 0.544 | 0.636 | 0.572 | 0.559 | 0.601 | 0.60 |
| **LACE tree-fitted, as-is** | **0.605** | 0.631 | 0.569 | 0.610 | 0.621 | 0.585 | 0.623 | 0.542 | 0.650 | 0.617 | 0.62 |
| SAM 3 zero-shot (uncropped) | 0.719 | 0.768 | 0.650 | 0.680 | 0.688 | 0.801 | 0.584 | 0.706 | 0.866 | 0.736 | 0.54 |
| SAM 3, box-clipped | 0.734 | 0.781 | 0.670 | 0.705 | 0.703 | 0.804 | 0.625 | 0.711 | 0.867 | 0.754 | 0.52 |
| ExG + Otsu (colour baseline) | 0.509 | 0.624 | 0.350 | 0.442 | 0.461 | 0.645 | 0.296 | 0.464 | 0.765 | 0.539 | 0.39 |
| Inscribed ellipse | 0.445 | 0.440 | 0.452 | 0.443 | 0.451 | 0.442 | 0.466 | 0.402 | 0.467 | 0.454 | 0.79 |
| Filled box | 0.412 | 0.410 | 0.414 | 0.423 | 0.414 | 0.394 | 0.457 | 0.363 | 0.416 | 0.430 | 1.00 |

Paired differences, non-partial plants, 95 % image bootstrap (2000 resamples, seed 0); last column = images where the first arm is ahead, of 772.

| comparison | mean | 95 % CI | images ahead |
|---|---|---|---|
| tree-fitted − ellipse | +0.160 | [0.158, 0.162] | 772/772 |
| tree-fitted − box | +0.193 | [0.191, 0.196] | 772/772 |
| tree-fitted − ExG | +0.097 | [0.086, 0.107] | 571/772 |
| tree-fitted − SAM 3 | −0.114 | [−0.120, −0.107] | 86/772 |
| test-time fit − ellipse | +0.135 | [0.130, 0.139] | 760/772 |
| test-time fit − ExG | +0.071 | [0.059, 0.083] | 520/772 |
| test-time fit − SAM 3 | −0.139 | [−0.148, −0.131] | 117/772 |
| test-time fit − α 0 | −0.010 | [−0.010, −0.009] | 145/772 |
| test-time fit − tree-fitted | −0.026 | [−0.029, −0.022] | 272/772 |

- **Per-image fit failures: 7/772** images (all from 05-15, the earliest date): `fit_image` raised "fg-init cells < K_fg 16". Both test-time arms score empty masks (IoU 0) there; not dropped. Listed in `summary.json` → `lace_tt_fit_failed`.
- ρ final fg mass per size band (mean over fitted images): 0.43 / 0.58 / 0.67 (pin π/4; pruning after the pin lowers it, as in sample60).
- sample60 reproduced at full scale: every sign and ordering holds; tree-fitted − ExG widened (+0.062 → +0.097) and its CI no longer touches 0 for the test-time fit (+0.071 [0.059, 0.083], was a tie).
- Runtime: pipeline ~16.2 s/img on MPS (15.0 s features), 204 images before the fit-failure crash (`logs/val/pipeline_part1_crashed.log`) + 568 resumed (152 min, `logs/val/pipeline.log`); SAM 3 618 s (`logs/val/sam_run.log`).
- Figure: `figs/examples_val.png`.

### Reading

- **The tree-fitted masker transfers to plants unchanged** and beats the ellipse on every one of 772 images, the box on all 772, and the standard colour index by +0.097.
- **ExG still wins on large isolated crops** (T3 0.765 vs 0.650; 06-05 0.645 vs 0.585). LACE's lead comes from small plants and weeds (ExG weed 0.350).
- **SAM 3 stays ahead** by 0.11 (tree-fitted) to 0.14 (test-time fit).
- **Per-image refit does not help** (−0.026 vs tree-fitted); the π/4 spatial prior costs 0.010.

## Result: sample60 (PROVISIONAL; superseded by val)

Mean per-instance mask IoU on identical tight GT boxes. The headline subset is the 711 non-partial plants (394 crop, 317 weed). Source: `results/sample60/summary.json`.

| arm | non-partial | crop | weed | 05-15 | 05-26 | 06-05 | T1 | T2 | T3 | incl. partial | fill |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LACE per-image test-time fit (α 0.3) | 0.564 | 0.568 | 0.559 | 0.613 | 0.572 | 0.523 | 0.626 | 0.543 | 0.522 | 0.580 | 0.63 |
| LACE test-time fit, α = 0 (sensitivity) | 0.572 | 0.575 | 0.568 | 0.614 | 0.584 | 0.533 | 0.624 | 0.562 | 0.529 | 0.590 | 0.61 |
| **LACE tree-fitted, as-is** | **0.595** | 0.624 | 0.559 | 0.588 | 0.624 | 0.573 | 0.605 | 0.544 | 0.635 | 0.612 | 0.62 |
| SAM 3 zero-shot (uncropped) | 0.726 | 0.777 | 0.662 | 0.656 | 0.702 | 0.795 | 0.594 | 0.724 | 0.859 | 0.743 | 0.53 |
| SAM 3, box-clipped | 0.738 | 0.787 | 0.677 | 0.681 | 0.716 | 0.798 | 0.627 | 0.728 | 0.860 | 0.757 | 0.51 |
| ExG + Otsu (colour baseline) | 0.533 | 0.653 | 0.384 | 0.450 | 0.502 | 0.618 | 0.325 | 0.517 | 0.757 | 0.566 | 0.41 |
| Inscribed ellipse | 0.438 | 0.436 | 0.441 | 0.418 | 0.451 | 0.440 | 0.450 | 0.405 | 0.460 | 0.451 | 0.79 |
| Filled box | 0.402 | 0.402 | 0.402 | 0.407 | 0.413 | 0.389 | 0.437 | 0.363 | 0.407 | 0.425 | 1.00 |

- GT fill is 0.40.
- Size terciles are split at a box √area of 33.9 px and 104.7 px.

Paired differences on non-partial plants. The CI is a 95 % image bootstrap (2000 resamples, seed 0). The last column counts images where the first arm is ahead, out of 60.

| comparison | mean | 95 % CI | images ahead |
|---|---|---|---|
| tree-fitted − ellipse | +0.157 | [0.148, 0.166] | 60/60 |
| tree-fitted − ExG | +0.062 | [0.030, 0.094] | 42/60 |
| tree-fitted − SAM 3 | −0.131 | [−0.153, −0.107] | 5/60 |
| test-time fit − ellipse | +0.126 | [0.109, 0.143] | 59/60 |
| test-time fit − ExG | +0.031 | [−0.007, 0.071] (tie) | 36/60 |
| test-time fit − SAM 3 | −0.162 | [−0.191, −0.130] | 8/60 |
| test-time fit − α 0 | −0.008 | [−0.011, −0.006] | 13/60 |
| test-time fit − tree-fitted | −0.031 | [−0.045, −0.017] | 18/60 |

### Reading (sample60, provisional)

- **The tree-fitted masker transfers to plants unchanged.** With no refit and frozen tree settings, it beats both geometric baselines and the standard colour index. Its masks are object-shaped: fill 0.62 against a GT fill of 0.40.
- **The colour baseline wins on large isolated crops.** ExG scores 0.653 on crops against 0.624, and 0.757 on the largest third (T3) against 0.635. LACE's overall lead comes from small plants and weeds (ExG 0.384).
- **SAM 3 stays well ahead,** by −0.13 to −0.16.
- **The spatial prior ρ costs a little.** With α = 0 the score is 0.008 higher, so the π/4 mass pin hurts, but only slightly.
  - Observed: the final ρ foreground mass is 0.47 / 0.60 / 0.66 per size band, not π/4. In `em.fit` the prototype pruning runs after the pin, so pruned prototypes take mass with them. The deployed tree model's mass is 0.71 / 0.68 / 0.69.

## How PhenoBench was chosen: disclose this in any write-up

PhenoBench was the **second** dataset tried, and it was picked after the first one failed. The high-level record is kept here because the folders that held those runs are slated for deletion.

1. **LIVECell probe (2026-11-13; 20 test images, 8 cell types, 5,448 cells, GT boxes, same frozen LACE).** Mean IoU per arm:

   | arm | mean IoU |
   |---|---|
   | LACE per-image fit | 0.554 |
   | Filled box | 0.551 |
   | Tree-fitted LACE | 0.555 |
   | Ellipse | 0.684 |
   | SAM 3 | 0.767 |

   - LACE masks filled 0.94 of the box, against 0.55 for the GT, so they were essentially the box.
   - Scale check (upsampling the images, same protocol): LACE scored 0.594 at ×2 and 0.642 at ×3. The gain at ×3 was +0.088 [0.070, 0.115], better on 20/20 images, but LACE stayed below the ellipse (−0.042 [−0.061, −0.013]).
   - Reading: the failure was largely object scale (cells span about 1.2 DINO patches), not domain alone.
2. **Ground-truth-only screen (no model run).** Columns: median object size in 16 px patches, GT fill of the box, inscribed-ellipse IoU, and crowding (the fraction of each box covered by other instances).

   | dataset | median size (patches) | fill | ellipse IoU | crowding |
   |---|---|---|---|---|
   | TCD test | 2.6 | 0.66 | 0.80 | 0.03 |
   | LIVECell | 1.2 | 0.55 | 0.68 | 0.29 |
   | PhenoBench (60-image sample) | 3.8 | 0.40 | 0.44 | 0.04 |
   | GlaS testA+B | 7.4 | 0.70 | 0.73 | 0.04 |

   PhenoBench was chosen because it has the most room above the ellipse, low crowding, objects big enough to resolve, and drone imagery taken top-down like TCD. GlaS was the runner-up and was not run.

## Protocol

- **Boxes:** the tight pixel extent [min, max+1) of each GT plant instance. Every instance, including partial plants (less than 50 % visible), is prompted and used in the fit. Headline scoring is on non-partial plants; partial plants are reported separately.
- **Arms** (`code/run.py`, all on identical boxes):
  - **`lace_tt`:** a per-image test-time fit (`lace_core.fit_image` = an `em.fit` port with frozen settings), then the deployed lace4px_simple render (α 0.3, κ 16, guided filter r 12 / ε 1e-4, τ = 0.33 × in-box max, clipped to the box).
  - **`lace_tt_a0`:** the same model rendered with α = 0.
  - **`lace_tree`:** the deployed tree masker `em_model_4p_fix.npz`, used as-is.
  - **`box`, `ellipse`:** the geometric baselines.
  - **`exg`:** ExG = 2g − r − b on chromatic coordinates, one Otsu threshold per image, clipped to the box (Woebbecke 1995). No tuning.
  - **`sam3` / `sam3_boxclip`** (`code/sam_run.py`): HF `Sam3TrackerModel` `facebook/sam3`, whole image, box prompts, `multimask_output=False`.
- **Features:** DINOv3 ViT-L/16 layer 24, 16 phases {0,4,8,12}², RGB. A 1024² image is exactly one deployed 1024 px window per pass. Features go through an fp16 round-trip.
  - `sample60` stored them in `features/sample60/` (gitignored).
  - `val` computes them in memory and discards them, because storing them would take about 103 GB.
- **Metric:** mean per-instance mask IoU (pairwise RLE, `_verbatim.per_crown_iou_rle`). Paired 95 % image-bootstrap CIs (2000 resamples, seed 0).

### Correctness evidence

- **`results/gate_parity_venv.json`:** `code/gate_parity.py`, run in this folder in the repo `.venv`. The ported apply/render path reproduces the deployed LACE masks byte for byte on 2 OAM-TCD val tiles (1,200/1,200 RLE identical). The reference is `feat_cache/refit4p_val_deployed`, whose deployed val AP50 is 0.7414.
- **`results/gate_xvenv_t457.json`:** the CPU arms run in the feature venv (`venv_feat_t457`, which `pipeline.py` uses) reproduce the sample60 masks written by `.venv` byte for byte, on 3 images × 6 arms. `gate_parity.py` itself cannot run in that venv because it has no `modal`.
- **`_verbatim.verify()`:** a source-identity check of every copied repo function. It runs at the start of every run.
- **`LC.check_frozen()`:** the settings match `em_model_4p_fix.npz`.

The code in `code/lace_core.py`, `code/_verbatim.py` and `code/gate_parity.py` is a byte-identical copy of the port first written for the LIVECell probe; only the usage line in `gate_parity.py`'s docstring differs.

## Data

PhenoBench v1.1.0, from https://www.phenobench.org.

- Licence: CC BY-NC-SA 4.0 per the zip README (`data/README.MD`); the website says CC BY-SA 4.0.
- Members were pulled from the 7.6 GB zip by HTTP range requests (`code/fetch.py`). Downloads are treated as untrusted data: names are checked against a strict pattern and files are checked to be PNG.
- The sha1 of every file is recorded in `results/val/files_sha1.json`. The 60 sample files are checked against the screen's original download record (`results/sample60/files_sha1_screen.json`).
- Layout: `data/val/{images,plant_instances,semantics}/` (gitignored).

## Commands (from `code/`; `PY=../../../.venv/bin/python`, `PYF=/Users/tompitts/dphil/feat_cache/venv_feat_t457/bin/python`, `PP=PYTHONPATH=/Users/tompitts/dphil/CanopyAI2`)

```
$PY -I -u fetch.py select ; for k in 0..7: $PY -I -u fetch.py files $k 8 ; $PY -I -u fetch.py sha1
$PP $PY -u gate_parity.py                                  # parity gate (.venv)
PB_RUN=xvenv_check $PP $PYF -u xvenv_check.py              # cross-venv gate
PB_RUN=val $PP $PYF -u pipeline.py                         # features in memory + CPU arms; ~16 s/img on MPS, ~3.4 h
PB_RUN=val $PP $PY -u sam_run.py                           # ~0.7 s/img
PB_RUN=val $PY -u summarise.py ; PB_RUN=val $PP $PY -u viz.py
```

The sample60 run used the older `feats.py` (stored features) then `run.py main()`, which reads `features/sample60/`.

## Layout

- `code/`: the scripts.
- `results/<run>/`:
  - `<arm>/<stem>.json`: native RLE, unrounded IoU and double-assignment, per instance.
  - `gt/`, `lace_tt/fit/` (fit reports plus ρ maps), `lace_tt/models/` (gitignored).
  - `selection.json`, `summary.json`.
- `results/gate_*.json`, `logs/<run>/`, `figs/examples_<run>.png`.
- `data/` and `features/` are gitignored.
