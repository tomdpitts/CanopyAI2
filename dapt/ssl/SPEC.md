# DAPT on Modal/CUDA — SPEC

> **STATUS 2026-07-14: v3 is the ACTIVE study — see the `## v3 study` section below
> (it is the runbook). Everything from `# M2 …` down to `## v3 study` is run-1/v1–v2
> history (old pool, 33-tile fixed split, NEON-included) retained for provenance;
> its numbers are superseded by v3. Result narrative: `dapt/REPORT.md`.**

---

# M2 (run-1, HISTORICAL) — DAPT on Modal/CUDA: single run, maximize power

Not a sweep. One adapted checkpoint, configured for the **highest probability of a
real, resolved DAPT − web gap** on the L2 paired test — by matching the pool to the
arid test sites and reading the result where the hypothesis (D4) says the effect
lives, while still reporting everything.

Baseline to beat: **web L2 mAP50 0.450 ± 0.015** (sat 0.354; paired web−sat +0.094
RESOLVED). L1 is underpowered — the claim rides on **L2**.

## Why these choices maximize power (the logic, before the knobs)

1. **DAPT only helps where the domain matches.** So maximize overlap between the SSL
   pool and the arid test sites. (Headline = FULL test per the 2026-07-07 amendment —
   AP is box-weighted and NEON is only ~24% of test boxes, so dilution is modest; the
   arid/NEON breakdown stays as the secondary mechanism readout, D4.)
2. **33-tile test → need a large effect + max power.** Use the **paired-gap
   bootstrap** (the only powerful test here), 5 seeds.
3. **One-shot (no sweep to catch degradation) → insure the downside.** Use the full
   DINOv3 objective **with Gram anchoring to the web checkpoint**: the DINO/iBOT/Koleo
   losses pull features toward arid (upside), Gram-anchor-to-web keeps dense patch
   features from collapsing during a single unswept run (bounds the floor near web).
   This is the one setting where Gram anchoring is worth it *at this pool size* — as
   collapse insurance for a run we can't sweep, not for its designed large-data role.

## Pool composition — the dominant lever (freeze once)

Maximize pool ↔ arid-test match, never touch the labelled tiles' pixels. **DONE** via
`build_pool.py` → `dapt/ssl/pool/` (tiles/, manifest.json, samples/coverage). **1,084
tiles.**

- **BRU162-center** (`splits2/BRU162_center_80pct.tif`) — near-identical domain to the
  BRU test tiles (same ortho/flight; labels are the L/R 10% strips). 198 tiles.
- **WON003 `WON003_10cm_right60.tif`** — right60 crop with training data already
  removed (no footprint masking needed). Adapts to WON, the highest-box-count test
  portion. 151 tiles.
- **CAN091/095/117** — arid breadth, unlabelled. 227/224/284 tiles.
- **Tile = 512 px @ native 0.1 m/px, 50% overlap.** 512 (not 256) is DINOv3-faithful —
  source > crop, so RRC global crops are native-res (no upsampling) — AND matches the
  detector's 512→32×32 grid. Overlap boosts the pool / cuts per-tile reuse over a 5k
  run; it's a pool-size choice, not a DINOv3 deviation.
- **Empty space = pure WHITE (>=250 on all channels)**, up to 41% of WON / 53% of CAN.
  Keep tiles only if >=95% valid; white excluded from stats. **d4 (k*90 rot + flip)**
  is the corner-safe rotation aug; arbitrary angles inject white corners — not used.
- **Arid RGB mean/std = [0.503, 0.470, 0.415] / [0.127, 0.117, 0.110]** (std well below
  ImageNet ~0.22 — arid is low-contrast). Export in the checkpoint's
  `preprocessor_config.json` so probe-time norm matches SSL.
- **Leakage:** safe by construction (BRU center / WON right60 / CAN unlabelled) — no
  per-tile masks needed.

## SSL config (frozen once, like the target encoder)

- **Objective:** full DINOv3 (DINO + iBOT + Koleo + **Gram anchoring, anchor =
  frozen web checkpoint**), Meta `dinov3` training repo on CUDA.
- **Init:** web `facebook/dinov3-vitl16-pretrain-lvd1689m`. Backbone ViT-L/16.
- **LR (as-executed, FROZEN for run-1):** peak effective **7.07e-5** (yaml
  `optim.lr: 1e-4` × the repo's `sqrt_wrt_1024` = ×4×√(32/1024)), 500-iter linear
  warmup, cosine to ~0. Hotter than the originally-intended ~1.8e-5 (factor-4
  convention missed pre-launch, caught live at iter ~1700, run kept) — accepted as
  the canonical run-1 config: cumulative-drift arithmetic means the DCP trail's
  first rungs cover the intended gentle regime (hot-500 ≈ intended-1250, hot-1500 ≈
  the whole intended 5k run) and later rungs extend into stronger adaptation.
  Layerwise decay 0.9 → probed blocks 3/6/9/12 train at ~0.8–2e-5 regardless.
- **Steps:** single **~5k** (≈ few hundred passes over the pool; enough to adapt,
  Gram bounds overfit/collapse). No sweep — this is the one-shot bet.
- **Crops: global 256 / local 112 — VERIFIED as Meta's own recipe** for our exact
  init: `dinov3_vitl16_lvd1689m_distilled.yaml` (and the 7B pretrain + gram-anchor
  configs) all use 256/112. The fork's 224/96 came from `vitl_im1k_lin834.yaml` (an
  IN-1k baseline) + schema defaults — not the released-model recipe. The **512-px
  source tiles** keep RRC native-res (source > crop). RandomRot90 (d4) patched in via
  `apply_repo_patches.py`; EMA teacher; bf16. Optional later: short 512-global tail
  (Meta's hi-res phase).
- **Gram: OFF for run-1 (decided 2026-07-07 after code reading).** Wiring is verified
  trivial (`gram.use_loss` + `gram.ckpt` → our teacher .pth loads via the same
  `init_fsdp_model_from_checkpoint` path already validated; gram crops share the
  geometric stage incl. our RandomRot90, so `gram_teacher_crops_size: 256` is exact).
  But Gram anchors the **patch-level similarity structure to web — and patch tokens
  are exactly what our probes read**: it would fight the dense-feature adaptation the
  experiment measures. Collapse insurance is instead the **500-iter DCP trail +
  val-based checkpoint selection** — later ckpts probing below web on val =
  degradation alarm, pick an earlier one.
- **Normalization: ImageNet everywhere for run-1** (see the Normalization decision
  section) — SSL under repo-default ImageNet stats, dapt-arm HF export ships ImageNet
  stats too, so SSL-time and probe-time norms agree and dapt vs web differs by
  weights only. Crops **256 global / 112 local** = DINOv3's multicrop recipe (2
  globals teacher+student, 8 locals student-only; local→global consistency is the
  objective). 512 globals is only Meta's brief final hi-res *tail*, not the main
  recipe; our probes evaluate at 512-px input anyway, so any hi-res drift from
  256-only continuation shows up directly in val checkpoint selection.
- **Seed** recorded (numpy+torch). Config hash logged.

## Modal infra

- **Image:** CUDA base + torch + xFormers/flash-attn + `dinov3` repo + rasterio/PIL.
- **Volume:** upload arid pool (~400 MB orthos) + web checkpoint + pool/exclusion
  config (`pool.json`: tif paths, exclusion boxes, tile size, arid mean/std).
- **GPU:** **A100-40GB — cost-optimized default** (H100 only if its $/hr is offset by
  ≥ the speedup). bf16 + gradient checkpointing to fit 512 globals. ~few GPU-h, ~$10–40.
- **Entry:** tile+exclude → SSL continuation → export adapted weights **in HF
  `AutoModel` format** so `dapt/backbone.py` loads it as arm `"dapt"` with zero
  pipeline change (add one entry to `MODEL_IDS`/a local path). If the repo can't emit
  HF format directly, include a state-dict → HF converter in the entrypoint.
- **Output:** pull `dapt.safetensors` back to `dapt/ckpt/`. Feature caching + probe
  training run **locally on the M4 Max, unchanged** (cheap, already working).

## Evaluation of the claim (identical probe/target config to web/sat)

- **Primary:** L2 MLP probe, **FULL test set**, **paired-gap bootstrap DAPT − web**,
  5 seeds. Resolved iff the gap CI clears 0 (same test that resolved web−sat).
- **Always also report:** arid (WON+BRU) and NEON subset gaps (mechanism readout),
  per-site, count-error, isolated-vs-touching. Transparency against cherry-picking.
- **Frozen:** target encoder, head capacity/loss, NMS, seeds — byte-identical to the
  web/sat arms. Only the backbone checkpoint differs.

## Honest caveats (surface, don't bury)

- **No shuffled-DAPT control in a one-run plan** ⇒ we can't fully separate "arid
  adaptation" from "any extra SSL." Partial substitute: the expected **pattern**
  (DAPT > web on BRU/WON, ≈0 on NEON) is itself evidence of *arid-specific* gain; a
  uniform lift across NEON too would instead suggest generic-SSL effects. State this.
- **Endpoint amended 2026-07-07 (pre-run): full-test primary, arid/NEON secondary.**
  Legitimate only because no DAPT data existed yet; the endpoint is now frozen.
- **One-shot LR/steps may miss the optimum.** Low LR + the DCP trail bound the
  downside; if the result is DAPT ≈ web, that's the valid **subsumption** outcome
  (narrow SSL subsumed by web features), diagnosable via pool size / step count.
- **WON leakage** is handled by construction: the pool's WON source is the
  user-cropped `WON003_10cm_right60.tif` (training data removed).

## Validation probe — cheap, sensitive, local (how to prove DAPT > web)

Baseline (do-not-beat-with-anything-fancier): **web `facebook/dinov3-vitl16-pretrain-lvd1689m`
(ViT-L/16)**. The probe is the existing `dapt/` frozen-feature pipeline — **no GPU
beyond feature extraction**. Cost per checkpoint = one forward pass over the 133
labelled tiles (~minutes, local M4 Max); every probe fit + bootstrap is free CPU.
That is what makes it cheap. Sensitivity comes from the **paired-gap bootstrap**
(cancels tile-to-tile noise) + 5 seeds, run at **linear**
capacity (cheapest, lowest-variance, most sensitive to the *representation*) and also
at **mlp** (matches the reported baseline). SSL is agnostic to the downstream head —
only the backbone checkpoint changes; the probe/head/target/NMS/seeds are byte-identical
to the web/sat arms.

**Wiring (zero pipeline change).** Modal exports each adapted checkpoint as an HF
`AutoModel` dir **including `preprocessor_config.json` with the arid mean/std it was
DAPT'd with**. Register it in `dapt/ssl/checkpoints.json`, e.g.
`{"dapt": "dapt/ckpt/dapt_hf"}`; `dapt/backbone.py` merges it into `MODEL_IDS`.

**Run (identical config to web/sat):**
```
.venv/bin/python -m dapt.cache_features --arm dapt
.venv/bin/python -m dapt.run_baseline --arms dapt web --capacity linear --seeds 0 1 2 3 4
.venv/bin/python -m dapt.run_baseline --arms dapt web --capacity mlp    --seeds 0 1 2 3 4
```
Prints `PAIRED dapt-web mAP50 gap ±95%CI, P(gap>0), RESOLVED?`. Headline the **arid
(WON+BRU) subset**; always also report full-test + **NEON** (OOD control ≈0) + per-site.
Resolved iff the paired-gap CI clears 0.

**Checkpoint selection = the anti-forgetting + don't-miss-benefits lever.** Because
we run a single DAPT (true peak 7.07e-5 as-executed, no separate LR sweep), **save
intermediate checkpoints on Modal (DCP every 500 iters)**, register each
(`dapt_s1000`, `dapt_s3000`, `dapt_s5000`, …), and probe all — **selecting the
checkpoint on full-val mAP50, never on the test gap** (test-based selection would
contaminate the headline number). Selection costs only feature extraction (no
retraining), converts the one-shot LR/step risk into a cheap post-hoc choice, and
*is* the forgetting alarm: if later checkpoints probe below web on val, features are
degrading → use an earlier one.

## Pre-registered endpoint (2026-07-03; AMENDED 2026-07-07 pre-run, before any DAPT
## data existed)

**Primary:** paired DAPT − web mAP50 gap, **L2 probe, 100% labels, FULL test set**
(33 tiles / ~362 crowns, NEON treated like WON/BRU), 95% CI must clear 0. **Sweep
checkpoint selected on full val** (never the test gap). Secondary (always reported):
arid (WON+BRU) and NEON subset gaps — the mechanism readout (gain concentrated in
arid ⇒ arid-specific adaptation; uniform ⇒ generic extra SSL), per-site, L1,
count-error, strata.
*Amendment rationale:* AP is box-weighted and WON dominates boxes (NEON = ~9% of val
boxes, ~24% of test boxes), so the dilution the original arid-only endpoint guarded
against is modest; full-test is the simpler, more general claim. Run-1 SSL pool
remains arid-only (that's the treatment, not the eval).

## Normalization decision (settled 2026-07-07 after two reversals — final)

**ImageNet norm everywhere for run-1** (SSL + dapt-arm extraction; the repo default,
no yaml override). Reasoning: (a) the web init's weights expect ImageNet-normed
inputs — switching stats mid-continuation (arid std 0.127 vs 0.229 ⇒ ~2× input
amplification) spends the short 5k-step budget re-adapting to the *norm* instead of
arid *content*; (b) Meta's sat493m does ship domain stats, but that's a 493M-image
from-init regime, and sat *lost* to web on our tasks anyway — weak precedent;
(c) with ImageNet on both, **dapt vs web differs by weights only** (clean
attribution). GOAL's "adapt RGB normalization to arid stats" is deliberately
deviated from for the short-continuation regime — arid stats are recorded in
`pool/manifest.json` as a follow-up variant if run-1 shows signal.

## File map (post-cleanup 2026-07-07; this section is the runbook)

- `build_pool.py` → `pool/{tiles/,manifest.json,samples/}` — DONE (1,084 tiles)
- `convert_hf_to_dinov3.py` — HF web ckpt → `dinov3_web_vitl_teacher.pth` (SSL init;
  DONE, 1.1 GB on disk). Needs a local dinov3 clone: `$DINOV3_REPO` or
  `~/.cache/dinov3` (the original clone lived in a /tmp scratchpad and was purged).
- `apply_repo_patches.py` + `arid_pool_dataset.py` — patch a fresh dinov3 clone:
  AridPool flat-folder dataset + RandomRot90 (d4) aug. Idempotent; Modal runs it.
- `dinov3_dapt_vitl.yaml` — the frozen SSL config (crops 256/112, ImageNet norm,
  released-ViT-L arch, 5k iters, DCP every 500). Preflight-verify notes in header.
- `local_smoke.py` — free CPU preflight (config parse, 0-missing-params load,
  native-vs-HF feature cosine). Re-run after any yaml/arch change.
- `modal/app.py` — dedicated app+volume, A100, 3 h hard cap (~$7), resumable;
  preflight then torchrun. Dataset root MUST stay `/vol/pool/tiles`.
- `convert_dinov3_to_hf.py` — adapted teacher → HF dir (+ arid preprocessor config);
  register per-checkpoint in `checkpoints.json` → `backbone.py` arm.
- Deleted in cleanup: `recover_won_coords.py` (WON right60 crop supersedes NCC
  footprint recovery), stale 1,747-tile pool (fork's builder filtered BLACK nodata but
  these orthos' nodata is WHITE → ~46% junk tiles + stats inflated to ~[0.68]/[0.26];
  correct content stats are [0.503,0.470,0.415]/[0.127,0.117,0.110]).

## Deep preflight 2026-07-07 — 4 launch-killing bugs found & fixed locally

1. `modal/app.py` overrode `schedules.lr.peak` — **no `schedules` section in the SSL
   schema** (fork knew, fixed the yaml, missed the CLI) → strict-merge crash on the
   box. Now `optim.lr=`.
2. `RandomRot90` used raw `torch.rot90` → `NotImplementedError` on the PIL images in
   the geometric stage; also torchvision ≥0.21 dispatches `transform()` not
   `_transform()`. Now `v2.functional.rotate` + both method names; **verified: full
   DataAugmentationDINO runs on real pool tiles, rotations active** (crop variance
   across draws).
3. `AridPool.super().__init__(root, ...)` bound `root` to `image_decoder`
   (ExtendedVisionDataset signature is `(image_decoder, target_decoder, *args)`) →
   "can not read image" on sample 0. Now keyword args; **verified:
   `make_dataset("AridPool:root=...")` loads all 1,084 tiles**.
4. Image used `from_registry(pytorch/pytorch, add_python=3.11)` — add_python's
   standalone interpreter can't see conda torch, so torch would arrive **unpinned**
   via torchmetrics' deps (or not at all). Now `debian_slim` + pinned
   `torch==2.7.1`/`torchvision==0.22.1` (PyPI wheels bundle cu126) + pillow; dinov3
   installed `--no-deps`. Also added `cpu=8, memory=32768` (yaml has num_workers=8).

Also verified: `init_fsdp_model_from_checkpoint` on a non-dir path does exactly
`torch.load(path)["teacher"]` (our converter's format) and the student-resume call
passes `keys_not_sharded=[rope_embed.periods, qkv.bias_mask]` (covers the synthesized
bias_mask buffer). DINO/iBOT heads start fresh (never released) — expected for
continuation; warmup + freeze_last_layer_epochs=1 handles head burn-in. Local smoke
re-passes on the patched clone (cosine 1.0000).

## Run-1 LR: as-executed config ADOPTED (decided 2026-07-07, mid-run)

The repo's `sqrt_wrt_1024` rule is `lr × 4 × sqrt(batch·world/1024)` — a **factor 4**
missing from our (and the fork's) preflight formula, so run-1 trains at **true peak
7.07e-5**, not the intended ~1.8e-5. Caught live at iter ~1700 (losses/grads
healthy); decision: **keep the run and adopt it as canonical — no seed-0 SSL
retrain.** Rationale: the DCP-500 trail from the hot run spans the intended run's
entire cumulative-drift range within its first ~3 rungs and extends beyond, so val
checkpoint-selection reads a wider adaptation sweep than the original design.
Deviation disclosed here; preflight formula fixed in app.py (a hypothetical future
cooler variant would be `--lr-peak 2.5e-5` under a NEW seed, only if the science
demands it — not part of run-1).

## Run order (handoff checklist)

1. **Pool — DONE.** 1,084 tiles @512 px native, 50% overlap, white-filtered,
   leakage-safe (BRU center / WON right60 / CAN). Rebuild:
   `python -m dapt.ssl.build_pool --tile 512 --stride 256`.
2. **Stage volume** — upload `pool/tiles/`, `dinov3_web_vitl_teacher.pth`, yaml to the
   Modal volume (`/vol/pool/tiles`, `/vol/dinov3_web_vitl_teacher.pth`).
3. **Modal SSL** — `modal run --detach dapt/ssl/modal/app.py::train [--seed N]`;
   preflight asserts CUDA, effective-LR sanity band, 0-missing-params, then
   torchrun 5k iters, DCP every 500. Seed feeds both `--seed` (fix_random_seeds) and
   `train.seed` (sampler stream); output dir is seed-scoped (`/vol/out_s<seed>`) so
   different seeds can't cross-resume. Resumable: re-invoke with the SAME seed to
   continue after timeout/kill.
4. **Export + pull** (chain built & tested 2026-07-07 on the web teacher):
   a. `modal run dapt/ssl/modal/app.py::extract --seed 42` — teacher-backbone-only
      .pth per DCP checkpoint (partial metadata-driven load, ~0.6 GB bf16 each) →
      `/vol/export/teacher_s42_i<iter>.pth`
   b. `modal volume get dinov3-dapt-arid-vol /export dapt/ssl/export`
   c. per checkpoint: `.venv/bin/python -m dapt.ssl.export_hf --teacher
      dapt/ssl/export/teacher_s42_i<iter>.pth` — unmap → load into a real web
      AutoModel (0-unexpected gate) → feature-sanity gate (must differ from web but
      cosine > 0.5; warns if ~identical) → `dapt/ckpt/dapt_s42_i<iter>_hf/` (fp32 +
      ImageNet preprocessor per the norm decision) → auto-registers arm
      `dapt_s42_i<iter>` in `checkpoints.json`.
5. **Probe locally** — `cache_features --arm dapt_sNNNN` then `run_baseline --arms
   dapt_sNNNN web` (linear + mlp, 5 seeds); **select the checkpoint on full-val
   mAP50** (never the test gap); later ckpts below web on val = degradation, use
   earlier. (TODO: small checkpoint-selection driver + arid/NEON subset breakdown in
   the report path.)
6. **Report once on test** — per the amended endpoint: paired gap ±95% CI, L2, FULL
   test; secondary arid/NEON subsets (mechanism readout), per-site, L1; RESOLVED iff
   CI clears 0. DAPT ≈ web = valid *subsumption* result.


## v3 study (ACTIVE — 2026-07-14) — repeated k-fold, leakage-safe, arid-only

Fresh start on re-annotated, higher-quality data with NEON removed (it was out of
domain). Same core question — does arid DAPT beat off-the-shelf DINOv3 for a
frozen-feature dryland crown detector — but re-engineered to **settle the
significance question** that v1/v2 left at a modest margin (effect ~+0.018–0.020,
per-SSL-seed spread ±0.003, but 33-tile test CI ±0.02 → unresolved).

**All v3 code + assets isolated under `dapt/v3/`** (data, k-fold runner, SSL app,
pool). **Brand-new Modal app + volume** (`dinov3-dapt-v3` / `dinov3-dapt-v3-vol`) —
nothing shared with the v1/v2 run.

### Data (`data/finetune/v3/`)
- **97 tiles, arid-only (WON+BRU), NEON dropped.** train/ 16 tiles (1,053 boxes,
  WON-only), test/ 81 tiles (1,375 boxes, 53 BRU + 28 WON). Annotations =
  `annotations.csv` (basenames + domain; 259 stale rows referencing images not in
  either folder → pruned on load).
- **CORRECTION 2026-07-14 (during execution):** the 19 BRU tiles originally in
  train/ were moved to test/ — all labelled BRU tiles come from the L/R 10% strips,
  which the pool ortho (BRU162_center_80pct) excludes by construction, so they are
  leakage-safe (user confirmed). Only the 16 WON tiles overlap the SSL pool.
- **The `test/` 81 tiles are DEFINITIVELY EXCLUDED from the DAPT SSL orthos**
  (leakage-safe holdout). The `train/` 16 WON tiles overlap the SSL pool (WON
  right50) → usable to *train* a probe but **never to evaluate the DAPT arm**.

### Evaluation — repeated k-fold over the leakage-safe tiles only
- **5-fold × 3 repeats over the 81 leakage-safe test tiles**, every arm identical
  folds. Each fold: ~65 train / ~16 held-out; pool out-of-fold predictions over all
  81 → **paired-gap bootstrap** (same powerful test as v1/v2). Multiple probe seeds.
- Why: eval only ever on pool-excluded tiles (**zero DAPT leakage**), yet all 81 tiles
  contribute to the estimate (**full power**) and each fold trains on ~65 box-dense
  tiles (**no train starvation** — the trap the fixed 35/65 split carried).
- Est. power: 81 eval tiles → CI half-width ≈ 0.011, so the ~+0.020 arid effect
  should **resolve** (P(gap>0) ≈ 0.98+). The 16 WON overlap tiles are held out of the
  headline (a "DAPT memorized them" objection); optional robustness run adds them as
  fixed extra training.
- **Probes:** L1 linear + L2 MLP, frozen-backbone, identical target-encoder/NMS across
  web / sat / dapt arms (reuses `dapt.head`/`dapt.targets`/`dapt.eval` unchanged).
  Full decoder head = optional later.

### DAPT SSL — new pool, leakage-safe
- **Pool orthos (all arid, all leakage-safe for the 81 test tiles):**
  `BRU162_center_80pct.tif`, `CAN091/095/117_10cm.tif`, `WON003_10cm_right50.tif`
  (pre-cropped so labelled tiles are removed — no footprint masking). ~2,180 raw
  512px/50%-overlap tiles → ~1,400 after the ≥95%-valid white filter (est.).
- **Protocol = P1K** (validated in v2): 1000 iters, DCP every 250, warmup 500 (so the
  i499 rung stays schedule-comparable), **ImageNet norm (FINAL — the old arid-stats
  norm variant is RETIRED: arid-only study, no NEON, no cross-domain norm question)**,
  Gram off, crops 256/112. Each seed → ~i249/i499/i749/i999 checkpoints; **val-select
  per seed** (inner val), pool the selected winners across seeds.
- **Seeds: 101, 201, 301, 401, 501 — CONFIRMED (5 seeds, ImageNet norm; user
  2026-07-14).**
- **Backbone must not see the 81 test tiles** — guaranteed by pool construction;
  `build_pool` asserts test-tile disjointness before upload.

### Modal GPU cost estimate (grounded in v2's measured P1K timing)
- **Per SSL seed (P1K, A100-40GB @ ~$2.10/hr):** ~11 min training + ~2 min
  startup/preflight ≈ 13 min GPU → **~$0.50/seed** (measured: v2 seed-43 ran ~11 min).
- **Marginal cost per ADDITIONAL seed ≈ $0.50.**
- **5 seeds ≈ $2.6; 3 seeds ≈ $1.6** (+ ~$0.10 CPU-extract + one-time image build,
  effectively free). All-in with margin/rebuilds: **$3–4 (5 seeds) / $2–2.5 (3 seeds)**.
- Extraction runs on a CPU Modal function (~$0.02/seed); all probing/k-fold is LOCAL
  (M4 Max, $0) — only wall-clock, not $.
- H100 would ~halve wall-time at ~$3.95/hr → similar $/seed; A100-40GB stays default.

### HANDOFF — execution checklist (state as of 2026-07-14: BUILT+VALIDATED, NOT RUN)

**Already done (no GPU spent):** `dapt/v3/` package built and smoke-tested —
`data.py` (97 tiles → 81 leakage-safe + 16 WON overlap after the 2026-07-14 BRU
correction, 259 stale rows auto-dropped;
web+sat v3 features already cached in `dapt/v3/cache/`), `kfold.py` (validated
end-to-end: web−sat +0.028 RESOLVED on a 1-seed smoke), `ssl/build_pool.py`,
`ssl/app.py` (fresh Modal app `dinov3-dapt-v3` / volume `dinov3-dapt-v3-vol`).
Locked decisions: 5 SSL seeds 101/201/301/401/501; ImageNet norm (arid-stats concept
retired); K=5 × 3 repeats; probe seeds 0–4; L1 linear + L2 MLP; budget ~$4.

**To execute, in order:**
1. **Pool:** `.venv/bin/python -m dapt.v3.ssl.build_pool --tile 512 --stride 256`
   → `dapt/v3/ssl/pool/{tiles,manifest.json}` (expect ~1,300–1,500 tiles).
2. **Stage volume:** `modal volume put dinov3-dapt-v3-vol dapt/v3/ssl/pool/tiles
   /pool/tiles` and `modal volume put dinov3-dapt-v3-vol
   dapt/ssl/dinov3_web_vitl_teacher.pth /dinov3_web_vitl_teacher.pth`
   (teacher .pth is 1.1 GB, lives in dapt/ssl/; regenerate via
   `dapt.ssl.convert_hf_to_dinov3` if missing).
3. **SSL (~$0.5/seed):** per seed S in 101 201 301 401 501:
   `modal run --detach dapt/v3/ssl/app.py::train --seed S` (P1K defaults baked in).
   Poll volume for `/out_v3_sS/ckpt/999`. Resume = re-run same seed.
4. **Extract (CPU):** `modal run dapt/v3/ssl/app.py::extract --seed S` →
   `/vol/export/teacher_v3_sS_i<it>.pth`; if it errors right after training, it's the
   volume-commit race — wait ~1 min and retry (known, harmless).
5. **Download + export:** `modal volume get dinov3-dapt-v3-vol
   /export/teacher_v3_sS_i<it>.pth dapt/v3/ssl/export/…` per file, then
   `.venv/bin/python -m dapt.ssl.export_hf --teacher <pth> --out
   dapt/v3/ckpt/dapt_v3_sS_i<it>_hf` (pass --out explicitly to keep v3 ckpts under
   dapt/v3/; registry `dapt/ssl/checkpoints.json` is intentionally shared — arm names
   `dapt_v3_*` can't collide). Cosine-vs-web gate: smooth decay to ~0.4–0.7 = normal
   adaptation; hard-fail only <0.15/NaN.
6. **Cache + per-seed val-select (local, free):**
   `.venv/bin/python -m dapt.v3.data --cache dapt_v3_sS_i249 …` then val-select among
   each seed's rungs — NOTE: `dapt.select_checkpoint` uses the OLD v1/v2 split, do
   NOT use it for v3; select instead on the k-fold inner-val (run `dapt.v3.kfold`
   with each rung vs web on 1–2 probe seeds and pick the rung with best OOF val
   mAP50, or simply carry both i499/i999 rungs into step 7 and select on inner-val
   there — either is val-only, never on the paired test gap).
7. **K-fold headline (local):** `.venv/bin/python -m dapt.v3.kfold --arms
   <pooled dapt winners…> web sat --k 5 --repeats 3 --seeds 0 1 2 3 4 --capacity mlp`
   (and `--capacity linear`) → `dapt/v3/artifacts/`. Paired gap arm[0]−arm[1];
   subsets reported = FULL/WON/BRU (all arid now).
8. **Report:** append a v3 section to `dapt/REPORT.md`; update the
   `project-dapt-detector` memory.

**Gotchas that already bit us once:** zsh does NOT word-split unquoted vars (pass
arm lists explicitly); Monitor greps must include failure signatures; probe runs are
MPS — don't run two heavy probe jobs concurrently; `train/` tiles must NEVER appear
in a DAPT-arm eval (headline = the 81 `test/` tiles only).

### Planned follow-ups (after v3-arid lands)
- **Restor OAM-TCD DAPT**, two variants: (a) detection — bboxes extracted from ITC
  masks, canopy masks ignored; (b) box-to-mask. Much larger pool ⇒ **same ~$0.50/seed**
  (SSL is step-bound, not epoch-bound; a bigger pool just lowers per-tile reuse).
- **NEON DAPT** — almost certainly valid: web LVD-1689M is curated *internet* images
  (NEON airborne tiles ~certainly absent); sat SAT-493M is *satellite* (NEON is
  airborne AOP, different sensor/altitude — likely absent, worth a quick check). So
  NEON is a fair separate-domain adaptation target, not a leakage risk.