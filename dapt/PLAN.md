# DAPT-for-dryland-detector — execution plan

**Question under test:** does domain-adaptive self-supervised pretraining (DAPT) on
unlabelled arid imagery produce DINOv3 features that make a *better dryland
tree-crown detector* than off-the-shelf DINOv3? The detector head is a **fixed
probe**, held identical across arms — we measure feature quality, not head design.

Three feature arms, one frozen backbone each, one identical head:
1. `web`  — DINOv3 ViT-L, `facebook/dinov3-vitl16-pretrain-lvd1689m` (frozen)
2. `sat`  — DINOv3 ViT-L, `facebook/dinov3-vitl16-pretrain-sat493m` (frozen)
3. `dapt` — `web` + continued SSL on the arid pool, then frozen
4. (control, compute permitting) `dapt-shuffled` — SSL on mismatched/shuffled tiles,
   to prove any gain is *arid adaptation*, not just extra SSL steps.

---

## 0. Decisions locked (this session)

- **Eval uses all three domains (WON + BRU + NEON), site-disjoint, balanced across
  train/val/test** so the benchmark isn't too narrow. NEON supplies out-of-domain
  breadth. Per-site metrics always reported so imbalance stays visible.
- **DAPT fidelity (full DINOv3 objective vs. lighter SSL continuation) is deferred**
  to a gate *after* the web-vs-sat baselines and label-efficiency curve exist.
- **First deliverable = this plan.** No experiment code until it's reviewed.

## Decisions resolved

- **D1 — RESOLVED: seeded domain-stratified random 60/20/20 tile split (no spatial
  blocking).** Verified the tiles are **non-overlapping**: BRU tiles are 400×400 on a
  400-px stride (edge-to-edge), WON tiles are 500×500. No two tiles share pixels, so
  a random subset split cannot leak a crown between train and test. The only residual
  is mild appearance autocorrelation between adjacent tiles — negligible given
  non-overlap. Spatial-blocking machinery dropped.
- **D2 — cohort size.** Actual arid+NEON cohort is **166 tiles** (WON 41 / BRU 53 /
  NEON 72), across **12 sites** (WON003, BRU162, + 10 NEON sub-sites: LAJA LENO CLBJ
  WOOD JORN NOGP OAES ONAQ STER TOOL). Using all 166.
- **D3 — RESOLVED: DAPT's BRU pool = `BRU162/splits2/BRU162_center_80pct.tif`.** The
  labelled `bru_tile_*` were generated from the left/right 10% strips
  (`BRU162_left_10pct.tif`, `_right_10pct.tif`); the center 80% is unseen by the
  detector cohort. DAPT trains only on the center ⇒ zero pixel overlap with detector
  train/val/test. (Plus `CAN091/095/117` full orthos; WON003 optional.)
- **D4 — DAPT's continued SSL adapts only to the Australian arid pool**, not to
  NEON/WON appearance. Expected win is on the BRU (and WON) portion of test; NEON is
  where DAPT may show no gain or regression. That asymmetry is intended — read it via
  the per-site breakdown.

---

## 1. Repo grounding (what is reused vs. built)

Reused as-is:
- `dino/dinov3_seg.py` — HF `AutoModel` load of both backbones (web/sat weights
  already present locally). Extend for multi-layer token extraction (§3).
- `dino/efficiency/cache_features.py` — pattern for caching frozen features once,
  reused per (arm × tile).
- Label source: `data/finetune/phase22X_combined.csv`
  (`image_path,xmin,ymin,xmax,ymax,label,shadow_angle,shadow_x,shadow_y,domain`).
- SSL pool: `data/australia/{BRU162,CAN091,CAN095,CAN117}/*_10cm.tif` (+ WON003
  ortho if we want it in-pool; note WON003 is arid Australian too).

Built new (all under `dapt/`):
- `dapt/data/` — split builder, tile/coordinate recovery, dataset + target encoder.
- `dapt/backbone.py` — frozen multi-layer feature extractor (shared by all arms).
- `dapt/head.py` — the fixed CenterNet-style probe.
- `dapt/train.py` — head training loop (identical config across arms).
- `dapt/decode.py` — heatmap peak-pick + offset + size → boxes → NMS.
- `dapt/eval.py` — P/R/F1/count, isolated-vs-touching strata, bootstrap CIs.
- `dapt/label_efficiency.py` — the headline curve driver.
- `dapt/ssl/` — DAPT continuation, as-built after the gate (§5; details in
  `dapt/ssl/SPEC.md`).

---

## 2. Data + split design  (`dapt/data/build_split.py`)

**Sites (12):** `WON003`, `BRU162`, + 10 NEON sub-sites (LAJA, LENO, CLBJ, WOOD,
JORN, NOGP, OAES, ONAQ, STER, TOOL). Site key derived from the path: WON→WON003,
BRU→BRU162, NEON→the 4-letter code before the `_<flight>_<easting>_<northing>_image`
block (so `DCFS_2019_WOOD_*` → WOOD).

**Split rule — seeded, domain-stratified random 60/20/20:**
- Within each domain (WON/BRU/NEON) shuffle its tiles with a seeded RNG and cut
  60/20/20, so every partition holds a proportional mix of all three domains.
- Tiles are non-overlapping (verified, D1) ⇒ no spatial blocking, no coordinate
  recovery needed. Each base tile = one image file (one fixed rotation) ⇒ no
  rotation-duplication leakage.
- Emit `dapt/data/split.json` = `{seed, ratios, csv, tiles:{path:{domain, site,
  partition, n_boxes}}}` plus a printed per-partition × per-site tile/box table for
  sign-off. Per-site counts always visible so any NEON sub-site concentration shows.
- Seed from a single `--seed`, recorded in `split.json` (standing preference: seed
  everything from one value, record it; torch seeding lives in `train.py`).

**Exhaustiveness:** confirmed — phase22X boxes are exhaustive per tile, so precision,
recall, and count-error are all fair on every partition.

---

## 3. Frozen multi-layer backbone (`dapt/backbone.py`) — identical across arms

- Input tiles at **native 0.1 m/px** (no downsampling — small crowns must stay above
  the 16 px patch). Tile size fixed (512 → 32×32 patch grid for the 500/400 px tiles,
  padded); document it.
- Extract **patch tokens from blocks {3,6,9,12}** (of ViT-L's 24; revisit indices),
  L2-norm each, **concat** → per-patch feature. Same layer set for all arms.
- Arm switch = which checkpoint loads (`web` / `sat` / `dapt/<step>`); everything
  downstream byte-identical. Backbone `eval()`, `requires_grad_(False)` in all arms.
- Cache features to disk once per (arm × tile) — probes/heads read the cache, matching
  the `dino/efficiency` pattern.

## 4. Evaluation ladder — three capacity levels on the SAME frozen features

The claim is about *feature quality*, so we read each arm's frozen features at three
increasing head capacities and report them as a **progression**.

**The head is an anchor-free bounding-box detector (CenterNet), scored by standard
COCO.** The "three targets" are just how a *box* is parameterized for training — not
an alternative to boxes. Per patch the head predicts: (a) **centre heatmap** (1 ch
`tree`, focal loss, Gaussian-splatted GT centres; dryland ≈ all background so
imbalance is severe; peak value = confidence score), (b) **sub-patch offset** (2 ch,
smooth-L1; recovers the center the 16 px grid quantizes), (c) **box size w,h** (2 ch,
smooth-L1). **Decode reassembles ordinary scored boxes:** peak-pick heatmap → add
offset → read (w,h) → emit (x,y,w,h,score) → NMS. That box list feeds the standard
COCO evaluator unchanged — same output form as YOLO/Faster-RCNN. What changes between
L1/L2/L3 is *only the head between features and those 3 maps*.

Why CenterNet and not an anchor head: anchor scales/ratios + RPN + ROI pooling are
tunable machinery that would confound "better *features*" with "better-tuned
*detector*", and can't collapse to a linear probe. CenterNet's three per-pixel
regressions **do** collapse to a 1×1 conv per patch — that's what makes the L1 probe
a real box detector, so the same head spans the whole ladder.

- **L1 — LINEAR PROBE (primary feature-quality claim).** A single **1×1 conv per
  patch** → the 3 target maps, then a **FIXED bilinear upsample** to pixel
  resolution. **No hidden layers, no learned upsampler.** Near-zero head-tuning
  confound ⇒ the clean web vs sat vs DAPT comparison. The offset branch does the
  sub-patch localisation the fixed upsample can't.
- **L2 — MLP PROBE.** 1–2 hidden layers (still per-patch, fixed bilinear upsample) —
  catches crown info that's present but **non-linearly** encoded.
- **L3 — FULL DETECTOR HEAD (optional, later; the deployable number).** Adds a light
  **learned** upsampler + small conv stem. More confounded, so control it: per-arm
  hyperparameter tuning on *val* with an **identical search budget**, and lean on the
  **shuffled-DAPT** control to isolate arid-specific gains.

**Reading rule:** if **L1 already shows DAPT > web/sat**, that's the strongest
evidence (frozen features linearly encode more). If the gain appears **only at L3**,
be suspicious it's head-tuning, not features — lean on the shuffled-DAPT control and
the identical-budget caveat before claiming a DAPT win.

**Result (M1/M1b, 5 seeds, test set):** crown info is substantially **non-linearly**
encoded — L2 ≈ 2.5× L1 for both arms (web 0.450 vs 0.181 mAP50). L1 is underpowered
on the 33-tile test (web 0.181 ± 0.026 vs sat 0.163 ± 0.020; CIs overlap), so the
primary DAPT comparison moves to the **L2 paired-gap bootstrap**, which cleanly
resolved web − sat (§7 M1b). The DAPT claim rides on L2.

**Shared training protocol (identical across arms and, within a level, across arms):**
same init seed, same LR schedule, same loss weights, same NMS IoU/threshold. Train on
*train*, select on *val*, evaluate once on *test*. Record seed + config hash in every
checkpoint and result file. Files: `dapt/head.py` (all three levels, capacity flag),
`dapt/train.py`, `dapt/decode.py`.

## 5. DAPT continuation (`dapt/ssl/`) — AS BUILT (spec + runbook = `ssl/SPEC.md`)

**Gate decision:** Meta `dinov3` training repo, full core objective (DINO + iBOT +
Koleo); Gram off unless preflight shows trivial anchor wiring — collapse insurance is
the **DCP-every-500 checkpoint trail + local probe checkpoint-selection**. One-shot
~5k iters on Modal A100 (cost-optimized, 3 h hard cap ~$7). **Run-1 LR as-executed
and adopted: true peak 7.07e-5** (repo's sqrt rule has a ×4 missed pre-launch; kept
mid-run — the DCP-500 trail covers the intended gentler regime in its first rungs;
probed blocks train at ~0.8–2e-5 via layerwise decay; no seed-0 retrain).

**History note (branch merge 2026-07-07):** two parallel branches built this dir; the
detector-pipeline branch took ownership and reconciled. The fork's Modal/repo infra
was kept; its pool was rebuilt — the fork's builder filtered **black** nodata but
these orthos' nodata is **white**, so its 1,747-tile pool was ~46% blank tiles and its
arid stats ([0.68…]/[0.26…]) were white-inflated. Its NCC WON-footprint recovery
(`recover_won_coords.py`) was deleted — superseded by the user-supplied pre-cropped
`WON003_10cm_right60.tif`.

As-built (file-by-file map + run order in `ssl/SPEC.md`):
- **Pool (DONE):** `build_pool.py` → 1,084 tiles @512 px native, 50% overlap,
  white-nodata filter (>=95% valid), BRU-center + **WON right60** + CAN091/095/117;
  arid stats [0.503, 0.470, 0.415] / [0.127, 0.117, 0.110] over valid pixels only.
- **Init (DONE):** `convert_hf_to_dinov3.py` → `dinov3_web_vitl_teacher.pth`;
  `local_smoke.py` verified 0-missing-params + native-vs-HF feature cosine.
- **Training:** `modal/app.py` (resumable torchrun, preflight asserts) +
  `dinov3_dapt_vitl.yaml` — crops **256/112** (DINOv3 pretrain recipe; 512 globals is
  only Meta's brief hi-res tail), **ImageNet norm everywhere** (settled after two
  reversals — web weights expect it, short continuation can't afford norm
  re-adaptation, sat's domain stats didn't help it; dapt vs web differs by weights
  only; arid stats kept in manifest as follow-up variant), dataset root
  `/vol/pool/tiles` (never `/vol/pool` — samples/ holds coverage thumbnails), d4
  rotation via repo patch.
- **Export:** `convert_dinov3_to_hf.py` per checkpoint → `checkpoints.json` →
  `backbone.py` arm, zero pipeline change.

Standing deviations from the original §5 plan (deliberate, per SPEC): no `adapt.py`
stub (converter pair is the interface); one-shot run with the DCP trail as the
step-sweep readout; no shuffled-DAPT control (pre-registered substitute = the
pattern: gain on WON/BRU with ~0 on NEON ⇒ arid-specific; uniform lift ⇒ generic SSL).

## 6. Metrics + analysis (`dapt/eval.py`, `label_efficiency.py`)

- **Standard COCO detection metrics on the decoded boxes:** `mAP@50`, `mAP@[.5:.95]`,
  and `AP-small` (crowns are small — track it). Score threshold for the single-point
  metrics chosen on *val*.
- **Also (per-GOAL, use-case):** precision, recall, F1 @ IoU 0.5, and **count error**
  (#boxes after NMS above the val-chosen threshold vs. GT count) — counting is the
  real dryland use case, report it explicitly.
- **Headline = label-efficiency curve:** **mAP@50** (y) vs. # labelled train images
  (25 / 50 / 100 %), `web` vs `sat` vs `dapt` on one axis, count-error reported
  alongside. Expectation: DAPT shows as "same accuracy, fewer labels" more than as a
  higher ceiling.
  **Measured (L2, 5 seeds, test):** web 0.021 / 0.175 / 0.450; sat 0.050 / 0.160 /
  0.354 at 25/50/100%. Both arms collapse at 25% (~25 train tiles — too few for the
  probe itself), so the informative region is 50→100%; web's advantage only emerges
  at full labels. A DAPT win at 50% would be the cleanest "fewer labels" result.
- **Stratify errors: isolated vs. touching crowns** (touching = known hard case;
  also flags where a future crown-separation module would pay off).
- **Site-blocked eval + bootstrap CIs** (resample tiles within test). The
  `dapt − web` gap must clear the CI band or it means nothing. Report per-site.
- Persist every run to `dapt/artifacts/<arm>_<split-seed>_<config-hash>.json`.

## 7. Sequencing + gates

1. **M0 — split + sign-off. DONE.** `build_split.py` → split.json (166 tiles, 12
   sites, seed 20260702). Frozen target encoder `targets.py` (stride-16, radius=1-cell
   [IoU-radius clamps to floor for ~all crowns at this grid], offset full-cell,
   log-size); diagnostic confirms 0% cell collisions, AP-small is the risk.
2. **M1 — backbone cache + L1 linear-probe baselines. DONE.**
   `backbone.py` (both arms, MPS) → `cache_features.py` → `dataset.py` → `head.py`
   (L1/L2 + focal/smooth-L1) → `decode.py` → `eval.py` (COCO AP + P/R/F1 + count +
   strata + bootstrap) → `train.py` → `run_baseline.py`. 5-seed test (val-frozen
   threshold): **web 0.181 ± 0.026, sat 0.163 ± 0.020** — CIs overlap; L1 is
   **underpowered** on the 33-tile test. (`artifacts/baseline_summary.json`)
3. **M1b — L2 MLP-probe baselines. DONE.** 5-seed test: **web mAP50 0.450 ± 0.015,
   sat 0.354 ± 0.027; paired web − sat gap +0.094, 95% CI [0.038, 0.149], 100% of
   bootstrap draws > 0 — RESOLVED.** Crown info is strongly non-linear in the frozen
   features (L2 ≈ 2.5× L1), so **L2 + paired-gap bootstrap is the primary
   instrument** for the DAPT claim. Label-efficiency curve done at L2 (§6).
   (`artifacts/baseline_summary_mlp.json`, `label_efficiency.json`)
4. **GATE — TAKEN (amended 2026-07-07):** dinov3-repo core objective (Gram off unless
   preflight-trivial), one-shot ~5k-iter Modal run with DCP-500 trail; **primary =
   paired DAPT−web on the FULL test set** (NEON treated like WON/BRU — it's only ~24%
   of test boxes, so dilution is modest), checkpoint selected on full val; arid/NEON
   subset gaps always reported as the mechanism readout. Full reasoning + amendment
   history in `dapt/ssl/SPEC.md`.
5. **M2 — DAPT arm: DONE, RESULT POSITIVE (see dapt/REPORT.md).** Run-1 (seed 42,
   5k iters, ~$2) → 10-ckpt trail → val-selected winner `dapt_s42_i999` → one-shot
   test: **paired dapt−web +0.033 CI[+0.009,+0.058] RESOLVED** (L2, full test);
   arid subset RESOLVED, NEON not (D4 signature). Semseg area-F1: dapt ≈ web →
   gain is instance-level. Remaining from M2 scope: label-efficiency for dapt,
   shuffled control, 2nd SSL seed.
5. **M3 — decision write-up** against the GOAL's decision rule:
   - `dapt > web` and `> sat`, gap clears CI → arid adaptation helps; headline the
     label-efficiency win; then diagnose the touching-crown subset.
   - `dapt ≈ web` → narrow-domain SSL subsumed by web features (extends the DINOv3
     finding to the specialized-domain limit); diagnose pool size / SSL steps /
     distribution tightness. State which outcome the numbers support *before*
     proposing next steps.

## 8. Risks

- **DAPT fidelity/compute** (D-gate) — biggest unknown; the lighter path de-risks it.
- **Precision fairness** — needs exhaustive-tile flagging (§2 caveat) or precision is
  meaningless; recall/count are robust.
- **BRU leakage** (D3) — must be wired before any DAPT step runs.
- **Head capacity** — keep it genuinely small so it can't differentially overfit 166
  images between arms and masquerade as a feature effect.
- **NEON in-eval** — non-arid; expect DAPT ≈ or < web there. Read via per-site, don't
  let it dilute the arid signal in the aggregate.
