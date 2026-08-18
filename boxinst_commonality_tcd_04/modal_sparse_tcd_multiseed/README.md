# `modal_sparse_tcd_multiseed` — zero-shot eval on unseen open-canopy TCD

**Question.** The settled phase-4 model scores mask mAP50 **0.630 ± 0.005** on the official 439 — a
cohort ~80% closed-canopy forest, matching its 900 training tiles. What does the *same, unmodified*
model do on **open canopy** (savanna, grassland, shrubland, desert) it has never seen?

No retraining, no masker refit, no re-tuning: same checkpoints, same `em_model_4p_fix.npz`, same
α=0.3 / κ×1.6 knobs, same `mask_thr=0.25`.

## Isolation

Own Modal app (`tcd-sparse-l24`) and own Volume (`tcd-sparse-vol`). The phase-4 volume
`tcd04-phase4-vol` is mounted at `/phase4` **for reading only** — `vol.commit()` is called on the
sparse volume and never on it, so nothing here can move the 0.630 result. Phase-4 *libraries* are
imported, never edited, so the eval path is bit-for-bit the one behind the headline.
`rm -rf` this folder + `modal volume delete tcd-sparse-vol` is a complete tidy-up.

## Files

| file | role |
|---|---|
| `tile_index.py` | 4608-tile index from local `_meta.json`: biome, RLE-safe coverage, 300 m scene clusters, seen-900 flag. `report()` prints the biome inventory. |
| `contact_sheet.py` | 3 random unseen tiles per biome with GT crowns drawn — the evidence for the sparse/not-sparse call |
| `build_slice.py` | selection + the four exclusion gates + the copy into `data/tcd_sparse` |
| `sparse_modal.py` | the A100 app: `verify` · `extract_4p` · `eval_sparse` · `band_sparse` · `score_subsets` |
| `manifest_sparse.json` | phase-4-shaped manifest (`feat_test` only) — joins to HF `restor/tcd` by `image_id`, never by filename |
| `tile_index.json`, `rgb_sha1.json` | caches; regenerable |

## Run order

```
modal run sparse_modal.py::verify                      # free, no GPU — join + pixel check
modal run sparse_modal.py::extract_4p                  # ~$0.7, ~20 min, 4.8 s/tile
modal run sparse_modal.py::band_sparse --seeds 0        # the testing seed
modal run sparse_modal.py::band_sparse --seeds 0,1,2   # final 3-seed band (seed 0 reused)
modal run sparse_modal.py::score_subsets --seeds 0,1,2 # CPU — the honest cuts
```

**Seed convention.** Seed **0** is the testing seed; **1 and 2** are added only for a final
3-seed band. Both entrypoints evaluate seeds in the order given and are per-seed idempotent, so
`--seeds 0,1,2` reuses a completed seed-0 rather than recomputing it.

**Comparator.** This eval runs the masker's stored knobs (α=0.3, κ×1.6) @ `mask_thr` 0.25, so the
439 reference in the SAME configuration is **seed-0 0.6250 / 0.2574 / box 0.6050** and the 3-seed
band **0.630 ± 0.005 / 0.257 / 0.603**. `score_subsets` picks whichever matches the run.
The vanilla α=1/κ=1 arm (seed-0 0.6203, 5-seed 0.615) is a *different configuration* and is not the
comparator here.

## Gates that must hold

- **`verify`** pixel-checks **all 236** tiles against HF (phase-4's own verify samples 20). Also
  confirms the local copies in `data/tcd_sparse` are byte-identical to the HF source.
- **`extract_4p`** runs the registration self-test *and* the layers-trap parity gate before
  extracting: cos vs the known-good slice must be **0.86003**, the exact value phase-4 recorded.
  A drift here means the features no longer match what the frozen checkpoints were trained on.
- **`score_subsets`** reuses phase-4's own `_full_metrics`. Validated by re-scoring the saved 439
  predictions: reproduces the published 0.6203 / 0.2436 / 0.605 / 0.6692 **exactly**.

## Reading the result

`score_subsets` reports four cuts, because the single 236-tile number is the optimistic one:

| cut | n | what it answers |
|---|---|---|
| `all` | 236 | the headline, but 130 tiles share a 300 m scene with a training tile |
| **`scene_clean`** | **106** | **the honest number** — no shared ortho with training |
| `train_pool` | 220 | drops the 16 tiles that were used to tune `mask_thr` and α/κ |
| `scene_clean_and_train_pool` | — | both restrictions at once — the strictest |
| `biome_{7,8,9,10,12,13}` | — | diagnostic: biome 7 is the genuinely open one |

Compare against the 439 reference carried in the output (seed-matched — see *Comparator* above).

**A drop is the expected result, not a failure** — it is the domain-gap measurement this exists to
make. Note also that GT density differs sharply by biome (crowns/tile 16.2 in biome 10 vs 105.5 in
biome 12), so per-biome AP is noisy on small tile counts.

Slice provenance, caveats and the exclusion argument: `data/tcd_sparse/README.md`.
