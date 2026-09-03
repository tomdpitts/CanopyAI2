# Prediction archive — LACE 439 seeds

**Rule: a prediction file that took GPU time to make never lives only in a session scratchpad.**
It goes on the Modal volume that produced it AND is copied here in the same session. Twice now a
figure could not be recomputed because masks were discarded (see `feedback-persist-masks`), and on
2026-09-02 the LACE seed 1/2 mask-IoU column had to be left as "seed 0 only" because these files
were believed lost. They were not lost — they were on `tcd04-phase4-vol` the whole time and simply
never pulled. Look on the volume BEFORE concluding anything is gone.

| file | seed | source on Modal | verified |
|---|---|---|---|
| `preds_knobbed_s0_product.json` | 0 | posterior-product rerank, built locally | Table 1 row 0.6615 |
| `preds_knobbed_s0_reranked.json` | 0 | 8-feat logreg rerank (superseded) | Table 1 row 0.6616 |
| `preds_knobbed_s1.json` | 1 | `tcd04-phase4-vol:out/preds_selfmask_fix_pw030_ks160_thr025_phase4_L24_s1/preds.json` | re-scored 2026-09-02 → **0.6358**, exactly the published seed-1 row |
| `preds_knobbed_s2.json` | 2 | `tcd04-phase4-vol:out/preds_selfmask_fix_pw030_ks160_thr025_phase4_L24_s2/preds.json` | published seed-2 row 0.6293 |
| `preds_knobbed_s1_product.json` | 1 | rebuilt from the above + `em_post_test_s1.json` | re-scored → **0.6639** and **162,017** dets, both exactly the published row |
| `preds_knobbed_s2_product.json` | 2 | rebuilt from the above + `em_post_test_s2.json` | re-scored → **0.6622** and **177,598** dets, both exactly the published row |

The `_product` files were rebuilt on 2026-09-02 with `apply_product.py` from the recovered raw
predictions plus `em_post_test_s{1,2}.json`; score them with `--score_floor 0` (the product is on
a different scale from the 0.05 decode floor — see that script's docstring). Reproducing both the
AP and the detection count exactly is what establishes these are the published artefacts and not
a lookalike run. With all three seeds present the Table 1 mask-IoU column can carry a proper
3-seed band instead of "seed 0 only".

`preds_*` schema: `{meta, preds{tile: {boxes_2048, scores, canopy_ignore, masks_rle}}}`, masks as
RLE at 512. Retrieve anything else with:

```bash
.venv/bin/modal volume ls  tcd04-phase4-vol out
.venv/bin/modal volume get tcd04-phase4-vol out/<path> <dest>
```
