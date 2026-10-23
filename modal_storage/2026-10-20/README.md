# canopyai-deepforest-data recovery — 2026-10-20

**What happened.** During the storage cleanup Tom deleted the Modal volume `canopyai-deepforest-data` (81 GB) by hand. It was still live: about 15 Modal entrypoints mount it at `/hfdata` with `HF_HOME=/hfdata/hf_cache, HF_DATASETS_OFFLINE=1` and call `load_dataset("restor/tcd")`. They include phase4, Restor, Detectree2, SAM 3, SelvaBox, gtbox, the native-raster rows and Tier B. The scripts only ever read `/hfdata/hf_cache`. The volume's other top-level dirs (`images/` 53 GB, `holdout/`, `phase*/`, `tcd_images.tar.gz`) are referenced by no current code and are **not** recovered.

**Rebuild** (`rebuild_hfdata.py`):
- Revision `d97d4da0ebbb6e249ae95ac5e19656babd972eb2`: the commit in every logged cache path, and still hub HEAD on 2026-10-20 (no commits since 2024-11-01).
- `datasets==4.0.0` and `numpy==2.2.6`, matching the Modal image pins.
- Cache path reproduced exactly: `datasets/restor___tcd/default/0.0.0/d97d4da…/tcd-{train-0000N-of-00006,test}.arrow`.
- Rows: 4169 train and 439 test.
- Local copy: `data/hf_cache_restor_tcd/` (6.9 GB, gitignored).
- Uploaded to `canopyai-deepforest-data:/hf_cache` (log: `upload_hf_cache.log`).

**Verification** (`verify.jsonl` per tile, `verify_summary.json`):
- **4605/4608** tiles pixel-identical to the local `data/tcd/{train,test}/*.tif`. The check compares the sha1 of decoded RGB, the same recipe as `build_manifest.py`, on every row, not just the manifest's 20/split sample.
- `coco_annotations` string, width/height and split identical for **all 4608**.
- `phase4/manifest.json` rgb_sha1: **40/40**. `manifest_sparse.json`: **236/236**.
- The 3 tiles that could not be checked (`train/tcd_tile_121`, `_1878`, `_231` = image_id 0, 1, 2) are **zero-byte local files** (likely from the 08-13 machine migration), not HF differences. None of them is in the 900/439 cohort. `data/tcd/experimental_ignore/top20_dryland/06_train_tcd_tile_231.tif` is also empty.

**Not recovered:** the rest of the old 20.3 GB `hf_cache` (the new one is 6.9 GB). It was most likely hub model weights pulled at run time. The scripts set `HF_HUB_OFFLINE=0` with the `huggingface` secret, so models re-download on first use.

**Still to do once the Modal spend limit is lifted:** run one in-container smoke test, e.g. `phase4_modal.py`'s join check, which asserts the image_id join plus the pixel-sha1 sample inside the real image.
