"""Validate the TCD data layer against the recorded sparse provenance.

Proves the RLE-aware decoder is correct *without* any model weights:
  - crown count (cat=2)        == provenance `trees`     (exact)
  - canopy_frac (cat=1 union)  ~= provenance `canopy_frac` (tol 1e-3)
Also sanity-checks the COCO GT builder and whether PIL can read the GeoTIFFs
(decides if the run script needs rasterio/tifffile).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tcd_data import load_tile, load_split  # noqa: E402
from eval import build_coco_gt  # noqa: E402

TEST_DIR = "data/tcd/test"
PROV = "data/tcd/experimental/sparse/_SUBSET_PROVENANCE.json"


def main():
    prov = json.load(open(PROV))["tiles"]
    prov = {r["stem"]: r for r in (prov if isinstance(prov, list) else prov.values())}
    print(f"[prov] {len(prov)} tiles with recorded trees/canopy_frac")

    n_ok_trees = n_ok_frac = n = 0
    worst_frac = 0.0
    for stem, rec in prov.items():
        mp = os.path.join(TEST_DIR, f"{stem}_meta.json")
        if not os.path.exists(mp):
            continue
        t = load_tile(mp)
        n += 1
        n_ok_trees += (len(t.crowns) == rec["trees"])
        df = abs(t.canopy_frac() - rec["canopy_frac"])
        worst_frac = max(worst_frac, df)
        n_ok_frac += (df < 1e-3)

    print(f"[validate] tiles checked        : {n}")
    print(f"[validate] crown-count exact     : {n_ok_trees}/{n}")
    print(f"[validate] canopy_frac (<1e-3)   : {n_ok_frac}/{n}  (worst |Δ|={worst_frac:.2e})")

    # COCO GT builder smoke + semantic mask shape on the full test split
    tiles = load_split(TEST_DIR)
    gt = build_coco_gt(tiles)
    n_crown = sum(a["iscrowd"] == 0 for a in gt["annotations"])
    n_ignore = sum(a["iscrowd"] == 1 for a in gt["annotations"])
    print(f"[coco-gt] images={len(gt['images'])} crown(iscrowd0)={n_crown} canopy(iscrowd1)={n_ignore}")
    sm = tiles[0].semantic_mask()
    print(f"[semantic] tile0 mask shape={sm.shape} fg_frac={sm.mean():.4f}")

    # can PIL read the GeoTIFF? (decides reader dep for the run)
    try:
        from PIL import Image
        arr = np.asarray(Image.open(tiles[0].image_path).convert("RGB"))
        print(f"[image] PIL read OK shape={arr.shape} dtype={arr.dtype} range=({arr.min()},{arr.max()})")
    except Exception as e:
        print(f"[image] PIL FAILED ({type(e).__name__}: {str(e)[:80]}) -> run script needs rasterio/tifffile")

    ok = (n_ok_trees == n and n_ok_frac == n)
    print(f"\n>>> DATA LAYER {'VALIDATED' if ok else 'MISMATCH -- investigate'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
