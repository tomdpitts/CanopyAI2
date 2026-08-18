"""Build val_gt.json — crown-polygon GT for the 108 VAL tiles (held-out knob tuning).

The 900 train tiles carry only BOXES in train_tiles_gt.json (the method is box-supervised),
so masker knobs could not be tuned off-test. The source OAM-TCD train metas do carry the
crown polygons; this script reads them for the val tids ONLY, in the exact format/convention
of prepare_test.py (`trees` = largest contour per ITC ann, `canopy` = all ignore rings), so
val mask-AP is scored by the identical core as the 439 test.

VAL ONLY by construction (partition=='val' in train_tiles_gt.json) — the 792 train tiles are
never read, so no mask label ever enters detector or masker fitting.

Usage: .venv/bin/python -m boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.make_val_gt
"""
import json
import os

from boxinst_commonality_tcd_04 import prepare_test as PT

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
TRAIN_DIR = os.path.join(PT.REPO, "data/tcd/train")
OUT = os.path.join(HERE, "val_gt.json")


def main():
    tg = json.load(open(os.path.join(PKG, "train_tiles_gt.json")))
    vals = sorted(t for t, v in tg.items() if v["partition"] == "val")
    gt, n_tree, n_can, n_empty = {}, 0, 0, 0
    for k, tid in enumerate(vals):
        meta = json.load(open(os.path.join(TRAIN_DIR, tid + "_meta.json")))
        H, W = meta["height"], meta["width"]
        trees, canopy = [], []
        for a in PT.anns_of(meta, 2):
            rs = [r for r in PT.seg_rings(a["segmentation"], H, W) if len(r) >= 6]
            if rs:
                trees.append(max(rs, key=len))
        for a in PT.anns_of(meta, 1):
            canopy += [r for r in PT.seg_rings(a["segmentation"], H, W) if len(r) >= 6]
        gt[tid] = {"trees": trees, "canopy": canopy, "W": W, "H": H}
        n_tree += len(trees); n_can += len(canopy); n_empty += (not trees)
        if (k + 1) % 25 == 0 or k + 1 == len(vals):
            print(f"  {k+1}/{len(vals)}", flush=True)
    json.dump(gt, open(OUT, "w"))
    print(f"val: {len(gt)} tiles ({n_empty} with no ITC), {n_tree} crown rings, "
          f"{n_can} canopy rings -> {OUT}")
    # box-count parity vs the box-only train GT (sanity: same anns, different geometry)
    nb = sum(len(tg[t]["boxes"]) for t in vals)
    print(f"box-GT count for the same tiles: {nb} (crown rings {n_tree}; "
          f"diff = degenerate/iscrowd anns)")


if __name__ == "__main__":
    main()
