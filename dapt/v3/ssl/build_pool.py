"""v3 DAPT SSL pool builder — arid-only, leakage-safe by construction.

Pool orthos (all leakage-safe for the 62 v3 test tiles):
  BRU162_center_80pct, CAN091/095/117, WON003_10cm_right50 (pre-cropped so labelled
  tiles are removed — no footprint masking needed).

Tiles the orthos at 512px/50% overlap at native 0.1 m/px, drops tiles with < min_valid
real content (nodata = pure WHITE, the orthos have large white corners), materializes
PNGs under dapt/v3/ssl/pool/tiles/<site>/ for the Modal ImageFolder dataloader, and
writes manifest.json (arid RGB mean/std over valid pixels).

Leakage guard: asserts NONE of the 62 v3 test tiles could originate inside these orthos
by construction (WON uses right50 = the test-excluded crop; BRU center-80; CAN
unlabelled). We additionally re-affirm the exclusion contract in the manifest.

Usage:
    .venv/bin/python -m dapt.v3.ssl.build_pool --tile 512 --stride 256
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

ORTHOS = [
    ("BRU162", "data/australia/BRU162/splits2/BRU162_center_80pct.tif",
     "center-80; labels are the L/R 10% strips"),
    ("CAN091", "data/australia/CAN091/CAN091_10cm.tif", "unlabelled"),
    ("CAN095", "data/australia/CAN095/CAN095_10cm.tif", "unlabelled"),
    ("CAN117", "data/australia/CAN117/CAN117_10cm.tif", "unlabelled"),
    ("WON003", "data/australia/WON003/WON003_10cm_right50.tif",
     "right50 crop; labelled/test tiles removed by construction"),
]
OUT = "dapt/v3/ssl/pool"


def tile_ortho(path, site, out_dir, tile, stride, min_valid, white_thr):
    im = np.asarray(Image.open(os.path.join(REPO, path)).convert("RGB"))
    H, W = im.shape[:2]
    os.makedirs(os.path.join(out_dir, site), exist_ok=True)
    kept, sums, sqs, npix = 0, np.zeros(3), np.zeros(3), 0
    for y in range(0, H - tile + 1, stride):
        for x in range(0, W - tile + 1, stride):
            t = im[y:y + tile, x:x + tile]
            white = np.all(t >= white_thr, axis=2)
            valid = 1.0 - white.mean()
            if valid < min_valid:
                continue
            Image.fromarray(t).save(
                os.path.join(out_dir, site, f"{site}_x{x}_y{y}.png"))
            m = ~white
            v = t[m].astype(np.float64) / 255.0
            sums += v.sum(0)
            sqs += (v ** 2).sum(0)
            npix += v.shape[0]
            kept += 1
    return kept, sums, sqs, npix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--min_valid", type=float, default=0.95)
    ap.add_argument("--white_thr", type=int, default=250)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    out_dir = os.path.join(REPO, args.out, "tiles")
    tot, S, Sq, N = 0, np.zeros(3), np.zeros(3), 0
    per_site = {}
    for site, path, note in ORTHOS:
        k, s, sq, n = tile_ortho(path, site, out_dir, args.tile, args.stride,
                                 args.min_valid, args.white_thr)
        per_site[site] = {"tiles": k, "ortho": path, "note": note}
        tot += k
        S += s
        Sq += sq
        N += n
        print(f"{site:8s} {k:4d} tiles  ({note})", flush=True)

    mean = (S / N)
    std = np.sqrt(Sq / N - mean ** 2)
    manifest = {
        "study": "v3", "tile": args.tile, "stride": args.stride,
        "min_valid": args.min_valid, "n_tiles": tot, "per_site": per_site,
        # QA diagnostics only (white-filter sanity). Normalization is ImageNet, final
        # — the old arid-stats norm variant is RETIRED (arid-only study, no NEON).
        "rgb_stats_qa": {"mean": [round(x, 4) for x in mean],
                         "std": [round(x, 4) for x in std]},
        "norm": "ImageNet (final)",
        "leakage_contract": "orthos are the v3 test-excluded crops (WON right50, "
        "BRU center80, CAN unlabelled); the 62 data/finetune/v3/test tiles cannot "
        "originate here by construction.",
    }
    json.dump(manifest, open(os.path.join(REPO, args.out, "manifest.json"), "w"),
              indent=2)
    print(f"\nTOTAL {tot} tiles | QA rgb mean={manifest['rgb_stats_qa']['mean']} "
          f"std={manifest['rgb_stats_qa']['std']}")
    print(f"wrote {args.out}/manifest.json")


if __name__ == "__main__":
    main()
