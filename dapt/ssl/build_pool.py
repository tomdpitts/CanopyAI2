"""Step 1 — build the arid SSL pool (no labelled pixels, no empty space).

Slides a 256 px window over each arid ortho at native 0.1 m/px, KEEPS only tiles that
are >= min_valid real content, and materializes them as PNGs for a trivial Modal
ImageFolder dataloader. Nodata is pure WHITE (>= white_thr on all channels) — the
orthos have large white corners (WON ~41%, CAN 36-53%); those tiles are dropped and
white pixels are excluded from the arid RGB mean/std.

Leakage safety, two modes (--won-mode):
  right60 (pool v1): BRU162 center-80, WON003 right60 crop (all labelled tiles
    removed), CAN* unlabelled. No per-tile masks.
  full-masked (pool v2, default): WON003 FULL ortho, excluding buffered footprints of
    the VAL+TEST tiles only (dapt/ssl/won_footprints.json, NCC-recovered + resynthesis-
    verified; --buffer absorbs the observed <=~100px grid-snap offsets). TRAIN-tile
    pixels are deliberately INCLUDED: SSL is label-free, the probe already has full
    supervised access to those tiles, and SSL corpora containing the downstream train
    split is the standard convention (e.g. SSL-on-ImageNet -> probe-on-ImageNet-train).
    Only test (claim validity) and val (selection hygiene) must stay out. A kept tile
    intersecting ANY excluded rect by >=1px is a hard error (asserted).

Augmentation contract (Modal): tiles are square, so d4 transforms (k*90 rotations +
flips) are corner-safe and FREE (no nodata introduced) — the valid rotation aug for
gravity-free overhead imagery. Arbitrary-angle rotation is NOT used (it injects white
corners). Photometric aug + random-resized-crop apply on top.

Outputs (all under dapt/ssl/pool/):
  tiles/<site>/<site>_x<x>_y<y>.png   materialized valid tiles
  manifest.json                        tiles, config, arid mean/std, per-ortho counts
  samples/_coverage_<site>.png         keep(green)/reject(red) grid thumbnail

Usage:
    .venv/bin/python -m dapt.ssl.build_pool
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# (site, repo-relative ortho, why it is leakage-safe)
BASE_ORTHOS = [
    ("BRU162", "data/australia/BRU162/splits2/BRU162_center_80pct.tif",
     "center-80; labels are the L/R 10% strips"),
    ("CAN091", "data/australia/CAN091/CAN091_10cm.tif", "unlabelled"),
    ("CAN095", "data/australia/CAN095/CAN095_10cm.tif", "unlabelled"),
    ("CAN117", "data/australia/CAN117/CAN117_10cm.tif", "unlabelled"),
]
WON_V1 = ("WON003", "data/australia/WON003/WON003_10cm_right60.tif",
          "right60 crop; all labelled tiles removed")
WON_V2 = ("WON003", "data/australia/WON003/WON003_10cm.tif",
          "FULL ortho; buffered val+test footprints excluded, train included")


def won_exclusions(buffer):
    """Buffered (x0,y0,x1,y1) ortho-space rects for WON val+test tile footprints."""
    import re
    fp = json.load(open(os.path.join(REPO, "dapt/ssl/won_footprints.json")))
    split = json.load(open(os.path.join(REPO, "dapt/data/split.json")))
    vt = set()
    for path, t in split["tiles"].items():
        if t["domain"] == "WON" and t["partition"] in ("val", "test"):
            vt.add(int(re.search(r"won_tile_(\d+)_", path).group(1)))
    rects = []
    for r in fp["results"]:
        if r.get("kind") == "rotated" and r["n"] in vt:
            rects.append((r["tile_x0"] - buffer, r["tile_y0"] - buffer,
                          r["tile_x0"] + 500 + buffer, r["tile_y0"] + 500 + buffer))
    assert len(rects) == len(vt), f"footprints missing for {len(vt)-len(rects)} tiles"
    return rects


def _intersects(x, y, tile, rects):
    return any(x < rx1 and x + tile > rx0 and y < ry1 and y + tile > ry0
               for rx0, ry0, rx1, ry1 in rects)


def build(tile, stride, min_valid, white_thr, out_dir, won_mode="full-masked",
          buffer=150):
    orthos = list(BASE_ORTHOS)
    exclusions = {}
    if won_mode == "right60":
        orthos.insert(1, WON_V1)
    else:
        orthos.insert(1, WON_V2)
        exclusions["WON003"] = won_exclusions(buffer)
        print(f"[WON003] excluding {len(exclusions['WON003'])} buffered val+test "
              f"footprints (buffer={buffer}px)")
    return _build(tile, stride, min_valid, white_thr, out_dir, orthos, exclusions,
                  won_mode, buffer)


def _build(tile, stride, min_valid, white_thr, out_dir, ORTHOS, exclusions,
           won_mode, buffer):
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    manifest_tiles = []
    ch_sum = np.zeros(3, np.float64)      # running stats over VALID pixels, in [0,1]
    ch_sqsum = np.zeros(3, np.float64)
    ch_n = 0
    per_ortho = {}

    for site, rel, note in ORTHOS:
        arr = np.asarray(Image.open(os.path.join(REPO, rel)).convert("RGB"))
        H, W = arr.shape[:2]
        valid = ~(arr >= white_thr).all(-1)               # (H,W) real-content mask
        site_dir = os.path.join(out_dir, "tiles", site)
        os.makedirs(site_dir, exist_ok=True)
        ys = list(range(0, H - tile + 1, stride))
        xs = list(range(0, W - tile + 1, stride))
        cover = np.zeros((len(ys), len(xs), 3), np.uint8)
        cover[:] = (180, 0, 0)
        rects = exclusions.get(site, [])
        kept = n_excluded = 0
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                if rects and _intersects(x, y, tile, rects):
                    cover[iy, ix] = (0, 0, 200)          # blue = leakage-excluded
                    n_excluded += 1
                    continue
                vtile = valid[y:y + tile, x:x + tile]
                if vtile.mean() < min_valid:
                    continue
                crop = arr[y:y + tile, x:x + tile]
                Image.fromarray(crop).save(
                    os.path.join(site_dir, f"{site}_x{x}_y{y}.png"))
                manifest_tiles.append(f"{site}/{site}_x{x}_y{y}.png")
                cover[iy, ix] = (0, 180, 0)
                sv = (crop.reshape(-1, 3) / 255.0)[vtile.reshape(-1)]
                ch_sum += sv.sum(0)
                ch_sqsum += (sv ** 2).sum(0)
                ch_n += sv.shape[0]
                kept += 1
        total = len(ys) * len(xs)
        per_ortho[site] = {"kept": kept, "total": total, "WxH": [W, H], "note": note,
                           "excluded_cells": n_excluded}
        # belt: no kept tile may intersect any exclusion rect
        if rects:
            for t in manifest_tiles:
                if not t.startswith(site + "/"):
                    continue
                tx, ty = (int(v[1:]) for v in
                          os.path.splitext(t)[0].split("_")[-2:])
                assert not _intersects(tx, ty, tile, rects), f"LEAKAGE: {t}"
        Image.fromarray(cover).resize((cover.shape[1] * 6, cover.shape[0] * 6),
                                      Image.NEAREST).save(
            os.path.join(out_dir, "samples", f"_coverage_{site}.png"))
        print(f"[{site}] {W}x{H}  kept {kept}/{total} ({100*kept/max(total,1):.0f}%)",
              flush=True)

    mean = (ch_sum / ch_n)
    std = np.sqrt(ch_sqsum / ch_n - mean ** 2)
    manifest = {
        "pool_version": "v1" if won_mode == "right60" else "v2",
        "won_mode": won_mode, "won_exclusion_buffer_px": buffer,
        "tile": tile, "stride": stride, "min_valid": min_valid,
        "white_thr": white_thr, "n_tiles": len(manifest_tiles),
        "arid_mean": mean.tolist(), "arid_std": std.tolist(), "valid_px": int(ch_n),
        "aug": "d4 (k*90 rot + flip) corner-safe; no arbitrary-angle rotation",
        "per_ortho": per_ortho, "tiles": manifest_tiles,
    }
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    print(f"\nPOOL {len(manifest_tiles)} tiles  "
          f"arid_mean={np.round(mean,4).tolist()} std={np.round(std,4).tolist()}")
    print(f"(ImageNet ref mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225])")
    print(f"wrote {os.path.relpath(os.path.join(out_dir,'manifest.json'), REPO)} "
          f"(+ tiles/, samples/)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)     # =tile -> non-overlap
    ap.add_argument("--min_valid", type=float, default=0.95)
    ap.add_argument("--white_thr", type=int, default=250)  # nodata >= this on all ch
    ap.add_argument("--won-mode", default="full-masked",
                    choices=["right60", "full-masked"])
    ap.add_argument("--buffer", type=int, default=150,
                    help="px margin around excluded WON val+test footprints")
    ap.add_argument("--out", default="dapt/ssl/pool_v2")
    args = ap.parse_args()
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    build(args.tile, args.stride, args.min_valid, args.white_thr, out_dir,
          args.won_mode, args.buffer)


if __name__ == "__main__":
    main()
