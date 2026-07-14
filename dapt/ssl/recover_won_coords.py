"""Jigsaw the data/finetune/v3 WON tiles back onto the WON003 orthomosaic.

The 41 `won_tile_{N}_rot{DEG}.png` tiles are axis-aligned 500px grid crops of
WON003_10cm.tif that were rotated by DEG (deg) about their centre AFTER
cropping (post hoc, corners filled black) — so the true ortho footprint of
every tile is the axis-aligned 500px square, not a rotated polygon. Positions
are not recorded anywhere, so we recover them by template matching:

  1. rotate the tile back to 0 deg (both sign conventions tried, best NCC kept),
  2. centre-crop the fully-valid core (500/sqrt(2) ~ 353 px -> use 348),
  3. coarse NCC (TM_CCOEFF_NORMED) against a 1/4-res ortho,
  4. refine at full resolution in a +/-40 px window around the coarse peak.

The 10 `WON003_tl_x{X}_y{Y}.png` tiles carry their coords in the filename and
are only NCC-verified in place. Outputs:
  claude_outputs/won_footprints.json        recovered coords + NCC per tile
  claude_outputs/won_tiles_unrotated_v2.png footprints drawn on 1/4-res ortho
"""
import glob
import json
import os
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORTHO = os.path.join(REPO, "data/australia/WON003/WON003_10cm.tif")
V3 = os.path.join(REPO, "data/finetune/v3")
OUT_JSON = os.path.join(REPO, "claude_outputs/won_footprints.json")
OUT_PNG = os.path.join(REPO, "claude_outputs/won_tiles_unrotated_v2.png")

TILE = 500
CORE = 348                # centre crop side, < 500/sqrt(2), valid for any angle
COARSE_DS = 4
REFINE_WIN = 40           # +/- px around upscaled coarse peak


def gray(a):
    return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)


def match(ortho_g, ortho_g_ds, tpl_g):
    """Coarse-to-fine TM_CCOEFF_NORMED. Returns (ncc, x, y) of template top-left."""
    tpl_ds = cv2.resize(tpl_g, None, fx=1 / COARSE_DS, fy=1 / COARSE_DS,
                        interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(ortho_g_ds, tpl_ds, cv2.TM_CCOEFF_NORMED)
    _, _, _, (cx, cy) = cv2.minMaxLoc(res)
    x0 = max(0, cx * COARSE_DS - REFINE_WIN)
    y0 = max(0, cy * COARSE_DS - REFINE_WIN)
    th, tw = tpl_g.shape
    win = ortho_g[y0:y0 + th + 2 * REFINE_WIN, x0:x0 + tw + 2 * REFINE_WIN]
    res = cv2.matchTemplate(win, tpl_g, cv2.TM_CCOEFF_NORMED)
    _, ncc, _, (fx, fy) = cv2.minMaxLoc(res)
    return float(ncc), int(x0 + fx), int(y0 + fy)


def main():
    ortho = np.array(Image.open(ORTHO).convert("RGB"))
    H, W = ortho.shape[:2]
    ortho_g = gray(ortho)
    ortho_g_ds = cv2.resize(ortho_g, None, fx=1 / COARSE_DS, fy=1 / COARSE_DS,
                            interpolation=cv2.INTER_AREA)
    results = []

    # -- rotated tiles: position unknown, recover by NCC ---------------------
    rot_paths = sorted(glob.glob(os.path.join(V3, "won_tile_*_rot*.png")))
    for p in rot_paths:
        name = os.path.basename(p)
        n, deg = map(int, re.match(r"won_tile_(\d+)_rot(\d+)\.png", name).groups())
        tile = Image.open(p).convert("RGB")
        best = None
        for sign in (+1, -1):
            unrot = tile.rotate(sign * deg, resample=Image.BICUBIC)
            c = (TILE - CORE) // 2
            core = gray(np.array(unrot)[c:c + CORE, c:c + CORE])
            ncc, x, y = match(ortho_g, ortho_g_ds, core)
            if best is None or ncc > best[0]:
                best = (ncc, x, y, sign)
        ncc, x, y, sign = best
        cx, cy = x + CORE / 2, y + CORE / 2  # tile centre in ortho px
        results.append(dict(file=name, kind="rotated", n=n, rot_deg=deg,
                            undo_sign=sign, ncc=round(ncc, 4),
                            center_x=cx, center_y=cy,
                            tile_x0=int(cx - TILE / 2), tile_y0=int(cy - TILE / 2)))
        print(f"{name:32s} ncc={ncc:.3f} sign={sign:+d} centre=({cx:.0f},{cy:.0f})",
              flush=True)

    # -- axis-aligned tl tiles: coords in filename, verify in place ----------
    for p in sorted(glob.glob(os.path.join(V3, "WON003_tl_x*_y*.png"))):
        name = os.path.basename(p)
        x0, y0 = map(int, re.match(r"WON003_tl_x(\d+)_y(\d+)\.png", name).groups())
        tile_g = gray(np.array(Image.open(p).convert("RGB")))
        patch = ortho_g[y0:y0 + TILE, x0:x0 + TILE]
        ncc = float(cv2.matchTemplate(patch, tile_g, cv2.TM_CCOEFF_NORMED)[0, 0])
        results.append(dict(file=name, kind="grid", ncc=round(ncc, 4),
                            center_x=x0 + TILE / 2, center_y=y0 + TILE / 2,
                            tile_x0=x0, tile_y0=y0))
        print(f"{name:32s} ncc={ncc:.3f} (at stated coords)", flush=True)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(dict(ortho=os.path.relpath(ORTHO, REPO), ortho_size=[W, H],
                       tile=TILE, results=results), f, indent=1)

    # -- visualization on 1/4-res copy of the ortho --------------------------
    ds = 4
    vis = Image.fromarray(ortho[::ds, ::ds].copy())
    draw = ImageDraw.Draw(vis, "RGBA")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except OSError:
        font = ImageFont.load_default()
    for r in results:
        cx, cy = r["center_x"] / ds, r["center_y"] / ds
        h = TILE / 2 / ds
        if r["kind"] == "rotated":
            # rotation was post hoc on the grid crop (black corner fill), so the
            # true footprint is the axis-aligned 500px square
            box = [cx - h, cy - h, cx + h, cy + h]
            col = (0, 255, 60) if r["ncc"] > 0.6 else (255, 40, 40)
            draw.rectangle(box, outline=col + (255,))
            draw.rectangle(box, fill=col + (40,))
            draw.text((cx - 14, cy - 8), f'{r["n"]}', fill=(255, 255, 0, 255), font=font)
        else:
            box = [cx - h, cy - h, cx + h, cy + h]
            draw.rectangle(box, outline=(40, 120, 255, 255))
            draw.rectangle(box, fill=(40, 120, 255, 50))
    vis.save(OUT_PNG)
    bad = [r for r in results if r["ncc"] <= 0.6]
    print(f"\n{len(results)} tiles placed, {len(bad)} below NCC 0.6: "
          f"{[r['file'] for r in bad]}")
    print(f"wrote {OUT_JSON}\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
