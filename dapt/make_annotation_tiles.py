"""Cut native-resolution annotation tiles from the WON003 top-left (SSL-safe) region.

Leakage geometry (verified): full ortho 7242x4996; the DAPT SSL slice is the RIGHT
60% (right60.tif = 4345 px, starting at col 2897). The left 40% [0,2897) was never
seen by SSL. A 1-tile (500px) buffer keeps annotation clear of that boundary, so tiles
are cut from cols [0, 2397] only, starting at the top-left, non-overlapping (stride =
tile). Mostly-nodata tiles (WON nodata is WHITE) are skipped.

Tiles match the existing WON test-tile spec: 500x500 px @ native 0.1 m/px. Filenames
encode the ortho pixel offset (traceable, non-overlap-checkable): WON003_tl_x{X}_y{Y}.png.

Usage:
    .venv/bin/python -m dapt.make_annotation_tiles          # writes tiles + overview
"""
import os

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
from dapt.data.cohort import REPO  # noqa: E402

ORTHO = os.path.join(REPO, "data/australia/WON003/WON003_10cm.tif")
OUT = os.path.join(REPO, "dapt/annotate/won_topleft")
OVERVIEW = os.path.join(REPO, "claude_outputs/won_topleft_annotation_grid.png")

TILE = 500                 # matches existing WON tiles (50 m @ 0.1 m/px)
STRIDE = 500               # non-overlapping (no crown in two tiles)
RIGHT60_START = 2897       # col where the SSL right-60 slice begins
BUFFER = 50                # thin margin before the SSL boundary (was a full tile)
X_MAX = RIGHT60_START - BUFFER            # last usable right edge = 2847
Y_MAX = 1500               # top strip only (above the bottom-left training tiles)
X_OFFSET = 300             # shift grid right to pull off the white top-left corner
MIN_VALID = 0.25           # admit heavily-partial tiles (nodata has no crowns)
# Optional manual exclusion of grid cells that overlap existing training tiles
# (their coords aren't recoverable in-repo; confirmed visually). List of (x0,y0).
EXCLUDE = set()
# Optional extra hand-placed tiles (x0,y0) added to the grid.
EXTRA = set()


def valid_frac(arr):
    white = np.all(arr >= 250, axis=-1)
    return 1.0 - white.mean()


def main(propose=False):
    os.makedirs(os.path.dirname(OVERVIEW), exist_ok=True)
    img = Image.open(ORTHO).convert("RGB")
    W, H = img.size
    full = np.asarray(img)

    # grid shifted right by X_OFFSET; rightmost tile capped at X_MAX (< 2897 line)
    x0s = list(range(X_OFFSET, X_MAX - TILE + 1, STRIDE))
    y0s = list(range(0, min(H, Y_MAX) - TILE + 1, STRIDE))   # top strip {0,500,1000}
    cells = [(x0, y0) for y0 in y0s for x0 in x0s] + sorted(EXTRA)
    kept, skipped, excluded = [], 0, 0
    for x0, y0 in cells:
        tile = full[y0:y0 + TILE, x0:x0 + TILE]
        vf = valid_frac(tile)
        if vf < MIN_VALID:
            skipped += 1
            continue
        if (x0, y0) in EXCLUDE:
            excluded += 1
            continue
        kept.append((x0, y0, vf))
        if not propose:
            os.makedirs(OUT, exist_ok=True)
            Image.fromarray(tile).save(
                os.path.join(OUT, f"WON003_tl_x{x0}_y{y0}.png"))

    # provenance overview: full ortho, region shading + numbered annotation tiles
    scale = 1600 / W
    ov = img.resize((int(W * scale), int(H * scale))).convert("RGB")
    d = ImageDraw.Draw(ov, "RGBA")
    Ho = ov.size[1]
    bx = RIGHT60_START * scale
    bufx = X_MAX * scale
    # right60 = SSL-trained + existing labelled WON tiles (blue wash)
    d.rectangle([bx, 0, ov.size[0], Ho], fill=(60, 90, 220, 55))
    # buffer strip (excluded, grey)
    d.rectangle([bufx, 0, bx, Ho], fill=(120, 120, 120, 70))
    # candidate annotation tiles: number + (x,y) label for exclusion reference
    for i, (x0, y0, vf) in enumerate(kept):
        col = (200, 160, 0, 255) if propose else (0, 190, 0, 255)
        d.rectangle([x0 * scale, y0 * scale, (x0 + TILE) * scale, (y0 + TILE) * scale],
                    outline=col, width=3)
        d.text((x0 * scale + 5, y0 * scale + 4),
               f"{i+1}\n{x0},{y0}", fill=(120, 80, 0, 255))
    d.line([bx, 0, bx, Ho], fill=(220, 0, 0, 255), width=3)
    kind = "PROPOSED (not yet written)" if propose else "GENERATED"
    leg = [f"WON003 test-tile split — {kind}",
           f"{'YELLOW' if propose else 'GREEN'} {len(kept)} = candidate tiles "
           "(left-40%, SSL never saw); label = #  x,y",
           f"GREY = {BUFFER}px buffer before SSL boundary",
           "BLUE = right-60% = DAPT SSL region",
           "-> tell me which #s overlap existing training tiles to EXCLUDE"]
    for j, t in enumerate(leg):
        d.text((10, 10 + 16 * j), t, fill=(0, 0, 0, 255))
    ov.save(OVERVIEW)

    verb = "candidates" if propose else "wrote"
    print(f"ortho {W}x{H}; usable cols [0,{X_MAX}] (right60 starts {RIGHT60_START}, "
          f"{BUFFER}px buffer)")
    print(f"{verb} {len(kept)} tiles ({len(x0s)} cols x up to {len(y0s)} rows); "
          f"skipped {skipped} mostly-white, {excluded} manually excluded")
    if not propose:
        print(f"tiles -> {os.path.relpath(OUT, REPO)}/")
    print(f"overview -> {os.path.relpath(OVERVIEW, REPO)}")


if __name__ == "__main__":
    import sys
    main(propose="--propose" in sys.argv)
