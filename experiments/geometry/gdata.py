"""Data loading for the deterministic crown->shadow filter experiment.

Everything downstream (candidate generation, the filter, evaluation) reads its
records from here so the cohort definition and the pixel/azimuth conventions live
in exactly one place.

Cohort
------
The three-arm filter test needs images that have BOTH ground-truth crown boxes AND
a usable shadow direction. Shadow direction is the annotation ``(shadow_x,
shadow_y)`` (compass-style: x = East/+col, y = North/up), mapped to an image-frame
azimuth by :func:`shadow_prior.geometry.vector_to_azimuth` -- the same convention
already verified against real crowns (WON base=215 deg, BRU base=118 deg).

Two NEON rows carry placeholder azimuths (exactly 0.0 and 270.0 deg) that are the
"could-not-compute" defaults, not measured directions; they are excluded from the
azimuth cohort and reported as such.

Rotation padding
----------------
The WON/BRU tiles are synthetic rotations with constant-0 fill, leaving pure-black
triangles at the corners (WON ~10% of pixels). A luminance shadow detector would
read those as shadow, so :func:`valid_mask` returns the non-padding region and the
filter must ignore hits that land outside it.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shadow_prior.geometry import vector_to_azimuth, azimuth_to_vector

COMBINED_CSV = "data/finetune/phase22X_combined.csv"

# NEON placeholder azimuths (deg) that mean "not computed", not a real direction.
_PLACEHOLDER_AZ_DEG = {0.0, 270.0}


def base_scene(path: str) -> str:
    """Strip the synthetic-rotation suffix so all rotations of a tile share a scene."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"_rot\d+$", "", stem)


@dataclass
class ImageRecord:
    path: str
    domain: str                       # WON | BRU | NEON  (acquisition)
    scene: str                        # base tile id (rotation-invariant)
    boxes: np.ndarray                 # (N,4) xyxy ground-truth crowns
    shadow_x: Optional[float]         # East/+col component (may be None)
    shadow_y: Optional[float]         # North/up  component (may be None)
    az_valid: bool                    # usable shadow direction present

    @property
    def azimuth(self) -> Optional[float]:
        """Image-frame shadow-displacement azimuth (radians), or None.

        ``azimuth_to_vector(azimuth) = (u_row, u_col)`` is the unit displacement
        from a crown toward its cast shadow, in [row-down, col-right] pixels.
        """
        if not self.az_valid:
            return None
        return vector_to_azimuth(self.shadow_x, self.shadow_y)


def load_records(csv_path: str = COMBINED_CSV) -> List[ImageRecord]:
    from collections import OrderedDict
    by_img = OrderedDict()
    for r in csv.DictReader(open(csv_path)):
        p = r["image_path"]
        az_s = r["shadow_angle"].strip()
        d = by_img.setdefault(p, {"domain": r["domain"], "boxes": [],
                                  "az_s": az_s, "sx": r["shadow_x"].strip(),
                                  "sy": r["shadow_y"].strip()})
        if all(r[k].strip() for k in ("xmin", "ymin", "xmax", "ymax")):
            x0, y0, x1, y1 = (float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
            if x1 > x0 and y1 > y0:
                d["boxes"].append([x0, y0, x1, y1])
    recs = []
    for p, d in by_img.items():
        sx = float(d["sx"]) if d["sx"] else None
        sy = float(d["sy"]) if d["sy"] else None
        az_deg = float(d["az_s"]) if d["az_s"] else None
        az_valid = (sx is not None and sy is not None and az_deg is not None
                    and not (d["domain"] == "NEON" and az_deg in _PLACEHOLDER_AZ_DEG))
        recs.append(ImageRecord(
            path=p, domain=d["domain"], scene=base_scene(p),
            boxes=np.asarray(d["boxes"], dtype=np.float32).reshape(-1, 4),
            shadow_x=sx, shadow_y=sy, az_valid=az_valid))
    return recs


def cohort(recs: List[ImageRecord]) -> List[ImageRecord]:
    """Images usable for the three-arm test: >=1 GT box AND a valid azimuth."""
    return [r for r in recs if r.az_valid and len(r.boxes) > 0]


def load_rgb(path: str) -> np.ndarray:
    """HxWx3 float32 in [0,1]."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def luminance(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def valid_mask(rgb: np.ndarray) -> np.ndarray:
    """True where the pixel is real imagery, False on rotation-padding (pure black).

    Rotation fill is exact 0 in all channels; real shadows are essentially never
    all-zero. We erode the mask by one blob-scale so the padding *edge* (an
    artificial bright/dark step) also cannot masquerade as a shadow boundary.
    """
    from scipy.ndimage import binary_erosion
    solid = ~np.all(rgb <= (1.0 / 255.0), axis=-1)   # not pure black
    # erode a little to drop the hard padding seam
    return binary_erosion(solid, iterations=3, border_value=1)


if __name__ == "__main__":
    from collections import Counter
    recs = load_records()
    ch = cohort(recs)
    print(f"images total: {len(recs)}")
    print(f"cohort (box + valid az): {len(ch)} | by domain: "
          f"{dict(Counter(r.domain for r in ch))}")
    print(f"excluded NEON placeholders: "
          f"{sum(1 for r in recs if r.domain=='NEON' and len(r.boxes)>0 and not r.az_valid)}")
    print(f"total GT boxes in cohort: {sum(len(r.boxes) for r in ch)}")
    szs = np.concatenate([np.sqrt((r.boxes[:,2]-r.boxes[:,0])*(r.boxes[:,3]-r.boxes[:,1]))
                          for r in ch])
    print(f"crown size (sqrt area) px: median={np.median(szs):.1f} "
          f"p10={np.percentile(szs,10):.1f} p90={np.percentile(szs,90):.1f}")
