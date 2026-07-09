"""Map a cohort tile (image_path + CSV `domain`) to a site key.

Sites (12): WON003, BRU162, and 10 NEON sub-sites. The NEON code is the 4-letter
uppercase token immediately before the `_<flight>_<easting>_<northing>_image` block,
so `DCFS_2019_WOOD_3_492000_5222000_image_...` resolves to WOOD (not DCFS).
"""
import os
import re

# ...<SITE>_<flight digit>_<6-digit easting>_<7-digit northing>_image...
_NEON_RE = re.compile(r"([A-Z]{4})_\d_\d{6}_\d{7}_image")

# NEON sites we expect in the cohort; used only to sanity-check extraction.
KNOWN_NEON = {"LAJA", "LENO", "CLBJ", "WOOD", "JORN", "NOGP", "OAES", "ONAQ",
              "STER", "TOOL"}


def tile_site(image_path: str, domain: str) -> str:
    """Return the site key for a tile. Raises on an unrecognisable NEON path."""
    if domain == "WON":
        return "WON003"
    if domain == "BRU":
        return "BRU162"
    if domain == "NEON":
        name = os.path.basename(image_path)
        m = _NEON_RE.search(name)
        if not m:
            raise ValueError(f"cannot parse NEON site from {name!r}")
        return m.group(1)
    raise ValueError(f"unknown domain {domain!r} for {image_path!r}")
