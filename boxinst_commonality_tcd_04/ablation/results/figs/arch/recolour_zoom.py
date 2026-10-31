"""Re-colour the existing zoom_cells.tex from BrBG to make_arch_panels.brbg_cool() (2026-10-29).

Exact and data-free: every cell colour was written as int(255*BrBG(i)) for one of BrBG's 256 LUT
entries, so each is mapped back to its LUT index and re-emitted from the shifted LUT. Avoids
re-running the pipeline (which needs the deployed Modal features). Original kept as
zoom_cells_BrBG_2026-10-29.tex. Run from this directory.
"""
import os
import re
import shutil

import matplotlib
import numpy as np

from make_arch_panels import brbg_cool

src, bak = "zoom_cells.tex", "zoom_cells_BrBG_2026-10-29.tex"
if not os.path.exists(bak):
    shutil.copy(src, bak)
old = (matplotlib.colormaps["BrBG"](np.arange(256))[:, :3] * 255).astype(int)
new = (brbg_cool()(np.arange(256))[:, :3] * 255).astype(int)
idx = {tuple(c): i for i, c in enumerate(old)}
n = {"lut": 0, "mask": 0, "kept": 0}


def sub(m):
    c = tuple(int(v) for v in m.groups())
    if c in idx:
        n["lut"] += 1
        r, g, b = new[idx[c]]
    elif c == (1, 102, 94):          # old hard-coded mask colour (= \bfg, ~BrBG LUT 230)
        n["mask"] += 1
        r, g, b = new[230]
    else:
        n["kept"] += 1
        r, g, b = c
    return f"rgb,255:red,{r};green,{g};blue,{b}"


txt = re.sub(r"rgb,255:red,(\d+);green,(\d+);blue,(\d+)", sub, open(bak).read())
open(src, "w").write(txt)
print(n)
