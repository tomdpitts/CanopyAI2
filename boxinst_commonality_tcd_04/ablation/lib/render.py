"""Figure composition for the qualitative panel.

Builds a vector (PDF/SVG) figure from crops of the panels already rendered by
phase4_research/viz_mask_compare.py and viz_beta_disagree.py. Those PNGs are the DEPLOYED
system's real outputs, so no re-inference is needed and no feature extraction is involved.

WHAT IS NOT DONE HERE, and why: a soft EM POSTERIOR heatmap would need the masker re-run
over DINOv3 features. The saved predictions carry only RLE masks already thresholded at
mask_thr=0.25, so the soft posterior is not recoverable from disk; and local DINOv3
features do not match the Modal ones that produced these results (transformers 5.12.1 vs
4.57.1, agreeing only at cos 0.86), so a locally recomputed posterior would not depict the
deployed model. The heatmap is therefore deferred with the other Modal items rather than
approximated.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def load_crop(path, box=None, max_px=1400):
    """Load a panel, optionally crop (l, t, r, b) in source px, and downscale for the PDF."""
    im = Image.open(path).convert("RGB")
    if box:
        im = im.crop(box)
    if max(im.size) > max_px:
        s = max_px / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    return np.asarray(im)


def grid(panels, out, ncol=2, title=None, figw=7.2, dpi=300):
    """panels: list of (image_array, caption). Writes `out` (.pdf) and a matching .svg."""
    n = len(panels)
    nrow = int(np.ceil(n / ncol))
    ar = np.mean([p[0].shape[0] / p[0].shape[1] for p in panels])
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(figw, figw / ncol * ar * nrow + 0.95 * nrow + 0.45))
    axes = np.atleast_1d(axes).ravel()
    for ax, (img, cap) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(cap, fontsize=5.6, loc="left", pad=3, linespacing=1.35)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4); sp.set_color("#888")
    for ax in axes[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=7.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975 if title else 1))
    fig.subplots_adjust(wspace=0.06)
    fig.savefig(out, dpi=dpi)
    fig.savefig(os.path.splitext(out)[0] + ".svg", dpi=dpi)
    plt.close(fig)
    return out
