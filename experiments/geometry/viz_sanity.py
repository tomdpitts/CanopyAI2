"""Visual proof that the annotated shadow direction points at the actual shadows.

For a handful of tiles per acquisition, draw each GT crown box and an arrow from
its centre along the image-frame shadow-displacement azimuth. If the convention is
right, arrows land on the dark cast shadows. This validates the geometry for THIS
experiment rather than trusting the prior verification blindly.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.geometry.gdata import load_records, cohort, load_rgb, valid_mask
from shadow_prior.geometry import azimuth_to_vector

OUT = "claude_outputs/geometry"
os.makedirs(OUT, exist_ok=True)


def main():
    recs = cohort(load_records())
    by_dom = {}
    for r in recs:
        by_dom.setdefault(r.domain, []).append(r)
    picks = []
    for dom in ("WON", "BRU", "NEON"):
        picks += by_dom[dom][:3]

    n = len(picks)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    L = 60.0  # arrow length px
    for ax, r in zip(axes, picks):
        rgb = load_rgb(r.path)
        ax.imshow(rgb)
        u_row, u_col = azimuth_to_vector(r.azimuth)
        for (x0, y0, x1, y1) in r.boxes:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor="lime", linewidth=1.0))
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            ax.arrow(cx, cy, L * u_col, L * u_row, color="red", width=1.2,
                     head_width=6, length_includes_head=True, alpha=0.9)
        vm = valid_mask(rgb)
        pad = (~vm).mean()
        ax.set_title(f"{r.domain} {r.scene}\naz={np.degrees(r.azimuth):.0f}deg pad={pad:.0%}",
                     fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Red arrow = annotated shadow-displacement direction (should point at cast shadow)",
                 fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "sanity_shadow_arrows.png")
    fig.savefig(p, dpi=90, bbox_inches="tight")
    print("wrote", p)


if __name__ == "__main__":
    main()
