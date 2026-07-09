"""Plot the L2 label-efficiency curve (test mAP50 vs #labels, web vs sat) with CI
bands. Static research figure -> claude_outputs/."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dapt.data.cohort import REPO
from dapt.dataset import CohortData

# fixed-order categorical slots from the validated reference palette
COLORS = {"web": "#2a78d6", "sat": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e3"

d = json.load(open(os.path.join(REPO, "dapt/artifacts/label_efficiency.json")))
data = CohortData("web")
n_train = {f: len(data.train_subset(f, 0)) for f in d["fracs"]}

fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=140)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

for arm in ("web", "sat"):
    pts = d["curve"][arm]
    xs = [n_train[p["frac"]] for p in pts]
    ys = [p["mAP50_mean"] for p in pts]
    lo = [p["ci95"][0] for p in pts]
    hi = [p["ci95"][1] for p in pts]
    c = COLORS[arm]
    ax.fill_between(xs, lo, hi, color=c, alpha=0.13, linewidth=0)
    ax.plot(xs, ys, color=c, lw=2, marker="o", ms=8, mec="#fcfcfb", mew=1.5,
            label=f"DINOv3-{arm}", zorder=3)
    ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), textcoords="offset points",
                xytext=(8, 2), color=c, fontsize=11, fontweight="bold")

# mark the ordering flip
ax.annotate("sat > web\n(p<0.05)", (n_train[0.25], 0.05), textcoords="offset points",
            xytext=(6, 14), color=MUTED, fontsize=8.5, ha="left")
ax.annotate("web > sat\n(p<0.05)", (n_train[1.0], 0.40), textcoords="offset points",
            xytext=(-4, -30), color=MUTED, fontsize=8.5, ha="right")

ax.set_xscale("log")
ax.set_xticks(list(n_train.values()))
ax.set_xticklabels([f"{n}\n({int(f*100)}%)" for f, n in n_train.items()])
ax.minorticks_off()
ax.set_xlabel("labelled training tiles", color=MUTED, fontsize=10)
ax.set_ylabel("test mAP@50", color=MUTED, fontsize=10)
ax.set_ylim(0, 0.6)
ax.set_title("Label efficiency (L2 MLP probe, frozen features, 5 seeds)",
             color=INK, fontsize=12, fontweight="bold", pad=12)
ax.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED)
ax.legend(frameon=False, loc="upper left", fontsize=10)
fig.text(0.5, -0.02, "shaded = 95% tile-bootstrap CI · steeply concave-up, "
         "unsaturated at 100% · backbone ordering flips with label budget",
         ha="center", color=MUTED, fontsize=8)

out = os.path.join(REPO, "claude_outputs", "dapt_label_efficiency.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight", facecolor="#fcfcfb")
print(f"wrote {out}  n_train={n_train}")
