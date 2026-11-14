"""3 examples (per date, the image with the median non-partial plant count) -> figs/examples_<RUN>.png. Columns: GT | LACE (alpha 0.3) | LACE tree-fitted | SAM 3 | ExG | ellipse. Outlines of non-partial plants only; panel
title = mean IoU over the image's non-partial plants.

  cd zeroshot_phenobench/code && ../../../.venv/bin/python -u viz.py
"""
import json, os

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as maskUtils

from common import RES, ROOT, RUN, stems, load_rgbf, gt_image

ARMS = [("lace_tt", "LACE (test-time fit)", "#e8590c"), ("lace_tree", "LACE (tree-fitted, as-is)", "#c2255c"), ("sam3", "SAM 3", "#1c7ed6"), ("exg", "ExG + Otsu", "#f59f00"),
        ("ellipse", "Ellipse", "#7048e8")]


def main():
    S = stems(); gt = {t: json.load(open(os.path.join(RES, "gt", t + ".json"))) for t in S}
    head = {t: [i for i, c in enumerate(gt[t]["cls"]) if c in ("crop", "weed")] for t in S}
    picks = []
    for d in sorted({t[:5] for t in S}):
        ts = sorted(t for t in S if t.startswith(d)); n = np.array([len(head[t]) for t in ts])
        picks.append(ts[int(np.argsort(n, kind="stable")[len(n) // 2])])
    R = {a: {t: json.load(open(os.path.join(RES, a, t + ".json"))) for t in picks} for a, _, _ in ARMS}
    fig, ax = plt.subplots(3, 6, figsize=(24, 12.6))
    for r, t in enumerate(picks):
        rgb = load_rgbf(t); boxes, gt_rles, _, cls, _ = gt_image(t); ins = head[t]
        cols = [("GT", "#37b24d", [maskUtils.decode(gt_rles[i]) for i in ins], None)]
        for a, lab, c in ARMS:
            ms = [maskUtils.decode({"size": R[a][t]["masks_rle"][i]["size"], "counts": R[a][t]["masks_rle"][i]["counts"].encode()}) for i in ins]
            cols.append((lab, c, ms, np.mean([R[a][t]["iou"][i] for i in ins])))
        for c, (lab, colr, ms, iou) in enumerate(cols):
            A = ax[r, c]; A.imshow(rgb)
            for i, m in zip(ins, ms):
                if m.any():
                    A.contour(m.astype(float), levels=[0.5], colors=colr, linewidths=0.9)
                b = boxes[i]; A.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="yellow", lw=0.4))
            A.set_xlim(-0.5, 1023.5); A.set_ylim(1023.5, -0.5); A.set_xticks([]); A.set_yticks([])
            A.set_title(lab + ("" if iou is None else f"  IoU {iou:.3f}"), fontsize=11)
            if c == 0:
                A.set_ylabel(f"{t}\n({len(ins)} non-partial plants)", fontsize=9)
        print(t, len(ins), flush=True)
    fig.suptitle(f"PhenoBench val, zero-shot on identical GT boxes (yellow). {RUN} ({len(S)} images).", fontsize=12)
    fig.tight_layout(); fp = os.path.join(ROOT, "figs", f"examples_{RUN}.png"); fig.savefig(fp, dpi=100); print(fp)


if __name__ == "__main__":
    main()
