"""Annotation cost vs mask AP on the OAM-TCD 439 holdout (native 2048), one point per Table 1 row.

x = estimated annotator-hours for every OAM-TCD label the row consumed in training OR model selection:
    n_tree * t_tree + n_canopy * T_POLY
  t_tree = T_POLY (79.2 s, polygon) if the row used crown masks, T_BOX (7.0 s, extreme clicking) if boxes.
  Canopy groups (cat 1) are always charged at T_POLY: every row consumes them as polygons (Restor as a
  class, LACE / Detectree2 as iscrowd ignore, Box2Mask blacked out on train crops, SelvaBox masked pixels).
  Per-instance times are the ones Box2Mask (TPAMI 2024, p.2) quotes: 79.2 s per COCO polygon (Lin et al.
  2014) and 7 s per box (Papadopoulos et al. 2017). The right whisker on box rows re-prices boxes at the
  conventional drawing time, 34.5 s (Su et al. 2012, as reported by Papadopoulos et al.).
  Instance counts are read from the OAM-TCD tile metadata (data/tcd/train/*_meta.json), i.e. the labels an
  annotator produced, not the post-filtering counts a loader kept.

Row label sets:
  LACE        792 train tiles as boxes + 108 val tiles as polygons (val masks select alpha, kappa, tau, guide)
  Detectree2  same 900 tiles, polygons
  Box2Mask    same 900 tiles, boxes (selected on val box AP50)
  Restor      all 4,169 OAM-TCD train tiles, polygons
  SelvaBox    folds 0-3 tiles holding >=1 crown (3,032 here vs 3,024 in their Table 19) + fold 4 (validation),
              boxes. SAM 3 and SelvaBox's non-OAM-TCD training data are not counted.
  No row's non-OAM-TCD pretraining is counted (COCO, detectree2 tropical polygons, SA-Co, DINOv3).

y = canopy-neutral mask AP50 / AP50:95 at 2048, Table 1 (paper.tex tab:oamtcd439). Sources:
  LACE  native_raster/NATIVE_RASTER.md "LACE + colour-guided render (pre-registered)", 3 seeds, sd ddof=1
  Restor rpn1000, DT2, Box2Mask  results_439/r2048/table_r2048.md (2048 columns)
  SelvaBox  NATIVE_RASTER.md SelvaBox 2048 (maxDets 600)

Outputs (next to this script): cost_pareto.pdf, cost_pareto.png (preview), cost_pareto_data.json.
Usage: .venv/bin/python boxinst_commonality_tcd_04/ablation/results/figs/make_cost_pareto.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPO = os.path.abspath(os.path.join(ROOT, ".."))
OUT = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(REPO, "data/tcd/train")
T_POLY, T_BOX, T_BOX_HI = 79.2, 7.0, 34.5

# (name, supervision class, AP50, AP50 sd, AP50:95, AP50:95 sd)
ROWS = [
    ("LACE (ours)",       "box",     0.6470, 0.0025, 0.2778, 0.0051),
    ("Restor Mask R-CNN", "mask",    0.6480, None,   0.3332, None),
    ("Detectree2",        "mask",    0.6297, None,   0.3099, None),   # seam-fixed 2026-10-14
    ("SelvaBox $\\rightarrow$ SAM 3", "boxsam",  0.5984, None,   0.2207, None),   # seam-fixed
    ("Box2Mask Swin-L",   "box",     0.5894, None,   0.2446, None),   # seam-fixed
    ("Box2Mask R-50",     "box",     0.4505, None,   0.1646, None),   # seam-fixed
]

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STYLE = {  # categorical slots 1-3 of the reference palette (all-pairs validated, light)
    "box":    dict(color="#2a78d6", marker="s", label="box labels"),
    "boxsam": dict(color="#1baf7a", marker="D", label="box labels + SAM 3"),
    "mask":   dict(color="#eb6834", marker="o", label="mask labels"),
}


def label_counts():
    tiles = {}
    for f in os.listdir(TRAIN):
        if not f.endswith("_meta.json"):
            continue
        m = json.load(open(os.path.join(TRAIN, f)))
        a = json.loads(m["coco_annotations"])
        tiles[f[:-len("_meta.json")]] = dict(
            fold=m.get("validation_fold"),
            tree=sum(x["category_id"] == 2 for x in a),
            canopy=sum(x["category_id"] == 1 for x in a))
    return tiles


def label_sets(tiles):
    gt = json.load(open(os.path.join(ROOT, "train_tiles_gt.json")))
    tr = [k for k, v in gt.items() if v["partition"] == "train"]
    va = [k for k, v in gt.items() if v["partition"] == "val"]
    sb = [k for k, v in tiles.items() if v["tree"] > 0]   # SelvaBox drops 0-annotation tiles; folds 0-4
    # each entry: list of (tile ids, tree label type)
    return {
        "LACE (ours)":       [(tr, "box"), (va, "poly")],
        "Detectree2":        [(tr + va, "poly")],
        "Box2Mask Swin-L":   [(tr + va, "box")],
        "Box2Mask R-50":     [(tr + va, "box")],
        "Restor Mask R-CNN": [(list(tiles), "poly")],
        "SelvaBox $\\rightarrow$ SAM 3": [(sb, "box")],
    }


def cost(tiles, parts, t_box):
    s, n = 0.0, dict(tiles=0, tree_box=0, tree_poly=0, canopy=0)
    for ids, kind in parts:
        nt = sum(tiles[k]["tree"] for k in ids)
        nc = sum(tiles[k]["canopy"] for k in ids)
        s += nt * (t_box if kind == "box" else T_POLY) + nc * T_POLY
        n["tiles"] += len(ids); n["tree_" + kind] += nt; n["canopy"] += nc
    return s / 3600.0, n


def frontier(pts):
    """Pareto set for (min cost, max AP); returns points sorted by cost."""
    pts = sorted(pts, key=lambda p: (p[0], -p[1]))
    out, best = [], -1
    for c, y in pts:
        if y > best:
            out.append((c, y)); best = y
    return out


def main():
    tiles = label_counts()
    sets = label_sets(tiles)
    data = []
    for name, sup, a50, a50sd, a5095, a5095sd in ROWS:
        h, n = cost(tiles, sets[name], T_BOX)
        h_hi, _ = cost(tiles, sets[name], T_BOX_HI)
        data.append(dict(name=name, supervision=sup, hours=h, hours_box34=h_hi, counts=n,
                         ap50=a50, ap50_sd=a50sd, ap5095=a5095, ap5095_sd=a5095sd))
    d = {r["name"]: r for r in data}
    lace, restor = d["LACE (ours)"], d["Restor Mask R-CNN"]
    ratio, ratio_hi = restor["hours"] / lace["hours"], restor["hours"] / lace["hours_box34"]

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 7.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
                         "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
                         "ytick.labelcolor": INK2, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), constrained_layout=True)
    # label offsets (points) per panel, tuned so nothing collides
    SB = "SelvaBox $\\rightarrow$ SAM 3"
    OFFS = {
        "ap50":   {"LACE (ours)": (6, 6), "Restor Mask R-CNN": (-4, 7), "Detectree2": (6, -6),
                   SB: (3, -8), "Box2Mask Swin-L": (3, -8), "Box2Mask R-50": (3, -8)},
        "ap5095": {"LACE (ours)": (-6, 7), "Restor Mask R-CNN": (-4, 7), "Detectree2": (5, -8),
                   SB: (3, -8), "Box2Mask Swin-L": (3, -8), "Box2Mask R-50": (3, -8)},
    }
    HA = {"LACE (ours)": {"ap50": "left", "ap5095": "right"}, "Restor Mask R-CNN": {"ap50": "right", "ap5095": "right"}}
    for ax, key, ylab, ylim in [(axes[0], "ap50", "Mask AP$_{50}$", (0.40, 0.69)),
                                (axes[1], "ap5095", "Mask AP$_{50:95}$", (0.14, 0.355))]:
        ax.set_xscale("log")
        ax.set_xlim(250, 11000)
        ax.set_ylim(*ylim)
        ax.grid(True, which="major", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=2.5, width=0.6)
        ax.set_xticks([300, 1000, 3000, 10000])
        ax.set_xticklabels(["300", "1k", "3k", "10k"])
        ax.set_xlabel("Estimated annotator-hours, OAM-TCD labels (log)", color=INK2)
        ax.set_ylabel(ylab, color=INK2)

        fr = frontier([(r["hours"], r[key]) for r in data])
        xs, ys = [], []
        for i, (c, y) in enumerate(fr):
            xs += [c]; ys += [y]
            if i + 1 < len(fr):
                xs += [fr[i + 1][0]]; ys += [y]
        xs += [ax.get_xlim()[1]]; ys += [fr[-1][1]]
        ax.plot(xs, ys, color=MUTED, lw=1.0, ls=(0, (3, 2)), zorder=1)

        for r in data:
            st = STYLE[r["supervision"]]
            x, y = r["hours"], r[key]
            if r["hours_box34"] > x * 1.001:   # box rows: whisker to the 34.5 s box price
                ax.plot([x, r["hours_box34"]], [y, y], color=st["color"], lw=0.9, alpha=0.55, zorder=2,
                        solid_capstyle="butt")
                ax.plot([r["hours_box34"]] * 2, [y - 0.003, y + 0.003], color=st["color"], lw=0.9,
                        alpha=0.55, zorder=2)
            sd = r[key + "_sd"]
            if sd:
                ax.errorbar(x, y, yerr=sd, fmt="none", ecolor=INK, elinewidth=0.8, capsize=0, zorder=4)
            is_lace = r["name"].startswith("LACE")
            ax.scatter(x, y, s=110 if is_lace else 30, marker="*" if is_lace else st["marker"],
                       facecolor=st["color"], edgecolor="white" if not is_lace else INK,
                       linewidth=0.6 if is_lace else 1.0, zorder=5)
            dx, dy = OFFS[key][r["name"]]
            ax.annotate(r["name"], (x, y), xytext=(dx, dy), textcoords="offset points",
                        ha=HA.get(r["name"], {}).get(key, "left"), va="center",
                        fontsize=7.5 if is_lace else 7, color=INK if is_lace else INK2,
                        fontweight="bold" if is_lace else "normal", zorder=6)

        if key == "ap50":   # headline bracket: LACE vs Restor at equal AP50
            yb = 0.672
            ax.annotate("", xy=(lace["hours"], yb), xytext=(restor["hours"], yb),
                        arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7, shrinkA=0, shrinkB=0))
            ax.text((lace["hours"] * restor["hours"]) ** 0.5, yb + 0.004,
                    f"{ratio:.0f}× fewer hours, same AP$_{{50}}$", ha="center", va="bottom",
                    fontsize=7.5, color=INK)

    handles = [Line2D([], [], ls="none", marker="*", ms=10, mfc=STYLE["box"]["color"], mec=INK, mew=0.6,
                      label="LACE (box labels)")]
    handles += [Line2D([], [], ls="none", marker=STYLE[k]["marker"], ms=5, mfc=STYLE[k]["color"], mec="white",
                       label=STYLE[k]["label"]) for k in ("box", "boxsam", "mask")]
    handles += [Line2D([], [], color=MUTED, lw=1.0, ls=(0, (3, 2)), label="Pareto frontier"),
                Line2D([], [], color=STYLE["box"]["color"], alpha=0.55, lw=0.9,
                       label="box priced at 34.5 s, not 7 s")]
    axes[0].legend(handles=handles, loc="lower right", fontsize=6.5, frameon=False, handlelength=1.6,
                   borderaxespad=0.2, labelcolor=INK2)

    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(os.path.join(OUT, f"cost_pareto.{ext}"), **kw)
    json.dump(dict(t_poly_s=T_POLY, t_box_s=T_BOX, t_box_hi_s=T_BOX_HI,
                   restor_over_lace=ratio, restor_over_lace_box34=ratio_hi, rows=data,
                   frontier_ap50=frontier([(r["hours"], r["ap50"]) for r in data]),
                   frontier_ap5095=frontier([(r["hours"], r["ap5095"]) for r in data])),
              open(os.path.join(OUT, "cost_pareto_data.json"), "w"), indent=1, ensure_ascii=False)
    for r in data:
        print(f"{r['name']:<20} {r['hours']:8.0f} h ({r['hours_box34']:6.0f} h @34.5s)  "
              f"AP50 {r['ap50']:.4f}  AP50:95 {r['ap5095']:.4f}  {r['counts']}")
    print(f"Restor/LACE hours: {ratio:.1f}x (box 7 s), {ratio_hi:.1f}x (box 34.5 s)")


if __name__ == "__main__":
    main()
