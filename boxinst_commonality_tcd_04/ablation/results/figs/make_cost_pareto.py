"""Annotation cost vs mask AP on the OAM-TCD 439 holdout (native 2048), one point per Table 1 row.

2026-11-24 re-run: rows are paper.tex tab:oamtcd439 as it stands (LACE = D1 detector + edge fix + learned confidence,
BoxTeacher added, SelvaBox -> SAM 3 dropped with its Table 1 row); AP values are read from results_439/TABLE1_CURRENT.json,
never typed in. Previous version (CenterNet-era LACE 0.647, 3 seeds) is in git history.

x = estimated annotator-hours for every OAM-TCD label the row consumed in training OR model selection:
    n_tree * t_tree + n_canopy * T_POLY
  t_tree = T_POLY (54 s, polygon) if the row used crown masks, T_BOX (7 s, bounding box) if boxes.
  Canopy groups (cat 1) are always charged at T_POLY: every row consumes them as polygons (Restor as a
  class, LACE / Detectree2 as iscrowd ignore, Box2Mask blacked out on train crops, SelvaBox masked pixels).
  Per-instance times (2026-11-26) are Jain & Grauman (ICCV 2013, Sec. 5 user study): "7 sec for bounding box,
  20 sec for sloppy contour, 54 sec for tight polygon", median per-image times from 101 MTurk users, 5 users per
  image per mode, same 420 images for every mode -- the only like-for-like box vs polygon timing found
  (PDF ~/dphil/refs/lace_bib_pdfs/jain2013.pdf). Replaces Box2Mask's 79.2 s (COCO polygon, draw only) / 7 s
  (Papadopoulos 2017 extreme clicking) pairing and the 34.5 s whisker (draw + verify), which mixed protocols.
  Instance counts are read from the OAM-TCD tile metadata (data/tcd/train/*_meta.json), i.e. the labels an
  annotator produced, not the post-filtering counts a loader kept.

Row label sets:
  LACE        792 train tiles as boxes + 108 val tiles as polygons (val masks select alpha, kappa, tau, guide, and
              select the learned confidence by val mask AP50; its training target is box-match, box IoU >= 0.5,
              dino_det/rerank3.py label "box50")
  Detectree2  same 900 tiles, polygons
  Box2Mask    same 900 tiles, boxes (selected on val box AP50)
  BoxTeacher  same 900 tiles, boxes (selected on val box AP50, box_supervised_baselines/tierb/TIERB.md)
  Restor      all 4,169 OAM-TCD train tiles, polygons
  No row's non-OAM-TCD pretraining is counted (COCO, detectree2 tropical polygons, DINOv3).

y = canopy-neutral mask AP50 / AP50:95 at 2048, Table 1 (paper.tex tab:oamtcd439), read from
  results_439/TABLE1_CURRENT.json (table1_439, full precision). All rows are single runs, so no seed band is drawn.

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
T_POLY, T_BOX = 54.0, 7.0

# (figure name, supervision class, method key in TABLE1_CURRENT.json table1_439) -- the six rows of paper.tex tab:oamtcd439
T1 = os.path.join(ROOT, "results_439", "TABLE1_CURRENT.json")
ROW_KEYS = [
    ("LACE (ours)",       "box",  "LACE (D1 variant) + edge fix + R3"),
    ("Restor Mask R-CNN", "mask", "Restor Mask R-CNN"),
    ("Detectree2",        "mask", "Detectree2"),
    ("BoxTeacher",        "box",  "BoxTeacher Swin-B"),
    ("Box2Mask Swin-L",   "box",  "Box2Mask Swin-L"),
    ("Box2Mask R-50",     "box",  "Box2Mask R-50"),
]


def table1_rows():
    t = {r["method"]: r for r in json.load(open(T1))["table1_439"]}
    rows = []
    for name, sup, key in ROW_KEYS:
        r = t[key]
        assert r["n_seeds"] == 1, (key, "seed band expected: draw it")
        rows.append((name, sup, r["AP50"], None, r["AP50_95"], None))
    return rows

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STYLE = {  # categorical slots 1-3 of the reference palette (all-pairs validated, light)
    "box":    dict(color="#2a78d6", marker="s", label="box labels"),
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
    # each entry: list of (tile ids, tree label type)
    return {
        "LACE (ours)":       [(tr, "box"), (va, "poly")],
        "Detectree2":        [(tr + va, "poly")],
        "Box2Mask Swin-L":   [(tr + va, "box")],
        "Box2Mask R-50":     [(tr + va, "box")],
        "BoxTeacher":        [(tr + va, "box")],
        "Restor Mask R-CNN": [(list(tiles), "poly")],
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
    for name, sup, a50, a50sd, a5095, a5095sd in table1_rows():
        h, n = cost(tiles, sets[name], T_BOX)
        data.append(dict(name=name, supervision=sup, hours=h, counts=n,
                         ap50=a50, ap50_sd=a50sd, ap5095=a5095, ap5095_sd=a5095sd))
    d = {r["name"]: r for r in data}
    lace, restor = d["LACE (ours)"], d["Restor Mask R-CNN"]
    ratio = restor["hours"] / lace["hours"]

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 7.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
                         "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
                         "ytick.labelcolor": INK2, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), constrained_layout=True)
    # label offsets (points) per panel, tuned so nothing collides
    OFFS = {
        "ap50":   {"LACE (ours)": (8, -9), "Restor Mask R-CNN": (-4, 8), "Detectree2": (6, -6),
                   "BoxTeacher": (3, -8), "Box2Mask Swin-L": (3, -8), "Box2Mask R-50": (3, -8)},
        "ap5095": {"LACE (ours)": (8, -9), "Restor Mask R-CNN": (-4, -8), "Detectree2": (5, -8),
                   "BoxTeacher": (3, -8), "Box2Mask Swin-L": (3, -8), "Box2Mask R-50": (3, -8)},
    }
    HA = {"Restor Mask R-CNN": {"ap50": "right", "ap5095": "right"}}
    for ax, key, ylab, ylim in [(axes[0], "ap50", "Mask AP$_{50}$", (0.40, 0.765)),
                                (axes[1], "ap5095", "Mask AP$_{50:95}$", (0.14, 0.365))]:
        ax.set_xscale("log")
        ax.set_xlim(200, 7000)
        ax.set_ylim(*ylim)
        ax.grid(True, which="major", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=2.5, width=0.6)
        ax.set_xticks([300, 1000, 3000])
        ax.set_xticklabels(["300", "1k", "3k"])
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

        if key == "ap50":   # headline bracket: LACE vs Restor (LACE is ahead on both axes since the D1 detector)
            yb = 0.738
            ax.annotate("", xy=(lace["hours"], yb), xytext=(restor["hours"], yb),
                        arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7, shrinkA=0, shrinkB=0))
            ax.text((lace["hours"] * restor["hours"]) ** 0.5, yb + 0.004,
                    f"{ratio:.0f}× fewer hours, {lace['ap50'] - restor['ap50']:+.3f} AP$_{{50}}$", ha="center", va="bottom",
                    fontsize=7.5, color=INK)

    handles = [Line2D([], [], ls="none", marker="*", ms=10, mfc=STYLE["box"]["color"], mec=INK, mew=0.6,
                      label="LACE (box labels)")]
    handles += [Line2D([], [], ls="none", marker=STYLE[k]["marker"], ms=5, mfc=STYLE[k]["color"], mec="white",
                       label=STYLE[k]["label"]) for k in ("box", "mask")]
    handles += [Line2D([], [], color=MUTED, lw=1.0, ls=(0, (3, 2)), label="Pareto frontier")]
    axes[0].legend(handles=handles, loc="lower right", fontsize=6.5, frameon=False, handlelength=1.6,
                   borderaxespad=0.2, labelcolor=INK2)

    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(os.path.join(OUT, f"cost_pareto.{ext}"), **kw)
    json.dump(dict(t_poly_s=T_POLY, t_box_s=T_BOX, time_source="Jain & Grauman ICCV 2013 user study",
                   restor_over_lace=ratio, rows=data,
                   frontier_ap50=frontier([(r["hours"], r["ap50"]) for r in data]),
                   frontier_ap5095=frontier([(r["hours"], r["ap5095"]) for r in data])),
              open(os.path.join(OUT, "cost_pareto_data.json"), "w"), indent=1, ensure_ascii=False)
    for r in data:
        print(f"{r['name']:<20} {r['hours']:8.0f} h  "
              f"AP50 {r['ap50']:.4f}  AP50:95 {r['ap5095']:.4f}  {r['counts']}")
    print(f"Restor/LACE hours: {ratio:.1f}x (box 7 s, polygon 54 s)")


if __name__ == "__main__":
    main()
