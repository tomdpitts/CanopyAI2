"""PhenoBench run summary -> results/summary.json + printed markdown.
Headline subset = non-partial plants (crop + weed; partial = <50 % visible, reported separately). Mean per-instance mask
IoU overall / crop / weed / per date / per size tercile (terciles of box sqrt-area over the headline subset). Fill =
predicted mask area / integer box-clip area (GT fill alongside). Paired image-bootstrap 95 % CI (2000 resamples, seed 0,
as the LIVECell probe). rho diagnostics from results/lace_tt/fit.

  cd zeroshot_phenobench/code && PB_RUN=<sample60|val> ../../../.venv/bin/python -u summarise.py
"""
import json, os

import numpy as np
from pycocotools import mask as maskUtils

from common import RES, ROOT, RUN, stems

ARMS = ["lace_tt", "lace_tt_a0", "lace_tree", "sam3", "sam3_boxclip", "exg", "ellipse", "box"]


def main():
    S = stems(); gt = {t: json.load(open(os.path.join(RES, "gt", t + ".json"))) for t in S}
    R = {a: {t: json.load(open(os.path.join(RES, a, t + ".json"))) for t in S} for a in ARMS}
    rows = []
    for t in S:
        g = gt[t]
        for a in ARMS:
            assert R[a][t]["valid"] == g["valid"] and len(R[a][t]["iou"]) == len(g["boxes"]), (a, t)
        for i, (b, c) in enumerate(zip(g["boxes"], g["cls"])):
            x0, y0, x1, y1 = b; clip = (int(np.ceil(y1)) - max(0, int(y0))) * (int(np.ceil(x1)) - max(0, int(x0)))
            r = {"img": t, "date": t[:5], "cls": c, "size": np.sqrt((x1 - x0) * (y1 - y0)), "gfill": g["gt_area"][i] / clip}
            for a in ARMS:
                m = R[a][t]["masks_rle"][i]; r[a] = R[a][t]["iou"][i]
                r["f_" + a] = maskUtils.area({"size": m["size"], "counts": m["counts"].encode()}) / clip
            rows.append(r)
    col = lambda k, rr: np.array([r[k] for r in rr])
    head = [r for r in rows if r["cls"] in ("crop", "weed")]
    edges = np.percentile(col("size", head), [100 / 3, 200 / 3]); terc = np.searchsorted(edges, col("size", head), side="right")
    subsets = {"all_incl_partial": np.ones(len(rows), bool)}
    out = {"run": RUN, "n_images": len(S), "n_images_with_plants": len({r["img"] for r in rows}), "n_head": len(head), "n_all": len(rows), "size_tercile_edges_px": [round(float(e), 1) for e in edges],
           "gt_fill_head": round(float(col("gfill", head).mean()), 4), "arms": {}}
    img, cls, date = col("img", head), col("cls", head), col("date", head)
    for a in ARMS:
        x = col(a, head)
        out["arms"][a] = {"mean_iou": round(float(x.mean()), 4), "crop": round(float(x[cls == "crop"].mean()), 4),
                          "weed": round(float(x[cls == "weed"].mean()), 4),
                          "per_date": {d: round(float(x[date == d].mean()), 4) for d in sorted(set(date))},
                          "per_tercile": {f"T{k+1}": round(float(x[terc == k].mean()), 4) for k in range(3)},
                          "all_incl_partial": round(float(col(a, rows).mean()), 4),
                          "fill": round(float(col("f_" + a, head).mean()), 4)}
    rng = np.random.default_rng(0); idx = rng.integers(0, len(S), (2000, len(S))); out["paired"] = {}
    for a, bs in (("lace_tt", ["sam3", "exg", "ellipse", "box", "lace_tt_a0", "lace_tree"]), ("lace_tree", ["sam3", "exg", "ellipse", "box"])):
        for b in bs:
            d = col(a, head) - col(b, head)
            sums = np.array([d[img == t].sum() for t in S]); cnt = np.array([(img == t).sum() for t in S]); ok = cnt > 0
            boot = sums[idx].sum(1) / np.maximum(cnt[idx].sum(1), 1)
            out["paired"][f"{a}_minus_{b}"] = {"mean": round(float(d.mean()), 4), "ci95": [round(float(q), 4) for q in np.percentile(boot, [2.5, 97.5])],
                                               "images_positive": int(sum(col(a, head)[img == t].mean() > col(b, head)[img == t].mean() for t in np.array(S)[ok])),
                                               "images_scored": int(ok.sum())}
    fits = [json.load(open(os.path.join(RES, "lace_tt", "fit", t + ".json"))) for t in S if os.path.exists(os.path.join(RES, "lace_tt", "fit", t + ".json"))]
    out["rho"] = {"fg_mass_per_band_mean_over_images": np.round(np.mean([f["rho_fg_mass"]["per_band_mean"] for f in fits], 0), 4).tolist(),
                  "map_mean_band_all": np.round(np.mean([np.mean(f["rho_fg_mass"]["maps"], 0) for f in fits], 0), 2).tolist(),
                  "k_effective": [f["k_effective"] for f in fits], "deviations": sorted({d for f in fits for d in f["deviations"]})}
    out["lace_tt_fit_failed"] = {t: R["lace_tt"][t]["fit_failed"] for t in S if R["lace_tt"][t].get("fit_failed")}   # scored as empty masks (IoU 0)
    json.dump(out, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    dates = sorted(set(date))
    print("| arm | non-partial | crop | weed | " + " | ".join(dates) + " | T1 | T2 | T3 | incl. partial | fill |")
    print("|---" * (len(dates) + 9) + "|")
    for a in ARMS:
        r = out["arms"][a]
        print(f"| {a} | **{r['mean_iou']:.3f}** | {r['crop']:.3f} | {r['weed']:.3f} | " + " | ".join(f"{r['per_date'][d]:.3f}" for d in dates) +
              " | " + " | ".join(f"{r['per_tercile'][k]:.3f}" for k in ("T1", "T2", "T3")) + f" | {r['all_incl_partial']:.3f} | {r['fill']:.2f} |")
    print(f"lace_tt fit failures (scored IoU 0): {len(out['lace_tt_fit_failed'])} images"); print(f"n non-partial {len(head)} (crop {int((cls=='crop').sum())}, weed {int((cls=='weed').sum())}); all {len(rows)}; GT fill {out['gt_fill_head']}; tercile edges {out['size_tercile_edges_px']}")
    print(json.dumps(out["paired"], indent=1)); print(json.dumps(out["rho"]["fg_mass_per_band_mean_over_images"]))
    print("rho mean map (8x8, all bands):"); [print("  " + " ".join(f"{v:.2f}" for v in row)) for row in out["rho"]["map_mean_band_all"]]


if __name__ == "__main__":
    main()
