"""A5 (PLAN.md §13, pre-registered 2026-10-15): A4's GT-box masker comparison broken down by the five
pre-declared strata (crown-area terciles, biome group, GT crowns per tile, canopy contact, crown contact).
Per-crown IoUs are read, not recomputed: LACE deployed guided render
(gtbox_guided_raw/gtbox_lace_guided_r2048.jsonl) and SAM 3 zero-shot / FT per-subtile
(audit_2026-10-14/selvabox_evidence/gtbox_sam_persub_tiles/). Crown features from test_gt.json with the
tab:gtbox rasteriser (gtbox_lib / evaluate.raster at 2048). Valid crowns only. Writes tierA/a5_strata.json.

  PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.tierA.a5_strata [--gt ... --sam_dir ... --lace ... --img_dir ... --out ...]
"""
import argparse, json, os
import numpy as np
from scipy.ndimage import binary_dilation
from pycocotools import mask as mu
from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as G

HERE = os.path.dirname(os.path.abspath(__file__))
NR = os.path.dirname(HERE)
PKG = os.path.dirname(NR)
REPO = os.path.dirname(PKG)
BIOME = {**{b: "Tr" for b in (1, 2, 3, 14)}, **{b: "Te" for b in (4, 5, 6, 11)}, **{b: "Sv" for b in (7, 8, 9, 10, 12, 13)}}
DENS = [(1, 10), (11, 30), (31, 60), (61, 100), (101, 200), (201, 10 ** 9)]


def crown_features(polys, canopy):
    """-> per crown (area_px, touches_canopy, touches_crown) on the 2048 raster."""
    rles, _ = G.gt_rles_and_valid(polys, res=2048)
    can = None
    if canopy:
        can = np.zeros((2048, 2048), bool)
        for c in G.E.raster(canopy, res=2048, scale=1.0):
            can |= np.asarray(c, bool)
    lab = np.zeros((2048, 2048), np.int32)              # crown id map (last writer wins on overlaps)
    masks = [mu.decode(r).astype(bool) for r in rles]
    for i, m in enumerate(masks):
        lab[m] = i + 1
    out = []
    for i, m in enumerate(masks):
        a = int(m.sum())
        if a == 0:
            out.append((0, False, False)); continue
        ys, xs = np.where(m)
        y0, y1, x0, x1 = max(0, ys.min() - 2), min(2048, ys.max() + 3), max(0, xs.min() - 2), min(2048, xs.max() + 3)
        w = binary_dilation(m[y0:y1, x0:x1], iterations=1)
        ring = w & ~m[y0:y1, x0:x1]
        tc = bool(can is not None and (can[y0:y1, x0:x1] & w).any())
        ids = lab[y0:y1, x0:x1][w]
        tk = bool(np.any((ids > 0) & (ids != i + 1))) or bool(np.any(lab[y0:y1, x0:x1][ring] > 0))
        out.append((a, tc, tk))
    return out


def summarise(rows, arms):
    def st(x):
        x = np.asarray(x, float)
        return {"mean_iou": round(float(x.mean()), 4), "ge50": round(float((x >= .5).mean()), 4),
                "ge75": round(float((x >= .75).mean()), 4), "ge90": round(float((x >= .9).mean()), 4)}
    res = {"n": len(rows), "n_tiles": len({r["tile"] for r in rows})}
    for a in ["lace"] + arms:
        res[a] = st([r[a] for r in rows])
    rs = np.random.RandomState(0)
    tiles = sorted({r["tile"] for r in rows})
    by = {t: [r for r in rows if r["tile"] == t] for t in tiles}
    for a in arms:
        d = {t: np.array([r["lace"] - r[a] for r in by[t]]) for t in tiles}
        boots = []
        for _ in range(2000):
            pick = rs.randint(0, len(tiles), len(tiles))
            boots.append(np.concatenate([d[tiles[j]] for j in pick]).mean())
        res[f"lace_minus_{a}"] = {"mean": round(float(np.concatenate(list(d.values())).mean()), 4),
                                  "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=os.path.join(PKG, "test_gt.json"))
    ap.add_argument("--sam_dir", default=os.path.join(NR, "audit_2026-10-14", "selvabox_evidence", "gtbox_sam_persub_tiles"))
    ap.add_argument("--lace", default=os.path.join(NR, "gtbox_guided_raw", "gtbox_lace_guided_r2048.jsonl"))
    ap.add_argument("--img_dir", default=os.path.join(REPO, "data", "tcd", "test"))
    ap.add_argument("--out", default=os.path.join(HERE, "a5_strata.json"))
    ap.add_argument("--no_biome", action="store_true")
    a = ap.parse_args()
    assert not os.path.exists(a.out), a.out
    gt = json.load(open(a.gt))
    lace = {}
    for line in open(a.lace):
        r = json.loads(line); lace[r["tile"]] = r
    arms = ["base_sub", "ft_sub"]
    rows = []
    tids = [t for t in sorted(gt) if gt[t]["trees"]]
    for k, t in enumerate(tids):
        s = json.load(open(os.path.join(a.sam_dir, t + ".json")))
        L = lace[t]; assert s["valid"] == L["valid"], t
        feats = crown_features(gt[t]["trees"], gt[t].get("canopy", []))
        biome = None if a.no_biome else json.load(open(os.path.join(a.img_dir, t + "_meta.json"))).get("biome")
        n_tile = int(np.sum(L["valid"]))
        for i, v in enumerate(L["valid"]):
            if not v:
                continue
            area, tc, tk = feats[i]
            rows.append({"tile": t, "area": area, "canopy": tc, "touch": tk, "n_tile": n_tile,
                         "biome": BIOME.get(int(biome), "other") if biome is not None else None,
                         "lace": L["iou"][i], **{x: s[x][i] for x in arms}})
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(tids)} tiles", flush=True)
    out = {"n_crowns": len(rows), "all": summarise(rows, arms), "strata": {}}
    e = np.percentile([r["area"] for r in rows], [100 / 3, 200 / 3])
    out["area_tercile_edges_px"] = [round(float(x), 1) for x in e]
    groups = {
        "area": [("small", lambda r: r["area"] < e[0]), ("mid", lambda r: e[0] <= r["area"] < e[1]), ("large", lambda r: r["area"] >= e[1])],
        "crowns_per_tile": [(f"{lo}-{hi if hi < 10**9 else '+'}", (lambda lo, hi: lambda r: lo <= r["n_tile"] <= hi)(lo, hi)) for lo, hi in DENS],
        "canopy_contact": [("touches_canopy", lambda r: r["canopy"]), ("no_canopy_contact", lambda r: not r["canopy"])],
        "crown_contact": [("touching", lambda r: r["touch"]), ("isolated", lambda r: not r["touch"])],
    }
    if not a.no_biome:
        groups["biome"] = [(b, (lambda b: lambda r: r["biome"] == b)(b)) for b in ("Tr", "Te", "Sv", "other")]
    for gname, defs in groups.items():
        out["strata"][gname] = {}
        for name, f in defs:
            sel = [r for r in rows if f(r)]
            if sel:
                out["strata"][gname][name] = summarise(sel, arms)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")
    for gname, d in out["strata"].items():
        for name, s in d.items():
            print(f"{gname:16s} {name:18s} n={s['n']:6d}  LACE {s['lace']['mean_iou']:.3f}  ZS {s['base_sub']['mean_iou']:.3f}  FT {s['ft_sub']['mean_iou']:.3f}  "
                  f"L-ZS {s['lace_minus_base_sub']['mean']:+.3f} {s['lace_minus_base_sub']['ci95']}  L-FT {s['lace_minus_ft_sub']['mean']:+.3f}")


if __name__ == "__main__":
    main()
