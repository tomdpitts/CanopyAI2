"""Local OAM-TCD tile index: biome + crown/canopy stats + scene clusters + seen-900 flag.

Everything the sparse-slice work needs about the 4608 local tiles, derived ONLY from
data/tcd/{train,test}/*_meta.json (biome, biome_name, country, bounds, crs, image_id,
coco_annotations) and the frozen seen-900 cohort. No HF pull, no GPU, no network.

`scene_id` is a 300 m single-linkage spatial cluster over tile-bound centroids (EPSG:3395
metres). Tiles are 205 m across, so one cluster ~= one OpenAerialMap ortho. It exists to
MEASURE how much of a candidate slice sits in a scene the detector already trained on; the
slice itself excludes by tile id only (user decision), so `scene_clean` is a report axis,
not a filter.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.modal_sparse_tcd_multiseed.tile_index
"""
import json
import os

import numpy as np
import pycocotools.mask as M

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                       # boxinst_commonality_tcd_04
REPO = os.path.dirname(PKG)
TCD = os.path.join(REPO, "data/tcd")
PHASE4_MANIFEST = os.path.join(PKG, "modal_tcd_multiseed/phase4/manifest.json")
TRAIN_GT = os.path.join(PKG, "train_tiles_gt.json")
CACHE = os.path.join(HERE, "tile_index.json")

SCENE_LINK_M = 300.0

# WWF/Olson terrestrial biomes, the taxonomy behind restor/tcd's `biome` field ([-1,14];
# -1 = no biome polygon matched). Same table as data/tcd/experimental_ignore/sparse.
BIOME = {
    -1: "UNMATCHED",
    1: "Trop/Subtrop Moist Broadleaf F.",
    2: "Trop/Subtrop Dry Broadleaf F.",
    3: "Trop/Subtrop Conifer F.",
    4: "Temperate Broadleaf & Mixed F.",
    5: "Temperate Conifer F.",
    6: "Boreal Forest / Taiga",
    7: "Trop/Subtrop Grassland, Savanna & Shrubland",
    8: "Temperate Grassland, Savanna & Shrubland",
    9: "Flooded Grassland & Savanna",
    10: "Montane Grassland & Shrubland",
    11: "Tundra",
    12: "Mediterranean Forest, Woodland & Scrub",
    13: "Desert & Xeric Shrubland",
    14: "Mangrove",
}
# Provisional open-canopy group -- a GEOGRAPHIC label, not a canopy-density measurement.
# Confirm/adjust from the contact sheet before building the slice.
SPARSE_DEFAULT = (7, 8, 9, 10, 12, 13)

# Local dir <-> HF restor/tcd split. The `val` in tcd_val_tile_* is a legacy filename;
# restor/tcd's held-out split is called `test`. (Same constant as phase4_modal.HF_SPLIT.)
DIRS = {"train": os.path.join(TCD, "train"), "test": os.path.join(TCD, "test")}


def seen_tiles():
    """The frozen 900 the detector was fitted on, cross-checked across both records."""
    man = json.load(open(PHASE4_MANIFEST))["feat_traintile"]
    gt = json.load(open(TRAIN_GT))
    assert set(man) == set(gt), (
        f"seen-900 disagree: manifest {len(man)} vs train_tiles_gt {len(gt)}, "
        f"symdiff {len(set(man) ^ set(gt))}")
    return {t: man[t]["image_id"] for t in man}


def seg_to_rle(seg, H, W):
    """COCO segmentation -> a single merged RLE, or None. Handles BOTH encodings.

    OAM-TCD mixes polygon and RLE segmentations (647 ITC + 3375 canopy anns are RLE,
    across 2142 of the 4608 tiles). Critically, **RLE annotations carry `area` == 0** in
    this dataset, so the COCO `area` field CANNOT be summed to get coverage -- it silently
    drops exactly the large canopy regions, which are the ones stored as RLE. We decode
    instead. Same failure mode prepare_test.seg_rings was written to fix for the GT.
    """
    if isinstance(seg, dict):
        rle = M.frPyObjects(seg, H, W) if isinstance(seg.get("counts"), list) else seg
        return M.merge(rle) if isinstance(rle, list) else rle
    if not seg:
        return None
    polys = [np.asarray(p, np.float64).ravel().tolist() for p in seg if len(p) >= 6]
    return M.merge(M.frPyObjects(polys, H, W)) if polys else None


def coverage(anns, H, W):
    """(canopy_frac, crown_frac) as true UNION areas -- overlap-safe and RLE-safe."""
    out = []
    for cat in (1, 2):
        rles = [r for r in (seg_to_rle(a["segmentation"], H, W)
                            for a in anns if a["category_id"] == cat) if r is not None]
        out.append(float(M.area(M.merge(rles))) / (H * W) if rles else 0.0)
    return out


def _meta_paths(split):
    d = DIRS[split]
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith("_meta.json"))


def _scene_ids(recs):
    """300 m single-linkage clusters over tile centroids -> {tid: scene_id}."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    c = np.array([[(r["bounds"][0] + r["bounds"][2]) / 2,
                   (r["bounds"][1] + r["bounds"][3]) / 2] for r in recs])
    pr = cKDTree(c).query_pairs(SCENE_LINK_M, output_type="ndarray")
    n = len(recs)
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return {r["tid"]: int(l) for r, l in zip(recs, lab)}


def build(verbose=True):
    seen = seen_tiles()
    recs = []
    for split in ("train", "test"):
        paths = _meta_paths(split)
        for i, p in enumerate(paths):
            if verbose and (i + 1) % 500 == 0:
                print(f"  {split} {i + 1}/{len(paths)}", flush=True)
            m = json.load(open(p))
            tid = os.path.basename(p).replace("_meta.json", "")
            anns = json.loads(m["coco_annotations"])
            # crowns = individual tree crowns (cat 2); canopy (cat 1) is the ignore region.
            # Both fractions are LABELLED coverage -- an annotation statistic, not ground
            # truth cover: a tile can read as sparse simply because its closed part was
            # left unlabelled.
            n_crowns = sum(a["category_id"] == 2 for a in anns)
            can, crown = coverage(anns, m["height"], m["width"])
            recs.append({
                "tid": tid, "split": split, "image_id": int(m["image_id"]),
                "biome": int(m.get("biome", -1)), "biome_name": m.get("biome_name") or "",
                "country": m.get("country"), "bounds": m["bounds"], "crs": m.get("crs"),
                "width": m["width"], "height": m["height"],
                "validation_fold": m.get("validation_fold"),
                "n_crowns": n_crowns,
                "canopy_frac": round(can, 5), "crown_frac": round(crown, 5),
                "seen": tid in seen,
            })
    assert len({r["image_id"] for r in recs}) == len(recs), "image_id not unique"
    assert sum(r["seen"] for r in recs) == len(seen), \
        f"only {sum(r['seen'] for r in recs)}/{len(seen)} seen tiles found on disk"

    scene = _scene_ids(recs)
    seen_scenes = {scene[r["tid"]] for r in recs if r["seen"]}
    for r in recs:
        r["scene_id"] = scene[r["tid"]]
        r["scene_clean"] = scene[r["tid"]] not in seen_scenes

    out = {"meta": {"n": len(recs), "n_seen": len(seen), "scene_link_m": SCENE_LINK_M,
                    "n_scenes": len(set(scene.values())),
                    "n_seen_scenes": len(seen_scenes), "biome_table": BIOME},
           "tiles": recs}
    json.dump(out, open(CACHE, "w"))
    if verbose:
        print(f"[tile_index] {len(recs)} tiles, {len(seen)} seen, "
              f"{out['meta']['n_scenes']} scenes ({len(seen_scenes)} touched by the 900) "
              f"-> {os.path.relpath(CACHE, REPO)}")
    return out


def load():
    return json.load(open(CACHE)) if os.path.exists(CACHE) else build(verbose=False)


def candidates(idx=None, biomes=SPARSE_DEFAULT, include_official_test=True):
    """Unseen tiles in `biomes`. source = train_pool | official_test."""
    idx = idx or load()
    out = []
    for r in idx["tiles"]:
        if r["seen"] or r["biome"] not in biomes:
            continue
        if r["split"] == "test" and not include_official_test:
            continue
        r = dict(r)
        r["source"] = "official_test" if r["split"] == "test" else "train_pool"
        out.append(r)
    return sorted(out, key=lambda r: r["tid"])


def report(idx=None, biomes=SPARSE_DEFAULT):
    """The inventory table: every biome, gross + scene-clean, grouped sparse/not."""
    idx = idx or load()
    T = idx["tiles"]
    codes = sorted({r["biome"] for r in T})
    lines = []
    hdr = (f"{'code':>4} {'biome':<44}{'unseen':>7}{'clean':>7}{'in439':>7}"
           f"{'crowns/tile':>13}{'crown%':>8}{'canopy%':>9}{'seen':>6}")
    for grp, title in ((True, "SPARSE / open-canopy (proposed)"),
                       (False, "CLOSED-CANOPY / other")):
        lines += ["", f"=== {title} ===", hdr]
        rows = [b for b in codes if (b in biomes) == grp]
        rows.sort(key=lambda b: -sum(1 for r in T
                                     if r["biome"] == b and not r["seen"]
                                     and r["split"] == "train"))
        tot = []
        for b in rows:
            un = [r for r in T if r["biome"] == b and not r["seen"] and r["split"] == "train"]
            cl = [r for r in un if r["scene_clean"]]
            te = [r for r in T if r["biome"] == b and r["split"] == "test"]
            sn = [r for r in T if r["biome"] == b and r["seen"]]
            tot += un
            name = BIOME.get(b, f"(code {b} — not a WWF biome)")
            cpt = np.mean([r["n_crowns"] for r in un]) if un else 0.0
            cf = np.mean([r["canopy_frac"] for r in un]) * 100 if un else 0.0
            kf = np.mean([r["crown_frac"] for r in un]) * 100 if un else 0.0
            lines.append(f"{b:>4} {name:<44}{len(un):>7}{len(cl):>7}{len(te):>7}"
                         f"{cpt:>13.1f}{kf:>7.1f}%{cf:>8.1f}%{len(sn):>6}")
        te = [r for r in T if (r["biome"] in biomes) == grp and r["split"] == "test"]
        lines.append(f"{'':>4} {'TOTAL':<44}{len(tot):>7}"
                     f"{sum(r['scene_clean'] for r in tot):>7}{len(te):>7}"
                     f"{np.mean([r['n_crowns'] for r in tot]):>13.1f}"
                     f"{np.mean([r['crown_frac'] for r in tot]) * 100:>7.1f}%"
                     f"{np.mean([r['canopy_frac'] for r in tot]) * 100:>8.1f}%")
    return "\n".join(lines)


if __name__ == "__main__":
    idx = build()
    print(report(idx))
