"""Where does LACE (D1) trail SAM 3 zs in mask IoU with GT? Same crown set and detection pairing as sam3_d1_iou.py (each GT
tree -> the D1 detection with highest box IoU >= 0.5; both maskers scored on that one detection). Per-crown table:
  gt geometry (area, fill = area/bbox area, aspect), det (box IoU, s, SAM score, SAM placement), context (other GT trees
  overlapping the det box, fraction of det box covered by other trees / canopy, tile edge, biome), and for each masker
  IoU, area ratio, and its pixel errors split: FP into other trees / canopy region / background, FN (missed GT).
Then strata: n, mean IoU LACE / SAM, gap, and share of the total gap mass (sum of SAM-LACE) per stratum.
Outputs: results/sam3_d1/gap_crowns_{split}.npz, gap_strata_{split}.json.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.sam3_d1_gap
"""
import json, os
import numpy as np
from pycocotools import mask as mu
from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
from boxinst_commonality_tcd_04.dino_det import sam3_on_d1 as S

FC = S.FC


def xywh(b):
    b = np.asarray(b, float).reshape(-1, 4); return np.c_[b[:, :2], b[:, 2:] - b[:, :2]]


def A(r):
    return float(mu.area(r))


def inter(a, b):
    return float(mu.area(mu.merge([a, b], intersect=True)))


def errs(m, g, others, canopy):
    """pixel FP of mask m split into other GT trees / canopy region / background, and FN. Assumes GT trees and canopy
    polygons are mutually disjoint (OAM-TCD convention), so each FP pixel lands in at most one of them."""
    mg = mu.merge([m, g], intersect=True); it = A(mg)
    fp, fn = A(m) - it, A(g) - it
    fpt = max(inter(m, others) - inter(mg, others), 0.0) if others else 0.0
    fpc = max(inter(m, canopy) - inter(mg, canopy), 0.0) if canopy else 0.0
    return fpt, fpc, max(fp - fpt - fpc, 0.0), fn


def crowns(split):
    L = json.load(open(os.path.join(S.RES, f"preds_dino_swinl_s0_{split}_sprime_r2048.json")))["preds"]
    M = json.load(open(os.path.join(S.OUT, f"preds_d1_sam3zs_gmean_{split}_r2048.json")))["preds"]
    D = S.det(split)
    tdir = os.path.join(FC, f"sam3_d1_{split}")
    g = json.load(open(S.SPLIT[split]["gt"])); gd, t2i = SL.build_gt_crowd(g, 2048)
    trees, can = {}, {}
    for a in gd["annotations"]:
        (can if a["iscrowd"] else trees).setdefault(a["image_id"], []).append(a)
    cols = {k: [] for k in ("tile", "biome", "gt_area", "gt_fill", "gt_aspect", "box_iou", "s", "sam_whole",
                            "n_nb", "box_cov_trees", "box_cov_canopy", "edge",
                            "l_iou", "s_iou", "l_ratio", "s_ratio", "l_fp_tree", "l_fp_can", "l_fp_bg", "l_fn",
                            "s_fp_tree", "s_fp_can", "s_fp_bg", "s_fn")}
    for t in sorted(g):
        G = trees.get(t2i[t], []); n = len(M[t]["masks_rle"])
        if not G or n == 0:
            continue
        meta = json.load(open(os.path.join(S.SPLIT[split]["rgb"], t + "_meta.json")))
        biome = meta.get("biome_name") or "unknown"
        sub = json.load(open(os.path.join(tdir, t + ".json")))["subtile"]
        enc = lambda r: dict(r, counts=r["counts"].encode())
        B = xywh(M[t]["boxes_2048"]); bi = mu.iou(B.tolist(), [x["bbox"] for x in G], [0] * len(G))
        gb = np.asarray([x["bbox"] for x in G], float)
        cand = lambda b, j: [k for k in np.nonzero((gb[:, 0] < b[0] + b[2]) & (gb[:, 0] + gb[:, 2] > b[0]) & (gb[:, 1] < b[1] + b[3]) & (gb[:, 1] + gb[:, 3] > b[1]))[0] if k != j]
        cmask = mu.merge([c["segmentation"] for c in can.get(t2i[t], [])]) if can.get(t2i[t]) else None
        for j, x in enumerate(G):
            i = int(bi[:, j].argmax())
            if bi[i, j] < 0.5:
                continue
            assert M[t]["boxes_2048"] == L[t]["boxes_2048"] == D[t]["boxes_2048"], t
            gm = x["segmentation"]; lm = enc(L[t]["masks_rle"][i]); sm = enc(M[t]["masks_rle"][i])
            bx = B[i]; brle = mu.frPyObjects(bx[None].astype(np.float64), 2048, 2048)[0]; barea = max(A(brle), 1.0)
            nb = [G[k]["segmentation"] for k in cand(bx, j) if inter(brle, G[k]["segmentation"]) > 0]
            others = mu.merge(nb) if nb else None
            ga = A(gm); w, h = x["bbox"][2], x["bbox"][3]
            cols["tile"].append(t); cols["biome"].append(biome); cols["gt_area"].append(ga)
            cols["gt_fill"].append(ga / max(w * h, 1.0)); cols["gt_aspect"].append(max(w, h) / max(min(w, h), 1.0))
            cols["box_iou"].append(float(bi[i, j])); cols["s"].append(float(D[t]["scores"][i]))
            cols["sam_whole"].append(sub[i] is None)
            cols["n_nb"].append(len(nb)); cols["box_cov_trees"].append(inter(brle, others) / barea if others else 0.0)
            cols["box_cov_canopy"].append(inter(brle, cmask) / barea if cmask else 0.0)
            cols["edge"].append(bool(bx[0] < 2 or bx[1] < 2 or bx[0] + bx[2] > 2046 or bx[1] + bx[3] > 2046))
            for p, m in (("l", lm), ("s", sm)):
                it = inter(m, gm); am = A(m)
                cols[f"{p}_iou"].append(it / max(am + ga - it, 1.0)); cols[f"{p}_ratio"].append(am / ga)
                fpt, fpc, fpb, fn = errs(m, gm, others, cmask)
                for k, v in (("fp_tree", fpt), ("fp_can", fpc), ("fp_bg", fpb), ("fn", fn)):
                    cols[f"{p}_{k}"].append(v / ga)
    return {k: np.asarray(v) for k, v in cols.items()}


def strata(C):
    gap = C["s_iou"] - C["l_iou"]; tot = gap.sum()
    def row(mask):
        n = int(mask.sum())
        if n == 0:
            return None
        e = {k: round(float(C[k][mask].mean()), 3) for k in ("l_fp_tree", "l_fp_can", "l_fp_bg", "l_fn", "s_fp_tree", "s_fp_can", "s_fp_bg", "s_fn", "l_ratio", "s_ratio")}
        return dict(n=n, frac_n=round(n / len(gap), 3), lace=round(float(C["l_iou"][mask].mean()), 3), sam=round(float(C["s_iou"][mask].mean()), 3),
                    gap=round(float(gap[mask].mean()), 3), gap_share=round(float(gap[mask].sum() / tot), 3),
                    lace_better=round(float((gap[mask] < 0).mean()), 3), err_per_gt_area=e)
    out = {"all": row(np.ones(len(gap), bool)), "n_crowns": len(gap), "total_gap_mass": round(float(tot), 1)}
    d = np.sqrt(C["gt_area"])  # px at 2048 (~ GSD 0.1 m)
    bins = {
        "crown_size_sqrt_px": (d, [0, 20, 30, 45, 70, 110, 1e9]),
        "box_iou": (C["box_iou"], [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]),
        "det_score": (C["s"], [0, 0.2, 0.3, 0.4, 0.5, 0.6, 1.01]),
        "gt_fill": (C["gt_fill"], [0, 0.5, 0.6, 0.7, 0.8, 1.01]),
        "gt_aspect": (C["gt_aspect"], [1, 1.25, 1.5, 2, 1e9]),
        "n_neighbours_in_box": (C["n_nb"], [0, 1, 2, 4, 7, 1e9]),
        "box_cov_other_trees": (C["box_cov_trees"], [0, 1e-9, 0.1, 0.25, 0.5, 1.01]),
        "box_cov_canopy": (C["box_cov_canopy"], [0, 1e-9, 0.1, 0.3, 0.6, 1.01]),
        "lace_area_ratio": (C["l_ratio"], [0, 0.5, 0.75, 0.9, 1.1, 1.33, 2, 1e9]),
    }
    for name, (v, e) in bins.items():
        out[name] = {f"[{e[k]:g},{e[k+1]:g})": row((v >= e[k]) & (v < e[k + 1])) for k in range(len(e) - 1)}
    out["sam_placement"] = {"subtile": row(~C["sam_whole"]), "whole_tile": row(C["sam_whole"])}
    out["tile_edge"] = {"interior": row(~C["edge"]), "edge": row(C["edge"])}
    out["biome"] = {b: row(C["biome"] == b) for b in sorted(set(C["biome"]))}
    return out


def main():
    for split in ("test", "sparse"):
        C = crowns(split)
        np.savez_compressed(os.path.join(S.OUT, f"gap_crowns_{split}.npz"), **C)
        R = strata(C); R["split"] = split
        json.dump(R, open(os.path.join(S.OUT, f"gap_strata_{split}.json"), "w"), indent=1)
        print(split, json.dumps(R["all"]), flush=True)


if __name__ == "__main__":
    main()
