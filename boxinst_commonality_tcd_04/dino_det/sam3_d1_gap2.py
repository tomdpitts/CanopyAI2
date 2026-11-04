"""Follow-up to sam3_d1_gap.py (same crown set / pairing). Per split:
 (a) mutually exclusive groups (first match wins): tile_edge -> touching (det box >= 10% covered by other GT trees)
     -> elongated (GT bbox aspect >= 1.5) -> rest; n, mean IoU LACE / SAM, gap, share of total gap mass.
 (b) where the pixel errors sit: FP and FN pixels binned by distance (px at 2048) from the GT crown boundary
     (<=2, 2-4, 4-8, 8-16, >16), per GT-area, LACE and SAM; all crowns and per group.
 (c) tile-edge crowns: GT pixels within 8 px of the tile border missed by each masker (per GT area).
Outputs results/sam3_d1/gap2_{split}.json.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.sam3_d1_gap2
"""
import json, os, time
import numpy as np
from pycocotools import mask as mu
from scipy.ndimage import distance_transform_edt as edt
from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
from boxinst_commonality_tcd_04.dino_det import sam3_on_d1 as S
from boxinst_commonality_tcd_04.dino_det.sam3_d1_gap import xywh

BANDS = [0, 2, 4, 8, 16, 1e9]; PAD = 24


def group(C):
    g = np.full(len(C["l_iou"]), "rest", object)
    el = C["gt_aspect"] >= 1.5; to = C["box_cov_trees"] >= 0.1; ed = C["edge"]
    g[el] = "elongated"; g[to] = "touching"; g[ed] = "tile_edge"
    return g


def main():
    for split in ("test", "sparse"):
        C = dict(np.load(os.path.join(S.OUT, f"gap_crowns_{split}.npz"), allow_pickle=True)); grp = group(C)
        L = json.load(open(os.path.join(S.RES, f"preds_dino_swinl_s0_{split}_sprime_r2048.json")))["preds"]
        M = json.load(open(os.path.join(S.OUT, f"preds_d1_sam3zs_gmean_{split}_r2048.json")))["preds"]
        g = json.load(open(S.SPLIT[split]["gt"])); gd, t2i = SL.build_gt_crowd(g, 2048)
        trees = {}
        for a in gd["annotations"]:
            if not a["iscrowd"]:
                trees.setdefault(a["image_id"], []).append(a)
        nb = len(BANDS) - 1; band = {k: np.zeros((len(grp), nb)) for k in ("l_fp", "l_fn", "s_fp", "s_fn")}
        border = np.full((len(grp), 2), np.nan); r = 0; t0 = time.time()
        for t in sorted(g):  # same iteration order and pairing as sam3_d1_gap.crowns
            G = trees.get(t2i[t], []); n = len(M[t]["masks_rle"])
            if not G or n == 0:
                continue
            B = xywh(M[t]["boxes_2048"]); bi = mu.iou(B.tolist(), [x["bbox"] for x in G], [0] * len(G))
            for j, x in enumerate(G):
                i = int(bi[:, j].argmax())
                if bi[i, j] < 0.5:
                    continue
                assert C["tile"][r] == t
                x0, y0, w, h = B[i]; gx, gy, gw, gh = x["bbox"]
                X0 = int(max(min(x0, gx) - PAD, 0)); Y0 = int(max(min(y0, gy) - PAD, 0))
                X1 = int(min(max(x0 + w, gx + gw) + PAD, 2048)); Y1 = int(min(max(y0 + h, gy + gh) + PAD, 2048))
                gm = mu.decode(x["segmentation"])[Y0:Y1, X0:X1].astype(bool); ga = max(gm.sum(), 1)
                d_out = edt(~gm); d_in = edt(gm)  # distance to GT for outside pixels / to GT exterior for inside pixels
                for p, P in (("l", L), ("s", M)):
                    m = mu.decode(dict(P[t]["masks_rle"][i], counts=P[t]["masks_rle"][i]["counts"].encode()))[Y0:Y1, X0:X1].astype(bool)
                    band[f"{p}_fp"][r] = np.histogram(d_out[m & ~gm], BANDS)[0] / ga
                    band[f"{p}_fn"][r] = np.histogram(d_in[gm & ~m], BANDS)[0] / ga
                    if C["edge"][r]:
                        yy, xx = np.mgrid[Y0:Y1, X0:X1]; near = (np.minimum(np.minimum(xx, yy), np.minimum(2047 - xx, 2047 - yy)) < 8) & gm
                        border[r, 0 if p == "l" else 1] = (near & ~m).sum() / ga
                r += 1
            print(f"  {t} {r} crowns {time.time()-t0:.0f}s", flush=True)
        assert r == len(grp), (r, len(grp))
        print(f"[{split}] {r} crowns {time.time()-t0:.0f}s", flush=True)
        gap = C["s_iou"] - C["l_iou"]; tot = gap.sum(); lab = [f"<={BANDS[1]}"] + [f"{BANDS[k]}-{BANDS[k+1]}" for k in range(1, nb - 1)] + [f">{BANDS[nb-1]}"]

        def summ(mask):
            o = dict(n=int(mask.sum()), frac_n=round(float(mask.mean()), 3), lace=round(float(C["l_iou"][mask].mean()), 3),
                     sam=round(float(C["s_iou"][mask].mean()), 3), gap=round(float(gap[mask].mean()), 3), gap_share=round(float(gap[mask].sum() / tot), 3))
            o["err_by_dist_px"] = {k: dict(zip(lab, np.round(band[k][mask].mean(0), 4).tolist())) for k in band}
            return o
        rec = {"split": split, "groups_order": "tile_edge > touching(box_cov_trees>=0.1) > elongated(aspect>=1.5) > rest",
               "all": summ(np.ones(len(grp), bool)), "groups": {k: summ(grp == k) for k in ("tile_edge", "touching", "elongated", "rest")},
               "edge_border8_missed_per_gt_area": {"lace": round(float(np.nanmean(border[:, 0])), 4), "sam": round(float(np.nanmean(border[:, 1])), 4)}}
        json.dump(rec, open(os.path.join(S.OUT, f"gap2_{split}.json"), "w"), indent=1); print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
