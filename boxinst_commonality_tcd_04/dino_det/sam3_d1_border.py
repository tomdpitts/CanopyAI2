"""Tile-border diagnostic (follow-up to sam3_d1_gap2.py). For every GT tree pixel within 16 px of the tile border, on
crowns in the gap set whose pairing is as sam3_d1_gap.py: miss rate (GT pixel not in mask) of LACE (D1) and SAM 3, by
border side (top/left/bottom/right) and distance band. Tests two code-level hypotheses:
  H1 top/left: guided_ablate_modal._prob shifts the 4 px posterior by delta=round(origin - s/2)=6 px, zero-filling the
     first 6 rows/cols -> prob 0 there -> LACE should miss ~all GT pixels at distance 0-5 on top/left only.
  H2 bottom/right: phase4_features_tcd._shift zero-fills a dy/dx (<=12 px) black strip -> outermost cells see black ->
     raised miss on bottom/right over ~0-12 px.
-> results/sam3_d1/border_{split}.json
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.sam3_d1_border
"""
import json, os
import numpy as np
from pycocotools import mask as mu
from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
from boxinst_commonality_tcd_04.dino_det import sam3_on_d1 as S
from boxinst_commonality_tcd_04.dino_det.sam3_d1_gap import xywh

BANDS = [0, 2, 4, 6, 8, 12, 16]; SIDES = ("top", "left", "bottom", "right")


def main():
    for split in ("test", "sparse"):
        L = json.load(open(os.path.join(S.RES, f"preds_dino_swinl_s0_{split}_sprime_r2048.json")))["preds"]
        M = json.load(open(os.path.join(S.OUT, f"preds_d1_sam3zs_gmean_{split}_r2048.json")))["preds"]
        g = json.load(open(S.SPLIT[split]["gt"])); gd, t2i = SL.build_gt_crowd(g, 2048)
        trees = {}
        for a in gd["annotations"]:
            if not a["iscrowd"]:
                trees.setdefault(a["image_id"], []).append(a)
        nb = len(BANDS) - 1; cnt = {sd: np.zeros((3, nb)) for sd in SIDES}  # rows: n GT px, LACE missed, SAM missed
        strip = {"top": lambda a: a[:16, :], "left": lambda a: a[:, :16].T,              # -> (16, 2048), row = distance
                 "bottom": lambda a: a[::-1][:16, :], "right": lambda a: a[:, ::-1][:, :16].T}
        for nt, t in enumerate(sorted(g)):
            if nt % 50 == 0:
                print(f"  {split} tile {nt}/{len(g)}", flush=True)
            G = trees.get(t2i[t], []); n = len(M[t]["masks_rle"])
            if not G or n == 0:
                continue
            B = xywh(M[t]["boxes_2048"]); bi = mu.iou(B.tolist(), [x["bbox"] for x in G], [0] * len(G))
            for j, x in enumerate(G):
                i = int(bi[:, j].argmax()); gx, gy, gw, gh = x["bbox"]
                if bi[i, j] < 0.5 or not (gx < 16 or gy < 16 or gx + gw > 2032 or gy + gh > 2032):
                    continue
                gm = mu.decode(x["segmentation"]).astype(bool)
                lm = mu.decode(dict(L[t]["masks_rle"][i], counts=L[t]["masks_rle"][i]["counts"].encode())).astype(bool)
                sm = mu.decode(dict(M[t]["masks_rle"][i], counts=M[t]["masks_rle"][i]["counts"].encode())).astype(bool)
                for sd in SIDES:
                    gs, ls, ss = strip[sd](gm), strip[sd](lm), strip[sd](sm)
                    for k in range(nb):
                        z = gs[BANDS[k]:BANDS[k + 1]]
                        if z.any():
                            cnt[sd][0, k] += z.sum(); cnt[sd][1, k] += (z & ~ls[BANDS[k]:BANDS[k + 1]]).sum(); cnt[sd][2, k] += (z & ~ss[BANDS[k]:BANDS[k + 1]]).sum()
        lab = [f"{BANDS[k]}-{BANDS[k+1]}" for k in range(nb)]
        rec = {"split": split, "bands_px": lab, "sides": {sd: {"gt_px": cnt[sd][0].astype(int).tolist(),
               "lace_miss": np.round(cnt[sd][1] / np.maximum(cnt[sd][0], 1), 3).tolist(),
               "sam_miss": np.round(cnt[sd][2] / np.maximum(cnt[sd][0], 1), 3).tolist()} for sd in SIDES}}
        json.dump(rec, open(os.path.join(S.OUT, f"border_{split}.json"), "w"), indent=1)
        print(split, lab, flush=True)
        for sd in SIDES:
            print(f"  {sd:6} LACE {rec['sides'][sd]['lace_miss']}  SAM {rec['sides'][sd]['sam_miss']}  n {rec['sides'][sd]['gt_px']}", flush=True)


if __name__ == "__main__":
    main()
