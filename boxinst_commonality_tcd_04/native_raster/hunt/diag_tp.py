"""Lead-finding on VAL: where do SAM masks score a TP (mask IoU >= 0.5, greedy in LACE product order, legacy canopy
rule) that LACE masks do not, and vice versa? Same boxes, same scores; only masks differ.
Per detection: tp/gt for each mask set, LACE & SAM IoU vs the crown SAM (or LACE) matched, LACE precision/recall vs
that crown, emptiness, box IoU with that crown, box short side, score rank.
-> hunt/diag_tp_val.json (summary) + feat_cache/hunt_diag_tp_val.pkl (per-detection table)."""
import json, os, pickle
import numpy as np
from pycocotools import mask as mu
from boxinst_commonality_tcd_04 import evaluate as E
from boxinst_commonality_tcd_04.native_raster.hunt import fastap as F
from boxinst_commonality_tcd_04.native_raster.guided_ablate_modal import box_iou_mat
from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as G

HERE = os.path.dirname(os.path.abspath(__file__)); FC = F.FC
OUT = os.path.join(HERE, "diag_tp_val.json"); assert not os.path.exists(OUT)


def main():
    gtc = F.gt_cache("val"); gt = json.load(open(F.VAL_GT)); tids = sorted(gtc)
    rows = []; SL, SS, SCS = [], [], []
    for t in tids:
        L = json.load(open(f"{FC}/boxfix_ap_val_tiles/{t}.json")); Sm = json.load(open(f"{FC}/hunt_sam_val_dets/{t}.json"))
        assert L["boxes_2048"] == Sm["boxes_2048"]
        sc = np.asarray(L["scores_product"], np.float64); bx = np.asarray(L["boxes_2048"]).reshape(-1, 4)
        lr, sr = F.enc(L["masks_rle"]["deployed"]), F.enc(Sm["masks_rle"])
        iL, gL_ign = F.tile_stats(lr, gtc[t]); iS, gS_ign = F.tile_stats(sr, gtc[t])
        SL.append((iL, gL_ign)); SS.append((iS, gS_ign)); SCS.append(sc)
        if not len(sc):
            continue
        oL, _, tpL, keepL, gidL = E.match_tile(iL, sc, gL_ign, 0.5)
        oS, _, tpS, keepS, gidS = E.match_tile(iS, sc, gS_ign, 0.5)
        tpl = np.zeros(len(sc), bool); tps = np.zeros(len(sc), bool); gl = np.full(len(sc), -1); gs = np.full(len(sc), -1)
        tpl[oL] = tpL; gl[oL] = gidL; tps[oS] = tpS; gs[oS] = gidS
        polys = gt[t]["trees"]; gb = G.gt_boxes_2048(polys).astype(np.float64) if polys else np.zeros((0, 4))
        biou = box_iou_mat(bx, gb) if len(gb) else np.zeros((len(bx), 0))
        from boxinst_commonality_tcd_04 import score_coco as SCC; la = SCC._areas(lr) if lr else []; sa = SCC._areas(sr) if sr else []
        ga = SCC._areas(gtc[t]["trees"]) if gtc[t]["trees"] else []
        for i in range(len(sc)):
            if not (tpl[i] or tps[i]):
                continue
            j = gs[i] if tps[i] else gl[i]
            inter = iL[i, j] * (la[i] + ga[j]) / (1 + iL[i, j]) if la[i] else 0.0
            rows.append(dict(tile=t, det=i, score=float(sc[i]), tpL=bool(tpl[i]), tpS=bool(tps[i]), crown=int(j),
                             same_crown=bool(gl[i] == gs[i]), iouL=float(iL[i, j]), iouS=float(iS[i, j]),
                             recL=float(inter / ga[j]), precL=float(inter / la[i]) if la[i] else 0.0, emptyL=bool(la[i] == 0),
                             areaL=int(la[i]), areaS=int(sa[i]), areaG=int(ga[j]), box_iou=float(biou[i, j]),
                             short=float(min(bx[i, 2] - bx[i, 0], bx[i, 3] - bx[i, 1]))))
    pickle.dump(rows, open(f"{FC}/hunt_diag_tp_val.pkl", "wb"))
    n_gt = {t: len(gtc[t]["trees"]) for t in tids}
    summ = {"AP50_legacy": {"lace": F.ap_halves(tids, SL, SCS, n_gt), "sam": F.ap_halves(tids, SS, SCS, n_gt)},
            "AP75_legacy": {"lace": F.ap_halves(tids, SL, SCS, n_gt, 0.75), "sam": F.ap_halves(tids, SS, SCS, n_gt, 0.75)}}
    R = rows
    onlyS = [r for r in R if r["tpS"] and not r["tpL"]]; onlyL = [r for r in R if r["tpL"] and not r["tpS"]]
    both = [r for r in R if r["tpL"] and r["tpS"]]
    summ["counts"] = {"both": len(both), "sam_only": len(onlyS), "lace_only": len(onlyL)}
    E_ = (622.0, 2456.3)
    def strat(rs):
        out = {}
        for st, lo, hi in (("small", 0, E_[0]), ("mid", E_[0], E_[1]), ("large", E_[1], 1e12)):
            x = [r for r in rs if lo <= r["areaG"] < hi]
            if not x:
                out[st] = {"n": 0}; continue
            f = lambda k: round(float(np.mean([r[k] for r in x])), 3)
            out[st] = {"n": len(x), "iouL": f("iouL"), "iouS": f("iouS"), "recL": f("recL"), "precL": f("precL"),
                       "emptyL": f("emptyL"), "box_iou": f("box_iou"), "frac_box_iou<0.5": round(float(np.mean([r["box_iou"] < .5 for r in x])), 3),
                       "areaL/areaG": round(float(np.median([r["areaL"] / max(r["areaG"], 1) for r in x])), 3),
                       "areaS/areaG": round(float(np.median([r["areaS"] / max(r["areaG"], 1) for r in x])), 3),
                       "median_score": round(float(np.median([r["score"] for r in x])), 4)}
        return out
    summ["sam_only_by_size"] = strat(onlyS); summ["lace_only_by_size"] = strat(onlyL); summ["both_by_size"] = strat(both)
    # failure modes of LACE on sam-only TPs
    fm = {"empty": 0, "under (recL<0.6, precL>=0.8)": 0, "over (precL<0.6, recL>=0.8)": 0, "both low": 0, "near miss 0.4-0.5": 0, "other": 0}
    for r in onlyS:
        if r["emptyL"]: fm["empty"] += 1
        elif r["recL"] < .6 and r["precL"] >= .8: fm["under (recL<0.6, precL>=0.8)"] += 1
        elif r["precL"] < .6 and r["recL"] >= .8: fm["over (precL<0.6, recL>=0.8)"] += 1
        elif r["recL"] < .6 and r["precL"] < .6: fm["both low"] += 1
        elif r["iouL"] >= .4: fm["near miss 0.4-0.5"] += 1
        else: fm["other"] += 1
    summ["lace_failure_on_sam_only"] = fm
    json.dump(summ, open(OUT, "w"), indent=1); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
