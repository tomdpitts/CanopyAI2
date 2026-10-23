"""Fast AP for iteration on the 108 val tiles: the LEGACY canopy arm of score_coco (evaluate.match_tile /
_greedy_ap, unmatched predictions >50% inside unlabelled canopy dropped) on RLEs at 2048. It tracks the COCO
canopy-crowd headline closely (test LACE s0: 0.6473 legacy vs 0.6472 crowd). Screening only: every reported
number is re-scored with score_coco's frozen protocol.
"""
import json, os, pickle
import numpy as np
from pycocotools import mask as mu
from boxinst_commonality_tcd_04 import evaluate as E, score_coco as S

FC = "/Users/tompitts/dphil/feat_cache"
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VAL_GT = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "val_gt.json")
TEST_GT = os.path.join(PKG, "test_gt.json")


def gt_cache(split):
    fp = f"{FC}/hunt_gt_{split}.pkl"
    if os.path.exists(fp):
        return pickle.load(open(fp, "rb"))
    gt = json.load(open(VAL_GT if split == "val" else TEST_GT))
    if split == "val":
        feat = f"{FC}/feat_4p_val"; gt = {t: v for t, v in gt.items() if os.path.exists(f"{feat}/{t}.npy")}
        assert len(gt) == 108
    out = {}
    for t in sorted(gt):
        tr = S._polys_to_rle(gt[t]["trees"], 2048)
        out[t] = {"trees": tr, "canopy": S._union(S._polys_to_rle(gt[t]["canopy"], 2048), 2048)}
    pickle.dump(out, open(fp + ".tmp", "wb")); os.replace(fp + ".tmp", fp)
    return out


def enc(rles):
    return [{"size": r["size"], "counts": r["counts"].encode() if isinstance(r["counts"], str) else r["counts"]} for r in rles]


def tile_stats(rles, gtt):
    """-> (iou (n_det, n_gt), ign (n_det,))"""
    rl = enc(rles); g = gtt["trees"]
    iou = (np.asarray(mu.iou(rl, g, [0] * len(g)), np.float64).reshape(len(rl), len(g)) if (g and rl)
           else np.zeros((len(rl), len(g))))
    ign = S._frac_inside(rl, gtt["canopy"]) > 0.5 if rl else np.zeros(0, bool)
    return iou, ign


def ap(stats, scores, n_gt, thr=0.5):
    """stats: list of (iou, ign) per tile, scores: list of arrays (same order)."""
    return E._greedy_ap([s[0] for s in stats], [np.asarray(x, np.float64) for x in scores], [s[1] for s in stats], n_gt, thr)


def ap_halves(tids, stats, scores, n_gt_by_tile, thr=0.5):
    out = {}
    for h, sel in (("all", tids), ("A", tids[0::2]), ("B", tids[1::2])):
        ix = [tids.index(t) for t in sel]
        out[h] = round(ap([stats[i] for i in ix], [scores[i] for i in ix], sum(n_gt_by_tile[t] for t in sel), thr), 4)
    return out
