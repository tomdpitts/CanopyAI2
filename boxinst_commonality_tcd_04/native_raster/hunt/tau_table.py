"""Per-detection table on VAL: IoU vs GT at every threshold of the deployed colour-guided render, plus intrinsic
signals available at inference. Detections: every val detection whose box overlaps some GT crown box at IoU >= 0.3;
its crown = the GT crown of max box IoU. Columns: tile, det, score, short, area_box, bimod (|2r-1| mean), pfg_mean
(cell r mean), peak q (in box), mean q (in box), mean prob (in box), fill at each tau, IoU at each tau, GT area, box IoU.
-> feat_cache/hunt_tau_table_val.npz (+ column names in hunt/diag/tau_table_cols.json)

  PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.hunt.tau_table
"""
import json, multiprocessing as mp, os
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); FC = "/Users/tompitts/dphil/feat_cache"
TAUS = (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55)
_W = {}


def _init():
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.native_raster import guided_ablate_modal as GA, guided_rgb_ext_val as GV
    from boxinst_commonality_tcd_04.native_raster.hunt import fastap as F
    _W.update(m=E.TCDMasker(f"{FC}/em_model_4p_fix.npz"), GA=GA, GV=GV, gt=json.load(open(F.VAL_GT)),
              man=json.load(open(GV.MANIFEST))["feat_traintile"])


def _tile(t):
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as G
    m, GA = _W["m"], _W["GA"]
    src = json.load(open(f"{FC}/boxfix_ap_val_tiles/{t}.json"))
    bx = np.asarray(src["boxes_2048"], np.float32).reshape(-1, 4); polys = _W["gt"][t]["trees"]
    if not len(bx) or not polys:
        return []
    gb = G.gt_boxes_2048(polys).astype(np.float64); biou = GA.box_iou_mat(bx.astype(np.float64), gb)
    rgbf = _W["GV"].load_rgb_verified(t, _W["man"]).astype(np.float32) / 255.0
    feat = np.load(f"{FC}/feat_4p_val/{t}.npy").astype(np.float32); g = feat.shape[-1]; zn = m.project(feat); del feat
    rows, gcache = [], {}
    for i, b in enumerate(bx):
        j = int(biou[i].argmax())
        if biou[i, j] < 0.3:
            continue
        if j not in gcache:
            gcache[j] = np.asarray(E.raster([polys[j]], res=2048, scale=1.0)[0])
        gm = gcache[j]
        clip, prob, r = GA._prob(m, zn, g, b, GA.ALPHA, GA.KAPPA)
        by0, by1, bx0, bx1 = clip; by1, bx1 = min(by1, 2048), min(bx1, 2048)
        wy0, wy1, wx0, wx1 = max(0, by0 - 16), min(2048, by1 + 16), max(0, bx0 - 16), min(2048, bx1 + 16)
        q = GA.color_guided_filter(rgbf[wy0:wy1, wx0:wx1], prob[wy0:wy1, wx0:wx1], 12, 1e-4)
        qb = q[by0 - wy0:by1 - wy0, bx0 - wx0:bx1 - wx0]; pb = prob[by0:by1, bx0:bx1]
        gw = gm[by0:by1, bx0:bx1]; ga = gm.sum()
        ious, fills = [], []
        for tau in TAUS:
            mk = qb >= tau; inter = (mk & gw).sum(); u = mk.sum() + ga - inter
            ious.append(inter / u if u else 0.0); fills.append(mk.mean() if mk.size else 0.0)
        rows.append([float(src["scores_product"][i]), float(min(b[2] - b[0], b[3] - b[1])), float((b[2] - b[0]) * (b[3] - b[1])),
                     float(np.abs(2 * r - 1).mean()), float(r.mean()), float(qb.max()) if qb.size else 0.0,
                     float(qb.mean()) if qb.size else 0.0, float(pb.mean()) if pb.size else 0.0, float(ga), float(biou[i, j])]
                    + fills + ious)
    return rows


def main():
    out = f"{FC}/hunt_tau_table_val.npz"; assert not os.path.exists(out)
    from boxinst_commonality_tcd_04.native_raster.hunt import fastap as F
    tids = sorted(F.gt_cache("val"))
    with mp.get_context("spawn").Pool(12, initializer=_init) as pool:
        R = [r for rows in pool.map(_tile, tids) for r in rows]
    cols = (["score", "short", "area_box", "bimod", "pfg_mean", "q_peak", "q_mean", "p_mean", "gt_area", "box_iou"]
            + [f"fill_{t}" for t in TAUS] + [f"iou_{t}" for t in TAUS])
    np.savez(out, X=np.asarray(R, np.float64)); json.dump(cols, open(os.path.join(HERE, "diag", "tau_table_cols.json"), "w"))
    print(len(R), "rows ->", out)


if __name__ == "__main__":
    main()
