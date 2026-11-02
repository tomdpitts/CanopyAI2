"""VAL-only diagnostics for the FCOS head (after its single test run: the motivation is test-informed, so any fix
found here needs its own pre-registered test run). 108 val tiles, LACE masker lace4px_simple, score_lite crowd arm.

  1. box size vs GT (matched IoU >= 0.5): FCOS vs LACE CenterNet s0 vs BoxTeacher
  2. near-duplicates: detections overlapping a higher-scored one at IoU in [0.5, 0.6) (survive NMS 0.6, not 0.5)
  3. full pipeline val mask AP50 (s' and detscore) for: CenterNet s0 boxes (gate: s' = 0.6752, FROZEN_simple), FCOS NMS 0.6
     (as tested), FCOS NMS 0.5, and FCOS NMS 0.6 with the masker box inflated by e in {0.03, 0.06}.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.diag_val
"""
import json, multiprocessing as mp, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); HERE = os.path.join(PKG, "dethead")
FCC = "/Users/tompitts/dphil/feat_cache"
OUT = os.path.join(HERE, "results", "diag_val.json")
VAL_GT = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "val_gt.json")
BT_VAL = os.path.join(PKG, "box_supervised_baselines", "tierb", "results", "20261101_192510", "boxteacher", "results",
                      "preds_boxteacher_sub_val_r2048_json".replace("_json", ".json"))
CFGS = [{"e": 0.0, "c": 0.33}, {"e": 0.03, "c": 0.33}, {"e": 0.06, "c": 0.33}]


def _job(args):
    t, boxes, scores, cfgs = args
    from boxinst_commonality_tcd_04.native_raster.hunt.btmask import btmask as BM
    if "m4" not in BM._W:
        BM._init("val")
    return t, BM.render_boxes("val", t, np.asarray(boxes, np.float32).reshape(-1, 4), scores, cfgs)


def render(sets, cfgs):
    """sets: {name: {tid: (boxes, scores)}} -> {name: {cfg: {tid: rec}}}"""
    out = {}
    with mp.get_context("spawn").Pool(12) as pool:
        for name, S in sets.items():
            t0 = time.time(); res = {}
            for t, r in pool.imap_unordered(_job, [(t, b, s, cfgs[name]) for t, (b, s) in S.items()]):
                res[t] = r
            out[name] = {k: {t: res[t][k] for t in res} for k in res[next(iter(res))]}
            print(f"[render] {name} {time.time()-t0:.0f}s", flush=True)
    return out


def iou(a, B):
    x0 = np.maximum(a[0], B[:, 0]); y0 = np.maximum(a[1], B[:, 1]); x1 = np.minimum(a[2], B[:, 2]); y1 = np.minimum(a[3], B[:, 3])
    i = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return i / np.maximum((a[2] - a[0]) * (a[3] - a[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - i, 1e-9)


def size_stats(S, vg):
    rx, mg = [], []
    for t, (B, sc) in S.items():
        if not len(B):
            continue
        B = np.asarray(B, float); o = np.argsort(-np.asarray(sc)); B = B[o][:600]
        for poly in vg[t]["trees"]:
            if len(poly) < 6:
                continue
            p = np.asarray(poly, float).reshape(-1, 2); g = np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])
            io = iou(g, B); j = io.argmax()
            if io[j] < 0.5:
                continue
            b = B[j]; gw, gh = g[2] - g[0], g[3] - g[1]
            rx.append(np.sqrt((b[2] - b[0]) * (b[3] - b[1]) / max(gw * gh, 1)))
            mg.append(np.mean([(g[0] - b[0]) / gw, (b[2] - g[2]) / gw, (g[1] - b[1]) / gh, (b[3] - g[3]) / gh]))
    rx, mg = np.array(rx), np.array(mg)
    return {"n_matched": len(rx), "size_ratio_med": round(float(np.median(rx)), 3),
            "side_margin_med": round(float(np.median(mg)), 3), "side_margin_p25_p75": [round(float(np.percentile(mg, 25)), 3), round(float(np.percentile(mg, 75)), 3)]}


def dup_stats(S):
    n = nd = 0
    for t, (B, sc) in S.items():
        B = np.asarray(B, float); o = np.argsort(-np.asarray(sc)); B = B[o]
        for i in range(1, len(B)):
            n += 1; m = iou(B[i], B[:i]).max()
            nd += 0.5 <= m < 0.6
    return {"n_det": n + len(S), "frac_iou_0.5_0.6_with_higher": round(nd / max(n, 1), 4), "per_tile": round(n / len(S), 1)}


def main():
    assert not os.path.exists(OUT), OUT
    import torch
    from boxinst_commonality_tcd_04.dethead import fcos as FC
    from boxinst_commonality_tcd_04.dethead.test import load_model
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    vg = json.load(open(VAL_GT)); tids = sorted(vg); assert len(tids) == 108
    # FCOS raw outputs on val (CPU; the MPS GPU is busy training the control)
    m = load_model(os.path.join(HERE, "runs", "fcos_s0", "best.pt"), "fcos")
    fc6, fc5 = {}, {}
    t0 = time.time()
    with torch.no_grad():
        for t in tids:
            out = m(torch.from_numpy(np.load(f"{FCC}/feat_4p_val/{t}.npy")).float()[None])
            for nms_iou, D in ((0.6, fc6), (0.5, fc5)):
                FC.NMS_IOU = nms_iou
                b, s = FC.decode(out); D[t] = (b.numpy().astype(float).tolist(), s.numpy().astype(float).tolist())
    FC.NMS_IOU = 0.6
    print(f"[fcos val decode] {time.time()-t0:.0f}s", flush=True)
    cn = {}
    for t in tids:
        d = json.load(open(f"{FCC}/boxfix_ap_val_tiles/{t}.json")); cn[t] = (d["boxes_2048"], d["scores_s"])
    bt = json.load(open(BT_VAL))["preds"]; btS = {t: (bt[t]["boxes_2048"], bt[t]["scores"]) for t in tids}
    rec = {"size_vs_gt": {"centernet_s0": size_stats(cn, vg), "fcos_nms06": size_stats(fc6, vg), "boxteacher": size_stats(btS, vg)},
           "duplicates": {"centernet_s0": dup_stats(cn), "fcos_nms06": dup_stats(fc6), "fcos_nms05": dup_stats(fc5)}}
    print(json.dumps(rec, indent=1), flush=True)
    R = render({"centernet_s0": cn, "fcos_nms06": fc6, "fcos_nms05": fc5},
               {"centernet_s0": CFGS[:1], "fcos_nms06": CFGS, "fcos_nms05": CFGS[:1]})
    gd, t2i = SL.build_gt_crowd(vg, 2048)
    rows = {}
    for name, S in (("centernet_s0", cn), ("fcos_nms06", fc6), ("fcos_nms05", fc5)):
        for cfg, M in R[name].items():
            for rank in ("sprime", "detscore"):
                preds = {t: {"boxes_2048": S[t][0], "scores": M[t]["s_prime"] if rank == "sprime" else S[t][1], "masks_rle": M[t]["masks_rle"]} for t in tids}
                r = SL.score(preds, gd, t2i, masks=True, floor=0.0 if rank == "sprime" else 0.05)
                rows[f"{name}|{cfg}|{rank}"] = {"mask": {k: r["mask"][k] for k in ("AP50", "AP75", "AP50_95")}, "box": {k: r["box"][k] for k in ("AP50", "AP75")}}
                print(f"[val] {name} {cfg} {rank}: {rows[f'{name}|{cfg}|{rank}']}", flush=True)
    rec["val_pipeline"] = rows
    rec["gate_centernet_sprime_eq_0.6752"] = rows["centernet_s0|e0.00_c0.33|sprime"]["mask"]["AP50"] == 0.6752
    rec["date"] = time.strftime("%Y-%m-%d %H:%M")
    json.dump(rec, open(OUT, "w"), indent=1)
    print("GATE", rec["gate_centernet_sprime_eq_0.6752"], flush=True)


if __name__ == "__main__":
    main()
