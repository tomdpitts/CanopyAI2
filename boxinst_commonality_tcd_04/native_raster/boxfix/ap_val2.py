"""Step 2 (2026-10-16): mask arms on LACE's own VAL detections, selected by val mask AP50 (product ranking fixed).
Context: on test, SAM 3 zs masks at LACE's product ranking = 0.664 +- 0.003 vs LACE masks 0.647 (a1_same_rank.py), and
matched-pair IoU SAM - LACE = +0.030 (pair_iou_test_s*). Ranking (product from the DEPLOYED posterior) and boxes are held
fixed in every arm; only the masks change. Arm = masker knobs (alpha, kappa), prompt-box enlargement f (masker box and
clip), clip padding (px), render variant (render_lib).

Pre-declared rule: an arm qualifies if val product AP50 >= deployed + 0.003 and it does not lose on BOTH halves; one
combination round of qualifying arms; then test once (3 seeds).
Per-tile JSON -> feat_cache/boxfix_ap_val2_tiles (resume). Scores -> boxfix/ap_val2_s0_r2048.json.

  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.boxfix.ap_val2 [--arms_json F --tag T]
"""
import argparse, json, multiprocessing as mp, os, tempfile, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); NR = os.path.dirname(HERE)
ARMS = {
    "deployed": {},
    "rhalf": {"render": {"r": "half", "c_r": 0.4}},
    "a0.15": {"alpha": 0.15}, "a0.5": {"alpha": 0.5}, "a1.0": {"alpha": 1.0},
    "k1.0": {"kappa": 1.0}, "k1.3": {"kappa": 1.3},
    "enl1.05": {"enlarge": 1.05}, "enl1.10": {"enlarge": 1.10},
    "clip+4": {"clip_pad": 4},
}
_W = {}


def _init(arms):
    from boxinst_commonality_tcd_04.native_raster.boxfix import ap_val as AV
    AV._init(); _W.update(AV._W); _W["arms"] = arms


def _post_key(a):
    return (a.get("alpha", 0.30), a.get("kappa", 1.60), a.get("enlarge", 1.0))


def _tile(args):
    tid, feat_dir, td = args
    dst = os.path.join(td, tid + ".json")
    if os.path.exists(dst):
        return tid, 0
    import torch
    from boxinst_commonality_tcd_04.detector import STRIDE8
    from boxinst_commonality_tcd_04.native_raster.boxfix import render_lib as R
    from dapt.decode import decode
    GA, GV, masker, arms = _W["GA"], _W["GV"], _W["masker"], _W["arms"]
    rgbf = GV.load_rgb_verified(tid, _W["man"]).astype(np.float32) / 255.0
    feat = np.load(os.path.join(feat_dir, tid + ".npy")).astype(np.float32)
    with torch.no_grad():
        det = _W["model"](torch.from_numpy(feat)[None])
    bx, sc = decode(det, score_thr=0.05, stride=STRIDE8, topk=600)
    bx, sc = bx.numpy(), sc.numpy()
    zn, g = masker.project(feat), feat.shape[-1]; del feat
    out = {"tile": tid, "boxes_2048": bx.tolist(), "scores_product": [], "masks_rle": {a: [] for a in arms}}
    keys = sorted({_post_key(v) for v in arms.values()})
    for b, s in zip(bx, sc):
        post = {}
        for k in keys:
            al, ka, f = k
            bb = b.astype(np.float64)
            if f != 1.0:
                cx, cy, hw, hh = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2, f * (bb[2] - bb[0]) / 2, f * (bb[3] - bb[1]) / 2
                bb = np.array([max(0, cx - hw), max(0, cy - hh), min(2048, cx + hw), min(2048, cy + hh)])
            clip, prob, r = GA._prob(masker, zn, g, bb.astype(np.float32), al, ka)
            post[k] = (clip, prob, bb)
            if k == (0.30, 1.60, 1.0):
                out["scores_product"].append(round(float(s * np.abs(2 * r - 1).mean() * r.mean()), 8))
        for a, v in arms.items():
            clip, prob, bb = post[_post_key(v)]
            p = v.get("clip_pad", 0)
            if p:
                clip = (max(0, clip[0] - p), min(2048, clip[1] + p), max(0, clip[2] - p), min(2048, clip[3] + p))
            out["masks_rle"][a].append(GA._rle(R.render_full(rgbf, prob, clip, bb, v.get("render", {}))))
    json.dump(out, open(dst + ".tmp", "w")); os.replace(dst + ".tmp", dst)
    return tid, len(bx)


def main():
    from boxinst_commonality_tcd_04 import score_coco as S
    from boxinst_commonality_tcd_04.native_raster.oracle_boxes_val import FEAT_DIR, VAL_GT
    ap = argparse.ArgumentParser(); ap.add_argument("--arms_json", default=None); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    arms = json.load(open(a.arms_json)) if a.arms_json else ARMS
    assert "deployed" in arms
    tag = f"_{a.tag}" if a.tag else ""
    OUT = os.path.join(HERE, f"ap_val2{tag}_s0_r2048.json"); assert not os.path.exists(OUT), OUT
    td = f"/Users/tompitts/dphil/feat_cache/boxfix_ap_val2{tag}_tiles"; os.makedirs(td, exist_ok=True)
    gt = json.load(open(VAL_GT))
    tids = sorted(t for t in gt if os.path.exists(os.path.join(FEAT_DIR, t + ".npy"))); assert len(tids) == 108
    todo = [t for t in tids if not os.path.exists(os.path.join(td, t + ".json"))]
    print(f"[ap_val2{tag}] render {len(todo)}/108 tiles, {len(arms)} arms: {list(arms)}", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(10, initializer=_init, initargs=(arms,)) as pool:
        for n, (t, nb) in enumerate(pool.imap_unordered(_tile, [(t, FEAT_DIR, td) for t in todo]), 1):
            print(f"  {n}/{len(todo)} {t} {nb} boxes {time.time()-t0:.0f}s", flush=True)
    recs = {t: json.load(open(os.path.join(td, t + ".json"))) for t in tids}
    n_det = sum(len(r["scores_product"]) for r in recs.values())
    res = {"n_det": n_det, "arms": arms, "AP": {}}
    halves = {"all": tids, "A": tids[0::2], "B": tids[1::2]}
    for h, ts in halves.items():
        sub = {t: gt[t] for t in ts}
        gt_fp, tid2img, tree_rles, cu = S.build_gt(sub, 2048, with_canopy=False)
        gt_cr, _, _, _ = S.build_gt(sub, 2048, with_canopy=True)
        ctx = (gt_fp, gt_cr, tid2img, tree_rles, cu, ts)
        for arm in arms:
            P = {t: {"boxes_2048": recs[t]["boxes_2048"], "scores": recs[t]["scores_product"],
                     "canopy_ignore": [False] * len(recs[t]["scores_product"]), "masks_rle": recs[t]["masks_rle"][arm]} for t in ts}
            fd, tmp = tempfile.mkstemp(suffix=".json"); os.close(fd)
            json.dump({"meta": {"mask_res": 2048}, "preds": P}, open(tmp, "w"))
            c = S._score_one(tmp, ctx, 2048, S.MAX_DETS, 0.0)["canopy_neutral_crowd"]; os.remove(tmp)
            res["AP"].setdefault(arm, {})[h] = {"mask": c["mask"], "box": c["box"]}
            print(f"  {h} {arm:10s} mask AP50 {c['mask']['AP50']:.4f} AP75 {c['mask']['AP75']:.4f} AP50:95 {c['mask']['AP50_95']:.4f}", flush=True)
    d = res["AP"]["deployed"]
    res["gate_deployed_AP50"] = {"got": d["all"]["mask"]["AP50"], "ref": 0.6636, "pass": abs(d["all"]["mask"]["AP50"] - 0.6636) <= 1e-4 and n_det == 40058}
    res["qualifies"] = {arm: (v["all"]["mask"]["AP50"] >= d["all"]["mask"]["AP50"] + 0.003 and
                              not (v["A"]["mask"]["AP50"] < d["A"]["mask"]["AP50"] and v["B"]["mask"]["AP50"] < d["B"]["mask"]["AP50"]))
                        for arm, v in res["AP"].items() if arm != "deployed"}
    print("GATE", res["gate_deployed_AP50"], "\nQUALIFY", res["qualifies"], flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
