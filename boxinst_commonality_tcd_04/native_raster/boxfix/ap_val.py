"""Stage 1: LACE detections on the 108 VAL tiles, mask AP50 + matched-pair mask IoU for render arms.

Detector det_phase4_L24_s0 (decode score_thr 0.05, topk 600), masker em_model_4p_fix.npz (0.3/1.6), product
s * bimod * pfg_mean -- exactly guided_rgb_ext_val.py. Arms (render_lib variants, posterior shared):
  deployed     r12 / eps 1e-4 / tau 0.35 (GATE: product AP50 == guided_rgb_ext_val_r2048.json cgf_r12_e0.0001|0.35, n_det 40,058)
  boxfix_v1    FROZEN_val_2026-10-15.json
  boxfix_noSE  boxfix_v1 without the superellipse
  deployed_SE  deployed + superellipse p2.5
Scoring: score_coco frozen protocol (2048, maxDets 600, floor 0, canopy crowd), product and s rankings, halves A/B.
Matched-pair IoU: pair_iou.py (pairs fixed by box IoU >= 0.5, product order; identical across arms).
Per-tile JSON -> feat_cache/boxfix_ap_val_tiles (resume); preds -> boxfix/preds_val_s0_r2048_<arm>.json.

  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.boxfix.ap_val
"""
import json, multiprocessing as mp, os, tempfile, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); NR = os.path.dirname(HERE)
TD = "/Users/tompitts/dphil/feat_cache/boxfix_ap_val_tiles"
FROZEN = json.load(open(os.path.join(HERE, "FROZEN_val_2026-10-15.json")))["variant"]
ARMS = {"deployed": {}, "boxfix_v1": FROZEN, "boxfix_noSE": {k: v for k, v in FROZEN.items() if k != "se_p"},
        "deployed_SE": {"se_p": FROZEN["se_p"]}}
GATE_REF = os.path.join(NR, "guided_rgb_ext_val_r2048.json")
_W = {}


def _init():
    import torch
    torch.set_num_threads(1)
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.phase4_lib_tcd import Detector4Phase
    from boxinst_commonality_tcd_04.native_raster.oracle_boxes_val import DET_PATH, EM_PATH
    from boxinst_commonality_tcd_04.native_raster import guided_rgb_ext_val as GV, guided_ablate_modal as GA
    ck = torch.load(DET_PATH, map_location="cpu", weights_only=False)
    m = Detector4Phase(ck["cfg"]["in_dim"], width=ck["cfg"]["width"], tower=ck["cfg"]["tower"]).eval()
    m.load_state_dict(ck["state"])
    _W.update(model=m, masker=E.TCDMasker(EM_PATH), GV=GV, GA=GA,
              man=json.load(open(GV.MANIFEST))["feat_traintile"])


def _tile(args):
    tid, feat_dir = args
    dst = os.path.join(TD, tid + ".json")
    if os.path.exists(dst):
        return tid, 0
    import torch
    from boxinst_commonality_tcd_04.detector import STRIDE8
    from boxinst_commonality_tcd_04.native_raster.boxfix import render_lib as R
    from dapt.decode import decode
    GA, GV, masker = _W["GA"], _W["GV"], _W["masker"]
    rgbf = GV.load_rgb_verified(tid, _W["man"]).astype(np.float32) / 255.0
    feat = np.load(os.path.join(feat_dir, tid + ".npy")).astype(np.float32)
    with torch.no_grad():
        det = _W["model"](torch.from_numpy(feat)[None])
    bx, sc = decode(det, score_thr=0.05, stride=STRIDE8, topk=600)
    bx, sc = bx.numpy(), sc.numpy()
    zn, g = masker.project(feat), feat.shape[-1]; del feat
    out = {"tile": tid, "boxes_2048": bx.tolist(), "scores_s": sc.tolist(), "scores_product": [],
           "masks_rle": {a: [] for a in ARMS}}
    for b, s in zip(bx, sc):
        clip, prob, r = GA._prob(masker, zn, g, b, GA.ALPHA, GA.KAPPA)
        out["scores_product"].append(round(float(s * np.abs(2 * r - 1).mean() * r.mean()), 8))
        for a, v in ARMS.items():
            out["masks_rle"][a].append(GA._rle(R.render_full(rgbf, prob, clip, b, v)))
    json.dump(out, open(dst + ".tmp", "w")); os.replace(dst + ".tmp", dst)
    return tid, len(bx)


def main():
    from boxinst_commonality_tcd_04 import score_coco as S
    from boxinst_commonality_tcd_04.native_raster.oracle_boxes_val import FEAT_DIR, VAL_GT
    OUT = os.path.join(HERE, "ap_val_s0_r2048.json"); assert not os.path.exists(OUT), OUT
    gt = json.load(open(VAL_GT))
    tids = sorted(t for t in gt if os.path.exists(os.path.join(FEAT_DIR, t + ".npy"))); assert len(tids) == 108
    os.makedirs(TD, exist_ok=True)
    todo = [t for t in tids if not os.path.exists(os.path.join(TD, t + ".json"))]
    print(f"[ap_val] render {len(todo)}/108 tiles, arms {list(ARMS)}", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(10, initializer=_init) as pool:
        for n, (t, nb) in enumerate(pool.imap_unordered(_tile, [(t, FEAT_DIR) for t in todo]), 1):
            print(f"  {n}/{len(todo)} {t} {nb} boxes {time.time()-t0:.0f}s", flush=True)
    recs = {t: json.load(open(os.path.join(TD, t + ".json"))) for t in tids}
    n_det = sum(len(r["scores_s"]) for r in recs.values())
    paths = {}
    for a in ARMS:
        for rank in ("product", "s"):
            fp = os.path.join(HERE, f"preds_val_s0_r2048_{a}_{rank}.json")
            if not os.path.exists(fp):
                P = {t: {"boxes_2048": r["boxes_2048"], "scores": r[f"scores_{rank}"],
                         "canopy_ignore": [False] * len(r["scores_s"]), "masks_rle": r["masks_rle"][a]} for t, r in recs.items()}
                json.dump({"meta": {"mask_res": 2048, "arm": a, "variant": ARMS[a], "rank": rank, "split": "val108",
                                    "det": "det_phase4_L24_s0.pt"}, "preds": P}, open(fp, "w"))
            paths[(a, rank)] = fp
    halves = {"all": tids, "A": tids[0::2], "B": tids[1::2]}
    res = {"n_det": n_det, "arms": ARMS, "AP": {}}
    for h, ts in halves.items():
        sub = {t: gt[t] for t in ts}
        gt_fp, tid2img, tree_rles, cu = S.build_gt(sub, 2048, with_canopy=False)
        gt_cr, _, _, _ = S.build_gt(sub, 2048, with_canopy=True)
        ctx = (gt_fp, gt_cr, tid2img, tree_rles, cu, ts)
        for (a, rank), fp in paths.items():
            P = json.load(open(fp)); P["preds"] = {t: P["preds"][t] for t in ts}
            fd, tmp = tempfile.mkstemp(suffix=".json"); os.close(fd); json.dump(P, open(tmp, "w"))
            c = S._score_one(tmp, ctx, 2048, S.MAX_DETS, 0.0)["canopy_neutral_crowd"]; os.remove(tmp)
            res["AP"].setdefault(a, {}).setdefault(rank, {})[h] = {"mask": c["mask"], "box": c["box"]}
            print(f"  {h} {a:12s} {rank:8s} mask AP50 {c['mask']['AP50']:.4f} AP75 {c['mask']['AP75']:.4f} AP50:95 {c['mask']['AP50_95']:.4f}", flush=True)
    ref = json.load(open(GATE_REF))["cells"]["cgf_r12_e0.0001|0.35"]
    got = res["AP"]["deployed"]["product"]["all"]["mask"]["AP50"]; want = ref["all"]["product"]["mask"]["AP50"]
    res["gate"] = {"ref": want, "got": got, "n_det": n_det, "ref_n_det": ref["n_det"],
                   "pass": abs(got - want) <= 1e-4 and n_det == ref["n_det"]}
    print(f"GATE deployed product AP50 {got} vs {want}, dets {n_det} vs {ref['n_det']} -> {'PASS' if res['gate']['pass'] else 'FAIL'}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
