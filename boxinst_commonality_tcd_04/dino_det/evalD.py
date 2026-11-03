"""Pre-registration D1: LOCAL masker + scoring on the stitched DINO boxes (tcd04-tierb-vol:runs/dino_swinl_s0/preds_<split>.json).

  select: val 108, ranking {detector s, s'} by score_lite crowd mask AP50 (ties within 0.001 -> detector s); val recall by size.
  test:   ONCE at the selected ranking: LACE masker on the 439 -> frozen score_coco (2048, maxDets 600, crowd arm); both rows.
Masks persisted (RLE): val -> feat_cache/dino_swinl_s0_val_mask.json; test -> feat_cache/dino_swinl_s0_test_mask/<tid>.json.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.evalD select|test
"""
import argparse, json, multiprocessing as mp, os, subprocess, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); REPO = os.path.dirname(PKG)
FC_ = "/Users/tompitts/dphil/feat_cache"
OUT = os.path.join(HERE, "results"); TAG = "dino_swinl_s0"
VAL_GT = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "val_gt.json")


def pull(split):
    os.makedirs(os.path.join(OUT, "modal"), exist_ok=True)
    fp = os.path.join(OUT, "modal", f"{TAG}_preds_{split}.json")
    if not os.path.exists(fp):
        subprocess.run([os.path.join(REPO, ".venv", "bin", "modal"), "volume", "get", "tcd04-tierb-vol",
                        f"runs/{TAG}/preds_{split}.json", fp], check=True)
    P = json.load(open(fp)); return P["preds"], P["meta"]


def select():
    from boxinst_commonality_tcd_04.dethead.diag_val import render
    from boxinst_commonality_tcd_04.dethead.diag_box import analyse
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    out = os.path.join(OUT, f"select_val_{TAG}.json"); assert not os.path.exists(out), out
    vg = json.load(open(VAL_GT)); tids = sorted(vg); assert len(tids) == 108
    gd, t2i = SL.build_gt_crowd(vg, 2048)
    P, meta = pull("val"); assert sorted(P) == tids
    S = {t: (P[t]["boxes_2048"], P[t]["scores"]) for t in tids}
    M = render({"d": S}, {"d": [{"e": 0.0, "c": 0.33}]})["d"]["e0.00_c0.33"]
    mfp = f"{FC_}/{TAG}_val_mask.json"
    if not os.path.exists(mfp):
        json.dump({t: {"boxes_2048": S[t][0], "scores": S[t][1], "s_prime": M[t]["s_prime"], "masks_rle": M[t]["masks_rle"]} for t in tids},
                  open(mfp + ".tmp", "w")); os.replace(mfp + ".tmp", mfp)
    rows = {}
    for rank in ("detscore", "sprime"):
        preds = {t: {"boxes_2048": S[t][0], "scores": S[t][1] if rank == "detscore" else M[t]["s_prime"], "masks_rle": M[t]["masks_rle"]} for t in tids}
        r = SL.score(preds, gd, t2i, masks=True, floor=0.05 if rank == "detscore" else 0.0)
        rows[rank] = {"mask": {k: r["mask"][k] for k in ("AP50", "AP75", "AP50_95")}, "box": {k: r["box"][k] for k in ("AP50", "AP75")}}
        print(f"[val {TAG}] {rank}: {rows[rank]}", flush=True)
    sel = "detscore" if rows["detscore"]["mask"]["AP50"] >= rows["sprime"]["mask"]["AP50"] - 0.001 else "sprime"
    diag = analyse(TAG, S, gd, t2i, 0.05)
    rec = {"rule": "highest val mask AP50; ties within 0.001 -> detscore", "rows": rows, "selected": sel,
           "box_diag_detscore": diag, "modal_meta": meta, "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(rec, open(out, "w"), indent=1); print(json.dumps({"selected": sel}), flush=True)


def test():
    from boxinst_commonality_tcd_04.dethead.evalC import _mask_job
    from boxinst_commonality_tcd_04 import score_coco as SC
    sel = json.load(open(os.path.join(OUT, f"select_val_{TAG}.json")))["selected"]
    sfp = os.path.join(OUT, f"summary_{TAG}_test.json"); assert not os.path.exists(sfp), sfp
    D, meta = pull("test"); tids = sorted(D); assert len(tids) == 439
    print(f"[{TAG}] test detections {sum(len(D[t]['scores']) for t in tids)}", flush=True)
    mdir = f"{FC_}/{TAG}_test_mask"; os.makedirs(mdir, exist_ok=True)
    jobs = [(t, D[t]["boxes_2048"], D[t]["scores"], mdir) for t in tids if not os.path.exists(os.path.join(mdir, t + ".json"))]
    t0 = time.time()
    if jobs:
        with mp.get_context("spawn").Pool(12) as pool:
            for k, _ in enumerate(pool.imap_unordered(_mask_job, jobs), 1):
                if k % 40 == 0 or k == len(jobs):
                    print(f"  mask {k}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    M = {t: json.load(open(os.path.join(mdir, t + ".json"))) for t in tids}
    out = {"tag": TAG, "selected": sel, "modal_meta": meta, "rows": {}}
    os.chdir(REPO)
    for rank, floor in (("detscore", 0.05), ("sprime", 0.0)):
        fp = os.path.join(OUT, f"preds_{TAG}_test_{rank}_r2048.json")
        if not os.path.exists(fp):
            preds = {t: {"boxes_2048": D[t]["boxes_2048"], "scores": D[t]["scores"] if rank == "detscore" else M[t]["s_prime"],
                         "masks_rle": M[t]["masks_rle"]} for t in tids}
            json.dump({"meta": {"mask_res": 2048, "model": f"DINO-Swin-L (prereg D1) + LACE lace4px_simple, ranking {rank}"}, "preds": preds},
                      open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
        sp = fp.replace("preds_", "scored_")
        if not os.path.exists(sp):
            json.dump(SC.score_many([fp], os.path.join(PKG, "test_gt.json"), res=2048, max_dets=600, score_floor=floor)[0], open(sp, "w"))
        c = json.load(open(sp))["canopy_neutral_crowd"]
        out["rows"][rank] = {"mask": {k: c["mask"][k] for k in ("AP50", "AP75", "AP50_95", "n_det")},
                             "box": {k: c["box"][k] for k in ("AP50", "AP75", "AP50_95")}, "floor": floor}
    json.dump(out, open(sfp, "w"), indent=1); print(json.dumps(out, indent=1, default=str), flush=True)


def _mask_job_sparse(args):
    t, boxes, scores, d = args
    from boxinst_commonality_tcd_04.native_raster.hunt.btmask import btmask as BM
    fp = os.path.join(d, t + ".json")
    if not os.path.exists(fp):
        if "m4" not in BM._W:
            BM._init("sparse")
        r = BM.render_boxes("sparse", t, np.asarray(boxes, np.float32).reshape(-1, 4), scores, [BM.FROZEN])[BM.cfg_name(BM.FROZEN)]
        json.dump(r, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    return t


def sparse():
    """Pre-registration D1-sparse: frozen D1 + LACE masker + s' (val-selected for D1), once on the sparse-236 holdout."""
    from boxinst_commonality_tcd_04 import score_coco as SC
    sel = json.load(open(os.path.join(OUT, f"select_val_{TAG}.json")))["selected"]
    sfp = os.path.join(OUT, f"summary_{TAG}_sparse.json"); assert not os.path.exists(sfp), sfp
    gt = os.path.join(REPO, "data", "tcd_sparse", "sparse_gt.json")
    D, meta = pull("sparse"); tids = sorted(D); assert len(tids) == 236 and set(tids) == set(json.load(open(gt))), "sparse ids"
    print(f"[{TAG}] sparse detections {sum(len(D[t]['scores']) for t in tids)}", flush=True)
    mdir = f"{FC_}/{TAG}_sparse_mask"; os.makedirs(mdir, exist_ok=True)
    jobs = [(t, D[t]["boxes_2048"], D[t]["scores"], mdir) for t in tids if not os.path.exists(os.path.join(mdir, t + ".json"))]
    t0 = time.time()
    if jobs:
        with mp.get_context("spawn").Pool(12) as pool:
            for k, _ in enumerate(pool.imap_unordered(_mask_job_sparse, jobs), 1):
                if k % 40 == 0 or k == len(jobs):
                    print(f"  mask {k}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    M = {t: json.load(open(os.path.join(mdir, t + ".json"))) for t in tids}
    out = {"tag": TAG, "split": "sparse236", "selected": sel, "modal_meta": meta, "rows": {}}
    os.chdir(REPO)
    for rank, floor in (("detscore", 0.05), ("sprime", 0.0)):
        fp = os.path.join(OUT, f"preds_{TAG}_sparse_{rank}_r2048.json")
        if not os.path.exists(fp):
            preds = {t: {"boxes_2048": D[t]["boxes_2048"], "scores": D[t]["scores"] if rank == "detscore" else M[t]["s_prime"],
                         "masks_rle": M[t]["masks_rle"]} for t in tids}
            json.dump({"meta": {"mask_res": 2048, "model": f"DINO-Swin-L (prereg D1-sparse) + LACE lace4px_simple, ranking {rank}"}, "preds": preds},
                      open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
        sp = fp.replace("preds_", "scored_")
        if not os.path.exists(sp):
            json.dump(SC.score_many([fp], gt, res=2048, max_dets=600, score_floor=floor)[0], open(sp, "w"))
        c = json.load(open(sp))["canopy_neutral_crowd"]
        out["rows"][rank] = {"mask": {k: c["mask"][k] for k in ("AP50", "AP75", "AP50_95", "n_det")},
                             "box": {k: c["box"][k] for k in ("AP50", "AP75", "AP50_95")}, "floor": floor}
    json.dump(out, open(sfp, "w"), indent=1); print(json.dumps(out, indent=1, default=str), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["select", "test", "sparse"]); a = ap.parse_args()
    {"select": select, "test": test, "sparse": sparse}[a.cmd]()
