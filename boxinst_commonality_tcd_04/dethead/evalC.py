"""Arm C (pre-registration 4): LOCAL masker + scoring on the boxes written by trainC_modal.predict.

  select: val 108, NMS {0.5, 0.6} x ranking {s', detector s}, score_lite crowd mask AP50 (ties within 0.001 -> NMS 0.5,
          detector s); plus val recall by size (diag_box.analyse) for the selected NMS. -> results/select_val_<tag>.json
  test:   ONCE, at the selected NMS: masker on the 439 -> frozen score_coco (2048, maxDets 600, crowd arm), both rankings
          (selected one is the headline). -> results/{preds,scored}_<tag>_nmsXX_<rank>_r2048.json, summary_<tag>_nmsXX.json
Downloads the Modal preds to results/modal/. Val masks are kept (RLE) in feat_cache/dethead_<tag>_val_nmsXX_mask.json.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.evalC select --tag fpnC_s0
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.evalC test --tag fpnC_s0
"""
import argparse, json, multiprocessing as mp, os, subprocess, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); HERE = os.path.join(PKG, "dethead"); REPO = os.path.dirname(PKG)
FC_ = "/Users/tompitts/dphil/feat_cache"
OUT = os.path.join(HERE, "results"); MOD = os.path.join(OUT, "modal")
VAL_GT = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "val_gt.json")


def fit_calib(preds, gd, t2i):
    """Per-level Platt params on these val detections (calibC's method: mask-TP labels at IoU 0.5, detector-s ranking)."""
    import boxinst_commonality_tcd_04.dethead.calibC as CB
    lab = CB.mask_labels(preds, gd, t2i)
    X, Y, Lv = [], [], []
    for t, r in preds.items():
        m = lab[t] >= 0; X.append(CB.logit(np.asarray(r["scores"])[m])); Y.append(lab[t][m]); Lv.append(CB.level(r["boxes_2048"])[m])
    X, Y, Lv = np.concatenate(X), np.concatenate(Y), np.concatenate(Lv); out = {}
    for l in range(4):
        m = Lv == l
        a, b = CB.fit(X[m], Y[m].astype(float)) if m.sum() >= 50 else (1.0, 0.0)
        out[str(l)] = {"a": round(a, 6), "b": round(b, 6), "n": int(m.sum())}
    return out


def pull(tag, split, nms):
    os.makedirs(MOD, exist_ok=True)
    name = f"{tag}_{split}_nms{nms}.json"; fp = os.path.join(MOD, name)
    if not os.path.exists(fp):
        subprocess.run([os.path.join(REPO, ".venv", "bin", "modal"), "volume", "get", "tcd04-phase4-vol",
                        f"dethead/preds/{name}", fp], check=True)
    P = json.load(open(fp)); return P["preds"], P["meta"]


def select(tag, calib=False):
    import boxinst_commonality_tcd_04.dethead.calibC as CB
    from boxinst_commonality_tcd_04.dethead.diag_val import render
    from boxinst_commonality_tcd_04.dethead.diag_box import analyse
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    out = os.path.join(OUT, f"select_val_{tag}.json"); assert not os.path.exists(out), out
    vg = json.load(open(VAL_GT)); tids = sorted(vg); assert len(tids) == 108
    gd, t2i = SL.build_gt_crowd(vg, 2048)
    sets, metas = {}, {}
    for nms in (0.5, 0.6):
        P, metas[nms] = pull(tag, "val", nms); assert sorted(P) == tids
        sets[f"nms{nms}"] = {t: (P[t]["boxes_2048"], P[t]["scores"]) for t in tids}
    R = render(sets, {k: [{"e": 0.0, "c": 0.33}] for k in sets})
    rows, cal = {}, {}
    for k, S in sets.items():
        M = R[k]["e0.00_c0.33"]
        mfp = f"{FC_}/dethead_{tag}_val_{k.replace('.', '')}_mask.json"
        if not os.path.exists(mfp):
            json.dump({t: {"boxes_2048": S[t][0], "scores": S[t][1], "s_prime": M[t]["s_prime"], "masks_rle": M[t]["masks_rle"]} for t in tids},
                      open(mfp + ".tmp", "w")); os.replace(mfp + ".tmp", mfp)
        if calib:                                           # pre-registration 6: per-level Platt (prereg 5/5a method), refit per NMS
            cal[k] = fit_calib({t: {"boxes_2048": S[t][0], "scores": S[t][1], "masks_rle": M[t]["masks_rle"]} for t in tids}, gd, t2i)
        for rank in (("sprime", "detscore", "calibrated") if calib else ("sprime", "detscore")):
            sc = {t: (M[t]["s_prime"] if rank == "sprime" else S[t][1] if rank == "detscore" else CB.apply(cal[k], S[t][0], S[t][1]).tolist()) for t in tids}
            preds = {t: {"boxes_2048": S[t][0], "scores": sc[t], "masks_rle": M[t]["masks_rle"]} for t in tids}
            r = SL.score(preds, gd, t2i, masks=True, floor=0.05 if rank == "detscore" else 0.0)
            rows[f"{k}|{rank}"] = {"mask": {kk: r["mask"][kk] for kk in ("AP50", "AP75", "AP50_95")}, "box": {kk: r["box"][kk] for kk in ("AP50", "AP75")}}
            print(f"[val {tag}] {k} {rank}: {rows[f'{k}|{rank}']}", flush=True)
    pref = [f"nms{n}|{r}" for n in (0.5, 0.6) for r in (("detscore", "sprime", "calibrated") if calib else ("detscore", "sprime"))]
    best = max(rows, key=lambda k: rows[k]["mask"]["AP50"])
    sel = next(k for k in pref if rows[best]["mask"]["AP50"] - rows[k]["mask"]["AP50"] <= 0.001)
    nms = float(sel.split("|")[0][3:])
    diag = analyse(tag, sets[sel.split("|")[0]], gd, t2i, 0.05)
    rec = {"rule": "highest val mask AP50; ties within 0.001 -> nms0.5, detscore", "rows": rows, "selected": sel, "nms": nms,
           "rank": sel.split("|")[1], "calib_params": cal, "box_diag_selected_detscore": diag, "modal_meta": {str(k): v for k, v in metas.items()},
           "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(rec, open(out, "w"), indent=1); print(json.dumps({k: rec[k] for k in ("selected", "nms", "rank")}), flush=True)


def _mask_job(args):
    t, boxes, scores, d = args
    from boxinst_commonality_tcd_04.native_raster.hunt.btmask import btmask as BM
    fp = os.path.join(d, t + ".json")
    if not os.path.exists(fp):
        if "m4" not in BM._W:
            BM._init("test")
        r = BM.render_boxes("test", t, np.asarray(boxes, np.float32).reshape(-1, 4), scores, [BM.FROZEN])[BM.cfg_name(BM.FROZEN)]
        json.dump(r, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    return t


def test(tag):
    import boxinst_commonality_tcd_04.dethead.calibC as CB
    sel = json.load(open(os.path.join(OUT, f"select_val_{tag}.json")))
    nms = sel["nms"]; ttag = f"{tag}_nms{int(round(nms * 10)):02d}"
    sfp = os.path.join(OUT, f"summary_{ttag}.json"); assert not os.path.exists(sfp), sfp
    D, meta = pull(tag, "test", nms); tids = sorted(D); assert len(tids) == 439
    print(f"[{ttag}] detections: {sum(len(D[t]['scores']) for t in tids)} on 439 tiles", flush=True)
    mdir = f"{FC_}/dethead_{ttag}_mask"; os.makedirs(mdir, exist_ok=True)
    jobs = [(t, D[t]["boxes_2048"], D[t]["scores"], mdir) for t in tids if not os.path.exists(os.path.join(mdir, t + ".json"))]
    t0 = time.time()
    if jobs:
        with mp.get_context("spawn").Pool(12) as pool:
            for k, _ in enumerate(pool.imap_unordered(_mask_job, jobs), 1):
                if k % 20 == 0 or k == len(jobs):
                    print(f"  mask {k}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    M = {t: json.load(open(os.path.join(mdir, t + ".json"))) for t in tids}
    files = {}
    cp = sel.get("calib_params", {}).get(f"nms{nms}")
    for rank, floor in (("sprime", 0.0), ("detscore", 0.05)) + ((("calibrated", 0.0),) if cp else ()):
        fp = os.path.join(OUT, f"preds_{ttag}_{rank}_r2048.json"); files[rank] = (fp, floor)
        if not os.path.exists(fp):
            sc = {t: (M[t]["s_prime"] if rank == "sprime" else D[t]["scores"] if rank == "detscore" else CB.apply(cp, D[t]["boxes_2048"], D[t]["scores"]).tolist()) for t in tids}
            preds = {t: {"boxes_2048": D[t]["boxes_2048"], "scores": sc[t], "masks_rle": M[t]["masks_rle"]} for t in tids}
            json.dump({"meta": {"mask_res": 2048, "model": f"dethead arm C {ttag} + LACE lace4px_simple, ranking {rank}"}, "preds": preds},
                      open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    from boxinst_commonality_tcd_04 import score_coco as SC
    os.chdir(REPO)
    out = {"tag": ttag, "selected": sel["selected"], "modal_meta": meta, "rows": {}}
    for rank, (fp, floor) in files.items():
        sp = fp.replace("preds_", "scored_")
        if not os.path.exists(sp):
            json.dump(SC.score_many([fp], os.path.join(PKG, "test_gt.json"), res=2048, max_dets=600, score_floor=floor)[0], open(sp, "w"))
        c = json.load(open(sp))["canopy_neutral_crowd"]
        out["rows"][rank] = {"mask": {k: c["mask"][k] for k in ("AP50", "AP75", "AP50_95", "n_det")},
                             "box": {k: c["box"][k] for k in ("AP50", "AP75", "AP50_95")}, "floor": floor}
    json.dump(out, open(sfp, "w"), indent=1); print(json.dumps(out, indent=1, default=str), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["select", "test"]); ap.add_argument("--tag", required=True)
    ap.add_argument("--calib", action="store_true", help="add the per-level calibrated ranking (pre-registration 6)")
    a = ap.parse_args()
    select(a.tag, a.calib) if a.cmd == "select" else test(a.tag)
