"""Pre-registration LACE-0 (README.md, 2026-11-26): LACE with no crown-mask label setting any tuned quantity.

Masker = deployed em_model_4p_fix.npz (beta 0.5); inference fixed a priori: alpha 1, kappa scale 1 (kappa 10), tau 0.5
absolute, no guided filter (r 0, bilinear posterior), clipped to box, EDGE_FILL on. s-hat: R3 machinery and box50 target,
selected by val BOX AP50 (OOF) by backward elimination from all 24 R1 features; fallback s-hat := s. ONE look on test 439
and sparse 236 with the frozen score_coco. Everything resumable: per-tile masks, feature npz, ledger.

  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.lace0 all
"""
import argparse, json, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

from boxinst_commonality_tcd_04.dino_det import abl_lib as L
from boxinst_commonality_tcd_04.dino_det import rerank as R
from boxinst_commonality_tcd_04.dino_det import rerank3 as R3

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); REPO = os.path.dirname(PKG)
FC = L.FC; OUT = os.path.join(HERE, "results", "lace0"); LEDGER = os.path.join(OUT, "ledger.jsonl"); NPROC = 14
SPEC = {"kind": "post4", "alpha": 1.0, "kappa": 1.0, "r": 0, "tau": 0.5}
DEP_SPEC = {"kind": "post4", "c_peak": L.C_PEAK, "alpha": 0.3, "kappa": 1.6}       # gate G1 only
ALL = list(R.NAMES)                                                                  # 24 R1 candidates
MODELS = list(R3.MODELS); DEF = {k: R3.DEF[k] for k in ("epochs", "dropout", "wd")}
RUNS, FRESH, TOL = R3.RUNS, R3.FRESH, 0.001
GT = {"val": R.VAL_GT, "test": R.TEST_GT, "sparse": os.path.join(REPO, "data", "tcd_sparse", "sparse_gt.json")}
DEPLOYED_PREDS = {s: os.path.join(HERE, "results", f"preds_dino_swinl_s0_{s}_R3_ef_r2048.json") for s in ("test", "sparse")}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ render
def render(split, t, boxes, scores, spec):
    from boxinst_commonality_tcd_04.native_raster.hunt.btmask import btmask as BM
    GA, RR = L._mods(); m = L.masker(L.DEPLOYED, True)
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    if not len(boxes):
        return [], []
    ctx = {"rgbf": BM._rgb(split, t), "GA": GA, "m4": m, "g4": 512}
    ctx["zn4"] = np.load(f"{BM.SPLIT[split]['z4']}/{t}.npy").astype(np.float32).reshape(-1, 128)
    M, SP = [], []
    for i, b in enumerate(boxes):
        post = GA._prob(m, ctx["zn4"], 512, b, spec["alpha"], spec["kappa"])
        det = {"i": i, "box": b, "cache": {("prob4", spec["alpha"], spec["kappa"]): post}}
        r = post[2]
        SP.append(round(float(scores[i] * np.abs(2 * r - 1).mean() * r.mean()), 8))
        M.append(GA._rle(RR.post4(ctx, det, spec)))
    return M, SP


def _rjob(a):
    split, t, boxes, scores, spec = a
    M, SP = render(split, t, boxes, scores, spec)
    return t, {"masks_rle": M, "s_prime": SP}


def dets(split):
    if split == "val":
        return json.load(open(f"{FC}/dino_swinl_s0_val_mask.json"))      # boxes_2048 + detector s (masker_scalars_d1 convention)
    return json.load(open(os.path.join(HERE, "results", f"preds_dino_swinl_s0_{split}_detscore_r2048.json")))["preds"]


def render_all(split, pool):
    D = dets(split); tids = sorted(json.load(open(GT[split]))); assert sorted(D) == tids, split
    d = f"{FC}/lace0_{split}"; os.makedirs(d, exist_ok=True); todo = [t for t in tids if not os.path.exists(f"{d}/{t}.json")]
    log(f"[render] {split}: {len(todo)}/{len(tids)} to do"); t0 = time.time()
    for n, (t, r) in enumerate(pool.imap_unordered(_rjob, [(split, t, D[t]["boxes_2048"], D[t]["scores"], SPEC) for t in todo]), 1):
        json.dump(r, open(f"{d}/{t}.json.tmp", "w")); os.replace(f"{d}/{t}.json.tmp", f"{d}/{t}.json")
        if n % 20 == 0 or n == len(todo):
            log(f"  render {split} {n}/{len(todo)} {time.time()-t0:.0f}s")
    F = {t: json.load(open(f"{d}/{t}.json")) for t in tids}
    return {t: {"boxes_2048": D[t]["boxes_2048"], "scores": D[t]["scores"], "s_prime": F[t]["s_prime"], "masks_rle": F[t]["masks_rle"]}
            for t in tids}


# ------------------------------------------------------------------ features (R1 extractor, own cache)
def features(split, P, pool):
    fp = f"{FC}/lace0_feats_{split}.npz"
    if os.path.exists(fp):
        Z = np.load(fp); return list(Z["tids"]), Z["tile"], Z["X"], (Z["y"] if "y" in Z else None)
    gt = json.load(open(GT[split])) if split == "val" else None; tids = sorted(P); out = {}; t0 = time.time()
    for k, (t, X, y) in enumerate(pool.imap_unordered(R._feats, [(split, t, P[t], gt[t] if gt else None) for t in tids]), 1):
        out[t] = (X, y)
        if k % 40 == 0 or k == len(tids):
            log(f"  feats {split} {k}/{len(tids)} {time.time()-t0:.0f}s")
    tile = np.concatenate([np.full(len(out[t][0]), i) for i, t in enumerate(tids)]); X = np.concatenate([out[t][0] for t in tids])
    d = {"tids": np.asarray(tids), "tile": tile, "X": X}
    if gt:
        d["y"] = np.concatenate([out[t][1] for t in tids])
    np.savez_compressed(fp + ".tmp.npz", **d); os.replace(fp + ".tmp.npz", fp)
    return tids, tile, X, d.get("y")


# ------------------------------------------------------------------ box fasteval (bbox COCOeval, crowd arm, maxDets 600)
def _btile(a):
    from pycocotools import mask as M
    boxes, gts, crowds = a; n = len(boxes)
    if n == 0:
        return [], np.zeros(0), len(gts)
    d = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
    iou = np.asarray(M.iou(d, gts, [0] * len(gts))).reshape(n, len(gts)) if gts else np.zeros((n, 0))
    ioa = np.asarray(M.iou(d, crowds, [1] * len(crowds))).reshape(n, len(crowds)).max(1) if crowds else np.zeros(n)
    return [(np.flatnonzero(r >= 0.5), r[r >= 0.5]) for r in iou], np.asarray(ioa, np.float64), len(gts)


def build_box(P, gt, pool):
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    tids = sorted(gt); assert sorted(P) == tids
    gd, t2i = SL.build_gt_crowd(gt, 2048); by = {i: ([], []) for i in t2i.values()}
    for a in gd["annotations"]:
        by[a["image_id"]][a["iscrowd"]].append(a["bbox"])
    tiles = pool.map(_btile, [(np.asarray(P[t]["boxes_2048"], np.float64).reshape(-1, 4).tolist(), by[t2i[t]][0], by[t2i[t]][1]) for t in tids])
    return {"tids": tids, "tiles": tiles}


# ------------------------------------------------------------------ selection worker
_W = {}


def _init(npz, ebox):
    import pickle
    Z = np.load(npz); E = pickle.load(open(ebox, "rb"))
    _W.update(X=Z["X"], y=Z["y"], tile=Z["tile"], F=R3.fold_sets(len(Z["tids"])), E=E)
    assert list(E["tids"]) == list(Z["tids"])


def _job(a):
    from boxinst_commonality_tcd_04.dino_det import fasteval as FE
    cfg, fname, seed = a; W = _W
    v, _ = R3.oof_fits(cfg, seed, fname, W["X"], W["y"], None, W["tile"], W["F"])
    return float(FE.ap(W["E"], v, thrs=[0.5])[0])


def cfg(model="mlp2x64", cols=tuple(ALL)):
    return {"model": model, **DEF, "label": "box50", "cols": tuple(cols)}


def key(c, runs):
    return json.dumps({**c, "cols": list(c["cols"]), "runs": [list(r) for r in runs]}, sort_keys=True)


def ledger():
    Lg = {}
    if os.path.exists(LEDGER):
        for ln in open(LEDGER):
            r = json.loads(ln); Lg[r["key"]] = r
    return Lg


def evaluate(pool, cfgs, base, runs=RUNS, tag=""):
    Lg = ledger(); todo = [c for c in cfgs if key(c, runs) not in Lg]
    if todo:
        t0 = time.time(); res = pool.map(_job, [(c, f, s) for c in todo for f, s in runs], chunksize=1)
        with open(LEDGER, "a") as fh:
            for i, c in enumerate(todo):
                a = np.asarray(res[i * len(runs):(i + 1) * len(runs)]); g = a - base
                r = {"key": key(c, runs), "cfg": {**c, "cols": list(c["cols"])}, "tag": tag, "box_ap50": a.round(6).tolist(),
                     "gain_mean": round(float(g.mean()), 5), "gain_sd": round(float(g.std(ddof=1)), 5), "n_pos": int((g > 0).sum()),
                     "date": time.strftime("%Y-%m-%d %H:%M")}
                fh.write(json.dumps(r) + "\n"); Lg[r["key"]] = r
        log(f"  [{tag}] {len(todo)} configs x {len(runs)} runs {time.time()-t0:.0f}s")
    return [Lg[key(c, runs)] for c in cfgs]


def stage2(pool, base):
    ref = evaluate(pool, [cfg()], base, tag="ref")[0]; log(f"[stage2] ref mlp2x64 x24 gain {ref['gain_mean']:+.4f} pos {ref['n_pos']}/15")
    for m in MODELS:
        cols = list(ALL); evaluate(pool, [cfg(m)], base, tag=f"stage2:{m}:24")
        while len(cols) > 1:
            rs = evaluate(pool, [cfg(m, tuple(c for c in cols if c != x)) for x in cols], base, tag=f"stage2:{m}:{len(cols)-1}")
            i = int(np.argmax([r["gain_mean"] for r in rs])); drop = cols[i]; cols = [c for c in cols if c != drop]
            log(f"  {m:8s} -{drop:15s} n={len(cols):2d} gain {rs[i]['gain_mean']:+.4f} pos {rs[i]['n_pos']}/15")
    return ref


def select(base):
    Lg = [r for r in ledger().values() if json.loads(r["key"])["runs"] == [list(x) for x in RUNS]]
    ref = next(r for r in Lg if r["key"] == key(cfg(), RUNS))
    adm = [r for r in Lg if r["gain_mean"] >= ref["gain_mean"] - TOL and r["gain_mean"] >= 0.005 and r["n_pos"] >= 13]
    adm.sort(key=lambda r: (len(r["cfg"]["cols"]), MODELS.index(r["cfg"]["model"]), -r["gain_mean"]))
    return ref, adm


def confirm(pool, base, adm):
    rf = evaluate(pool, [cfg()], base, FRESH, "confirm:ref")[0]; tried = []
    for r in adm:
        c = {**r["cfg"], "cols": tuple(r["cfg"]["cols"])}; f = evaluate(pool, [c], base, FRESH, "confirm")[0]
        ok = bool(f["gain_mean"] >= rf["gain_mean"] - TOL and f["gain_mean"] >= 0.005 and f["n_pos"] >= 13)
        tried.append({"cfg": f["cfg"], "fresh_gain_mean": f["gain_mean"], "fresh_n_pos": f["n_pos"], "pass": ok})
        log(f"  [confirm] {f['cfg']['model']} {f['cfg']['cols']} fresh gain {f['gain_mean']:+.4f} pos {f['n_pos']}/15 {'PASS' if ok else 'fail'}")
        if ok:
            return {"fresh_ref_gain": rf["gain_mean"], "tried": tried, "chosen": f["cfg"]}
    return {"fresh_ref_gain": rf["gain_mean"], "tried": tried, "chosen": None}


# ------------------------------------------------------------------ scoring
def score(P, split, name):
    from boxinst_commonality_tcd_04 import score_coco as SC
    fp = os.path.join(OUT, f"preds_{name}_r2048.json"); sp = fp.replace("preds_", "scored_")
    if not os.path.exists(sp):
        json.dump({"meta": {"mask_res": 2048, "model": f"LACE-0 (no mask labels) {name}"}, "preds": P}, open(fp + ".tmp", "w"))
        os.replace(fp + ".tmp", fp)
        cwd = os.getcwd(); os.chdir(REPO)
        json.dump(SC.score_many([fp], GT[split], res=2048, max_dets=600, score_floor=0.0)[0], open(sp + ".tmp", "w")); os.chdir(cwd)
        os.replace(sp + ".tmp", sp)
    c = json.load(open(sp))["canopy_neutral_crowd"]
    return {"mask": {m: c["mask"][m] for m in ("AP50", "AP75", "AP50_95")}, "box": {m: c["box"][m] for m in ("AP50", "AP75")}}, fp


def main():
    import multiprocessing as mp, pickle, contextlib, io
    from boxinst_commonality_tcd_04.dino_det import fasteval as FE
    from boxinst_commonality_tcd_04.dino_det.masker_scalars_d1 import boot
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    os.makedirs(OUT, exist_ok=True); fout = os.path.join(OUT, "lace0.json")
    rec = json.load(open(fout)) if os.path.exists(fout) else {"date_start": time.strftime("%Y-%m-%d %H:%M"), "spec": SPEC}
    save = lambda: json.dump(rec, open(fout, "w"), indent=1)
    vg = json.load(open(GT["val"]))
    with mp.get_context("spawn").Pool(12) as pool:
        if "gates" not in rec:
            D = dets("val"); tids = sorted(vg)
            for t in tids[:3]:                                                      # G1
                M, _ = render("val", t, D[t]["boxes_2048"], D[t]["scores"], DEP_SPEC)
                assert [m["counts"] for m in M] == [m["counts"] for m in json.load(open(f"{FC}/edgefix_val/{t}.json"))["masks_rle"]], f"G1 {t}"
            log("[gate] G1 PASS: deployed spec reproduces edgefix_val masks (3 tiles)")
            Z = np.load(os.path.join(R.FC_, "rerankR1_val.npz")); zt = list(Z["tids"]); Pv0 = json.load(open(f"{FC}/dino_swinl_s0_val_mask.json"))
            for t in zt[:3]:                                                        # G3
                _, X, y = R._feats(("val", t, Pv0[t], vg[t])); i = zt.index(t)
                assert np.allclose(X, Z["X"][Z["tile"] == i]) and np.array_equal(y, Z["y"][Z["tile"] == i]), f"G3 {t}"
            log("[gate] G3 PASS: R1 feature extractor reproduces rerankR1_val.npz (3 tiles)")
            rec["gates"] = {"G1": "PASS", "G3": "PASS"}; save()
        Pv = render_all("val", pool)
        tids, tile, X, y = features("val", Pv, pool)
        eb = f"{FC}/lace0_fasteval_box_val.pkl"
        if not os.path.exists(eb):
            E = build_box(Pv, vg, pool); pickle.dump(E, open(eb + ".tmp", "wb")); os.replace(eb + ".tmp", eb)
        E = pickle.load(open(eb, "rb"))
        if "G2" not in rec["gates"]:                                                 # G2
            gd, t2i = SL.build_gt_crowd(vg, 2048); g2 = {}
            for name, f in {"s": lambda r: r["scores"], "s_prime": lambda r: r["s_prime"],
                            "reverse": lambda r: list(np.arange(len(r["scores"]))[::-1] / 600.0)}.items():
                sc = {t: f(Pv[t]) for t in tids}; fast = round(float(FE.ap(E, sc, thrs=[0.5])[0]), 4)
                pr = {t: {"boxes_2048": Pv[t]["boxes_2048"], "scores": sc[t], "masks_rle": Pv[t]["masks_rle"]} for t in tids}
                with contextlib.redirect_stdout(io.StringIO()):
                    ref = round(SL.score(pr, gd, t2i, masks=False, floor=0.0)["box"]["AP50"], 4)
                g2[name] = [fast, ref]; assert fast == ref, f"G2 {name} {fast} != {ref}"
            rec["gates"]["G2"] = {"PASS": g2}; save(); log(f"[gate] G2 PASS: box fasteval == score_lite {g2}")
        base = rec["gates"]["G2"]["PASS"]["s_prime"][0]
        rec["val"] = {"n_dets": int(len(y)), "pos": int((y == 1).sum()), "neg": int((y == 0).sum()), "ign": int((y < 0).sum()),
                      "box_ap50_s": rec["gates"]["G2"]["PASS"]["s"][0], "box_ap50_sprime": base}; save()
    if "selection" not in rec:
        with mp.get_context("spawn").Pool(NPROC, initializer=_init, initargs=(f"{FC}/lace0_feats_val.npz", eb)) as pool:
            ref = stage2(pool, base); _, adm = select(base)
            log(f"[select] {len(adm)} admissible")
            for r in adm[:10]:
                log(f"   {r['cfg']['model']:8s} n={len(r['cfg']['cols'])} gain {r['gain_mean']:+.4f} pos {r['n_pos']}/15 {r['cfg']['cols']}")
            cf = confirm(pool, base, adm) if adm else {"tried": [], "chosen": None}
        rec["selection"] = {"ref": {k: ref[k] for k in ("gain_mean", "gain_sd", "n_pos")}, "n_admissible": len(adm),
                            "top": [{"cfg": r["cfg"], "gain_mean": r["gain_mean"], "n_pos": r["n_pos"]} for r in adm[:20]], **cf}
        save(); log(f"[select] chosen {rec['selection']['chosen']} (None -> s-hat := s)")
    ch = rec["selection"]["chosen"]
    if ch:
        c = {**ch, "cols": tuple(ch["cols"])}; cols = [R.NAMES.index(x) for x in c["cols"]]; Z = np.load(f"{FC}/lace0_feats_val.npz")
        F = R3.fold_sets(len(Z["tids"]))
        fits = [f for s in range(5) for f in R3.oof_fits(c, s, "rr", Z["X"], Z["y"], None, Z["tile"], F)[1]]; assert len(fits) == 25
    with mp.get_context("spawn").Pool(12) as pool:
        for split in ("test", "sparse"):
            if split in rec:
                continue
            P = render_all(split, pool); tids, tile, X, _ = features(split, P, pool)
            assert np.allclose(np.concatenate([P[t]["s_prime"] for t in tids]), X[:, 1])
            sh = R.split_scores(tids, tile, np.mean([f(X[:, cols]) for f in fits], 0)) if ch else {t: P[t]["scores"] for t in tids}
            rows, fps = {}, {"deployed": DEPLOYED_PREDS[split]}
            for rk, sc in (("shat", sh), ("s", {t: P[t]["scores"] for t in tids})):
                pr = {t: {"boxes_2048": P[t]["boxes_2048"], "scores": sc[t], "masks_rle": P[t]["masks_rle"]} for t in tids}
                rows[rk], fps[rk] = score(pr, split, f"lace0_{split}_{rk}"); log(f"[{split}] LACE-0 {rk}: {rows[rk]}")
            dep = json.load(open(DEPLOYED_PREDS[split].replace("preds_", "scored_")))["canopy_neutral_crowd"]
            rec[split] = {"rows": rows, "deployed": {"mask": {m: dep["mask"][m] for m in ("AP50", "AP75", "AP50_95")}, "box": {"AP50": dep["box"]["AP50"]}},
                          "boot": boot(fps, GT[split], "deployed")}
            save(); log(f"[{split}] boot {rec[split]['boot']}")
    rec["date_end"] = time.strftime("%Y-%m-%d %H:%M"); rec["note"] = "ONE look per pre-registration LACE-0; single detector run"; save()
    log(json.dumps(rec, indent=1))


if __name__ == "__main__":
    _a = argparse.ArgumentParser(); _a.add_argument("cmd", choices=["all"]); _a.parse_args(); main()
