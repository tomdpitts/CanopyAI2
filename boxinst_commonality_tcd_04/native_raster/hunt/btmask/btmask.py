"""LACE's masker (lace4px_simple) on BoxTeacher B2's stitched detections. Pre-registration: btmask/README.md.

  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.hunt.btmask.btmask gate
  ... btmask val                       # 9-config grid on the 108 val tiles -> val_grid.json + selection
  ... btmask test                      # arms A, B0, B* (from val_grid.json) on the 439 test tiles -> compare.json

Per-tile JSON checkpoints in feat_cache/btmask_<split>_<cfg>/ (resume). Outputs refuse to overwrite.
"""
import argparse, json, multiprocessing as mp, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); HUNT = os.path.dirname(HERE); NR = os.path.dirname(HUNT)
PKG = os.path.dirname(NR); REPO = os.path.dirname(PKG)
FC = "/Users/tompitts/dphil/feat_cache"
BT = os.path.join(PKG, "box_supervised_baselines", "tierb", "results", "20261101_192510", "boxteacher", "results")
TOPK = 600
FROZEN = {"e": 0.0, "c": 0.33}                       # lace4px_simple: post4, c_peak 0.33, r 12, eps 1e-4, a 0.3, k 1.6
GRID = [{"e": e, "c": c} for e in (0.0, 0.05, 0.10) for c in (0.25, 0.33, 0.40)]
SPLIT = {"test": {"z4": f"{FC}/hunt_z4p_test", "gt": os.path.join(PKG, "test_gt.json"), "n": 439},
         "val": {"z4": f"{FC}/hunt_z4p_val", "gt": os.path.join(PKG, "modal_tcd_multiseed", "phase4", "val_gt.json"), "n": 108}}
_W = {}


def cfg_name(c):
    return f"e{c['e']:.2f}_c{c['c']:.2f}"


def bt_top(split):
    """BoxTeacher's stitched detections, top TOPK per tile by its score (stable order)."""
    P = json.load(open(os.path.join(BT, f"preds_boxteacher_sub_{split}_r2048.json")))["preds"]
    out = {}
    for t, r in P.items():
        o = np.argsort(-np.asarray(r["scores"], np.float64), kind="stable")[:TOPK]
        out[t] = {k: [r[k][i] for i in o] for k in ("boxes_2048", "scores", "masks_rle")}
    return out


def _rgb(split, t):
    from boxinst_commonality_tcd_04.native_raster import guided_ablate_modal as GA, guided_rgb_ext_val as GV
    if split == "test":
        man = json.load(open(os.path.join(PKG, "modal_tcd_multiseed", "phase4", "manifest.json")))["feat_test"]
        return GA._rgb(os.path.join(REPO, "data", "tcd", "test"), t, man)
    man = json.load(open(GV.MANIFEST))["feat_traintile"]
    return GV.load_rgb_verified(t, man).astype(np.float32) / 255.0


def _init(split):
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.native_raster import guided_ablate_modal as GA
    from boxinst_commonality_tcd_04.native_raster.hunt import render as R
    m4 = E.TCDMasker(f"{FC}/em_model_4p_fix.npz"); m4.s, m4.origin = 4, 8.0
    _W.update(GA=GA, R=R, m4=m4, split=split)


def inflate(b, e):
    x0, y0, x1, y1 = (float(v) for v in b)
    if e == 0:
        return np.asarray(b, np.float32)
    dw, dh = e * (x1 - x0), e * (y1 - y0)
    return np.array([max(0.0, x0 - dw), max(0.0, y0 - dh), min(2048.0, x1 + dw), min(2048.0, y1 + dh)], np.float32)


def render_boxes(split, t, boxes, scores, cfgs):
    """-> {cfg_name: {"masks_rle": [...], "s_prime": [...]}}. post4 exactly as hunt/render.py (spec of lace4px_simple
    with c_peak = cfg c), on the box inflated by cfg e; s' = s*mean|2r-1|*mean(r) from that box's 4 px posterior."""
    GA, R, m4 = _W["GA"], _W["R"], _W["m4"]
    out = {cfg_name(c): {"masks_rle": [], "s_prime": []} for c in cfgs}
    if not len(boxes):
        return out
    ctx = {"rgbf": _rgb(split, t), "GA": GA, "m4": m4, "g4": 512}
    ctx["zn4"] = np.load(f"{SPLIT[split]['z4']}/{t}.npy").astype(np.float32).reshape(-1, 128)
    es = sorted({c["e"] for c in cfgs})
    for i, b in enumerate(boxes):
        for e in es:
            bb = inflate(b, e)
            post = GA._prob(m4, ctx["zn4"], 512, bb, 0.3, 1.6)
            det = {"i": i, "box": bb, "cache": {("prob4", 0.3, 1.6): post}}
            r = post[2]
            sp = round(float(scores[i] * np.abs(2 * r - 1).mean() * r.mean()), 8) if scores is not None else None
            for c in cfgs:
                if c["e"] != e:
                    continue
                k = cfg_name(c)
                out[k]["masks_rle"].append(GA._rle(R.post4(ctx, det, {"kind": "post4", "c_peak": c["c"]})))
                out[k]["s_prime"].append(sp)
    return out


def _job(args):
    split, t, boxes, scores, cfgs = args
    todo = [c for c in cfgs if not os.path.exists(f"{FC}/btmask_{split}_{cfg_name(c)}/{t}.json")]
    if todo:
        res = render_boxes(split, t, boxes, scores, todo)
        for c in todo:
            d = f"{FC}/btmask_{split}_{cfg_name(c)}"; os.makedirs(d, exist_ok=True)
            fp = f"{d}/{t}.json"
            json.dump(res[cfg_name(c)], open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    return t


def run(split, cfgs, procs=12):
    B = bt_top(split); tids = sorted(B); assert len(tids) == SPLIT[split]["n"], len(tids)
    jobs = [(split, t, B[t]["boxes_2048"], B[t]["scores"], cfgs) for t in tids
            if not all(os.path.exists(f"{FC}/btmask_{split}_{cfg_name(c)}/{t}.json") for c in cfgs)]
    print(f"[{split}] {len(jobs)}/{len(tids)} tiles to render, cfgs {[cfg_name(c) for c in cfgs]}", flush=True)
    t0 = time.time()
    if jobs:
        with mp.get_context("spawn").Pool(procs, initializer=_init, initargs=(split,)) as pool:
            for k, _ in enumerate(pool.imap_unordered(_job, jobs), 1):
                if k % 10 == 0 or k == len(jobs):
                    print(f"  {k}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    return B, tids


def preds_for(split, B, tids, c, ranking):
    """ranking 'bt' = BoxTeacher's score (as arm A); 'sprime' = LACE's s' with s = BoxTeacher's score."""
    preds = {}
    for t in tids:
        rec = json.load(open(f"{FC}/btmask_{split}_{cfg_name(c)}/{t}.json"))
        sc = B[t]["scores"] if ranking == "bt" else rec["s_prime"]
        preds[t] = {"boxes_2048": B[t]["boxes_2048"], "scores": sc, "masks_rle": rec["masks_rle"]}
    return preds


# ------------------------------------------------------------------ gates
def gate():
    """Masker code: render_boxes on LACE s0's published boxes == published lace4px_simple masks (3 test tiles)."""
    _init("test")
    P = json.load(open(os.path.join(HUNT, "test", "preds_lace_s0_r2048_lace4px_simple.json")))["preds"]
    D = json.load(open(os.path.join(NR, "preds_lace_s0_r2048_cgf_r12_tau035_detscore.json")))["preds"]
    tiles = [t for t in sorted(P) if P[t]["scores"]][:3]; n = eq = 0; dmax = 0.0
    for t in tiles:
        assert D[t]["boxes_2048"] == P[t]["boxes_2048"]
        r = render_boxes("test", t, np.asarray(P[t]["boxes_2048"], np.float32).reshape(-1, 4), D[t]["scores"], [FROZEN])[cfg_name(FROZEN)]
        for a, b in zip(r["masks_rle"], P[t]["masks_rle"]):
            n += 1; eq += a["counts"] == b["counts"]
        dmax = max(dmax, float(np.max(np.abs(np.asarray(r["s_prime"]) - np.asarray(P[t]["scores"])))))
    ok = eq == n and dmax <= 1e-6
    rec = {"gate": "render_boxes(LACE s0 published boxes) == published lace4px_simple masks + own4 s'",
           "tiles": tiles, "masks_identical": f"{eq}/{n}", "s_prime_max_abs_diff": dmax, "pass": ok,
           "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(rec, open(os.path.join(HERE, "gate_masker.json"), "w"), indent=1)
    print(json.dumps(rec), flush=True)
    return ok


# ------------------------------------------------------------------ val grid
def val():
    fp = os.path.join(HERE, "val_grid.json"); assert not os.path.exists(fp), fp
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    B, tids = run("val", GRID)
    vg = json.load(open(SPLIT["val"]["gt"])); vg = {t: vg[t] for t in tids}
    gd, t2i = SL.build_gt_crowd(vg, 2048)
    rows = {}
    A = {t: {k: B[t][k] for k in ("boxes_2048", "scores", "masks_rle")} for t in tids}
    rows["A_boxteacher"] = SL.score(A, gd, t2i, masks=True)["mask"]
    for c in GRID:
        rows[cfg_name(c)] = SL.score(preds_for("val", B, tids, c, "bt"), gd, t2i, masks=True)["mask"]
        print(f"[val] {cfg_name(c)} {rows[cfg_name(c)]}", flush=True)
    best = max(GRID, key=lambda c: (rows[cfg_name(c)]["AP50"], c == FROZEN))
    sel = FROZEN if rows[cfg_name(best)]["AP50"] - rows[cfg_name(FROZEN)]["AP50"] <= 0.001 else best
    rec = {"rule": "highest val mask AP50 (score_lite crowd, BT ranking); within 0.001 of frozen -> frozen",
           "rows": rows, "best": cfg_name(best), "selected": cfg_name(sel), "selected_cfg": sel,
           "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(rec, open(fp, "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("best", "selected")}), flush=True)


# ------------------------------------------------------------------ test
def test():
    out = os.path.join(HERE, "compare.json"); assert not os.path.exists(out), out
    sel = json.load(open(os.path.join(HERE, "val_grid.json")))["selected_cfg"]
    cfgs = [FROZEN] + ([sel] if sel != FROZEN else [])
    B, tids = run("test", cfgs)
    from boxinst_commonality_tcd_04 import score_coco as SC
    from boxinst_commonality_tcd_04.native_raster.hunt import fastap as F
    files = {}

    def write(name, preds, model):
        fp = os.path.join(HERE, f"preds_{name}_r2048.json")
        if not os.path.exists(fp):
            json.dump({"meta": {"mask_res": 2048, "model": model}, "preds": preds}, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
        files[name] = fp
    write("A_boxteacher_top600", {t: {k: B[t][k] for k in ("boxes_2048", "scores", "masks_rle")} for t in tids},
          "BoxTeacher B2 test, top 600/tile by its score, its own masks (arm A)")
    for c in cfgs:
        n = cfg_name(c)
        write(f"B_lace_{n}_btrank", preds_for("test", B, tids, c, "bt"), f"LACE post4 {n} on BoxTeacher boxes, BoxTeacher score")
        write(f"B_lace_{n}_sprime", preds_for("test", B, tids, c, "sprime"), f"LACE post4 {n} on BoxTeacher boxes, s' (s = BoxTeacher score)")
    os.chdir(REPO)
    floors = {k: (0.0 if k.endswith("_sprime") else 0.05) for k in files}
    for k, fp in files.items():
        sp = fp.replace("preds_", "scored_")
        if not os.path.exists(sp):
            json.dump(SC.score_many([fp], SPLIT["test"]["gt"], res=2048, max_dets=600, score_floor=floors[k])[0], open(sp, "w"))
    met = {}
    for k, fp in files.items():
        r = json.load(open(fp.replace("preds_", "scored_")))["canopy_neutral_crowd"]
        met[k] = {kk: r["mask"][kk] for kk in ("AP50", "AP75", "AP50_95", "n_det")} | {"box_AP50": r["box"]["AP50"], "floor": floors[k]}
    pub = json.load(open(os.path.join(BT, "scored_boxteacher_sub_test_r2048.json")))["canopy_neutral_crowd"]["mask"]
    gate_A = all(met["A_boxteacher_top600"][k] == pub[k] for k in ("AP50", "AP75", "AP50_95"))
    # paired tile bootstrap (legacy arm), same BoxTeacher ranking: B - A
    gtc = F.gt_cache("test"); T = sorted(gtc); n_gt = {t: len(gtc[t]["trees"]) for t in T}
    PA = json.load(open(files["A_boxteacher_top600"]))["preds"]
    stA = [F.tile_stats(PA[t]["masks_rle"], gtc[t]) for t in T]; sc = [PA[t]["scores"] for t in T]
    boot = {}
    for c in cfgs:
        n = cfg_name(c); PB = json.load(open(files[f"B_lace_{n}_btrank"]))["preds"]
        stB = [F.tile_stats(PB[t]["masks_rle"], gtc[t]) for t in T]
        rng = np.random.default_rng(0); d = []
        for _ in range(2000):
            ix = rng.integers(0, len(T), len(T)); ng = sum(n_gt[T[i]] for i in ix)
            d.append(F.ap([stB[i] for i in ix], [sc[i] for i in ix], ng) - F.ap([stA[i] for i in ix], [sc[i] for i in ix], ng))
        full = F.ap(stB, sc, sum(n_gt.values())) - F.ap(stA, sc, sum(n_gt.values()))
        boot[n] = {"legacy_AP50_B_minus_A": round(float(full), 4),
                   "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}
    rec = {"gate_A_reproduces_published": gate_A, "selected_val_cfg": cfg_name(sel), "metrics": met, "boot": boot,
           "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(rec, open(out, "w"), indent=1)
    print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["gate", "val", "test"]); a = ap.parse_args()
    if a.cmd == "gate":
        assert gate(), "GATE FAILED"
    elif a.cmd == "val":
        val()
    else:
        test()
