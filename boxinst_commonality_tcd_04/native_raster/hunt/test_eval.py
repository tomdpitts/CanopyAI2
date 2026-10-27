"""TEST evaluation of ONE frozen render spec (chosen on val), 3 seeds, 439 tiles — run once per frozen candidate.

Boxes + LACE product scores: published preds_lace_s{s}_r2048_cgf_r12_tau035_guidedrun.json (verbatim). Masks: the
frozen spec through hunt/render.py (same code as the val arms), local CPU, Modal-cached 8 px test features
(feat_4p_test_dl) and the local 16-phase 4 px features (hunt_z4p_test, gated cos vs Modal). Per-tile JSON checkpoints
in feat_cache/hunt_test_<name>_s<s>/ (resume). Outputs (refuse to overwrite) in hunt/test/:
  preds_lace_s{s}_r2048_<name>.json, scored_lace_s{s}_r2048_<name>.json  (score_coco: 2048, maxDets 600, floor 0, crowd)
  compare_<name>.json: per seed AP50/AP75/AP50:95 vs SAM 3 zs masks at the SAME LACE ranking
  (boxfix/scored_a1_sam3zs_lace_product_s{s}_r2048.json) and vs LACE deployed; paired tile-bootstrap 95% CI of the
  AP50 difference (legacy canopy arm via fastap, 2000 resamples) per seed.
GATE: spec 'deployed' on s0's first 3 tiles reproduces the published masks (byte-identical RLE).

--split sparse: the same on the sparse-236 slice (published preds_lace_sparse_s{s}_r2048_cgf_r12_tau035_*, Modal 8 px
feat_4p_sparse, local hunt_z4p_sparse, GT data/tcd_sparse/sparse_gt.json); checkpoints feat_cache/hunt_sparse_<name>_s<s>/,
outputs hunt/test/{preds,scored}_lace_sparse_s{s}_r2048_<name>.json. No SAM comparison exists there.
--detscore: also writes/scores the same masks ranked by the raw detector score s ({preds,scored}_..._<name>_detscore.json).
--no_compare (implied by --split sparse): skip compare_<name>.json; write summary_<split>_<name>.json (per-seed mask and
box metrics, mean +- sd ddof 1, for s' and, with --detscore, s). Resumes from existing tile/preds/scored files.

  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.hunt.test_eval FROZEN.json NAME [--split sparse] [--detscore] [--no_compare]
"""
import argparse, json, multiprocessing as mp, os, sys, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); NR = os.path.dirname(HERE); PKG = os.path.dirname(NR); REPO = os.path.dirname(PKG)
FC = "/Users/tompitts/dphil/feat_cache"
OUTD = os.path.join(HERE, "test")
_W = {}
SPLITS = {"test": {"pre": "lace", "n": 439, "rgb": os.path.join(REPO, "data", "tcd", "test"),
                   "man": (os.path.join(PKG, "modal_tcd_multiseed", "phase4", "manifest.json"), "feat_test"),
                   "feat8": f"{FC}/feat_4p_test_dl/feat_4p_test", "z4": f"{FC}/hunt_z4p_test", "gt": os.path.join(PKG, "test_gt.json")},
          "sparse": {"pre": "lace_sparse", "n": 236, "rgb": os.path.join(REPO, "data", "tcd_sparse", "test"),
                     "man": (os.path.join(REPO, "data", "tcd_sparse", "slice_manifest.json"), "tiles"),
                     "feat8": f"{FC}/feat_4p_sparse_dl/feat_4p_sparse", "z4": f"{FC}/hunt_z4p_sparse",
                     "gt": os.path.join(REPO, "data", "tcd_sparse", "sparse_gt.json")}}
SP = SPLITS["test"]


def pub(s):
    return os.path.join(NR, f"preds_{SP['pre']}_s{s}_r2048_cgf_r12_tau035_guidedrun.json")


def detscore(s):
    """Same boxes as pub(s), raw detector score s (for spec "rank": s' recomputed from the render's own posterior)."""
    return os.path.join(NR, f"preds_{SP['pre']}_s{s}_r2048_cgf_r12_tau035_detscore.json")


def _init(spec, split="test"):
    global SP
    SP = SPLITS[split]
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.native_raster import guided_ablate_modal as GA
    from boxinst_commonality_tcd_04.native_raster.hunt import render as R
    from boxinst_commonality_tcd_04.native_raster.hunt.pxlace import PixelLACE
    m = E.TCDMasker(f"{FC}/em_model_4p_fix.npz"); m4 = E.TCDMasker(f"{FC}/em_model_4p_fix.npz"); m4.s, m4.origin = 4, 8.0
    _W.update(m=m, m4=m4, GA=GA, R=R, PL=PixelLACE(m), PL4=PixelLACE(m4), spec=spec,
              man=json.load(open(SP["man"][0]))[SP["man"][1]])


def render_tile(t, boxes, s_raw=None):
    """-> (masks_rle, scores). scores is None unless spec["rank"] is set: then s' = s_raw * mean|2r-1| * mean(r) with r the
    box's cell posterior -- 8 px (own8, = the published product) or 4 px (own4, m4, the arm's alpha/kappa)."""
    m, GA, R = _W["m"], _W["GA"], _W["R"]
    bx = np.asarray(boxes, np.float32).reshape(-1, 4)
    out = []; rank = _W["spec"].get("rank"); sc = [] if rank else None
    if not len(bx):
        return out, sc
    ctx = {"rgbf": GA._rgb(SP["rgb"], t, _W["man"]), "m": m, "PL": _W["PL"], "GA": GA}
    feat = np.load(f"{SP['feat8']}/{t}.npy").astype(np.float32)
    g = feat.shape[-1]; zn = m.project(feat); del feat
    ctx.update(zn=zn, g=g, Z=zn.reshape(g, g, -1), boxes=bx)
    z4 = f"{SP['z4']}/{t}.npy"
    if os.path.exists(z4):
        ctx.update(m4=_W["m4"], zn4=np.load(z4).astype(np.float32).reshape(-1, 128), g4=512, PL4=_W["PL4"])
        ctx["Z4"] = ctx["zn4"].reshape(512, 512, 128)
    spec = _W["spec"]
    for i, b in enumerate(bx):
        clip, prob, r = GA._prob(m, zn, g, b, GA.ALPHA, GA.KAPPA)
        det = {"i": i, "box": b, "clip": clip, "prob": prob, "r": r, "cache": {}}
        if rank:
            if rank == "own8":
                rr = r
            else:
                key = ("prob4", spec.get("alpha", 0.3), spec.get("kappa", 1.6))
                if key not in det["cache"]:
                    det["cache"][key] = GA._prob(_W["m4"], ctx["zn4"], ctx["g4"], b, key[1], key[2])
                rr = det["cache"][key][2]
            sc.append(round(float(s_raw[i] * np.abs(2 * rr - 1).mean() * rr.mean()), 8))
        out.append(GA._rle(R.RENDER[spec["kind"]](ctx, det, spec)))
    return out, sc


def _job(args):
    t, boxes, d, s_raw = args
    fp = os.path.join(d, t + ".json")
    if not os.path.exists(fp):
        ms, sc = render_tile(t, boxes, s_raw)
        rec = {"masks_rle": ms} if sc is None else {"masks_rle": ms, "scores": sc}
        json.dump(rec, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    return t


def gate(split="test"):
    _init({"kind": "deployed"}, split)
    P = json.load(open(pub(0)))["preds"]
    tiles = [t for t in sorted(P) if P[t]["scores"]][:3]; n = eq = 0
    for t in tiles:
        ms, _ = render_tile(t, P[t]["boxes_2048"])
        for a, b in zip(ms, P[t]["masks_rle"]):
            n += 1; eq += a["counts"] == b["counts"]
    print(f"[gate] deployed spec vs published masks: {eq}/{n} byte-identical", flush=True)
    return eq == n


def gate_rank(split="test"):
    """own8 ranking recomputed from the 8 px posterior + raw detector score == published product (s0, first 3 tiles)."""
    _init({"kind": "deployed", "rank": "own8"}, split)
    P = json.load(open(pub(0)))["preds"]; D = json.load(open(detscore(0)))["preds"]
    tiles = [t for t in sorted(P) if P[t]["scores"]][:3]; dmax = 0.0; n = 0
    for t in tiles:
        assert D[t]["boxes_2048"] == P[t]["boxes_2048"], t
        _, sc = render_tile(t, P[t]["boxes_2048"], D[t]["scores"])
        dmax = max(dmax, float(np.max(np.abs(np.asarray(sc) - np.asarray(P[t]["scores"]))))); n += len(sc)
    print(f"[gate] own8 s' vs published product: max |diff| {dmax:.2e} over {n} boxes", flush=True)
    # tolerance: published test products were computed on Modal (float32 path); local recomputation differs by ~1.5e-6,
    # the same offset accepted by the A2 gate (tierA/a2_gate.json, score diff <= 1.5e-6). A code error is >> 1e-5.
    return dmax <= 1e-5


def boot_diff(statsA, statsB, scores, n_gt, tids, n=2000, seed=0):
    from boxinst_commonality_tcd_04.native_raster.hunt import fastap as F
    rng = np.random.default_rng(seed); T = len(tids); d = []
    for _ in range(n):
        ix = rng.integers(0, T, T)
        ng = sum(n_gt[tids[i]] for i in ix)
        a = F.ap([statsA[i] for i in ix], [scores[i] for i in ix], ng); b = F.ap([statsB[i] for i in ix], [scores[i] for i in ix], ng)
        d.append(a - b)
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


def _metrics(fp):
    r = json.load(open(fp))["canopy_neutral_crowd"]
    return {k: r["mask"][k] for k in ("AP50", "AP75", "AP50_95")} | {"box_AP50": r["box"]["AP50"]}


def _band(rows):
    out = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows]); out[k] = [round(float(v.mean()), 4), round(float(v.std(ddof=1)), 4)]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("spec_fp"); ap.add_argument("name")
    ap.add_argument("--split", default="test", choices=sorted(SPLITS)); ap.add_argument("--detscore", action="store_true")
    ap.add_argument("--no_compare", action="store_true")
    a = ap.parse_args(); spec_fp, name, split = a.spec_fp, a.name, a.split
    global SP
    SP = SPLITS[split]; compare = split == "test" and not a.no_compare
    spec = json.load(open(spec_fp)); spec = spec.get("spec", spec)
    os.makedirs(OUTD, exist_ok=True)
    cmp_fp = os.path.join(OUTD, f"compare_{name}.json")
    sum_fp = os.path.join(OUTD, f"summary_{split}_{name}.json")
    assert not os.path.exists(cmp_fp if compare else sum_fp), (cmp_fp if compare else sum_fp)
    assert gate(split), "GATE FAILED"
    if spec.get("rank"):
        assert gate_rank(split), "RANK GATE FAILED"
    need4 = "px4" in json.dumps(spec) or "post4" in json.dumps(spec)
    if need4:
        n4 = len([f for f in os.listdir(SP["z4"]) if f.endswith(".npy") and not f.endswith(".tmp.npy")])
        assert n4 == SP["n"], f"4 px {split} features incomplete: {n4}/{SP['n']}"
    from boxinst_commonality_tcd_04 import score_coco as SC
    from boxinst_commonality_tcd_04.native_raster.hunt import fastap as F
    res = {"spec": spec, "name": name, "split": split, "seeds": {}}
    paths, dpaths = [], []
    for s in (0, 1, 2):
        P = json.load(open(pub(s)))["preds"]; tids = sorted(P); assert len(tids) == SP["n"]
        D = json.load(open(detscore(s)))["preds"] if (spec.get("rank") or a.detscore) else None
        if D is not None:
            assert all(D[t]["boxes_2048"] == P[t]["boxes_2048"] for t in tids), "detscore boxes differ"
        d = f"{FC}/hunt_{split}_{name}_s{s}"; os.makedirs(d, exist_ok=True)
        todo = [(t, P[t]["boxes_2048"], d, D[t]["scores"] if spec.get("rank") else None) for t in tids if not os.path.exists(os.path.join(d, t + ".json"))]
        print(f"[{split} {name} s{s}] {len(todo)}/{SP['n']} tiles to render", flush=True); t0 = time.time()
        if todo:
            with mp.get_context("spawn").Pool(12, initializer=_init, initargs=(spec, split)) as pool:
                for k, _ in enumerate(pool.imap_unordered(_job, todo), 1):
                    if k % 20 == 0 or k == len(todo):
                        print(f"  s{s} {k}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
        fp = os.path.join(OUTD, f"preds_{SP['pre']}_s{s}_r2048_{name}.json")
        dfp = fp.replace(".json", "_detscore.json")
        for out_fp, key in ((fp, "own"), (dfp, "det")) if a.detscore else ((fp, "own"),):
            if os.path.exists(out_fp):
                continue
            preds = {}
            for t in tids:
                rec = json.load(open(os.path.join(d, t + ".json")))
                sc = D[t]["scores"] if key == "det" else (rec["scores"] if spec.get("rank") else P[t]["scores"])
                preds[t] = {"boxes_2048": P[t]["boxes_2048"], "scores": sc, "masks_rle": rec["masks_rle"]}
            rk = "raw detector score s" if key == "det" else ("own-posterior s'" if spec.get("rank") else "published product")
            json.dump({"meta": {"mask_res": 2048, "model": f"LACE {split} s{s} boxes, ranking {rk}, hunt render {name}", "spec": spec},
                       "preds": preds}, open(out_fp + ".tmp", "w")); os.replace(out_fp + ".tmp", out_fp)
        paths.append(fp); dpaths.append(dfp)
    os.chdir(REPO)
    allp = paths + (dpaths if a.detscore else [])
    todo_sc = [p for p in allp if not os.path.exists(p.replace("preds_", "scored_"))]      # resume: keep scored files
    for p, r in zip(todo_sc, SC.score_many(todo_sc, SP["gt"], res=2048, max_dets=600, score_floor=0.0) if todo_sc else []):
        json.dump(r, open(p.replace("preds_", "scored_"), "w"))
    if not compare:
        own = [_metrics(p.replace("preds_", "scored_")) for p in paths]
        summ = {"spec": spec, "name": name, "split": split, "files": [p.replace("preds_", "scored_") for p in allp],
                "rank_own": {"per_seed": own, "band": _band(own)}}
        if a.detscore:
            det = [_metrics(p.replace("preds_", "scored_")) for p in dpaths]
            summ["rank_detscore"] = {"per_seed": det, "band": _band(det)}
        json.dump(summ, open(sum_fp, "w"), indent=1)
        print(json.dumps({k: v["band"] for k, v in summ.items() if k.startswith("rank_")}), flush=True)
        return
    gtc = F.gt_cache("test")
    for s, fp in zip((0, 1, 2), paths):
        r = json.load(open(fp.replace("preds_", "scored_")))
        sam = json.load(open(os.path.join(NR, "boxfix", f"scored_a1_sam3zs_lace_product_s{s}_r2048.json")))["canopy_neutral_crowd"]["mask"]
        dep = json.load(open(os.path.join(NR, f"scored_lace_s{s}_r2048_cgf_r12_tau035_guidedrun.json")))["canopy_neutral_crowd"]["mask"]
        me = r["canopy_neutral_crowd"]["mask"]
        a1 = json.load(open(os.path.join(NR, "tierA", f"scored_a1_lace_s{s}_sam3zs_gmean.json")))["canopy_neutral_crowd"]["mask"]
        c1 = os.path.join(OUTD, f"scored_lace_s{s}_r2048_lace4px_g30.json")
        c1 = json.load(open(c1))["canopy_neutral_crowd"]["mask"] if os.path.exists(c1) else None
        # paired tile bootstrap (legacy arm) vs SAM
        P = json.load(open(fp))["preds"]; S = json.load(open(os.path.join(NR, "boxfix", f"preds_a1_sam3zs_lace_product_s{s}_r2048.json")))["preds"]
        tids = sorted(gtc); stA, stB, sc = [], [], []
        for t in tids:
            stA.append(F.tile_stats(P[t]["masks_rle"], gtc[t])); stB.append(F.tile_stats(S[t]["masks_rle"], gtc[t])); sc.append(P[t]["scores"])
        n_gt = {t: len(gtc[t]["trees"]) for t in tids}
        la, sa = F.ap(stA, sc, sum(n_gt.values())), F.ap(stB, sc, sum(n_gt.values()))
        res["seeds"][s] = {"lace_new": me, "sam_own_rank_A1": a1, "AP50_minus_sam_own_rank": round(me["AP50"] - a1["AP50"], 4),
                           "box": r["canopy_neutral_crowd"]["box"], "cand1": c1,
                           "sam_at_lace_rank": sam, "lace_deployed": dep,
                           "AP50_minus_sam": round(me["AP50"] - sam["AP50"], 4), "AP50_minus_deployed": round(me["AP50"] - dep["AP50"], 4),
                           "legacy_AP50": {"lace_new": round(la, 4), "sam": round(sa, 4)},
                           "ci95_AP50_minus_sam_tile_boot_legacy": boot_diff(stA, stB, sc, n_gt, tids)}
        print(f"[test {name} s{s}] AP50 {me['AP50']:.4f} AP50:95 {me['AP50_95']:.4f} | SAM own rank (A1) {a1['AP50']:.4f} / {a1['AP50_95']:.4f} | "
              f"diff {me['AP50']-a1['AP50']:+.4f} | cand1 {c1['AP50'] if c1 else float('nan'):.4f}", flush=True)
        print(f"[test {name} s{s}] AP50 {me['AP50']:.4f} | SAM@LACE-rank {sam['AP50']:.4f} | deployed {dep['AP50']:.4f} | "
              f"diff vs SAM {me['AP50']-sam['AP50']:+.4f} CI {res['seeds'][s]['ci95_AP50_minus_sam_tile_boot_legacy']}", flush=True)
    a = np.array([res["seeds"][s]["lace_new"]["AP50"] for s in (0, 1, 2)]); b = np.array([res["seeds"][s]["sam_at_lace_rank"]["AP50"] for s in (0, 1, 2)])
    c = np.array([res["seeds"][s]["sam_own_rank_A1"]["AP50"] for s in (0, 1, 2)])
    a95 = np.array([res["seeds"][s]["lace_new"]["AP50_95"] for s in (0, 1, 2)])
    res["band_vs_sam_own_rank"] = {"lace_new_AP50_95": [round(a95.mean(), 4), round(a95.std(ddof=1), 4)],
                                   "sam_own_rank_AP50": [round(c.mean(), 4), round(c.std(ddof=1), 4)],
                                   "paired_diff": [round((a - c).mean(), 4), round((a - c).std(ddof=1), 4)]}
    print(json.dumps(res["band_vs_sam_own_rank"]), flush=True)
    res["band"] = {"lace_new_AP50": [round(a.mean(), 4), round(a.std(ddof=1), 4)], "sam_AP50": [round(b.mean(), 4), round(b.std(ddof=1), 4)],
                   "paired_diff": [round((a - b).mean(), 4), round((a - b).std(ddof=1), 4)]}
    json.dump(res, open(cmp_fp, "w"), indent=1); print(json.dumps(res["band"]), flush=True)


if __name__ == "__main__":
    main()
