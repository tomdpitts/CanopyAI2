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

  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.hunt.test_eval FROZEN.json NAME
"""
import json, multiprocessing as mp, os, sys, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); NR = os.path.dirname(HERE); PKG = os.path.dirname(NR); REPO = os.path.dirname(PKG)
FC = "/Users/tompitts/dphil/feat_cache"
OUTD = os.path.join(HERE, "test")
_W = {}


def pub(s):
    return os.path.join(NR, f"preds_lace_s{s}_r2048_cgf_r12_tau035_guidedrun.json")


def _init(spec):
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.native_raster import guided_ablate_modal as GA
    from boxinst_commonality_tcd_04.native_raster.hunt import render as R
    from boxinst_commonality_tcd_04.native_raster.hunt.pxlace import PixelLACE
    m = E.TCDMasker(f"{FC}/em_model_4p_fix.npz"); m4 = E.TCDMasker(f"{FC}/em_model_4p_fix.npz"); m4.s, m4.origin = 4, 8.0
    _W.update(m=m, m4=m4, GA=GA, R=R, PL=PixelLACE(m), PL4=PixelLACE(m4), spec=spec,
              man=json.load(open(os.path.join(PKG, "modal_tcd_multiseed", "phase4", "manifest.json")))["feat_test"])


def render_tile(t, boxes):
    m, GA, R = _W["m"], _W["GA"], _W["R"]
    bx = np.asarray(boxes, np.float32).reshape(-1, 4)
    out = []
    if not len(bx):
        return out
    ctx = {"rgbf": GA._rgb(os.path.join(REPO, "data", "tcd", "test"), t, _W["man"]), "m": m, "PL": _W["PL"], "GA": GA}
    feat = np.load(f"{FC}/feat_4p_test_dl/feat_4p_test/{t}.npy").astype(np.float32)
    g = feat.shape[-1]; zn = m.project(feat); del feat
    ctx.update(zn=zn, g=g, Z=zn.reshape(g, g, -1), boxes=bx)
    z4 = f"{FC}/hunt_z4p_test/{t}.npy"
    if os.path.exists(z4):
        ctx.update(m4=_W["m4"], zn4=np.load(z4).astype(np.float32).reshape(-1, 128), g4=512, PL4=_W["PL4"])
        ctx["Z4"] = ctx["zn4"].reshape(512, 512, 128)
    spec = _W["spec"]
    for i, b in enumerate(bx):
        clip, prob, r = GA._prob(m, zn, g, b, GA.ALPHA, GA.KAPPA)
        det = {"i": i, "box": b, "clip": clip, "prob": prob, "r": r, "cache": {}}
        out.append(GA._rle(R.RENDER[spec["kind"]](ctx, det, spec)))
    return out


def _job(args):
    t, boxes, d = args
    fp = os.path.join(d, t + ".json")
    if not os.path.exists(fp):
        json.dump({"masks_rle": render_tile(t, boxes)}, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    return t


def gate():
    _init({"kind": "deployed"})
    P = json.load(open(pub(0)))["preds"]
    tiles = [t for t in sorted(P) if P[t]["scores"]][:3]; n = eq = 0
    for t in tiles:
        ms = render_tile(t, P[t]["boxes_2048"])
        for a, b in zip(ms, P[t]["masks_rle"]):
            n += 1; eq += a["counts"] == b["counts"]
    print(f"[gate] deployed spec vs published masks: {eq}/{n} byte-identical", flush=True)
    return eq == n


def boot_diff(statsA, statsB, scores, n_gt, tids, n=2000, seed=0):
    from boxinst_commonality_tcd_04.native_raster.hunt import fastap as F
    rng = np.random.default_rng(seed); T = len(tids); d = []
    for _ in range(n):
        ix = rng.integers(0, T, T)
        ng = sum(n_gt[tids[i]] for i in ix)
        a = F.ap([statsA[i] for i in ix], [scores[i] for i in ix], ng); b = F.ap([statsB[i] for i in ix], [scores[i] for i in ix], ng)
        d.append(a - b)
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


def main():
    spec_fp, name = sys.argv[1], sys.argv[2]
    spec = json.load(open(spec_fp)); spec = spec.get("spec", spec)
    os.makedirs(OUTD, exist_ok=True)
    cmp_fp = os.path.join(OUTD, f"compare_{name}.json"); assert not os.path.exists(cmp_fp), cmp_fp
    assert gate(), "GATE FAILED"
    need4 = "px4" in json.dumps(spec) or "post4" in json.dumps(spec)
    if need4:
        n4 = len([f for f in os.listdir(f"{FC}/hunt_z4p_test") if f.endswith(".npy") and not f.endswith(".tmp.npy")])
        assert n4 == 439, f"4 px test features incomplete: {n4}/439"
    from boxinst_commonality_tcd_04 import score_coco as SC
    from boxinst_commonality_tcd_04.native_raster.hunt import fastap as F
    gtc = F.gt_cache("test"); res = {"spec": spec, "name": name, "seeds": {}}
    paths = []
    for s in (0, 1, 2):
        P = json.load(open(pub(s)))["preds"]; tids = sorted(P); assert len(tids) == 439
        d = f"{FC}/hunt_test_{name}_s{s}"; os.makedirs(d, exist_ok=True)
        todo = [(t, P[t]["boxes_2048"], d) for t in tids if not os.path.exists(os.path.join(d, t + ".json"))]
        print(f"[test {name} s{s}] {len(todo)}/439 tiles to render", flush=True); t0 = time.time()
        with mp.get_context("spawn").Pool(12, initializer=_init, initargs=(spec,)) as pool:
            for k, _ in enumerate(pool.imap_unordered(_job, todo), 1):
                if k % 20 == 0 or k == len(todo):
                    print(f"  s{s} {k}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
        fp = os.path.join(OUTD, f"preds_lace_s{s}_r2048_{name}.json")
        if not os.path.exists(fp):
            preds = {t: {"boxes_2048": P[t]["boxes_2048"], "scores": P[t]["scores"],
                         "masks_rle": json.load(open(os.path.join(d, t + ".json")))["masks_rle"]} for t in tids}
            json.dump({"meta": {"mask_res": 2048, "model": f"LACE s{s} boxes+product, hunt render {name}", "spec": spec}, "preds": preds}, open(fp, "w"))
        paths.append(fp)
    os.chdir(REPO)
    todo_sc = [p for p in paths if not os.path.exists(p.replace("preds_", "scored_"))]      # resume: keep scored seeds
    for p, r in zip(todo_sc, SC.score_many(todo_sc, os.path.join(PKG, "test_gt.json"), res=2048, max_dets=600, score_floor=0.0) if todo_sc else []):
        json.dump(r, open(p.replace("preds_", "scored_"), "w"))
    for s, fp in zip((0, 1, 2), paths):
        r = json.load(open(fp.replace("preds_", "scored_")))
        sam = json.load(open(os.path.join(NR, "boxfix", f"scored_a1_sam3zs_lace_product_s{s}_r2048.json")))["canopy_neutral_crowd"]["mask"]
        dep = json.load(open(os.path.join(NR, f"scored_lace_s{s}_r2048_cgf_r12_tau035_guidedrun.json")))["canopy_neutral_crowd"]["mask"]
        me = r["canopy_neutral_crowd"]["mask"]
        # paired tile bootstrap (legacy arm) vs SAM
        P = json.load(open(fp))["preds"]; S = json.load(open(os.path.join(NR, "boxfix", f"preds_a1_sam3zs_lace_product_s{s}_r2048.json")))["preds"]
        tids = sorted(gtc); stA, stB, sc = [], [], []
        for t in tids:
            stA.append(F.tile_stats(P[t]["masks_rle"], gtc[t])); stB.append(F.tile_stats(S[t]["masks_rle"], gtc[t])); sc.append(P[t]["scores"])
        n_gt = {t: len(gtc[t]["trees"]) for t in tids}
        la, sa = F.ap(stA, sc, sum(n_gt.values())), F.ap(stB, sc, sum(n_gt.values()))
        res["seeds"][s] = {"lace_new": me, "sam_at_lace_rank": sam, "lace_deployed": dep,
                           "AP50_minus_sam": round(me["AP50"] - sam["AP50"], 4), "AP50_minus_deployed": round(me["AP50"] - dep["AP50"], 4),
                           "legacy_AP50": {"lace_new": round(la, 4), "sam": round(sa, 4)},
                           "ci95_AP50_minus_sam_tile_boot_legacy": boot_diff(stA, stB, sc, n_gt, tids)}
        print(f"[test {name} s{s}] AP50 {me['AP50']:.4f} | SAM@LACE-rank {sam['AP50']:.4f} | deployed {dep['AP50']:.4f} | "
              f"diff vs SAM {me['AP50']-sam['AP50']:+.4f} CI {res['seeds'][s]['ci95_AP50_minus_sam_tile_boot_legacy']}", flush=True)
    a = np.array([res["seeds"][s]["lace_new"]["AP50"] for s in (0, 1, 2)]); b = np.array([res["seeds"][s]["sam_at_lace_rank"]["AP50"] for s in (0, 1, 2)])
    res["band"] = {"lace_new_AP50": [round(a.mean(), 4), round(a.std(ddof=1), 4)], "sam_AP50": [round(b.mean(), 4), round(b.std(ddof=1), 4)],
                   "paired_diff": [round((a - b).mean(), 4), round((a - b).std(ddof=1), 4)]}
    json.dump(res, open(cmp_fp, "w"), indent=1); print(json.dumps(res["band"]), flush=True)


if __name__ == "__main__":
    main()
