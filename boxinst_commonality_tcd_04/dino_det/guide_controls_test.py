"""Pre-registration GC-test (README.md, 2026-11-04, Tom: "i think 439 is enough. run it locally"). ONE look, test 439.
Guide controls for the colour-guided render on D1: same frozen D1 test boxes/scores, lace4px_simple post4 (c_peak 0.33), EDGE_FILL on;
only the upsampling step differs. Arms: guided (deployed; masks reused from results/edgefix/, gated by byte-identical re-render of the
first 3 sorted tiles), bilinear (r 0), blur (constant guide, box 2r+1, r 12), wrongtile (colour guide from the next test tile, sorted,
wrapping). Rankings: s' (primary; = "+ edge fix" row) and frozen R3 (secondary; = "+ edge fix + R3" row).
Frozen score_coco per arm x ranking; fasteval asserted equal; paired tile bootstrap (2000, seed 0) guided - each control on AP50 / AP75 /
AP50:95. Masks -> feat_cache/guidectl_test_<arm>/<tid>.json (resume). -> results/guidectl/guidectl_test.json
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.guide_controls_test > native_raster/logs/guidectl_test_2026-11-04.log 2>&1
"""
import json, multiprocessing as mp, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); REPO = os.path.dirname(PKG)
FC = "/Users/tompitts/dphil/feat_cache"; RES = os.path.join(HERE, "results"); OUT = os.path.join(RES, "guidectl"); EF = os.path.join(RES, "edgefix")
SPLIT = "test"; C_PEAK = 0.33; N_BOOT, SEED = 2000, 0
ARMS = {"guided": {}, "bilinear": {"r": 0}, "blur": {"guide": "blur"}, "wrongtile": {"guide": "wrong"}}
CONTROLS = ("bilinear", "blur", "wrongtile")


def _job(args):
    """Exactly btmask.render_boxes (e = 0, one cfg) with extra post4 spec keys per arm; posterior computed once per box."""
    t, other, boxes, scores, arms = args
    from boxinst_commonality_tcd_04.native_raster.hunt.btmask import btmask as BM
    if "m4" not in BM._W:
        BM._init(SPLIT)
    GA, R, m4 = BM._W["GA"], BM._W["R"], BM._W["m4"]
    assert GA.EDGE_FILL is True
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4); out = {a: {"masks_rle": [], "s_prime": []} for a in arms}
    if not len(boxes):
        return t, out
    ctx = {"rgbf": BM._rgb(SPLIT, t), "GA": GA, "m4": m4, "g4": 512}
    if "wrongtile" in arms:
        ctx["rgbf_wrong"] = BM._rgb(SPLIT, other); assert ctx["rgbf_wrong"].shape == ctx["rgbf"].shape
    ctx["zn4"] = np.load(f"{BM.SPLIT[SPLIT]['z4']}/{t}.npy").astype(np.float32).reshape(-1, 128)
    for i, b in enumerate(boxes):
        bb = BM.inflate(b, 0.0); post = GA._prob(m4, ctx["zn4"], 512, bb, 0.3, 1.6); r = post[2]
        sp = round(float(scores[i] * np.abs(2 * r - 1).mean() * r.mean()), 8)
        det = {"i": i, "box": bb, "cache": {("prob4", 0.3, 1.6): post}}
        for a in arms:
            out[a]["masks_rle"].append(GA._rle(R.post4(ctx, det, {"kind": "post4", "c_peak": C_PEAK, **ARMS[a]})))
            out[a]["s_prime"].append(sp)
    return t, out


def boot3(Dn, Db, tids):
    """s2_final.boot plus AP75: paired tile bootstrap, sorted-id order with repeats."""
    from boxinst_commonality_tcd_04.dino_det import s2_final as F
    rng = np.random.default_rng(SEED); d = []
    for _ in range(N_BOOT):
        s = [tids[i] for i in np.sort(rng.integers(0, len(tids), len(tids)))]; a, b = F.ap_from(Dn, s), F.ap_from(Db, s)
        d.append((a[0] - b[0], a[5] - b[5], a.mean() - b.mean()))
    d = np.asarray(d); a, b = F.ap_from(Dn, tids), F.ap_from(Db, tids); r = lambda v: round(float(v), 4)
    full = (a[0] - b[0], a[5] - b[5], a.mean() - b.mean())
    return {m: {"diff": r(full[j]), "ci95": [r(np.percentile(d[:, j], 2.5)), r(np.percentile(d[:, j], 97.5))], "p_le_0": r(np.mean(d[:, j] <= 0))}
            for j, m in enumerate(("AP50", "AP75", "AP50_95"))}


def main():
    from boxinst_commonality_tcd_04 import score_coco as SC
    from boxinst_commonality_tcd_04.dino_det import fasteval as FE, s2_final as F, sam3_on_d1 as S
    os.makedirs(OUT, exist_ok=True); rec = {"date_start": time.strftime("%Y-%m-%d %H:%M"), "prereg": "dino_det/README.md GC-test"}
    Pe = json.load(open(os.path.join(EF, "preds_d1_edgefix_test_sprime_r2048.json")))["preds"]
    R3 = json.load(open(os.path.join(RES, "preds_dino_swinl_s0_test_R3_ef_r2048.json")))["preds"]
    Dp = S.det(SPLIT); tids = sorted(Pe); assert len(tids) == 439 and sorted(R3) == tids
    for t in tids:
        assert Pe[t]["boxes_2048"] == Dp[t]["boxes_2048"] == R3[t]["boxes_2048"], t
    nxt = {t: tids[(i + 1) % len(tids)] for i, t in enumerate(tids)}
    # gate: guided re-render of the first 3 tiles == published edge-fixed masks (and s')
    with mp.get_context("spawn").Pool(3) as pool:
        gate = {}
        for t, r in pool.imap_unordered(_job, [(t, nxt[t], Dp[t]["boxes_2048"], Dp[t]["scores"], ("guided",)) for t in tids[:3]]):
            g = r["guided"]; same = sum(a["counts"] == b["counts"] for a, b in zip(g["masks_rle"], Pe[t]["masks_rle"]))
            sp = float(np.abs(np.asarray(g["s_prime"], float) - np.asarray(Pe[t]["scores"], float)).max(initial=0))
            gate[t] = {"n": len(Pe[t]["masks_rle"]), "identical": int(same), "s_prime_maxabs": sp}; print("[gate]", t, gate[t], flush=True)
    ok = all(v["identical"] == v["n"] and v["s_prime_maxabs"] < 1e-6 for v in gate.values()); rec["gate"] = {"tiles": gate, "pass": ok}
    print("[gate]", "PASS" if ok else "FAIL", flush=True); assert ok
    # render controls (resume per tile)
    dirs = {a: f"{FC}/guidectl_test_{a}" for a in CONTROLS}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    todo = [t for t in tids if not all(os.path.exists(f"{dirs[a]}/{t}.json") for a in CONTROLS)]; t0 = time.time()
    print(f"[render] {len(todo)}/{len(tids)} tiles to render, arms {CONTROLS}", flush=True)
    with mp.get_context("spawn").Pool(12) as pool:
        for n, (t, r) in enumerate(pool.imap_unordered(_job, [(t, nxt[t], Dp[t]["boxes_2048"], Dp[t]["scores"], CONTROLS) for t in todo]), 1):
            for a in CONTROLS:
                json.dump(r[a], open(f"{dirs[a]}/{t}.json.tmp", "w")); os.replace(f"{dirs[a]}/{t}.json.tmp", f"{dirs[a]}/{t}.json")
            if n % 20 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    M = {"guided": {t: Pe[t]["masks_rle"] for t in tids}}
    for a in CONTROLS:
        A = {t: json.load(open(f"{dirs[a]}/{t}.json")) for t in tids}
        for t in tids:
            assert np.abs(np.asarray(A[t]["s_prime"], float) - np.asarray(Pe[t]["scores"], float)).max(initial=0) < 1e-6, (a, t)
        M[a] = {t: A[t]["masks_rle"] for t in tids}
    rec["n_masks"] = int(sum(len(M["guided"][t]) for t in tids))
    rec["n_masks_changed_vs_guided"] = {a: int(sum(x["counts"] != y["counts"] for t in tids for x, y in zip(M[a][t], M["guided"][t]))) for a in CONTROLS}
    # score_coco per arm x ranking (guided rows = the published scored files)
    rank = {"sprime": {t: Pe[t]["scores"] for t in tids}, "R3": {t: R3[t]["scores"] for t in tids}}
    pf = lambda a, k: os.path.join(OUT, f"preds_d1_gc_{a}_{k}_test_r2048.json")
    scored_fp = {("guided", "sprime"): os.path.join(EF, "scored_d1_edgefix_test_sprime_r2048.json"),
                 ("guided", "R3"): os.path.join(RES, "scored_dino_swinl_s0_test_R3_ef_r2048.json")}
    new = []
    for a in CONTROLS:
        for k in rank:
            fp = pf(a, k); scored_fp[(a, k)] = fp.replace("preds_", "scored_")
            if not os.path.exists(fp):
                json.dump({"meta": {"mask_res": 2048, "model": f"LACE (D1 variant) + edge fix, guide control '{a}' (GC-test), ranking {k}", "rank": k},
                           "preds": {t: {"boxes_2048": Pe[t]["boxes_2048"], "scores": list(map(float, rank[k][t])), "masks_rle": M[a][t]} for t in tids}},
                          open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
            if not os.path.exists(scored_fp[(a, k)]):
                new.append((a, k))
    if new:
        t0 = time.time(); cwd = os.getcwd(); os.chdir(REPO)
        sc = SC.score_many([pf(a, k) for a, k in new], S.SPLIT[SPLIT]["gt"], res=2048, max_dets=600, score_floor=0.0); os.chdir(cwd)
        for (a, k), s in zip(new, sc):
            json.dump(s, open(scored_fp[(a, k)], "w"))
        print(f"[score] score_coco {len(new)} rows {time.time()-t0:.0f}s", flush=True)
    rows = {k: {a: {m: json.load(open(scored_fp[(a, k)]))["canopy_neutral_crowd"]["mask"][m] for m in ("AP50", "AP75", "AP50_95")} for a in ARMS} for k in rank}
    print("[rows]", json.dumps(rows), flush=True); rec["rows"] = rows
    # fasteval (asserted == score_coco) + paired bootstrap guided - control
    G = json.load(open(S.SPLIT[SPLIT]["gt"])); assert sorted(G) == tids
    E = {a: FE.build({t: {"masks_rle": M[a][t]} for t in tids}, G, os.path.join(FC, f"fasteval_test_gc_{a}.pkl")) for a in ARMS}
    rec["boot"] = {}
    for k in rank:
        D = {a: F.per_tile(E[a], rank[k]) for a in ARMS}
        for a in ARMS:
            got = FE.summary(F.ap_from(D[a], tids)); assert all(abs(got[m] - rows[k][a][m]) < 1.5e-4 for m in got), (k, a, got, rows[k][a])
        t0 = time.time(); rec["boot"][k] = {f"guided_minus_{a}": boot3(D["guided"], D[a], tids) for a in CONTROLS}
        print(f"[boot {k}] {time.time()-t0:.0f}s {json.dumps(rec['boot'][k])}", flush=True)
        json.dump(rec, open(os.path.join(OUT, "guidectl_test.json"), "w"), indent=1)
    rec["date_end"] = time.strftime("%Y-%m-%d %H:%M"); rec["note"] = "run once per pre-registration GC-test; test 439 only; single detector seed"
    json.dump(rec, open(os.path.join(OUT, "guidectl_test.json"), "w"), indent=1); print("[guidectl] DONE", flush=True)


if __name__ == "__main__":
    main()
