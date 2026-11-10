"""Pre-registration IL-test (README.md, 2026-11-04, Tom: "run c and update appendix with results"). ONE look, test 439.
Interlaced vs interpolated masker features on D1: same frozen D1 test boxes/scores, lace4px_simple post4 (c_peak 0.33), EDGE_FILL on;
only the 4 px lattice zn4 the masker reads differs. Arms: real4 (deployed, 16 offsets; masks reused from results/edgefix/, gated by
byte-identical re-render of the first 3 sorted tiles), interp8 (4 offsets {0,8}^2, 8 px -> 4 px bilinear), interp16 (native 16 px grid
-> 4 px bilinear). Interpolation on the unnormalised whitened projection (affine in the DINOv3 features), then L2-norm + fp16 as zn4.
Rankings: own s' (primary) and detector s (secondary). Frozen score_coco per arm x ranking; fasteval asserted equal; paired tile
bootstrap (2000, seed 0) real4 - each interpolated arm on AP50 / AP75 / AP50:95. Masks -> feat_cache/interlace_test_<arm>/<tid>.json
(resume). -> results/interlace/interlace_test.json
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.interlace_test > boxinst_commonality_tcd_04/native_raster/logs/interlace_test_2026-11-04.log 2>&1
"""
import json, multiprocessing as mp, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); REPO = os.path.dirname(PKG)
FC = "/Users/tompitts/dphil/feat_cache"; RES = os.path.join(HERE, "results"); OUT = os.path.join(RES, "interlace"); EF = os.path.join(RES, "edgefix")
F4P = f"{FC}/feat_4p_test_dl/feat_4p_test"
SPLIT = "test"; C_PEAK = 0.33; N_BOOT, SEED = 2000, 0
ARMS = ("real4", "interp8", "interp16")
CONTROLS = ("interp8", "interp16")
STEP = {"interp8": 2, "interp16": 4}                   # 4 px cells per source cell


def _interp_matrix(n_src, step, n_dst=512):
    """(n_dst, n_src) bilinear weights: dst cell Y sits at source coordinate Y/step (source centre g -> dst cell step*g); clamped."""
    A = np.zeros((n_dst, n_src), np.float64)
    for y in range(n_dst):
        u = min(y / step, n_src - 1); i0 = int(np.floor(u)); i1 = min(i0 + 1, n_src - 1); w = u - i0
        A[y, i0] += 1 - w; A[y, i1] += w
    return A


def arm_feats(m4, t):
    """-> {arm: (512*512, 128) float32 unit-norm zn4} for the interpolated arms, plus gate/cosine stats vs real4."""
    f = np.load(f"{F4P}/{t}.npy").astype(np.float32)                          # (1024, 256, 256), 8 px, phases {0,8}^2
    v8 = ((f.reshape(1024, -1).T - m4.mu) @ m4.U / m4.scale).reshape(256, 256, 128)   # unnormalised projection
    z4 = np.load(f"{FC}/hunt_z4p_{SPLIT}/{t}.npy").astype(np.float32)                   # (512, 512, 128) real 4 px, unit
    nrm = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
    c = (nrm(v8) * z4[0::2, 0::2]).sum(-1).ravel()
    st = {"gate_cos_mean": float(c.mean()), "gate_cos_p1": float(np.percentile(c, 1))}
    out = {}
    for a, src in (("interp8", v8), ("interp16", v8[0::2, 0::2])):
        n, s = src.shape[0], STEP[a]; A = _interp_matrix(n, s).astype(np.float32)
        z = nrm(np.einsum("xj,yjc->yxc", A, np.einsum("yi,ijc->yjc", A, src))).astype(np.float16).astype(np.float32)
        lat = np.zeros((512, 512), bool); lat[0:n * s:s, 0:n * s:s] = True
        st[f"{a}_lattice_maxabs"] = float(np.abs(z[lat] - nrm(src).reshape(-1, 128)).max())
        cs = (z * z4).sum(-1)
        st[f"{a}_cos_real4_offlattice_mean"] = float(cs[~lat].mean()); st[f"{a}_cos_real4_offlattice_p1"] = float(np.percentile(cs[~lat], 1))
        out[a] = z.reshape(-1, 128)
    return out, st


def _job(args):
    """Exactly btmask.render_boxes (e = 0, one cfg) with the arm's zn4; posterior computed per arm."""
    t, boxes, scores, arms = args
    from boxinst_commonality_tcd_04.native_raster.hunt.btmask import btmask as BM
    if "m4" not in BM._W:
        BM._init(SPLIT)
    GA, R, m4 = BM._W["GA"], BM._W["R"], BM._W["m4"]
    assert GA.EDGE_FILL is True
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4); out = {a: {"masks_rle": [], "s_prime": []} for a in arms}
    Z, st = ({}, {}) if arms == ("real4",) else arm_feats(m4, t)
    if "real4" in arms:
        Z["real4"] = np.load(f"{BM.SPLIT[SPLIT]['z4']}/{t}.npy").astype(np.float32).reshape(-1, 128)
    rgbf = BM._rgb(SPLIT, t) if len(boxes) else None
    for a in arms:
        ctx = {"rgbf": rgbf, "GA": GA, "m4": m4, "g4": 512, "zn4": Z[a]}
        for i, b in enumerate(boxes):
            bb = BM.inflate(b, 0.0); post = GA._prob(m4, ctx["zn4"], 512, bb, 0.3, 1.6); r = post[2]
            sp = round(float(scores[i] * np.abs(2 * r - 1).mean() * r.mean()), 8)
            det = {"i": i, "box": bb, "cache": {("prob4", 0.3, 1.6): post}}
            out[a]["masks_rle"].append(GA._rle(R.post4(ctx, det, {"kind": "post4", "c_peak": C_PEAK})))
            out[a]["s_prime"].append(sp)
    return t, out, st


def boot3(Dn, Db, tids):
    """s2_final.boot plus AP75: paired tile bootstrap, sorted-id order with repeats (= guide_controls_test.boot3)."""
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
    os.makedirs(OUT, exist_ok=True); rec = {"date_start": time.strftime("%Y-%m-%d %H:%M"), "prereg": "dino_det/README.md IL-test"}
    jf = os.path.join(OUT, "interlace_test.json")
    Pe = json.load(open(os.path.join(EF, "preds_d1_edgefix_test_sprime_r2048.json")))["preds"]
    Dp = S.det(SPLIT); tids = sorted(Pe); assert len(tids) == 439
    for t in tids:
        assert Pe[t]["boxes_2048"] == Dp[t]["boxes_2048"], t
    # gate: real4 re-render of the first 3 tiles == published edge-fixed masks (and s')
    with mp.get_context("spawn").Pool(3) as pool:
        gate = {}
        for t, r, _ in pool.imap_unordered(_job, [(t, Dp[t]["boxes_2048"], Dp[t]["scores"], ("real4",)) for t in tids[:3]]):
            g = r["real4"]; same = sum(a["counts"] == b["counts"] for a, b in zip(g["masks_rle"], Pe[t]["masks_rle"]))
            sp = float(np.abs(np.asarray(g["s_prime"], float) - np.asarray(Pe[t]["scores"], float)).max(initial=0))
            gate[t] = {"n": len(Pe[t]["masks_rle"]), "identical": int(same), "s_prime_maxabs": sp}; print("[gate]", t, gate[t], flush=True)
    ok = all(v["identical"] == v["n"] and v["s_prime_maxabs"] < 1e-6 for v in gate.values()); rec["gate_real4"] = {"tiles": gate, "pass": ok}
    print("[gate real4]", "PASS" if ok else "FAIL", flush=True); assert ok
    # render interpolated arms (resume per tile); per-tile feature gates + cosines -> _stats.jsonl
    dirs = {a: f"{FC}/interlace_test_{a}" for a in CONTROLS}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    stf = f"{FC}/interlace_test_interp16/_stats.jsonl"
    todo = [t for t in tids if not all(os.path.exists(f"{dirs[a]}/{t}.json") for a in CONTROLS)]; t0 = time.time()
    print(f"[render] {len(todo)}/{len(tids)} tiles to render, arms {CONTROLS}", flush=True)
    with mp.get_context("spawn").Pool(10) as pool:
        for n, (t, r, st) in enumerate(pool.imap_unordered(_job, [(t, Dp[t]["boxes_2048"], Dp[t]["scores"], CONTROLS) for t in todo]), 1):
            assert st["gate_cos_mean"] >= 0.999 and st["gate_cos_p1"] >= 0.99, (t, st)
            assert st["interp8_lattice_maxabs"] < 1e-3 and st["interp16_lattice_maxabs"] < 1e-3, (t, st)
            open(stf, "a").write(json.dumps({"tile": t, **st}) + "\n")
            for a in CONTROLS:
                json.dump(r[a], open(f"{dirs[a]}/{t}.json.tmp", "w")); os.replace(f"{dirs[a]}/{t}.json.tmp", f"{dirs[a]}/{t}.json")
            if n % 20 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} {time.time()-t0:.0f}s  last {t} {json.dumps({k: round(v, 4) for k, v in st.items()})}", flush=True)
    S_ = {}
    for l in open(stf):
        d = json.loads(l); S_[d.pop("tile")] = d
    assert sorted(S_) == tids
    rec["feat_stats_mean_over_tiles"] = {k: round(float(np.mean([S_[t][k] for t in tids])), 5) for k in S_[tids[0]]}
    rec["feat_stats_min_over_tiles"] = {k: round(float(np.min([S_[t][k] for t in tids])), 5) for k in S_[tids[0]]}
    print("[feat]", json.dumps(rec["feat_stats_mean_over_tiles"]), flush=True)
    M = {"real4": {t: Pe[t]["masks_rle"] for t in tids}}; SP = {"real4": {t: Pe[t]["scores"] for t in tids}}
    for a in CONTROLS:
        A = {t: json.load(open(f"{dirs[a]}/{t}.json")) for t in tids}
        M[a] = {t: A[t]["masks_rle"] for t in tids}; SP[a] = {t: A[t]["s_prime"] for t in tids}
    rec["n_masks"] = int(sum(len(M["real4"][t]) for t in tids))
    rec["n_masks_changed_vs_real4"] = {a: int(sum(x["counts"] != y["counts"] for t in tids for x, y in zip(M[a][t], M["real4"][t]))) for a in CONTROLS}
    # score_coco per arm x ranking (real4 s' = the published edge-fix scored file)
    rank = {"sprime": SP, "dets": {a: {t: Dp[t]["scores"] for t in tids} for a in ARMS}}
    pf = lambda a, k: os.path.join(OUT, f"preds_d1_il_{a}_{k}_test_r2048.json")
    scored_fp = {("real4", "sprime"): os.path.join(EF, "scored_d1_edgefix_test_sprime_r2048.json")}
    new = []
    for a in ARMS:
        for k in rank:
            if (a, k) in scored_fp:
                continue
            fp = pf(a, k); scored_fp[(a, k)] = fp.replace("preds_", "scored_")
            if not os.path.exists(fp):
                json.dump({"meta": {"mask_res": 2048, "model": f"LACE (D1 variant) + edge fix, feature arm '{a}' (IL-test), ranking {k}", "rank": k},
                           "preds": {t: {"boxes_2048": Pe[t]["boxes_2048"], "scores": list(map(float, rank[k][a][t])), "masks_rle": M[a][t]} for t in tids}},
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
    json.dump(rec, open(jf, "w"), indent=1)
    # fasteval (asserted == score_coco) + paired bootstrap real4 - control
    G = json.load(open(S.SPLIT[SPLIT]["gt"])); assert sorted(G) == tids
    E = {a: FE.build({t: {"masks_rle": M[a][t]} for t in tids}, G, os.path.join(FC, f"fasteval_test_il_{a}.pkl")) for a in ARMS}
    rec["boot"] = {}
    for k in rank:
        D = {a: F.per_tile(E[a], rank[k][a]) for a in ARMS}
        for a in ARMS:
            got = FE.summary(F.ap_from(D[a], tids)); assert all(abs(got[m] - rows[k][a][m]) < 1.5e-4 for m in got), (k, a, got, rows[k][a])
        t0 = time.time(); rec["boot"][k] = {f"real4_minus_{a}": boot3(D["real4"], D[a], tids) for a in CONTROLS}
        print(f"[boot {k}] {time.time()-t0:.0f}s {json.dumps(rec['boot'][k])}", flush=True)
        json.dump(rec, open(jf, "w"), indent=1)
    rec["date_end"] = time.strftime("%Y-%m-%d %H:%M"); rec["note"] = "run once per pre-registration IL-test; test 439 only; single detector seed"
    json.dump(rec, open(jf, "w"), indent=1); print("[interlace] DONE", flush=True)


if __name__ == "__main__":
    main()
