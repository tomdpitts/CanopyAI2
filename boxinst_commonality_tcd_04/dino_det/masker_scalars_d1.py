"""Pre-registration ABL-D1 item 3 (README.md, 2026-11-10): tab:masker inference scalars on LACE (D1 + edge fix), ONE look.
val 108 (own s' only): alpha in {0, 0.3, 1.0} x kappa in {10, 16} -> directional claims of the footnote.
test 439: (alpha, kappa) = (0, 16), (1.0, 10), (0.3, 16) deployed (reused, 3-tile byte gate); rankings ŝ (primary, frozen R3
heads re-applied to each arm's s') and own s'; frozen score_coco; paired tile bootstrap deployed - arm (2000, seed 0).
Masks feat_cache/abl_masker_<split>_a<alpha>_k<ks>/ (resume). -> results/abl_d1/masker_scalars.json
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.masker_scalars_d1
"""
import json, multiprocessing as mp, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

from boxinst_commonality_tcd_04.dino_det import abl_lib as L

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); REPO = os.path.dirname(PKG)
RES = os.path.join(HERE, "results"); OUT = os.path.join(RES, "abl_d1"); FC = L.FC; N_BOOT, SEED = 2000, 0
VAL_GRID = [(a, k) for a in (0.0, 0.3, 1.0) for k in (1.0, 1.6)]           # kappa = 10 * k
TEST_ARMS = [(0.0, 1.6), (1.0, 1.0)]; DEP = (0.3, 1.6)


def tag(split, a, k):
    return f"abl_masker_{split}_a{a:g}_k{k:g}"


def _job(args):
    split, t, boxes, scores, a, k = args
    M, SP = L.render(split, t, boxes, scores, alpha=a, ks=k)
    return t, {"masks_rle": M, "s_prime": SP}


def render_all(split, Dp, a, k, tids, pool):
    d = f"{FC}/{tag(split, a, k)}"; os.makedirs(d, exist_ok=True); t0 = time.time()
    todo = [t for t in tids if not os.path.exists(f"{d}/{t}.json")]
    print(f"[masker] render {split} a={a:g} k={k:g}: {len(todo)}/{len(tids)}", flush=True)
    for n, (t, r) in enumerate(pool.imap_unordered(_job, [(split, t, Dp[t]["boxes_2048"], Dp[t]["scores"], a, k) for t in todo]), 1):
        json.dump(r, open(f"{d}/{t}.json.tmp", "w")); os.replace(f"{d}/{t}.json.tmp", f"{d}/{t}.json")
        if n % 40 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    return {t: json.load(open(f"{d}/{t}.json")) for t in tids}


def score(P, gtp, name):
    from boxinst_commonality_tcd_04 import score_coco as SC
    fp = os.path.join(OUT, f"preds_{name}_r2048.json"); sp = fp.replace("preds_", "scored_")
    if not os.path.exists(sp):
        json.dump({"meta": {"mask_res": 2048, "model": f"LACE (D1) + edge fix, ABL-D1 tab:masker {name}"}, "preds": P}, open(fp + ".tmp", "w"))
        os.replace(fp + ".tmp", fp)
        cwd = os.getcwd(); os.chdir(REPO); json.dump(SC.score_many([fp], gtp, res=2048, max_dets=600, score_floor=0.0)[0], open(sp, "w")); os.chdir(cwd)
    c = json.load(open(sp))["canopy_neutral_crowd"]
    return {"mask": {m: c["mask"][m] for m in ("AP50", "AP75", "AP50_95")}, "box": {m: c["box"][m] for m in ("AP50",)}}, fp


def boot(fps, gtp, base):
    """Paired tile bootstrap base - each other key, AP50 and AP50:95 (edgefix_test pattern)."""
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    from boxinst_commonality_tcd_04.dino_det import sam3_on_d1 as S
    G = json.load(open(gtp)); gd, t2i = SL.build_gt_crowd(G, 2048); ids = np.array([t2i[t] for t in sorted(G)]); D = {}
    for k, f in fps.items():
        P = json.load(open(f))["preds"]; D[k] = S.per_tile({t: P[t] for t in sorted(G)}, gd, t2i, 0.0)
        for i in ids:
            if i not in D[k]:
                ng = sum(1 for a in gd["annotations"] if a["image_id"] == i and not a["iscrowd"])
                D[k][i] = (np.zeros(0), np.zeros((10, 0), bool), np.zeros((10, 0), bool), ng)
    full = {k: [S.ap_at(ids, D[k], SL.REC_THRS, j) for j in range(10)] for k in D}
    rng = np.random.default_rng(SEED); dd = {k: [] for k in D if k != base}
    for b in range(N_BOOT):
        s = ids[rng.integers(0, len(ids), len(ids))]; A = {k: [S.ap_at(s, D[k], SL.REC_THRS, j) for j in range(10)] for k in D}
        for k in dd:
            dd[k].append((A[base][0] - A[k][0], np.mean(A[base]) - np.mean(A[k])))
    out = {}
    for k, v in dd.items():
        v = np.asarray(v)
        out[f"{base}_minus_{k}"] = {m: {"diff": round(float(full[base][0] - full[k][0]) if j == 0 else float(np.mean(full[base]) - np.mean(full[k])), 4),
                                        "ci95": [round(float(np.percentile(v[:, j], 2.5)), 4), round(float(np.percentile(v[:, j], 97.5)), 4)]}
                                    for j, m in enumerate(("AP50", "AP50_95"))}
    return out


def main():
    from boxinst_commonality_tcd_04.native_raster.hunt.btmask import btmask as BM
    os.makedirs(OUT, exist_ok=True); fout = os.path.join(OUT, "masker_scalars.json")
    rec = json.load(open(fout)) if os.path.exists(fout) else {"date_start": time.strftime("%Y-%m-%d %H:%M")}
    with mp.get_context("spawn").Pool(12) as pool:
        # ---------------- val grid, own s'
        if "val" not in rec:
            gtp = BM.SPLIT["val"]["gt"]; Dv = json.load(open(f"{FC}/dino_swinl_s0_val_mask.json"))  # edgefix_val.BASE: boxes + detector s
            tids = sorted(json.load(open(gtp))); assert sorted(Dv) == tids, "val tids"
            ef = f"{FC}/edgefix_val"                                   # gate: deployed val render == cached edge-fixed val masks
            for t in tids[:3]:
                M, _ = L.render("val", t, Dv[t]["boxes_2048"], Dv[t]["scores"])
                assert [m["counts"] for m in M] == [m["counts"] for m in json.load(open(f"{ef}/{t}.json"))["masks_rle"]], f"val gate {t}"
            print("[masker] val deployed-render gate PASS (3 tiles byte-identical)", flush=True)
            rows = {}
            for a, k in VAL_GRID:
                F = render_all("val", Dv, a, k, tids, pool)
                P = {t: {"boxes_2048": Dv[t]["boxes_2048"], "scores": F[t]["s_prime"], "masks_rle": F[t]["masks_rle"]} for t in tids}
                rows[f"a{a:g}_k{10*k:g}"], _ = score(P, gtp, f"val_a{a:g}_k{k:g}_sprime")
                print(f"[masker] val a={a:g} kappa={10*k:g}: {rows[f'a{a:g}_k{10*k:g}']['mask']}", flush=True)
            ap = lambda a, kk: rows[f"a{a:g}_k{kk:g}"]["mask"]["AP50"]
            claims = {"alpha_lt1_beats_alpha1_every_kappa": all(min(ap(a, kk) for a in (0.0, 0.3)) > ap(1.0, kk) for kk in (10, 16)),
                      "kappa16_beats_kappa10_every_alpha": all(ap(a, 16) > ap(a, 10) for a in (0.0, 0.3, 1.0)),
                      "note": "alpha<1 claim read as: every alpha<1 cell beats alpha=1 at the same kappa"}
            rec["val"] = {"rows": rows, "claims": claims}
            json.dump(rec, open(fout, "w"), indent=1)
        # ---------------- test, 3 arms x {ŝ, s'}
        if "test" not in rec:
            gtp = BM.SPLIT["test"]["gt"]; tids = sorted(json.load(open(gtp)))
            from boxinst_commonality_tcd_04.dino_det import sam3_on_d1 as S
            Dp = S.det("test"); Ef = json.load(open(os.path.join(RES, "edgefix", "preds_d1_edgefix_test_sprime_r2048.json")))["preds"]
            Pub = json.load(open(os.path.join(RES, "preds_dino_swinl_s0_test_R3_ef_r2048.json")))["preds"]
            for t in tids[:3]:
                M, SP = L.render("test", t, Dp[t]["boxes_2048"], Dp[t]["scores"])
                assert [m["counts"] for m in M] == [m["counts"] for m in Ef[t]["masks_rle"]], f"test gate {t}"
            print("[masker] test deployed-render gate PASS (3 tiles byte-identical)", flush=True)
            r3 = L.r3_heads()
            dev = max(np.abs(r3(Dp[t]["boxes_2048"], Dp[t]["scores"], Ef[t]["scores"]) - np.asarray(Pub[t]["scores"])).max(initial=0) for t in tids)
            assert dev < 1e-6, f"R3 gate {dev}"; print(f"[masker] R3 gate PASS max |d| {dev:.1e}", flush=True)
            arms = {f"a{DEP[0]:g}_k{10*DEP[1]:g}": {t: {"masks_rle": Ef[t]["masks_rle"], "s_prime": Ef[t]["scores"]} for t in tids}}
            for a, k in TEST_ARMS:
                arms[f"a{a:g}_k{10*k:g}"] = render_all("test", Dp, a, k, tids, pool)
            rows, fps = {}, {"shat": {}, "sprime": {}}
            for name, F in arms.items():
                for rk in ("shat", "sprime"):
                    sc = {t: (r3(Dp[t]["boxes_2048"], Dp[t]["scores"], F[t]["s_prime"]).tolist() if rk == "shat" else F[t]["s_prime"]) for t in tids}
                    P = {t: {"boxes_2048": Dp[t]["boxes_2048"], "scores": sc[t], "masks_rle": F[t]["masks_rle"]} for t in tids}
                    rows[f"{name}_{rk}"], fps[rk][name] = score(P, gtp, f"test_{name}_{rk}")
                    print(f"[masker] test {name} {rk}: {rows[f'{name}_{rk}']['mask']}", flush=True)
            dep = f"a{DEP[0]:g}_k{10*DEP[1]:g}"
            assert abs(rows[f"{dep}_shat"]["mask"]["AP50"] - 0.7141) < 5e-5, "deployed ŝ != Table 1"
            assert abs(rows[f"{dep}_sprime"]["mask"]["AP50"] - 0.7044) < 5e-5, "deployed s' != edge-fix row"
            rec["test"] = {"rows": rows, "boot": {rk: boot(fps[rk], gtp, dep) for rk in ("shat", "sprime")}}
            rec["date_end"] = time.strftime("%Y-%m-%d %H:%M"); rec["note"] = "ONE look per pre-registration ABL-D1 item 3; single detector run"
            json.dump(rec, open(fout, "w"), indent=1)
    print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()
