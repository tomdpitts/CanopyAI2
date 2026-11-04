"""Pre-registration R3 (README.md, 2026-11-04): streamline R2 by ablation on val, local compute only; one frozen config to test + sparse.

Every config = 15 OOF runs on val (5 seeds x fold sets rr/perm1/perm2, tile-grouped 5-fold), scored by fasteval (parity-checked
against score_lite). gain = run mask AP50 - s' (0.7384). Every result is appended to results/rerankR3_ledger.jsonl, which is
also the resume cache (a config already in the ledger is never re-run).

  stage1 : machinery screen on R2's 14 features
  stage2 : backward feature elimination, 14 -> 1 features, for every model in MODELS
  select : pre-registered rule -> results/rerankR3_select.json
  confirm: fresh randomness (seeds 5-9 x perm3/4/5) for the selected config vs R2; walk the candidate list until one passes
  final  : ONCE. 25-model mean on the test 439 and sparse-236, frozen score_coco, bootstrap vs s' and vs R2
  side   : AP50:95 checks (val only): box-IoU targets, s'/s exponent, mask dilation/erosion
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.rerank3 all
"""
import argparse, json, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

from boxinst_commonality_tcd_04.dino_det import rerank as R
from boxinst_commonality_tcd_04.dino_det import fasteval as FE

LEDGER = os.path.join(R.OUT, "rerankR3_ledger.jsonl"); BASE = 0.7384; TOL = 0.001; NPROC = 14
R2COLS = [R.NAMES[c] for c in R.SCORE + R.GEOM + R.CTX]
MODELS = {"logreg": [], "mlp1x8": [8], "mlp1x16": [16], "mlp1x64": [64], "mlp2x64": [64, 64]}  # simplest first
DEF = {"model": "mlp2x64", "epochs": 30, "dropout": 0.1, "wd": 1e-3, "label": "box50"}
RUNS = [(f, s) for f in ("rr", "perm1", "perm2") for s in range(5)]
FRESH = [(f, s) for f in ("perm3", "perm4", "perm5") for s in range(5, 10)]


def fold_sets(n):
    out = {"rr": [i % R.K for i in range(n)]}
    for s in range(1, 6):
        p = np.random.default_rng(s).permutation(n); f = np.zeros(n, int); f[p] = np.arange(n) % R.K; out[f"perm{s}"] = f.tolist()
    return out


def soft_targets(P, vg, tids, y):
    """Box IoU of each det with its greedy match (R1 labels: s' order, box IoU >= 0.5); 0 if unmatched; asserts the matched set == y==1."""
    out = []
    for t in tids:
        b = np.asarray(P[t]["boxes_2048"], np.float64).reshape(-1, 4); sp = np.asarray(P[t]["s_prime"]); v = np.zeros(len(b))
        gb = np.asarray([[min(p[0::2]), min(p[1::2]), max(p[0::2]), max(p[1::2])] for p in vg[t]["trees"] if len(p) >= 6],
                        np.float64).reshape(-1, 4)
        if len(gb) and len(b):
            bi = R._box_iou(b, gb); taken = np.zeros(len(gb), bool)
            for i in np.argsort(-sp, kind="mergesort"):
                cc = np.where(~taken & (bi[i] >= 0.5), bi[i], -1.0); j = int(cc.argmax())
                if cc[j] >= 0.5:
                    taken[j] = True; v[i] = cc[j]
        out.append(v)
    v = np.concatenate(out); assert ((v > 0) == (y == 1)).all(); return v


# ------------------------------------------------------------------ worker
_W = {}


def _init():
    import json as _j
    Z = np.load(os.path.join(R.FC_, "rerankR1_val.npz")); P = _j.load(open(os.path.join(R.FC_, f"{R.TAG}_val_mask.json")))
    vg = _j.load(open(R.VAL_GT)); tids = list(Z["tids"])
    _W.update(X=Z["X"], y=Z["y"], tile=Z["tile"], tids=tids, E=FE.build(None, None, os.path.join(R.FC_, "fasteval_val.pkl")),
              F=fold_sets(len(tids)), soft=soft_targets(P, vg, tids, Z["y"]))
    assert list(_W["E"]["tids"]) == tids


def fit(X, y, cfg, seed):
    import torch
    torch.set_num_threads(1); nn = torch.nn
    torch.manual_seed(seed); mu, sd = X.mean(0), X.std(0) + 1e-6; layers, d = [], X.shape[1]
    for h in MODELS[cfg["model"]]:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(cfg["dropout"])]; d = h
    net = nn.Sequential(*layers, nn.Linear(d, 1)) if layers else nn.Linear(d, 1)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=cfg["wd"])
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed); lossf = nn.BCEWithLogitsLoss()
    for _ in range(cfg["epochs"]):
        net.train()
        for idx in torch.randperm(len(Xt), generator=g).split(1024):
            opt.zero_grad(); lossf(net(Xt[idx]).squeeze(-1), yt[idx]).backward(); opt.step()
    net.eval()

    def pred(Z):
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor((Z - mu) / sd, dtype=torch.float32)).squeeze(-1)).numpy().astype(np.float64)
    return pred


def targets(cfg, y, soft):
    lab = cfg["label"]
    if lab == "box50":
        return y >= 0, y.astype(float)
    if lab == "crowdneg":
        return np.ones(len(y), bool), (y == 1).astype(float)
    if lab == "soft":
        return y >= 0, soft
    if lab == "multi":
        return y >= 0, np.mean([soft >= t for t in FE.THR], 0)
    raise ValueError(lab)


def oof_fits(cfg, seed, fname, X, y, soft, tile, F):
    cols = [R.NAMES.index(c) for c in cfg["cols"]]; fold = np.asarray(F[fname])[tile]; m, tgt = targets(cfg, y, soft)
    out = np.zeros(len(X)); fits = []
    for k in range(R.K):
        tr = (fold != k) & m; f = fit(X[tr][:, cols], tgt[tr], cfg, seed); te = fold == k; out[te] = f(X[te][:, cols]); fits.append(f)
    return out, fits


def _job(a):
    cfg, fname, seed = a; W = _W
    v, _ = oof_fits(cfg, seed, fname, W["X"], W["y"], W["soft"], W["tile"], W["F"])
    return FE.ap(W["E"], v).tolist()


# ------------------------------------------------------------------ ledger
def key(cfg, runs):
    return json.dumps({**cfg, "cols": list(cfg["cols"]), "runs": [list(r) for r in runs]}, sort_keys=True)


def ledger():
    L = {}
    if os.path.exists(LEDGER):
        for ln in open(LEDGER):
            r = json.loads(ln); L[r["key"]] = r
    return L


def evaluate(pool, cfgs, runs=RUNS, tag=""):
    L = ledger(); todo = [c for c in cfgs if key(c, runs) not in L]
    if todo:
        t0 = time.time(); res = pool.map(_job, [(c, f, s) for c in todo for f, s in runs], chunksize=1)
        with open(LEDGER, "a") as fh:
            for i, c in enumerate(todo):
                a = np.asarray(res[i * len(runs):(i + 1) * len(runs)]); g = a[:, 0] - BASE
                r = {"key": key(c, runs), "cfg": {**c, "cols": list(c["cols"])}, "runs": [list(x) for x in runs], "tag": tag,
                     "ap": a.round(6).tolist(), "gain_mean": round(float(g.mean()), 5), "gain_sd": round(float(g.std(ddof=1)), 5),
                     "n_pos": int((g > 0).sum()), "ap5095_mean": round(float(a.mean(1).mean()), 5),
                     "ap75_mean": round(float(a[:, 5].mean()), 5), "date": time.strftime("%Y-%m-%d %H:%M")}
                fh.write(json.dumps(r) + "\n"); L[r["key"]] = r
        print(f"  [{tag}] {len(todo)} configs x {len(runs)} runs {time.time()-t0:.0f}s", flush=True)
    return [L[key(c, runs)] for c in cfgs]


def show(r, ref=None):
    c = r["cfg"]; d = "" if ref is None else f" vsR2 {r['gain_mean']-ref['gain_mean']:+.4f}"
    return (f"{c['model']:8s} ep{c['epochs']} do{c['dropout']} wd{c['wd']} {c['label']:8s} n={len(c['cols']):2d} "
            f"gain {r['gain_mean']:+.4f}±{r['gain_sd']:.4f} pos {r['n_pos']:2d}/{len(r['ap'])} AP50:95 {r['ap5095_mean']:.4f}{d}")


def cfg(**kw):
    return {**DEF, "cols": tuple(R2COLS), **kw}


# ------------------------------------------------------------------ stages
def stage1(pool):
    print("[stage1] machinery screen, 14 features", flush=True)
    C = [cfg()] + [cfg(model=m) for m in MODELS if m != "mlp2x64"] + [
        cfg(dropout=0.0), cfg(epochs=10), cfg(epochs=60), cfg(wd=1e-2), cfg(wd=0.0), cfg(label="crowdneg")]
    rs = evaluate(pool, C, tag="stage1"); ref = rs[0]
    old = json.load(open(os.path.join(R.OUT, "rerankR2_cv.json")))
    assert [round(x, 4) for x in np.asarray(ref["ap"])[:, 0]] == [r["AP50"] for r in old["runs"]], "R2 reference not reproduced"
    print("  R2 reference reproduced run-for-run", flush=True)
    for r in rs:
        print("  " + show(r, ref), flush=True)
    return ref


def stage2(pool):
    ref = evaluate(pool, [cfg()])[0]; path = {}
    for m in MODELS:
        cols = list(R2COLS); steps = [{"cols": list(cols), **{k: v for k, v in evaluate(pool, [cfg(model=m)])[0].items() if k in ("gain_mean", "n_pos", "ap5095_mean")}}]
        print(f"[stage2] {m}: start {show(evaluate(pool, [cfg(model=m)])[0], ref)}", flush=True)
        while len(cols) > 1:
            C = [cfg(model=m, cols=tuple(c for c in cols if c != x)) for x in cols]
            rs = evaluate(pool, C, tag=f"stage2:{m}:{len(cols)-1}")
            i = int(np.argmax([r["gain_mean"] for r in rs])); drop = cols[i]; cols = [c for c in cols if c != drop]
            steps.append({"dropped": drop, "cols": list(cols), "gain_mean": rs[i]["gain_mean"], "n_pos": rs[i]["n_pos"],
                          "ap5095_mean": rs[i]["ap5095_mean"]})
            print(f"  {m} -{drop:15s} -> {show(rs[i], ref)}", flush=True)
        path[m] = steps
    json.dump(path, open(os.path.join(R.OUT, "rerankR3_paths.json"), "w"), indent=1)
    return path


def candidates():
    """Pre-registered rule: admissible = mean gain >= R2 mean - TOL, mean gain >= 0.005, n_pos >= 13/15, box50 label.
    Order: fewest features, then simplest model (MODELS order), then default machinery first, then higher mean."""
    L = [r for r in ledger().values() if r["runs"] == [list(x) for x in RUNS]]
    ref = next(r for r in L if r["key"] == key(cfg(), RUNS)); mo = list(MODELS)
    adm = [r for r in L if r["cfg"]["label"] == "box50" and r["gain_mean"] >= ref["gain_mean"] - TOL and r["gain_mean"] >= 0.005
           and r["n_pos"] >= 13]
    isdef = lambda c: all(c[k] == DEF[k] for k in ("epochs", "dropout", "wd"))
    adm.sort(key=lambda r: (len(r["cfg"]["cols"]), mo.index(r["cfg"]["model"]), not isdef(r["cfg"]), -r["gain_mean"]))
    return ref, adm


def select():
    ref, adm = candidates()
    rec = {"R2_ref": {k: ref[k] for k in ("gain_mean", "gain_sd", "n_pos", "ap5095_mean")}, "tol": TOL, "n_admissible": len(adm),
           "ranked": [{"cfg": r["cfg"], "gain_mean": r["gain_mean"], "gain_sd": r["gain_sd"], "n_pos": r["n_pos"],
                       "ap5095_mean": r["ap5095_mean"]} for r in adm[:40]], "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(rec, open(os.path.join(R.OUT, "rerankR3_select.json"), "w"), indent=1)
    print(f"[select] {len(adm)} admissible; top 10:", flush=True)
    for r in adm[:10]:
        print("  " + show(r, ref) + f"  {r['cfg']['cols']}", flush=True)
    return ref, adm


def confirm(pool):
    ref, adm = candidates(); out = os.path.join(R.OUT, "rerankR3_confirm.json")
    rf = evaluate(pool, [cfg()], FRESH, "confirm:R2")[0]; tried = []
    for r in adm:
        c = {**r["cfg"], "cols": tuple(r["cfg"]["cols"])}; f = evaluate(pool, [c], FRESH, "confirm")[0]
        ok = bool(f["gain_mean"] >= rf["gain_mean"] - TOL and f["gain_mean"] >= 0.005 and f["n_pos"] >= 13)
        tried.append({"cfg": f["cfg"], "fresh_gain_mean": f["gain_mean"], "fresh_n_pos": f["n_pos"], "fresh_ap5095": f["ap5095_mean"],
                      "pass": ok})
        print(f"  [confirm] {show(f, rf)} {'PASS' if ok else 'fail'}", flush=True)
        if ok:
            break
    else:
        raise SystemExit("no candidate confirmed")
    rec = {"R2_fresh": {k: rf[k] for k in ("gain_mean", "gain_sd", "n_pos", "ap5095_mean")}, "tried": tried, "chosen": tried[-1]["cfg"],
           "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(rec, open(out, "w"), indent=1); return rec["chosen"]


def final(chosen):
    """ONCE. Frozen chosen config: 25 models (5 seeds x rr folds) trained on val, mean prediction -> test 439 and sparse-236."""
    from boxinst_commonality_tcd_04 import score_coco as SC
    out = os.path.join(R.OUT, "rerankR3_final.json"); assert not os.path.exists(out), out
    c = {**chosen, "cols": tuple(chosen["cols"])}; _init(); W = _W; vg = json.load(open(R.VAL_GT))
    fits = [f for s in range(5) for f in oof_fits(c, s, "rr", W["X"], W["y"], W["soft"], W["tile"], W["F"])[1]]; assert len(fits) == 25
    cols = [R.NAMES.index(x) for x in c["cols"]]
    # val bootstrap of the rr seed-0 OOF scores vs s' (the R2 gate's third criterion)
    Pv = json.load(open(os.path.join(R.FC_, f"{R.TAG}_val_mask.json"))); v0, _ = oof_fits(c, 0, "rr", W["X"], W["y"], W["soft"], W["tile"], W["F"])
    sc0 = R.split_scores(W["tids"], W["tile"], v0)
    vb = R.boot({t: {**Pv[t], "scores": sc0[t]} for t in W["tids"]}, {t: {**Pv[t], "scores": Pv[t]["s_prime"]} for t in W["tids"]}, vg, 0.0, 0.0)
    print(f"[final] val boot rr s0 vs s': {vb}", flush=True)
    rec = {"chosen": chosen, "val_boot_rr_s0_vs_sprime": vb}
    for split, gtp in (("test", R.TEST_GT), ("sparse", os.path.join(R.REPO, "data", "tcd_sparse", "sparse_gt.json"))):
        G = json.load(open(gtp)); assert not set(vg) & set(G)
        Z = np.load(os.path.join(R.FC_, f"rerankR1_{split}.npz")); tids = list(Z["tids"]); X = Z["X"]; tile = Z["tile"]
        assert not np.allclose(X[:, 0], X[:, 1]), "s == s' (the R2 test bug)"
        St = json.load(open(os.path.join(R.OUT, f"preds_{R.TAG}_{split}_sprime_r2048.json")))["preds"]; assert sorted(St) == tids == sorted(G)
        assert np.allclose(np.concatenate([St[t]["scores"] for t in tids]), X[:, 1])
        v = np.mean([f(X[:, cols]) for f in fits], 0); sc = R.split_scores(tids, tile, v)
        preds = {t: {"boxes_2048": St[t]["boxes_2048"], "scores": sc[t], "masks_rle": St[t]["masks_rle"]} for t in tids}
        fp = os.path.join(R.OUT, f"preds_{R.TAG}_{split}_rerankR3_r2048.json")
        json.dump({"meta": {"mask_res": 2048, "model": f"D1 + LACE masker, R3 streamlined reranker {c['model']} on {list(c['cols'])} (POST-HOC; val-trained, 25-model mean)"},
                   "preds": preds}, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
        cwd = os.getcwd(); os.chdir(R.REPO); s = SC.score_many([fp], gtp, res=2048, max_dets=600, score_floor=0.0)[0]; os.chdir(cwd)
        json.dump(s, open(fp.replace("preds_", "scored_"), "w")); cr = s["canopy_neutral_crowd"]
        ref = {k: json.load(open(os.path.join(R.OUT, f"preds_{R.TAG}_{split}_{k}_r2048.json")))["preds"] for k in ("sprime", "rerankR2")}
        bt = {f"vs_{k}": R.boot(preds, P, G, 0.0, 0.0) for k, P in ref.items()}
        rec[split] = {"mask": {k: cr["mask"][k] for k in ("AP50", "AP75", "AP50_95")}, "box": {k: cr["box"][k] for k in ("AP50", "AP75")}, "boot": bt}
        if split == "sparse":
            keep = [t for t in tids if t not in set(json.load(open(R.TEST_GT)))]
            rec[split]["boot_vs_sprime_220_not_in_439"] = {**R.boot({t: preds[t] for t in keep}, {t: ref["sprime"][t] for t in keep},
                                                                    {t: G[t] for t in keep}, 0.0, 0.0), "n": len(keep)}
        print(f"[final] {split}: {json.dumps(rec[split])}", flush=True)
    rec["confirmed"] = bool(rec["sparse"]["boot"]["vs_sprime"]["ci95"][0] > 0); rec["date"] = time.strftime("%Y-%m-%d %H:%M")
    json.dump(rec, open(out, "w"), indent=1); print(json.dumps(rec, indent=1), flush=True)


# ------------------------------------------------------------------ AP50:95 side (val only)
def side(pool, chosen):
    out = os.path.join(R.OUT, "rerankR3_side_ap5095.json"); rec = json.load(open(out)) if os.path.exists(out) else {}
    E = FE.build(None, None, os.path.join(R.FC_, "fasteval_val.pkl")); Z = np.load(os.path.join(R.FC_, "rerankR1_val.npz"))
    s, sp = Z["X"][:, 0], Z["X"][:, 1]
    if "exponent" not in rec:   # score = s * (s'/s)^a: a=0 is detector s, a=1 is s'
        rec["exponent"] = {f"{a:.2f}": FE.summary(FE.ap(E, s * (sp / np.maximum(s, 1e-9)) ** a)) for a in np.arange(0, 1.76, 0.25)}
        print("[side] exponent", json.dumps(rec["exponent"]), flush=True)
    c = {**chosen, "cols": tuple(chosen["cols"])}
    C = [cfg(), cfg(label="soft"), cfg(label="multi"), c, {**c, "label": "soft"}, {**c, "label": "multi"}]
    rs = evaluate(pool, C, tag="side:labels")
    rec["labels"] = [{"cfg": r["cfg"], "gain_mean_ap50": r["gain_mean"], "ap5095_mean": r["ap5095_mean"], "ap75_mean": r["ap75_mean"]} for r in rs]
    for r in rs:
        print("  [side] " + show(r), flush=True)
    rec["sprime_val"] = FE.summary(FE.ap(E, sp)); rec["s_val"] = FE.summary(FE.ap(E, s))
    json.dump(rec, open(out, "w"), indent=1)
    return rec


def _morph_tile(a):
    from pycocotools import mask as M
    from scipy import ndimage as ndi
    rles, k = a; out = []
    for r in rles:
        m = M.decode(r); ys, xs = np.nonzero(m)
        if len(ys) == 0:
            out.append(r); continue
        p = abs(k) + 2; y0, y1, x0, x1 = max(0, ys.min() - p), min(2048, ys.max() + p + 1), max(0, xs.min() - p), min(2048, xs.max() + p + 1)
        c = m[y0:y1, x0:x1].astype(bool); st = ndi.generate_binary_structure(2, 1)
        c = ndi.binary_dilation(c, st, iterations=k) if k > 0 else ndi.binary_erosion(c, st, iterations=-k, border_value=0)
        if not c.any():
            c = m[y0:y1, x0:x1].astype(bool)  # an erosion that empties the mask keeps the original
        m2 = np.zeros_like(m); m2[y0:y1, x0:x1] = c; e = M.encode(np.asfortranarray(m2)); e["counts"] = e["counts"].decode(); out.append(e)
    return out


def side_morph():
    """Mask dilation (+k) / erosion (-k) by k px (4-connected) at 2048, ranking fixed at s'. Val only."""
    import multiprocessing as mp
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    out = os.path.join(R.OUT, "rerankR3_side_ap5095.json"); rec = json.load(open(out)) if os.path.exists(out) else {}
    rec.setdefault("morph", {}); vg = json.load(open(R.VAL_GT)); P = json.load(open(os.path.join(R.FC_, f"{R.TAG}_val_mask.json")))
    tids = sorted(P); sp = {t: P[t]["s_prime"] for t in tids}
    for k in (-3, -2, -1, 1, 2, 3):
        if str(k) in rec["morph"]:
            continue
        t0 = time.time(); cache = os.path.join(R.FC_, f"fasteval_val_morph{k:+d}.pkl")
        if not os.path.exists(cache):
            with mp.get_context("spawn").Pool(NPROC) as pool:
                M2 = pool.map(_morph_tile, [([SL._as_bytes(r) for r in P[t]["masks_rle"]], k) for t in tids])
            E = FE.build({t: {"masks_rle": m} for t, m in zip(tids, M2)}, vg, cache)
        else:
            E = FE.build(None, None, cache)
        rec["morph"][str(k)] = FE.summary(FE.ap(E, sp)); print(f"[side] morph {k:+d}: {rec['morph'][str(k)]} {time.time()-t0:.0f}s", flush=True)
        json.dump(rec, open(out, "w"), indent=1)
    return rec


def main():
    import multiprocessing as mp
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=["all", "stage1", "stage2", "select", "confirm", "final", "side"]); a = a.parse_args()
    with mp.get_context("spawn").Pool(NPROC, initializer=_init) as pool:
        if a.cmd in ("all", "stage1"):
            stage1(pool)
        if a.cmd in ("all", "stage2"):
            stage2(pool)
        if a.cmd in ("all", "select"):
            select()
        chosen = None
        if a.cmd in ("all", "confirm"):
            cp = os.path.join(R.OUT, "rerankR3_confirm.json")
            chosen = json.load(open(cp))["chosen"] if os.path.exists(cp) else confirm(pool)
            print(f"[confirm] chosen {chosen}", flush=True)
        if a.cmd in ("all", "side"):
            chosen = chosen or json.load(open(os.path.join(R.OUT, "rerankR3_confirm.json")))["chosen"]
            side(pool, chosen)
    if a.cmd in ("all", "side"):
        side_morph()
    if a.cmd in ("all", "final"):
        if not os.path.exists(os.path.join(R.OUT, "rerankR3_final.json")):
            final(json.load(open(os.path.join(R.OUT, "rerankR3_confirm.json")))["chosen"])


if __name__ == "__main__":
    main()
