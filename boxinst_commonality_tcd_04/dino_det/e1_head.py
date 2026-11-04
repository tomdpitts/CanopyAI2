"""Pre-registration E1 (README.md): a scoring head on DINO's final-decoder features (256-d) + log s'. Val box labels.

  pull <split>: modal volume -> feat_cache/e1_emb/<split>/<tid>.npy (+ DONE.json; parity must be clean)
  cv          : heads (a) logreg, (b) MLP 1x64, (c) MLP 2x64; 5 seeds x 3 fold assignments; selection + gate -> results/e1_cv.json
  test        : ONLY if the gate passed. The 25-model mean on the test 439 -> frozen score_coco; bootstrap vs s' and vs R2 -> results/e1_test.json
  sparse      : ONLY after test. The same frozen models on sparse-236 -> results/e1_sparse.json
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.e1_head pull val|cv|test|sparse
"""
import argparse, contextlib, io, json, os, subprocess, time
import numpy as np

from boxinst_commonality_tcd_04.dino_det import rerank as R
from boxinst_commonality_tcd_04.dino_det.rerank2 import SEEDS, fold_sets

EMB = os.path.join(R.FC_, "e1_emb"); HEADS = ["logreg", "mlp1", "mlp2"]
SPARSE_GT = os.path.join(R.REPO, "data", "tcd_sparse", "sparse_gt.json")


def pull(split):
    """`modal volume get <dir> <existing parent>` creates <parent>/emb_<split>/ (the npz packing was not served: warm container)."""
    os.makedirs(EMB, exist_ok=True); d = os.path.join(EMB, split); src = os.path.join(EMB, f"emb_{split}")
    if not os.path.exists(os.path.join(d, "DONE.json")):
        subprocess.run([os.path.join(R.REPO, ".venv", "bin", "modal"), "volume", "get", "--force", "tcd04-tierb-vol",
                        f"runs/{R.TAG}/emb_{split}", EMB], check=True)
        os.replace(src, d)
    rec = json.load(open(os.path.join(d, "DONE.json"))); print(json.dumps(rec), flush=True)
    assert not rec["parity_fail"] and rec["n_files"] == rec["n_tiles"], rec


def load_emb(split, tids, counts):
    E = [np.load(os.path.join(EMB, split, f"{t}.npy")).astype(np.float32) for t in tids]
    for t, e, n in zip(tids, E, counts):
        assert e.shape == (n, 256), (t, e.shape, n)
    return np.concatenate(E)


def inputs(E, sp):
    return np.column_stack([E, np.log(np.maximum(sp, 1e-6))])


def fit(X, y, head, seed):
    import torch
    torch.manual_seed(seed); mu, sd = X.mean(0), X.std(0) + 1e-6; d = X.shape[1]; nn = torch.nn
    net = {"logreg": lambda: nn.Linear(d, 1),
           "mlp1": lambda: nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1)),
           "mlp2": lambda: nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1),
                                         nn.Linear(64, 1))}[head]()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed); lossf = nn.BCEWithLogitsLoss()
    for _ in range(30):
        net.train()
        for idx in torch.randperm(len(Xt), generator=g).split(1024):
            opt.zero_grad(); lossf(net(Xt[idx]).squeeze(1), yt[idx]).backward(); opt.step()
    net.eval()

    def pred(Z):
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor((Z - mu) / sd, dtype=torch.float32)).squeeze(1)).numpy().astype(np.float64)
    return pred


def oof(tids, tile, X, y, head, seed, tf):
    fold = np.asarray(tf)[tile]; out = np.zeros(len(X)); fits = []
    for k in range(R.K):
        tr = (fold != k) & (y >= 0); f = fit(X[tr], y[tr], head, seed); te = fold == k; out[te] = f(X[te]); fits.append(f)
    return out, fits


def val_data():
    vg = json.load(open(R.VAL_GT)); P = json.load(open(os.path.join(R.FC_, f"{R.TAG}_val_mask.json")))
    tids, tile, Xr, y = R.features("val", P, vg)
    E = load_emb("val", tids, [len(P[t]["scores"]) for t in tids])
    return vg, P, tids, tile, inputs(E, Xr[:, 1]), y, Xr


def cv():
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    out = os.path.join(R.OUT, "e1_cv.json"); assert not os.path.exists(out), out
    vg, P, tids, tile, X, y, Xr = val_data(); gd, t2i = SL.build_gt_crowd(vg, 2048)

    def val_ap(v):
        sc = R.split_scores(tids, tile, v)
        pr = {t: {"boxes_2048": P[t]["boxes_2048"], "scores": sc[t], "masks_rle": P[t]["masks_rle"]} for t in tids}
        with contextlib.redirect_stdout(io.StringIO()):
            return round(SL.score(pr, gd, t2i, masks=True, floor=0.0)["mask"]["AP50"], 4), pr
    base, Pb = val_ap(Xr[:, 1]); assert abs(base - 0.7384) < 5e-4, base
    F = fold_sets(len(tids)); runs = {h: [] for h in HEADS}; pr00 = {}; t0 = time.time()
    for fname, tf in F.items():
        for seed in SEEDS:
            for h in HEADS:
                a, pr = val_ap(oof(tids, tile, X, y, h, seed, tf)[0]); runs[h].append(round(a - base, 4))
                if fname == "rr" and seed == 0:
                    pr00[h] = pr
            print(f"[E1] {fname} s{seed}: " + " ".join(f"{h} {runs[h][-1]:+.4f}" for h in HEADS) + f" {time.time()-t0:.0f}s", flush=True)
    mean = {h: float(np.mean(runs[h])) for h in HEADS}; best = max(HEADS, key=lambda h: mean[h])
    chosen = next(h for h in HEADS if mean[h] >= mean[best] - 0.002)
    g = np.asarray(runs[chosen]); bt = R.boot(pr00[chosen], Pb, vg, 0.0, 0.0)
    gate = bool(g.mean() >= 0.005 and (g > 0).sum() >= 13 and bt["ci95"][0] > 0)
    rec = {"base_sprime_AP50": base, "gain_runs": runs,
           "gain_mean_sd": {h: [round(mean[h], 4), round(float(np.std(runs[h], ddof=1)), 4)] for h in HEADS},
           "chosen": chosen, "n_pos": int((g > 0).sum()), "boot_rr_seed0_vs_sprime": bt, "gate_pass": gate,
           "R2_val_gain_ref": 0.0061, "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(rec, open(out, "w"), indent=1); print(json.dumps({k: v for k, v in rec.items() if k != "gain_runs"}, indent=1), flush=True)


def _apply(split, sprime_file, gt_path, ref_files, tag):
    from boxinst_commonality_tcd_04 import score_coco as SC
    cvr = json.load(open(os.path.join(R.OUT, "e1_cv.json"))); assert cvr["gate_pass"], "gate failed: must not run"
    out = os.path.join(R.OUT, f"e1_{tag}.json"); assert not os.path.exists(out), out
    _, _, tids_v, tile_v, Xv, yv, _ = val_data(); tf = fold_sets(len(tids_v))["rr"]
    fits = [f for s in SEEDS for f in oof(tids_v, tile_v, Xv, yv, cvr["chosen"], s, tf)[1]]; assert len(fits) == 25
    St = json.load(open(sprime_file))["preds"]; tids = sorted(St)
    E = load_emb(split, tids, [len(St[t]["scores"]) for t in tids])
    sp = np.concatenate([np.asarray(St[t]["scores"], np.float64) for t in tids])
    tile = np.concatenate([np.full(len(St[t]["scores"]), i) for i, t in enumerate(tids)])
    v = np.mean([f(inputs(E, sp)) for f in fits], 0); sc = R.split_scores(tids, tile, v)
    preds = {t: {"boxes_2048": St[t]["boxes_2048"], "scores": sc[t], "masks_rle": St[t]["masks_rle"]} for t in tids}
    fp = os.path.join(R.OUT, f"preds_{R.TAG}_{split}_e1_r2048.json")
    json.dump({"meta": {"mask_res": 2048, "model": f"D1 + LACE masker, E1 head ({cvr['chosen']}) on DINO decoder features + log s', val-trained, box labels"},
               "preds": preds}, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    os.chdir(R.REPO)
    c = SC.score_many([fp], gt_path, res=2048, max_dets=600, score_floor=0.0)[0]
    json.dump(c, open(fp.replace("preds_", "scored_"), "w")); cr = c["canopy_neutral_crowd"]
    G = json.load(open(gt_path))
    boots = {k: R.boot(preds, json.load(open(f))["preds"], G, 0.0, 0.0) for k, f in ref_files.items() if os.path.exists(f)}
    rec = {"split": split, "chosen": cvr["chosen"], "mask": {k: cr["mask"][k] for k in ("AP50", "AP75", "AP50_95")},
           "box": {k: cr["box"][k] for k in ("AP50", "AP75")}, "boot": boots, "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(rec, open(out, "w"), indent=1); print(json.dumps(rec, indent=1), flush=True)


def test():
    sp = os.path.join(R.OUT, f"preds_{R.TAG}_test_sprime_r2048.json")
    _apply("test", sp, R.TEST_GT, {"vs_sprime": sp, "vs_R2": os.path.join(R.OUT, f"preds_{R.TAG}_test_rerankR2_r2048.json")}, "test")


def sparse():
    assert os.path.exists(os.path.join(R.OUT, "e1_test.json")), "sparse runs after test"
    sp = os.path.join(R.OUT, f"preds_{R.TAG}_sparse_sprime_r2048.json")
    _apply("sparse", sp, SPARSE_GT, {"vs_sprime": sp}, "sparse")


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=["pull", "cv", "test", "sparse"]); a.add_argument("split", nargs="?")
    a = a.parse_args()
    pull(a.split) if a.cmd == "pull" else {"cv": cv, "test": test, "sparse": sparse}[a.cmd]()
