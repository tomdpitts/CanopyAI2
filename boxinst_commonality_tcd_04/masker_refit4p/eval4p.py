"""Evaluate refitted 4 px maskers on frozen D1 boxes (pre-registration: masker_refit4p/README.md).

  val   --tag deployed|n120|n250   render the frozen D1 val boxes with that masker (lace4px_simple render path, alpha 0.3,
                                   kappa x1.6, tau 0.33 x box peak, guided r12 eps 1e-4 -- all unchanged) and score val
                                   mask AP50 under s' (score_lite crowd arm, maxDets 600, floor 0). 'deployed' is the parity
                                   gate: it must reproduce D1 + edge fix val s' 0.7414.
  select                           apply the pre-registered rule -> results/select.json
  final                            selected masker ONCE on test 439 and sparse 236 (frozen score_coco) + paired tile
                                   bootstrap vs the deployed masker's D1 s' predictions with the edge fix
                                   (dino_det/results/edgefix/preds_d1_edgefix_*; same boxes, same detector scores).
Masks are persisted as RLE (feat_cache/refit4p_<split>_<tag>/<tid>.json per-tile checkpoints = resume).
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.masker_refit4p.eval4p val --tag deployed
"""
import argparse, json, multiprocessing as mp, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); REPO = os.path.dirname(PKG)
FC = "/Users/tompitts/dphil/feat_cache"
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
D1 = os.path.join(PKG, "dino_det", "results")
NPZ = {"deployed": f"{FC}/em_model_4p_fix.npz", "n120": os.path.join(RES, "fit_n120", "em_model_z4p_n120.npz"),
       "n250": os.path.join(RES, "fit_n250", "em_model_z4p_n250.npz")}
VAL_GT = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "val_gt.json")
GT = {"test": os.path.join(PKG, "test_gt.json"), "sparse": os.path.join(REPO, "data", "tcd_sparse", "sparse_gt.json")}
N_SPLIT = {"val": 108, "test": 439, "sparse": 236}
# D1 val s' WITH the shared top/left edge fix (GA.EDGE_FILL = True, adopted 2026-11-04; dino_det/SAM3_D1.md
# "+ top/left fix" 0.7414 / 0.3392 / 0.3788). The pre-fix 0.7384 is not reproducible with the shared flag on.
PARITY = 0.7414
GAIN, TIE = 0.003, 0.001   # pre-registered


def _init(split, npz):
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.native_raster import guided_ablate_modal as GA
    from boxinst_commonality_tcd_04.native_raster.hunt import render as R
    from boxinst_commonality_tcd_04.native_raster.hunt.btmask import btmask as BM
    assert GA.EDGE_FILL is True, "R4P is pre-registered with the shared edge fix on"
    m4 = E.TCDMasker(npz); m4.s, m4.origin = 4, 8.0          # = btmask._init, masker path swapped
    BM._W.update(GA=GA, R=R, m4=m4, split=split)


def _job(args):
    split, t, boxes, scores, d = args
    from boxinst_commonality_tcd_04.native_raster.hunt.btmask import btmask as BM
    fp = os.path.join(d, t + ".json")
    if not os.path.exists(fp):
        r = BM.render_boxes(split, t, np.asarray(boxes, np.float32).reshape(-1, 4), scores, [BM.FROZEN])[BM.cfg_name(BM.FROZEN)]
        json.dump(r, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    return t


def d1_boxes(split):
    P = json.load(open(os.path.join(D1, "modal", f"dino_swinl_s0_preds_{split}.json")))["preds"]
    assert len(P) == N_SPLIT[split], (split, len(P))
    return P


def render(split, tag):
    import hashlib
    npz = NPZ[tag]; sha = hashlib.sha1(open(npz, "rb").read()).hexdigest()
    P = d1_boxes(split); tids = sorted(P)
    d = f"{FC}/refit4p_{split}_{tag}"; os.makedirs(d, exist_ok=True)
    jobs = [(split, t, P[t]["boxes_2048"], P[t]["scores"], d) for t in tids if not os.path.exists(os.path.join(d, t + ".json"))]
    print(f"[render {split} {tag}] {len(jobs)}/{len(tids)} tiles to render, masker sha1 {sha[:12]}", flush=True)
    t0 = time.time()
    if jobs:
        with mp.get_context("spawn").Pool(12, initializer=_init, initargs=(split, npz)) as pool:
            for k, _ in enumerate(pool.imap_unordered(_job, jobs), 1):
                if k % 20 == 0 or k == len(jobs):
                    print(f"  {k}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    M = {t: json.load(open(os.path.join(d, t + ".json"))) for t in tids}
    return P, M, tids, sha


def val(tag):
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    out = os.path.join(RES, f"val_{tag}.json"); assert not os.path.exists(out), out
    P, M, tids, sha = render("val", tag)
    gd, t2i = SL.build_gt_crowd(json.load(open(VAL_GT)), 2048)
    preds = {t: {"boxes_2048": P[t]["boxes_2048"], "scores": M[t]["s_prime"], "masks_rle": M[t]["masks_rle"]} for t in tids}
    r = SL.score(preds, gd, t2i, masks=True, floor=0.0)
    rec = {"tag": tag, "masker_sha1": sha, "ranking": "s'", "mask": {k: r["mask"][k] for k in ("AP50", "AP75", "AP50_95")},
           "box": {k: r["box"][k] for k in ("AP50", "AP75")}, "date": time.strftime("%Y-%m-%d %H:%M")}
    if tag == "deployed":
        rec["parity"] = {"expected": PARITY, "pass": abs(rec["mask"]["AP50"] - PARITY) < 5e-5}
    json.dump(rec, open(out, "w"), indent=1); print(json.dumps(rec), flush=True)


def select():
    out = os.path.join(RES, "select.json"); assert not os.path.exists(out), out
    V = {t: json.load(open(os.path.join(RES, f"val_{t}.json"))) for t in ("deployed", "n120", "n250")}
    assert V["deployed"]["parity"]["pass"], "parity gate failed: no refit is evaluated"
    base = V["deployed"]["mask"]["AP50"]
    ok = [t for t in ("n120", "n250") if V[t]["mask"]["AP50"] >= base + GAIN]
    best = max((V[t]["mask"]["AP50"] for t in ok), default=None)
    sel = next((t for t in ok if V[t]["mask"]["AP50"] >= best - TIE), None)   # ok is ordered smallest first
    rec = {"rule": f"adopt iff val s' mask AP50 >= deployed + {GAIN}; among passing arms the smallest within {TIE} of the best",
           "val_AP50": {t: V[t]["mask"]["AP50"] for t in V}, "passing": ok, "selected": sel,
           "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(rec, open(out, "w"), indent=1); print(json.dumps(rec, indent=1), flush=True)


def final():
    from boxinst_commonality_tcd_04 import score_coco as SC
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    from boxinst_commonality_tcd_04.dino_det.boot_vs_bt import ap, per_tile
    tag = json.load(open(os.path.join(RES, "select.json")))["selected"]
    assert tag, "no arm passed the val gate: test and sparse are not run"
    out = os.path.join(RES, f"final_{tag}.json"); assert not os.path.exists(out), out
    rec = {"tag": tag, "rows": {}, "boot": {}}
    for split in ("test", "sparse"):
        P, M, tids, sha = render(split, tag)
        fp = os.path.join(RES, f"preds_d1_{split}_{tag}_sprime_r2048.json")
        if not os.path.exists(fp):
            preds = {t: {"boxes_2048": P[t]["boxes_2048"], "scores": M[t]["s_prime"], "masks_rle": M[t]["masks_rle"]} for t in tids}
            json.dump({"meta": {"mask_res": 2048, "model": f"D1 boxes + LACE masker refit {tag} (sha1 {sha}), ranking s'"}, "preds": preds},
                      open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
        sp = fp.replace("preds_", "scored_")
        if not os.path.exists(sp):
            cwd = os.getcwd(); os.chdir(REPO)
            json.dump(SC.score_many([fp], GT[split], res=2048, max_dets=600, score_floor=0.0)[0], open(sp, "w")); os.chdir(cwd)
        c = json.load(open(sp))["canopy_neutral_crowd"]
        rec["rows"][split] = {"mask": {k: c["mask"][k] for k in ("AP50", "AP75", "AP50_95")}, "box": {k: c["box"][k] for k in ("AP50", "AP75")}}
        # paired tile bootstrap vs the deployed masker on the same D1 boxes (2,000 resamples, seed 0)
        g = json.load(open(GT[split])); gtids = sorted(g); gd, t2i = SL.build_gt_crowd(g, 2048); ids = np.array([t2i[t] for t in gtids])
        D = {}
        for k, f in (("refit", fp), ("deployed", os.path.join(D1, "edgefix", f"preds_d1_edgefix_{split}_sprime_r2048.json"))):
            Q = json.load(open(f)); Q = Q.get("preds", Q)
            D[k] = per_tile({t: Q[t] for t in gtids}, gd, t2i, 0.0)
            for i in ids:
                if i not in D[k]:
                    D[k][i] = (np.zeros(0), np.zeros(0, bool), sum(1 for a in gd["annotations"] if a["image_id"] == i and not a["iscrowd"]))
        rng = np.random.default_rng(0); diffs = []
        for _ in range(2000):
            s = ids[rng.integers(0, len(ids), len(ids))]
            diffs.append(ap(s, D["refit"], SL.REC_THRS) - ap(s, D["deployed"], SL.REC_THRS))
        full = ap(ids, D["refit"], SL.REC_THRS) - ap(ids, D["deployed"], SL.REC_THRS)
        rec["boot"][split] = {"diff_AP50": round(full, 4), "ci95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
                              "p_le_0": round(float(np.mean(np.asarray(diffs) <= 0)), 4)}
        print(f"[final {split}] {rec['rows'][split]} boot {rec['boot'][split]}", flush=True)
    json.dump(rec, open(out, "w"), indent=1); print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser(); ap_.add_argument("cmd", choices=["val", "select", "final"]); ap_.add_argument("--tag")
    a = ap_.parse_args()
    {"val": lambda: val(a.tag), "select": select, "final": final}[a.cmd]()
