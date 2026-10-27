"""Mask arms on LACE's VAL detections, scored by fast legacy AP (fastap; exact vs COCO crowd on the deployed arm).

Per tile (worker): decode boxes (as ap_val: det_phase4_L24_s0, thr 0.05, topk 600) -- taken verbatim from
feat_cache/boxfix_ap_val_tiles so boxes and product scores are identical across all arms -- project features once,
then for each detection compute the inputs an arm needs (cell posterior bilinear `prob` as GA._prob; pixel posterior
via PixelLACE when asked) and the arm's mask; per tile return (iou matrix, ign) per arm. Nothing heavy is stored:
masks are deterministic and re-rendered for the finalists.

An arm is a JSON dict: {"kind": <render kind>, ...params}. Kinds in `RENDER`. Results ->
hunt/results/<batch>.json (refuses to overwrite): AP50/AP75 all/A/B per arm + gate on 'deployed'.

  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.hunt.arms BATCH ARMS.json [--tiles N] [--save arm1,arm2]
  (--z4: restrict to tiles with hunt_z4p_val features -- a SUBSET, compare arms only with the in-batch 'sam' arm)
  (--oracles F.json: {name: [arm, ...]} CEILING arms that pick per detection the member mask with the best GT IoU)
  (--save: per-tile masks -> feat_cache/hunt_masks/<batch>__<arm>/, readable by diag_pair as hunt_masks/<batch>__<arm>:-)
"""
import json, multiprocessing as mp, os, sys, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); NR = os.path.dirname(HERE)
FC = "/Users/tompitts/dphil/feat_cache"
NPZ = f"{FC}/em_model_4p_fix.npz"
_W = {}


def _init(arms, batch="", save=(), oracles=None):
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.native_raster import guided_ablate_modal as GA, guided_rgb_ext_val as GV
    from boxinst_commonality_tcd_04.native_raster.hunt import fastap as F, render as R
    from boxinst_commonality_tcd_04.native_raster.hunt.pxlace import PixelLACE
    m = E.TCDMasker(NPZ)
    assert abs(m.prior_weight - GA.ALPHA) < 1e-9 and abs(m.kappa_scale - GA.KAPPA) < 1e-9
    m4 = E.TCDMasker(NPZ); m4.s, m4.origin = 4, 8.0          # same masker on the real-4px lattice (feat16p.py)
    _W["m4"] = m4; _W["PL4"] = PixelLACE(m4)
    _W.update(m=m, GA=GA, GV=GV, F=F, R=R, PL=PixelLACE(m), arms=arms, gtc=F.gt_cache("val"),
              man=json.load(open(GV.MANIFEST))["feat_traintile"], batch=batch, save=list(save), oracles=oracles or {})


def _tile(t):
    m, GA, F, R = _W["m"], _W["GA"], _W["F"], _W["R"]
    src = json.load(open(f"{FC}/boxfix_ap_val_tiles/{t}.json"))
    bx = np.asarray(src["boxes_2048"], np.float32).reshape(-1, 4)
    ctx = {"rgbf": _W["GV"].load_rgb_verified(t, _W["man"]).astype(np.float32) / 255.0, "m": m, "PL": _W["PL"], "GA": GA}
    if len(bx):
        feat = np.load(f"{FC}/feat_4p_val/{t}.npy").astype(np.float32)
        g = feat.shape[-1]; zn = m.project(feat); del feat
        ctx.update(zn=zn, g=g, Z=zn.reshape(g, g, -1), boxes=bx, scores=np.asarray(src["scores_product"]))
        ctx["cell_rgb"] = None
        if any(s.get("guide") == "wrong" for s in _W["arms"].values()):   # control: RGB of the next val tile (sorted)
            order = sorted(_W["gtc"]); other = order[(order.index(t) + 1) % len(order)]
            ctx["rgbf_wrong"] = _W["GV"].load_rgb_verified(other, _W["man"]).astype(np.float32) / 255.0
        z4 = f"{FC}/hunt_z4p_val/{t}.npy"
        if os.path.exists(z4):
            ctx.update(m4=_W["m4"], zn4=np.load(z4).astype(np.float32).reshape(-1, 128), g4=512, PL4=_W["PL4"])
            ctx["Z4"] = ctx["zn4"].reshape(512, 512, 128)
        if any(s["kind"] == "sam" for s in _W["arms"].values()):
            ctx["sam"] = json.load(open(f"{FC}/hunt_sam_val_dets/{t}.json"))["masks_rle"]
    masks = {a: [] for a in _W["arms"]}
    comp = {a: [] for a, spec in _W["arms"].items() if "compete" in spec}   # (win, mask window, strength window)
    # optional per-arm ranking (spec "rank"): "own8" = s*bimod*pfg_mean from the 8 px cell posterior (must reproduce
    # scores_product), "own4" = the same product from the 4 px cell posterior (m4, alpha/kappa as the arm). Default:
    # the published scores_product.
    s_raw = np.asarray(src["scores_s"], np.float64)
    rank = {a: [] for a, spec in _W["arms"].items() if spec.get("rank")}
    for i, b in enumerate(bx):
        clip, prob, r = GA._prob(m, ctx["zn"], ctx["g"], b, GA.ALPHA, GA.KAPPA)
        det = {"i": i, "box": b, "clip": clip, "prob": prob, "r": r, "cache": {}}
        for a in rank:
            spec = _W["arms"][a]
            if spec["rank"] == "own8":
                rr = r
            else:
                key = ("prob4", spec.get("alpha", 0.3), spec.get("kappa", 1.6))
                if key not in det["cache"]:
                    det["cache"][key] = GA._prob(_W["m4"], ctx["zn4"], ctx["g4"], b, key[1], key[2])
                rr = det["cache"][key][2]
            rank[a].append(round(float(s_raw[i] * np.abs(2 * rr - 1).mean() * rr.mean()), 8))
        for a, spec in _W["arms"].items():
            mk = R.RENDER[spec["kind"]](ctx, det, spec)
            if a in comp:
                by0, by1, bx0, bx1 = clip; win = (max(0, by0 - 16), min(2048, by1 + 16), max(0, bx0 - 16), min(2048, bx1 + 16))
                comp[a].append((win, mk[win[0]:win[1], win[2]:win[3]].copy(), prob[win[0]:win[1], win[2]:win[3]].copy()))
                masks[a].append(None)
            else:
                masks[a].append(GA._rle(mk))
    for a, items in comp.items():                            # instance competition (tile level)
        c = _W["arms"][a]["compete"]; bi = GA.box_iou_mat(bx.astype(np.float64), bx.astype(np.float64)) if len(bx) else None
        for i, (wi, mi, si) in enumerate(items):
            keep = mi.copy()
            for j, (wj, mj, sj) in enumerate(items):
                if j == i or bi[i, j] >= c.get("iou_max", 0.3):
                    continue
                y0, y1, x0, x1 = max(wi[0], wj[0]), min(wi[1], wj[1]), max(wi[2], wj[2]), min(wi[3], wj[3])
                if y0 >= y1 or x0 >= x1:
                    continue
                si_ = si[y0 - wi[0]:y1 - wi[0], x0 - wi[2]:x1 - wi[2]]; sj_ = sj[y0 - wj[0]:y1 - wj[0], x0 - wj[2]:x1 - wj[2]]
                mj_ = mj[y0 - wj[0]:y1 - wj[0], x0 - wj[2]:x1 - wj[2]] if c.get("need_mask", True) else True
                lose = mj_ & (sj_ > si_ + c.get("delta", 0.1))
                keep[y0 - wi[0]:y1 - wi[0], x0 - wi[2]:x1 - wi[2]] &= ~lose
            full = np.zeros((2048, 2048), bool); full[wi[0]:wi[1], wi[2]:wi[3]] = keep
            masks[a][i] = GA._rle(full)
    for a in _W.get("save", []):                         # finalists: keep masks for diag_pair / figures
        d = f"{FC}/hunt_masks/{_W['batch']}__{a}"; os.makedirs(d, exist_ok=True)
        json.dump({"boxes_2048": src["boxes_2048"], "scores_product": src["scores_product"], "masks_rle": masks[a]},
                  open(f"{d}/{t}.json.tmp", "w")); os.replace(f"{d}/{t}.json.tmp", f"{d}/{t}.json")
    st = {a: F.tile_stats(ms, _W["gtc"][t]) for a, ms in masks.items()}
    for name, members in _W.get("oracles", {}).items():      # CEILING ONLY (uses GT): per det, the member arm
        if not len(bx):                                       # whose mask has the highest IoU with any crown
            st[name] = st[members[0]]; continue
        best = np.stack([st[a][0].max(1) if st[a][0].shape[1] else np.zeros(len(bx)) for a in members]).argmax(0)
        st[name] = (np.stack([st[members[k]][0][i] for i, k in enumerate(best)]) if st[members[0]][0].shape[1] else st[members[0]][0],
                    np.array([st[members[k]][1][i] for i, k in enumerate(best)], bool))
    sc = {a: rank.get(a, src["scores_product"]) for a in st}
    return t, st, sc


def main():
    batch, arms_fp = sys.argv[1], sys.argv[2]
    ntiles = int(sys.argv[sys.argv.index("--tiles") + 1]) if "--tiles" in sys.argv else None
    save = sys.argv[sys.argv.index("--save") + 1].split(",") if "--save" in sys.argv else []
    oracles = json.load(open(sys.argv[sys.argv.index("--oracles") + 1])) if "--oracles" in sys.argv else {}
    out = os.path.join(HERE, "results", f"{batch}.json"); os.makedirs(os.path.dirname(out), exist_ok=True)
    assert not os.path.exists(out), out
    arms = json.load(open(arms_fp)); assert "deployed" in arms
    from boxinst_commonality_tcd_04.native_raster.hunt import fastap as F
    gtc = F.gt_cache("val"); tids = sorted(gtc)[:ntiles] if ntiles else sorted(gtc)
    if "--z4" in sys.argv:                                   # only tiles whose 4px features exist (subset!)
        tids = [t for t in tids if os.path.exists(f"{FC}/hunt_z4p_val/{t}.npy")]; ntiles = len(tids)
    t0 = time.time(); res = {}
    with mp.get_context("spawn").Pool(12, initializer=_init, initargs=(arms, batch, save, oracles)) as pool:
        for n, (t, st, sc) in enumerate(pool.imap_unordered(_tile, tids), 1):
            res[t] = (st, sc)
            if n % 12 == 0 or n == len(tids):
                print(f"  [{batch}] {n}/{len(tids)} {time.time()-t0:.0f}s", flush=True)
    n_gt = {t: len(gtc[t]["trees"]) for t in tids}
    summ = {"batch": batch, "n_tiles": len(tids), "arms": arms, "AP50": {}, "AP75": {}}
    for a in list(arms) + list(oracles):
        S = [res[t][0][a] for t in tids]; SC = [res[t][1][a] for t in tids]
        summ["AP50"][a] = F.ap_halves(tids, S, SC, n_gt); summ["AP75"][a] = F.ap_halves(tids, S, SC, n_gt, .75)
    if ntiles is None:
        summ["gate_deployed"] = {"got": summ["AP50"]["deployed"]["all"], "ref": 0.6636, "pass": abs(summ["AP50"]["deployed"]["all"] - 0.6636) < 1e-4}
    for a, spec in arms.items():                             # GATE: own8 ranking must reproduce the published product
        if spec.get("rank") == "own8":
            dmax = max((float(np.max(np.abs(np.asarray(res[t][1][a]) - np.asarray(res[t][1]["deployed"])))) for t in tids if len(res[t][1][a])), default=0.0)
            summ[f"gate_rank_own8__{a}"] = {"max_abs_score_diff": dmax, "pass": dmax <= 1e-8}
            print(f"  GATE rank own8 [{a}] max |score - scores_product| = {dmax:.2e} -> {'PASS' if dmax <= 1e-8 else 'FAIL'}", flush=True)
    summ["sam_val_ref"] ={"AP50": {"all": 0.6776, "A": 0.6844, "B": 0.6721}, "AP75": {"all": 0.2839}}
    json.dump(summ, open(out, "w"), indent=1)
    summ["oracles"] = oracles
    d = summ["AP50"]["deployed"]
    for a in list(arms) + list(oracles):
        x = summ["AP50"][a]
        print(f"  {a:34s} AP50 {x['all']:.4f} ({x['all']-d['all']:+.4f}) A {x['A']:.4f} B {x['B']:.4f}  AP75 {summ['AP75'][a]['all']:.4f}", flush=True)
    print("  SAM val ref AP50 0.6776 (A 0.6844, B 0.6721), AP75 0.2839", flush=True)


if __name__ == "__main__":
    main()
