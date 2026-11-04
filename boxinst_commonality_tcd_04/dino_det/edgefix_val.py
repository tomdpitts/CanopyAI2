"""Top/left tile-border fix for the LACE masker, VAL ONLY (108 tiles, frozen D1 val boxes). Record: dino_det/SAM3_D1.md.

Cause (dino_det/sam3_d1_border.py): guided_ablate_modal._prob bilinearly upsamples the 4 px cell posterior to 2048 and
shifts it by delta = round((origin - s/2)/scale) = 6 px down/right, ZERO-filling rows/cols 0..5 -> prob 0 on the top/left
tile band -> LACE misses ~55% of GT px at 0-2 px from top/left (SAM ~12%).
Fix ("edge"): fill the vacated band by edge replication (rows 0..delta-1 = row delta; cols likewise). Applied by wrapping
GA._prob inside this script's workers only (shared code untouched). Cell posterior r, hence s', is unchanged -> ranking
identical; only masks near the top/left border change.

Steps: (1) parity: baseline render (no fix) of 3 val tiles == cached D1 val masks (feat_cache/dino_swinl_s0_val_mask.json);
(2) render all 108 with the fix -> feat_cache/edgefix_val/<tid>.json (resume); (3) score_lite crowd arm, s' ranking,
baseline vs fix: mask AP50/AP75/AP50:95 + paired tile bootstrap of fix - baseline (2,000, seed 0); matched-crown mean IoU
(all / tile-edge crowns) and top/left border miss rates. -> results/edgefix/edgefix_val.json
Decision rule (written before running): carry to a single pre-registered test run (Tom decides) iff val AP50 >= baseline
- 0.001 AND edge-crown mean IoU improves; AP75 reported either way.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.edgefix_val
"""
import json, multiprocessing as mp, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE)
FC = "/Users/tompitts/dphil/feat_cache"; BASE = f"{FC}/dino_swinl_s0_val_mask.json"; TDIR = f"{FC}/edgefix_val"
OUT = os.path.join(HERE, "results", "edgefix"); VAL_GT = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "val_gt.json")
CFG = {"e": 0.0, "c": 0.33}; N_BOOT, SEED = 2000, 0


def _install(fix):
    """fix False = pre-2026-11-04 render (zero-filled band); True = edge-replicated band. Since adoption the fix lives
    in guided_ablate_modal._prob behind GA.EDGE_FILL; this toggles that flag (results identical to the original
    wrapper: same two assignments, same order)."""
    from boxinst_commonality_tcd_04.native_raster.hunt.btmask import btmask as BM
    if "m4" not in BM._W:
        BM._init("val")
    BM._W["GA"].EDGE_FILL = bool(fix)
    return BM


def _job(args):
    t, boxes, scores, fix = args
    BM = _install(fix)
    r = BM.render_boxes("val", t, np.asarray(boxes, np.float32).reshape(-1, 4), scores, [CFG])[BM.cfg_name(CFG)]
    return t, r


def main():
    from pycocotools import mask as mu
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    from boxinst_commonality_tcd_04.dino_det import sam3_on_d1 as S
    os.makedirs(OUT, exist_ok=True); os.makedirs(TDIR, exist_ok=True)
    B = json.load(open(BASE)); vg = json.load(open(VAL_GT)); tids = sorted(vg); assert sorted(B) == tids and len(tids) == 108
    # (1) parity
    par = {}
    with mp.get_context("spawn").Pool(3) as pool:
        for t, r in pool.imap_unordered(_job, [(t, B[t]["boxes_2048"], B[t]["scores"], False) for t in tids[:3]]):
            same = [a["counts"] == b["counts"] for a, b in zip(r["masks_rle"], B[t]["masks_rle"])]
            sp = np.abs(np.asarray(r["s_prime"]) - np.asarray(B[t]["s_prime"])).max() if len(same) else 0.0
            par[t] = {"n": len(same), "masks_identical": int(sum(same)), "s_prime_maxabs": float(sp)}
            print("[parity]", t, par[t], flush=True)
    assert all(v["masks_identical"] == v["n"] and v["s_prime_maxabs"] < 1e-6 for v in par.values()), par
    # (2) render with fix
    todo = [t for t in tids if not os.path.exists(f"{TDIR}/{t}.json")]; t0 = time.time()
    print(f"[edgefix] {len(todo)}/108 to render", flush=True)
    with mp.get_context("spawn").Pool(12) as pool:
        for n, (t, r) in enumerate(pool.imap_unordered(_job, [(t, B[t]["boxes_2048"], B[t]["scores"], True) for t in todo]), 1):
            json.dump(r, open(f"{TDIR}/{t}.json.tmp", "w")); os.replace(f"{TDIR}/{t}.json.tmp", f"{TDIR}/{t}.json")
            print(f"  {n}/{len(todo)} {t} {time.time()-t0:.0f}s", flush=True)
    F = {t: json.load(open(f"{TDIR}/{t}.json")) for t in tids}
    for t in tids:
        assert np.abs(np.asarray(F[t]["s_prime"], float) - np.asarray(B[t]["s_prime"], float)).max(initial=0) < 1e-6, t
    n_changed = sum(a["counts"] != b["counts"] for t in tids for a, b in zip(F[t]["masks_rle"], B[t]["masks_rle"]))
    # (3) score
    gd, t2i = SL.build_gt_crowd(vg, 2048); ids = np.array([t2i[t] for t in tids])
    rows, D = {}, {}
    for name, src in (("baseline", B), ("edgefix", F)):
        preds = {t: {"boxes_2048": B[t]["boxes_2048"], "scores": B[t]["s_prime"], "masks_rle": src[t]["masks_rle"]} for t in tids}
        r = SL.score(preds, gd, t2i, masks=True, floor=0.0); rows[name] = {k: r["mask"][k] for k in ("AP50", "AP75", "AP50_95")}
        D[name] = S.per_tile(preds, gd, t2i, 0.0)
        for i in ids:
            if i not in D[name]:
                ng = sum(1 for a in gd["annotations"] if a["image_id"] == i and not a["iscrowd"])
                D[name][i] = (np.zeros(0), np.zeros((10, 0), bool), np.zeros((10, 0), bool), ng)
        print(f"[val {name}] {rows[name]}", flush=True)
    rng = np.random.default_rng(SEED); dd = []
    for b in range(N_BOOT):
        s = ids[rng.integers(0, len(ids), len(ids))]
        a = [S.ap_at(s, D["edgefix"], SL.REC_THRS, k) - S.ap_at(s, D["baseline"], SL.REC_THRS, k) for k in range(10)]
        dd.append((a[0], a[5], float(np.mean(a))))
    dd = np.asarray(dd); boot = {m: {"diff": round(rows["edgefix"][m] - rows["baseline"][m], 4),
                                     "ci95": [round(float(np.percentile(dd[:, j], 2.5)), 4), round(float(np.percentile(dd[:, j], 97.5)), 4)]}
                                 for j, m in enumerate(("AP50", "AP75", "AP50_95"))}
    # matched-crown IoU (pairing as sam3_d1_gap) and top/left border miss
    trees = {}
    for a in gd["annotations"]:
        if not a["iscrowd"]:
            trees.setdefault(a["image_id"], []).append(a)
    iou = {"baseline": [], "edgefix": []}; edge = []; miss = {k: np.zeros((2, 2, 3)) for k in ("baseline", "edgefix")}  # side x [n, miss] x band
    bands = [0, 4, 8, 16]
    for t in tids:
        G = trees.get(t2i[t], []); bx = np.asarray(B[t]["boxes_2048"], float).reshape(-1, 4)
        if not G or not len(bx):
            continue
        xy = np.c_[bx[:, :2], bx[:, 2:] - bx[:, :2]]; bi = mu.iou(xy.tolist(), [x["bbox"] for x in G], [0] * len(G))
        for j, x in enumerate(G):
            i = int(bi[:, j].argmax())
            if bi[i, j] < 0.5:
                continue
            e = bool(xy[i, 0] < 2 or xy[i, 1] < 2 or xy[i, 0] + xy[i, 2] > 2046 or xy[i, 1] + xy[i, 3] > 2046); edge.append(e)
            gx, gy = x["bbox"][:2]; near = gx < 16 or gy < 16
            gm = mu.decode(x["segmentation"]).astype(bool) if near else None
            for name, src in (("baseline", B), ("edgefix", F)):
                m = dict(src[t]["masks_rle"][i], counts=src[t]["masks_rle"][i]["counts"].encode())
                iou[name].append(float(mu.iou([m], [x["segmentation"]], [0])[0, 0]))
                if near:
                    mm = mu.decode(m).astype(bool)
                    for sd, (gs, ms) in enumerate(((gm[:16], mm[:16]), (gm[:, :16].T, mm[:, :16].T))):
                        for k in range(3):
                            z = gs[bands[k]:bands[k + 1]]; miss[name][sd, 0, k] += z.sum(); miss[name][sd, 1, k] += (z & ~ms[bands[k]:bands[k + 1]]).sum()
    edge = np.asarray(edge); I = {k: np.asarray(v) for k, v in iou.items()}
    rec = {"date": time.strftime("%Y-%m-%d %H:%M"), "parity_baseline_render": par, "n_masks": int(sum(len(B[t]["masks_rle"]) for t in tids)),
           "n_masks_changed": int(n_changed), "mask_AP_sprime": rows, "edgefix_minus_baseline_boot": boot,
           "matched_crown_mean_iou": {k: {"all": round(float(v.mean()), 4), "tile_edge": round(float(v[edge].mean()), 4), "interior": round(float(v[~edge].mean()), 4)} for k, v in I.items()},
           "n_crowns": {"all": int(len(edge)), "tile_edge": int(edge.sum())},
           "border_miss_top_left": {k: {sd: dict(zip(["0-4", "4-8", "8-16"], np.round(v[s_, 1] / np.maximum(v[s_, 0], 1), 3).tolist())) for s_, sd in enumerate(("top", "left"))} for k, v in miss.items()}}
    rec["decision"] = {"AP50_ok": rows["edgefix"]["AP50"] >= rows["baseline"]["AP50"] - 0.001,
                       "edge_iou_up": rec["matched_crown_mean_iou"]["edgefix"]["tile_edge"] > rec["matched_crown_mean_iou"]["baseline"]["tile_edge"]}
    json.dump(rec, open(os.path.join(OUT, "edgefix_val.json"), "w"), indent=1); print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()
