"""CPU finalisation, run in the img_cpu container with the FROZEN project code:
  * stitching  = native_raster/stitch_seam.load_tile_crowns(edge_px=10) + stitch_res.dedup/serialize
                 (the seam-fixed path of the DT2/Box2Mask rows, same parameters)
  * scoring    = score_coco.score_many (res 2048, maxDets 600, floor 0.05, canopy crowd)
  * GT boxes   = box_supervised_baselines/gtbox_lib (tab:gtbox protocol) + A4's tile bootstrap
Every output file is written once (resume = skip if present); nothing is overwritten.
"""
from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool

import numpy as np

from tierb import crops as C
from tierb import lace_gate as LG
from tierb import score_lite as SL
from tierb import tierb_common as TC

PKG_R = "/root/proj/boxinst_commonality_tcd_04"
GT = {"test": f"{PKG_R}/test_gt.json", "sparse": "/root/proj/data/tcd_sparse/sparse_gt.json",
      "val": f"{PKG_R}/modal_tcd_multiseed/phase4/val_gt.json"}
EDGE_PX, IOU_THR, CONT_THR, FLOOR = 10, 0.7, 0.85, 0.05
# A stored LACE row per split, re-scored in THIS container before any Tier B file is scored: the
# frozen scorer's numbers depend on the PIL rasteriser, so the container must reproduce the
# environment that produced every published row (py3.10, Pillow 12.3.0, pycocotools 2.0.11) EXACTLY.
PARITY = {"test": (f"{PKG_R}/native_raster/preds_lace_s0_r2048_cgf_r12_tau035_guidedrun.json",
                   f"{PKG_R}/native_raster/scored_lace_s0_r2048_cgf_r12_tau035_guidedrun.json", 0.0),
          "sparse": (f"{PKG_R}/native_raster/preds_lace_sparse_s0_r2048_cgf_r12_tau035_guidedrun.json",
                     f"{PKG_R}/native_raster/scored_lace_sparse_s0_r2048_cgf_r12_tau035_guidedrun.json", 0.0)}
MEM_BYTES = 128 * 2 ** 30          # = finalize's memory request (tierb_modal.py)


class ScorerParityError(RuntimeError):
    pass


def score_files(entries, split):
    """entries [(preds_path, floor)] -> [score_coco result]. Exactly score_coco.score_many (one GT
    build, shared ctx, _score_one per file) but with a per-file floor, so the LACE parity row
    (published at floor 0) and the Tier B rows (floor 0.05) are scored against ONE GT build.
    The first entry of every call is the LACE parity row; it must equal its stored file exactly."""
    from boxinst_commonality_tcd_04 import score_coco as SC
    gt = json.load(open(GT[split]))
    tids = sorted(gt)
    print(f"[score] building {split} GT at 2048 (frozen score_coco.build_gt) ...", flush=True)
    gt_fp, tid2img, tree_rles, canopy_union = SC.build_gt(gt, 2048, with_canopy=False)
    gt_cr, _, _, _ = SC.build_gt(gt, 2048, with_canopy=True)
    ctx = (gt_fp, gt_cr, tid2img, tree_rles, canopy_union, tids)
    pp, ps, pf = PARITY[split]
    par = SC._score_one(pp, ctx, 2048, 600, pf)
    ref = json.load(open(ps))
    bad = []
    for arm, kind in (("canopy_neutral_crowd", "mask"), ("canopy_neutral_crowd", "box"), ("canopy_fp", "mask")):
        for k in ("AP50", "AP75", "AP50_95", "n_det"):
            if par[arm][kind][k] != ref[arm][kind][k]:
                bad.append((arm, kind, k, par[arm][kind][k], ref[arm][kind][k]))
    if bad:
        raise ScorerParityError(f"{split}: container scorer != stored LACE row: {bad}")
    print(f"[score] {split} parity PASS vs {os.path.basename(ps)}", flush=True)
    return [SC._score_one(p, ctx, 2048, 600, f) for p, f in entries]


def _w(path, obj):
    assert not os.path.exists(path), f"REFUSING to overwrite {path}"
    TC.atomic_json(path, obj)


# ------------------------------------------------------------------ B1/B2 preds
def assemble_whole(m, split, tids):
    d = TC.run_dir(m, "predict", split, "whole")
    preds = {}
    for t in tids:
        r = json.load(open(os.path.join(d, f"{t}.json")))
        preds[t] = {"boxes_2048": r["boxes_2048"], "scores": r["scores"], "masks_rle": r["masks_rle"]}
    return preds


_SCFG = {}


def _stitch_init(pred_dir):
    _SCFG["pred_dir"] = pred_dir


def _stitch_one(tid):
    from boxinst_commonality_tcd_04.native_raster import stitch_res as ST
    from boxinst_commonality_tcd_04.native_raster import stitch_seam as SS
    crowns, nd = SS.load_tile_crowns(_SCFG["pred_dir"], tid, 2048, FLOOR, EDGE_PX)
    crowns = ST.dedup(crowns, IOU_THR, CONT_THR)
    return tid, ST.serialize_tile(crowns, 2048), nd


def stitch_sub(m, split, tids):
    """The frozen stitcher holds one dense 2048^2 bool (4 MiB) per crown, and a tile can carry up
    to 9 x 600 subtile predictions, so workers are sized from the largest tile actually present.
    Per-tile JSONL checkpoint (as run_stitch_seam.py): a dropped container resumes, never restarts."""
    d = TC.run_dir(m, "predict", split, "sub")
    n_tile = {}
    for t in tids:
        dn = os.path.join(d, f"{t}.done")
        assert os.path.exists(dn), f"{split}/{t} sub incomplete"
        n_tile[t] = sum(json.load(open(dn))["n"].values())
    max_n = max(n_tile.values()) if n_tile else 1
    workers = int(max(1, min(28, (0.55 * MEM_BYTES) // (4.2e6 * 1.3 * max(max_n, 1)))))
    ck = TC.run_dir(m, "results", f"stitch_{split}.jsonl")
    done = {r["tid"]: r for r in TC.read_jsonl(ck)}
    todo = sorted((t for t in tids if t not in done), key=lambda t: -n_tile[t])
    print(f"[stitch] {m}/{split}: {len(done)} done, {len(todo)} to do; max {max_n} preds/tile -> "
          f"{workers} workers", flush=True)
    if todo:
        with Pool(workers, _stitch_init, (d,)) as pool:
            for k, (tid, rec, nd) in enumerate(pool.imap_unordered(_stitch_one, todo, chunksize=1)):
                TC.append_jsonl(ck, {"tid": tid, "rec": rec, "n_edge_dropped": nd})
                done[tid] = {"tid": tid, "rec": rec, "n_edge_dropped": nd}
                if (k + 1) % 20 == 0:
                    print(f"[stitch] {m}/{split} {k+1}/{len(todo)}", flush=True)
    return {t: done[t]["rec"] for t in tids}, sum(done[t]["n_edge_dropped"] for t in tids)


def _meta(m, split, geom, extra=None):
    return {"model": f"tierb {m} {geom} {split}", "mask_res": 2048, "n_tiles": None,
            "geometry": geom, "min_score": FLOOR, **(extra or {})}


def build_preds_d2(m, split):
    """-> {geom: path} with preds files for every geometry in crops.GEOMETRIES (written once)."""
    tids = TC.manifest()["ids"][split]
    out = {}
    for geom in C.GEOMETRIES:
        p = TC.run_dir(m, "results", f"preds_{m}_{geom}_{split}_r2048.json")
        if not os.path.exists(p):
            if geom == "whole":
                preds, extra = assemble_whole(m, split, tids), {"source": "model boxes, unrounded scores"}
            else:
                preds, drops = stitch_sub(m, split, tids)
                extra = {"stitcher": "native_raster/stitch_seam.py (seam-fixed)", "edge_px": EDGE_PX,
                         "dedup": {"iou_thr": IOU_THR, "cont_thr": CONT_THR},
                         "n_subtile_preds_dropped": drops}
            meta = _meta(m, split, geom, extra)
            meta["n_tiles"] = len(preds)
            _w(p, {"meta": meta, "preds": preds})
        out[geom] = p
    return out


def finalize_d2(m):
    res = {"method": m, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    # 1. geometry: fixed to 'sub' before any number (amendment 2026-10-29, TIERB.md section 000; was a
    #    val choice between whole and sub). Val is still stitched and scored, for the record only.
    choice_p = TC.run_dir(m, "results", "geometry_choice.json")
    if not os.path.exists(choice_p):
        assert C.GEOMETRIES == ("sub",), C.GEOMETRIES
        vp = build_preds_d2(m, "val")
        vg = json.load(open(GT["val"]))
        vg = {t: vg[t] for t in TC.manifest()["ids"]["val"]}
        gd, t2i = SL.build_gt_crowd(vg, 2048)
        val = {g: SL.score(json.load(open(p))["preds"], gd, t2i, masks=True) for g, p in vp.items()}
        _w(choice_p, {"rule": "fixed 'sub' (3x3 1024 subtiles = the training canvas, as DT2/Box2Mask), "
                              "pre-registered 2026-10-29; val scored for the record only",
                      "val": val, "primary": "sub"})
    choice = json.load(open(choice_p))
    res["val"], res["primary"] = choice["val"], choice["primary"]
    # 2. test + sparse, crops.GEOMETRIES, frozen scorer
    rows = []
    for split in [x for x in ("test", "sparse")
                  if os.path.isdir(TC.run_dir(m, "predict", x))]:   # sparse only if it was predicted
        paths = build_preds_d2(m, split)
        todo = [(g, p) for g, p in paths.items()
                if not os.path.exists(p.replace("preds_", "scored_"))]
        if todo:
            scored = score_files([(p, FLOOR) for _, p in todo], split)
            for (g, p), s in zip(todo, scored):
                _w(p.replace("preds_", "scored_"), s)
        for g, p in paths.items():
            s = json.load(open(p.replace("preds_", "scored_")))
            c = s["canopy_neutral_crowd"]
            rows.append({"name": f"{m}/{split}/{g}{' (primary)' if g == res['primary'] else ''}",
                         "split": split, "ranking": "product", "geometry": g,
                         "AP50": c["mask"]["AP50"], "AP50_95": c["mask"]["AP50_95"],
                         "AP75": c["mask"]["AP75"], "box_AP50": c["box"]["AP50"],
                         "n_det": c["mask"]["n_det"]})
    res["rows"] = rows
    res["summary"] = {r["name"]: {k: r[k] for k in ("AP50", "AP50_95", "box_AP50")} for r in rows}
    TC.atomic_json(TC.run_dir(m, "results", "summary.json"), res, indent=1)
    return res


# ------------------------------------------------------------------ B3 (MAL)
def _mal_tile(name, t):
    return json.load(open(TC.run_dir("mal", "predict", name, f"{t}.json")))


def _boot(diff_by_tile, n=2000, seed=0):
    """A4's tile bootstrap (gtbox_sam_persub.py `boot`), verbatim."""
    rs = np.random.RandomState(seed)
    tiles = list(diff_by_tile)
    s = [np.concatenate([diff_by_tile[tiles[j]] for j in rs.randint(0, len(tiles), len(tiles))]).mean()
         for _ in range(n)]
    return [round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4)]


def gtbox_eval(name, split):
    from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as G
    gt = json.load(open(GT[split]))
    tids = [t for t in TC.manifest()["ids"][split] if gt[t]["trees"]]
    per = {}
    for t in tids:
        r = _mal_tile(name, t)
        g, valid = G.gt_rles_and_valid(gt[t]["trees"], res=2048)
        pr = [{"size": x["size"], "counts": x["counts"].encode()} for x in r["masks_rle"]]
        iou = G.per_crown_iou_rle(pr, g)
        per[t] = {"iou": iou.round(5).tolist(), "valid": valid.tolist()}
    allv = np.concatenate([np.asarray(per[t]["iou"])[np.asarray(per[t]["valid"], bool)] for t in tids])
    out = {"n_tiles": len(tids), "n_crowns": int(len(allv)), "mean_iou": round(float(allv.mean()), 4),
           "ge50": round(float((allv >= .5).mean()), 4), "ge75": round(float((allv >= .75).mean()), 4),
           "ge90": round(float((allv >= .9).mean()), 4)}
    return out, per


def finalize_mal():
    res = {"method": "mal", "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    man = TC.manifest()
    # GT boxes: test (paired vs LACE, A4 protocol), sparse, val
    gtp = TC.run_dir("mal", "results", "gtbox_mal.json")
    if not os.path.exists(gtp):
        test_sum, test_per = gtbox_eval("test_gt", "test")
        lace = {}
        for line in open(f"{PKG_R}/{LG.LACE_GTBOX_PERCROWN}"):          # adopted 4 px masker (2026-10-27)
            r = json.loads(line)
            lace[r["tile"]] = r
        d = {}
        for t, v in test_per.items():
            va = np.asarray(v["valid"], bool)
            assert va.tolist() == lace[t]["valid"], f"{t}: validity differs from LACE's file"
            d[t] = np.asarray(v["iou"])[va] - np.asarray(lace[t]["iou"])[va]
        diff = float(np.concatenate(list(d.values())).mean())
        a4 = json.load(open(f"{PKG_R}/native_raster/audit_2026-10-14/selvabox_evidence/gtbox_sam_persub.json"))["summary"]
        sparse_sum = (gtbox_eval("sparse_gt", "sparse")[0]
                      if os.path.isdir(TC.run_dir("mal", "predict", "sparse_gt")) else None)
        val_sum, _ = gtbox_eval("val_gt", "val")
        _w(gtp, {"test": test_sum, "test_paired_vs_lace": {"mean_diff": round(diff, 4), "ci95": _boot(d),
                                                         "lace_mean_iou": LG.LACE_GTBOX_MEAN_IOU,
                                                         "lace_file": LG.LACE_GTBOX_PERCROWN},
                 "a4_reference": {k: a4[k] for k in ("lace", "ft_sub", "base_sub", "ft_whole", "base_whole")},
                 "sparse": sparse_sum, "val": val_sum, "per_tile_test": test_per})
    g = json.load(open(gtp))
    res["gtbox"] = {k: g[k] for k in ("test", "test_paired_vs_lace", "a4_reference", "sparse", "val")}
    # detector boxes -> frozen scorer, ranked by the detector's own score
    rows = []
    sets = {"test": [n for n in ("lace_s0", "lace_s1", "lace_s2", "selvabox")
                     if os.path.isdir(TC.run_dir("mal", "predict", n))],
            "sparse": [n for n in ("lace_sparse_s0", "lace_sparse_s1", "lace_sparse_s2")
                       if os.path.isdir(TC.run_dir("mal", "predict", n))]}
    for split, names in sets.items():
        if not names:
            continue
        paths, labels = [], []
        for name in names:
            # MAL's labeler has no confidence output, so its native ranking is the detector score s
            # (Tom 2026-10-27: no invented ranking). Gated against LACE's FULL s' row.
            for rk in ("detscore",):
                p = TC.run_dir("mal", "results", f"preds_mal_{name}_r2048.json")
                if not os.path.exists(p):
                    preds = {}
                    for t in man["ids"][split]:
                        r = _mal_tile(name, t)
                        sc = r["scores"]
                        preds[t] = {"boxes_2048": r["boxes_2048"], "scores": sc, "masks_rle": r["masks_rle"]}
                    how = "detector score s"
                    _w(p, {"meta": {"model": f"tierb MAL masks on {name} boxes, ranked by {how}",
                                    "mask_res": 2048, "n_tiles": len(preds), "min_score": FLOOR}, "preds": preds})
                paths.append(p); labels.append((name, rk))
        todo = [p for p in paths if not os.path.exists(p.replace("preds_", "scored_"))]
        if todo:
            for p, s in zip(todo, score_files([(p, FLOOR) for p in todo], split)):
                _w(p.replace("preds_", "scored_"), s)
        for (name, rk), p in zip(labels, paths):
            c = json.load(open(p.replace("preds_", "scored_")))["canopy_neutral_crowd"]
            rows.append({"name": f"mal/{name}", "split": split, "ranking": rk,
                         "boxes": name, "AP50": c["mask"]["AP50"], "AP50_95": c["mask"]["AP50_95"],
                         "AP75": c["mask"]["AP75"], "box_AP50": c["box"]["AP50"], "n_det": c["mask"]["n_det"]})
    for split, pref in (("test", "mal/lace_s"), ("sparse", "mal/lace_sparse_s")):
        for rk in ("detscore",):
            lr = [r for r in rows if r["name"].startswith(pref) and r["ranking"] == rk]
            if len(lr) < 2:
                continue
            band = {k: (round(float(np.mean([r[k] for r in lr])), 4), round(float(np.std([r[k] for r in lr], ddof=1)), 4))
                    for k in ("AP50", "AP50_95", "box_AP50")}
            rows.append({"name": f"{pref}{{0,1,2}} mean", "split": split, "ranking": rk,
                         "AP50": band["AP50"][0], "AP50_95": band["AP50_95"][0], "band_sd": band})
    res["rows"] = rows
    res["summary"] = {"gtbox_test_mean_iou": g["test"]["mean_iou"],
                      "gtbox_paired_vs_lace": g["test_paired_vs_lace"],
                      **{r["name"]: {k: r[k] for k in ("AP50", "AP50_95") if k in r} for r in rows}}
    TC.atomic_json(TC.run_dir("mal", "results", "summary.json"), res, indent=1)
    return res


def lace_gate(m, res):
    if m == "mal":
        # LACE-box rows (MAL masks ranked by s) vs LACE's FULL row (own s'): seed means vs LACE's
        # 3-seed mean when all 3 seeds were masked, else seed 0 vs LACE seed 0. SelvaBox-box rows are not gated.
        rows = [dict(r, lace_ref="product") for r in res["rows"] if r["name"].endswith("mean")]
        for split, n0 in (("test", "mal/lace_s0"), ("sparse", "mal/lace_sparse_s0")):
            if not any(r["split"] == split for r in rows):
                rows += [dict(r, lace_ref="product_s0") for r in res["rows"] if r["name"] == n0]
        return LG.evaluate(rows, gtbox=res["gtbox"]["test_paired_vs_lace"])
    return LG.evaluate(res["rows"])
