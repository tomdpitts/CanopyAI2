"""SAM 3 zero-shot masks on LACE (D1 variant) boxes (plan approved by Tom 2026-11-03; record in dino_det/SAM3_D1.md).
Boxes + detector scores s: frozen D1 preds (results/preds_dino_swinl_s0_{test,sparse}_detscore_r2048.json, s >= 0.05).
Prompting = Tier A1 exactly (tierA/a1_lace_sam3_persub.py: facebook/sam3 HF Sam3Tracker, no fine-tuning, local MPS, box on
the most central 1024 subtile containing it else whole 2048 tile, multimask off, pasted at 2048). Per-tile JSON checkpoints
in feat_cache/sam3_d1_<split>/ (resume by listing); sparse tiles whose boxes equal the test tile's are copied, not re-run.
Rows: SAM's own ranking sqrt(s*SAM) (floor 0) and detector s (floor 0.05). Never LACE's s'.

  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.sam3_on_d1 parity          # A1 reproduction gate
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.sam3_on_d1 run --split test [--limit N]
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.sam3_on_d1 score --split test
"""
import argparse, contextlib, io, json, os, shutil, time
import numpy as np
from PIL import Image
from pycocotools import mask as mu
from boxinst_commonality_tcd_04.native_raster.tierA import a1_lace_sam3_persub as A1

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); REPO = os.path.dirname(PKG)
RES = os.path.join(HERE, "results"); OUT = os.path.join(RES, "sam3_d1"); FC = "/Users/tompitts/dphil/feat_cache"
SPLIT = {"test": dict(n=439, rgb=os.path.join(REPO, "data", "tcd", "test"), gt=os.path.join(PKG, "test_gt.json")),
         "sparse": dict(n=236, rgb=os.path.join(REPO, "data", "tcd_sparse", "test"), gt=os.path.join(REPO, "data", "tcd_sparse", "sparse_gt.json"))}
N_BOOT, SEED = 2000, 0


def det(split):
    return json.load(open(os.path.join(RES, f"preds_dino_swinl_s0_{split}_detscore_r2048.json")))["preds"]


def sam_tile(sam, proc, rgb, B):
    rles, sams, where = [None] * len(B), [None] * len(B), [None] * len(B)
    for key, idx in A1.assign(B).items():
        if key is None:
            ms, sc = A1.prompt(sam, proc, rgb, B[idx])
        else:
            oy, ox = key
            ms, sc = A1.prompt(sam, proc, rgb.crop((ox, oy, ox + A1.SUB, oy + A1.SUB)), B[idx] - np.array([ox, oy, ox, oy]))
        for i, m, s in zip(idx, ms, sc):
            if key is None:
                full = m.astype(np.uint8)
            else:
                full = np.zeros((2048, 2048), np.uint8); full[oy:oy + A1.SUB, ox:ox + A1.SUB] = m
            rles[i] = {"size": [2048, 2048], "counts": mu.encode(np.asfortranarray(full))["counts"].decode("ascii")}
            sams[i] = s; where[i] = list(key) if key else None
    return rles, sams, where


def env_json(path):
    import platform, torch, transformers, PIL
    if not os.path.exists(path):
        json.dump({"device": A1.DEV, "machine": platform.machine(), "platform": platform.platform(), "python": platform.python_version(),
                   "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
                   "pillow": PIL.__version__, "started": time.strftime("%Y-%m-%d %H:%M")}, open(path, "w"), indent=1)


def parity():
    """Gate: re-prompt A1 s0's stored boxes on 2 tiles; masks must match the stored A1 masks (IoU) and SAM scores."""
    sam, proc = A1.load_base(); d = os.path.join(PKG, "native_raster", "tierA", "a1_tiles_s0"); out = {}
    for t in ("tcd_val_tile_0", "tcd_val_tile_1"):
        r = json.load(open(os.path.join(d, t + ".json"))); B = np.asarray(r["boxes_2048"], np.float64).reshape(-1, 4)
        rgb = Image.open(SPLIT["test"]["rgb"] + f"/{t}.tif").convert("RGB")
        rles, sams, _ = sam_tile(sam, proc, rgb, B)
        iou = mu.iou(rles, r["masks_rle"], [0] * len(rles)).diagonal()
        ds = np.abs(np.asarray(sams) - np.asarray(r["sam_scores"]))
        out[t] = {"n": len(B), "iou_min": round(float(iou.min()), 4), "iou_mean": round(float(iou.mean()), 5),
                  "frac_iou_ge_0.99": round(float((iou >= 0.99).mean()), 4), "sam_score_absdiff_max": round(float(ds.max()), 4)}
        print(t, out[t], flush=True)
    out["pass"] = all(v["iou_mean"] >= 0.99 for k, v in out.items() if k != "pass")
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, "parity_vs_a1_s0.json"), "w"), indent=1)
    print("[parity]", "PASS" if out["pass"] else "FAIL", flush=True)


def run(split, limit):
    D = det(split); tids = sorted(D); assert len(tids) == SPLIT[split]["n"]
    tdir = os.path.join(FC, f"sam3_d1_{split}"); os.makedirs(tdir, exist_ok=True); env_json(os.path.join(tdir, "_env.json"))
    if split == "sparse":  # reuse test tiles whose D1 boxes are identical
        T, tt = det("test"), os.path.join(FC, "sam3_d1_test")
        for t in tids:
            src = os.path.join(tt, t + ".json")
            if t in T and T[t]["boxes_2048"] == D[t]["boxes_2048"] and os.path.exists(src) and not os.path.exists(os.path.join(tdir, t + ".json")):
                shutil.copy(src, os.path.join(tdir, t + ".json")); print(f"  reuse test {t}", flush=True)
    done = {f[:-5] for f in os.listdir(tdir) if f.endswith(".json") and not f.startswith("_")}
    todo = [t for t in tids if t not in done][:limit]
    nb = sum(len(D[t]["scores"]) for t in todo)
    print(f"[sam3_d1 {split}] {len(tids)} tiles, {len(done)} done, {len(todo)} to do ({nb} boxes)", flush=True)
    sam, proc = A1.load_base(); t0 = time.time(); nd = 0
    for n, t in enumerate(todo):
        B = np.asarray(D[t]["boxes_2048"], np.float64).reshape(-1, 4)
        rles, sams, where = ([], [], [])
        if len(B):
            rgb = Image.open(os.path.join(SPLIT[split]["rgb"], t + ".tif")).convert("RGB"); assert rgb.size == (2048, 2048)
            rles, sams, where = sam_tile(sam, proc, rgb, B)
        rec = {"boxes_2048": D[t]["boxes_2048"], "scores": D[t]["scores"], "sam_scores": sams, "masks_rle": rles,
               "subtile": where, "device": A1.DEV}
        tmp = os.path.join(tdir, t + ".json.tmp"); json.dump(rec, open(tmp, "w")); os.replace(tmp, os.path.join(tdir, t + ".json"))
        nd += len(B); el = time.time() - t0
        print(f"  {len(done)+n+1}/{len(tids)} {t} {len(B)} boxes {el:.0f}s {nd/el:.1f} box/s eta {el/nd*(nb-nd):.0f}s", flush=True)
    print(f"[sam3_d1 {split}] done", flush=True)


# ---- scoring + paired bootstrap (AP50 and AP50:95; per-tile COCOeval statuses, as dino_det/boot_vs_bt.py) ----
def per_tile(preds, gd, t2i, floor):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    dets = SL.build_dt(preds, t2i, 2048, floor, with_masks=True)
    cg = COCO(); cg.dataset = gd; cg.createIndex()
    with contextlib.redirect_stdout(io.StringIO()):
        cd = cg.loadRes([dict(d) for d in dets]); ev = COCOeval(cg, cd, "segm")
        ev.params.maxDets = [600]; ev.params.areaRng = [[0, 1e10]]; ev.params.areaRngLbl = ["all"]; ev.evaluate()
    out = {}
    for e in ev.evalImgs:
        if e is None:
            continue
        # per IoU threshold: dets matched to crowd (canopy) are ignored at that threshold only, as COCOeval accumulate
        out[e["image_id"]] = (np.asarray(e["dtScores"]), np.asarray(e["dtMatches"]) > 0,
                              ~np.asarray(e["dtIgnore"], bool), int((~np.asarray(e["gtIgnore"], bool)).sum()))
    return out


def ap_at(tiles, data, rec_thrs, k):
    sc = np.concatenate([data[t][0][data[t][2][k]] for t in tiles]); tp = np.concatenate([data[t][1][k][data[t][2][k]] for t in tiles])
    npig = sum(data[t][3] for t in tiles)
    o = np.argsort(-sc, kind="mergesort"); tp = tp[o]
    tpc = np.cumsum(tp); fpc = np.cumsum(~tp); rc = tpc / max(npig, 1); pr = tpc / np.maximum(tpc + fpc, np.spacing(1))
    pr = np.maximum.accumulate(pr[::-1])[::-1] if len(pr) else pr
    idx = np.searchsorted(rc, rec_thrs, side="left"); q = np.where(idx < len(pr), pr[np.minimum(idx, max(len(pr) - 1, 0))] if len(pr) else 0.0, 0.0)
    return float(q.mean())


def aps(tiles, data, rec_thrs):
    a = [ap_at(tiles, data, rec_thrs, k) for k in range(10)]
    return a[0], float(np.mean(a))


def score(split):
    from boxinst_commonality_tcd_04 import score_coco as SC
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    D = det(split); tids = sorted(D); tdir = os.path.join(FC, f"sam3_d1_{split}"); gtp = SPLIT[split]["gt"]
    rows = {"sam3zs_gmean": (lambda r: np.sqrt(np.asarray(r["scores"], float) * np.clip(np.asarray(r["sam_scores"], float), 0, None)).round(6), 0.0, "sqrt(s*SAM)"),
            "sam3zs_det": (lambda r: np.asarray(r["scores"], float), 0.05, "D1 detector s")}
    files = {}
    for name, (f, floor, rank) in rows.items():
        fp = os.path.join(OUT, f"preds_d1_{name}_{split}_r2048.json")
        if not os.path.exists(fp):
            preds = {}
            for t in tids:
                r = json.load(open(os.path.join(tdir, t + ".json"))); sc = f(r)
                assert r["boxes_2048"] == D[t]["boxes_2048"] and len(sc) == len(r["masks_rle"]) and np.all(np.isfinite(sc)), t
                preds[t] = {"boxes_2048": r["boxes_2048"], "scores": [float(x) for x in sc], "masks_rle": r["masks_rle"]}
            json.dump({"meta": {"mask_res": 2048, "model": "LACE (D1 variant) boxes + SAM 3 zero-shot (A1 prompting, local MPS)", "rank": rank},
                       "preds": preds}, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
        sp = fp.replace("preds_", "scored_")
        if not os.path.exists(sp):
            cwd = os.getcwd(); os.chdir(REPO)
            json.dump(SC.score_many([fp], gtp, res=2048, max_dets=600, score_floor=floor)[0], open(sp, "w")); os.chdir(cwd)
        files[name] = (fp, floor)
    files["lace_d1_sprime"] = (os.path.join(RES, f"preds_dino_swinl_s0_{split}_sprime_r2048.json"), 0.0)
    rec = {"split": split, "n_boot": N_BOOT, "seed": SEED, "full": {}, "lace_d1_minus": {},
           "note": "paired tile bootstrap; test-set sampling only, single detector seed; LACE row = frozen D1 s' (0.7035 test), not later refinements"}
    for k in files:
        sp = files[k][0].replace("preds_", "scored_"); c = json.load(open(sp))["canopy_neutral_crowd"]
        rec["full"][k] = {"mask": {m: c["mask"][m] for m in ("AP50", "AP75", "AP50_95")}, "box_AP50": c["box"]["AP50"]}
    g = json.load(open(gtp)); gd, t2i = SL.build_gt_crowd(g, 2048); ids = np.array([t2i[t] for t in sorted(g)])
    S = {}
    for k, (fp, floor) in files.items():
        P = json.load(open(fp)); P = P.get("preds", P); S[k] = per_tile({t: P[t] for t in sorted(g)}, gd, t2i, floor)
        for i in ids:
            if i not in S[k]:
                ng = sum(1 for a in gd["annotations"] if a["image_id"] == i and not a["iscrowd"])
                S[k][i] = (np.zeros(0), np.zeros((10, 0), bool), np.zeros((10, 0), bool), ng)
        a50, a5095 = aps(ids, S[k], SL.REC_THRS)
        rec["full"][k]["boot_reproduces"] = {"AP50": round(a50, 4), "AP50_95": round(a5095, 4)}
        print(f"[{split} {k}] scorer {rec['full'][k]['mask']}  bootstrap-full AP50 {a50:.4f} AP50:95 {a5095:.4f}", flush=True)
    rng = np.random.default_rng(SEED); diffs = {k: [] for k in files if k != "lace_d1_sprime"}; t0 = time.time()
    full = {k: aps(ids, S[k], SL.REC_THRS) for k in files}
    for b in range(N_BOOT):
        s = ids[rng.integers(0, len(ids), len(ids))]; L = aps(s, S["lace_d1_sprime"], SL.REC_THRS)
        for k in diffs:
            o = aps(s, S[k], SL.REC_THRS); diffs[k].append((L[0] - o[0], L[1] - o[1]))
        if (b + 1) % 200 == 0:
            print(f"  boot {b+1}/{N_BOOT} {time.time()-t0:.0f}s", flush=True)
    for k, v in diffs.items():
        v = np.asarray(v)
        rec["lace_d1_minus"][k] = {m: {"diff": round(full["lace_d1_sprime"][j] - full[k][j], 4),
                                       "ci95": [round(float(np.percentile(v[:, j], 2.5)), 4), round(float(np.percentile(v[:, j], 97.5)), 4)],
                                       "p_le_0": round(float(np.mean(v[:, j] <= 0)), 4)} for j, m in enumerate(("AP50", "AP50_95"))}
    json.dump(rec, open(os.path.join(OUT, f"summary_sam3_d1_{split}.json"), "w"), indent=1); print(json.dumps(rec, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["parity", "run", "score"])
    ap.add_argument("--split", choices=["test", "sparse"], default="test"); ap.add_argument("--limit", type=int, default=None); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"parity": parity, "run": lambda: run(a.split, a.limit), "score": lambda: score(a.split)}[a.cmd]()


if __name__ == "__main__":
    main()
