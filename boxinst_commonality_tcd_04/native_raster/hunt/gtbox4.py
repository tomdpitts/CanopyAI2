"""tab:gtbox LACE row for a frozen hunt render spec: masks prompted with the ground-truth crown boxes, 439 test tiles.

Same protocol as guided_ablate_modal.gtbox_guided (the published LACE row): boxes = gtbox_lib.gt_boxes_2048 (polygon
extents), GT RLE at 2048 and validity (>= 4 px at the 512 reference) from gtbox_lib.gt_rles_and_valid, per-crown IoU
(pairwise, gtbox_lib.per_crown_iou_rle), summary over valid crowns (gtbox_lib.summarise). Render = hunt/render.py through
test_eval.render_tile (same ctx: 8 px Modal test features + local 4 px hunt_z4p_test); no ranking.
Per-tile checkpoints (iou unrounded, valid, double, masks_rle) in feat_cache/hunt_gtbox4_<name>/; summary -> hunt/test/gtbox4_<name>.json.
GATE: spec 'deployed' on the first 2 GT tiles reproduces native_raster/gtbox_guided_raw/gtbox_lace_guided_r2048.jsonl
per-crown IoU (5 dp) and validity exactly, and that file's summary is the published 0.748 / 0.940 / 0.647 / 0.051.

  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.hunt.gtbox4 FROZEN.json NAME
"""
import json, multiprocessing as mp, os, sys, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

from boxinst_commonality_tcd_04.native_raster.hunt import test_eval as TE

HERE = TE.HERE; NR = TE.NR; PKG = TE.PKG; FC = TE.FC
_G = {}


def _init(spec):
    TE._init(spec, "test")
    _G["gt"] = json.load(open(os.path.join(PKG, "test_gt.json")))


def tile(t):
    from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as L
    polys = _G["gt"][t]["trees"]
    boxes = L.gt_boxes_2048(polys)
    gt_rles, valid = L.gt_rles_and_valid(polys, res=2048)
    ms, _ = TE.render_tile(t, boxes.tolist())
    pr = [{"size": r["size"], "counts": r["counts"].encode()} for r in ms]
    return {"tile": t, "iou": L.per_crown_iou_rle(pr, gt_rles).tolist(), "valid": valid.tolist(),
            "double": L.double_assignment_rle(pr), "masks_rle": ms}


def _job(args):
    t, d = args
    fp = os.path.join(d, t + ".json")
    if not os.path.exists(fp):
        json.dump(tile(t), open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    return t


def summary(recs):
    from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as L
    iou = np.concatenate([np.asarray(r["iou"])[np.asarray(r["valid"], bool)] for r in recs if r["iou"]])
    return L.summarise(iou)


def gate():
    ref = {}
    for line in open(os.path.join(NR, "gtbox_guided_raw", "gtbox_lace_guided_r2048.jsonl")):
        r = json.loads(line); ref[r["tile"]] = r
    s = summary(list(ref.values()))
    ok_sum = (round(s["mean_iou"], 3), round(s["rates"][">=0.5"], 3), round(s["rates"][">=0.75"], 3), round(s["rates"][">=0.9"], 3)) == (0.748, 0.940, 0.647, 0.051)
    print(f"[gate] published guided jsonl summary {s} -> {'PASS' if ok_sum else 'FAIL'}", flush=True)
    _init({"kind": "deployed"})
    tiles = [t for t in sorted(_G["gt"]) if _G["gt"][t]["trees"]][:2]; ok = True
    for t in tiles:
        o = tile(t)
        eq = np.round(o["iou"], 5).tolist() == ref[t]["iou"] and o["valid"] == ref[t]["valid"]
        print(f"[gate] deployed spec {t}: {len(o['iou'])} crowns, iou/valid identical to published: {eq}", flush=True); ok &= eq
    return ok and ok_sum, s


def main():
    spec_fp, name = sys.argv[1], sys.argv[2]
    spec = json.load(open(spec_fp)); spec = {k: v for k, v in spec.get("spec", spec).items() if k != "rank"}
    out = os.path.join(HERE, "test", f"gtbox4_{name}.json"); assert not os.path.exists(out), out
    ok, ref_sum = gate(); assert ok, "GATE FAILED"
    gt = json.load(open(os.path.join(PKG, "test_gt.json")))
    tiles = [t for t in sorted(gt) if gt[t]["trees"]]
    d = f"{FC}/hunt_gtbox4_{name}"; os.makedirs(d, exist_ok=True)
    todo = [(t, d) for t in tiles if not os.path.exists(os.path.join(d, t + ".json"))]
    print(f"[gtbox4 {name}] {len(todo)}/{len(tiles)} tiles to render, spec {spec}", flush=True); t0 = time.time()
    if todo:
        with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", 12)), initializer=_init, initargs=(spec,)) as pool:
            for k, _ in enumerate(pool.imap_unordered(_job, todo), 1):
                if k % 20 == 0 or k == len(todo):
                    print(f"  {k}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    recs = [json.load(open(os.path.join(d, t + ".json"))) for t in tiles]
    s = summary(recs)
    json.dump({"name": name, "spec": spec, "summary_valid_crowns": s, "reference_deployed_guided": ref_sum,
               "n_crowns_total": int(sum(len(r["iou"]) for r in recs)), "tiles_dir": d}, open(out, "w"), indent=1)
    print(f"[gtbox4 {name}] {s} -> {out}", flush=True)


if __name__ == "__main__":
    main()
