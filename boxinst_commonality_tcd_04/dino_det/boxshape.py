"""Geometric-masker control: LACE's deployed detections and ranking (D1 + edge fix + R3 = \\hat s), with each LACE mask replaced by
(a) the filled detector box or (b) the ellipse inscribed in it. Scores, boxes, floor and order are identical across arms, so any AP
difference is mask geometry alone. No \\hat s refit (ranking is held fixed by design). CPU only.
Pre-registration BOXSHAPE, dino_det/README.md (2026-11-11). NOT RUN until Tom greenlights.

GATE: the "lace" arm (unchanged masks) must reproduce scored_dino_swinl_s0_{split}_R3_ef_r2048.json mask AP50 / AP50:95 exactly (4 dp).
Outputs: results/preds_boxshape_{box,ellipse}_{split}_r2048.json (RLE at 2048), results/boxshape_{split}.json (AP + paired bootstrap
LACE − arm, 2,000 resamples, seed 0, via t1_boot.per_tile/ap_all).

  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.boxshape --split {test,sparse}
"""
import json, os, time
import numpy as np
from pycocotools import mask as maskUtils

from boxinst_commonality_tcd_04.dino_det import t1_boot as TB

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results"); RES = 2048; NV = 72


def _rle(poly):
    r = maskUtils.merge(maskUtils.frPyObjects([poly], RES, RES))
    r["counts"] = r["counts"].decode("ascii"); return r


def shape_masks(boxes, kind):
    out = []
    th = np.linspace(0, 2 * np.pi, NV, endpoint=False)
    for x0, y0, x1, y1 in np.clip(np.asarray(boxes, np.float64).reshape(-1, 4), 0, RES):
        if kind == "box":
            poly = [x0, y0, x1, y0, x1, y1, x0, y1]
        else:
            cx, cy, ax, ay = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
            poly = np.stack([cx + ax * np.cos(th), cy + ay * np.sin(th)], 1).ravel().tolist()
        out.append(_rle(poly))
    return out


def main(split):
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    out = os.path.join(R, f"boxshape_{split}.json"); assert not os.path.exists(out), f"exists, one look only: {out}"
    src, sfp, floor = TB.ROWS[split]["LACE"]
    P = json.load(open(src)); meta, P = P.get("meta", {}), P.get("preds", P)
    g = json.load(open(TB.GT[split])); tids = sorted(g); gd, t2i = SL.build_gt_crowd(g, RES); ids = np.array([t2i[t] for t in tids])
    arms = {"lace": {t: P[t] for t in tids if t in P}}
    for kind in ("box", "ellipse"):
        t0 = time.time()
        arms[kind] = {t: {"boxes_2048": rec["boxes_2048"], "scores": rec["scores"], "masks_rle": shape_masks(rec["boxes_2048"], kind)}
                      for t, rec in arms["lace"].items()}
        fp = os.path.join(R, f"preds_boxshape_{kind}_{split}_r2048.json")
        json.dump({"meta": {"src": src, "kind": kind, "src_meta": meta, "note": "LACE boxes + hat s scores, mask = " + kind}, "preds": arms[kind]}, open(fp, "w"))
        print(f"[{split}] rendered {kind} -> {fp} [{time.time() - t0:.0f}s]", flush=True)
    res, D = {}, {}
    for k, pr in arms.items():
        t0 = time.time(); s = SL.score(pr, gd, t2i, masks=True, floor=floor); res[k] = {"mask": s["mask"], "box": s["box"]}
        print(f"[{split}] {k}: mask {s['mask']} [{time.time() - t0:.0f}s]", flush=True)
        D[k] = TB.per_tile(pr, gd, t2i, floor)
        for i in ids:
            if i not in D[k]:
                D[k][i] = (np.zeros(0), np.zeros((10, 0), bool), np.zeros((10, 0), bool),
                           sum(1 for a in gd["annotations"] if a["image_id"] == i and not a["iscrowd"]))
    ref = json.load(open(sfp))["canopy_neutral_crowd"]["mask"]
    for m in ("AP50", "AP50_95"):
        if round(res["lace"]["mask"][m], 4) != ref[m]:
            raise SystemExit(f"GATE FAIL lace {m}: {res['lace']['mask'][m]} vs scored {ref[m]}")
    print(f"[{split}] gate PASS", flush=True)
    rng = np.random.default_rng(TB.SEED); diffs = {k: {"AP50": [], "AP50_95": []} for k in ("box", "ellipse")}
    for b in range(TB.N):
        s = ids[rng.integers(0, len(ids), len(ids))]; a = TB.ap_all(s, D["lace"], SL.REC_THRS)
        for k in diffs:
            c = TB.ap_all(s, D[k], SL.REC_THRS); diffs[k]["AP50"].append(a[0] - c[0]); diffs[k]["AP50_95"].append(a.mean() - c.mean())
        if (b + 1) % 200 == 0:
            print(f"[{split}] resample {b + 1}/{TB.N}", flush=True)
    rec = {"split": split, "scored": res, "LACE_minus": {k: {m: {"diff": round(res["lace"]["mask"][m] - res[k]["mask"][m], 4),
           "ci95": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)],
           "p_le_0": round(float(np.mean(np.asarray(v) <= 0)), 4)} for m, v in d.items()} for k, d in diffs.items()},
           "note": "same boxes, scores (hat s) and order in every arm; only the mask differs. Paired tile bootstrap, test sampling only."}
    json.dump(rec, open(out, "w"), indent=1, default=float); print(json.dumps(rec["LACE_minus"], indent=1), flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--split", choices=["test", "sparse"], required=True)
    main(p.parse_args().split)
