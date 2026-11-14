"""SAMEBOX: mask-shape-only effect at identical ranking. D1 boxes, 3-input hat s (no log p m) for every arm; arms differ only in mask:
LACE masks (T3 no_logpm), inscribed ellipse, filled box (BOXSHAPE-2). Paired tile bootstrap of LACE - ellipse, LACE - box.
Pre-registration SAMEBOX, dino_det/README.md (2026-11-13). CPU only.

GATE: each arm re-scores to boxshape_nm_{split}.json to 4 dp.

  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.samebox --split {test,sparse}
"""
import json, os
import numpy as np

from boxinst_commonality_tcd_04.dino_det import t1_boot as TB
from boxinst_commonality_tcd_04.dino_det.boxshape import R, RES

MET = {"AP50": lambda a: a[0], "AP75": lambda a: a[5], "AP50_95": lambda a: a.mean()}


def main(split):
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    out = os.path.join(R, f"samebox_{split}.json"); assert not os.path.exists(out), f"exists, one look only: {out}"
    _, _, floor = TB.ROWS[split]["LACE"]
    src = {"lace_masks": f"preds_dino_swinl_s0_{split}_t3_no_logpm_r2048.json",
           "ellipse": f"preds_boxshape_nm_ellipse_{split}_r2048.json", "box": f"preds_boxshape_nm_box_{split}_r2048.json"}
    ref = json.load(open(os.path.join(R, f"boxshape_nm_{split}.json")))["scored"]; refk = {"lace_masks": "lace_nologpm", "ellipse": "ellipse", "box": "box"}
    g = json.load(open(TB.GT[split])); tids = sorted(g); gd, t2i = SL.build_gt_crowd(g, RES); ids = np.array([t2i[t] for t in tids])
    res, D = {}, {}
    for k, f in src.items():
        P = json.load(open(os.path.join(R, f))); P = P.get("preds", P); pr = {t: P[t] for t in tids}
        s = SL.score(pr, gd, t2i, masks=True, floor=floor); res[k] = s["mask"]; print(f"[{split}] {k}: mask {s['mask']}", flush=True)
        for m in ("AP50", "AP75", "AP50_95"):
            if round(s["mask"][m], 4) != round(ref[refk[k]]["mask"][m], 4):
                raise SystemExit(f"GATE FAIL {k} {m}: {s['mask'][m]} vs {ref[refk[k]]['mask'][m]}")
        D[k] = TB.per_tile(pr, gd, t2i, floor)
        for i in ids:
            if i not in D[k]:
                D[k][i] = (np.zeros(0), np.zeros((10, 0), bool), np.zeros((10, 0), bool),
                           sum(1 for a in gd["annotations"] if a["image_id"] == i and not a["iscrowd"]))
    print(f"[{split}] gate PASS", flush=True)
    rng = np.random.default_rng(TB.SEED); diffs = {k: {m: [] for m in MET} for k in ("ellipse", "box")}
    for b in range(TB.N):
        s = ids[rng.integers(0, len(ids), len(ids))]; a = TB.ap_all(s, D["lace_masks"], SL.REC_THRS)
        for k in diffs:
            c = TB.ap_all(s, D[k], SL.REC_THRS)
            for m, fn in MET.items():
                diffs[k][m].append(fn(a) - fn(c))
        if (b + 1) % 200 == 0:
            print(f"[{split}] resample {b + 1}/{TB.N}", flush=True)
    rec = {"split": split, "scored": res, "src": src,
           "LACEmasks_minus": {k: {m: {"diff": round(res["lace_masks"][m] - res[k][m], 4),
               "ci95": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)],
               "p_le_0": round(float(np.mean(np.asarray(v) <= 0)), 4)} for m, v in d.items()} for k, d in diffs.items()},
           "note": "all arms: same D1 boxes, same 3-input hat s (no log p m) scores and order; only the mask differs. Paired tile bootstrap, test sampling only."}
    json.dump(rec, open(out, "w"), indent=1, default=float); print(json.dumps(rec["LACEmasks_minus"], indent=1), flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--split", choices=["test", "sparse"], required=True)
    main(p.parse_args().split)
