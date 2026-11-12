"""BOXSHAPE-2: geometric-masker control without access to the masker's ranking signal. Same D1 boxes as deployed LACE; filled-box and
inscribed-ellipse masks ranked by the 3-input \\hat s refit WITHOUT log(p̄m) (s, log h, IoU_up; T3 "no_logpm" scores), since a
geometric masker has no posterior. Compared against deployed LACE (4-input \\hat s, LACE masks). Supersedes BOXSHAPE for tab:masker.
Pre-registration BOXSHAPE-2, dino_det/README.md (2026-11-11). CPU only.

GATES: (1) the T3 no_logpm preds have boxes identical and in the same order as deployed (checked 2026-11-11, both splits);
(2) no_logpm preds with LACE masks reproduce t3_abl.json rows.no_logpm mask AP50 / AP50:95 to 4 dp;
(3) deployed LACE reproduces scored_dino_swinl_s0_{split}_R3_ef_r2048.json to 4 dp.

  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.boxshape_nm --split {test,sparse}
"""
import json, os
import numpy as np

from boxinst_commonality_tcd_04.dino_det import t1_boot as TB
from boxinst_commonality_tcd_04.dino_det.boxshape import shape_masks, R, RES


def main(split):
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    out = os.path.join(R, f"boxshape_nm_{split}.json"); assert not os.path.exists(out), f"exists, one look only: {out}"
    src, sfp, floor = TB.ROWS[split]["LACE"]
    nm_src = os.path.join(R, f"preds_dino_swinl_s0_{split}_t3_no_logpm_r2048.json")
    A = json.load(open(src))["preds"]; B = json.load(open(nm_src)); B = B.get("preds", B)
    g = json.load(open(TB.GT[split])); tids = sorted(g); gd, t2i = SL.build_gt_crowd(g, RES); ids = np.array([t2i[t] for t in tids])
    for t in tids:  # gate 1
        assert np.allclose(np.asarray(A[t]["boxes_2048"]), np.asarray(B[t]["boxes_2048"])), t
    arms = {"lace": {t: A[t] for t in tids}, "lace_nologpm": {t: B[t] for t in tids}}
    for kind in ("box", "ellipse"):
        arms[kind] = {t: {"boxes_2048": B[t]["boxes_2048"], "scores": B[t]["scores"], "masks_rle": shape_masks(B[t]["boxes_2048"], kind)} for t in tids}
        json.dump({"meta": {"src_boxes_scores": nm_src, "kind": kind, "note": "D1 boxes, 3-input hat s (no log p m), mask = " + kind},
                   "preds": arms[kind]}, open(os.path.join(R, f"preds_boxshape_nm_{kind}_{split}_r2048.json"), "w"))
    res, D = {}, {}
    for k, pr in arms.items():
        s = SL.score(pr, gd, t2i, masks=True, floor=floor); res[k] = {"mask": s["mask"], "box": s["box"]}
        print(f"[{split}] {k}: mask {s['mask']}", flush=True)
        if k in ("lace", "box", "ellipse"):
            D[k] = TB.per_tile(pr, gd, t2i, floor)
            for i in ids:
                if i not in D[k]:
                    D[k][i] = (np.zeros(0), np.zeros((10, 0), bool), np.zeros((10, 0), bool),
                               sum(1 for a in gd["annotations"] if a["image_id"] == i and not a["iscrowd"]))
    ref3 = json.load(open(sfp))["canopy_neutral_crowd"]["mask"]; ref2 = json.load(open(os.path.join(R, "t3_abl.json")))[split]["rows"]["no_logpm"]["mask"]
    for m in ("AP50", "AP50_95"):
        if res["lace"]["mask"][m] != ref3[m]:
            raise SystemExit(f"GATE 3 FAIL {m}: {res['lace']['mask'][m]} vs {ref3[m]}")
        if res["lace_nologpm"]["mask"][m] != ref2[m]:
            raise SystemExit(f"GATE 2 FAIL {m}: {res['lace_nologpm']['mask'][m]} vs {ref2[m]}")
    print(f"[{split}] gates PASS", flush=True)
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
           "note": "geometric arms ranked by 3-input hat s (no log p m); LACE = deployed 4-input hat s. Paired tile bootstrap, test sampling only."}
    json.dump(rec, open(out, "w"), indent=1, default=float); print(json.dumps(rec["LACE_minus"], indent=1), flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--split", choices=["test", "sparse"], required=True)
    main(p.parse_args().split)
