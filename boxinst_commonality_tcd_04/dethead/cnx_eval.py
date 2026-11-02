"""Pre-registration 3: val decode selection and the single test run for the ConvNeXt-L ATSS/GFL head.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.cnx_eval select --tag cnxl_s0
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.cnx_eval test --tag cnxl_s0
Test features are computed live (backbone on the test tile, never cached). Masker / scorer as test.py.
"""
import argparse, json, multiprocessing as mp, os, time
import numpy as np

from boxinst_commonality_tcd_04.dethead import cnx as C
from boxinst_commonality_tcd_04.dethead.diag_val import CFGS, VAL_GT, render
from boxinst_commonality_tcd_04.dethead.test import OUT, PKG, PUB_S0, REPO, _mask_job

HERE = os.path.join(PKG, "dethead"); FCC = C.FCC


def _head(tag, device):
    import torch
    h = C.CNXHead(); ck = torch.load(os.path.join(HERE, "runs", tag, "best.pt"), map_location="cpu", weights_only=False)
    h.load_state_dict(ck["state"]); return h.eval().to(device), ck["cfg"]


def select(tag):
    import torch
    out = os.path.join(OUT, f"select_val_{tag}.json"); assert not os.path.exists(out), out
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    from boxinst_commonality_tcd_04.dethead.diag_box import analyse
    vg = json.load(open(VAL_GT)); tids = sorted(vg); gd, t2i = SL.build_gt_crowd(vg, 2048)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); h, _ = _head(tag, dev)
    sets = {"nms0.5": {}, "nms0.6": {}}
    with torch.no_grad():
        for t in tids:
            outs = h([f.float()[None].to(dev) for f in C.load_feats(t)])
            for k, n in (("nms0.5", 0.5), ("nms0.6", 0.6)):
                b, s = C.decode(outs, n); sets[k][t] = (b.numpy().astype(float).tolist(), s.numpy().astype(float).tolist())
    R = render(sets, {k: CFGS[:1] for k in sets}); rows = {}
    for k, S in sets.items():
        M = R[k]["e0.00_c0.33"]
        for rank in ("sprime", "detscore"):
            preds = {t: {"boxes_2048": S[t][0], "scores": M[t]["s_prime"] if rank == "sprime" else S[t][1], "masks_rle": M[t]["masks_rle"]} for t in tids}
            r = SL.score(preds, gd, t2i, masks=True, floor=0.0 if rank == "sprime" else 0.05)
            rows[f"{k}|{rank}"] = {"mask": {kk: r["mask"][kk] for kk in ("AP50", "AP75", "AP50_95")}, "box": {kk: r["box"][kk] for kk in ("AP50", "AP75")}}
            print(f"[val {tag}] {k} {rank}: {rows[f'{k}|{rank}']}", flush=True)
    pref = ["nms0.5|detscore", "nms0.5|sprime", "nms0.6|detscore", "nms0.6|sprime"]
    best = max(rows, key=lambda k: rows[k]["mask"]["AP50"])
    sel = next(k for k in pref if rows[best]["mask"]["AP50"] - rows[k]["mask"]["AP50"] <= 0.001)
    diag = analyse(tag, sets[sel.split("|")[0]], gd, t2i, 0.05)
    rec = {"rule": "highest val mask AP50; ties within 0.001 -> nms0.5, detscore", "rows": rows, "selected": sel,
           "nms": float(sel.split("|")[0][3:]), "rank": sel.split("|")[1], "box_diag_selected_detscore": diag, "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(rec, open(out, "w"), indent=1); print(json.dumps({k: rec[k] for k in ("selected", "nms", "rank")}), flush=True)


def test(tag):
    import torch
    sel = json.load(open(os.path.join(OUT, f"select_val_{tag}.json"))); nms_iou = sel["nms"]
    otag = f"{tag}_nms{int(round(nms_iou * 10)):02d}"
    sfp = os.path.join(OUT, f"summary_{otag}.json"); assert not os.path.exists(sfp), sfp
    tids = sorted(json.load(open(PUB_S0))["preds"]); assert len(tids) == 439
    ddir, mdir = f"{FCC}/dethead_{otag}_det", f"{FCC}/dethead_{otag}_mask"; os.makedirs(ddir, exist_ok=True); os.makedirs(mdir, exist_ok=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    h, cfg = _head(tag, dev); bb = C.load_backbone(dev); t0 = time.time()
    with torch.no_grad():
        for k, t in enumerate(tids, 1):
            fp = os.path.join(ddir, t + ".json")
            if os.path.exists(fp):
                continue
            feats = C.extract(bb, C.rgb_test(t), dev)                     # live, never cached
            b, s = C.decode(h([f.float()[None].to(dev) for f in feats]), nms_iou)
            json.dump({"boxes_2048": b.numpy().astype(float).tolist(), "scores": s.numpy().astype(float).tolist()}, open(fp + ".tmp", "w"))
            os.replace(fp + ".tmp", fp)
            if k % 50 == 0:
                print(f"  det {k}/439 {time.time()-t0:.0f}s", flush=True)
    del bb
    D = {t: json.load(open(os.path.join(ddir, t + ".json"))) for t in tids}
    print(f"[{otag}] detections {sum(len(D[t]['scores']) for t in tids)}", flush=True)
    jobs = [(t, D[t]["boxes_2048"], D[t]["scores"], mdir) for t in tids if not os.path.exists(os.path.join(mdir, t + ".json"))]
    if jobs:
        with mp.get_context("spawn").Pool(12) as pool:
            for k, _ in enumerate(pool.imap_unordered(_mask_job, jobs), 1):
                if k % 100 == 0 or k == len(jobs):
                    print(f"  mask {k}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    M = {t: json.load(open(os.path.join(mdir, t + ".json"))) for t in tids}
    from boxinst_commonality_tcd_04 import score_coco as SC
    os.chdir(REPO); res = {"tag": otag, "head": "cnxl_atss_gfl", "nms": nms_iou, "val_selected": sel["selected"], "ckpt_cfg": cfg, "rows": {}}
    for rank, floor in (("sprime", 0.0), ("detscore", 0.05)):
        fp = os.path.join(OUT, f"preds_{otag}_{rank}_r2048.json")
        if not os.path.exists(fp):
            preds = {t: {"boxes_2048": D[t]["boxes_2048"], "scores": M[t]["s_prime"] if rank == "sprime" else D[t]["scores"], "masks_rle": M[t]["masks_rle"]} for t in tids}
            json.dump({"meta": {"mask_res": 2048, "model": f"ConvNeXt-L ATSS/GFL {otag} + LACE lace4px_simple, ranking {rank}"}, "preds": preds}, open(fp + ".tmp", "w"))
            os.replace(fp + ".tmp", fp)
        sp = fp.replace("preds_", "scored_")
        if not os.path.exists(sp):
            json.dump(SC.score_many([fp], os.path.join(PKG, "test_gt.json"), res=2048, max_dets=600, score_floor=floor)[0], open(sp, "w"))
        c = json.load(open(sp))["canopy_neutral_crowd"]
        res["rows"][rank] = {"mask": {k: c["mask"][k] for k in ("AP50", "AP75", "AP50_95", "n_det")}, "box": {k: c["box"][k] for k in ("AP50", "AP75", "AP50_95")}, "floor": floor}
    json.dump(res, open(sfp, "w"), indent=1, default=str); print(json.dumps(res["rows"], indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["select", "test"]); ap.add_argument("--tag", required=True)
    a = ap.parse_args(); select(a.tag) if a.cmd == "select" else test(a.tag)
