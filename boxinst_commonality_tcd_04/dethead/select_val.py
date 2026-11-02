"""Val decode selection (pre-registration 2): NMS {0.5, 0.6} x ranking {s', detector s}, full LACE pipeline (masker
lace4px_simple), score_lite crowd mask AP50 on the 108 val tiles. Highest wins; ties within 0.001 -> NMS 0.5, detector s.
Also records val recall by GT size (box, IoU 0.5) for the selected decode.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.select_val --tag fpn_s0 --head fpn
"""
import argparse, json, os, time
import numpy as np

from boxinst_commonality_tcd_04.dethead.diag_val import CFGS, FCC, HERE, VAL_GT, render


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--head", required=True); a = ap.parse_args()
    out = os.path.join(HERE, "results", f"select_val_{a.tag}.json"); assert not os.path.exists(out), out
    import torch
    from boxinst_commonality_tcd_04.dethead.test import load_model
    from boxinst_commonality_tcd_04.dethead.train import predict
    from boxinst_commonality_tcd_04.dethead.diag_box import analyse
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    vg = json.load(open(VAL_GT)); tids = sorted(vg); gd, t2i = SL.build_gt_crowd(vg, 2048)
    m = load_model(os.path.join(HERE, "runs", a.tag, "best.pt"), a.head); dev = torch.device("cpu")
    sets = {}
    for nms in (0.5, 0.6):
        S = {}
        for t in tids:
            b, s = predict(m, a.head, torch.from_numpy(np.load(f"{FCC}/feat_4p_val/{t}.npy")), dev, nms)
            S[t] = (b.numpy().astype(float).tolist(), s.numpy().astype(float).tolist())
        sets[f"nms{nms}"] = S
    R = render(sets, {k: CFGS[:1] for k in sets})
    rows = {}
    for k, S in sets.items():
        M = R[k]["e0.00_c0.33"]
        for rank in ("sprime", "detscore"):
            preds = {t: {"boxes_2048": S[t][0], "scores": M[t]["s_prime"] if rank == "sprime" else S[t][1], "masks_rle": M[t]["masks_rle"]} for t in tids}
            r = SL.score(preds, gd, t2i, masks=True, floor=0.0 if rank == "sprime" else 0.05)
            rows[f"{k}|{rank}"] = {"mask": {kk: r["mask"][kk] for kk in ("AP50", "AP75", "AP50_95")}, "box": {kk: r["box"][kk] for kk in ("AP50", "AP75")}}
            print(f"[val {a.tag}] {k} {rank}: {rows[f'{k}|{rank}']}", flush=True)
    pref = ["nms0.5|detscore", "nms0.5|sprime", "nms0.6|detscore", "nms0.6|sprime"]
    best = max(rows, key=lambda k: rows[k]["mask"]["AP50"])
    sel = next(k for k in pref if rows[best]["mask"]["AP50"] - rows[k]["mask"]["AP50"] <= 0.001)
    nms = float(sel.split("|")[0][3:])
    diag = analyse(a.tag, sets[sel.split("|")[0]], gd, t2i, 0.05)
    rec = {"rule": "highest val mask AP50; ties within 0.001 -> nms0.5, detscore", "rows": rows, "selected": sel, "nms": nms,
           "rank": sel.split("|")[1], "box_diag_selected_detscore": diag, "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(rec, open(out, "w"), indent=1); print(json.dumps({k: rec[k] for k in ("selected", "nms", "rank")}), flush=True)


if __name__ == "__main__":
    main()
