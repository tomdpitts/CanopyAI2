"""Soft box on the 108 val tiles at 2048 (tau 0.40): does freeing the mask from the box clip
raise mask AP50? Same harness as oracle_boxes_val.py (seed-0 detector, deployed masker,
detector-score ranking, score_coco crowd arm, floor 0).

Arms: deployed (E.pred_instance_masks) + soft(margin, lam) for each grid cell. Each arm is
scored on ALL 108 tiles and on two fixed halves (seed-0 permutation): the config is SELECTED
on half A and REPORTED on half B, so the reported gain is not inflated by the choice.

GATE: the deployed arm must reproduce val_knobs_tau_r2048_s0_local.json cell 0.40 (0.6045).
softbox.py's margin=0 path is separately proven bit-identical to the deployed masks.

    .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.softbox_val [--limit 20]
"""
from __future__ import annotations

import argparse, json, os, platform, tempfile, time

import numpy as np
import torch
from pycocotools import mask as maskUtils

from boxinst_commonality_tcd_04 import evaluate as E
from boxinst_commonality_tcd_04 import score_coco as S
from boxinst_commonality_tcd_04.detector import STRIDE8
from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.phase4_lib_tcd import Detector4Phase
from boxinst_commonality_tcd_04.native_raster import oracle_boxes_val as O
from boxinst_commonality_tcd_04.native_raster import softbox as SB
from dapt.decode import decode

HERE = os.path.dirname(os.path.abspath(__file__))
RES, TAU, PW, KS = 2048, 0.40, 0.30, 1.60
GRID = [(0.15, 0.10), (0.15, 0.30), (0.30, 0.10), (0.30, 0.30)]      # (margin, lam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--grid", default=None, help="e.g. '0.2:0.2,0.3:0.1' to override GRID")
    a = ap.parse_args()
    grid = GRID if not a.grid else [tuple(float(x) for x in c.split(":")) for c in a.grid.split(",")]
    dst = os.path.join(HERE, f"softbox_val_r{RES}{'_lim%d' % a.limit if a.limit else ''}.json")
    gt = json.load(open(O.VAL_GT))
    tiles = sorted(t for t in gt if os.path.exists(os.path.join(O.FEAT_DIR, t + ".npy")))
    if a.limit:
        tiles = tiles[:a.limit]
    gt = {t: gt[t] for t in tiles}
    perm = np.random.default_rng(0).permutation(len(tiles))
    half = {"A": sorted(tiles[i] for i in perm[:len(tiles) // 2]),
            "B": sorted(tiles[i] for i in perm[len(tiles) // 2:])}

    out = {"res": RES, "tau": TAU, "prior_weight": PW, "kappa_scale": KS, "n_val_tiles": len(tiles),
           "ranking": "detector score s", "halves": {k: len(v) for k, v in half.items()},
           "environment": {"platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__},
           "arms": {}}
    if os.path.exists(dst):
        prev = json.load(open(dst))
        assert prev.get("n_val_tiles") == len(tiles) and prev.get("tau") == TAU, f"{dst}: config mismatch"
        out["arms"] = prev.get("arms", {}); print(f"[softbox] resuming: {list(out['arms'])}", flush=True)

    ck = torch.load(O.DET_PATH, map_location="cpu", weights_only=False); cfg = ck["cfg"]
    model = Detector4Phase(cfg["in_dim"], width=cfg["width"], tower=cfg["tower"]).eval()
    model.load_state_dict(ck["state"]); masker = E.TCDMasker(O.EM_PATH)
    boxes, scores, zns, gs = {}, {}, {}, {}
    t0 = time.time()
    for k, tid in enumerate(tiles):
        feat = np.load(os.path.join(O.FEAT_DIR, tid + ".npy")).astype(np.float32)
        with torch.no_grad():
            bx, sc = decode(model(torch.from_numpy(feat)[None]), score_thr=0.05, stride=STRIDE8, topk=600)
        boxes[tid], scores[tid] = bx.numpy(), sc.numpy()
        zns[tid], gs[tid] = masker.project(feat), feat.shape[-1]
    n_det = int(sum(len(v) for v in boxes.values()))
    print(f"[softbox] detector: {n_det} dets in {time.time()-t0:.0f}s", flush=True)
    if not a.limit:
        assert n_det == 40058, n_det

    ctx = {}
    for name, ts in (("all", tiles), ("A", half["A"]), ("B", half["B"])):
        g_ = {t: gt[t] for t in ts}
        fp, t2i, tr, cu = S.build_gt(g_, RES, with_canopy=False)
        cr, _, _, _ = S.build_gt(g_, RES, with_canopy=True)
        ctx[name] = (fp, cr, t2i, tr, cu, ts)
    print("[softbox] GT rasterised", flush=True)

    tmp = os.path.join(tempfile.gettempdir(), "softbox_cell.json")
    arms = [("deployed", None)] + [(f"soft m{m:.2f} lam{l:.2f}", (m, l)) for m, l in grid]
    for name, cfgm in arms:
        if name in out["arms"]:
            print(f"  {name}: cached", flush=True); continue
        tc = time.time(); preds = {}; grow = []
        for tid in tiles:
            bx, sc = boxes[tid], scores[tid]
            if not len(bx):
                preds[tid] = {"boxes_2048": [], "scores": [], "canopy_ignore": [], "masks_rle": []}; continue
            if cfgm is None:
                pm = E.pred_instance_masks(masker, zns[tid], gs[tid], bx, res=RES, scale=1.0, mask_thr=TAU,
                                           prior_weight=PW, kappa_scale=KS)
            else:
                pm = SB.soft_instance_masks(masker, zns[tid], gs[tid], bx, RES, 1.0, TAU, PW, KS,
                                            margin=cfgm[0], lam=cfgm[1], clip="window")
            preds[tid] = {"boxes_2048": bx.tolist(), "scores": sc.tolist(), "canopy_ignore": [False] * len(bx),
                          "masks_rle": [{"size": [RES, RES], "counts": maskUtils.encode(
                              np.asfortranarray(m.astype(np.uint8)))["counts"].decode("ascii")} for m in pm]}
        json.dump({"meta": {"mask_res": RES}, "preds": preds}, open(tmp, "w"))
        rec = {"margin": cfgm and cfgm[0], "lam": cfgm and cfgm[1]}
        for part in ("all", "A", "B"):
            r = S._score_one(tmp, ctx[part], RES, S.MAX_DETS, 0.0)["canopy_neutral_crowd"]["mask"]
            rec[part] = {k: r[k] for k in ("AP50", "AP75", "AP50_95")}
        rec["secs"] = round(time.time() - tc, 1)
        out["arms"][name] = rec
        if name == "deployed" and not a.limit:
            ref = json.load(open(os.path.join(HERE, "val_knobs_tau_r2048_s0_local.json")))["cells"]["0.40"]["AP50"]
            out["gate"] = {"ref_AP50": ref, "got_AP50": rec["all"]["AP50"], "pass": abs(rec["all"]["AP50"] - ref) <= 1e-4}
            print(f"  GATE deployed AP50 {rec['all']['AP50']} vs {ref} -> {'PASS' if out['gate']['pass'] else 'FAIL'}", flush=True)
            if not out["gate"]["pass"]:
                json.dump(out, open(dst, "w"), indent=2); raise SystemExit("gate failed")
        json.dump(out, open(dst, "w"), indent=2)
        print(f"  {name}: AP50 all {rec['all']['AP50']:.4f} | A {rec['A']['AP50']:.4f} | B {rec['B']['AP50']:.4f}"
              f"   AP75 all {rec['all']['AP75']:.4f}   ({rec['secs']:.0f}s) [saved]", flush=True)

    dep = out["arms"]["deployed"]
    soft = {k: v for k, v in out["arms"].items() if k != "deployed"}
    if soft:
        pick = max(soft, key=lambda k: soft[k]["A"]["AP50"])
        out["selection"] = {"selected_on": "half A AP50", "config": pick,
                            "half_B_delta_AP50": round(soft[pick]["B"]["AP50"] - dep["B"]["AP50"], 4),
                            "half_B_delta_AP75": round(soft[pick]["B"]["AP75"] - dep["B"]["AP75"], 4),
                            "all_delta_AP50": round(soft[pick]["all"]["AP50"] - dep["all"]["AP50"], 4)}
        print("[softbox] selection:", out["selection"], flush=True)
    json.dump(out, open(dst, "w"), indent=2); print("->", dst, flush=True)


if __name__ == "__main__":
    main()
