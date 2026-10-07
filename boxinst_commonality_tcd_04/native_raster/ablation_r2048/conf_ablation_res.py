"""tab:factors (confidence/ablation.py) at an arbitrary scoring raster, without editing it.

The published script hardcodes RES=512, its cache path, and knobbed_s0.json as the mask
source. This wrapper overrides exactly those three things on the imported module and then
calls its unmodified main(). Everything else -- features (em_diag_s0.json), the val split
(GT polygon extents, raster-independent), the fitting, the bootstrap -- is the published code.

Masks: at 512 the published knobbed_s0.json; at 2048 native_raster/
preds_lace_s0_r2048_tau040_detscore.json, whose boxes/scores/order are byte-identical to
knobbed_s0.json (rescore_detector.py gate), so em_diag_s0.json still aligns box-for-box.

Gate: --res 512 must reproduce confidence/_ap_cache_s0.pkl array-for-array and
confidence/ablation_results.json value-for-value.

  python -m boxinst_commonality_tcd_04.native_raster.ablation_r2048.conf_ablation_res \
      --res 512|2048 --out <json> [--full]
"""
import json, os, pickle, sys

import numpy as np

from boxinst_commonality_tcd_04 import evaluate as E
from boxinst_commonality_tcd_04.confidence import ablation as A

HERE = os.path.dirname(os.path.abspath(__file__))
NR = os.path.dirname(HERE)
MASKS = {512: os.path.join(A.P4, "preds", "knobbed_s0.json"),
         2048: os.path.join(NR, "preds_lace_s0_r2048_tau040_detscore.json")}


def build_cache_res(res, preds_path, cache_path):
    from pycocotools import mask as M
    scale = 2048.0 / res
    gt = json.load(open(os.path.join(A.PKG, "test_gt.json")))
    P = json.load(open(preds_path))["preds"]
    D, n_gt = {}, 0
    for i, t in enumerate(sorted(gt)):
        trs = [M.encode(np.asfortranarray(m.astype(np.uint8)))
               for m in E.raster(gt[t]["trees"], res=res, scale=scale)]
        n_gt += len(trs)
        r = P.get(t)
        if not r or not r["scores"]:
            continue
        rl = [{"size": x["size"], "counts": x["counts"].encode()} for x in r["masks_rle"]]
        assert all(tuple(x["size"]) == (res, res) for x in rl), f"{t}: mask size != {res}"
        can = [M.encode(np.asfortranarray(m.astype(np.uint8)))
               for m in E.raster(gt[t]["canopy"], res=res, scale=scale)]
        canu = (M.merge(can) if can
                else M.encode(np.asfortranarray(np.zeros((res, res), np.uint8))))
        ar = np.asarray([float(M.area(x)) for x in rl])
        inter = np.asarray([float(M.area(M.merge([x, canu], intersect=1))) for x in rl])
        iou = (np.asarray(M.iou(rl, trs, [0] * len(trs)), np.float64)
               .reshape(len(rl), len(trs)) if trs else np.zeros((len(rl), 0)))
        D[t] = dict(iou=iou.astype(np.float32),
                    ign=np.divide(inter, ar, out=np.zeros(len(rl)), where=ar > 0) > 0.5,
                    sc=np.asarray(r["scores"], np.float64))
        if (i + 1) % 25 == 0:
            print(f"  cache@{res} {i+1}/{len(gt)}", flush=True)
    pickle.dump(dict(D=D, n_gt=n_gt), open(cache_path, "wb"))
    return D, n_gt


def main():
    res = int(sys.argv[sys.argv.index("--res") + 1]); del sys.argv[sys.argv.index("--res"):sys.argv.index("--res") + 2]
    out = sys.argv[sys.argv.index("--out") + 1]
    assert not os.path.exists(out), f"REFUSING to overwrite {out}"
    A.RES, A.SCALE = res, 2048.0 / res
    A.CACHE = os.path.join(HERE, f"_ap_cache_s0_r{res}.pkl")
    A.build_cache = lambda: build_cache_res(res, MASKS[res], A.CACHE)
    print(f"[conf_ablation_res] res={res} masks={MASKS[res]} cache={A.CACHE}", flush=True)
    A.main()


if __name__ == "__main__":
    main()
