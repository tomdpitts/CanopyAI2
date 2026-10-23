"""16-phase ("real 4 px") LACE features, local MPS. Same frozen backbone and layer as the deployed masker features
(DINOv3-web ViT-L/16, block 24, L2-normed patch tokens; phase4_features_tcd.py), same 2x2 1024-px windowing per pass,
run on the tile shifted by every (dy, dx) in {0,4,8,12}^2 and interleaved: phase (dy,dx) patch (gy,gx) is centred at
pixel (dx + 8 + 16 gx, dy + 8 + 16 gy), so the union is a 4 px lattice with cell (Y,X) centred at (8 + 4X, 8 + 4Y)
-> masker geometry s = 4, origin = 8. Stored: the masker PROJECTION only (TCDMasker.project: whiten to 128-d, unit
norm), (512, 512, 128) fp16 per tile (64 MB) in feat_cache/hunt_z4p_<split>/.

GATE (per tile, logged): the 4 phases at {0,8}^2 form the 8 px grid; their projections are compared with the
projection of the Modal-cached feat_4p (cosine per cell). Local MPS vs Modal CUDA features are never bit-identical
(project_environment memory), so the gate is cosine: mean >= 0.995 and 1st percentile >= 0.98, else abort.

  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.hunt.feat16p --split val [--tiles t1 t2 ...]
"""
import argparse, json, os, time
import numpy as np
from PIL import Image

FC = "/Users/tompitts/dphil/feat_cache"
REPO = "/Users/tompitts/dphil/CanopyAI2"
SPLITS = {"val": (f"{REPO}/data/tcd/train", f"{FC}/feat_4p_val"), "test": (f"{REPO}/data/tcd/test", f"{FC}/feat_4p_test_dl/feat_4p_test")}
SHIFTS = (0, 4, 8, 12)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True); ap.add_argument("--tiles", nargs="*")
    ap.add_argument("--tile_list", default=None)
    a = ap.parse_args()
    import torch
    from dapt.backbone import FrozenDinoV3Features
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.phase4_features_tcd import _shift, _extract_2048
    rgb_dir, feat4p = SPLITS[a.split]
    out = f"{FC}/hunt_z4p_{a.split}"; os.makedirs(out, exist_ok=True)
    m = E.TCDMasker(f"{FC}/em_model_4p_fix.npz")
    tids = a.tiles or (json.load(open(a.tile_list)) if a.tile_list else sorted(f[:-4] for f in os.listdir(feat4p) if f.endswith(".npy")))
    net = FrozenDinoV3Features("web", layers=(24,), device="mps")
    gate_log = os.path.join(out, "_gate.jsonl")
    print(f"[feat16p {a.split}] {len(tids)} tiles -> {out}", flush=True)
    t0 = time.time(); done = 0
    for k, t in enumerate(tids):
        fp = os.path.join(out, t + ".npy")
        if os.path.exists(fp):
            continue
        arr = np.asarray(Image.open(os.path.join(rgb_dir, t + ".tif")).convert("RGB"), np.float32) / 255.0
        assert arr.shape == (2048, 2048, 3)
        Z = np.zeros((512, 512, 128), np.float32)
        for dy in SHIFTS:
            for dx in SHIFTS:
                f = _extract_2048(net, _shift(arr, dy, dx))                     # (1024,128,128) fp32
                f = f.astype(np.float16).astype(np.float32)                     # match the fp16 storage of feat_4p
                Z[dy // 4::4, dx // 4::4] = m.project(f).reshape(128, 128, 128)
        ref = m.project(np.load(os.path.join(feat4p, t + ".npy")).astype(np.float32)).reshape(256, 256, 128)
        sub = Z[0::2, 0::2]                                                      # 8 px cells = shifts {0,8}^2
        cos = (sub * ref).sum(-1).ravel()
        g = {"tile": t, "cos_mean": float(cos.mean()), "cos_p1": float(np.percentile(cos, 1)), "cos_min": float(cos.min())}
        open(gate_log, "a").write(json.dumps(g) + "\n")
        assert g["cos_mean"] >= 0.995 and g["cos_p1"] >= 0.98, f"GATE FAIL {g}"
        np.save(fp + ".tmp.npy", Z.astype(np.float16)); os.replace(fp + ".tmp.npy", fp)
        done += 1; el = time.time() - t0
        print(f"  {k+1}/{len(tids)} {t} cos mean {g['cos_mean']:.4f} p1 {g['cos_p1']:.4f}  {el/done:.0f}s/tile", flush=True)
    print(f"[feat16p {a.split}] done", flush=True)


if __name__ == "__main__":
    main()
