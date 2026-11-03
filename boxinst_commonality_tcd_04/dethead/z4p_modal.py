"""Modal H100 (amendment 4a; G2 ran on A100): 16-phase ("real 4 px") L24 features for the detector (pre-registration 4, dethead/README.md).

Definition = native_raster/hunt/feat16p.py verbatim (same backbone/layer/windowing/shifts/interleave/projection), on the
HF restor/tcd tiles joined by manifest image_id with a per-tile RGB sha1 check. Output (512,512,128) fp16 per tile to
tcd04-phase4-vol:z4p_{train,test}/<tid>.npy. Per-tile gate G1 vs the same volume's feat_4p_* (abort on fail).
Resumable per tile (tmp + rename), sharded, logs on the volume (z4p_logs/<split>_<shard>.log), committed every 5 tiles.

  python native_raster/detach.py deploy ../../dethead/z4p_modal.py      (or: .venv/bin/modal deploy <this file>)
  python native_raster/detach.py spawn tcd04-dethead-z4p extract --split train --shard 0 --nshards 4
  (G2 first: --tiles tcd_tile_1012,tcd_tile_1041 then compare locally with hunt_z4p_val)
"""
import json
import os

import modal

APP_NAME = "tcd04-dethead-z4p"
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REPO = os.path.dirname(PKG)
PH4 = os.path.join(PKG, "modal_tcd_multiseed", "phase4")
STUBS = os.path.join(PH4, "stubs")

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("tcd04-phase4-vol")
hfvol = modal.Volume.from_name("canopyai-deepforest-data")
hf_secret = modal.Secret.from_name("huggingface")

P = "/root/proj"
PKG_R = f"{P}/boxinst_commonality_tcd_04"
PH4_R = f"{PKG_R}/modal_tcd_multiseed/phase4"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.1", "torchvision==0.27.1", "numpy==2.2.6",
                 "transformers==4.57.1", "datasets==4.0.0", "pillow", "contourpy", "pycocotools")
    .env({"HF_HOME": "/hfdata/hf_cache", "HF_HUB_OFFLINE": "0", "HF_DATASETS_OFFLINE": "1"})
)
for rel in ("dapt/__init__.py", "dapt/backbone.py", "dapt/targets.py", "dapt/decode.py", "dapt/eval.py", "dapt/head.py"):
    image = image.add_local_file(os.path.join(REPO, rel), f"{P}/{rel}")
for rel in ("__init__.py", "detector.py", "evaluate.py", "em.py", "prepare_test.py", "cache_test.py", "cache_train_tiles.py"):
    image = image.add_local_file(os.path.join(PKG, rel), f"{PKG_R}/{rel}")
image = image.add_local_file(os.path.join(PKG, "modal_tcd_multiseed", "__init__.py"), f"{PKG_R}/modal_tcd_multiseed/__init__.py")
for rel in ("__init__.py", "phase4_features_tcd.py", "phase4_lib_tcd.py", "manifest.json", "em_model_4p_fix.npz"):
    image = image.add_local_file(os.path.join(PH4, rel), f"{PH4_R}/{rel}")
for pkg, files in {"boxinst": ("__init__.py", "cache_feats.py"), "boxinst_commonality": ("__init__.py", "em.py"),
                   "boxinst_tcd": ("__init__.py", "build_canopy.py", "cache.py", "prepare.py")}.items():
    for f in files:
        image = image.add_local_file(os.path.join(STUBS, pkg, f), f"{P}/{pkg}/{f}")

VOL = "/vol"
SPLIT = {"train": ("feat_traintile", "train", f"{VOL}/feat_4p_train", f"{VOL}/z4p_train"),
         "test": ("feat_test", "test", f"{VOL}/feat_4p_test", f"{VOL}/z4p_test")}
SHIFTS = (0, 4, 8, 12)
EM_SHA1 = "2f0eaa3f0a17501a482b96b8feacf53f32a18d4e"


@app.function(gpu="H100", image=image, volumes={VOL: vol, "/hfdata": hfvol}, timeout=8 * 3600, cpu=4, memory=32768,
              secrets=[hf_secret], retries=modal.Retries(max_retries=3, initial_delay=10.0))
def extract(split: str = "train", shard: int = 0, nshards: int = 1, tiles: str = ""):
    import hashlib, sys, time
    import numpy as np
    import torch
    if P not in sys.path:
        sys.path.insert(0, P)
    from datasets import load_dataset
    from dapt.backbone import FrozenDinoV3Features
    from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.phase4_features_tcd import _extract_2048, _shift

    mkey, hfsplit, feat4p, out = SPLIT[split]
    os.makedirs(out, exist_ok=True); os.makedirs(f"{VOL}/z4p_logs", exist_ok=True)
    logp = f"{VOL}/z4p_logs/{split}_{shard}of{nshards}{'_' + tiles.replace(',', '+') if tiles else ''}.log"

    def log(msg):
        line = f"{time.strftime('%F %T')} {msg}"
        print(line, flush=True)
        with open(logp, "a") as f:
            f.write(line + "\n")

    emp = f"{PH4_R}/em_model_4p_fix.npz"
    assert hashlib.sha1(open(emp, "rb").read()).hexdigest() == EM_SHA1, "masker npz sha1 mismatch"
    d = np.load(emp, allow_pickle=False); mu, U, scale = d["mu"], d["U"], d["scale"]

    def project(feat):                                     # = TCDMasker.project
        z = (feat.reshape(feat.shape[0], -1).T.astype(np.float32) - mu) @ U / scale
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)

    man = json.load(open(f"{PH4_R}/manifest.json"))[mkey]
    ds = load_dataset("restor/tcd", split=hfsplit)
    row_of = {int(iid): i for i, iid in enumerate(ds["image_id"])}
    tids = tiles.split(",") if tiles else sorted(man)[shard::nshards]
    todo = [t for t in tids if not os.path.exists(os.path.join(out, t + ".npy"))]
    net = FrozenDinoV3Features("web", layers=(24,), device="cuda")
    log(f"[z4p {split} {shard}/{nshards}] gpu={torch.cuda.get_device_name(0)} out_dim={net.out_dim} "
        f"{len(todo)}/{len(tids)} to do -> {out}")
    assert net.out_dim == 1024
    t0 = time.time(); done = 0
    for k, t in enumerate(todo):
        img = ds[row_of[man[t]["image_id"]]]["image"].convert("RGB")
        a8 = np.asarray(img)
        if "rgb_sha1" in man[t]:                         # the manifest hashes a 20-tile sample per split (amendment 4b)
            assert hashlib.sha1(a8.tobytes()).hexdigest() == man[t]["rgb_sha1"], f"{t}: RGB sha1 mismatch"
        assert (img.width, img.height) == (man[t]["width"], man[t]["height"]), f"{t}: size mismatch"
        arr = a8.astype(np.float32) / 255.0
        assert arr.shape == (2048, 2048, 3)
        Z = np.zeros((512, 512, 128), np.float32)
        for dy in SHIFTS:
            for dx in SHIFTS:
                f = _extract_2048(net, _shift(arr, dy, dx))                     # (1024,128,128) fp32
                f = f.astype(np.float16).astype(np.float32)                     # = feat16p (fp16 storage of feat_4p)
                Z[dy // 4::4, dx // 4::4] = project(f).reshape(128, 128, 128)
        ref = project(np.load(os.path.join(feat4p, t + ".npy")).astype(np.float32)).reshape(256, 256, 128)
        cos = (Z[0::2, 0::2] * ref).sum(-1).ravel()
        g = {"tile": t, "cos_mean": float(cos.mean()), "cos_p1": float(np.percentile(cos, 1)), "cos_min": float(cos.min())}
        with open(f"{VOL}/z4p_logs/gate_{split}.jsonl", "a") as f:
            f.write(json.dumps(g) + "\n")
        assert g["cos_mean"] >= 0.995 and g["cos_p1"] >= 0.98, f"GATE G1 FAIL {g}"
        fp = os.path.join(out, t + ".npy")
        np.save(fp + ".tmp.npy", Z.astype(np.float16)); os.replace(fp + ".tmp.npy", fp)
        done += 1; el = time.time() - t0
        log(f"  {k+1}/{len(todo)} {t} G1 cos {g['cos_mean']:.5f} p1 {g['cos_p1']:.4f}  {el/done:.1f}s/tile  "
            f"ETA {(len(todo)-k-1)*el/done/60:.0f} min")
        if done % 5 == 0:
            vol.commit()
    vol.commit()
    n = len([f for f in os.listdir(out) if f.endswith(".npy") and not f.endswith(".tmp.npy")])
    log(f"[z4p {split} {shard}/{nshards}] DONE {done} new; {n} tiles in {out}")
    return {"split": split, "shard": shard, "new": done, "total_in_dir": n}
