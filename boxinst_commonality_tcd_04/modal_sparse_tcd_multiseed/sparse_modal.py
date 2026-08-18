"""Modal A100 app: zero-shot eval of the SETTLED phase-4 model on the sparse-canopy TCD slice.

Question: the 4-phase L24 detector + beta=0.5 self-mask masker scores mask mAP50 0.630 +/- 0.005
on the official 439 -- a cohort that is ~80% closed-canopy forest, matching its 900 training
tiles. What does it do on OPEN CANOPY (savanna, grassland, shrubland, desert) that it has never
seen? No retraining, no refit, no re-tuning: the same checkpoints, the same masker npz, the same
alpha/kappa knobs.

Data: data/tcd_sparse (236 tiles, biomes 7/8/9/10/12/13), built by build_slice.py, tile-disjoint
from the 900 by tile id + image_id + rgb sha1. Joined to HF restor/tcd by image_id via
manifest_sparse.json -- never by filename, same rule as phase4.

ISOLATED: its own app + its own Volume (tcd-sparse-vol). The phase-4 volume is mounted for the
detector checkpoints and masker npz and is NEVER committed to -- nothing here can alter the 439
result. `rm -rf` this folder + `modal volume delete tcd-sparse-vol` is a complete tidy-up.

Stages:
    modal run sparse_modal.py::verify          # image_id + pixel join, free, no GPU
    modal run sparse_modal.py::extract_4p      # 4-phase L24 feats -> /vol   (~$0.8)
    modal run sparse_modal.py::band_sparse --seeds 0,1,2       # eval, ~$2/seed
    modal run sparse_modal.py::score_subsets --seeds 0,1,2     # CPU: the honest cuts
"""
import json
import os

import modal

APP_NAME = "tcd-sparse-l24"
VOL_NAME = "tcd-sparse-vol"                   # scratch: sparse feats + outputs
PHASE4_VOL = "tcd04-phase4-vol"               # READ-ONLY here: detector ckpts + masker npz
HFDATA_VOL = "canopyai-deepforest-data"       # read-only: HF restor/tcd + hub cache

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                   # boxinst_commonality_tcd_04
REPO = os.path.dirname(PKG)
PH4 = os.path.join(PKG, "modal_tcd_multiseed/phase4")
STUBS = os.path.join(PH4, "stubs")            # reuse phase4's stubs verbatim

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
p4vol = modal.Volume.from_name(PHASE4_VOL)
hfvol = modal.Volume.from_name(HFDATA_VOL)
hf_secret = modal.Secret.from_name("huggingface")

P = "/root/proj"
PKG_R = f"{P}/boxinst_commonality_tcd_04"
PH4_R = f"{PKG_R}/modal_tcd_multiseed/phase4"
SP_R = f"{PKG_R}/modal_sparse_tcd_multiseed"

# transformers 4.57 is the phase-4 pin. DO NOT BUMP: the detector checkpoints were trained on
# features from this exact version, and the local cache is 5.12 (cos 0.86) -- a bump would
# silently shift the features under a frozen checkpoint.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.1", "torchvision==0.27.1", "numpy==2.2.6",
                 "transformers==4.57.1", "datasets==4.0.0", "pillow", "contourpy",
                 "pycocotools")
    .env({"HF_HOME": "/hfdata/hf_cache", "HF_HUB_OFFLINE": "0",
          "HF_DATASETS_OFFLINE": "1"})
)
for rel in ("dapt/__init__.py", "dapt/backbone.py", "dapt/targets.py",
            "dapt/decode.py", "dapt/eval.py", "dapt/head.py"):
    image = image.add_local_file(os.path.join(REPO, rel), f"{P}/{rel}")
for rel in ("__init__.py", "detector.py", "train_detector_tiles.py", "evaluate.py",
            "em.py", "prepare_test.py", "cache_test.py", "cache_train_tiles.py",
            "test_gt.json", "train_tiles_gt.json"):
    image = image.add_local_file(os.path.join(PKG, rel), f"{PKG_R}/{rel}")
image = image.add_local_file(os.path.join(PKG, "vault", "em_model.npz"),
                             f"{PKG_R}/vault/em_model.npz")
image = image.add_local_file(os.path.join(PKG, "modal_tcd_multiseed", "__init__.py"),
                             f"{PKG_R}/modal_tcd_multiseed/__init__.py")
# phase-4 libraries are IMPORTED, never edited -- the eval path is bit-for-bit the 439 one.
for rel in ("__init__.py", "phase4_features_tcd.py", "phase4_lib_tcd.py",
            "phase4_fit_tcd.py", "ref_feat_tcd.npz", "manifest.json"):
    image = image.add_local_file(os.path.join(PH4, rel), f"{PH4_R}/{rel}")
for rel in ("__init__.py", "manifest_sparse.json"):
    image = image.add_local_file(os.path.join(HERE, rel), f"{SP_R}/{rel}")
image = image.add_local_file(os.path.join(REPO, "data/tcd_sparse/sparse_gt.json"),
                             f"{SP_R}/sparse_gt.json")
image = image.add_local_file(os.path.join(REPO, "data/tcd_sparse/slice_manifest.json"),
                             f"{SP_R}/slice_manifest.json")
STUB_FILES = {
    "boxinst": ("__init__.py", "cache_feats.py"),
    "boxinst_commonality": ("__init__.py", "em.py"),
    "boxinst_tcd": ("__init__.py", "build_canopy.py", "cache.py", "prepare.py"),
}
for pkg, files in STUB_FILES.items():
    for f in files:
        image = image.add_local_file(os.path.join(STUBS, pkg, f), f"{P}/{pkg}/{f}")

VOL = "/vol"
P4 = "/phase4"                                  # the phase-4 volume, read-only by convention
FEAT_SPARSE = f"{VOL}/feat_4p_sparse"           # (1024,256,256) real-8px L24, 236 tiles
OUT = f"{VOL}/out"
MANIFEST = f"{SP_R}/manifest_sparse.json"
SLICE_MANIFEST = f"{SP_R}/slice_manifest.json"
SPARSE_GT = f"{SP_R}/sparse_gt.json"
REF_NPZ = f"{PH4_R}/ref_feat_tcd.npz"
PH4_MANIFEST = f"{PH4_R}/manifest.json"        # only to resolve the parity ref tile
N_TILES = 236


def _setup_path():
    import sys
    if P not in sys.path:
        sys.path.insert(0, P)


def _load_hf_test():
    """restor/tcd `test` is the official 439; the rest of the corpus is in `train`. The sparse
    slice draws from BOTH, so index both splits and let image_id do the routing."""
    from datasets import load_dataset
    idx = {}
    ds = {s: load_dataset("restor/tcd", split=s) for s in ("train", "test")}
    for s, d in ds.items():
        for i, iid in enumerate(d["image_id"]):
            idx[int(iid)] = (s, i)
    return idx, ds


def _det_ckpt(seed):
    return f"{P4}/out/det_phase4_L24_s{seed}.pt"


def _masker_npz():
    """The settled masker: beta=0.5, geometry-fixed grid, alpha=0.3/kappa=1.6 stored inside."""
    return f"{P4}/out/em_model_4p_fix.npz"


@app.function(image=image, volumes={"/hfdata": hfvol}, timeout=1800, cpu=4,
              memory=32768, secrets=[hf_secret])
def verify():
    """Prove the join before any GPU. Stricter than phase4's: the slice records rgb_sha1 for
    EVERY tile (not a 20-tile sample), so all 236 are pixel-verified against HF."""
    import hashlib

    import numpy as np
    _setup_path()
    man = json.load(open(MANIFEST))["feat_test"]
    idx, ds = _load_hf_test()
    print(f"[verify] {len(man)} sparse tiles vs HF train={len(ds['train'])} "
          f"test={len(ds['test'])}", flush=True)

    def cat2(r):
        return sum(a["category_id"] == 2 for a in json.loads(r["coco_annotations"]))

    miss, wh_bad, box_bad, hash_ok, hash_bad = [], [], [], 0, []
    for k, (tid, rec) in enumerate(sorted(man.items())):
        iid = rec["image_id"]
        if iid not in idx:
            miss.append((tid, iid)); continue
        s, i = idx[iid]
        row = ds[s][i]
        if row["width"] != rec["width"] or row["height"] != rec["height"]:
            wh_bad.append((tid, iid))
        if cat2(row) != rec["n_cat2"]:
            box_bad.append((tid, iid))
        arr = np.asarray(row["image"].convert("RGB"))
        if (hashlib.sha1(arr.tobytes()).hexdigest() == rec["rgb_sha1"]
                and list(arr.shape) == rec["rgb_shape"]):
            hash_ok += 1
        else:
            hash_bad.append((tid, iid))
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(man)}", flush=True)
    print(f"[verify] missing={len(miss)} wh_bad={len(wh_bad)} box_bad={len(box_bad)} "
          f"pixel_sha1 ok={hash_ok} bad={len(hash_bad)}", flush=True)
    assert not (miss or wh_bad or box_bad or hash_bad), \
        f"join failed: miss={miss[:5]} wh={wh_bad[:5]} box={box_bad[:5]} hash={hash_bad[:5]}"
    assert hash_ok == len(man) == N_TILES, f"only {hash_ok}/{len(man)} pixel-verified"
    print(f"[verify] JOIN OK — all {hash_ok} tiles pixel-identical to HF", flush=True)
    return {"n": len(man), "pixel_hash_checked": hash_ok}


@app.function(gpu="A100", image=image, volumes={"/vol": vol, "/hfdata": hfvol},
              timeout=4 * 3600, cpu=8, memory=49152, secrets=[hf_secret])
def extract_4p():
    """4-phase real-8px L24 features for the 236 sparse tiles. Same backbone, same layers,
    same interleave as the 439 -- the reg self-test + layers-trap parity gate run FIRST and
    abort in seconds if anything differs. No native-4096: the self-mask masker reads these
    same 4-phase cells. Idempotent per-tile; resumable."""
    import time

    import numpy as np
    import torch
    _setup_path()
    assert torch.cuda.is_available(), "no CUDA"
    from dapt.backbone import FrozenDinoV3Features
    from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4 import \
        phase4_features_tcd as p4

    net = FrozenDinoV3Features("web", layers=(21, 22, 23, 24), device="cuda")
    print(f"[extract_4p] gpu={torch.cuda.get_device_name(0)} out_dim={net.out_dim} "
          f"layers={net.layers}", flush=True)
    assert net.out_dim == 4096 and net.layers == (21, 22, 23, 24), \
        f"bad backbone: dim={net.out_dim} layers={net.layers}"

    man = json.load(open(MANIFEST))["feat_test"]
    idx, ds = _load_hf_test()

    # ---- abort-fast reg self-test + layers-trap parity gate, on the SAME reference tile
    # phase4 used, so a passing gate means these features match the ones the frozen
    # checkpoints were trained on. Nothing about the eval is trustworthy if this drifts.
    ref = np.load(REF_NPZ, allow_pickle=True)
    ref_tid, ref_arr = str(ref["tid"]), ref["ref"].astype(np.float32)
    riid = json.load(open(PH4_MANIFEST))["feat_test"][ref_tid]["image_id"]
    hs, ii = idx[riid]
    rimg = ds[hs][ii]["image"].convert("RGB")
    p4.registration_self_test(net, rimg)
    _, native0 = p4.feat_4phase(net, rimg, want_native=True)
    got = native0[:, ::16, ::16].astype(np.float32).ravel()
    want = ref_arr.ravel()
    cos = float(got @ want / (np.linalg.norm(got) * np.linalg.norm(want) + 1e-9))
    print(f"[parity] phase(0,0) full-4096 vs known-good: cos={cos:.5f}", flush=True)
    # Same threshold as phase4: guards the catastrophic layers-scramble regime (cos ~0.17);
    # ~0.86 is benign transformers 4.57-vs-5.12 drift, and every phase-4 artifact we load
    # was produced under 4.57 too, so this run is self-consistent with the checkpoints.
    assert cos > 0.5, f"catastrophic feature mismatch (cos {cos:.4f}) — check layers"

    os.makedirs(FEAT_SPARSE, exist_ok=True)
    todo = [t for t in sorted(man)
            if not os.path.exists(os.path.join(FEAT_SPARSE, t + ".npy"))]
    print(f"[extract_4p] {len(todo)}/{len(man)} to extract", flush=True)
    t0 = time.time()
    for k, tid in enumerate(todo):
        s, i = idx[man[tid]["image_id"]]
        asm = p4.feat_4phase(net, ds[s][i]["image"].convert("RGB"))
        np.save(os.path.join(FEAT_SPARSE, tid + ".npy"), asm)
        if (k + 1) % 25 == 0 or k + 1 == len(todo):
            dt = time.time() - t0
            print(f"  {k+1}/{len(todo)}  {dt/(k+1):.2f}s/tile  "
                  f"ETA {(len(todo)-k-1)*dt/(k+1)/60:.0f} min  "
                  f"est_cost≈${dt/3600*2.1:.1f}", flush=True)
            vol.commit()
    vol.commit()
    n = len(os.listdir(FEAT_SPARSE))
    print(f"[extract_4p] done: {n}/{N_TILES}", flush=True)
    assert n == N_TILES, f"incomplete: {n}/{N_TILES}"
    return {"feat_4p_sparse": n}


@app.function(gpu="A100", image=image, volumes={"/vol": vol, P4: p4vol},
              timeout=4 * 3600, cpu=8, memory=65536)
def eval_sparse(seed: int = 0, mask_thr: float = 0.25, save_preds: bool = True,
                prior_weight: float = -1.0, kappa_scale: float = -1.0):
    """Zero-shot eval of the seed-s phase-4 detector + the settled beta=0.5-fixed masker on
    the sparse slice. Checkpoints and masker come from the phase-4 volume UNCHANGED; the
    knobs default to the masker's own stored alpha=0.3/kappa=1.6 (sentinel -1.0), i.e. the
    exact configuration behind the 0.630 headline. Nothing is written to the phase-4 volume."""
    import torch
    _setup_path()
    assert torch.cuda.is_available(), "no CUDA"
    from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4 import phase4_lib_tcd as L

    ckpt, npz = _det_ckpt(seed), _masker_npz()
    for p in (ckpt, npz):
        assert os.path.exists(p), f"{p} missing on the phase-4 volume"
    n = len(os.listdir(FEAT_SPARSE)) if os.path.exists(FEAT_SPARSE) else 0
    assert n == N_TILES, f"feat_4p_sparse {n}/{N_TILES} — run extract_4p"

    pw = None if prior_weight < 0 else prior_weight
    ks = None if kappa_scale < 0 else kappa_scale
    pk = "" if (pw, ks) == (None, None) else \
        f"_pw{int(round((pw or 1)*100)):03d}_ks{int(round((ks or 1)*100)):03d}"
    tag = f"sparse_L24_s{seed}{pk}"
    os.makedirs(OUT, exist_ok=True)
    preds_dir = os.path.join(OUT, f"preds_{tag}") if save_preds else None

    res = L.eval_4p_selfmask(ckpt, FEAT_SPARSE, SPARSE_GT, npz,
                             os.path.join(OUT, f"eval_{tag}.json"),
                             mask_thr=mask_thr, device="cuda", save_preds_dir=preds_dir,
                             prior_weight=pw, kappa_scale=ks)
    res.update({"tag": tag, "seed": seed, "slice": "tcd_sparse", "n_expected": N_TILES,
                "mask_thr": mask_thr})
    json.dump(res, open(os.path.join(OUT, f"results_{tag}.json"), "w"), indent=2)
    vol.commit()                                  # NOTE: p4vol is never committed
    print(f"[eval_sparse] s{seed}: mask={res['mask_mAP50']}/{res['mask_mAP50_95']} "
          f"box={res['box_mAP50']}  (439 seed-0 ref, same config: 0.6250/0.2574 mask, "
          f"0.6050 box)", flush=True)
    return res


@app.function(gpu="A100", image=image, volumes={"/vol": vol, P4: p4vol},
              timeout=8 * 3600, cpu=8, memory=65536)
def band_sparse(seeds: str = "0,1,2", mask_thr: float = 0.25):
    """Zero-shot band on the sparse slice. Convention: **seed 0 is the testing seed**; 1 and 2
    are added only for a final 3-seed band. Seeds are evaluated in the order given, so a
    `--seeds 0,1,2` run produces the seed-0 number first and extends it. Detectors differ by
    seed; the masker is seed-independent and reused across all three."""
    import time

    import numpy as np
    import torch
    _setup_path()
    assert torch.cuda.is_available(), "no CUDA"
    from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4 import phase4_lib_tcd as L

    npz = _masker_npz()
    assert os.path.exists(npz), f"{npz} missing on the phase-4 volume"
    n = len(os.listdir(FEAT_SPARSE)) if os.path.exists(FEAT_SPARSE) else 0
    assert n == N_TILES, f"feat_4p_sparse {n}/{N_TILES} — run extract_4p"
    os.makedirs(OUT, exist_ok=True)
    sl = [int(x) for x in str(seeds).split(",") if x.strip()]
    print(f"[band_sparse] gpu={torch.cuda.get_device_name(0)} seeds={sl} "
          f"mask_thr={mask_thr} em={os.path.basename(npz)} n_tiles={n}", flush=True)

    out, t0 = {}, time.time()
    for seed in sl:
        tag = f"sparse_L24_s{seed}"
        res_fp = os.path.join(OUT, f"results_{tag}.json")
        if os.path.exists(res_fp):                       # per-seed idempotent
            print(f"[band_sparse] seed {seed}: results exist -> reuse", flush=True)
            out[seed] = json.load(open(res_fp)); continue
        ckpt = _det_ckpt(seed)
        assert os.path.exists(ckpt), f"{ckpt} missing on the phase-4 volume"
        ts = time.time()
        res = L.eval_4p_selfmask(ckpt, FEAT_SPARSE, SPARSE_GT, npz,
                                 os.path.join(OUT, f"eval_{tag}.json"),
                                 mask_thr=mask_thr, device="cuda",
                                 save_preds_dir=os.path.join(OUT, f"preds_{tag}"))
        res.update({"tag": tag, "seed": seed, "slice": "tcd_sparse",
                    "n_expected": N_TILES, "mask_thr": mask_thr,
                    "eval_min": round((time.time() - ts) / 60, 1)})
        json.dump(res, open(res_fp, "w"), indent=2)
        vol.commit()                                     # p4vol is NEVER committed
        out[seed] = res
        print(f"[band_sparse] seed {seed}: mask={res['mask_mAP50']}/"
              f"{res['mask_mAP50_95']} box={res['box_mAP50']} "
              f"({res['eval_min']}min)", flush=True)
    band = {k: (round(float(np.mean([out[s][k] for s in out])), 4),
                round(float(np.std([out[s][k] for s in out])), 4))
            for k in ("mask_mAP50", "mask_mAP50_95", "box_mAP50")}
    print(f"[band_sparse] total {(time.time() - t0) / 60:.1f} min", flush=True)
    summary = {"seeds": sorted(out), "per_seed": out, "band_mean_std": band}
    json.dump(summary, open(os.path.join(OUT, "band_sparse.json"), "w"), indent=2)
    vol.commit()
    print(f"[band_sparse] {band}", flush=True)
    return summary


@app.function(image=image, volumes={"/vol": vol}, timeout=4 * 3600, cpu=8, memory=65536)
def score_subsets(seeds: str = "0,1,2"):
    """Re-score the SAVED predictions over the cuts that matter, on CPU, with no re-inference.

    The headline over all 236 is the optimistic number: 130 tiles share a 300 m scene with a
    training tile, and 16 come from the official 439 that tuned mask_thr and alpha/kappa. The
    `scene_clean` and `train_pool` cuts are the honest ones. Per-biome is the diagnostic.

    Uses phase4_lib._full_metrics on rebuilt per-tile lists, so every number is produced by
    the same scorer as the 439 headline."""
    import numpy as np
    _setup_path()
    from pycocotools import mask as maskUtils
    from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4 import phase4_lib_tcd as L
    import boxinst_commonality_tcd_04.evaluate as E

    gt = json.load(open(SPARSE_GT))
    sm = json.load(open(SLICE_MANIFEST))["tiles"]
    all_res = {}
    for seed in [int(x) for x in seeds.split(",")]:
        pf = os.path.join(OUT, f"preds_sparse_L24_s{seed}", "preds.json")
        assert os.path.exists(pf), f"{pf} missing — run band_sparse --seeds {seed} first"
        blob = json.load(open(pf))
        preds, op_thr = blob["preds"], blob["meta"]["op_thr"]

        per = {}
        for tid, p in preds.items():
            pm = np.array([maskUtils.decode({"size": r["size"],
                                             "counts": r["counts"].encode("ascii")}
                                            ).astype(bool) for r in p["masks_rle"]]) \
                if p["masks_rle"] else np.zeros((0, E.RES, E.RES), bool)
            sc = np.array(p["scores"], np.float32)
            pbox = np.array(p["boxes_2048"], np.float32).reshape(-1, 4) / E.SCALE
            gm = np.array(E.raster(gt[tid]["trees"]))
            can = np.array(E.raster(gt[tid]["canopy"]))
            can = can.any(0) if len(can) else np.zeros((E.RES, E.RES), bool)
            gb = (np.array([[*(np.asarray(t).reshape(-1, 2).min(0)),
                             *(np.asarray(t).reshape(-1, 2).max(0))]
                            for t in gt[tid]["trees"]], np.float32).reshape(-1, 4) / E.SCALE
                  if gt[tid]["trees"] else np.zeros((0, 4), np.float32))
            ign_b = np.zeros(len(pbox), bool)
            for i, (x0, y0, x1, y1) in enumerate(pbox):
                sub = can[slice(max(0, int(y0)), int(np.ceil(y1))),
                          slice(max(0, int(x0)), int(np.ceil(x1)))]
                ign_b[i] = sub.size > 0 and sub.mean() > 0.5
            per[tid] = (pbox, sc, gb, ign_b, pm, gm,
                        np.array(p["canopy_ignore"], bool))

        def score(tids):
            tids = [t for t in tids if t in per]
            if not tids:
                return None
            cols = list(zip(*(per[t] for t in tids)))
            det, seg = L._full_metrics(*cols, op_thr, box_ious=(0.4, 0.5),
                                       mask_ious=(0.5,))
            return {"n_tiles": len(tids),
                    "n_gt": int(sum(len(per[t][5]) for t in tids)),
                    "mask_mAP50": seg["mask_mAP50"],
                    "mask_mAP50_95": seg["mask_mAP50_95"],
                    "box_mAP50": det["box_mAP50"], "box_mAP40": det["box_mAP40"]}

        cuts = {
            "all": list(per),
            "scene_clean": [t for t in per if sm[t]["scene_clean"]],
            "train_pool": [t for t in per if sm[t]["source"] == "train_pool"],
            "scene_clean_and_train_pool": [t for t in per if sm[t]["scene_clean"]
                                           and sm[t]["source"] == "train_pool"],
        }
        for b in sorted({sm[t]["biome"] for t in per}):
            cuts[f"biome_{b}"] = [t for t in per if sm[t]["biome"] == b]
        all_res[seed] = {k: score(v) for k, v in cuts.items()}
        print(f"[score_subsets] seed {seed}: "
              f"{json.dumps({k: (v or {}).get('mask_mAP50') for k, v in all_res[seed].items()})}",
              flush=True)

    band = {}
    for cut in all_res[list(all_res)[0]]:
        vals = [all_res[s][cut] for s in all_res if all_res[s][cut]]
        if vals:
            band[cut] = {"n_tiles": vals[0]["n_tiles"], "n_gt": vals[0]["n_gt"], **{
                m: [round(float(np.mean([v[m] for v in vals])), 4),
                    round(float(np.std([v[m] for v in vals])), 4)]
                for m in ("mask_mAP50", "mask_mAP50_95", "box_mAP50")}}
    # Reference = the 439 run in the SAME configuration this eval uses: beta=0.5-fixed with
    # the masker's stored knobs (alpha=0.3, kappa x1.6) @ mask_thr 0.25. Seed-0 and the
    # 3-seed band are both carried so a seed-0-only run is anchored to the seed-0 number,
    # not to the band. (The vanilla alpha=1/kappa=1 arm scores 0.6203 on seed 0 / 0.615
    # 5-seed -- a different configuration, not a comparator for this run.)
    ref = {"config": "beta0.5-fixed, alpha=0.3, kappa=1.6, mask_thr=0.25",
           "seed0": {"mask_mAP50": 0.6250, "mask_mAP50_95": 0.2574, "box_mAP50": 0.6050},
           "band_012": {"mask_mAP50": 0.630, "mask_mAP50_95": 0.257, "box_mAP50": 0.603}}
    ref_used = ref["seed0"] if len(all_res) == 1 else ref["band_012"]
    out = {"per_seed": all_res, "band_mean_std": band,
           "reference_439": ref, "reference_used": ref_used,
           "delta_vs_439_all": ({m: round(band["all"][m][0] - ref_used[m], 4)
                                 for m in ref_used} if "all" in band else None)}
    json.dump(out, open(os.path.join(OUT, "subsets_sparse.json"), "w"), indent=2)
    vol.commit()
    print(json.dumps(band, indent=2), flush=True)
    return out
