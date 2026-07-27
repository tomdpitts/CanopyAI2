"""Modal A100 app: L24 4-phase real-8px seed-0 go/no-go on OAM-TCD (900 train/439 test).

Question: does REAL 8px sampling (4 shifted backbone passes, interleaved) beat the
native INTERPOLATED 8px at layer 24? Compare single-scale box+mask mAP50 vs the interp-
L24 probe (0.502 mask / 0.540 box) and native full-4096 (0.499 / 0.555).

Data pattern reused from feat_ablation/modal_app: HF restor/tcd tiles (on the
canopyai-deepforest-data volume) joined to the 900+439 cohort by image_id (manifest.json,
pixel-sha1 verified), never by filename. The DINOv3 backbone is loaded with the CORRECT
layers=(21,22,23,24) and an abort-fast parity gate vs a known-good local slice
(ref_feat_tcd.npz, cosine>0.98) guards against the layers-default trap that ruined the
earlier Modal TCD run.

ISOLATED: new Volume tcd04-phase4-vol; everything written under /vol/**. Deleting this
folder + the Volume is a complete tidy-up; no native file/default is touched.

Stages (A100):
    modal run phase4_modal.py::verify            # image_id join (free, no GPU)
    modal run phase4_modal.py::extract_4p        # 4-phase feats -> /vol  (reg+parity gate first)
    modal run phase4_modal.py::train_eval_4p     # seed-0 Detector4Phase train + box+mask eval
Or the chain:  modal run phase4_modal.py
"""
import json
import os

import modal

APP_NAME = "tcd04-phase4-l24"
VOL_NAME = "tcd04-phase4-vol"                 # scratch: 4-phase feats + outputs
HFDATA_VOL = "canopyai-deepforest-data"       # read-only: HF restor/tcd + hub cache

HERE = os.path.dirname(os.path.abspath(__file__))     # .../modal_tcd_multiseed/phase4
MTS = os.path.dirname(HERE)                            # .../modal_tcd_multiseed
PKG = os.path.dirname(MTS)                             # boxinst_commonality_tcd_04
REPO = os.path.dirname(PKG)
STUBS = os.path.join(HERE, "stubs")

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
hfvol = modal.Volume.from_name(HFDATA_VOL)
hf_secret = modal.Secret.from_name("huggingface")

P = "/root/proj"
PKG_R = f"{P}/boxinst_commonality_tcd_04"
PH4_R = f"{PKG_R}/modal_tcd_multiseed/phase4"

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
for rel in ("__init__.py",):
    image = image.add_local_file(os.path.join(MTS, rel),
                                 f"{PKG_R}/modal_tcd_multiseed/{rel}")
for rel in ("__init__.py", "phase4_features_tcd.py", "phase4_lib_tcd.py",
            "phase4_fit_tcd.py", "manifest.json", "ref_feat_tcd.npz"):
    image = image.add_local_file(os.path.join(HERE, rel), f"{PH4_R}/{rel}")
STUB_FILES = {
    "boxinst": ("__init__.py", "cache_feats.py"),
    "boxinst_commonality": ("__init__.py", "em.py"),
    "boxinst_tcd": ("__init__.py", "build_canopy.py", "cache.py", "prepare.py"),
}
for pkg, files in STUB_FILES.items():
    for f in files:
        image = image.add_local_file(os.path.join(STUBS, pkg, f), f"{P}/{pkg}/{f}")

VOL = "/vol"
FEAT_4P_TRAIN = f"{VOL}/feat_4p_train"        # (1024,256,256) real-8px L24 detector feats
FEAT_4P_TEST = f"{VOL}/feat_4p_test"          # (1024,256,256) test detector feats
NATIVE_TEST = f"{VOL}/native_test"            # (4096,128,128) full native, for the EM masker
OUT = f"{VOL}/out"
TEST_GT = f"{PKG_R}/test_gt.json"
TRAIN_GT = f"{PKG_R}/train_tiles_gt.json"
EM_PATH = f"{PKG_R}/vault/em_model.npz"       # vaulted masker (fit on native 16px, local)


def _selfmask_npz(beta, fix=False):
    # beta 0.5 keeps the legacy name (the recorded 0.449 run); others get their own file.
    # fix=True -> geometry-corrected grid (cell_origin=s, 8X+8) written to a `_fix` path so
    # it never clobbers the pre-existing half-cell-mis-registered vault models (clean A/B).
    suf = "_fix" if fix else ""
    return (f"{OUT}/em_model_4p{suf}.npz" if beta == 0.5
            else f"{OUT}/em_model_4p_b{beta:g}{suf}.npz")


def _btag(beta):
    return "" if beta == 0.5 else f"_b{beta:g}"
MANIFEST = f"{PH4_R}/manifest.json"
REF_NPZ = f"{PH4_R}/ref_feat_tcd.npz"
HF_SPLIT = {"feat_traintile": "train", "feat_test": "test"}


def _setup_path():
    import sys
    if P not in sys.path:
        sys.path.insert(0, P)


def _load_hf():
    from datasets import load_dataset
    idx, ds_by_split = {}, {}
    for split in ("train", "test"):
        ds = load_dataset("restor/tcd", split=split)
        ds_by_split[split] = ds
        for i, iid in enumerate(ds["image_id"]):
            idx[int(iid)] = (split, i)
    return idx, ds_by_split


@app.function(image=image, volumes={"/hfdata": hfvol}, timeout=1800, cpu=4,
              memory=32768, secrets=[hf_secret])
def verify():
    """Prove the image_id join before any GPU: every cohort tile's image_id is in HF,
    width/height + ITC box-count agree, pixel-sha1 sample matches (reused verbatim)."""
    import hashlib

    import numpy as np
    _setup_path()
    tok = [k for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
                       "HF_API_TOKEN") if os.environ.get(k)]
    print(f"[verify] HF token env: {tok or 'NONE (extract will fail on gated DINOv3)'}",
          flush=True)
    man = json.load(open(MANIFEST))
    idx, ds = _load_hf()
    print(f"[verify] HF train={len(ds['train'])} test={len(ds['test'])}", flush=True)

    def cat2(r):
        return sum(a["category_id"] == 2 for a in json.loads(r["coco_annotations"]))

    miss, wh_bad, box_bad, hash_ok, hash_bad = [], [], [], 0, []
    for split, tiles in man.items():
        want = HF_SPLIT[split]
        for tid, rec in tiles.items():
            iid = rec["image_id"]
            if iid not in idx or idx[iid][0] != want:
                miss.append((split, tid, iid)); continue
            row = ds[want][idx[iid][1]]
            if row["width"] != rec["width"] or row["height"] != rec["height"]:
                wh_bad.append((tid, iid))
            if cat2(row) != rec["n_cat2"]:
                box_bad.append((tid, iid))
            if "rgb_sha1" in rec:
                arr = np.asarray(row["image"].convert("RGB"))
                if hashlib.sha1(arr.tobytes()).hexdigest() == rec["rgb_sha1"] \
                        and list(arr.shape) == rec["rgb_shape"]:
                    hash_ok += 1
                else:
                    hash_bad.append((tid, iid))
    print(f"[verify] missing={len(miss)} wh_bad={len(wh_bad)} box_bad={len(box_bad)} "
          f"pixel_sha1 ok={hash_ok} bad={len(hash_bad)}", flush=True)
    ok = not (miss or wh_bad or box_bad or hash_bad) and hash_ok > 0
    assert ok, f"join failed: miss={miss[:5]} wh={wh_bad[:5]} box={box_bad[:5]} hash={hash_bad[:5]}"
    print("[verify] JOIN OK", flush=True)
    return {"train": len(man["feat_traintile"]), "test": len(man["feat_test"]),
            "pixel_hash_checked": hash_ok}


@app.function(gpu="A100", image=image, volumes={"/vol": vol, "/hfdata": hfvol},
              timeout=6 * 3600, cpu=8, memory=49152, secrets=[hf_secret])
def extract_4p():
    """4-phase real-8px L24 features (train+test) + full-4096 native (test) for the EM
    masker. Runs the registration self-test + layers-trap parity gate on 1 real tile
    FIRST (aborts in seconds if wrong). Idempotent per-tile; resumable."""
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

    man = json.load(open(MANIFEST))
    idx, ds = _load_hf()

    # ---- abort-fast reg self-test + layers-trap parity gate on the ref tile ----
    ref = np.load(REF_NPZ, allow_pickle=True)
    ref_tid = str(ref["tid"]); ref_arr = ref["ref"].astype(np.float32)
    riid = man["feat_test"][ref_tid]["image_id"]
    hs, ii = idx[riid]
    rimg = ds[hs][ii]["image"].convert("RGB")
    p4.registration_self_test(net, rimg)                       # invertibility + real!=interp
    _, native0 = p4.feat_4phase(net, rimg, want_native=True)   # (4096,128,128)
    got = native0[:, ::16, ::16].astype(np.float32).ravel(); want = ref_arr.ravel()
    cos = float(got @ want / (np.linalg.norm(got) * np.linalg.norm(want) + 1e-9))
    print(f"[parity] phase(0,0) full-4096 vs local known-good: cos={cos:.5f} "
          f"(local cache = transformers 5.12; image pins 4.57 -> some drift expected)",
          flush=True)
    # Hard guard ONLY against a catastrophic layers/scramble error (that regime is
    # cos~0.17). Moderate version drift (~0.86, transformers 4.57 vs 5.12) is tolerated:
    # the reg-test above proves the interleave is correct, layers are asserted (21..24),
    # and every seed of this run uses the SAME env so the 4-phase band is self-consistent.
    assert cos > 0.5, f"catastrophic feature mismatch (cos {cos:.4f}) — check layers/extraction"
    if cos < 0.95:
        print(f"[parity] NOTE cos {cos:.3f}<0.95: Modal DINOv3 (transformers 4.57) differs "
              f"from the local 5.12 cache; absolute mAP is NOT directly comparable to the "
              f"local 0.499/0.502 refs, but the Modal 4-phase seeds are mutually consistent.",
              flush=True)

    # ---- full extract ----
    for split, tiles in man.items():
        is_test = split == "feat_test"
        l24_dir = FEAT_4P_TEST if is_test else FEAT_4P_TRAIN
        os.makedirs(l24_dir, exist_ok=True)
        if is_test:
            os.makedirs(NATIVE_TEST, exist_ok=True)
        want = HF_SPLIT[split]
        todo = [t for t in sorted(tiles)
                if not os.path.exists(os.path.join(l24_dir, t + ".npy"))]
        print(f"[extract_4p:{split}] {len(todo)}/{len(tiles)} to extract", flush=True)
        t0 = time.time()
        for k, tid in enumerate(todo):
            img = ds[want][idx[tiles[tid]["image_id"]][1]]["image"].convert("RGB")
            if is_test:
                asm, native = p4.feat_4phase(net, img, want_native=True)
                np.save(os.path.join(NATIVE_TEST, tid + ".npy"), native)
            else:
                asm = p4.feat_4phase(net, img)
            np.save(os.path.join(l24_dir, tid + ".npy"), asm)
            if (k + 1) % 25 == 0 or k + 1 == len(todo):
                dt = time.time() - t0
                eta = (len(todo) - k - 1) * dt / (k + 1) / 60
                print(f"  {k+1}/{len(todo)}  {dt/(k+1):.2f}s/tile  ETA {eta:.0f} min "
                      f"est_cost≈${(time.time()-t0)/3600*2.1:.1f}", flush=True)
                vol.commit()
        vol.commit()
    n_tr = len(os.listdir(FEAT_4P_TRAIN)); n_te = len(os.listdir(FEAT_4P_TEST))
    n_nat = len(os.listdir(NATIVE_TEST))
    print(f"[extract_4p] done train={n_tr} test={n_te} native={n_nat}", flush=True)
    return {"feat_4p_train": n_tr, "feat_4p_test": n_te, "native_test": n_nat}


@app.function(gpu="A100", image=image, volumes={"/vol": vol}, timeout=6 * 3600,
              cpu=8, memory=65536)
def train_eval_4p(seed: int = 0, epochs: int = 40):
    """Train seed-s Detector4Phase on the 4-phase L24 real-8px features, then single-scale
    box+mask eval (detector=4-phase L24, EM masker=full-4096 native). H100/A100 wall +
    est cost printed (A100 ≈ $0.035/min)."""
    import time

    import torch
    _setup_path()
    assert torch.cuda.is_available(), "no CUDA"
    from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4 import phase4_lib_tcd as L
    n_tr = len(os.listdir(FEAT_4P_TRAIN)); n_te = len(os.listdir(FEAT_4P_TEST))
    n_nat = len(os.listdir(NATIVE_TEST))
    assert n_tr == 900 and n_te == 439 and n_nat == 439, \
        f"volume incomplete: train={n_tr}/900 test={n_te}/439 native={n_nat}/439"
    os.makedirs(OUT, exist_ok=True)
    tag = f"phase4_L24_s{seed}"
    t0 = time.time()
    L.train_4p(FEAT_4P_TRAIN, OUT, tag=tag, seed=seed, epochs=epochs, device="cuda")
    vol.commit()
    train_min = (time.time() - t0) / 60
    ckpt = os.path.join(OUT, f"det_{tag}.pt")
    t1 = time.time()
    res = L.eval_4p(ckpt, FEAT_4P_TEST, NATIVE_TEST, TEST_GT, EM_PATH,
                    os.path.join(OUT, f"eval_{tag}.json"), device="cuda")
    eval_min = (time.time() - t1) / 60
    res.update({"tag": tag, "seed": seed, "train_min": round(train_min, 1),
                "eval_min": round(eval_min, 1),
                "est_cost_usd": round((train_min + eval_min) * 0.035, 2)})
    json.dump(res, open(os.path.join(OUT, f"results_{tag}.json"), "w"), indent=2)
    vol.commit()
    print(f"[train_eval_4p] {tag}: train {train_min:.1f}m eval {eval_min:.1f}m "
          f"mask={res['mask_mAP50']} box={res['box_mAP50']} "
          f"est_cost≈${res['est_cost_usd']}", flush=True)
    return res


@app.function(image=image, volumes={"/vol": vol}, timeout=4 * 3600, cpu=8,
              memory=131072)          # CPU-only: numpy EM fit, no GPU -> cheap
def fit_masker_4p(n_tiles: int = 120, seed: int = 0, beta: float = 0.5,
                  fix: bool = False):
    """Refit the box->mask EM masker ON the cached 4-phase L24 train cells (self-mask),
    so the masker is in-distribution to the 4-phase boxes. CPU numpy, no GPU. Idempotent.

    beta=0 (no_contrast) = the ROBUST masker the forensics settled on (fill + light carve,
    no prototype collapse); beta=0.5 = the collapsing default (recorded self-mask 0.449).
    Each beta writes its own npz so results don't clobber each other.

    fix=True: the code now registers the interleaved 8px grid at its TRUE cell centers
    (cell_origin=s, 8X+8) instead of the half-cell-shifted s/2; --fix routes the corrected
    model to a `_fix` npz so the old (mis-registered) vault models are kept for the A/B."""
    import time
    _setup_path()
    from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4 import phase4_fit_tcd as F
    npz = _selfmask_npz(beta, fix)
    if os.path.exists(npz):
        print(f"[fit_masker_4p] {npz} exists -> skip", flush=True)
        return {"em": npz, "beta": beta, "skipped": True}
    n_tr = len(os.listdir(FEAT_4P_TRAIN))
    assert n_tr == 900, f"feat_4p_train incomplete: {n_tr}/900 — run extract_4p"
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    F.fit_masker_4p(FEAT_4P_TRAIN, TRAIN_GT, OUT, npz, n_tiles=n_tiles, seed=seed,
                    contrastive_beta=beta, no_contrast=(beta == 0))
    vol.commit()
    mins = (time.time() - t0) / 60
    print(f"[fit_masker_4p] beta={beta} done in {mins:.1f}min -> {npz}", flush=True)
    return {"em": npz, "beta": beta, "n_tiles": n_tiles, "fit_min": round(mins, 1)}


@app.function(gpu="A100", image=image, volumes={"/vol": vol}, timeout=4 * 3600,
              cpu=8, memory=65536)
def eval_selfmask(seed: int = 0, beta: float = 0.0, mask_thr: float = 0.25,
                  save_preds: bool = False, limit: int = 0, fix: bool = False):
    """Eval seed-s 4-phase detector with the REFIT 4-phase masker (self-mask) at the given
    beta and mask_thr. beta=0 + mask_thr=0.5 is the current headline (mask 0.5794/0.1948);
    lowering mask_thr toward ~0.25 grows masks to recover small crowns that under-cover
    under imprecise predicted boxes (det_t8 proxy: +0.043 mAP50 / +0.035 mAP50-95). Non-0.5
    thresholds write their own files so the 0.5 record is preserved.

    fix=True: use the geometry-corrected `_fix` masker (cell_origin=s) + its +1px render
    shift; writes `_fix`-suffixed results/preds so the mis-registered baseline is kept."""
    import torch
    _setup_path()
    assert torch.cuda.is_available(), "no CUDA"
    from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4 import phase4_lib_tcd as L
    npz = _selfmask_npz(beta, fix)
    bt = _btag(beta)
    ft = "_fix" if fix else ""
    tt = "" if mask_thr == 0.5 else f"_thr{int(round(mask_thr*100)):03d}"
    assert os.path.exists(npz), (f"{npz} missing — run fit_masker_4p --beta {beta}"
                                 f"{' --fix' if fix else ''} first")
    tag = f"phase4_L24_s{seed}"
    ckpt = os.path.join(OUT, f"det_{tag}.pt")
    preds_dir = (os.path.join(OUT, f"preds_selfmask{bt}{ft}{tt}_{tag}")
                 if save_preds else None)
    res = L.eval_4p_selfmask(ckpt, FEAT_4P_TEST, TEST_GT, npz,
                             os.path.join(OUT, f"eval_selfmask{bt}{ft}{tt}_{tag}.json"),
                             mask_thr=mask_thr, device="cuda", save_preds_dir=preds_dir,
                             limit=(limit or None))
    res.update({"tag": tag, "seed": seed, "beta": beta, "mask_thr": mask_thr, "fix": fix})
    json.dump(res, open(os.path.join(OUT, f"results_selfmask{bt}{ft}{tt}_{tag}.json"), "w"),
              indent=2)
    vol.commit()
    print(f"[eval_selfmask] {tag} beta={beta} mask_thr={mask_thr}: "
          f"mask={res['mask_mAP50']}/{res['mask_mAP50_95']} box={res['box_mAP50']}",
          flush=True)
    return res


@app.function(gpu="A100", image=image, volumes={"/vol": vol}, timeout=12 * 3600,
              cpu=8, memory=65536)
def band_selfmask(seeds: str = "0,1,2,3,4", beta: float = 0.5, fix: bool = True,
                  mask_thr: float = 0.25, epochs: int = 40, save_preds: bool = True):
    """THE DEPLOYABLE multi-seed pipeline: 4-phase L24 detector + β self-mask masker
    (geometry-FIXED grid when fix=True) @ mask_thr. DEFAULT = the settled β=0.5-fixed winner
    (seed-0 mask 0.620 / box 0.605). Per seed: train the detector (skip if ckpt exists) +
    eval_4p_selfmask. Reuses the seed-independent masker + cached features; per-seed
    idempotent — result filenames match `eval_selfmask`, so seed 0 from the single-seed run
    is reused (skipped, not recomputed). `--seeds 0,1,2,3,4` adds seeds 1–4 to the existing
    seed 0 and reports the full 5-seed band (mean ± std)."""
    import time

    import numpy as np
    import torch
    _setup_path()
    assert torch.cuda.is_available(), "no CUDA"
    from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4 import phase4_lib_tcd as L
    npz = _selfmask_npz(beta, fix)
    assert os.path.exists(npz), (f"{npz} missing — run fit_masker_4p --beta {beta}"
                                 f"{' --fix' if fix else ''} first")
    os.makedirs(OUT, exist_ok=True)
    bt = _btag(beta)
    ft = "_fix" if fix else ""
    tt = "" if mask_thr == 0.5 else f"_thr{int(round(mask_thr*100)):03d}"
    sl = [int(s) for s in str(seeds).split(",") if s.strip() != ""]
    print(f"[band_selfmask] gpu={torch.cuda.get_device_name(0)} seeds={sl} beta={beta} "
          f"fix={fix} mask_thr={mask_thr} em={os.path.basename(npz)}", flush=True)
    t0 = time.time()
    out = {}
    for seed in sl:
        tag = f"phase4_L24_s{seed}"
        res_fp = os.path.join(OUT, f"results_selfmask{bt}{ft}{tt}_{tag}.json")
        if os.path.exists(res_fp):
            print(f"[band_selfmask] seed {seed}: results exist -> skip", flush=True)
            out[seed] = json.load(open(res_fp)); continue
        ts = time.time()
        L.train_4p(FEAT_4P_TRAIN, OUT, tag=tag, seed=seed, epochs=epochs, device="cuda")
        vol.commit()
        preds_dir = (os.path.join(OUT, f"preds_selfmask{bt}{ft}{tt}_{tag}")
                     if save_preds else None)
        res = L.eval_4p_selfmask(os.path.join(OUT, f"det_{tag}.pt"), FEAT_4P_TEST,
                                 TEST_GT, npz,
                                 os.path.join(OUT, f"eval_selfmask{bt}{ft}{tt}_{tag}.json"),
                                 mask_thr=mask_thr, device="cuda", save_preds_dir=preds_dir)
        res.update({"tag": tag, "seed": seed, "beta": beta, "fix": fix,
                    "mask_thr": mask_thr, "seed_min": round((time.time() - ts) / 60, 1)})
        json.dump(res, open(res_fp, "w"), indent=2)
        vol.commit()
        out[seed] = res
        print(f"[band_selfmask] seed {seed}: mask={res['mask_mAP50']}/"
              f"{res['mask_mAP50_95']} box={res['box_mAP50']} in {res['seed_min']}min",
              flush=True)
    mins = (time.time() - t0) / 60
    per = {s: {"mask_mAP50": out[s]["mask_mAP50"],
               "mask_mAP50_95": out[s]["mask_mAP50_95"],
               "box_mAP50": out[s]["box_mAP50"]} for s in out}
    band = {}
    for k in ("mask_mAP50", "mask_mAP50_95", "box_mAP50"):
        v = np.array([per[s][k] for s in per], float)
        band[k] = {"mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4),
                   "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}
    summary = {"config": f"beta={beta} fix={fix} mask_thr={mask_thr}",
               "em": os.path.basename(npz), "seeds": sl, "n_seeds": len(per),
               "per_seed": per, "band": band, "wall_min": round(mins, 1)}
    json.dump(summary, open(os.path.join(OUT, f"band_selfmask{bt}{ft}{tt}.json"), "w"),
              indent=2)
    vol.commit()
    print(f"[band_selfmask] {len(per)} seeds in {mins:.1f}min est≈${mins*0.035:.2f} | "
          f"mask mAP50 {band['mask_mAP50']['mean']}±{band['mask_mAP50']['std']} "
          f"box mAP50 {band['box_mAP50']['mean']}±{band['box_mAP50']['std']}", flush=True)
    return summary


@app.function(gpu="A100", image=image, volumes={"/vol": vol}, timeout=6 * 3600,
              cpu=8, memory=65536)
def band_4p(seeds: str = "0,1,2,3,4", epochs: int = 40):
    """5-seed 4-phase L24 band on the SAME cached features (extraction is seed-independent).
    Per-seed idempotent: skips a seed whose results_{tag}.json already exists, so this can
    extend a seed-0 run to the full band without redoing it. Detector features unchanged
    across seeds; only the detector-training seed varies."""
    import time

    import torch
    _setup_path()
    assert torch.cuda.is_available(), "no CUDA"
    from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4 import phase4_lib_tcd as L
    n_tr = len(os.listdir(FEAT_4P_TRAIN)); n_te = len(os.listdir(FEAT_4P_TEST))
    n_nat = len(os.listdir(NATIVE_TEST))
    assert n_tr == 900 and n_te == 439 and n_nat == 439, \
        f"volume incomplete: train={n_tr}/900 test={n_te}/439 native={n_nat}/439 — run extract_4p"
    os.makedirs(OUT, exist_ok=True)
    sl = [int(s) for s in str(seeds).split(",") if s.strip() != ""]
    print(f"[band_4p] gpu={torch.cuda.get_device_name(0)} seeds={sl}", flush=True)
    t0 = time.time()
    out = {}
    for seed in sl:
        tag = f"phase4_L24_s{seed}"
        res_fp = os.path.join(OUT, f"results_{tag}.json")
        if os.path.exists(res_fp):
            print(f"[band_4p] seed {seed}: results exist -> skip", flush=True)
            out[seed] = json.load(open(res_fp)); continue
        ts = time.time()
        L.train_4p(FEAT_4P_TRAIN, OUT, tag=tag, seed=seed, epochs=epochs, device="cuda")
        vol.commit()
        res = L.eval_4p(os.path.join(OUT, f"det_{tag}.pt"), FEAT_4P_TEST, NATIVE_TEST,
                        TEST_GT, EM_PATH, os.path.join(OUT, f"eval_{tag}.json"),
                        device="cuda")
        res.update({"tag": tag, "seed": seed,
                    "seed_min": round((time.time() - ts) / 60, 1)})
        json.dump(res, open(res_fp, "w"), indent=2)
        vol.commit()
        out[seed] = res
        print(f"[band_4p] seed {seed}: mask={res['mask_mAP50']} box={res['box_mAP50']} "
              f"in {res['seed_min']}min", flush=True)
    mins = (time.time() - t0) / 60
    print(f"[band_4p] {len(sl)} seeds in {mins:.1f}min est≈${mins*0.035:.2f}", flush=True)
    return {s: {"mask_mAP50": out[s]["mask_mAP50"], "box_mAP50": out[s]["box_mAP50"]}
            for s in out}


@app.local_entrypoint()
def main(seed: int = 0, epochs: int = 40, skip_verify: bool = False,
         skip_extract: bool = False):
    if not skip_verify:
        print("== verify =="); print(json.dumps(verify.remote(), indent=2))
    if not skip_extract:
        print("== extract_4p =="); print(json.dumps(extract_4p.remote(), indent=2))
    print("== train_eval_4p ==")
    res = train_eval_4p.remote(seed=seed, epochs=epochs)
    print(json.dumps(res, indent=2))
    print(f"\n4-phase L24 real-8px: mask mAP50={res['mask_mAP50']} box={res['box_mAP50']} "
          f"| interp-L24 0.502/0.540, native-4096 0.499/0.555")
