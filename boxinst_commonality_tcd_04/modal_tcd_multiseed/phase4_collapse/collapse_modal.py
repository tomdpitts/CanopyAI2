"""Prototype-collapse ablation: is K=16 scaffolding, and does collapse buy box-robustness?

Sibling of phase4/ (same pattern as phase4_sam/): reuses phase4's Volume, features,
detector checkpoints and fit recipe, but writes ONLY under /vol/collapse/.

  modal run collapse_modal.py::fit_grid      # Stage 1  CPU  ~$2   K x EM-seed fits
  modal run collapse_modal.py::gate1         #          free  read Stage-1 gate
  modal run collapse_modal.py::eval_grid     # Stage 2  A100 ~$4   GT vs predicted boxes

WHY THE OUTPUT PATHS ARE NOT phase4's
    phase4_modal._selfmask_npz(beta, fix) keys on beta and fix but NOT on k or EM seed, so
    (beta=0.5, fix=True) resolves to /vol/out/em_model_4p_fix.npz -- THE DEPLOYED MASKER.
    A K=2 fit would target that exact path, and phase4_modal.fit_masker_4p's
    `if os.path.exists(npz): skip` would silently return the K=16 model labelled as K=2.
    So this app defines its own `_npz()` carrying k and em_seed, writes under /vol/collapse/,
    and `_assert_isolated()` refuses to run if any output path escapes that prefix.

STAGE 1 (CPU, no eval) -- the scaffolding question and the missing noise floor
    K in {2,16} x EM seed in {0,1,2} at beta=0.5. The deployed band's +/-0.005 is DETECTOR
    variance only (one masker, fit at seed 0, shared across detector seeds), so the spread of
    cos(C_bar_i, C_bar_j) between K=16 seeds is the first measurement of masker-init variance
    in this project. It is also the floor against which the K=2 vs K=16 difference -- and
    Stage 2's AP differences -- must be judged.

STAGE 2 (GPU) -- the sign flip at ONE config
    The two existing maskers (em_model_4p_b0_fix.npz, rank 10.83; em_model_4p_fix.npz,
    rank 1.66) evaluated under GT boxes and predicted boxes on the 108-tile val split.
    No refits. Existing evidence for the flip mixes 16px/local/GT-box with 8px/Modal/
    predicted-box, so box precision is confounded with stride and stack; this holds
    everything but box precision fixed.

    Val is genuinely held out from the masker, and this was checked rather than assumed:
    make_load_train_4p filters partition=="train" before taking the first 120 tiles, and
    train_tiles_gt.json marks all 108 val tiles partition=="val" -- so the fit/val overlap is
    ZERO. Those records also carry only boxes and canopy, no crown polygons, so the fit cannot
    read mask labels at all. eval_grid asserts the overlap is empty before spending.
"""
import json
import os

import modal

APP_NAME = "tcd04-phase4-collapse"
VOL_NAME = "tcd04-phase4-vol"                 # phase4's volume: features + checkpoints

HERE = os.path.dirname(os.path.abspath(__file__))
MTS = os.path.dirname(HERE)
PH4 = os.path.join(MTS, "phase4")
PKG = os.path.dirname(MTS)
REPO = os.path.dirname(PKG)
STUBS = os.path.join(PH4, "stubs")

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=False)

P = "/root/proj"
PKG_R = f"{P}/boxinst_commonality_tcd_04"
PH4_R = f"{PKG_R}/modal_tcd_multiseed/phase4"
COL_R = f"{PKG_R}/modal_tcd_multiseed/phase4_collapse"

# Image mirrors phase4_modal.py's exactly (same pins), plus this folder's lib. Rebuilt here
# rather than imported so the folder stays self-contained, as phase4_sam/ does.
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
image = image.add_local_file(os.path.join(MTS, "__init__.py"),
                             f"{PKG_R}/modal_tcd_multiseed/__init__.py")
for rel in ("__init__.py", "phase4_features_tcd.py", "phase4_lib_tcd.py",
            "phase4_fit_tcd.py", "manifest.json", "val_gt.json"):
    image = image.add_local_file(os.path.join(PH4, rel), f"{PH4_R}/{rel}")
for rel in ("__init__.py", "collapse_lib.py"):
    image = image.add_local_file(os.path.join(HERE, rel), f"{COL_R}/{rel}")
STUB_FILES = {
    "boxinst": ("__init__.py", "cache_feats.py"),
    "boxinst_commonality": ("__init__.py", "em.py"),
    "boxinst_tcd": ("__init__.py", "build_canopy.py", "cache.py", "prepare.py"),
}
for pkg, files in STUB_FILES.items():
    for f in files:
        image = image.add_local_file(os.path.join(STUBS, pkg, f), f"{P}/{pkg}/{f}")

VOL = "/vol"
FEAT_4P_TRAIN = f"{VOL}/feat_4p_train"        # holds BOTH the 792 train and 108 val tiles
OUT_PH4 = f"{VOL}/out"                        # phase4's outputs -- READ ONLY from here
COLLAPSE = f"{VOL}/collapse"                  # everything this app writes
TRAIN_GT = f"{PKG_R}/train_tiles_gt.json"
VAL_GT = f"{PH4_R}/val_gt.json"

BETA = 0.5
KS = (2, 16)
EM_SEEDS = (0, 1, 2)
DET_SEED = 0
FIT_TILES = 120                               # the deployed recipe


def _cell(k, em_seed, beta, contrast=None):
    """Cell name. EVERY axis that changes the fit is in the name -- see module docstring.

    `contrast` is the E-step within-box recentring flag. It is a SEPARATE mechanism from the
    M-step repulsion beta, but phase4_modal ties them (no_contrast = beta == 0), so the two
    existing arms only cover the diagonal of the 2x2. Cells fitted before this axis was
    exposed carry no _c tag and are implicitly contrast == (beta != 0); new cells always tag
    it, so nothing already computed can be overwritten.
    """
    base = f"k{k}_s{em_seed}_b{beta:g}"
    return base if contrast is None else f"{base}_c{int(bool(contrast))}"


def _npz(k, em_seed, beta=BETA, contrast=None):
    c = _cell(k, em_seed, beta, contrast)
    return f"{COLLAPSE}/{c}/em_{c}_fix.npz"


def _assert_isolated(*paths):
    """Refuse to write anywhere but /vol/collapse/. Guards against the silent-skip trap."""
    for p in paths:
        assert p.startswith(COLLAPSE + "/"), f"REFUSING: {p} escapes {COLLAPSE}"
        assert not p.startswith(OUT_PH4 + "/"), f"REFUSING: {p} is inside phase4 out"


def _setup_path():
    import sys
    if P not in sys.path:
        sys.path.insert(0, P)


# LOCAL paths for the same two files -- _fit_clean_val_tiles runs in the local entrypoint,
# not in the container, so it must not use the /root/proj paths.
TRAIN_GT_LOCAL = os.path.join(PKG, "train_tiles_gt.json")
VAL_GT_LOCAL = os.path.join(PH4, "val_gt.json")


def _fit_clean_val_tiles():
    """Val tids the masker fit could have seen. Verified to be EMPTY -- kept as an assertion.

    make_load_train_4p (phase4_fit_tcd.py) filters `v["partition"] == "train"` BEFORE taking
    the first n_tiles, and train_tiles_gt.json marks the 108 val tiles as partition="val", so
    no val tile can enter the masker fit. (A naive sorted(all 900)[:120] suggests an 11-tile
    overlap; that ignores the partition filter and is wrong.)

    Further: train_tiles_gt.json records carry only ['boxes', 'canopy', 'partition'] -- no
    crown polygons -- so the fit is structurally incapable of reading mask labels.

    Returns (clean, overlap). `overlap` must be empty; the caller asserts it.
    """
    import json as _j
    gt = _j.load(open(TRAIN_GT_LOCAL))
    va = set(_j.load(open(VAL_GT_LOCAL)))
    fit_tiles = set(sorted(t for t, v in gt.items()
                           if v["partition"] == "train")[:FIT_TILES])
    return sorted(va - fit_tiles), sorted(va & fit_tiles)


@app.function(image=image, volumes={"/vol": vol}, timeout=4 * 3600, cpu=8, memory=131072)
def _fit_one(k: int, em_seed: int, beta: float = BETA, n_tiles: int = FIT_TILES,
             contrast: bool = None):
    """One masker fit. CPU-only numpy EM -- no GPU, matching phase4_modal.fit_masker_4p."""
    import time
    import numpy as np
    _setup_path()
    from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4 import phase4_fit_tcd as F
    from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4_collapse import collapse_lib as CL

    npz = _npz(k, em_seed, beta, contrast)
    _assert_isolated(npz)
    # contrast=None reproduces phase4's coupling; an explicit value decouples the two axes
    use_contrast = (beta != 0) if contrast is None else bool(contrast)
    out_dir = os.path.dirname(npz)            # per-cell dir: em_fit_report.json is a FIXED
    os.makedirs(out_dir, exist_ok=True)       # name in em.py:292 and would otherwise clobber
    if os.path.exists(npz):
        print(f"[fit] {npz} exists -> reuse", flush=True)
    else:
        t0 = time.time()
        F.fit_masker_4p(FEAT_4P_TRAIN, TRAIN_GT, out_dir, npz, n_tiles=n_tiles,
                        seed=em_seed, k=k, contrastive_beta=beta,
                        no_contrast=(not use_contrast))
        print(f"[fit] k={k} s={em_seed} b={beta:g} contrast={use_contrast} done in "
              f"{(time.time()-t0)/60:.1f}min", flush=True)
        vol.commit()
    st = CL.prototype_stats(np.load(npz, allow_pickle=True)["C"])
    st.update({"k_init": k, "em_seed": em_seed, "beta": beta, "contrast": use_contrast,
               "cell": _cell(k, em_seed, beta, contrast), "npz": npz})
    json.dump(st, open(os.path.join(out_dir, "proto_stats.json"), "w"), indent=2)
    vol.commit()
    print(f"[fit] k={k} s={em_seed}: K_eff={st['K']} rank={st['effective_rank']} "
          f"cos={st['pairwise_cos_mean']}", flush=True)
    return st


@app.local_entrypoint()
def fit_grid(ks: str = "2,16", seeds: str = "0,1,2", beta: float = BETA):
    """STAGE 1 -- the K x EM-seed fit grid. CPU only, no eval, ~$2."""
    kl = [int(x) for x in ks.split(",")]
    sl = [int(x) for x in seeds.split(",")]
    cells = [(k, s) for k in kl for s in sl]
    print(f"[fit_grid] {len(cells)} fits: k={kl} x seed={sl} beta={beta}", flush=True)
    rows = list(_fit_one.starmap([(k, s, beta) for k, s in cells]))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(rows, open(os.path.join(HERE, "results", "stage1_fits.json"), "w"), indent=2)
    _report_gate1(rows)


def _report_gate1(rows):
    """Gate G1: is K=16 scaffolding, decoration, or is EM init itself unstable?

    The comparison is between-K agreement vs WITHIN-K=16 agreement across EM seeds. Using
    mean-vs-min (not min-vs-min) matters: with only 3 seeds there are 3 pairs, and the
    minimum of 3 samples is noisy enough to flip the verdict on its own.
    """
    import itertools
    import math

    def _cos(a, b):
        n = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
        return sum(x * y for x, y in zip(a, b)) / (n + 1e-12)

    def _mean(v):
        return sum(v) / len(v) if v else float("nan")

    by_k = {}
    for r in rows:
        by_k.setdefault(r["k_init"], []).append(r)
    print(f"\n{'k':>4}{'seed':>6}{'K_eff':>7}{'eff rank':>10}{'cos mean':>10}")
    for k in sorted(by_k):
        for r in sorted(by_k[k], key=lambda x: x["em_seed"]):
            print(f"{k:>4}{r['em_seed']:>6}{r['K']:>7}{r['effective_rank']:>10.3f}"
                  f"{r['pairwise_cos_mean']:>10.4f}")

    within = {k: [_cos(a["C_bar"], b["C_bar"]) for a, b in itertools.combinations(rs, 2)]
              for k, rs in by_k.items()}
    print("\nEM-init stability -- cos(C_bar) between seeds of the SAME k:")
    for k in sorted(within):
        if within[k]:
            print(f"  k={k:<3} mean {_mean(within[k]):.4f}  min {min(within[k]):.4f}  "
                  f"(n={len(within[k])} pairs)")
    if not (2 in by_k and 16 in by_k):
        print("\nGATE G1: need both k=2 and k=16 to judge; rerun the full grid.")
        return
    floor_mean, floor_min = _mean(within[16]), min(within[16])
    cross = [_cos(a["C_bar"], b["C_bar"]) for a in by_k[2] for b in by_k[16]]
    print(f"\ncross k=2 vs k=16 cos(C_bar): mean {_mean(cross):.4f}  min {min(cross):.4f}")

    print("\nGATE G1: ", end="")
    if floor_mean < 0.80:
        print(f"EM INIT UNSTABLE (k=16 seeds agree only at {floor_mean:.3f}).\n"
              "  The shared-masker design is fragile and the published +/-0.005 band "
              "understates true variance.\n  STOP -- that is a larger finding than either "
              "question here; reassess before Stage 2.")
    elif _mean(cross) >= floor_min:
        print(f"MIXTURE REMOVABLE -- cross agreement {_mean(cross):.4f} is within the "
              f"k=16 seed floor [{floor_min:.4f}, {_mean(within[16]):.4f}].\n"
              "  k=16 is decoration; k=2 finds the same direction. Stage 2 runs beta arms "
              "only; write the simplification result.")
    else:
        print(f"SCAFFOLDING CANDIDATE -- cross agreement {_mean(cross):.4f} sits below the "
              f"k=16 seed floor {floor_min:.4f}.\n"
              "  Over-provisioning changes WHERE the fit lands. Stage 2 should add the k=2 "
              "arm (--include-k2 2,0) to price it in AP.")


@app.function(gpu="A100", image=image, volumes={"/vol": vol}, timeout=6 * 3600,
              cpu=8, memory=65536)
def _eval_one(em_path: str, box_source: str, tag: str, exclude: list,
              mask_thr: float = 0.25):
    _setup_path()
    from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4_collapse import collapse_lib as CL
    out = f"{COLLAPSE}/eval/{tag}.json"
    _assert_isolated(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res = CL.eval_boxsource(em_path, FEAT_4P_TRAIN, VAL_GT, box_source, out,
                            ckpt_path=f"{OUT_PH4}/det_phase4_L24_s{DET_SEED}.pt",
                            mask_thr=mask_thr, device="cuda", exclude=exclude)
    res["tag"] = tag
    vol.commit()
    return res


@app.local_entrypoint()
def eval_grid(include_k2: str = ""):
    """STAGE 2 -- GT vs predicted boxes for the two existing maskers, on the val split.

    include_k2: "k,seed" (e.g. "2,0") adds the Stage-1 k=2 masker as a third arm -- pass it
    only if gate G1 came back SCAFFOLDING CANDIDATE.
    """
    arms = [("b0", f"{OUT_PH4}/em_model_4p_b0_fix.npz"),
            ("b05", f"{OUT_PH4}/em_model_4p_fix.npz")]
    if include_k2:
        k, s = [int(x) for x in include_k2.split(",")]
        arms.append((f"k{k}s{s}", _npz(k, s)))
    clean, overlap = _fit_clean_val_tiles()
    assert not overlap, (f"{len(overlap)} val tiles entered the masker fit -- the partition "
                         f"filter in make_load_train_4p is not holding: {overlap[:5]}")
    print(f"[eval_grid] val: {len(clean)} tiles, none seen by the masker fit "
          f"(partition filter verified)", flush=True)
    # One pass per (arm, box source): with no fit/val overlap there is no clean-vs-all split
    # to make, so the earlier double-scoring is unnecessary -- and halves the GPU bill.
    jobs = [(em, src, f"{name}_{src}_all", [])
            for name, em in arms
            for src in ("gt", "pred")]
    rows = list(_eval_one.starmap(jobs))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(rows, open(os.path.join(HERE, "results", "stage2_boxsource.json"), "w"),
              indent=2)
    print(f"\n{'arm':<18}{'crownIoU':>10}{'AP50':>9}{'AP50-95':>10}{'nTile':>7}")
    for r in rows:
        print(f"{r['tag']:<18}{r['mean_crown_iou']:>10.4f}{r['mask_mAP50']:>9.4f}"
              f"{r['mask_mAP50_95']:>10.4f}{r['n_tiles']:>7}")
    _report_gate2(rows)


def _report_gate2(rows):
    d = {r["tag"]: r for r in rows}
    print("\nGATE G2 (mean crown IoU, beta=0 -> beta=0.5):")
    for scope in ("all", "clean"):
        try:
            gt = d[f"b05_gt_{scope}"]["mean_crown_iou"] - d[f"b0_gt_{scope}"]["mean_crown_iou"]
            pr = d[f"b05_pred_{scope}"]["mean_crown_iou"] - d[f"b0_pred_{scope}"]["mean_crown_iou"]
        except KeyError:
            continue
        verdict = ("SIGN FLIP CONFIRMED" if gt < 0 < pr else
                   "no flip -- same sign" if gt * pr > 0 else "ambiguous")
        print(f"  [{scope:5s}] GT-box delta {gt:+.4f}   predicted-box delta {pr:+.4f}"
              f"   -> {verdict}")
    print("  Judge both deltas against the Stage-1 noise floor before believing either.")


@app.local_entrypoint()
def factorial(k: int = 16, em_seed: int = 0, evals: str = "1"):
    """The 2x2 component ablation: M-step repulsion x E-step recentring.

    phase4 ties the two (no_contrast = beta == 0), so the recorded arms are only the
    DIAGONAL -- (beta=0.5, recentred) at 0.6203 and (beta=0, absolute) at 0.5790. The joint
    +0.0413 AP50 / +0.0483 crown IoU therefore cannot be attributed to either component. This
    fills the two off-diagonal cells so the split is measurable.

    Prior expectation from the dryland proxies (boxinst_commonality/README.md:107-119), where
    the two ARE separated: removing repulsion moves `corner` 0.37 -> 0.41, removing recentring
    moves it 0.37 -> 0.61. Recentring looks like the dominant term, which would mean the
    accuracy gain is mostly E-step and prototype collapse is close to free.

    Only the two missing cells are fitted; the diagonal is read from the existing arms.
    """
    cells = [(0.0, True), (0.5, False)]          # off-diagonal only
    print(f"[factorial] fitting {len(cells)} missing cells at k={k} seed={em_seed}", flush=True)
    for beta, contrast in cells:
        _assert_isolated(_npz(k, em_seed, beta, contrast))
    rows = list(_fit_one.starmap([(k, em_seed, b, FIT_TILES, c) for b, c in cells]))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(rows, open(os.path.join(HERE, "results", "factorial_fits.json"), "w"), indent=2)
    print(f"\n{'cell':<22}{'K_eff':>7}{'eff rank':>10}{'cos mean':>10}")
    for r in rows:
        print(f"{r['cell']:<22}{r['K']:>7}{r['effective_rank']:>10.3f}"
              f"{r['pairwise_cos_mean']:>10.4f}")
    if evals != "1":
        return
    jobs = []
    for beta, contrast in cells:
        em = _npz(k, em_seed, beta, contrast)
        tag = _cell(k, em_seed, beta, contrast)
        for src in ("gt", "pred"):
            jobs.append((em, src, f"{tag}_{src}", []))
    ev = list(_eval_one.starmap(jobs))
    json.dump(ev, open(os.path.join(HERE, "results", "factorial_evals.json"), "w"), indent=2)
    print(f"\n{'arm':<28}{'crownIoU':>10}{'AP50':>9}")
    for r in ev:
        print(f"{r['tag']:<28}{r['mean_crown_iou']:>10.4f}{r['mask_mAP50']:>9.4f}")
    _report_factorial(ev)


def _report_factorial(ev):
    """Split the joint gain into its M-step and E-step parts (GT-box crown IoU)."""
    d = {r["tag"]: r["mean_crown_iou"] for r in ev}
    base = {"gt": (0.7402, 0.7885), "pred": (0.6717, 0.7193)}   # (beta0/absolute, beta0.5/recentred)
    print("\nCOMPONENT SPLIT (mean crown IoU)")
    for src in ("gt", "pred"):
        lo, hi = base[src]
        rec_only = next((v for t, v in d.items() if t.endswith(f"_{src}") and "_b0_c1" in t), None)
        rep_only = next((v for t, v in d.items() if t.endswith(f"_{src}") and "_b0.5_c0" in t), None)
        if rec_only is None or rep_only is None:
            continue
        print(f"  [{src:4s}] both off {lo:.4f} | recentring only {rec_only:.4f} | "
              f"repulsion only {rep_only:.4f} | both on {hi:.4f}")
        print(f"         recentring alone {rec_only - lo:+.4f}   repulsion alone "
              f"{rep_only - lo:+.4f}   joint {hi - lo:+.4f}")
        dom = "RECENTRING" if (rec_only - lo) > (rep_only - lo) else "REPULSION"
        print(f"         -> {dom} dominates; interaction "
              f"{(hi - lo) - (rec_only - lo) - (rep_only - lo):+.4f}")


@app.local_entrypoint()
def beta_sweep(betas: str = "0.1,0.25,0.5", k: int = 16, em_seed: int = 0, evals: str = "1"):
    """Single-axis beta sweep at 8px with E-step recentring HELD ON for every arm.

    No previous sweep did this. masker_lab/sweep.py (16px) sets no_contrast=(beta == 0), so its
    beta=0 row is generative+absolute while the rest are contrastive+recentred -- the 0->0.1
    drop there conflates adding recentring with adding repulsion. Holding contrast=True across
    all beta isolates the repulsion term, which the `contrast` axis now makes possible.

    Motivation: with recentring on, the two endpoints TIE at 8px (predicted-box crown IoU
    0.7188 at beta=0 vs 0.7193 at beta=0.5), and the 16px sweep shows collapse is graded rather
    than binary (effective rank 13.0 / 11.1 / 5.4 / 1.7 at beta 0 / 0.1 / 0.25 / 0.5). Equal
    endpoints plus graded geometry is the shape in which an interior optimum can hide.

    beta=0 and beta=0.5 at contrast=True already exist and are reused, not refitted.
    """
    bl = [float(b) for b in betas.split(",")]
    print(f"[beta_sweep] k={k} seed={em_seed} betas={bl} contrast=True", flush=True)
    for b in bl:
        _assert_isolated(_npz(k, em_seed, b, True))
    rows = list(_fit_one.starmap([(k, em_seed, b, FIT_TILES, True) for b in bl]))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(rows, open(os.path.join(HERE, "results", "beta_sweep_fits.json"), "w"), indent=2)
    print(f"\n{'cell':<24}{'K_eff':>7}{'eff rank':>10}{'cos mean':>10}")
    for r in sorted(rows, key=lambda x: x["beta"]):
        print(f"{r['cell']:<24}{r['K']:>7}{r['effective_rank']:>10.3f}"
              f"{r['pairwise_cos_mean']:>10.4f}")
    if evals != "1":
        return
    jobs = []
    for b in bl:
        em, tag = _npz(k, em_seed, b, True), _cell(k, em_seed, b, True)
        for src in ("gt", "pred"):
            jobs.append((em, src, f"{tag}_{src}", []))
    ev = list(_eval_one.starmap(jobs))
    json.dump(ev, open(os.path.join(HERE, "results", "beta_sweep_evals.json"), "w"), indent=2)
    # Fallbacks only. beta=0 was fitted by THIS harness in the factorial run
    # (cell k16_s0_b0_c1); the deployed beta=0.5 model was fitted by phase4, so if 0.5 is in
    # the sweep it is refitted here and the freshly measured value must win.
    known = {0.0: (0.7895, 0.7188)}
    got = {}
    for r in ev:
        b = float(r["tag"].split("_b")[1].split("_c")[0])
        got.setdefault(b, {})["gt" if r["tag"].endswith("_gt") else "pred"] = r["mean_crown_iou"]
    for b, (o, pr) in known.items():
        g = got.setdefault(b, {})
        g.setdefault("gt", o)          # setdefault, not update: never clobber a measured cell
        g.setdefault("pred", pr)
    print(f"\n{'beta':>6}{'oracle IoU':>13}{'pred IoU':>11}")
    for b in sorted(got):
        g = got[b]
        print(f"{b:>6.2f}{g.get('gt', float('nan')):>13.4f}{g.get('pred', float('nan')):>11.4f}")
    best = max(got, key=lambda b: got[b].get("pred", -1))
    interior = best not in (min(got), max(got))
    print(f"\n-> best predicted-box beta = {best:g}"
          f"{'  INTERIOR OPTIMUM' if interior else '  (endpoint; no interior optimum)'}")


@app.local_entrypoint()
def eval_cell(k: int = 2, em_seed: int = 0, beta: float = 0.0, contrast: str = "1",
              src: str = "pred"):
    """Re-run ONE eval cell, disconnect-proof.

    Uses .spawn() rather than .remote()/.starmap(): the local client dispatches and exits
    immediately, so nothing client-side is holding the app open and a dropped connection
    cannot take the run down with it. That is what killed the k2 predicted-box eval at
    100/108 -- --detach alone did not survive it, because the entrypoint was still iterating
    starmap results when the socket died.

    Writes exactly one file, /vol/collapse/eval/<cell>_<src>.json. Check with
    `modal volume ls tcd04-phase4-vol collapse/eval` that it does not already exist -- this
    does not guard against overwriting, because the volume is not visible from the client.
    """
    c = bool(int(contrast))
    em = _npz(k, em_seed, beta, c)
    tag = f"{_cell(k, em_seed, beta, c)}_{src}"
    out = f"{COLLAPSE}/eval/{tag}.json"
    _assert_isolated(em, out)
    assert src in ("gt", "pred")
    print(f"[eval_cell] em  = {em}")
    print(f"[eval_cell] out = {out}")
    call = _eval_one.spawn(em, src, tag, [])
    print(f"[eval_cell] spawned {call.object_id} -- client exiting; poll the volume for the "
          f"output file.")


@app.local_entrypoint()
def beta_band(seeds: str = "0,1,2"):
    """3-EM-seed band for beta=0 vs beta=0.5 at K=16 -- the noise floor AND the beta claim.

    Two quantities from one run:
      1. Masker-seed variance. The published 0.630 +/- 0.005 varies the DETECTOR and holds one
         masker (em_model_4p_fix.npz, EM seed 0) fixed, so it contains no masker variance at
         all. Every masker-swap comparison in this folder has therefore been judged without a
         floor. Refitting at three EM seeds with the detector fixed isolates it.
      2. The beta claim with error bars. The beta sweep (0/0.1/0.25/0.5, spread 0.0011 while
         effective rank falls 11.4 -> 1.7) is single-seed; banding the two endpoints is what
         makes "beta is inert" reportable rather than observational.

    Reuses everything already on the volume: the three beta=0.5 fits from fit_grid, the beta=0
    seed-0 fit from factorial, and its predicted-box eval. Only the two missing beta=0 fits and
    the five missing evals are run. Nothing existing is overwritten -- verified against a live
    volume listing before launch.
    """
    sl = [int(x) for x in seeds.split(",")]
    # beta=0 arms are tagged _c1; beta=0.5 arms are the untagged fit_grid cells (contrast is
    # implicitly True there, since phase4 couples no_contrast = beta == 0)
    need_fits = [(16, s, 0.0, FIT_TILES, True) for s in sl if s != 0]
    for a in need_fits:
        _assert_isolated(_npz(a[0], a[1], a[2], a[4]))
    if need_fits:
        print(f"[beta_band] fitting {len(need_fits)} missing beta=0 cells", flush=True)
        for r in _fit_one.starmap(need_fits):
            print(f"  fitted {r['cell']}: K_eff={r['K']} rank={r['effective_rank']:.3f}",
                  flush=True)

    jobs = []
    for s in sl:
        for beta, contrast in ((0.0, True), (0.5, None)):
            em = _npz(16, s, beta, contrast)
            tag = f"{_cell(16, s, beta, contrast)}_pred"
            if beta == 0.0 and s == 0:
                print(f"  skip {tag} (already measured, 0.7188 -- not overwriting)", flush=True)
                continue
            _assert_isolated(em, f"{COLLAPSE}/eval/{tag}.json")
            jobs.append((em, tag))
    print(f"[beta_band] spawning {len(jobs)} predicted-box evals", flush=True)
    for em, tag in jobs:
        call = _eval_one.spawn(em, "pred", tag, [])
        print(f"  spawned {tag}  ({call.object_id})", flush=True)
    print("[beta_band] client exiting; poll the volume for eval/*.json", flush=True)
