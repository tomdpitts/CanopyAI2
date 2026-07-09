"""Label-efficiency probe: does a directional shadow prior left-shift the curve?

Frozen DINOv3 (web ViT-L) features, cached once. A linear head is trained on the
azimuth cohort (data/finetune, 500px tiles, boxes+azimuth) under three rungs --
features only / features+correct-shadow / features+within-acquisition-shuffled-
shadow -- across a sweep of training-set sizes N (distinct scenes).

Thesis: the *correct* shadow channel reaches a target F1 at smaller N than the
no-prior and shuffled rungs (a left-shift). correct-vs-shuffled isolates
direction information from added channel capacity. Unit of analysis = scene;
folds are scene-clustered (stratified by acquisition). Primary stat at each N is
the Nadeau-Bengio corrected resampled t-test on the per-fold correct-shuffled
deltas.

NOTE: across-scene axis only (the materialised cohort has one rotation per scene;
the clean within-acquisition rotation axis must be regenerated). Metric is
box-union foreground F1, a fast detection proxy; crown AP50 is the follow-up.

Env overrides: N_SWEEP, N_FOLDS, SEEDS, STEPS, DINO_MODEL.
"""
from __future__ import annotations

import collections
import csv
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dino"))
from shadow_prior.config import ShadowFeatureConfig          # noqa: E402
from shadow_prior.geometry import vector_to_azimuth           # noqa: E402
from shadow_prior.shadow_feature import compute_shadow_feature, shuffle_azimuths  # noqa: E402
from shadow_prior.folds import make_scene_clustered_folds     # noqa: E402
from shadow_prior.stats import corrected_resampled_ttest      # noqa: E402
from dinov3_seg import Dinov3Seg                               # noqa: E402

MODEL = os.environ.get("DINO_MODEL", "facebook/dinov3-vitl16-pretrain-lvd1689m")
CSV_FILES = ["data/finetune/phase22X_train.csv", "data/finetune/phase22X_val_fixed.csv"]
IMG = 512                       # resize 500 -> 512 (divisible by patch 16)
SHADOW_CFG = ShadowFeatureConfig(offset_min=10.0, offset_max=80.0, offset_steps=14,
                                 aggregation="max", n_channels=2)  # crown-scale bracket; crown+shadow channels
N_SWEEP = [int(x) for x in os.environ.get("N_SWEEP", "10,30,60,90").split(",")]
N_FOLDS = int(os.environ.get("N_FOLDS", "5"))
SEEDS = list(range(int(os.environ.get("SEEDS", "3"))))
STEPS = int(os.environ.get("STEPS", "400"))
LR = 1e-3
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RUNGS = ("none", "correct", "shuffled")
ARTIFACT = "dino/artifacts/shadow_probe.json"


# ---- data ------------------------------------------------------------------ #
def base_scene(p):
    return re.sub(r"_rot\d+$", "", os.path.splitext(os.path.basename(p))[0])


def load_cohort():
    rows = []
    for fn in CSV_FILES:
        if os.path.exists(fn):
            rows += list(csv.DictReader(open(fn)))
    by_img = collections.OrderedDict()
    for r in rows:
        if r["shadow_angle"].strip() == "":
            continue
        d = by_img.setdefault(r["image_path"], {"boxes": [], "domain": r["domain"],
                                                 "sx": float(r["shadow_x"]), "sy": float(r["shadow_y"])})
        if all(r[k].strip() != "" for k in ("xmin", "ymin", "xmax", "ymax")):
            x0, y0, x1, y1 = (float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
            if x1 > x0 and y1 > y0:
                d["boxes"].append([x0, y0, x1, y1])
    recs = []
    for p, d in by_img.items():
        if not d["boxes"]:
            continue
        recs.append({"path": p, "scene": base_scene(p), "domain": d["domain"],
                     "azimuth": vector_to_azimuth(d["sx"], d["sy"]),
                     "boxes": np.array(d["boxes"], dtype=np.float32)})
    return recs


def box_mask(boxes, src_size=500):
    s = IMG / src_size
    m = np.zeros((IMG, IMG), np.int64)
    for x0, y0, x1, y1 in boxes * s:
        m[int(y0):int(round(y1)), int(x0):int(round(x1))] = 1
    return m


def pool_shadow(rgb, azimuth):
    s = compute_shadow_feature(rgb, azimuth, SHADOW_CFG).astype(np.float32)   # (k,H,W)
    return F.avg_pool2d(torch.from_numpy(s)[None], kernel_size=16, stride=16)[0]  # (k,h,w)


# ---- feature cache (DINOv3 frozen, computed once) -------------------------- #
def build_cache(recs, model):
    cache = {}
    for i, r in enumerate(recs):
        rgb = np.asarray(Image.open(r["path"]).convert("RGB").resize((IMG, IMG)), np.float32) / 255.0
        x = torch.from_numpy(rgb.transpose(2, 0, 1))[None].to(DEVICE)
        cache[r["path"]] = {"rgb": rgb, "feat": model.features(x)[0].cpu(),
                            "shadow": pool_shadow(rgb, r["azimuth"]),     # correct-azimuth shadow
                            "mask": torch.from_numpy(box_mask(r["boxes"]))}
        if i % 40 == 0:
            print(f"[cache] {i}/{len(recs)}", flush=True)
    return cache


def shuffled_shadow_lookup(items, cache, az_over):
    """Precompute the within-acquisition-shuffled shadow once per run (perf + covers test)."""
    return {r["path"]: pool_shadow(cache[r["path"]]["rgb"], az_over[r["scene"]]) for r in items}


# ---- head ------------------------------------------------------------------ #
def stack(cache, rec, rung, shuf_lookup):
    feat = cache[rec["path"]]["feat"]
    if rung == "none":
        return feat
    sh = cache[rec["path"]]["shadow"] if rung == "correct" else shuf_lookup[rec["path"]]
    return torch.cat([feat, sh], dim=0)


def class_weight(items, cache):
    fg = float(np.mean([cache[r["path"]]["mask"].float().mean().item() for r in items])) + 1e-6
    return torch.tensor([1.0, min(20.0, (1 - fg) / fg)], device=DEVICE)


def train_head(items, cache, rung, in_ch, seed, shuf_lookup):
    torch.manual_seed(seed)
    head = nn.Conv2d(in_ch, 2, kernel_size=1).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=class_weight(items, cache))
    rng = np.random.default_rng(seed)
    head.train()
    for _ in range(STEPS):
        batch = [items[j] for j in rng.choice(len(items), size=min(16, len(items)), replace=False)]
        xs = torch.stack([stack(cache, r, rung, shuf_lookup) for r in batch]).to(DEVICE)
        ys = torch.stack([cache[r["path"]]["mask"] for r in batch]).to(DEVICE)
        logits = F.interpolate(head(xs), size=(IMG, IMG), mode="bilinear", align_corners=False)
        loss = lossf(logits, ys)
        opt.zero_grad(); loss.backward(); opt.step()
    return head


@torch.no_grad()
def eval_head(head, items, cache, rung, shuf_lookup):
    head.eval()
    tp = fp = fn = 0
    for r in items:
        x = stack(cache, r, rung, shuf_lookup)[None].to(DEVICE)
        logits = F.interpolate(head(x), size=(IMG, IMG), mode="bilinear", align_corners=False)
        pred = (logits[0, 1] > logits[0, 0]).cpu().numpy()
        g = cache[r["path"]]["mask"].numpy().astype(bool)
        tp += int((pred & g).sum()); fp += int((pred & ~g).sum()); fn += int((~pred & g).sum())
    return 2 * tp / (2 * tp + fp + fn + 1e-9)


# ---- driver ---------------------------------------------------------------- #
def main():
    t0 = time.time()
    recs = load_cohort()
    scenes = np.array([r["scene"] for r in recs])
    acqs = np.array([r["domain"] for r in recs], dtype=object)
    print(f"[data] {len(recs)} scenes | domains={dict(collections.Counter(acqs))} "
          f"N_SWEEP={N_SWEEP} folds={N_FOLDS} seeds={len(SEEDS)} steps={STEPS} dev={DEVICE}", flush=True)

    model = Dinov3Seg(MODEL, device=DEVICE)
    cache = build_cache(recs, model)
    C = cache[recs[0]["path"]]["feat"].shape[0]
    k = cache[recs[0]["path"]]["shadow"].shape[0]
    in_ch = {"none": C, "correct": C + k, "shuffled": C + k}
    print(f"[cache] done C={C} k={k} ({time.time()-t0:.0f}s)", flush=True)

    fa = make_scene_clustered_folds(scenes, n_splits=N_FOLDS, acquisition_ids=acqs, seed=0)
    ntr, nte = fa.train_test_scene_counts(0)
    res = {rg: {N: [] for N in N_SWEEP} for rg in RUNGS}
    fold_delta = {N: [] for N in N_SWEEP}

    for fold in range(N_FOLDS):
        tr = [recs[i] for i in fa.train_indices(fold)]
        te = [recs[i] for i in fa.test_indices(fold)]
        tr_scenes = list(dict.fromkeys(r["scene"] for r in tr))
        for N in N_SWEEP:
            Nuse = min(N, len(tr_scenes))
            per_seed = {rg: [] for rg in RUNGS}
            for seed in SEEDS:
                rng = np.random.default_rng(1000 * fold + seed)
                pick = set(rng.choice(tr_scenes, size=Nuse, replace=False))
                items = [r for r in tr if r["scene"] in pick]
                # within-acquisition shuffled azimuth over train-pick + test scenes
                union = items + te
                u_sc = list(dict.fromkeys(r["scene"] for r in union))
                u_acq = {r["scene"]: r["domain"] for r in union}
                u_az = {r["scene"]: r["azimuth"] for r in union}
                shuf = shuffle_azimuths(np.array([u_az[s] for s in u_sc]),
                                        np.array([u_acq[s] for s in u_sc], dtype=object), seed=seed)
                az_over = {s: float(v) for s, v in zip(u_sc, shuf)}
                shuf_lookup = shuffled_shadow_lookup(union, cache, az_over)
                for rg in RUNGS:
                    head = train_head(items, cache, rg, in_ch[rg], seed, shuf_lookup)
                    f1 = eval_head(head, te, cache, rg, shuf_lookup)
                    res[rg][N].append(f1); per_seed[rg].append(f1)
            d = float(np.mean(per_seed["correct"]) - np.mean(per_seed["shuffled"]))
            fold_delta[N].append(d)
            print(f"[fold {fold} N={N:3d}] none={np.mean(per_seed['none']):.3f} "
                  f"correct={np.mean(per_seed['correct']):.3f} shuffled={np.mean(per_seed['shuffled']):.3f} "
                  f"Δ(c-s)={d:+.3f}", flush=True)

    out = {"model": MODEL, "n_sweep": N_SWEEP, "n_folds": N_FOLDS, "seeds": len(SEEDS), "steps": STEPS,
           "curve": {rg: {N: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                          for N, v in res[rg].items()} for rg in RUNGS},
           "stats_correct_vs_shuffled": {}}
    for N in N_SWEEP:
        d = np.array(fold_delta[N])
        try:
            out["stats_correct_vs_shuffled"][N] = corrected_resampled_ttest(d, ntr, nte).to_dict()
        except Exception as e:
            out["stats_correct_vs_shuffled"][N] = {"mean_delta": float(d.mean()), "error": str(e)}
    out["wall_clock_s"] = round(time.time() - t0, 1)
    os.makedirs("dino/artifacts", exist_ok=True)
    json.dump(out, open(ARTIFACT, "w"), indent=2)

    print("\n===== EFFICIENCY CURVE (mean F1) =====", flush=True)
    print(f"{'N':>5} | {'none':>7} {'correct':>8} {'shuffled':>9} | {'Δ(c-s)':>8} {'NB p':>7}", flush=True)
    for N in N_SWEEP:
        c = out["curve"]; st = out["stats_correct_vs_shuffled"][N]
        print(f"{N:>5} | {c['none'][N]['mean']:>7.3f} {c['correct'][N]['mean']:>8.3f} "
              f"{c['shuffled'][N]['mean']:>9.3f} | {st.get('mean_delta', float('nan')):>+8.3f} "
              f"{st.get('p_value', float('nan')):>7.3f}", flush=True)
    print(f"\nartifact: {ARTIFACT} | {out['wall_clock_s']}s", flush=True)


if __name__ == "__main__":
    main()
