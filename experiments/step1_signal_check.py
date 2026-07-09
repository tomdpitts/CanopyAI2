"""Step 1 -- privileged-input signal check (detection) on the finetune cohort.

Asks the gating question cheaply: when we hand the detector the *correct* shadow
feature as an extra input channel, does it beat the *within-acquisition-shuffled*
feature (rung 2 vs rung 3) above the seed-noise floor, on scene-clustered CV?

This is the upper bound from decision-history: if direction can't help even when
handed in directly, no train-time conditioning architecture will. A null here (or
effect < MDE) kills the idea; a positive sets the ceiling worth engineering toward.

Detector: torchvision RetinaNet (ResNet-50-FPN), COCO-pretrained, as a stand-in for
DeepForest weights (swapping in DeepForest's backbone is a one-line change). The
input stem is inflated 3 -> 3+k channels with the shadow channels zero-initialised,
so rung-1 (RGB) starts identical to stock. Only the input channels differ across
rungs; architecture/schedule/seed are held fixed.

NOTE this is a *reduced* pilot (few folds/epochs, CPU/MPS): it proves the loop and
gives a first, likely-underpowered read. The seed-variance MDE tells us whether the
pilot is even powered to conclude anything.
"""

from __future__ import annotations

import os, sys, csv, re, json, time, math, collections
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shadow_prior.config import ShadowFeatureConfig
from shadow_prior.geometry import vector_to_azimuth
from shadow_prior.shadow_feature import compute_shadow_feature, shuffle_azimuths
from shadow_prior.folds import make_scene_clustered_folds
from shadow_prior.stats import corrected_resampled_ttest, permutation_test_paired, seed_variance

# ---- pilot config (override via env) --------------------------------------- #
N_FOLDS    = int(os.environ.get("N_FOLDS", 3))
EPOCHS     = int(os.environ.get("EPOCHS", 30))
SEED_RUNS  = int(os.environ.get("SEED_RUNS", 3))    # seeds PER (fold,rung) cell now
LR         = float(os.environ.get("LR", 1e-3))      # AdamW head LR
WARMUP_EP  = float(os.environ.get("WARMUP_EP", 3))  # epochs of linear LR warmup
MIN_SIZE   = int(os.environ.get("MIN_SIZE", 384))
TRAIN_LAYERS = int(os.environ.get("TRAIN_LAYERS", 1))  # trainable backbone stages (0-5)
COLLAPSE_EPS = float(os.environ.get("COLLAPSE_EPS", 0.05))  # AP50 below this = collapse
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
CSV_FILES = ["data/finetune/phase22X_train.csv", "data/finetune/phase22X_val.csv"]
SHADOW_CFG = ShadowFeatureConfig(offset_min=2.0, offset_max=25.0, offset_steps=8,
                                 aggregation="max", n_channels=1)
ARTIFACT = "artifacts/step1_signal_check.json"


# ---- data ------------------------------------------------------------------ #
def base_scene(p):
    stem = os.path.splitext(os.path.basename(p))[0]
    return re.sub(r"_rot\d+$", "", stem)


def load_records():
    rows = []
    for fn in CSV_FILES:
        rows += list(csv.DictReader(open(fn)))
    by_img = collections.OrderedDict()
    for r in rows:
        if r["shadow_angle"].strip() == "":
            continue  # azimuth-bearing tiles only
        p = r["image_path"]
        d = by_img.setdefault(p, {"boxes": [], "domain": r["domain"],
                                  "sx": float(r["shadow_x"]), "sy": float(r["shadow_y"])})
        if any(r[k].strip() == "" for k in ("xmin", "ymin", "xmax", "ymax")):
            continue  # azimuth present but no box on this row
        x0, y0, x1, y1 = (float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
        if x1 > x0 and y1 > y0:
            d["boxes"].append([x0, y0, x1, y1])
    recs = []
    for p, d in by_img.items():
        if not d["boxes"]:
            continue
        recs.append({"path": p, "scene": base_scene(p), "acq": d["domain"],
                     "azimuth": vector_to_azimuth(d["sx"], d["sy"]),
                     "boxes": np.array(d["boxes"], dtype=np.float32)})
    return recs


class DetDataset(Dataset):
    """Emits (image CxHxW float, target{boxes,labels}). Rung sets the extra channel:
    rgb -> none; correct -> shadow feat at true azimuth; shuffled -> at permuted az."""
    def __init__(self, recs, rung, scene_azimuth_override=None):
        self.recs = recs
        self.rung = rung
        self.az_override = scene_azimuth_override or {}
        self._cache = {}

    def __len__(self):
        return len(self.recs)

    def _rgb(self, path):
        if path not in self._cache:
            self._cache[path] = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return self._cache[path]

    def __getitem__(self, i):
        r = self.recs[i]
        rgb = self._rgb(r["path"])                      # HxWx3 in [0,1]
        chw = np.transpose(rgb, (2, 0, 1))
        if self.rung == "rgb":
            stack = chw
        else:
            az = self.az_override.get(r["scene"], r["azimuth"]) if self.rung == "shuffled" else r["azimuth"]
            feat = compute_shadow_feature(rgb, az, SHADOW_CFG)   # (k,H,W)
            stack = np.concatenate([chw, feat.astype(np.float32)], axis=0)
        target = {"boxes": torch.as_tensor(r["boxes"]),
                  "labels": torch.ones((len(r["boxes"]),), dtype=torch.int64)}
        return torch.as_tensor(stack, dtype=torch.float32), target


def collate(batch):
    return tuple(zip(*batch))


# ---- model ----------------------------------------------------------------- #
def freeze_backbone_bn(model):
    """Freeze every BatchNorm in train mode: keep running stats fixed and stop
    grads (a fine-tune divergence source on tiny data). torchvision's backbone
    already uses FrozenBatchNorm2d; this is the safety net for any live BN."""
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


def build_model(in_ch):
    model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1,
                                   trainable_backbone_layers=TRAIN_LAYERS)
    c = model.backbone.out_channels
    na = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(c, na, num_classes=2)
    if in_ch != 3:
        old = model.backbone.body.conv1
        new = torch.nn.Conv2d(in_ch, old.out_channels, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3:].zero_()                   # zero-init shadow channels
        model.backbone.body.conv1 = new
    model.transform.min_size = (MIN_SIZE,)
    model.transform.max_size = max(MIN_SIZE + 64, 640)
    model.transform.image_mean = [0.485, 0.456, 0.406] + [0.0] * (in_ch - 3)
    model.transform.image_std  = [0.229, 0.224, 0.225] + [1.0] * (in_ch - 3)
    model.score_thresh = 0.01   # keep low-confidence preds so the PR curve is populated
    return model


def in_channels_for(rung):
    return 3 if rung == "rgb" else 3 + SHADOW_CFG.n_channels


def train_eval(recs_tr, recs_va, rung, seed, az_override):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model(in_channels_for(rung)).to(DEVICE)
    model.train(); freeze_backbone_bn(model)   # BN frozen for the whole fine-tune
    dl_tr = DataLoader(DetDataset(recs_tr, rung, az_override), batch_size=2,
                       shuffle=True, collate_fn=collate, num_workers=0)
    params = [p for p in model.parameters() if p.requires_grad]
    # AdamW + low LR + multi-epoch warmup + grad clip + NaN guard: the combination
    # that drives AP50=0.0 collapses to zero across seeds (the acceptance check).
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    warmup_iters = max(1, int(WARMUP_EP * len(dl_tr)))
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup_iters)
    it = 0
    for ep in range(EPOCHS):
        for imgs, targets in dl_tr:
            imgs = [im.to(DEVICE) for im in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss = sum(model(imgs, targets).values())
            opt.zero_grad()
            if not torch.isfinite(loss):
                continue  # skip a non-finite step rather than corrupt the model
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            opt.step()
            if it < warmup_iters:
                sched.step()
            it += 1
    return evaluate_ap50(model, recs_va, rung, az_override)


@torch.no_grad()
def evaluate_ap50(model, recs_va, rung, az_override, iou_thr=0.5):
    model.eval()
    ds = DetDataset(recs_va, rung, az_override)
    scores, tps, n_gt = [], [], 0
    for i in range(len(ds)):
        img, target = ds[i]
        out = model([img.to(DEVICE)])[0]
        gt = target["boxes"].numpy()
        n_gt += len(gt)
        pb = out["boxes"].cpu().numpy(); ps = out["scores"].cpu().numpy()
        order = np.argsort(-ps)
        matched = np.zeros(len(gt), dtype=bool)
        for j in order:
            scores.append(ps[j])
            if len(gt) == 0:
                tps.append(0); continue
            ious = _iou(pb[j], gt)
            k = int(np.argmax(ious)) if len(ious) else -1
            if k >= 0 and ious[k] >= iou_thr and not matched[k]:
                matched[k] = True; tps.append(1)
            else:
                tps.append(0)
    return _ap(np.array(scores), np.array(tps), n_gt)


def _iou(b, gts):
    x0 = np.maximum(b[0], gts[:, 0]); y0 = np.maximum(b[1], gts[:, 1])
    x1 = np.minimum(b[2], gts[:, 2]); y1 = np.minimum(b[3], gts[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_g = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
    return inter / (area_b + area_g - inter + 1e-9)


def _ap(scores, tps, n_gt):
    if len(scores) == 0 or n_gt == 0:
        return 0.0
    order = np.argsort(-scores); tps = tps[order]
    tp = np.cumsum(tps); fp = np.cumsum(1 - tps)
    rec = tp / n_gt; prec = tp / (tp + fp)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# ---- driver ---------------------------------------------------------------- #
def main():
    t0 = time.time()
    recs = load_records()
    scenes = np.array([r["scene"] for r in recs])
    acqs = np.array([r["acq"] for r in recs])
    print(f"[data] {len(recs)} azimuth tiles | acquisitions={dict(collections.Counter(acqs))}", flush=True)

    fa = make_scene_clustered_folds(scenes, n_splits=N_FOLDS, acquisition_ids=acqs, seed=0)
    print(f"[folds] effective_n (scenes) per fold = {fa.effective_n()}", flush=True)

    # within-acquisition shuffled azimuth, keyed by scene (rung 3)
    sc_az = {r["scene"]: r["azimuth"] for r in recs}
    sc_acq = {r["scene"]: r["acq"] for r in recs}
    uscenes = list(sc_az.keys())
    shuf = shuffle_azimuths(np.array([sc_az[s] for s in uscenes]),
                            np.array([sc_acq[s] for s in uscenes], dtype=object), seed=0)
    az_override = {s: float(v) for s, v in zip(uscenes, shuf)}

    from scipy import stats as sps
    rungs = ["rgb", "correct", "shuffled"]
    # runs[rung][fold] = [AP50 per seed]; every (fold,rung) cell is seeded so a
    # single divergence can't masquerade as the fold's result (user fix #2).
    runs = {rg: [[None] * SEED_RUNS for _ in range(N_FOLDS)] for rg in rungs}
    collapses = []
    for fold in range(N_FOLDS):
        tr = [recs[i] for i in fa.train_indices(fold)]
        va = [recs[i] for i in fa.test_indices(fold)]
        for rg in rungs:
            for s in range(SEED_RUNS):
                t1 = time.time()
                v = train_eval(tr, va, rg, seed=s, az_override=az_override)
                runs[rg][fold][s] = v
                bad = v < COLLAPSE_EPS
                if bad:
                    collapses.append({"rung": rg, "fold": fold, "seed": s, "ap50": v})
                print(f"[fold {fold} | {rg:8} | seed {s}] AP50={v:.4f}  "
                      f"({time.time()-t1:.0f}s){'  <-- COLLAPSE' if bad else ''}", flush=True)

    # robust per-fold value = median over NON-collapsed seeds (NaN if all collapse)
    def agg(vals):
        good = [x for x in vals if x >= COLLAPSE_EPS]
        return float(np.median(good)) if good else float("nan")
    per_fold = {rg: [agg(runs[rg][f]) for f in range(N_FOLDS)] for rg in rungs}

    # recomputed seed floor: pooled within-config seed std (residuals about each
    # cell's median, non-collapsed seeds only) -> MDE for a paired SEED_RUNS design.
    resid = []
    for rg in rungs:
        for f in range(N_FOLDS):
            good = [x for x in runs[rg][f] if x >= COLLAPSE_EPS]
            if len(good) >= 2:
                m = np.median(good); resid += [x - m for x in good]
    pooled_seed_std = float(np.std(resid, ddof=1)) if len(resid) >= 2 else float("nan")
    if np.isfinite(pooled_seed_std):
        za, zb = float(sps.norm.ppf(0.975)), float(sps.norm.ppf(0.8))
        mde_pooled = float((za + zb) * np.sqrt(2) * pooled_seed_std / np.sqrt(SEED_RUNS))
    else:
        mde_pooled = float("nan")

    d_23 = np.array([per_fold["correct"][f] - per_fold["shuffled"][f] for f in range(N_FOLDS)])
    d_21 = np.array([per_fold["correct"][f] - per_fold["rgb"][f] for f in range(N_FOLDS)])
    n_train, n_test = fa.train_test_scene_counts(0)

    out = {"config": {"n_folds": N_FOLDS, "epochs": EPOCHS, "lr": LR, "optimizer": "adamw",
                      "warmup_epochs": WARMUP_EP, "trainable_backbone_layers": TRAIN_LAYERS,
                      "seeds_per_cell": SEED_RUNS, "min_size": MIN_SIZE, "device": DEVICE,
                      "collapse_eps": COLLAPSE_EPS, "shadow_cfg": SHADOW_CFG.to_dict(),
                      "detector": "torchvision retinanet_resnet50_fpn (DeepForest stand-in)"},
           "effective_n": fa.effective_n(),
           "ap50_runs": runs, "per_fold_median": per_fold,
           "n_collapses": len(collapses), "collapses": collapses,
           "pooled_seed_std": pooled_seed_std, "mde_pooled": mde_pooled,
           "delta_correct_minus_shuffled": d_23.tolist(),
           "delta_correct_minus_rgb": d_21.tolist()}

    stable = len(collapses) == 0 and bool(np.all(np.isfinite(d_23)))
    if np.all(np.isfinite(d_23)) and N_FOLDS >= 2:
        out["nadeau_bengio_2v3"] = corrected_resampled_ttest(d_23, n_train, n_test).to_dict()
        out["permutation_2v3"] = permutation_test_paired(d_23).to_dict()
    floor = [x for x in runs["correct"][0] if x >= COLLAPSE_EPS]
    if len(floor) >= 2:
        out["seed_variance_correct_fold0"] = seed_variance(floor, n_seeds_paired=SEED_RUNS).to_dict()
    out["acceptance_zero_collapses"] = (len(collapses) == 0)
    out["wall_clock_s"] = round(time.time() - t0, 1)

    os.makedirs("artifacts", exist_ok=True)
    json.dump(out, open(ARTIFACT, "w"), indent=2)

    print("\n===== STEP 1 SUMMARY =====", flush=True)
    for rg in rungs:
        print(f"  {rg:8} seeds/fold: {[[round(x,3) for x in runs[rg][f]] for f in range(N_FOLDS)]} "
              f"-> per-fold median {[round(x,3) for x in per_fold[rg]]}", flush=True)
    print(f"collapses (AP50<{COLLAPSE_EPS}): {len(collapses)}  "
          f"{'-> ACCEPTANCE MET (training stable)' if len(collapses)==0 else '-> NOT STABLE, do not interpret'}", flush=True)
    print(f"delta correct-shuffled (per fold): {np.round(d_23,4).tolist()}  mean={np.nanmean(d_23):.4f}", flush=True)
    print(f"delta correct-rgb      (per fold): {np.round(d_21,4).tolist()}  mean={np.nanmean(d_21):.4f}", flush=True)
    if "nadeau_bengio_2v3" in out:
        nb = out["nadeau_bengio_2v3"]
        print(f"Nadeau-Bengio 2v3: mean={nb['mean_delta']:.4f} CI=({nb['ci_low']:.4f},{nb['ci_high']:.4f}) "
              f"dz={nb['effect_size_dz']:.3f} p={nb['p_value']:.3f}", flush=True)
    if "permutation_2v3" in out:
        print(f"permutation 2v3 p={out['permutation_2v3']['p_value']:.3f}", flush=True)
    print(f"pooled seed_std={pooled_seed_std:.4f}  MDE(pooled)={mde_pooled:.4f}", flush=True)
    if stable and np.isfinite(mde_pooled):
        above = abs(np.nanmean(d_23)) > mde_pooled
        ci = out.get("nadeau_bengio_2v3", {})
        ci_excl0 = ci and (ci["ci_low"] > 0 or ci["ci_high"] < 0)
        print(f">>> VERDICT: |mean 2v3|={abs(np.nanmean(d_23)):.4f} vs MDE={mde_pooled:.4f} | "
              f"CI_excludes_0={bool(ci_excl0)} -> "
              f"{'SIGNAL above floor (pursue)' if (above and ci_excl0) else 'no effect above floor (null/underpowered)'}", flush=True)
    else:
        print(">>> VERDICT: training NOT stable (collapses present) -- rung comparison not interpretable", flush=True)
    print(f"wall clock: {out['wall_clock_s']}s | artifact: {ARTIFACT}", flush=True)


if __name__ == "__main__":
    main()
