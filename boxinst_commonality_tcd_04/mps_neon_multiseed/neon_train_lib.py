"""NEON box-only detector: resumable training (faithful to the 0.492 TCD recipe) +
box-only evaluation. Isolated in mps_neon_multiseed/; reuses the repo's Detector8 /
det_loss / TileData / decode so the RECIPE is byte-identical to train_detector_tiles.

Recipe (unchanged from the 0.492 run): width 256, tower 3, Adam lr 1e-3 wd 1e-4,
cosine over `epochs`, bs 3, eval_every 5, best-on-val checkpoint, aggressive early
stopping (min_epochs 12, es_patience 2, es_min_delta 0.005). ONLY additions vs
train_detector_tiles.train: (1) full-state checkpoint every eval + resume on restart
(Modal reallocates GPUs mid-flight), (2) a `commit` hook fired after every checkpoint
so the best model + resume state are durably persisted to the Volume, (3) box-only
eval that emits predictions for the NEON scorer (NO EM box->mask step).

Geometry: NEON tiles/patches are 400px @ 0.1 m/px -> load_tile pads to 512 -> 32x32
DINOv3 grid -> Detector8 predicts at 8px stride on a 64x64 grid (canvas 512). Same
8px stride as the TCD recipe.
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
from PIL import Image, ImageDraw

from dapt.backbone import pick_device
from dapt.decode import decode
from dapt.eval import full_report, pick_threshold
from dapt.targets import TargetConfig, encode
from boxinst_commonality_tcd_04.detector import (STRIDE8, Detector8,
                                                 canopy_cell_mask, det_loss)

CANVAS = 512                         # padded NEON tile -> grid 64 at 8px


# --- data loading: TileData / infer / set_seed copied VERBATIM from
# train_detector_tiles.py (same target encoding as the 0.492 recipe), so this module
# is self-contained (no cache_train_tiles/prepare_test/boxinst import chain -> the
# Modal image needs no stubs). TileData takes the feature dir directly.
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def _canopy_px(polys, res):
    m = Image.new("L", (res, res), 0); d = ImageDraw.Draw(m)
    for poly in polys:
        if poly and len(poly) >= 6:
            d.polygon([tuple(v) for v in np.asarray(poly).reshape(-1, 2)], fill=1)
    return np.asarray(m, bool)


class TileData:
    def __init__(self, feat_dir, canvas=CANVAS, gt_path=None):
        self.cdir = feat_dir
        gt = json.load(open(gt_path))
        self.gt = {t: v for t, v in gt.items()
                   if os.path.exists(os.path.join(self.cdir, t + ".npy"))}
        cfg = TargetConfig(grid=canvas // 8, stride=8)
        self.enc, self.ign, self.boxes = {}, {}, {}
        for t, v in self.gt.items():
            bx = np.array(v["boxes"], np.float32).reshape(-1, 4)
            e = encode(bx, cfg)
            self.enc[t] = {k: torch.from_numpy(e[k]) for k in
                           ("heatmap", "offset", "size", "reg_mask")}
            self.ign[t] = torch.from_numpy(
                canopy_cell_mask(_canopy_px(v.get("canopy", []), res=canvas),
                                 cfg.grid))
            self.boxes[t] = bx

    def partition(self, name):
        return sorted(t for t, v in self.gt.items() if v["partition"] == name)

    def _feat(self, t):
        return torch.from_numpy(np.load(os.path.join(self.cdir, t + ".npy")))

    def batch(self, tids, device):
        f = torch.stack([self._feat(t) for t in tids]).float().to(device)
        st = lambda k: torch.stack([self.enc[t][k] for t in tids]).to(device)  # noqa: E731
        return {"feat": f, "heatmap": st("heatmap"), "offset": st("offset"),
                "size": st("size"), "reg_mask": st("reg_mask"),
                "ignore": torch.stack([self.ign[t] for t in tids]).to(device),
                "boxes": [self.boxes[t] for t in tids]}


@torch.no_grad()
def infer(model, data, tids, device, score_thr=0.05, topk=600):
    model.eval()
    preds, gts = [], []
    for t in tids:
        det = model(data._feat(t).float()[None].to(device))
        bx, sc = decode(det.cpu(), score_thr=score_thr, stride=STRIDE8, topk=topk)
        preds.append((bx.numpy(), sc.numpy())); gts.append(data.boxes[t])
    return preds, gts


def make_data(feat_dir, gt_path):
    return TileData(feat_dir, canvas=CANVAS, gt_path=gt_path)


def _cfg(in_dim, width, tower, thr, seed, tag, best_epoch):
    return {"in_dim": in_dim, "width": width, "tower": tower, "score_thr": thr,
            "nms_iou": 0.5, "seed": seed, "tag": tag, "best_epoch": best_epoch,
            "canvas": CANVAS, "stride": STRIDE8, "data": "NEON hand-annotated patches"}


def train_resumable(feat_dir, gt_path, ckpt_dir, tag="neon_s0", seed=0, epochs=40,
                    bs=3, eval_every=5, lr=1e-3, wd=1e-4, width=256, tower=3,
                    min_epochs=12, es_patience=2, es_min_delta=0.005,
                    device=None, commit=None):
    os.makedirs(ckpt_dir, exist_ok=True)
    set_seed(seed)
    device = pick_device(device)
    data = make_data(feat_dir, gt_path)
    tr, va = data.partition("train"), data.partition("val")
    assert tr and va, f"empty split: train={len(tr)} val={len(va)}"
    in_dim = data._feat(tr[0]).shape[0]
    model = Detector8(in_dim, width=width, tower=tower).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = np.random.default_rng(seed)
    best = {"mAP50": -1.0, "state": None, "epoch": -1, "thr": 0.2}
    start_ep, no_improve = 0, 0
    state_fp = os.path.join(ckpt_dir, f"state_{tag}.pt")
    best_fp = os.path.join(ckpt_dir, f"det_{tag}.pt")

    if os.path.exists(state_fp):                 # RESUME after preemption
        st = torch.load(state_fp, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); best = st["best"]
        start_ep, no_improve = st["epoch"] + 1, st["no_improve"]
        rng = np.random.default_rng(); rng.bit_generator.state = st["rng"]
        print(f"[resume] {tag}: from ep{start_ep} (best ep{best['epoch']} "
              f"boxAP50={best['mAP50']:.3f})", flush=True)

    npar = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[{tag}] device={device} in_dim={in_dim} params={npar:.2f}M "
          f"train/val={len(tr)}/{len(va)} seed={seed} cosine early-stop "
          f"(p{es_patience}/min{min_epochs}) start_ep={start_ep}", flush=True)

    for ep in range(start_ep, epochs):
        model.train()
        order = list(tr); rng.shuffle(order)
        losses = []
        for i in range(0, len(order), bs):
            b = data.batch(order[i:i + bs], device)
            det = model(b["feat"])
            l_hm, l_off, l_size, l_giou = det_loss(det, b)
            loss = l_hm + l_off + l_size + l_giou
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append([l_hm.item(), l_off.item(), l_size.item(), l_giou.item()])
        sched.step()
        if (ep + 1) % eval_every == 0 or ep + 1 == epochs:
            m = np.mean(losses, 0)
            preds, gts = infer(model, data, va, device)
            thr, _ = pick_threshold(preds, gts)
            rep = full_report(preds, gts, thr)
            lr_now = opt.param_groups[0]["lr"]
            print(f"  ep{ep+1:3d} hm={m[0]:.3f} off={m[1]:.3f} size={m[2]:.3f} "
                  f"giou={m[3]:.3f} | val boxAP50={rep['mAP50']:.3f} "
                  f"F1={rep['f1']:.3f}@{thr:.2f} lr={lr_now:.1e}", flush=True)
            improved = rep["mAP50"] > best["mAP50"] + es_min_delta
            if rep["mAP50"] > best["mAP50"]:
                best = {"mAP50": rep["mAP50"], "epoch": ep + 1, "thr": thr,
                        "state": {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}}
                torch.save({"state": best["state"],
                            "cfg": _cfg(in_dim, width, tower, thr, seed, tag,
                                        ep + 1)}, best_fp)
                print(f"    ^ saved best ep{ep+1} boxAP50={best['mAP50']:.3f}",
                      flush=True)
                if commit:
                    commit()
            no_improve = 0 if improved else no_improve + 1
            # durable resume state after EVERY eval (survives mid-flight kill)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "best": best, "epoch": ep,
                        "no_improve": no_improve,
                        "rng": rng.bit_generator.state}, state_fp)
            if commit:
                commit()
            if ep + 1 >= min_epochs and no_improve >= es_patience:
                print(f"    early-stop: {no_improve} evals flat (best ep"
                      f"{best['epoch']})", flush=True)
                break
    open(os.path.join(ckpt_dir, f"done_{tag}.flag"), "w").write(str(best["epoch"]))
    if commit:
        commit()
    print(f"[{tag}] best ep{best['epoch']} val boxAP50={best['mAP50']:.3f} "
          f"-> {best_fp}", flush=True)
    return best


@torch.no_grad()
def predict_boxes(ckpt_path, eval_feat_dir, eval_gt_json, rgb_dir, out_pred_json,
                  score_thr=0.01, topk=600, device=None):
    """Box-only inference (NO EM). Per 194 eval tile: features -> detector -> decode ->
    boxes in tile pixel coords (pad region dropped). Writes {plot: {boxes, scores}}."""
    device = pick_device(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = Detector8(cfg["in_dim"], width=cfg["width"], tower=cfg["tower"]
                      ).to(device).eval()
    model.load_state_dict(ck["state"])
    gt = json.load(open(eval_gt_json))
    preds = {}
    for k, plot in enumerate(sorted(gt)):
        fp = os.path.join(eval_feat_dir, plot + ".npy")
        if not os.path.exists(fp):
            preds[plot] = {"boxes": [], "scores": []}
            continue
        feat = np.load(fp).astype(np.float32)
        det = model(torch.from_numpy(feat)[None].to(device))
        bx, sc = decode(det.cpu(), score_thr=score_thr, stride=STRIDE8, topk=topk)
        bx, sc = bx.numpy(), sc.numpy()
        W, H = Image.open(os.path.join(rgb_dir, plot + ".tif")).size   # true tile size
        cx = (bx[:, 0] + bx[:, 2]) / 2; cy = (bx[:, 1] + bx[:, 3]) / 2
        keep = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)             # drop pad region
        bx, sc = bx[keep], sc[keep]
        bx[:, [0, 2]] = bx[:, [0, 2]].clip(0, W)                       # clip to tile
        bx[:, [1, 3]] = bx[:, [1, 3]].clip(0, H)
        preds[plot] = {"boxes": bx.tolist(), "scores": sc.tolist()}
        if (k + 1) % 50 == 0 or k + 1 == len(gt):
            print(f"  predicted {k+1}/{len(gt)}", flush=True)
    json.dump(preds, open(out_pred_json, "w"))
    print(f"wrote {out_pred_json}", flush=True)
    return preds


def score_predictions(pred_json, gt_json, out_json, op_score_thr=None):
    """Score box predictions with the NEON macro-average scorer at IoU 0.4 + PR sweep.
    If op_score_thr is None, report the sweep + the best-F1 operating point."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scorer import evaluate, pr_curve
    preds = json.load(open(pred_json))
    gt = json.load(open(gt_json))
    gt_arr = {p: np.array(b, float).reshape(-1, 4) for p, b in gt.items()}
    pr_in = {p: {"boxes": np.array(v["boxes"], float).reshape(-1, 4),
                 "scores": np.array(v["scores"], float)}
             for p, v in preds.items()}
    curve = pr_curve(pr_in, gt_arr, iou_thr=0.4,
                     thresholds=np.round(np.arange(0.0, 0.95, 0.02), 3))
    # best-F1 operating point on the sweep
    def f1(p):
        return 0.0 if (p["mean_precision"] + p["mean_recall"]) == 0 else \
            2 * p["mean_precision"] * p["mean_recall"] / (p["mean_precision"] + p["mean_recall"])
    best = max(curve, key=f1)
    op = op_score_thr if op_score_thr is not None else best["score_thr"]
    e = evaluate(pr_in, gt_arr, iou_thr=0.4, score_thr=op, nan_precision="zero")
    res = {"iou_thr": 0.4, "operating_score_thr": op,
           "mean_precision": round(e["mean_precision"], 4),
           "mean_recall": round(e["mean_recall"], 4),
           "best_f1_point": {"score_thr": best["score_thr"],
                             "P": round(best["mean_precision"], 4),
                             "R": round(best["mean_recall"], 4)},
           "target_paper_P": 0.659, "target_paper_R": 0.790,
           "n_plots": len(gt), "pr_curve": curve}
    json.dump(res, open(out_json, "w"), indent=2)
    print(f"[score] ours @IoU0.4 best-F1: P={best['mean_precision']:.3f} "
          f"R={best['mean_recall']:.3f} @thr{best['score_thr']:.2f}  "
          f"(paper Table 3: P0.659/R0.790)", flush=True)
    return res
