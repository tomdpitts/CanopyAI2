"""4-phase detector head + train/eval, byte-identical recipe to the native seed-0 run.

The ONLY differences vs neon_train_lib's native path:
  - features are (4096,64,64) REAL 8px (phase4_features.interleave), not (4096,32,32);
  - Detector4Phase drops DetectorS's internal bilinear upsample (features arrive at 8px).
Everything else - TargetConfig(grid=64, stride=8), encode/decode, det_loss, Adam lr1e-3
wd1e-4, cosine, bs3, eval_every5, early-stop(min12,p2,delta5e-3) - is imported verbatim
from the shared modules so this is a clean apples-to-apples of feature quality only.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from PIL import Image

from dapt.backbone import pick_device
from dapt.decode import decode
from dapt.eval import full_report, pick_threshold
from boxinst_commonality_tcd_04.detector import Detector8
# reuse the native data pipeline + loss VERBATIM (stride=8 -> grid 64, same targets)
from boxinst_commonality_tcd_04.modal_neon_multiseed.neon_train_lib import (
    TileData, _det_loss, infer, make_data)

STRIDE = 8
CANVAS = 512


class Detector4Phase(Detector8):
    """Detector8 on a pre-assembled REAL 8px grid: identical modules & params, but NO
    internal interpolate (native DetectorS(up=2) bilinearly upsamples 16px->8px; here the
    64x64 features are already 8px). stem+tower+up run at 64x64; heads predict (5,64,64)
    at stride 8."""

    def forward(self, feat):                       # feat (B,4096,64,64)
        x = self.tower(self.stem(feat))
        x = self.up(x)                             # no interpolate
        return torch.cat([self.hm(x), self.reg(x)], dim=1)


def _cfg(in_dim, width, tower, thr, seed, tag, best_epoch):
    return {"in_dim": in_dim, "width": width, "tower": tower, "score_thr": thr,
            "nms_iou": 0.5, "seed": seed, "tag": tag, "best_epoch": best_epoch,
            "up": "4phase", "stride": STRIDE, "canvas": CANVAS,
            "data": "NEON hand-annotated patches, 4-phase real-8px features"}


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def train_seed(feat_dir, gt_path, ckpt_dir, tag="phase4_s0", seed=0, epochs=40, bs=3,
               eval_every=5, lr=1e-3, wd=1e-4, width=256, tower=3, min_epochs=12,
               es_patience=2, es_min_delta=0.005, device=None, commit=None, data=None):
    """Native recipe, Detector4Phase. Resumable (full state each eval) for H100 preemption."""
    os.makedirs(ckpt_dir, exist_ok=True)
    set_seed(seed)
    device = pick_device(device)
    if data is None:
        data = make_data(feat_dir, gt_path, preload=True, stride=STRIDE)
    tr, va = data.partition("train"), data.partition("val")
    assert tr and va, f"empty split train={len(tr)} val={len(va)}"
    in_dim = data._feat(tr[0]).shape[0]
    gshape = tuple(data._feat(tr[0]).shape)
    assert gshape == (4096, 64, 64), f"expected (4096,64,64) real-8px feats, got {gshape}"
    model = Detector4Phase(in_dim, width=width, tower=tower).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = np.random.default_rng(seed)
    best = {"mAP50": -1.0, "state": None, "epoch": -1, "thr": 0.2}
    start_ep, no_improve = 0, 0
    state_fp = os.path.join(ckpt_dir, f"state_{tag}.pt")
    best_fp = os.path.join(ckpt_dir, f"det_{tag}.pt")
    if os.path.exists(state_fp):
        st = torch.load(state_fp, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); best = st["best"]
        start_ep, no_improve = st["epoch"] + 1, st["no_improve"]
        rng = np.random.default_rng(); rng.bit_generator.state = st["rng"]
        print(f"[resume] {tag}: ep{start_ep} (best ep{best['epoch']} "
              f"AP50={best['mAP50']:.3f})", flush=True)
    npar = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[{tag}] device={device} in_dim={in_dim} feats{gshape} params={npar:.2f}M "
          f"train/val={len(tr)}/{len(va)} seed={seed} start_ep={start_ep}", flush=True)
    t_run = time.time()
    for ep in range(start_ep, epochs):
        model.train()
        order = list(tr); rng.shuffle(order)
        losses, t_ep = [], time.time()
        for i in range(0, len(order), bs):
            b = data.batch(order[i:i + bs], device)
            det = model(b["feat"])
            l_hm, l_off, l_size, l_giou = _det_loss(det, b, STRIDE)
            loss = l_hm + l_off + l_size + l_giou
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append([l_hm.item(), l_off.item(), l_size.item(), l_giou.item()])
        sched.step()
        m = np.mean(losses, 0)
        print(f"  ep{ep+1:3d}/{epochs} loss={sum(m):.3f} (hm={m[0]:.3f} off={m[1]:.3f} "
              f"size={m[2]:.3f} giou={m[3]:.3f}) {time.time()-t_ep:.1f}s/ep "
              f"lr={opt.param_groups[0]['lr']:.1e}", flush=True)
        if (ep + 1) % eval_every == 0 or ep + 1 == epochs:
            preds, gts = infer(model, data, va, device, stride=STRIDE)
            thr, _ = pick_threshold(preds, gts)
            rep = full_report(preds, gts, thr)
            print(f"    val boxAP50={rep['mAP50']:.3f} F1={rep['f1']:.3f}@{thr:.2f}",
                  flush=True)
            improved = rep["mAP50"] > best["mAP50"] + es_min_delta
            if rep["mAP50"] > best["mAP50"]:
                best = {"mAP50": rep["mAP50"], "epoch": ep + 1, "thr": thr,
                        "state": {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}}
                torch.save({"state": best["state"],
                            "cfg": _cfg(in_dim, width, tower, thr, seed, tag, ep + 1)},
                           best_fp)
                print(f"    ^ saved best ep{ep+1} AP50={best['mAP50']:.3f}", flush=True)
                if commit:
                    commit()
            no_improve = 0 if improved else no_improve + 1
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "best": best, "epoch": ep,
                        "no_improve": no_improve, "rng": rng.bit_generator.state},
                       state_fp)
            if commit:
                commit()
            if ep + 1 >= min_epochs and no_improve >= es_patience:
                print(f"    early-stop (best ep{best['epoch']})", flush=True)
                break
    open(os.path.join(ckpt_dir, f"done_{tag}.flag"), "w").write(str(best["epoch"]))
    if commit:
        commit()
    print(f"[{tag}] best ep{best['epoch']} AP50={best['mAP50']:.3f} in "
          f"{(time.time()-t_run)/60:.1f} min -> {best_fp}", flush=True)
    return best


@torch.no_grad()
def predict_boxes_4p(ckpt_path, eval_feat_dir, eval_gt_json, rgb_dir, out_pred_json,
                     score_thr=0.01, topk=600, device=None, eval_cache=None):
    """Box-only inference with Detector4Phase (stride 8). Mirrors neon_train_lib.
    predict_boxes: features (4096,64,64) -> decode -> boxes in tile px (pad dropped)."""
    device = pick_device(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = Detector4Phase(cfg["in_dim"], width=cfg["width"], tower=cfg["tower"]
                           ).to(device).eval()
    model.load_state_dict(ck["state"])
    gt = json.load(open(eval_gt_json))
    preds = {}
    for k, plot in enumerate(sorted(gt)):
        if eval_cache is not None:
            if plot not in eval_cache:
                preds[plot] = {"boxes": [], "scores": []}; continue
            feat = eval_cache[plot]
        else:
            fp = os.path.join(eval_feat_dir, plot + ".npy")
            if not os.path.exists(fp):
                preds[plot] = {"boxes": [], "scores": []}; continue
            feat = np.load(fp).astype(np.float32)
        det = model(torch.from_numpy(feat)[None].to(device))
        bx, sc = decode(det.cpu(), score_thr=score_thr, stride=STRIDE, topk=topk)
        bx, sc = bx.numpy(), sc.numpy()
        W, H = Image.open(os.path.join(rgb_dir, plot + ".tif")).size
        cx = (bx[:, 0] + bx[:, 2]) / 2; cy = (bx[:, 1] + bx[:, 3]) / 2
        keep = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
        bx, sc = bx[keep], sc[keep]
        bx[:, [0, 2]] = bx[:, [0, 2]].clip(0, W); bx[:, [1, 3]] = bx[:, [1, 3]].clip(0, H)
        preds[plot] = {"boxes": bx.tolist(), "scores": sc.tolist()}
        if (k + 1) % 50 == 0 or k + 1 == len(gt):
            print(f"  predicted {k+1}/{len(gt)}", flush=True)
    json.dump(preds, open(out_pred_json, "w"))
    print(f"wrote {out_pred_json}", flush=True)
    return preds
