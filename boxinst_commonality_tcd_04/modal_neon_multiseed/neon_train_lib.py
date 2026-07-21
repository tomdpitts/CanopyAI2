"""NEON box-only detector: resumable training (faithful to the 0.492 TCD recipe) +
box-only evaluation. Isolated in modal_neon_multiseed/; reuses the repo's Detector8 /
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

import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss

from dapt.backbone import pick_device
from dapt.decode import decode
from dapt.eval import full_report, pick_threshold
from dapt.head import _masked_smooth_l1, focal_heatmap_loss
from dapt.targets import TargetConfig, encode
from boxinst_commonality_tcd_04.detector import (STRIDE8, Detector8,
                                                 canopy_cell_mask, det_loss)

CANVAS = 512                         # padded NEON tile -> grid 64 at 8px (up=2)


class DetectorS(Detector8):
    """Detector8 with a configurable feature-upsample factor. up=2 -> 8px stride (the
    0.492/native recipe); up=4 -> 4px stride: a 128x128 output grid that halves CenterNet
    cell collisions again in dense canopy (the same trick as the original 16->8 upsample).
    Features stay 16px (the 4px grid is interpolated), so this aids localization/collision,
    not feature resolution. STRIDE = 16 // up (DINOv3 patch16 -> 32x32 grid on 512)."""
    def __init__(self, in_dim, width=256, tower=3, up=2):
        super().__init__(in_dim, width=width, tower=tower)
        self.up_factor = up

    def forward(self, feat):
        x = self.tower(self.stem(feat))
        x = F.interpolate(x, scale_factor=self.up_factor, mode="bilinear",
                          align_corners=False)
        x = self.up(x)
        return torch.cat([self.hm(x), self.reg(x)], dim=1)


def _det_loss(det, tgt, stride, w_size=0.1):
    """det_loss (verbatim from detector.py) but with `stride` passed in, so the GIoU box
    conversion is correct at 4px as well as 8px (the shared det_loss hardcodes STRIDE8)."""
    hm, off, size = det[:, :1], det[:, 1:3], det[:, 3:5]
    l_hm = focal_heatmap_loss(hm, tgt["heatmap"], tgt.get("ignore"))
    l_off = _masked_smooth_l1(off, tgt["offset"], tgt["reg_mask"])
    l_size = _masked_smooth_l1(size, tgt["size"], tgt["reg_mask"])
    b, gy, gx = tgt["reg_mask"].nonzero(as_tuple=True)
    if len(b):
        def to_box(o, s):
            cx = (gx.float() + o[b, 0, gy, gx]) * stride
            cy = (gy.float() + o[b, 1, gy, gx]) * stride
            w = s[b, 0, gy, gx].clamp(max=8).exp()
            h = s[b, 1, gy, gx].clamp(max=8).exp()
            return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        l_giou = generalized_box_iou_loss(to_box(off, size),
                                          to_box(tgt["offset"], tgt["size"]),
                                          reduction="mean")
    else:
        l_giou = det.sum() * 0.0
    return l_hm, l_off, l_size * w_size, l_giou


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
    def __init__(self, feat_dir, canvas=CANVAS, gt_path=None, preload=False, stride=8):
        self.cdir = feat_dir
        gt = json.load(open(gt_path))
        self.gt = {t: v for t, v in gt.items()
                   if os.path.exists(os.path.join(self.cdir, t + ".npy"))}
        # RAM cache: NEON features (~8MB each) fit in memory, so preload once and
        # skip the per-batch Volume np.load that dominated wall-clock (~1.5min/epoch).
        # THREADED: np.load is I/O-bound and drops the GIL, so parallel reads saturate
        # the (slow, small-file) Modal Volume bandwidth -> ~13min -> a few min.
        self._cache = {}
        if preload:
            import time as _t
            from concurrent.futures import ThreadPoolExecutor
            keys = list(self.gt)
            t0 = _t.time()

            def _ld(t):
                return t, torch.from_numpy(
                    np.load(os.path.join(self.cdir, t + ".npy")))
            with ThreadPoolExecutor(max_workers=16) as ex:
                for j, (t, arr) in enumerate(ex.map(_ld, keys)):
                    self._cache[t] = arr
                    if (j + 1) % 400 == 0 or j + 1 == len(keys):
                        print(f"  [preload] {j+1}/{len(keys)} feats "
                              f"({_t.time()-t0:.0f}s)", flush=True)
        cfg = TargetConfig(grid=canvas // stride, stride=stride)
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
        if t in self._cache:
            return self._cache[t]
        return torch.from_numpy(np.load(os.path.join(self.cdir, t + ".npy")))

    def batch(self, tids, device):
        f = torch.stack([self._feat(t) for t in tids]).float().to(device)
        st = lambda k: torch.stack([self.enc[t][k] for t in tids]).to(device)  # noqa: E731
        return {"feat": f, "heatmap": st("heatmap"), "offset": st("offset"),
                "size": st("size"), "reg_mask": st("reg_mask"),
                "ignore": torch.stack([self.ign[t] for t in tids]).to(device),
                "boxes": [self.boxes[t] for t in tids]}


@torch.no_grad()
def infer(model, data, tids, device, score_thr=0.05, topk=600, stride=8):
    model.eval()
    preds, gts = [], []
    for t in tids:
        det = model(data._feat(t).float()[None].to(device))
        bx, sc = decode(det.cpu(), score_thr=score_thr, stride=stride, topk=topk)
        preds.append((bx.numpy(), sc.numpy())); gts.append(data.boxes[t])
    return preds, gts


def make_data(feat_dir, gt_path, preload=False, stride=8):
    return TileData(feat_dir, canvas=CANVAS, gt_path=gt_path, preload=preload,
                    stride=stride)


def _cfg(in_dim, width, tower, thr, seed, tag, best_epoch, up=2, stride=8):
    return {"in_dim": in_dim, "width": width, "tower": tower, "score_thr": thr,
            "nms_iou": 0.5, "seed": seed, "tag": tag, "best_epoch": best_epoch,
            "up": up, "stride": stride,
            "canvas": CANVAS, "data": "NEON hand-annotated patches"}


def train_resumable(feat_dir, gt_path, ckpt_dir, tag="neon_s0", seed=0, epochs=40,
                    bs=3, eval_every=5, lr=1e-3, wd=1e-4, width=256, tower=3,
                    min_epochs=12, es_patience=2, es_min_delta=0.005,
                    device=None, commit=None, preload=True, log_every_steps=100,
                    up=2, data=None):
    import time
    stride = 16 // up                    # DINOv3 patch16 -> up=2:8px, up=4:4px
    os.makedirs(ckpt_dir, exist_ok=True)
    set_seed(seed)
    device = pick_device(device)
    # `data` may be a pre-built (already-preloaded) TileData shared across seeds
    # (amortized multiseed): identical to building it here since it's read-only during
    # training and set_seed/model-init happen per call. Build it only if not supplied.
    if data is None:
        data = make_data(feat_dir, gt_path, preload=preload, stride=stride)
    tr, va = data.partition("train"), data.partition("val")
    assert tr and va, f"empty split: train={len(tr)} val={len(va)}"
    in_dim = data._feat(tr[0]).shape[0]
    model = DetectorS(in_dim, width=width, tower=tower, up=up).to(device)
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

    t_run = time.time()
    ep_times = []
    n_steps = (len(tr) + bs - 1) // bs
    for ep in range(start_ep, epochs):
        model.train()
        order = list(tr); rng.shuffle(order)
        losses = []
        t_ep = time.time()
        for s, i in enumerate(range(0, len(order), bs)):
            b = data.batch(order[i:i + bs], device)
            det = model(b["feat"])
            l_hm, l_off, l_size, l_giou = _det_loss(det, b, stride)
            loss = l_hm + l_off + l_size + l_giou
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append([l_hm.item(), l_off.item(), l_size.item(), l_giou.item()])
            # intra-epoch heartbeat so a slow epoch is never a blind window
            if log_every_steps and ((s + 1) % log_every_steps == 0):
                el = time.time() - t_ep
                print(f"    ep{ep+1} step {s+1}/{n_steps} loss={loss.item():.3f} "
                      f"{(s+1)/el:.1f} it/s", flush=True)
        sched.step()
        # per-epoch summary + ETA on EVERY epoch (not just eval epochs)
        dt = time.time() - t_ep; ep_times.append(dt)
        m_ep = np.mean(losses, 0)
        eta = np.mean(ep_times[-5:]) * (epochs - ep - 1) / 60
        print(f"  ep{ep+1:3d}/{epochs} loss={sum(m_ep):.3f} "
              f"(hm={m_ep[0]:.3f} off={m_ep[1]:.3f} size={m_ep[2]:.3f} "
              f"giou={m_ep[3]:.3f}) {dt:.1f}s/ep lr={opt.param_groups[0]['lr']:.1e} "
              f"ETA<={eta:.1f}min", flush=True)
        if (ep + 1) % eval_every == 0 or ep + 1 == epochs:
            m = np.mean(losses, 0)
            preds, gts = infer(model, data, va, device, stride=stride)
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
                                        ep + 1, up=up, stride=stride)}, best_fp)
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


def preload_eval(eval_feat_dir, eval_gt_json):
    """Threaded one-time preload of the 194 eval features into a {plot: float32 array}
    dict, so an amortized multiseed run reads the Volume once, not per-seed."""
    from concurrent.futures import ThreadPoolExecutor
    gt = json.load(open(eval_gt_json))
    plots = [p for p in sorted(gt)
             if os.path.exists(os.path.join(eval_feat_dir, p + ".npy"))]

    def _ld(p):
        return p, np.load(os.path.join(eval_feat_dir, p + ".npy")).astype(np.float32)
    cache = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for p, a in ex.map(_ld, plots):
            cache[p] = a
    print(f"  [preload-eval] {len(cache)} eval feats cached", flush=True)
    return cache


@torch.no_grad()
def predict_boxes(ckpt_path, eval_feat_dir, eval_gt_json, rgb_dir, out_pred_json,
                  score_thr=0.01, topk=600, device=None, eval_cache=None):
    """Box-only inference (NO EM). Per 194 eval tile: features -> detector -> decode ->
    boxes in tile pixel coords (pad region dropped). Writes {plot: {boxes, scores}}.
    `eval_cache` (dict plot->feat) reuses pre-loaded features across seeds if given."""
    device = pick_device(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    up = cfg.get("up", 2); stride = cfg.get("stride", 8)   # old ckpts = 8px
    model = DetectorS(cfg["in_dim"], width=cfg["width"], tower=cfg["tower"], up=up
                      ).to(device).eval()
    model.load_state_dict(ck["state"])
    gt = json.load(open(eval_gt_json))
    preds = {}
    for k, plot in enumerate(sorted(gt)):
        if eval_cache is not None:
            if plot not in eval_cache:
                preds[plot] = {"boxes": [], "scores": []}
                continue
            feat = eval_cache[plot]
        else:
            fp = os.path.join(eval_feat_dir, plot + ".npy")
            if not os.path.exists(fp):
                preds[plot] = {"boxes": [], "scores": []}
                continue
            feat = np.load(fp).astype(np.float32)
        det = model(torch.from_numpy(feat)[None].to(device))
        bx, sc = decode(det.cpu(), score_thr=score_thr, stride=stride, topk=topk)
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


@torch.no_grad()
def predict_boxes_multiscale(ckpt_path, net, eval_feat_dir, eval_gt_json, rgb_dir,
                             out_pred_json, up=2, qwin=240, score_thr=0.01, topk=600,
                             ms_nms_iou=0.5, device=None):
    """Native (cached 400px features) + UPSCALE arm for tiny crowns: each tile is split
    into overlapping `qwin`px quadrants, each enlarged `up`x to fit the 512 pad so a
    ~19px crown becomes ~38px (~2.5 feature cells). Upscale boxes are mapped back to tile
    coords and cross-scale NMS-merged with the native detections. Inference-only; the
    backbone `net` must be the SAME layers=(21,22,23,24) extractor. Writes {plot:{boxes,
    scores}}."""
    from torchvision.ops import nms
    from PIL import Image
    from boxinst_commonality_tcd_04.modal_neon_multiseed import neon_features as nf
    device = pick_device(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = Detector8(cfg["in_dim"], width=cfg["width"], tower=cfg["tower"]
                      ).to(device).eval()
    model.load_state_dict(ck["state"])
    gt = json.load(open(eval_gt_json))
    preds = {}
    for k, plot in enumerate(sorted(gt)):
        W, H = Image.open(os.path.join(rgb_dir, plot + ".tif")).size
        boxes_all, scores_all = [], []
        # native arm (cached features)
        fp = os.path.join(eval_feat_dir, plot + ".npy")
        if os.path.exists(fp):
            det = model(torch.from_numpy(np.load(fp).astype(np.float32))[None].to(device))
            bx, sc = decode(det.cpu(), score_thr=score_thr, stride=STRIDE8, topk=topk)
            boxes_all.append(bx.numpy()); scores_all.append(sc.numpy())
        # upscale arm: overlapping quadrants
        img = Image.open(os.path.join(rgb_dir, plot + ".tif")).convert("RGB")
        xs = sorted({0, max(0, W - qwin)}); ys = sorted({0, max(0, H - qwin)})
        for qy in ys:
            for qx in xs:
                qw, qh = min(qwin, W - qx), min(qwin, H - qy)
                quad = img.crop((qx, qy, qx + qw, qy + qh)).resize(
                    (qw * up, qh * up), Image.BILINEAR)
                feat, _ = nf.feat_for_pil(net, quad)
                det = model(torch.from_numpy(feat.astype(np.float32))[None].to(device))
                bx, sc = decode(det.cpu(), score_thr=score_thr, stride=STRIDE8, topk=topk)
                bx = bx.numpy(); sc = sc.numpy()
                cx = (bx[:, 0] + bx[:, 2]) / 2; cy = (bx[:, 1] + bx[:, 3]) / 2
                keep = (cx >= 0) & (cx < qw * up) & (cy >= 0) & (cy < qh * up)
                bx, sc = bx[keep], sc[keep]
                bx = bx / up                                     # upscaled -> quad coords
                bx[:, [0, 2]] += qx; bx[:, [1, 3]] += qy         # quad -> tile coords
                boxes_all.append(bx); scores_all.append(sc)
        if boxes_all:
            B = np.concatenate(boxes_all); S = np.concatenate(scores_all)
        else:
            B = np.zeros((0, 4)); S = np.zeros(0)
        # confine to tile, drop pad, cross-scale NMS
        cx = (B[:, 0] + B[:, 2]) / 2; cy = (B[:, 1] + B[:, 3]) / 2
        m = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
        B, S = B[m], S[m]
        B[:, [0, 2]] = B[:, [0, 2]].clip(0, W); B[:, [1, 3]] = B[:, [1, 3]].clip(0, H)
        if len(B):
            ki = nms(torch.from_numpy(B).float(), torch.from_numpy(S).float(),
                     ms_nms_iou).numpy()
            B, S = B[ki], S[ki]
        preds[plot] = {"boxes": B.tolist(), "scores": S.tolist()}
        if (k + 1) % 50 == 0 or k + 1 == len(gt):
            print(f"  ms-predicted {k+1}/{len(gt)}", flush=True)
    json.dump(preds, open(out_pred_json, "w"))
    print(f"wrote {out_pred_json}", flush=True)
    return preds


def _soft_nms(boxes, scores, sigma=0.5, score_min=1e-3):
    """Gaussian soft-NMS: keep all boxes but decay a box's score by exp(-iou^2/sigma)
    against every higher-scoring kept box. Recovers dense-canopy crowns that hard NMS
    deletes, and merges TTA duplicates by consensus. Returns (boxes, decayed_scores)."""
    b = np.asarray(boxes, float).reshape(-1, 4).copy()
    s = np.asarray(scores, float).copy()
    if len(b) == 0:
        return b, s
    kb, ks = [], []
    while True:
        i = int(np.argmax(s))
        if s[i] < score_min:
            break
        bi = b[i]; kb.append(bi.copy()); ks.append(s[i]); s[i] = -1
        rest = np.where(s >= score_min)[0]
        if len(rest) == 0:
            continue
        x0 = np.maximum(bi[0], b[rest, 0]); y0 = np.maximum(bi[1], b[rest, 1])
        x1 = np.minimum(bi[2], b[rest, 2]); y1 = np.minimum(bi[3], b[rest, 3])
        inter = (x1 - x0).clip(0) * (y1 - y0).clip(0)
        ai = (bi[2] - bi[0]) * (bi[3] - bi[1])
        ar = (b[rest, 2] - b[rest, 0]) * (b[rest, 3] - b[rest, 1])
        iou = inter / (ai + ar - inter + 1e-9)
        s[rest] *= np.exp(-(iou ** 2) / sigma)
    return np.array(kb).reshape(-1, 4), np.array(ks)


_FLIPS = {  # name -> (PIL transpose, box remap given tile W,H)
    "id":  (None,                    lambda b, W, H: b),
    "h":   ("FLIP_LEFT_RIGHT",       lambda b, W, H: np.stack([W - b[:, 2], b[:, 1], W - b[:, 0], b[:, 3]], 1)),
    "v":   ("FLIP_TOP_BOTTOM",       lambda b, W, H: np.stack([b[:, 0], H - b[:, 3], b[:, 2], H - b[:, 1]], 1)),
    "hv":  ("ROTATE_180",            lambda b, W, H: np.stack([W - b[:, 2], H - b[:, 3], W - b[:, 0], H - b[:, 1]], 1)),
}


@torch.no_grad()
def predict_boxes_tta(ckpt_path, net, eval_feat_dir, eval_gt_json, rgb_dir, out_pred_json,
                      views=("id", "h", "v", "hv"), sigma=0.5, score_thr=0.01, topk=600,
                      merge="soft", nms_iou=0.5, device=None):
    """Flip-TTA + soft-NMS (inference-only). Detects on each flip view (identity uses the
    cached feature; flips re-extract), un-flips boxes to the tile frame, pools all views,
    and soft-NMS-merges. Recovers borderline true crowns (view-consensus) and dense crowns
    (soft-NMS) at low FP risk. views=('id',) => soft-NMS only, no TTA."""
    from PIL import Image
    from boxinst_commonality_tcd_04.modal_neon_multiseed import neon_features as nf
    device = pick_device(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = Detector8(cfg["in_dim"], width=cfg["width"], tower=cfg["tower"]
                      ).to(device).eval()
    model.load_state_dict(ck["state"])
    gt = json.load(open(eval_gt_json))
    preds = {}
    for k, plot in enumerate(sorted(gt)):
        img = Image.open(os.path.join(rgb_dir, plot + ".tif")).convert("RGB")
        W, H = img.size
        B, S = [], []
        for vn in views:
            tr, remap = _FLIPS[vn]
            if vn == "id" and os.path.exists(os.path.join(eval_feat_dir, plot + ".npy")):
                feat = np.load(os.path.join(eval_feat_dir, plot + ".npy")).astype(np.float32)
            else:
                vimg = img if tr is None else img.transpose(getattr(Image, tr))
                feat, _ = nf.feat_for_pil(net, vimg)
                feat = feat.astype(np.float32)
            det = model(torch.from_numpy(feat)[None].to(device))
            bx, sc = decode(det.cpu(), score_thr=score_thr, stride=STRIDE8, topk=topk,
                            nms_iou=0.9)                     # permissive: soft-NMS does merging
            bx = bx.numpy()
            if len(bx):
                bx = remap(bx, W, H)
            B.append(bx.reshape(-1, 4)); S.append(sc.numpy())
        B = np.concatenate(B) if B else np.zeros((0, 4)); S = np.concatenate(S) if S else np.zeros(0)
        cx = (B[:, 0] + B[:, 2]) / 2; cy = (B[:, 1] + B[:, 3]) / 2
        m = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
        B, S = B[m], S[m]
        if merge == "soft":
            B, S = _soft_nms(B, S, sigma=sigma)          # keep-all-decayed (dense recovery)
        elif len(B):                                     # "hard": consolidate TTA views
            from torchvision.ops import nms
            ki = nms(torch.from_numpy(B).float(), torch.from_numpy(S).float(),
                     nms_iou).numpy()
            B, S = B[ki], S[ki]
        if len(B):
            B[:, [0, 2]] = B[:, [0, 2]].clip(0, W); B[:, [1, 3]] = B[:, [1, 3]].clip(0, H)
        preds[plot] = {"boxes": B.tolist(), "scores": S.tolist()}
        if (k + 1) % 50 == 0 or k + 1 == len(gt):
            print(f"  tta-predicted {k+1}/{len(gt)} (views={list(views)})", flush=True)
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


def run_multiseed(feat_dir, gt_path, eval_feat_dir, eval_gt_json, rgb_dir, ckpt_dir,
                  seeds, up=2, epochs=40, device=None, commit=None):
    """AMORTIZED multiseed: preload the train TileData and the eval features ONCE, then
    train+eval each seed reusing the in-RAM caches (vs the old one-container-per-seed
    pattern that paid cold-start + preload N times). Recipe is byte-identical to the
    single-seed path — `data` is read-only during training and set_seed/model-init happen
    per seed inside train_resumable. Per-seed idempotent skip (results json + done flag)
    so a preempted container resumes without redoing finished seeds."""
    import time
    stride = 16 // up
    device = pick_device(device)
    t_pre = time.time()
    data = make_data(feat_dir, gt_path, preload=True, stride=stride)   # TRAIN, once
    eval_cache = preload_eval(eval_feat_dir, eval_gt_json)             # EVAL, once
    print(f"[multiseed] preloaded train+eval in {(time.time()-t_pre)/60:.1f} min; "
          f"seeds={list(seeds)} up={up} stride={stride}", flush=True)
    results = {}
    for seed in seeds:
        tag = f"neon_s{seed}" + ("" if up == 2 else f"_up{up}")
        res_fp = os.path.join(ckpt_dir, f"results_{tag}.json")
        done_fp = os.path.join(ckpt_dir, f"done_{tag}.flag")
        if os.path.exists(res_fp) and os.path.exists(done_fp):
            r = json.load(open(res_fp))
            if "best_f1_point" in r:
                print(f"[multiseed] seed {seed} ({tag}): already done -> skip",
                      flush=True)
                results[seed] = r
                continue
        t_s = time.time()
        best = train_resumable(feat_dir, gt_path, ckpt_dir, tag=tag, seed=seed,
                               epochs=epochs, device=device, up=up, commit=commit,
                               data=data)
        ckpt = os.path.join(ckpt_dir, f"det_{tag}.pt")
        predict_boxes(ckpt, eval_feat_dir, eval_gt_json, rgb_dir,
                      os.path.join(ckpt_dir, f"preds_{tag}.json"), device=device,
                      eval_cache=eval_cache)
        res = score_predictions(os.path.join(ckpt_dir, f"preds_{tag}.json"),
                                eval_gt_json, res_fp)
        res["best_val_boxAP50"] = round(best["mAP50"], 4)
        res["best_epoch"] = best["epoch"]
        res["seed_min"] = round((time.time() - t_s) / 60, 1)
        json.dump(res, open(res_fp, "w"), indent=2)
        if commit:
            commit()
        bf = res["best_f1_point"]
        print(f"[multiseed] seed {seed} ({tag}): best-F1 P={bf['P']} R={bf['R']} "
              f"in {res['seed_min']} min", flush=True)
        results[seed] = res
    return results
