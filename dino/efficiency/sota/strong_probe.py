"""Stronger frozen-DINOv3 semantic head, to beat SegFormer-b5 in the MATCHED protocol.

Upgrades over the 1x1 linear probe: (1) multi-layer feature fusion (concat the last
K transformer blocks), (2) a small 3x3 conv decoder instead of 1x1, (3) more train
tiles + steps, (4) multi-scale sliding-window inference. Backbone stays frozen.
Eval uses the same dino/eval semantic protocol the calibration used, so the number
is directly comparable to SegFormer-b5's matched-protocol 0.878.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))    # dino/
from tcd_data import load_split  # noqa: E402
from eval import aggregate_semantic, semantic_counts  # noqa: E402

ROOT = os.path.join(HERE, "..", "..", "..")


class MultiLayerDinov3(nn.Module):
    def __init__(self, model_id, n_layers, device):
        super().__init__()
        self.bb = AutoModel.from_pretrained(model_id).eval().to(device)
        for p in self.bb.parameters():
            p.requires_grad_(False)
        self.patch = getattr(self.bb.config, "patch_size", 16)
        self.C = self.bb.config.hidden_size
        self.n_layers = n_layers
        self.device = device
        proc = AutoImageProcessor.from_pretrained(model_id)
        self.register_buffer("mean", torch.tensor(proc.image_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(proc.image_std).view(1, 3, 1, 1))
        self.to(device)

    @torch.no_grad()
    def features(self, x):
        x = (x - self.mean) / self.std
        hs = self.bb(pixel_values=x, output_hidden_states=True).hidden_states
        h, w = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        feats = [hl[:, -(h * w):, :] for hl in hs[-self.n_layers:]]
        f = torch.cat(feats, -1)                                # (B, h*w, C*n_layers)
        return f.transpose(1, 2).reshape(x.shape[0], self.C * self.n_layers, h, w)


class Decoder(nn.Module):
    def __init__(self, cin, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 2, 1))

    def forward(self, f, size):
        return F.interpolate(self.net(f), size=size, mode="bilinear", align_corners=False)


def read_rgb(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--train-tiles", type=int, default=800)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--scales", default="512,768")
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.time()

    enc = MultiLayerDinov3(a.model, a.layers, dev)
    dec = Decoder(enc.C * a.layers).to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=a.lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.LinearLR(opt, 0.05, 1.0, total_iters=max(1, a.steps // 10))

    train = load_split(os.path.join(ROOT, "data/tcd/train"))[: a.train_tiles]
    test = load_split(os.path.join(ROOT, "data/tcd/test"))
    if a.eval_limit:
        test = test[: a.eval_limit]
    print(f"[strong] train={len(train)} test={len(test)} layers={a.layers} dev={dev}", flush=True)

    rng = np.random.default_rng(0)
    s = a.crop

    def sample_crop():
        t = train[rng.integers(len(train))]
        rgb = read_rgb(t.image_path); m = t.semantic_mask().astype(np.int64)
        H, W = m.shape
        y0 = rng.integers(0, max(1, H - s)); x0 = rng.integers(0, max(1, W - s))
        return rgb[y0:y0 + s, x0:x0 + s].transpose(2, 0, 1), m[y0:y0 + s, x0:x0 + s]

    dec.train()
    for it in range(a.steps):
        crs, cms = zip(*[sample_crop() for _ in range(a.batch)])
        xt = torch.from_numpy(np.stack(crs)).to(dev)
        yt = torch.from_numpy(np.stack(cms)).to(dev)
        logits = dec(enc.features(xt), (s, s))
        loss = lossf(logits, yt)
        opt.zero_grad()
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            opt.step()
        sched.step()
        if it % 200 == 0:
            print(f"[strong] step {it}/{a.steps} loss={loss.item():.4f}", flush=True)

    scales = [int(s) for s in a.scales.split(",")]
    dec.eval()
    counts = []
    for i, t in enumerate(test):
        rgb = read_rgb(t.image_path); H, W = rgb.shape[:2]
        acc = np.zeros((2, H, W), np.float32); cnt = np.zeros((H, W), np.float32)
        for s in scales:
            ys = sorted(set(list(range(0, max(1, H - s) + 1, s)) + [max(0, H - s)]))
            xs = sorted(set(list(range(0, max(1, W - s) + 1, s)) + [max(0, W - s)]))
            for y in ys:
                for x in xs:
                    cr = rgb[y:y + s, x:x + s]
                    xt = torch.from_numpy(cr.transpose(2, 0, 1))[None].to(dev)
                    with torch.no_grad():
                        lg = dec(enc.features(xt), cr.shape[:2])[0].cpu().numpy()
                    acc[:, y:y + s, x:x + s] += lg
                    cnt[y:y + s, x:x + s] += 1
        pred = (acc[1] > acc[0]) & (cnt > 0)
        counts.append(semantic_counts(pred, t.semantic_mask()))
        if i % 50 == 0:
            print(f"[strong] eval {i}/{len(test)}", flush=True)
    res = aggregate_semantic(counts)
    out = {"model": a.model, "layers": a.layers, "train_tiles": len(train), "steps": a.steps,
           "scales": scales, "micro_f1": res["micro_f1"], "micro_iou": res["micro_iou"],
           "wall_clock_s": round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(HERE, "strong_result.json"), "w"), indent=2)
    print(f"\n>>> STRONG frozen DINOv3 multi-layer: F1={res['micro_f1']:.4f} IoU={res['micro_iou']:.4f} "
          f"(vs SegFormer-b5 matched 0.878) [{out['wall_clock_s']}s]", flush=True)


if __name__ == "__main__":
    main()
