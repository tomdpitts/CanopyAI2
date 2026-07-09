"""Light fine-tune of DINOv3-web ViT-L on OAM-TCD semantic segmentation.

Unfreeze the last 4 transformer blocks (20-23) + the final norm; freeze everything
before (patch embed + blocks 0-19). Because nothing before block 20 is trainable,
the frozen prefix runs grad-free automatically and the backward pass only traverses
the last 4 blocks -> fast + light on MPS. Multi-layer (last-4 hidden states) 3x3
conv decoder on top. Full 4169-tile train set with a held-out val slice for
best-checkpoint selection; final eval on the 439-tile holdout in the matched
protocol (fg=cat1u2, sliding-window). Saves trainable backbone params + decoder.

Baselines to beat (matched protocol): frozen probe 0.874 / frozen strong head 0.876
/ SegFormer-b5 0.8945.
"""
import argparse
import json
import math
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
sys.path.insert(0, os.path.join(HERE, ".."))          # dino/
ROOT = os.path.join(HERE, "..", "..")
from tcd_data import load_split  # noqa: E402
from eval import aggregate_semantic, semantic_counts  # noqa: E402

MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
N_BLOCKS = 24
PROG = os.path.join(HERE, "progress.jsonl")
CKPT = os.path.join(HERE, "ckpt", "ft_best.pt")


class FTEncoder(nn.Module):
    def __init__(self, last_k, n_feat_layers, device):
        super().__init__()
        self.bb = AutoModel.from_pretrained(MODEL_ID).to(device)
        self.patch = getattr(self.bb.config, "patch_size", 16)
        self.C = self.bb.config.hidden_size
        self.n_feat = n_feat_layers
        self.device = device
        keep = set(range(N_BLOCKS - last_k, N_BLOCKS))
        n_train = 0
        for name, p in self.bb.named_parameters():
            tr = False
            if name.startswith("model.layer."):
                idx = int(name.split(".")[2])
                tr = idx in keep
            if name.startswith("norm."):          # final norm
                tr = True
            p.requires_grad_(tr)
            if tr:
                n_train += p.numel()
        self.n_train = n_train
        proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.register_buffer("mean", torch.tensor(proc.image_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(proc.image_std).view(1, 3, 1, 1))
        self.to(device)

    def features(self, x):
        x = (x - self.mean) / self.std
        hs = self.bb(pixel_values=x, output_hidden_states=True).hidden_states
        h, w = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        feats = [hl[:, -(h * w):, :] for hl in hs[-self.n_feat:]]
        return torch.cat(feats, -1).transpose(1, 2).reshape(x.shape[0], self.C * self.n_feat, h, w)


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


@torch.no_grad()
def evaluate(enc, dec, tiles, dev, size=512, scales=(512,), flips=False):
    """Sliding-window (multi-scale) semantic eval. flips=True adds h/v-flip TTA:
    average logits over the original + horizontally + vertically flipped tile."""
    enc.eval(); dec.eval()

    def _sliding(rgb):
        H, W = rgb.shape[:2]
        acc = np.zeros((2, H, W), np.float32); cnt = np.zeros((H, W), np.float32)
        for s in scales:
            ys = sorted(set(list(range(0, max(1, H - s) + 1, s)) + [max(0, H - s)]))
            xs = sorted(set(list(range(0, max(1, W - s) + 1, s)) + [max(0, W - s)]))
            for y in ys:
                for x in xs:
                    cr = np.ascontiguousarray(rgb[y:y + s, x:x + s].transpose(2, 0, 1))
                    lg = dec(enc.features(torch.from_numpy(cr)[None].to(dev)), cr.shape[1:])[0].cpu().numpy()
                    acc[:, y:y + s, x:x + s] += lg; cnt[y:y + s, x:x + s] += 1
        return acc / np.maximum(cnt, 1), cnt

    counts = []
    for t in tiles:
        rgb = read_rgb(t.image_path)
        logits, cnt = _sliding(rgb)
        if flips:                                          # un-flip each view before averaging
            logits = logits + _sliding(np.ascontiguousarray(rgb[:, ::-1]))[0][:, :, ::-1]
            logits = logits + _sliding(np.ascontiguousarray(rgb[::-1]))[0][:, ::-1, :]
        counts.append(semantic_counts((logits[1] > logits[0]) & (cnt > 0), t.semantic_mask()))
    enc.train(); dec.train()
    return aggregate_semantic(counts)


def log(rec):
    with open(PROG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("[ft] " + " ".join(f"{k}={v}" for k, v in rec.items()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-k", type=int, default=4)
    ap.add_argument("--feat-layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr-bb", type=float, default=3e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--val-tiles", type=int, default=200)
    ap.add_argument("--val-eval", type=int, default=80)     # subset for periodic val
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--train-limit", type=int, default=0)   # 0 = full 4169
    ap.add_argument("--eval-limit", type=int, default=0)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    open(PROG, "w").close()
    t0 = time.time()

    enc = FTEncoder(a.last_k, a.feat_layers, dev)
    dec = Decoder(enc.C * a.feat_layers).to(dev)
    log({"event": "init", "trainable_bb_M": round(enc.n_train / 1e6, 2),
         "decoder_M": round(sum(p.numel() for p in dec.parameters()) / 1e6, 2), "dev": dev})

    all_train = load_split(os.path.join(ROOT, "data/tcd/train"))
    rng = np.random.default_rng(0)
    rng.shuffle(all_train)
    val = all_train[:a.val_tiles]
    train = all_train[a.val_tiles:]
    if a.train_limit:
        train = train[:a.train_limit]
    test = load_split(os.path.join(ROOT, "data/tcd/test"))
    if a.eval_limit:
        test = test[:a.eval_limit]
    val_eval = val[:a.val_eval]
    log({"event": "data", "train": len(train), "val": len(val), "test": len(test)})

    opt = torch.optim.AdamW(
        [{"params": [p for p in enc.bb.parameters() if p.requires_grad], "lr": a.lr_bb},
         {"params": dec.parameters(), "lr": a.lr_head}], weight_decay=1e-4)

    def lr_factor(step):
        if step < a.warmup:
            return step / a.warmup
        prog = (step - a.warmup) / max(1, a.steps - a.warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    lossf = nn.CrossEntropyLoss()

    s = a.crop

    def sample():
        t = train[rng.integers(len(train))]
        rgb = read_rgb(t.image_path); m = t.semantic_mask().astype(np.int64)
        H, W = m.shape
        y0 = rng.integers(0, max(1, H - s)); x0 = rng.integers(0, max(1, W - s))
        return rgb[y0:y0 + s, x0:x0 + s].transpose(2, 0, 1), m[y0:y0 + s, x0:x0 + s]

    enc.train(); dec.train()
    best_f1 = -1.0
    run_loss = []
    for it in range(1, a.steps + 1):
        crs, cms = zip(*[sample() for _ in range(a.batch)])
        xt = torch.from_numpy(np.stack(crs)).to(dev)
        yt = torch.from_numpy(np.stack(cms)).to(dev)
        logits = dec(enc.features(xt), (s, s))
        loss = lossf(logits, yt)
        opt.zero_grad()
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in enc.bb.parameters() if p.requires_grad], 1.0)
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            opt.step()
        sched.step()
        run_loss.append(float(loss))
        if it % 100 == 0:
            log({"step": it, "loss": round(np.mean(run_loss[-100:]), 4),
                 "lr_bb": round(opt.param_groups[0]["lr"], 7), "min": round((time.time() - t0) / 60, 1)})
        if it % a.eval_every == 0 or it == a.steps:
            vf = evaluate(enc, dec, val_eval, dev)["micro_f1"]
            improved = vf > best_f1
            log({"step": it, "val_f1": round(vf, 4), "best": round(max(vf, best_f1), 4),
                 "improved": improved, "min": round((time.time() - t0) / 60, 1)})
            if improved:
                best_f1 = vf
                os.makedirs(os.path.dirname(CKPT), exist_ok=True)
                torch.save({"bb_trainable": {n: p.detach().cpu() for n, p in enc.bb.named_parameters()
                                             if p.requires_grad},
                            "decoder": {k: v.cpu() for k, v in dec.state_dict().items()},
                            "step": it, "val_f1": vf, "args": vars(a)}, CKPT)

    # final: reload best, eval full test multi-scale
    ck = torch.load(CKPT, map_location=dev)
    msd = enc.bb.state_dict()
    for n, v in ck["bb_trainable"].items():
        msd[n] = v.to(dev)
    enc.bb.load_state_dict(msd)
    dec.load_state_dict({k: v.to(dev) for k, v in ck["decoder"].items()})
    res = evaluate(enc, dec, test, dev, scales=(512, 768))
    log({"event": "FINAL", "best_val_step": ck["step"], "best_val_f1": round(ck["val_f1"], 4),
         "test_f1": round(res["micro_f1"], 4), "test_iou": round(res["micro_iou"], 4),
         "total_min": round((time.time() - t0) / 60, 1)})
    json.dump({"test_f1": res["micro_f1"], "test_iou": res["micro_iou"],
               "best_val_f1": ck["val_f1"], "best_step": ck["step"]},
              open(os.path.join(HERE, "ft_result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
