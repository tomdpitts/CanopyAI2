"""Fork the v3 detector head and adapt it on BRU (dryland) tiles only.

Backbone = v3 fine-tuned DINOv3, FROZEN. Head = v3 decoder, forked and retrained on
BRU crown boxes (rasterised to masks). Head-only (BRU is ~50 tiles). Seeds BOTH numpy
and torch and RECORDS the seed (project convention). Saves a checkpoint that
viz_predict.py can load directly.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "dino", "efficiency"))
from finetune import Decoder, FTEncoder  # noqa: E402
from data import IMG, load_cohort, load_rgb512  # noqa: E402

CKPT = os.path.join(HERE, "ckpt", "bru_head.pt")


def box_mask(boxes, size=IMG):
    m = np.zeros((size, size), np.int64)
    for x0, y0, x1, y1 in boxes:
        m[max(0, int(y0)):min(size, int(np.ceil(y1))), max(0, int(x0)):min(size, int(np.ceil(x1)))] = 1
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--v3-ckpt", default=os.path.join(HERE, "ckpt", "ft_v3_last4_best.pt"))
    ap.add_argument("--backbone", default="v3", choices=["v3", "web"])   # web = original unmodified DINOv3
    ap.add_argument("--fresh-head", action="store_true")                 # random-init head (no TCD fork)
    ap.add_argument("--domains", default="BRU")                          # e.g. "WON,BRU" (Australian only)
    ap.add_argument("--out", default="bru_head")                         # ckpt basename
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    # --- seed BOTH rngs and record it -------------------------------------- #
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    if dev == "mps":
        torch.mps.manual_seed(a.seed)
    print(f"[bru] SEED={a.seed} dev={dev}", flush=True)
    t0 = time.time()

    # --- frozen backbone (original web OR v3-finetuned) + head ------------- #
    enc = FTEncoder(0, 4, dev)                     # last_k=0 -> fully frozen; feat = last-4 layers
    hidden, bb_save, last_k_save = 128, {}, 0      # 'web' defaults (original DINOv3, empty bb_trainable)
    if a.backbone == "v3":
        ck = torch.load(a.v3_ckpt, map_location=dev)
        msd = enc.bb.state_dict()
        for n, v in ck["bb_trainable"].items():
            msd[n] = v.to(dev)
        enc.bb.load_state_dict(msd)                # backbone = v3 fine-tuned
        hidden = ck["args"].get("hidden", 128)
        bb_save, last_k_save = {n: v.cpu() for n, v in ck["bb_trainable"].items()}, 4
    dec = Decoder(enc.C * 4, hidden=hidden).to(dev)
    if a.backbone == "v3" and not a.fresh_head:
        dec.load_state_dict({k: v.to(dev) for k, v in ck["decoder"].items()})   # fork v3 head
        head_init = "forked-v3"
    else:
        head_init = "fresh"
    print(f"[bru] backbone={a.backbone}(frozen) head={head_init} hidden={hidden}", flush=True)

    # --- selected domains only (no TCD) ------------------------------------ #
    doms = a.domains.split(",")
    recs = [r for r in load_cohort() if r.acq in doms]
    data = [(load_rgb512(r).astype(np.float32) / 255.0, box_mask(r.boxes)) for r in recs]
    n_box = sum(len(r.boxes) for r in recs)
    print(f"[bru] domains={a.domains} tiles={len(data)} crowns={n_box}", flush=True)

    opt = torch.optim.AdamW(dec.parameters(), lr=a.lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    enc.eval(); dec.train()
    for it in range(1, a.steps + 1):
        idx = rng.integers(len(data), size=a.batch)
        crs, cms = [], []
        for i in idx:
            rgb, m = data[i]
            if rng.random() < 0.5:
                rgb = rgb[:, ::-1]; m = m[:, ::-1]
            if rng.random() < 0.5:
                rgb = rgb[::-1]; m = m[::-1]
            crs.append(np.ascontiguousarray(rgb.transpose(2, 0, 1))); cms.append(np.ascontiguousarray(m))
        xt = torch.from_numpy(np.stack(crs)).to(dev)
        yt = torch.from_numpy(np.stack(cms)).to(dev)
        with torch.no_grad():
            feats = enc.features(xt)
        logits = dec(feats, (IMG, IMG))
        loss = lossf(logits, yt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        if it % 50 == 0:
            print(f"[bru] step {it}/{a.steps} loss={loss.item():.4f}", flush=True)

    out_ckpt = os.path.join(HERE, "ckpt", a.out + ".pt")
    torch.save({"bb_trainable": bb_save,
                "decoder": {k: v.cpu() for k, v in dec.state_dict().items()},
                "step": a.steps, "val_f1": -1.0, "seed": a.seed,
                "args": {"last_k": last_k_save, "feat_layers": 4, "hidden": hidden},
                "note": f"backbone={a.backbone}(frozen) head={head_init} domains={a.domains}"},
               out_ckpt)
    print(f">>> saved {out_ckpt}  SEED={a.seed}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
