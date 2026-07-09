"""Corrected light fine-tune of DINOv3-web ViT-L on OAM-TCD — v2.

Fixes every lesson from v1 (v1 overfit: val 0.9235 but 439-test 0.864 < frozen 0.874,
because a random-tile val leaked spatial neighbours):
  - Selection val = SPATIALLY-GROUPED hold-out from train (whole 2km bins) so it
    tracks the geographically-separated test. The 439 test is never touched.
  - Lighter touch: last **2** blocks, LR 1e-5, small decoder (hidden 128, last-2
    features), weight decay 0.05, flip augmentation.
  - **Two-phase schedule**: (1) backbone frozen, warm up the fresh head alone;
    (2) unfreeze last-2 blocks + head jointly. Avoids the random head perturbing
    pretrained weights.
  - **Early stopping** on grouped-val (patience), keeping the best checkpoint.

Beat targets (matched protocol): frozen probe 0.874 · SegFormer-b5 0.8945.
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
ROOT = os.path.join(HERE, "..", "..")
from finetune import Decoder, FTEncoder, evaluate, read_rgb  # noqa: E402
from tcd_data import load_split  # noqa: E402
from split import fold_split  # noqa: E402

PROG = os.path.join(HERE, "progress_v2.jsonl")
CKPT = os.path.join(HERE, "ckpt", "ft_v2_best.pt")


def log(rec):
    with open(PROG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("[ftv2] " + " ".join(f"{k}={v}" for k, v in rec.items()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-k", type=int, default=2)
    ap.add_argument("--feat-layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--head-warmup", type=int, default=400)   # phase-1 head-only steps
    ap.add_argument("--steps", type=int, default=2000)        # phase-2 max steps
    ap.add_argument("--lr-warmup", type=int, default=100)     # LR ramp within phase 2
    ap.add_argument("--patience", type=int, default=3)        # evals w/o improvement -> stop
    ap.add_argument("--min-delta", type=float, default=0.0005)  # min val gain to count (small:
    #  fixes the earlier-vs-later tie-break WITHOUT prematurely tripping early-stop on slow anneal gains)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr-bb", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--val-fold", type=int, default=0)   # Restor official fold used as val
    ap.add_argument("--val-eval", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-limit", type=int, default=0)      # cap final test tiles (smoke)
    ap.add_argument("--tag", default="v2")                    # names progress/ckpt/result files
    ap.add_argument("--flips-final", action="store_true")     # h/v-flip TTA on the final test eval
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    global PROG, CKPT
    PROG = os.path.join(HERE, f"progress_{a.tag}.jsonl")
    CKPT = os.path.join(HERE, "ckpt", f"ft_{a.tag}_best.pt")
    open(PROG, "w").close()
    t0 = time.time()

    enc = FTEncoder(a.last_k, a.feat_layers, dev)
    dec = Decoder(enc.C * a.feat_layers, hidden=a.hidden).to(dev)
    bb_train_names = {n for n, p in enc.bb.named_parameters() if p.requires_grad}

    def set_bb(flag):
        for n, p in enc.bb.named_parameters():
            if n in bb_train_names:
                p.requires_grad_(flag)

    log({"event": "init", "trainable_bb_M": round(enc.n_train / 1e6, 2),
         "decoder_M": round(sum(p.numel() for p in dec.parameters()) / 1e6, 2), "dev": dev})

    all_train = load_split(os.path.join(ROOT, "data/tcd/train"))
    train, val, info = fold_split(all_train, val_fold=a.val_fold)
    test = load_split(os.path.join(ROOT, "data/tcd/test"))
    if a.eval_limit:
        test = test[:a.eval_limit]
    rng = np.random.default_rng(1)
    val_eval = list(val); rng.shuffle(val_eval); val_eval = val_eval[:a.val_eval]
    log({"event": "split", **info, "test": len(test), "val_eval": len(val_eval)})

    s = a.crop

    def sample():
        t = train[rng.integers(len(train))]
        rgb = read_rgb(t.image_path); m = t.semantic_mask().astype(np.int64)
        H, W = m.shape
        y0 = rng.integers(0, max(1, H - s)); x0 = rng.integers(0, max(1, W - s))
        cr = rgb[y0:y0 + s, x0:x0 + s]; cm = m[y0:y0 + s, x0:x0 + s]
        if rng.random() < 0.5:
            cr = cr[:, ::-1]; cm = cm[:, ::-1]
        if rng.random() < 0.5:
            cr = cr[::-1]; cm = cm[::-1]
        return np.ascontiguousarray(cr.transpose(2, 0, 1)), np.ascontiguousarray(cm)

    lossf = nn.CrossEntropyLoss()
    state = {"best": -1.0, "since": 0}

    def save_best(step, phase, vf):
        os.makedirs(os.path.dirname(CKPT), exist_ok=True)
        torch.save({"bb_trainable": {n: p.detach().cpu() for n, p in enc.bb.named_parameters()
                                     if n in bb_train_names},
                    "decoder": {k: v.cpu() for k, v in dec.state_dict().items()},
                    "step": step, "phase": phase, "val_f1": vf, "args": vars(a)}, CKPT)

    def eval_and_track(step, phase):
        vf = evaluate(enc, dec, val_eval, dev)["micro_f1"]
        imp = vf > state["best"] + a.min_delta   # keep earlier ckpt at effectively-equal F1
        log({"phase": phase, "step": step, "grouped_val_f1": round(vf, 4),
             "best": round(max(vf, state["best"]), 4), "improved": imp,
             "since_improve": 0 if imp else state["since"] + 1, "min": round((time.time() - t0) / 60, 1)})
        if imp:
            state["best"] = vf; state["since"] = 0; save_best(step, phase, vf)
        else:
            state["since"] += 1
        return imp

    def train_step(opt, clip_params):
        crs, cms = zip(*[sample() for _ in range(a.batch)])
        xt = torch.from_numpy(np.stack(crs)).to(dev)
        yt = torch.from_numpy(np.stack(cms)).to(dev)
        loss = lossf(dec(enc.features(xt), (s, s)), yt)
        opt.zero_grad()
        if torch.isfinite(loss):
            loss.backward()
            for grp in clip_params:
                torch.nn.utils.clip_grad_norm_(grp, 1.0)
            opt.step()
        return float(loss)

    enc.train(); dec.train()

    # ---- Phase 1: head warmup (backbone frozen) ----------------------------- #
    set_bb(False)
    opt_h = torch.optim.AdamW(dec.parameters(), lr=a.lr_head, weight_decay=a.wd)
    for it in range(1, a.head_warmup + 1):
        loss = train_step(opt_h, [dec.parameters()])
        if it % 100 == 0:
            log({"phase": "warmup", "step": it, "loss": round(loss, 4), "min": round((time.time() - t0) / 60, 1)})
        if it % a.eval_every == 0 or it == a.head_warmup:
            eval_and_track(it, "warmup")
    log({"event": "warmup_done", "best": round(state["best"], 4), "min": round((time.time() - t0) / 60, 1)})

    # ---- Phase 2: joint (unfreeze last-k) + early stopping ------------------ #
    set_bb(True)
    state["since"] = 0
    opt = torch.optim.AdamW(
        [{"params": [p for p in enc.bb.parameters() if p.requires_grad], "lr": a.lr_bb},
         {"params": dec.parameters(), "lr": a.lr_head}], weight_decay=a.wd)

    def lr_factor(step):
        if step < a.lr_warmup:
            return step / a.lr_warmup
        prog = (step - a.lr_warmup) / max(1, a.steps - a.lr_warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    stopped = a.steps
    for it in range(1, a.steps + 1):
        loss = train_step(opt, [[p for p in enc.bb.parameters() if p.requires_grad], dec.parameters()])
        sched.step()
        if it % 100 == 0:
            log({"phase": "joint", "step": it, "loss": round(loss, 4),
                 "lr_bb": round(opt.param_groups[0]["lr"], 7), "min": round((time.time() - t0) / 60, 1)})
        if it % a.eval_every == 0 or it == a.steps:
            eval_and_track(it, "joint")
            if state["since"] >= a.patience:
                stopped = it
                log({"event": "EARLY_STOP", "step": it, "best": round(state["best"], 4),
                     "patience": a.patience})
                break

    # ---- Final: load best, eval full 439 test (multi-scale) ----------------- #
    ck = torch.load(CKPT, map_location=dev)
    msd = enc.bb.state_dict()
    for n, v in ck["bb_trainable"].items():
        msd[n] = v.to(dev)
    enc.bb.load_state_dict(msd)
    dec.load_state_dict({k: v.to(dev) for k, v in ck["decoder"].items()})
    res = evaluate(enc, dec, test, dev, scales=(512, 768), flips=a.flips_final)
    log({"event": "FINAL", "tag": a.tag, "flips": a.flips_final,
         "best_ckpt_phase": ck["phase"], "best_ckpt_step": ck["step"],
         "best_grouped_val_f1": round(ck["val_f1"], 4), "stopped_at": stopped,
         "test_f1": round(res["micro_f1"], 4), "test_iou": round(res["micro_iou"], 4),
         "vs_frozen_0.874": round(res["micro_f1"] - 0.874, 4),
         "vs_segformer_0.8945": round(res["micro_f1"] - 0.8945, 4),
         "total_min": round((time.time() - t0) / 60, 1)})
    json.dump({"tag": a.tag, "flips": a.flips_final, "test_f1": res["micro_f1"],
               "test_iou": res["micro_iou"], "best_grouped_val_f1": ck["val_f1"],
               "best_phase": ck["phase"], "best_step": ck["step"], "stopped_at": stopped},
              open(os.path.join(HERE, f"ft_{a.tag}_result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
