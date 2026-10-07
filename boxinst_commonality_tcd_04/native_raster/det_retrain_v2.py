"""Detector retrain v2 — `train_detector_tiles.train` with three knobs exposed, plus resume.

WHAT CHANGES vs the published recipe (decisions in NATIVE_RASTER.md "DETECTOR RETRAIN"):
  - loss weights: size x w_size (published 0.1), GIoU x w_giou (published 1.0);
  - checkpoint/early-stop metric: val box `es_metric` from dapt.eval.full_report
    ("mAP50" published, "mAP50_95" for the pilot) — still box-only;
  - min_epochs / es_patience / es_min_delta / epochs are arguments (unchanged semantics).
Model, targets, data, optimiser, schedule, batch order and eval are the published code paths:
TileData, infer, set_seed, Detector8 (swapped to Detector4Phase by the caller, as
phase4_lib_tcd.train_4p does), det_loss, pick_threshold, full_report are imported, not copied.

LEGACY GATE. With w_size=0.1, w_giou=1.0, es_metric="mAP50" the loop is the published
`train` line for line (same RNG consumption, same op order), so it must reproduce it bitwise:
gate_det_retrain_v2.py checks that before any GPU run.

RESUME. A full training state (model, optimiser, scheduler, RNGs, best, patience) is written to
`state_{tag}.pt` after every epoch; a restarted call continues from it. Resumed runs are not
guaranteed bitwise equal to an uninterrupted run (cuDNN), which is recorded in the cfg.

NEVER OVERWRITES: refuses to start if det_{tag}.pt exists without a matching state file.
"""
from __future__ import annotations

import os

import numpy as np
import torch

import boxinst_commonality_tcd_04.train_detector_tiles as T
from boxinst_commonality_tcd_04.detector import STRIDE8


def train_v2(args, art_dir, persist=None, log=print):
    """args: Namespace with the published train() fields plus w_size, w_giou, es_metric.
    persist(path) is called after every file write (Modal: vol.commit)."""
    T.set_seed(args.seed)
    device = T.pick_device(args.device)
    data = T.TileData(args.arm, canvas=getattr(args, "canvas", 2048),
                      gt_path=getattr(args, "gt_path", None))
    tr, va = data.partition("train"), data.partition("val")
    in_dim = data._feat(tr[0]).shape[0]
    model = T.Detector8(in_dim, width=args.width, tower=args.tower).to(device)
    npar = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    metric = args.es_metric
    ckpt_path = os.path.join(art_dir, f"det_{args.tag}.pt")
    state_path = os.path.join(art_dir, f"state_{args.tag}.pt")
    os.makedirs(art_dir, exist_ok=True)
    log(f"[{args.tag}] device={device} in_dim={in_dim} params={npar:.2f}M "
        f"train/val={len(tr)}/{len(va)} seed={args.seed} w_size={args.w_size} "
        f"w_giou={args.w_giou} es_metric={metric} min_ep={args.min_epochs} "
        f"patience={args.es_patience} min_delta={args.es_min_delta} cap={args.epochs}")

    rng = np.random.default_rng(args.seed)
    best = {"metric": -1, "state": None, "epoch": -1, "thr": 0.2, "rep": None}
    no_improve, start, resumed_from = 0, 0, None
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); rng.bit_generator.state = st["np_rng"]
        torch.set_rng_state(st["torch_rng"])
        if torch.cuda.is_available() and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        best, no_improve, start = st["best"], st["no_improve"], st["epoch"]
        resumed_from = start
        log(f"[{args.tag}] RESUME from epoch {start} (best ep{best['epoch']} "
            f"{metric}={best['metric']:.4f}, no_improve={no_improve})")
        if st.get("stopped"):
            log(f"[{args.tag}] state says training already finished -> nothing to do")
            return best
    else:
        assert not os.path.exists(ckpt_path), \
            f"{ckpt_path} exists with no state file -- refusing to overwrite"

    stopped = False
    for ep in range(start, args.epochs):
        model.train()
        order = list(tr); rng.shuffle(order)
        losses = []
        for i in range(0, len(order), args.bs):
            b = data.batch(order[i:i + args.bs], device)
            det = model(b["feat"])
            l_hm, l_off, l_size, l_giou = T.det_loss(det, b, w_size=args.w_size)
            l_giou = l_giou * args.w_giou if args.w_giou != 1.0 else l_giou
            loss = l_hm + l_off + l_size + l_giou
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append([l_hm.item(), l_off.item(), l_size.item(), l_giou.item()])
        sched.step()
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            m = np.mean(losses, 0)
            preds, gts = T.infer(model, data, va, device)
            thr, _ = T.pick_threshold(preds, gts)
            rep = T.full_report(preds, gts, thr)
            v = rep[metric]
            log(f"  ep{ep+1:3d} hm={m[0]:.3f} off={m[1]:.3f} size={m[2]:.3f} "
                f"giou={m[3]:.3f} | val boxAP50={rep['mAP50']:.4f} "
                f"boxAP50:95={rep['mAP50_95']:.4f} F1={rep['f1']:.3f}@{thr:.2f} "
                f"lr={opt.param_groups[0]['lr']:.1e}")
            improved = v > best["metric"] + args.es_min_delta
            if v > best["metric"]:
                best = {"metric": v, "epoch": ep + 1, "thr": thr,
                        "rep": {k: rep[k] for k in ("mAP50", "mAP50_95", "f1")},
                        "state": {k: t.cpu().clone() for k, t in model.state_dict().items()}}
                cfg = {"tag": args.tag, "seed": args.seed, "in_dim": in_dim,
                       "grid_stride": STRIDE8, "width": args.width, "tower": args.tower,
                       "best_epoch": best["epoch"], "score_thr": best["thr"],
                       "nms_iou": 0.5, "n_train": len(tr), "arm": args.arm,
                       "val_boxAP50": round(rep["mAP50"], 4),
                       "val_boxAP50_95": round(rep["mAP50_95"], 4),
                       "es_metric": metric, "w_size": args.w_size, "w_giou": args.w_giou,
                       "min_epochs": args.min_epochs, "es_patience": args.es_patience,
                       "es_min_delta": args.es_min_delta, "epochs_cap": args.epochs,
                       "resumed_from_epoch": resumed_from,
                       "recipe": "det_retrain_v2 (native_raster/NATIVE_RASTER.md)",
                       "data": "full 2048 train tiles, no windowing"}
                torch.save({"state": best["state"], "cfg": cfg}, ckpt_path)
                log(f"    ^ saved best ep{best['epoch']} {metric}={v:.4f}")
                if persist:
                    persist(ckpt_path)
            no_improve = 0 if improved else no_improve + 1
            if ep + 1 >= args.min_epochs and no_improve >= args.es_patience:
                log(f"    early-stop: {no_improve} evals without >{args.es_min_delta} "
                    f"improvement (best ep{best['epoch']})")
                stopped = True
        last = stopped or ep + 1 == args.epochs
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "np_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available()
                    else None,
                    "best": best, "no_improve": no_improve, "epoch": ep + 1,
                    "stopped": last}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
        if persist:
            persist(state_path)
        if stopped:
            break
    log(f"[{args.tag}] best ep{best['epoch']} val {metric}={best['metric']:.4f} "
        f"-> {ckpt_path}")
    return best
