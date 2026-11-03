"""Pre-registration D1 (dino_det/README.md): DINO-Swin-L (detrex, COCO init) trained on LACE's OAM-TCD boxes with Tier B's
data path, LACE's canopy rule, Tier B's plateau schedule and BoxTeacher's selection set. Runs INSIDE the Modal container
(dino_modal.py), one process per GPU (detectron2 launch).

Run dir RUN: train.log, evals.jsonl, plateau.json, last.pth (+.tmp), model_best.pth, best.json, STOPPED.
Budget: BUDGET_FILE (written by dino_modal before launch) = {"cap_usd", "spent_usd_before", "usd_per_hour", "t0"}; rank 0
stops at an epoch end when spent_before + elapsed * rate + one more epoch > cap.
"""
from __future__ import annotations

import argparse, json, os, random, sys, time

import numpy as np
import torch

DETREX = os.environ.get("DETREX_ROOT", "/opt/detrex")
CFG = "projects/dino/configs/dino-swin/dino_swin_large_384_5scale_36ep.py"
COCO_CKPT = "/root/ckpt/dino_swin_large_384_5scale_36ep.pth"
VAL_SUB_GT = "/root/tierb_data/val_sub_gt_1024.json"
TRAIN_GT = "/root/proj/train_tiles_gt.json"
LR, WD, BB_FACTOR, CLIP, PER_GPU = 1e-4, 1e-4, 0.1, 0.1, 2
CANOPY_IOA = 0.5


# ------------------------------------------------------------------ canopy ignore (pre-registration D1)
def install_canopy_patch():
    """Unmatched queries whose predicted box is >= 50% canopy get no negative focal loss (final, aux, encoder outputs);
    matched and denoising queries unchanged. Canopy rasters ride in batched_inputs['canopy'] (TierBMapper)."""
    import torch.nn.functional as F
    from detrex.modeling.criterion.criterion import SetCriterion
    from detrex.layers import box_cxcywh_to_xyxy
    sys.path.insert(0, DETREX)
    from projects.dino.modeling.dino import DINO
    from projects.dino.modeling.dn_criterion import DINOCriterion

    fwd0, prep0 = DINO.forward, DINO.prepare_targets

    def forward(self, batched_inputs):
        self._tcd_canopy = ([x["canopy"].to(self.device) for x in batched_inputs]
                            if self.training and "canopy" in batched_inputs[0] else None)
        try:
            return fwd0(self, batched_inputs)
        finally:
            self._tcd_canopy = None

    def prepare_targets(self, targets):
        out = prep0(self, targets)
        if getattr(self, "_tcd_canopy", None) is not None:
            for t, c in zip(out, self._tcd_canopy):
                ii = torch.zeros((c.shape[0] + 1, c.shape[1] + 1), dtype=torch.float32, device=c.device)
                ii[1:, 1:] = c.float().cumsum(0).cumsum(1)
                t["canopy_ii"] = ii
        return out

    dn0 = DINOCriterion.compute_dn_loss

    def compute_dn_loss(self, *a, **k):
        self._tcd_in_dn = True
        try:
            return dn0(self, *a, **k)
        finally:
            self._tcd_in_dn = False

    def loss_labels(self, outputs, targets, indices, num_boxes, **kwargs):
        assert self.loss_class_type == "focal_loss"
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        onehot = torch.zeros([*src_logits.shape[:2], src_logits.shape[2] + 1], dtype=src_logits.dtype, device=src_logits.device)
        onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        onehot = onehot[:, :, :-1]
        w = torch.ones(src_logits.shape[:2], dtype=src_logits.dtype, device=src_logits.device)
        if not getattr(self, "_tcd_in_dn", False) and "canopy_ii" in targets[0]:
            with torch.no_grad():
                for b, t in enumerate(targets):
                    ii = t["canopy_ii"]; H, W = ii.shape[0] - 1, ii.shape[1] - 1
                    bx = box_cxcywh_to_xyxy(outputs["pred_boxes"][b].detach().float()) * torch.tensor([W, H, W, H], device=ii.device)
                    x0 = bx[:, 0].floor().clamp(0, W).long(); x1 = bx[:, 2].ceil().clamp(0, W).long()
                    y0 = bx[:, 1].floor().clamp(0, H).long(); y1 = bx[:, 3].ceil().clamp(0, H).long()
                    area = ((x1 - x0) * (y1 - y0)).clamp(min=1).float()
                    cov = (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]) / area
                    w[b] = torch.where((target_classes[b] == self.num_classes) & (cov >= CANOPY_IOA), 0.0, 1.0)
        prob = src_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(src_logits, onehot, reduction="none")
        p_t = prob * onehot + (1 - prob) * (1 - onehot)
        loss = ce * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            loss = (self.alpha * onehot + (1 - self.alpha) * (1 - onehot)) * loss
        loss = loss * w.unsqueeze(-1)
        return {"loss_class": loss.mean(1).sum() / num_boxes * src_logits.shape[1]}     # = released reduction

    DINO.forward, DINO.prepare_targets = forward, prepare_targets
    DINOCriterion.compute_dn_loss = compute_dn_loss
    SetCriterion.loss_labels = loss_labels


def build_model(device):
    from detectron2.config import LazyConfig, instantiate
    from detrex.checkpoint import DetectionCheckpointer
    cfg = LazyConfig.load(os.path.join(DETREX, CFG))
    cfg.model.num_classes = 1
    cfg.model.device = device
    model = instantiate(cfg.model).to(device)
    assert model.criterion.num_classes == 1 and model.num_classes == 1
    load0 = torch.load
    torch.load = lambda f, *a, **k: load0(f, *a, **{**k, "weights_only": False})
    try:
        DetectionCheckpointer(model).load(COCO_CKPT)
    finally:
        torch.load = load0
    return model, cfg


def make_optimizer(model):
    bb = [p for n, p in model.named_parameters() if p.requires_grad and "backbone" in n]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and "backbone" not in n]
    return torch.optim.AdamW([{"params": bb, "lr": LR * BB_FACTOR, "base": LR * BB_FACTOR},
                              {"params": rest, "lr": LR, "base": LR}], lr=LR, weight_decay=WD)


def _collate(x):
    return x


class _Mapped(torch.utils.data.Dataset):
    def __init__(self, dicts, idx, mapper):
        self.dicts, self.idx, self.mapper = dicts, idx, mapper

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.mapper(self.dicts[self.idx[i]])


def log(run, msg):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(os.path.join(run, "train.log"), "a") as f:
        f.write(line + "\n")


def atomic_save(obj, path):
    torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)


@torch.no_grad()
def window_infer(model, rgb, oy, ox, size=1024):
    a = rgb[oy:oy + size, ox:ox + size]
    out = model([{"image": torch.as_tensor(np.ascontiguousarray(a.transpose(2, 0, 1))), "height": size, "width": size}])[0]["instances"]
    return out.pred_boxes.tensor.cpu().numpy(), out.scores.cpu().numpy()


def evaluate(model, val_gt, subs, tids, rank, world):
    from tierb import tierb_common as TC
    from tierb import score_lite as SL
    model.eval(); dets = []
    for t in tids[rank::world]:
        rgb = TC.load_rgb(t)
        for iid, oy, ox in subs[t]:
            b, s = window_infer(model, rgb, oy, ox)
            dets += SL.window_dets(iid, b.tolist(), s.tolist())
    model.train()
    return dets


def main_worker(args):
    import detectron2.utils.comm as comm
    from torch.nn.parallel import DistributedDataParallel as DDP
    sys.path.insert(0, DETREX); sys.path.insert(0, "/root")
    from tierb import crops as C, plateau as PL, score_lite as SL, tierb_common as TC
    from tierb.d2 import d2_data as D
    rank, world = comm.get_rank(), comm.get_world_size()
    dev = f"cuda:{comm.get_local_rank()}"; torch.cuda.set_device(dev)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False      # fp32, as released (amp off)
    run = args.run; os.makedirs(run, exist_ok=True)
    seed = 0
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed + rank)
    install_canopy_patch()
    model, _ = build_model(dev)
    opt = make_optimizer(model)
    plateau = PL.Plateau(os.path.join(run, "plateau.json"), **(PL.SMOKE if args.smoke else {}))
    epoch0 = 0
    lp = os.path.join(run, "last.pth")
    if os.path.exists(lp):
        ck = torch.load(lp, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); epoch0 = ck["epoch"]
        if rank == 0:
            log(run, f"[resume] from epoch {epoch0}")
    plateau.load()
    ddp = DDP(model, device_ids=[comm.get_local_rank()], find_unused_parameters=False) if world > 1 else model
    dicts = D.train_dicts(TRAIN_GT)
    mapper = D.TierBMapper(image_format="RGB", aug_type="none")
    val_gt = json.load(open(VAL_SUB_GT)); subs = {}
    for im in val_gt["images"]:
        subs.setdefault(im["tile_id"], []).append((im["id"], im["oy"], im["ox"]))
    val_tids = sorted(subs)
    if args.smoke:
        val_tids = val_tids[:1]
    n_iter = len(dicts) // (PER_GPU * world)
    if args.smoke:
        n_iter = args.smoke_iters
    budget = json.load(open(args.budget)) if args.budget and os.path.exists(args.budget) else None
    if rank == 0:
        log(run, f"[dino] world {world} per_gpu {PER_GPU} batch {PER_GPU * world} iters/epoch {n_iter} crops {len(dicts)} "
                 f"lr {LR} bb x{BB_FACTOR} clip {CLIP} plateau {plateau.params or 'default'} budget {budget}")
    max_epochs = 1 if args.smoke else PL.MAX_EPOCHS
    t_start = time.time(); ep_secs = []
    for ep in range(epoch0, max_epochs):
        if plateau.state["stop"]:
            break
        for g in opt.param_groups:
            g["lr"] = g["base"] * plateau.lr_factor
        perm = np.random.default_rng(1000 + ep).permutation(len(dicts))           # same on every rank
        mine = [int(perm[((it * world + rank) * PER_GPU + j) % len(dicts)]) for it in range(n_iter) for j in range(PER_GPU)]
        dl = torch.utils.data.DataLoader(_Mapped(dicts, mine, mapper), batch_size=PER_GPU, shuffle=False, num_workers=4,
                                         collate_fn=_collate, persistent_workers=False, prefetch_factor=2,
                                         multiprocessing_context="fork")     # CPU-only workers; spawn would pickle everything
        te = time.time(); losses = []
        for it, batch in enumerate(dl):
            ld = ddp(batch); loss = sum(ld.values())
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP, 2)
            opt.step(); losses.append(float(loss.detach()))
            if rank == 0 and (it in (0, 9, 49) or (it + 1) % 50 == 0):
                log(run, f"  ep{ep+1} it{it+1}/{n_iter} loss {np.mean(losses[-10:]):.3f} {time.time()-te:.0f}s "
                         f"mem {torch.cuda.max_memory_allocated(dev)/2**30:.1f} GB")
        # ---- epoch end: eval (all ranks), record, checkpoint, budget
        t_eval = time.time()
        dets = evaluate(model, val_gt, subs, val_tids, rank, world)
        allp = comm.gather(dets, dst=0)
        stop_budget = torch.zeros(1, device=dev)
        if rank == 0:
            merged = [d for p in allp for d in p]
            gt = val_gt if not args.smoke else {**val_gt, "images": [i for i in val_gt["images"] if i["tile_id"] in val_tids],
                                                "annotations": [a for a in val_gt["annotations"] if a["image_id"] in {i["id"] for i in val_gt["images"] if i["tile_id"] in val_tids}]}
            box = SL.cocoeval(gt, merged, "bbox")
            best = json.load(open(os.path.join(run, "best.json"))) if os.path.exists(os.path.join(run, "best.json")) else None
            rec = {"epoch": ep + 1, "val_box": box, "train_loss": round(float(np.mean(losses)), 4), "train_s": round(t_eval - te, 1),
                   "eval_s": round(time.time() - t_eval, 1), "lr_factor": plateau.lr_factor, "time": time.strftime("%F %T")}
            if best is None or box["AP50"] > best["val_box"]["AP50"]:
                atomic_save({"model": model.state_dict(), "epoch": ep + 1, "val_box": box}, os.path.join(run, "model_best.pth"))
                TC.atomic_json(os.path.join(run, "best.json"), rec, indent=1); rec["new_best"] = True
            st = plateau.record(ep + 1, box["AP50"])
            rec["plateau"] = {k: st[k] for k in ("best", "best_epoch", "bad", "decays", "lr_factor", "stop", "reason")}
            atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1}, lp)
            TC.append_jsonl(os.path.join(run, "evals.jsonl"), rec)
            ep_secs.append(time.time() - te)
            if budget:
                spent = budget["spent_usd_before"] + (time.time() - budget["t0"]) / 3600 * budget["usd_per_hour"]
                nxt = np.mean(ep_secs) / 3600 * budget["usd_per_hour"]
                rec["spend_usd"] = round(spent, 2)
                if spent + nxt > budget["cap_usd"] - budget.get("reserve_usd", 0):
                    stop_budget[0] = 1
                    open(os.path.join(run, "STOPPED"), "w").write(f"budget: spent ${spent:.2f} + next epoch ${nxt:.2f} > cap")
            log(run, f"[eval] ep {ep+1} val box AP50 {box['AP50']} AP75 {box['AP75']} loss {rec['train_loss']} "
                     f"train {rec['train_s']}s eval {rec['eval_s']}s spend ${rec.get('spend_usd')} plateau {rec['plateau']}"
                     f"{' NEW BEST' if rec.get('new_best') else ''}")
        comm.synchronize()
        torch.distributed.broadcast(stop_budget, 0) if world > 1 else None
        plateau.load()
        if stop_budget.item() > 0:
            if rank == 0:
                log(run, "[stop] budget cap reached; checkpoint written, resumable")
            break
    if rank == 0:
        log(run, f"[dino] DONE plateau {plateau.state.get('reason')} best ep {plateau.state.get('best_epoch')} "
                 f"AP50 {plateau.state.get('best')} total {time.time()-t_start:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--smoke_iters", type=int, default=20)
    ap.add_argument("--budget", default="")
    args = ap.parse_args()
    from detectron2.engine import launch
    launch(main_worker, args.gpus, dist_url="auto", args=(args,))


if __name__ == "__main__":
    main()
