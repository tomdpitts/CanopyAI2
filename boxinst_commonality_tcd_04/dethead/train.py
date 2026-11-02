"""Train an 8 px detector head on LACE's frozen L24 features, LACE's recipe (dethead/README.md). Local (MPS).

  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.train --head fcos --tag fcos_s0
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.train --head centernet --tag cn_local_s0   # control
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.train --head fpn --tag fpn_s0               # pyramid (prereg 2)

Checkpoints every epoch (runs/<tag>/last.pt, resume) and on every new best (runs/<tag>/best.pt).
"""
import argparse, json, os, random, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from dapt.eval import average_precision
from boxinst_commonality_tcd_04.dethead import fcos as FC

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(PKG, "dethead")
FEAT = {"train": "/Users/tompitts/dphil/feat_cache/feat_4p_train", "val": "/Users/tompitts/dphil/feat_cache/feat_4p_val"}
GT = os.path.join(PKG, "train_tiles_gt.json")
RECIPE = dict(lr=1e-3, wd=1e-4, epochs=40, bs=3, eval_every=5, min_epochs=12, es_patience=2, es_min_delta=0.005)


class Data:
    def __init__(self, head):
        from boxinst_commonality_tcd_04.detector import canopy_cell_mask
        from boxinst_commonality_tcd_04.train_detector_tiles import canopy_px
        gt = json.load(open(GT))
        self.tr = sorted(t for t, v in gt.items() if v["partition"] == "train")
        self.va = sorted(t for t, v in gt.items() if v["partition"] == "val")
        self.boxes = {t: np.asarray(v["boxes"], np.float32).reshape(-1, 4) for t, v in gt.items()}
        self.dir = {**{t: FEAT["train"] for t in self.tr}, **{t: FEAT["val"] for t in self.va}}
        self.tgt, self.nfb = {}, 0
        t0 = time.time()
        for t in self.tr:
            ign = torch.from_numpy(canopy_cell_mask(canopy_px(gt[t].get("canopy", []), res=2048), FC.G))
            if head == "fpn":
                levels, nfb = FC.encode_fpn(self.boxes[t]); self.nfb += nfb; rec = {}
                cpx = canopy_px(gt[t].get("canopy", []), res=2048)
                for l, (pos, ltrb, ctr) in enumerate(levels):
                    g = FC.LEVELS[l][1]
                    rec |= {f"pos{l}": torch.from_numpy(pos), f"ltrb{l}": torch.from_numpy(ltrb), f"ctr{l}": torch.from_numpy(ctr),
                            f"ignore{l}": torch.from_numpy(canopy_cell_mask(cpx, g))}
                self.tgt[t] = rec
            elif head == "fcos":
                pos, ltrb, ctr, nfb = FC.encode(self.boxes[t]); self.nfb += nfb
                self.tgt[t] = {"pos": torch.from_numpy(pos), "ltrb": torch.from_numpy(ltrb), "ctr": torch.from_numpy(ctr), "ignore": ign}
            else:
                from dapt.targets import TargetConfig, encode
                e = encode(self.boxes[t], TargetConfig(grid=256, stride=8))
                self.tgt[t] = {k: torch.from_numpy(e[k]) for k in ("heatmap", "offset", "size", "reg_mask")} | {"ignore": ign}
        print(f"[data] {len(self.tr)} train / {len(self.va)} val; targets {time.time()-t0:.0f}s; fcos fallback boxes {self.nfb}", flush=True)

    def feat(self, t):
        return torch.from_numpy(np.load(os.path.join(self.dir[t], t + ".npy")))

    def batch(self, tids):
        f = torch.stack([self.feat(t) for t in tids])
        return f, {k: torch.stack([self.tgt[t][k] for t in tids]) for k in self.tgt[tids[0]]}


def build(head):
    if head == "fcos":
        return FC.FCOS8(1024, 256, 3)
    if head == "fpn":
        return FC.FCOS8FPN(1024, 256, 3)
    from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.phase4_lib_tcd import Detector4Phase
    return Detector4Phase(1024, width=256, tower=3)


@torch.no_grad()
def predict(model, head, feat, device, nms_iou=None):
    out = model(feat.float()[None].to(device))
    if head == "fpn":
        return FC.decode_fpn(out, nms_iou)
    if head == "fcos":
        if nms_iou is None:
            return FC.decode(out)
        old_, FC.NMS_IOU = FC.NMS_IOU, nms_iou
        try:
            return FC.decode(out)
        finally:
            FC.NMS_IOU = old_
    from dapt.decode import decode
    return decode(out.cpu(), score_thr=0.05, stride=8, topk=600)          # = train_detector_tiles.infer


def evaluate(model, head, data, device):
    model.eval(); preds, gts = [], []
    with ThreadPoolExecutor(4) as ex:
        for t, f in zip(data.va, ex.map(data.feat, data.va)):
            b, s = predict(model, head, f, device)
            preds.append((b.numpy(), s.numpy())); gts.append(data.boxes[t])
    model.train()
    return average_precision(preds, gts, 0.5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--head", choices=["fcos", "centernet", "fpn"], required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); R = RECIPE
    run = os.path.join(HERE, "runs", a.tag); os.makedirs(run, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = Data(a.head)
    assert all(os.path.exists(os.path.join(data.dir[t], t + ".npy")) for t in data.tr + data.va), "features incomplete"
    model = build(a.head).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=R["lr"], weight_decay=R["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, R["epochs"])
    st = {"epoch": 0, "best": -1.0, "best_epoch": -1, "no_improve": 0, "evals": []}
    lp = os.path.join(run, "last.pt")
    if os.path.exists(lp):
        ck = torch.load(lp, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); st = ck["st"]
        torch.set_rng_state(ck["torch_rng"])
        print(f"[resume] from epoch {st['epoch']}", flush=True)
    rng = np.random.default_rng(a.seed)
    for _ in range(st["epoch"]):                                  # replay the shuffles of completed epochs
        rng.shuffle(list(data.tr))
    print(f"[{a.tag}] head={a.head} device={device} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M recipe={R}", flush=True)
    stopped = False
    for ep in range(st["epoch"], R["epochs"]):
        model.train(); order = list(data.tr); rng.shuffle(order); t0 = time.time(); L = []
        chunks = [order[i:i + R["bs"]] for i in range(0, len(order), R["bs"])]
        with ThreadPoolExecutor(2) as ex:
            fut = ex.submit(data.batch, chunks[0])
            for k in range(len(chunks)):
                f, tg = fut.result()
                if k + 1 < len(chunks):
                    fut = ex.submit(data.batch, chunks[k + 1])
                f = f.float().to(device); tg = {kk: v.to(device) for kk, v in tg.items()}
                out = model(f)
                if a.head == "fpn":
                    parts = FC.loss_fpn(out, tg)
                elif a.head == "fcos":
                    parts = FC.loss(out, tg)
                else:
                    from boxinst_commonality_tcd_04.detector import det_loss
                    parts = det_loss(out, tg)
                loss = sum(parts)
                opt.zero_grad(); loss.backward(); opt.step()
                L.append([p.item() for p in parts])
        sched.step(); st["epoch"] = ep + 1
        msg = f"  ep{ep+1:3d} loss {np.round(np.mean(L, 0), 4).tolist()} {time.time()-t0:.0f}s lr {opt.param_groups[0]['lr']:.1e}"
        if (ep + 1) % R["eval_every"] == 0 or ep + 1 == R["epochs"]:
            ap50 = evaluate(model, a.head, data, device)
            st["evals"].append({"epoch": ep + 1, "val_boxAP50": round(ap50, 4)})
            msg += f" | val boxAP50 {ap50:.4f}"
            improved = ap50 > st["best"] + R["es_min_delta"]
            if ap50 > st["best"]:
                st["best"], st["best_epoch"] = ap50, ep + 1
                torch.save({"state": model.state_dict(), "cfg": {"head": a.head, "tag": a.tag, "seed": a.seed, "best_epoch": ep + 1,
                            "val_boxAP50": round(ap50, 4), "recipe": R}}, os.path.join(run, "best.pt.tmp"))
                os.replace(os.path.join(run, "best.pt.tmp"), os.path.join(run, "best.pt")); msg += " ^best"
            st["no_improve"] = 0 if improved else st["no_improve"] + 1
            if ep + 1 >= R["min_epochs"] and st["no_improve"] >= R["es_patience"]:
                stopped = True; msg += f" | early-stop (best ep{st['best_epoch']})"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "st": st,
                    "torch_rng": torch.get_rng_state()}, lp + ".tmp"); os.replace(lp + ".tmp", lp)
        print(msg, flush=True)
        if stopped:
            break
    json.dump(st, open(os.path.join(run, "train_summary.json"), "w"), indent=1)
    print(f"[{a.tag}] DONE best ep{st['best_epoch']} val boxAP50 {st['best']:.4f}", flush=True)


if __name__ == "__main__":
    main()
