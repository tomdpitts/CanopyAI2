"""Train the ATSS / GFL head on frozen ConvNeXt-L features (pre-registration 3). Same recipe and checkpointing as
train.py. Val: cached features; selection metric = val box mAP50 (dapt), as every LACE detector.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.train_cnx --tag cnxl_s0
"""
import argparse, json, os, random, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from dapt.eval import average_precision
from boxinst_commonality_tcd_04.dethead import cnx as C
from boxinst_commonality_tcd_04.dethead.train import RECIPE, GT, HERE


class Data:
    def __init__(self):
        from boxinst_commonality_tcd_04.detector import canopy_cell_mask
        from boxinst_commonality_tcd_04.train_detector_tiles import canopy_px
        gt = json.load(open(GT))
        self.tr = sorted(t for t, v in gt.items() if v["partition"] == "train")
        self.va = sorted(t for t, v in gt.items() if v["partition"] == "val")
        self.boxes = {t: np.asarray(v["boxes"], np.float32).reshape(-1, 4) for t, v in gt.items()}
        self.tgt = {}; t0 = time.time(); nopos = 0
        for t in self.tr:
            maps = C.encode_atss(self.boxes[t]); cpx = canopy_px(gt[t].get("canopy", []), res=2048)
            got = set()
            for mp in maps:
                got |= set(mp[mp >= 0].tolist())
            nopos += len(self.boxes[t]) - len(got)
            self.tgt[t] = {**{f"idx{l}": torch.from_numpy(mp.astype(np.int64)) for l, mp in enumerate(maps)},
                           **{f"ignore{l}": torch.from_numpy(canopy_cell_mask(cpx, g)) for l, g in enumerate(C.GRIDS)}}
        print(f"[data] {len(self.tr)} train / {len(self.va)} val; ATSS targets {time.time()-t0:.0f}s; boxes without a positive {nopos}", flush=True)

    def batch(self, tids):
        feats = [torch.stack(fs) for fs in zip(*[C.load_feats(t) for t in tids])]
        tg = {k: torch.stack([self.tgt[t][k] for t in tids]) for k in self.tgt[tids[0]]}
        tg["gtb"] = [torch.from_numpy(self.boxes[t]) for t in tids]
        return feats, tg


@torch.no_grad()
def predict(head, feats, device, nms_iou=0.5):
    return C.decode(head([f.float()[None].to(device) for f in feats]), nms_iou)


def evaluate(head, data, device):
    head.eval(); preds, gts = [], []
    with ThreadPoolExecutor(4) as ex:
        for t, f in zip(data.va, ex.map(C.load_feats, data.va)):
            b, s = predict(head, f, device, 0.5)                    # selection decode fixed at NMS 0.5 (LACE's)
            preds.append((b.numpy(), s.numpy())); gts.append(data.boxes[t])
    head.train()
    return average_precision(preds, gts, 0.5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); R = RECIPE
    run = os.path.join(HERE, "runs", a.tag); os.makedirs(run, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = Data()
    assert all(os.path.exists(f"{C.CACHE}/s{k+1}/{t}.npy") for t in data.tr + data.va for k in range(4)), "features incomplete"
    head = C.CNXHead().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=R["lr"], weight_decay=R["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, R["epochs"])
    st = {"epoch": 0, "best": -1.0, "best_epoch": -1, "no_improve": 0, "evals": []}
    lp = os.path.join(run, "last.pt")
    if os.path.exists(lp):
        ck = torch.load(lp, map_location="cpu", weights_only=False)
        head.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); st = ck["st"]
        torch.set_rng_state(ck["torch_rng"]); print(f"[resume] from epoch {st['epoch']}", flush=True)
    rng = np.random.default_rng(a.seed)
    for _ in range(st["epoch"]):
        rng.shuffle(list(data.tr))
    print(f"[{a.tag}] ConvNeXt-L + ATSS/GFL device={device} params={sum(p.numel() for p in head.parameters())/1e6:.2f}M recipe={R}", flush=True)
    stopped = False
    for ep in range(st["epoch"], R["epochs"]):
        order = list(data.tr); rng.shuffle(order); t0 = time.time(); L = []
        chunks = [order[i:i + R["bs"]] for i in range(0, len(order), R["bs"])]
        with ThreadPoolExecutor(2) as ex:
            fut = ex.submit(data.batch, chunks[0])
            for k in range(len(chunks)):
                feats, tg = fut.result()
                if k + 1 < len(chunks):
                    fut = ex.submit(data.batch, chunks[k + 1])
                feats = [f.float().to(device) for f in feats]
                tg = {kk: (v.to(device) if torch.is_tensor(v) else [x.to(device) for x in v]) for kk, v in tg.items()}
                q, g = C.gfl_loss(head(feats), tg); loss = q + g
                opt.zero_grad(); loss.backward(); opt.step(); L.append([q.item(), g.item()])
        sched.step(); st["epoch"] = ep + 1
        msg = f"  ep{ep+1:3d} loss {np.round(np.mean(L, 0), 4).tolist()} {time.time()-t0:.0f}s lr {opt.param_groups[0]['lr']:.1e}"
        if (ep + 1) % R["eval_every"] == 0 or ep + 1 == R["epochs"]:
            ap50 = evaluate(head, data, device); st["evals"].append({"epoch": ep + 1, "val_boxAP50": round(ap50, 4)})
            msg += f" | val boxAP50 {ap50:.4f}"
            improved = ap50 > st["best"] + R["es_min_delta"]
            if ap50 > st["best"]:
                st["best"], st["best_epoch"] = ap50, ep + 1
                torch.save({"state": head.state_dict(), "cfg": {"head": "cnxl_atss_gfl", "tag": a.tag, "seed": a.seed, "best_epoch": ep + 1,
                            "val_boxAP50": round(ap50, 4), "recipe": R, "backbone": C.MODEL}}, os.path.join(run, "best.pt.tmp"))
                os.replace(os.path.join(run, "best.pt.tmp"), os.path.join(run, "best.pt")); msg += " ^best"
            st["no_improve"] = 0 if improved else st["no_improve"] + 1
            if ep + 1 >= R["min_epochs"] and st["no_improve"] >= R["es_patience"]:
                stopped = True; msg += f" | early-stop (best ep{st['best_epoch']})"
        torch.save({"model": head.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "st": st,
                    "torch_rng": torch.get_rng_state()}, lp + ".tmp"); os.replace(lp + ".tmp", lp)
        print(msg, flush=True)
        if stopped:
            break
    json.dump(st, open(os.path.join(run, "train_summary.json"), "w"), indent=1)
    print(f"[{a.tag}] DONE best ep{st['best_epoch']} val boxAP50 {st['best']:.4f}", flush=True)


if __name__ == "__main__":
    main()
