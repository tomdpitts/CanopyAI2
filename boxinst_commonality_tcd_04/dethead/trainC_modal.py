"""Modal A100: train / predict arm C (pre-registration 4, dethead/README.md). Detector only; the LACE masker and all
scoring for selection and test run LOCALLY on the boxes this writes (published masker inputs).

Inputs on tcd04-phase4-vol: feat_4p_train|test (1024,256,256) fp16, z4p_train|test (512,512,128) fp16; RGB = HF restor/tcd
joined by manifest image_id (size asserted; sha1 where the manifest has one). Targets: train_tiles_gt.json (train partition).
Early-stop metric: val box AP at IoU 0.6, crowd arm (score_lite GT from val_gt.json, canopy = iscrowd), maxDets 600,
floor 0.05, decode NMS 0.6 (the FCOS default used for every head's training eval).
Everything under /vol/dethead/: runs/<tag>/{last.pt,best.pt,train.log,train_summary.json}, preds/<tag>_<split>_nms<x>.json.
Checkpoint + volume commit every epoch; resume from last.pt (shuffle order replayed, as train.py).

  .venv/bin/python boxinst_commonality_tcd_04/native_raster/detach.py deploy <abs path to this file>
  ... detach.py spawn tcd04-dethead-armc train --tag fpnC_s0
  ... detach.py spawn tcd04-dethead-armc predict --tag fpnC_s0 --split val
"""
import json
import os

import modal

APP_NAME = "tcd04-dethead-armc"
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REPO = os.path.dirname(PKG)
PH4 = os.path.join(PKG, "modal_tcd_multiseed", "phase4")
STUBS = os.path.join(PH4, "stubs")

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("tcd04-phase4-vol")
hfvol = modal.Volume.from_name("canopyai-deepforest-data")
hf_secret = modal.Secret.from_name("huggingface")

P = "/root/proj"
PKG_R = f"{P}/boxinst_commonality_tcd_04"
PH4_R = f"{PKG_R}/modal_tcd_multiseed/phase4"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.1", "torchvision==0.27.1", "numpy==2.2.6",
                 "transformers==4.57.1", "datasets==4.0.0", "pillow", "contourpy", "pycocotools")
    .env({"HF_HOME": "/hfdata/hf_cache", "HF_HUB_OFFLINE": "0", "HF_DATASETS_OFFLINE": "1"})
)
for rel in ("dapt/__init__.py", "dapt/backbone.py", "dapt/targets.py", "dapt/decode.py", "dapt/eval.py", "dapt/head.py"):
    image = image.add_local_file(os.path.join(REPO, rel), f"{P}/{rel}")
for rel in ("__init__.py", "detector.py", "evaluate.py", "em.py", "prepare_test.py", "cache_test.py", "cache_train_tiles.py",
            "train_detector_tiles.py", "train_tiles_gt.json", "dethead/__init__.py", "dethead/fcos.py",
            "box_supervised_baselines/__init__.py", "box_supervised_baselines/tierb/__init__.py",
            "box_supervised_baselines/tierb/score_lite.py"):
    image = image.add_local_file(os.path.join(PKG, rel), f"{PKG_R}/{rel}")
image = image.add_local_file(os.path.join(PKG, "modal_tcd_multiseed", "__init__.py"), f"{PKG_R}/modal_tcd_multiseed/__init__.py")
for rel in ("__init__.py", "manifest.json", "val_gt.json"):
    image = image.add_local_file(os.path.join(PH4, rel), f"{PH4_R}/{rel}")
for pkg, files in {"boxinst": ("__init__.py", "cache_feats.py"), "boxinst_commonality": ("__init__.py", "em.py"),
                   "boxinst_tcd": ("__init__.py", "build_canopy.py", "cache.py", "prepare.py")}.items():
    for f in files:
        image = image.add_local_file(os.path.join(STUBS, pkg, f), f"{P}/{pkg}/{f}")

VOL = "/vol"
ROOT = f"{VOL}/dethead"
FEAT = {"train": (f"{VOL}/feat_4p_train", f"{VOL}/z4p_train", "feat_traintile", "train"),
        "test": (f"{VOL}/feat_4p_test", f"{VOL}/z4p_test", "feat_test", "test")}
RECIPE = dict(lr=1e-3, wd=1e-4, epochs=40, bs=3, eval_every=5, min_epochs=12, es_patience=2, es_min_delta=0.005,
              es_metric="crowd box AP@0.6, NMS 0.6, floor 0.05, maxDets 600", seed=0)


def _path():
    import sys
    if P not in sys.path:
        sys.path.insert(0, P)


class Tiles:
    """Per-tile inputs: 8 px feats (1024,256,256) fp16, 4 px feats (128,512,512) fp16, RGB (3,2048,2048) uint8."""

    def __init__(self, split):
        self.f8, self.z4, mkey, self.hfsplit = FEAT[split]
        self.man = json.load(open(f"{PH4_R}/manifest.json"))[mkey]
        self.ds = None

    def _hf(self):
        if self.ds is None:                                   # lazily, once per DataLoader worker
            from datasets import load_dataset
            self.ds = load_dataset("restor/tcd", split=self.hfsplit)
            self.row = {int(i): k for k, i in enumerate(self.ds["image_id"])}
        return self.ds

    def get(self, t):
        import hashlib
        import numpy as np
        import torch
        f8 = torch.from_numpy(np.load(os.path.join(self.f8, t + ".npy")))
        z4 = torch.from_numpy(np.load(os.path.join(self.z4, t + ".npy"))).permute(2, 0, 1).contiguous()
        ds = self._hf(); img = ds[self.row[self.man[t]["image_id"]]]["image"].convert("RGB")
        assert (img.width, img.height) == (self.man[t]["width"], self.man[t]["height"]) == (2048, 2048), t
        a = np.asarray(img)
        if "rgb_sha1" in self.man[t]:
            assert hashlib.sha1(a.tobytes()).hexdigest() == self.man[t]["rgb_sha1"], f"{t}: RGB sha1"
        return f8, z4, torch.from_numpy(a.copy()).permute(2, 0, 1).contiguous()


def _to_dev(f8, z4, rgb, dev):
    return f8.to(dev, non_blocking=True).float(), z4.to(dev, non_blocking=True).float(), rgb.to(dev, non_blocking=True).float() / 255.0


def _val_gt():
    """Crowd-arm val GT (score_lite.build_gt_crowd), cached on the volume (rasterising 108 tiles takes minutes)."""
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    fp = f"{ROOT}/val_gt_crowd_2048.json"
    if os.path.exists(fp):
        d = json.load(open(fp)); return SL.gt_from_json(d["gt"]), {k: int(v) for k, v in d["t2i"].items()}
    vg = json.load(open(f"{PH4_R}/val_gt.json")); gd, t2i = SL.build_gt_crowd(vg, 2048)
    json.dump({"gt": SL.gt_to_json(gd), "t2i": t2i}, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    return gd, t2i


def _box_ap(preds, gd, t2i, iou=0.6):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    dets = SL.build_dt(preds, t2i, 2048, 0.05, with_masks=False)
    if not dets:
        return 0.0
    cg = COCO(); cg.dataset = gd; cg.createIndex(); cd = cg.loadRes([dict(d) for d in dets])
    ev = COCOeval(cg, cd, "bbox"); ev.params.iouThrs = np.array([iou]); ev.params.recThrs = SL.REC_THRS
    ev.params.maxDets = [600]; ev.params.areaRng = [[0, 1e10]]; ev.params.areaRngLbl = ["all"]
    ev.evaluate(); ev.accumulate()
    p = ev.eval["precision"][0, :, 0, 0, 0]
    return float(p[p > -1].mean()) if (p > -1).any() else 0.0


@app.function(gpu="A100", image=image, volumes={VOL: vol, "/hfdata": hfvol}, timeout=12 * 3600, cpu=8, memory=98304,
              secrets=[hf_secret], retries=modal.Retries(max_retries=3, initial_delay=30.0))
def train(tag: str = "fpnC_s0", smoke: bool = False, head: str = "C"):
    """smoke=True: 6 train / 6 val tiles, 1 epoch, tag <tag>_smoke (pipeline + memory + speed check; no number used)."""
    import random, time
    import numpy as np
    import torch
    _path()
    from boxinst_commonality_tcd_04.detector import canopy_cell_mask
    from boxinst_commonality_tcd_04.train_detector_tiles import canopy_px
    from boxinst_commonality_tcd_04.dethead import fcos as FC
    R = dict(RECIPE, epochs=1) if smoke else RECIPE; L = FC.LEVELS_C
    tag = tag + "_smoke" if smoke else tag
    run = f"{ROOT}/runs/{tag}"; os.makedirs(run, exist_ok=True)
    logp = f"{run}/train.log"

    def log(m):
        line = f"{time.strftime('%F %T')} {m}"; print(line, flush=True)
        with open(logp, "a") as f:
            f.write(line + "\n")

    random.seed(R["seed"]); np.random.seed(R["seed"]); torch.manual_seed(R["seed"])
    dev = torch.device("cuda")
    gt = json.load(open(f"{PKG_R}/train_tiles_gt.json"))
    tr = sorted(t for t, v in gt.items() if v["partition"] == "train"); va = sorted(t for t, v in gt.items() if v["partition"] == "val")
    assert (len(tr), len(va)) == (792, 108), (len(tr), len(va))
    if smoke:
        tr, va = tr[:6], va[:6]
    t0 = time.time(); tgt, nfb = {}, 0
    for t in tr:
        levels, n = FC.encode_fpn(np.asarray(gt[t]["boxes"], np.float32).reshape(-1, 4), L); nfb += n
        cpx = canopy_px(gt[t].get("canopy", []), res=2048); rec = {}
        for l, (pos, ltrb, ctr) in enumerate(levels):
            rec |= {f"pos{l}": torch.from_numpy(pos), f"ltrb{l}": torch.from_numpy(ltrb), f"ctr{l}": torch.from_numpy(ctr),
                    f"ignore{l}": torch.from_numpy(canopy_cell_mask(cpx, L[l][1]))}
        tgt[t] = rec
    log(f"[{tag}] targets {time.time()-t0:.0f}s; {len(tr)} train / {len(va)} val; P2 fallback boxes {nfb}")
    tiles = Tiles("train"); tiles._hf()                     # load once in the parent; forked workers share it
    for d in (tiles.f8, tiles.z4):
        miss = [t for t in tr + va if not os.path.exists(os.path.join(d, t + ".npy"))]
        assert not miss, f"{d}: {len(miss)} missing, e.g. {miss[:3]}"
    gd, t2i = _val_gt()

    class DS(torch.utils.data.Dataset):
        def __init__(self, chunks): self.chunks = chunks
        def __len__(self): return len(self.chunks)
        def __getitem__(self, i):
            ts = self.chunks[i]; xs = [tiles.get(t) for t in ts]
            return (torch.stack([x[0] for x in xs]), torch.stack([x[1] for x in xs]), torch.stack([x[2] for x in xs]),
                    {k: torch.stack([tgt[t][k] for t in ts]) for k in tgt[ts[0]]})

    class VDS(torch.utils.data.Dataset):
        def __len__(self): return len(va)
        def __getitem__(self, i): return (va[i],) + tiles.get(va[i])

    def evaluate(model):
        model.eval(); preds = {}
        dl = torch.utils.data.DataLoader(VDS(), batch_size=None, num_workers=6)
        with torch.no_grad():
            for t, f8, z4, rgb in dl:
                out = model(*_to_dev(f8[None], z4[None], rgb[None], dev))
                b, s = FC.decode_fpn(out, 0.6, L, qfl=C2)
                preds[t] = {"boxes_2048": b.numpy().astype(float).tolist(), "scores": s.numpy().astype(float).tolist()}
        model.train()
        return _box_ap(preds, gd, t2i, 0.6)

    C2 = head == "C2"                                         # pre-registration 6: QFL, 3-layer P2, EMA
    model = (FC.FCOSC2() if C2 else FC.FCOSC()).to(dev)
    ema = None
    if C2:
        import copy
        ema = copy.deepcopy(model).eval()
        for q in ema.parameters():
            q.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=R["lr"], weight_decay=R["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, R["epochs"])
    st = {"epoch": 0, "best": -1.0, "best_epoch": -1, "no_improve": 0, "evals": [], "ema_n": 0}
    lp = f"{run}/last.pt"
    if os.path.exists(lp):
        ck = torch.load(lp, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); st = ck["st"]
        if C2:
            ema.load_state_dict(ck["ema"])
        torch.set_rng_state(ck["torch_rng"]); log(f"[resume] from epoch {st['epoch']}")
    if st.get("done"):
        log(f"[{tag}] already done"); return st
    rng = np.random.default_rng(R["seed"])
    for _ in range(st["epoch"]):
        rng.shuffle(list(tr))
    log(f"[{tag}] arm {head} gpu={torch.cuda.get_device_name(0)} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M recipe={R}")
    stopped = False; ep0 = st["epoch"]
    for ep in range(st["epoch"], R["epochs"]):
        order = list(tr); rng.shuffle(order); te = time.time(); Ls = []
        chunks = [order[i:i + R["bs"]] for i in range(0, len(order), R["bs"])]
        dl = torch.utils.data.DataLoader(DS(chunks), batch_size=None, shuffle=False, num_workers=6, pin_memory=True, prefetch_factor=2)
        model.train()
        for k, (f8, z4, rgb, tg) in enumerate(dl):
            x = _to_dev(f8, z4, rgb, dev); tg = {kk: v.to(dev, non_blocking=True) for kk, v in tg.items()}
            parts = (FC.loss_qfl(model(*x), tg, L) if C2 else FC.loss_fpn(model(*x), tg, L)); loss = sum(parts)
            opt.zero_grad(); loss.backward(); opt.step(); Ls.append([p.item() for p in parts])
            if C2:
                n = st["ema_n"]; d = min(0.999, (1 + n) / (10 + n)); st["ema_n"] = n + 1
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            if ep == ep0 and k in (0, 9, 49):                      # first epoch of this container only
                log(f"  ep{ep+1} it{k+1}/{len(chunks)} loss {np.round(Ls[-1], 4).tolist()} {time.time()-te:.0f}s "
                    f"mem {torch.cuda.max_memory_allocated()/2**30:.1f} GB")
        sched.step(); st["epoch"] = ep + 1
        msg = f"  ep{ep+1:3d} loss {np.round(np.mean(Ls, 0), 4).tolist()} {time.time()-te:.0f}s lr {opt.param_groups[0]['lr']:.1e}"
        if (ep + 1) % R["eval_every"] == 0 or ep + 1 == R["epochs"]:
            ap = evaluate(ema if C2 else model); st["evals"].append({"epoch": ep + 1, "val_crowd_boxAP60": round(ap, 4)})
            msg += f" | val crowd boxAP@0.6 {ap:.4f}"
            improved = ap > st["best"] + R["es_min_delta"]
            if ap > st["best"]:
                st["best"], st["best_epoch"] = ap, ep + 1
                torch.save({"state": (ema if C2 else model).state_dict(), "cfg": {"head": "fpnC2" if C2 else "fpnC", "tag": tag, "ema": C2, "best_epoch": ep + 1,
                            "val_crowd_boxAP60": round(ap, 4), "recipe": R}}, f"{run}/best.pt.tmp")
                os.replace(f"{run}/best.pt.tmp", f"{run}/best.pt"); msg += " ^best"
            st["no_improve"] = 0 if improved else st["no_improve"] + 1
            if ep + 1 >= R["min_epochs"] and st["no_improve"] >= R["es_patience"]:
                stopped = True; msg += f" | early-stop (best ep{st['best_epoch']})"
        st["done"] = stopped or ep + 1 == R["epochs"]
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "st": st,
                    "torch_rng": torch.get_rng_state()} | ({"ema": ema.state_dict()} if C2 else {}), lp + ".tmp"); os.replace(lp + ".tmp", lp)
        log(msg); vol.commit()
        if stopped:
            break
    json.dump(st, open(f"{run}/train_summary.json", "w"), indent=1)
    log(f"[{tag}] DONE best ep{st['best_epoch']} val crowd boxAP@0.6 {st['best']:.4f}"); vol.commit()
    return st


@app.function(gpu="A100", image=image, volumes={VOL: vol, "/hfdata": hfvol}, timeout=4 * 3600, cpu=8, memory=65536,
              secrets=[hf_secret], retries=modal.Retries(max_retries=3, initial_delay=30.0))
def predict(tag: str = "fpnC_s0", split: str = "val", nms: str = "0.5,0.6", ckpt: str = "best.pt", with_level: int = 0, head: str = "C"):
    """Raw decode of the given checkpoint on val (108) or test (439) -> preds/<tag>_<split>_nms<x>.json per NMS value.
    Test is run ONCE, after val selection, at the selected NMS only (pre-registration 4)."""
    import time
    import numpy as np
    import torch
    _path()
    from boxinst_commonality_tcd_04.dethead import fcos as FC
    L = FC.LEVELS_C; dev = torch.device("cuda")
    if split == "val":
        gt = json.load(open(f"{PKG_R}/train_tiles_gt.json")); tids = sorted(t for t, v in gt.items() if v["partition"] == "val")
        tiles = Tiles("train"); assert len(tids) == 108
    else:
        tids = sorted(json.load(open(f"{PH4_R}/manifest.json"))["feat_test"]); tiles = Tiles("test"); assert len(tids) == 439
    tiles._hf()
    ck = torch.load(f"{ROOT}/runs/{tag}/{ckpt}", map_location="cpu", weights_only=False)
    C2 = head == "C2"
    model = (FC.FCOSC2() if C2 else FC.FCOSC()).to(dev); model.load_state_dict(ck["state"] if "state" in ck else ck["model"]); model.eval()
    nl = [float(x) for x in str(nms).split(",")]; out = {x: {} for x in nl}           # detach.py passes "0.5" as a float
    os.makedirs(f"{ROOT}/preds", exist_ok=True); t0 = time.time()

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(tids)
        def __getitem__(self, i): return (tids[i],) + tiles.get(tids[i])

    with torch.no_grad():
        for k, (t, f8, z4, rgb) in enumerate(torch.utils.data.DataLoader(DS(), batch_size=None, num_workers=6)):
            o = model(*_to_dev(f8[None], z4[None], rgb[None], dev))
            for x in nl:
                b, s, lv = FC.decode_fpn(o, x, L, return_level=True, qfl=C2)
                out[x][t] = {"boxes_2048": b.numpy().astype(float).tolist(), "scores": s.numpy().astype(float).tolist()}
                if with_level:
                    out[x][t]["levels"] = lv.numpy().astype(int).tolist()            # 0..3 = P2..P5 (pre-registration 5)
            if (k + 1) % 50 == 0:
                print(f"  {k+1}/{len(tids)} {time.time()-t0:.0f}s", flush=True)
    files = []
    for x in nl:
        fp = f"{ROOT}/preds/{tag}_{split}_nms{x}{'_lv' if with_level else ''}.json"
        json.dump({"meta": {"tag": tag, "split": split, "nms": x, "ckpt": ckpt, "cfg": ck.get("cfg")}, "preds": out[x]},
                  open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp); files.append(fp)
    vol.commit()
    return files
