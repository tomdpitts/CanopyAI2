"""SelvaBox -> SAM 3 seam fix (2026-10-14): CanopyRS's released edge band (geodataset
`_filter_edge_polygons`, 5% of the 1024 subtile = 51.2 px) applied at INTERIOR subtile edges
only, before floor/NMS, replayed from the persisted `raw_dets` of
`selvabox_native_modal.py` (tiles on tcd04-baselines-vol:out/native_raster/selvabox_bench_r2048/,
READ ONLY here). Selection logic: `selvabox_seam.py` (shipped into the image).

SAM 3 masks are reused for every kept box the original run already prompted (same key: 2-dp
box, 4-dp score); only boxes that were NOT prompted before are sent to SAM 3, on the same
1777 px image, same processor, same 64-box batching, multimask_output=False. `verify_reuse`
tests the premise first: re-prompting stored boxes (same batching, then a different batch
composition) must reproduce the stored masks.

  mode "off": edge band OFF (oracle-tie-broken replay) -> AP-level gate vs published row
  mode "on" : edge band ON at interior edges            -> the fixed row

Outputs (NEW dirs, verified absent at launch): out/native_raster/selvabox_seamfix_r2048_{on,off}/
tiles/<tid>.json, run.log, assembled preds_*.json. Per-tile atomic writes, vol.commit() every
CKPT_EVERY tiles, resume by listing. Launch ONLY via detach.py:

    python detach.py deploy selvabox_seamfix_modal.py
    python detach.py spawn tcd04-selvabox-seamfix verify_reuse --tids tcd_val_tile_125,tcd_val_tile_1
    python detach.py spawn tcd04-selvabox-seamfix refill --mode off
    python detach.py spawn tcd04-selvabox-seamfix refill --mode on
    python detach.py spawn tcd04-selvabox-seamfix assemble --mode on
"""
import json
import os

import modal

APP = "tcd04-selvabox-seamfix"
OUT_VOL = "tcd04-baselines-vol"
HFDATA_VOL = "canopyai-deepforest-data"
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)

SAM_REPO = "CanopyRS/sam3-multi-selvabox-selvamask-FT"
SAM_FILE = "sam3_selvamask_ft_model_best.pt"
SAM_BASE = "facebook/sam3"
NATIVE, LOW, SAM_TILE, BOX_BATCH = 2048, 512, 1777, 64
CKPT_EVERY = 5

app = modal.App(APP)
vol = modal.Volume.from_name(OUT_VOL)
hfvol = modal.Volume.from_name(HFDATA_VOL)
hf_secret = modal.Secret.from_name("huggingface")

# image: verbatim from selvabox_native_modal.py (cached build) + the selection module
image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel")
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
    .apt_install("git", "g++", "ninja-build", "libgl1", "libglib2.0-0")
    .env({"TORCH_CUDA_ARCH_LIST": "8.0", "FORCE_CUDA": "1"})
    .pip_install("numpy<2", "opencv-python-headless", "pycocotools", "pillow",
                 "datasets==4.0.0", "huggingface_hub", "timm",
                 "transformers>=5.12", "scipy")
    .run_commands(
        "git clone --recursive https://github.com/IDEA-Research/detrex.git /opt/detrex",
        "pip install --no-build-isolation -e /opt/detrex/detectron2",
        "pip install --no-build-isolation -e /opt/detrex",
    )
    .env({"HF_HOME": "/hfdata/hf_cache", "HF_HUB_OFFLINE": "0",
          "HF_DATASETS_OFFLINE": "1", "DETREX_ROOT": "/opt/detrex"})
    .add_local_file(os.path.join(PKG, "modal_tcd_multiseed", "phase4", "manifest.json"),
                    "/root/manifest.json")
    .add_local_file(os.path.join(PKG, "test_gt.json"), "/root/test_gt.json")
    .add_local_file(os.path.join(HERE, "selvabox_seam.py"), "/root/selvabox_seam.py")
)

SRC_TILES = "/vol/out/native_raster/selvabox_bench_r2048/tiles"


def _out(mode):
    assert mode in ("on", "off"), mode
    d = f"/vol/out/native_raster/selvabox_seamfix_r{NATIVE}_{mode}"
    return d, os.path.join(d, "tiles"), os.path.join(d, "run.log")


def _log(path, msg):
    print(msg, flush=True)
    with open(path, "a") as fh:
        fh.write(msg + "\n")


def _env_versions():
    import numpy, PIL, torch, transformers
    return {"torch": torch.__version__, "numpy": numpy.__version__, "pillow": PIL.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def _load_ft_sam3():
    """Verbatim selvabox_native_modal._load_ft_sam3."""
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    proc = Sam3TrackerProcessor.from_pretrained(SAM_BASE)
    model = Sam3TrackerModel.from_pretrained(SAM_BASE).to("cuda").eval()
    sd = torch.load(hf_hub_download(SAM_REPO, SAM_FILE), map_location="cpu")
    sd = sd.get("model_state_dict", sd)
    exp = set(model.state_dict().keys()); got = set(sd.keys())
    matched = len(exp & got)
    if matched != len(exp) or (got - exp):
        raise RuntimeError(f"FT SAM 3 state dict mismatch: matched {matched}/{len(exp)}, "
                           f"{len(got - exp)} unexpected")
    model.load_state_dict(sd, strict=False)
    return model, proc


def _dataset():
    from datasets import load_dataset
    manifest = json.load(open("/root/manifest.json"))["feat_test"]
    ds = load_dataset("restor/tcd", split="test")
    idx = {int(i): k for k, i in enumerate(ds["image_id"])}
    return ds, idx, manifest


def _sam_masks(sam, proc, rgb, B, batches):
    """Same path as selvabox_native_modal.predict. B: (n,4) whole-tile 2048 boxes.
    batches: list of index lists into B (the prompt grouping). Returns per-box dict."""
    import numpy as np
    import torch
    from PIL import Image
    from pycocotools import mask as maskUtils

    def enc(m_uint8, res):
        small = m_uint8 if m_uint8.shape[0] == res else \
            np.asarray(Image.fromarray(m_uint8 * 255).resize((res, res), Image.BILINEAR)) >= 128
        r = maskUtils.encode(np.asfortranarray(np.asarray(small, np.uint8)))
        return {"size": [res, res], "counts": r["counts"].decode("ascii")}

    sam_img = rgb.resize((SAM_TILE, SAM_TILE), Image.BILINEAR)
    f = SAM_TILE / 2048.0
    inputs = proc(images=sam_img, input_boxes=[(B * f).tolist()], return_tensors="pt")
    with torch.no_grad():
        emb = sam.get_image_embeddings(inputs["pixel_values"].to("cuda"))
    all_boxes = inputs["input_boxes"]; osz = inputs["original_sizes"]
    out = {}
    for idx in batches:
        with torch.no_grad():
            so = sam(input_boxes=all_boxes[:, idx].to("cuda"), image_embeddings=emb,
                     multimask_output=False)
        m = proc.post_process_masks(so.pred_masks.cpu(), osz)[0]
        m = m.squeeze(1).numpy().astype(np.uint8)
        iou = getattr(so, "iou_scores", None)
        ssc = np.asarray(iou.cpu()).reshape(len(m), -1)[:, 0].round(4).tolist() if iou is not None \
            else [None] * len(m)
        for j, mm, sc in zip(idx, m, ssc):
            assert mm.shape == (SAM_TILE, SAM_TILE), mm.shape
            out[int(j)] = {"m1777": enc(mm, SAM_TILE), "m2048": enc(mm, NATIVE),
                           "m512": enc(mm, LOW), "sam": sc, "px": int(mm.sum())}
    return out


@app.function(gpu="A100", image=image, volumes={"/vol": vol, "/hfdata": hfvol},
              timeout=3600, cpu=8, memory=65536, secrets=[hf_secret])
def verify_reuse(tids: str = "tcd_val_tile_125,tcd_val_tile_1", n_boxes: int = 256):
    """Re-prompt the first n stored boxes of each tile (a) with the original 64-box batching and
    (b) shuffled into 37-box batches; compare to the stored masks_rle_1777 exactly."""
    import numpy as np
    from pycocotools import mask as maskUtils
    out_fp = "/vol/out/native_raster/selvabox_seamfix_verify_reuse.json"
    assert not os.path.exists(out_fp), f"REFUSING to overwrite {out_fp}"
    sam, proc = _load_ft_sam3()
    ds, idx, manifest = _dataset()
    res = {"env": _env_versions(), "n_boxes": n_boxes, "tiles": {}}
    for tid in tids.split(","):
        r = json.load(open(os.path.join(SRC_TILES, tid + ".json")))
        n = min(n_boxes, len(r["scores"]))
        B = np.asarray(r["boxes_2048"][:n], np.float64)
        rgb = ds[idx[int(manifest[tid]["image_id"])]]["image"].convert("RGB")
        same = [list(range(i, min(i + BOX_BATCH, n))) for i in range(0, n, BOX_BATCH)]
        perm = np.random.RandomState(0).permutation(n).tolist()
        shuf = [perm[i:i + 37] for i in range(0, n, 37)]
        tile = {}
        for name, batches in (("same_batching", same), ("shuffled_37", shuf)):
            o = _sam_masks(sam, proc, rgb, B, batches)
            eq, pxdiff = 0, []
            for j in range(n):
                st = r["masks_rle_1777"][j]
                if o[j]["m1777"]["counts"] == st["counts"]:
                    eq += 1
                else:
                    a = maskUtils.decode({"size": st["size"], "counts": st["counts"].encode()})
                    b = maskUtils.decode({"size": o[j]["m1777"]["size"],
                                          "counts": o[j]["m1777"]["counts"].encode()})
                    pxdiff.append(int((a != b).sum()))
            tile[name] = {"identical": eq, "of": n, "n_diff": len(pxdiff),
                          "max_px_diff": max(pxdiff) if pxdiff else 0,
                          "sam_score_eq": sum(o[j]["sam"] == r["sam_scores"][j] for j in range(n))}
        res["tiles"][tid] = tile
        print(tid, json.dumps(tile), flush=True)
    json.dump(res, open(out_fp, "w"), indent=1)
    vol.commit()
    return res


@app.function(gpu="A100", image=image, volumes={"/vol": vol, "/hfdata": hfvol},
              timeout=8 * 3600, cpu=8, memory=65536, secrets=[hf_secret])
def refill(mode: str = "on", limit: int = 0):
    import sys
    import numpy as np
    sys.path.insert(0, "/root")
    import selvabox_seam as SB
    out_dir, tiles_dir, log = _out(mode)
    os.makedirs(tiles_dir, exist_ok=True)
    edge = SB.EDGE_FRAC if mode == "on" else None
    _log(log, f"[seamfix-{mode}] start env={json.dumps(_env_versions())} edge_frac={edge} "
              f"src={SRC_TILES} -> {out_dir}")
    gt = json.load(open("/root/test_gt.json"))
    manifest = json.load(open("/root/manifest.json"))["feat_test"]
    tids = [t for t in sorted(gt) if t in manifest]
    if limit:
        tids = tids[:limit]
    done = {f[:-5] for f in os.listdir(tiles_dir) if f.endswith(".json")}
    todo = [t for t in tids if t not in done]
    _log(log, f"[seamfix-{mode}] {len(tids)} tiles, {len(done)} done, {len(todo)} to do")
    sam = proc = ds = None
    tot_new = tot_reuse = 0
    for n, tid in enumerate(todo):
        r = json.load(open(os.path.join(SRC_TILES, tid + ".json")))
        old = {SB.key(b, s): i for i, (b, s) in enumerate(zip(r["boxes_2048"], r["scores"]))}
        B, S, nd = SB.select(r["raw_dets"], edge, set(old))
        keys = [SB.key(b, s) for b, s in zip(B, S)]
        src = [old.get(k) for k in keys]
        new_ix = [j for j, s_ in enumerate(src) if s_ is None]
        fresh = {}
        if new_ix:
            if sam is None:
                sam, proc = _load_ft_sam3()
                ds, idx, _ = _dataset()
            rgb = ds[idx[int(manifest[tid]["image_id"])]]["image"].convert("RGB")
            Bn = B[new_ix]
            o = _sam_masks(sam, proc, rgb, Bn,
                           [list(range(i, min(i + BOX_BATCH, len(new_ix))))
                            for i in range(0, len(new_ix), BOX_BATCH)])
            fresh = {new_ix[j]: o[j] for j in range(len(new_ix))}
        rec = {"boxes_2048": np.round(B, 2).tolist(), "scores": np.round(S, 4).tolist(),
               "sam_scores": [], "masks_rle_1777": [], "masks_rle_2048": [],
               "masks_rle_512": [], "reused": [], "n_edge_dropped": nd, "mode": mode}
        for j, s_ in enumerate(src):
            if s_ is not None:
                rec["sam_scores"].append(r["sam_scores"][s_] if r["sam_scores"] else None)
                rec["masks_rle_1777"].append(r["masks_rle_1777"][s_])
                rec["masks_rle_2048"].append(r["masks_rle_2048"][s_])
                rec["masks_rle_512"].append(r["masks_rle_512"][s_])
                rec["reused"].append(True)
            else:
                f_ = fresh[j]
                rec["sam_scores"].append(f_["sam"]); rec["masks_rle_1777"].append(f_["m1777"])
                rec["masks_rle_2048"].append(f_["m2048"]); rec["masks_rle_512"].append(f_["m512"])
                rec["reused"].append(False)
        tot_new += len(new_ix); tot_reuse += len(src) - len(new_ix)
        tmp = os.path.join(tiles_dir, tid + ".json.tmp")
        json.dump(rec, open(tmp, "w"))
        os.replace(tmp, os.path.join(tiles_dir, tid + ".json"))
        if (n + 1) % CKPT_EVERY == 0 or n + 1 == len(todo):
            vol.commit()
            _log(log, f"  {n+1}/{len(todo)} this run ({len(done)+n+1}/{len(tids)} total)  "
                      f"reused={tot_reuse} prompted={tot_new}")
    vol.commit()
    _log(log, f"[seamfix-{mode}] refill done")
    return {"mode": mode, "tiles": len(tids), "this_run": len(todo),
            "reused": tot_reuse, "prompted": tot_new}


@app.function(image=image, volumes={"/vol": vol}, timeout=3600, cpu=2, memory=32768)
def assemble(mode: str = "on"):
    out_dir, tiles_dir, log = _out(mode)
    gt = json.load(open("/root/test_gt.json"))
    manifest = json.load(open("/root/manifest.json"))["feat_test"]
    tids = [t for t in sorted(gt) if t in manifest]
    missing = [t for t in tids if not os.path.exists(os.path.join(tiles_dir, t + ".json"))]
    assert not missing, f"{len(missing)} tiles missing, e.g. {missing[:3]}"
    base = {"model": "SelvaBox DINO-Swin-L (NQOS-selvamask-FT) -> SAM 3 (selvamask-FT), 1024px "
                     "@0.5 ovl @0.1 m/px, IoU-NMS 0.7, " +
                     ("CanopyRS edge band 0.05 at INTERIOR subtile edges (seam fix)" if mode == "on"
                      else "no edge band (seam-fix GATE replay)"),
            "mode": mode, "edge_rule": "geodataset _filter_edge_polygons, 5% of 1024 px, "
                                       "interior subtile edges only, before floor/NMS",
            "selection": "selvabox_seam.select(raw_dets, ..., stored_keys) oracle tie-break",
            "source_tiles": SRC_TILES, "tiles_dir": tiles_dir, "n_tiles": len(tids),
            "score_floor": 0.05, "keep_dets": 1600, "sam_tile": SAM_TILE}
    for key, res, name in (("masks_rle_2048", NATIVE, f"preds_selvabox_seamfix_{mode}_r{NATIVE}.json"),
                           ("masks_rle_1777", SAM_TILE, f"preds_selvabox_seamfix_{mode}_r{SAM_TILE}_native.json"),
                           ("masks_rle_512", LOW, f"preds_selvabox_seamfix_{mode}_r{LOW}.json")):
        fp = os.path.join(out_dir, name)
        assert not os.path.exists(fp), f"REFUSING to overwrite {fp}"
        preds, tot, nre, nd = {}, 0, 0, 0
        for t in tids:
            r = json.load(open(os.path.join(tiles_dir, t + ".json")))
            preds[t] = {"boxes_2048": r["boxes_2048"], "scores": r["scores"],
                        "sam_scores": r["sam_scores"], "masks_rle": r[key]}
            tot += len(r["scores"]); nre += sum(r["reused"]); nd += r["n_edge_dropped"]
        json.dump({"meta": dict(base, mask_res=res, n_crowns=tot, n_reused=nre,
                                n_prompted=tot - nre, n_raw_edge_dropped=nd), "preds": preds},
                  open(fp, "w"))
        _log(log, f"[seamfix-{mode}] wrote {fp}: {len(preds)} tiles, {tot} crowns "
                  f"({nre} reused, {tot-nre} prompted)")
    vol.commit()
    return {"out_dir": out_dir}
