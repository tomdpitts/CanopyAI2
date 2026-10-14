"""Local (MPS/CPU) twin of selvabox_seamfix_modal.py, written 2026-10-14 when the Modal
workspace was paused. Same SAM 3 FT weights (HF cache), same processor path, same 64-box
batching; image from data/tcd/test/<tid>.tif instead of the HF dataset.

  verify : re-prompt stored boxes and compare to the stored A100 masks (premise for reuse AND
           for running the refill in this environment at all)
  refill : --mode off|on, per-tile JSON checkpoints under --out_dir/tiles, resume by listing
  assemble: --mode off|on -> preds_selvabox_seamfix_{mode}_r2048.json (2048 masks)

  PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.selvabox_seamfix_local \
      verify --tids tcd_val_tile_125,tcd_val_tile_1 --device mps
"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils

from boxinst_commonality_tcd_04.native_raster import selvabox_seam as SB

ROOT = "/Users/tompitts/dphil/CanopyAI2"
SRC_TILES = "/Users/tompitts/dphil/feat_cache/selvabox_bench_r2048_tiles/tiles"  # copy of the volume
TIF = os.path.join(ROOT, "data/tcd/test/{}.tif")
NR = os.path.dirname(os.path.abspath(__file__))
SAM_REPO = "CanopyRS/sam3-multi-selvabox-selvamask-FT"
SAM_FILE = "sam3_selvamask_ft_model_best.pt"
SAM_BASE = "facebook/sam3"
NATIVE, LOW, SAM_TILE, BOX_BATCH = 2048, 512, 1777, 64


def load_sam(device):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    proc = Sam3TrackerProcessor.from_pretrained(SAM_BASE)
    model = Sam3TrackerModel.from_pretrained(SAM_BASE).to(device).eval()
    sd = torch.load(hf_hub_download(SAM_REPO, SAM_FILE), map_location="cpu")
    sd = sd.get("model_state_dict", sd)
    exp = set(model.state_dict().keys()); got = set(sd.keys())
    if len(exp & got) != len(exp) or (got - exp):
        raise RuntimeError("FT SAM 3 state dict mismatch")
    model.load_state_dict(sd, strict=False)
    return model, proc


def load_rgb(tid):
    return Image.open(TIF.format(tid)).convert("RGB")


def enc(m_uint8, res):
    small = m_uint8 if m_uint8.shape[0] == res else \
        np.asarray(Image.fromarray(m_uint8 * 255).resize((res, res), Image.BILINEAR)) >= 128
    r = maskUtils.encode(np.asfortranarray(np.asarray(small, np.uint8)))
    return {"size": [res, res], "counts": r["counts"].decode("ascii")}


def sam_masks(sam, proc, rgb, B, batches, device):
    import torch
    assert rgb.size == (2048, 2048), rgb.size
    sam_img = rgb.resize((SAM_TILE, SAM_TILE), Image.BILINEAR)
    f = SAM_TILE / 2048.0
    inputs = proc(images=sam_img, input_boxes=[(B * f).tolist()], return_tensors="pt")
    with torch.no_grad():
        emb = sam.get_image_embeddings(inputs["pixel_values"].to(device))
    all_boxes = inputs["input_boxes"]; osz = inputs["original_sizes"]
    out = {}
    for idx in batches:
        with torch.no_grad():
            so = sam(input_boxes=all_boxes[:, idx].to(device), image_embeddings=emb,
                     multimask_output=False)
        m = proc.post_process_masks(so.pred_masks.cpu(), osz)[0]
        m = m.squeeze(1).numpy().astype(np.uint8)
        iou = getattr(so, "iou_scores", None)
        ssc = np.asarray(iou.cpu()).reshape(len(m), -1)[:, 0].round(4).tolist() if iou is not None \
            else [None] * len(m)
        for j, mm, sc in zip(idx, m, ssc):
            assert mm.shape == (SAM_TILE, SAM_TILE), mm.shape
            out[int(j)] = {"m1777": enc(mm, SAM_TILE), "m2048": enc(mm, NATIVE),
                           "m512": enc(mm, LOW), "sam": sc}
    return out


def _dec(r):
    return maskUtils.decode({"size": r["size"], "counts": r["counts"].encode()})


def verify(a):
    out_fp = os.path.join(NR, "seamfix", f"selvabox_verify_reuse_local_{a.device}.json")
    assert not os.path.exists(out_fp), f"REFUSING to overwrite {out_fp}"
    sam, proc = load_sam(a.device)
    res = {"device": a.device, "n_boxes": a.n_boxes, "tiles": {}}
    for tid in a.tids.split(","):
        r = json.load(open(os.path.join(SRC_TILES, tid + ".json")))
        n = min(a.n_boxes, len(r["scores"]))
        B = np.asarray(r["boxes_2048"][:n], np.float64)
        rgb = load_rgb(tid)
        same = [list(range(i, min(i + BOX_BATCH, n))) for i in range(0, n, BOX_BATCH)]
        perm = np.random.RandomState(0).permutation(n).tolist()
        shuf = [perm[i:i + 37] for i in range(0, n, 37)]
        tile = {}
        outs = {}
        for name, batches in (("same_batching", same), ("shuffled_37", shuf)):
            t0 = time.time()
            o = sam_masks(sam, proc, rgb, B, batches, a.device)
            outs[name] = o
            eq, px, ious = 0, [], []
            for j in range(n):
                st = r["masks_rle_1777"][j]
                if o[j]["m1777"]["counts"] == st["counts"]:
                    eq += 1; ious.append(1.0)
                else:
                    x, y = _dec(st).astype(bool), _dec(o[j]["m1777"]).astype(bool)
                    px.append(int((x != y).sum()))
                    ious.append(float((x & y).sum() / max((x | y).sum(), 1)))
            sd = [abs(o[j]["sam"] - r["sam_scores"][j]) for j in range(n)] if r["sam_scores"] else []
            tile[name] = {"identical": eq, "of": n, "median_px_diff": float(np.median(px)) if px else 0,
                          "max_px_diff": max(px) if px else 0, "mean_iou_vs_stored": float(np.mean(ious)),
                          "min_iou_vs_stored": float(np.min(ious)),
                          "frac_iou_ge_0.99": float(np.mean(np.array(ious) >= 0.99)),
                          "max_sam_score_absdiff": float(max(sd)) if sd else None,
                          "sec": round(time.time() - t0, 1)}
        tile["same_vs_shuffled_identical"] = sum(
            outs["same_batching"][j]["m1777"]["counts"] == outs["shuffled_37"][j]["m1777"]["counts"]
            for j in range(n))
        res["tiles"][tid] = tile
        print(tid, json.dumps(tile, indent=1), flush=True)
    json.dump(res, open(out_fp, "w"), indent=1)
    print("->", out_fp)


def refill(a):
    out_dir = a.out_dir or os.path.join(NR, "seamfix", f"selvabox_seamfix_{a.mode}_tiles_{a.device}")
    tiles_dir = os.path.join(out_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)
    edge = SB.EDGE_FRAC if a.mode == "on" else None
    tids = sorted(json.load(open(os.path.join(ROOT, "boxinst_commonality_tcd_04/test_gt.json"))))
    done = {f[:-5] for f in os.listdir(tiles_dir) if f.endswith(".json")}
    todo = [t for t in tids if t not in done]
    print(f"[seamfix-{a.mode}] {len(tids)} tiles, {len(done)} done, {len(todo)} to do "
          f"edge={edge} device={a.device} -> {tiles_dir}", flush=True)
    sam = proc = None
    tot_new = tot_reuse = 0
    t0 = time.time()
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
                sam, proc = load_sam(a.device)
            o = sam_masks(sam, proc, load_rgb(tid), B[new_ix],
                          [list(range(i, min(i + BOX_BATCH, len(new_ix))))
                           for i in range(0, len(new_ix), BOX_BATCH)], a.device)
            fresh = {new_ix[j]: o[j] for j in range(len(new_ix))}
        rec = {"boxes_2048": np.round(B, 2).tolist(), "scores": np.round(S, 4).tolist(),
               "sam_scores": [], "masks_rle_1777": [], "masks_rle_2048": [], "masks_rle_512": [],
               "reused": [], "n_edge_dropped": nd, "mode": a.mode, "device": a.device}
        for j, s_ in enumerate(src):
            if s_ is not None:
                rec["sam_scores"].append(r["sam_scores"][s_] if r["sam_scores"] else None)
                for k in ("masks_rle_1777", "masks_rle_2048", "masks_rle_512"):
                    rec[k].append(r[k][s_])
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
        if (n + 1) % 5 == 0 or n + 1 == len(todo):
            el = time.time() - t0
            print(f"  {n+1}/{len(todo)}  reused={tot_reuse} prompted={tot_new}  {el:.0f}s  "
                  f"eta {el/(n+1)*(len(todo)-n-1):.0f}s", flush=True)
    print(f"[seamfix-{a.mode}] refill done", flush=True)


def assemble(a):
    out_dir = a.out_dir or os.path.join(NR, "seamfix", f"selvabox_seamfix_{a.mode}_tiles_{a.device}")
    tiles_dir = os.path.join(out_dir, "tiles")
    tids = sorted(json.load(open(os.path.join(ROOT, "boxinst_commonality_tcd_04/test_gt.json"))))
    missing = [t for t in tids if not os.path.exists(os.path.join(tiles_dir, t + ".json"))]
    assert not missing, f"{len(missing)} tiles missing"
    fp = os.path.join(NR, "seamfix", f"preds_selvabox_seamfix_{a.mode}_r{NATIVE}.json")
    assert not os.path.exists(fp), f"REFUSING to overwrite {fp}"
    preds, tot, nre, nd = {}, 0, 0, 0
    for t in tids:
        r = json.load(open(os.path.join(tiles_dir, t + ".json")))
        preds[t] = {"boxes_2048": r["boxes_2048"], "scores": r["scores"],
                    "sam_scores": r["sam_scores"], "masks_rle": r["masks_rle_2048"]}
        tot += len(r["scores"]); nre += sum(r["reused"]); nd += r["n_edge_dropped"]
    meta = {"model": "SelvaBox DINO-Swin-L (NQOS-selvamask-FT) -> SAM 3 (selvamask-FT), 1024px @0.5 "
                     "ovl @0.1 m/px, IoU-NMS 0.7, " +
                     ("CanopyRS edge band 0.05 at INTERIOR subtile edges (seam fix)"
                      if a.mode == "on" else "no edge band (seam-fix GATE replay)"),
            "mode": a.mode, "mask_res": NATIVE, "n_tiles": len(tids), "n_crowns": tot,
            "n_reused_A100_masks": nre, "n_prompted_locally": tot - nre, "device": a.device,
            "n_raw_edge_dropped": nd, "score_floor": 0.05, "keep_dets": 1600, "sam_tile": SAM_TILE,
            "edge_rule": "geodataset _filter_edge_polygons, 5% of 1024 px, interior subtile edges "
                         "only, before floor/NMS",
            "selection": "selvabox_seam.select(raw_dets, ..., stored_keys) oracle tie-break",
            "source_tiles": "tcd04-baselines-vol:out/native_raster/selvabox_bench_r2048/tiles "
                            "(local copy " + SRC_TILES + ")"}
    json.dump({"meta": meta, "preds": preds}, open(fp, "w"))
    print(f"-> {fp}: {len(preds)} tiles, {tot} crowns ({nre} reused, {tot-nre} prompted)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify", "refill", "assemble"])
    ap.add_argument("--tids", default="tcd_val_tile_125,tcd_val_tile_1")
    ap.add_argument("--n_boxes", type=int, default=256)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--mode", default="on", choices=["on", "off"])
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    {"verify": verify, "refill": refill, "assemble": assemble}[a.cmd](a)


if __name__ == "__main__":
    main()
