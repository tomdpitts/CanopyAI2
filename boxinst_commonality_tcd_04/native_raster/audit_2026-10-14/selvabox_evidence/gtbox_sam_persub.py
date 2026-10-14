"""Is LACE's masker really better than SAM 3 once SAM gets the released per-subtile placement?
tab:gtbox protocol EXACTLY (gtbox_lib: polygon-extent GT boxes, E.raster GT masks at 2048, 512-ref
validity, per_crown_iou_rle), on EVERY test tile with crowns, local MPS, HF Sam3Tracker:
  ft_whole   FT SAM 3, 2048 tile -> 1777 (tab:gtbox arm b path)
  ft_sub     FT SAM 3, each box on its most-central 1024 subtile -> 1777 (CanopyRS placement)
  base_whole zero-shot facebook/sam3, 2048 tile (tab:gtbox arm a path)
  base_sub   zero-shot, each box on its most-central 1024 subtile (no resize; processor -> 1008)
Crowns wider than any subtile fall back to that model's whole-tile mask.
Parity: ft_whole / base_whole vs the paper's stored per-crown IoUs (results_439/gtbox_sam3_{ft,base}_r2048.json).
LACE: deployed guided render per-crown IoUs (gtbox_guided_raw/gtbox_lace_guided_r2048.jsonl).
Paired differences vs LACE with tile-bootstrap 95% CIs.

  PYTHONPATH=. .venv/bin/python boxinst_commonality_tcd_04/native_raster/audit_2026-10-14/selvabox_evidence/gtbox_sam_persub.py
"""
import json, os, time
import numpy as np
from PIL import Image
from pycocotools import mask as mu
from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as G
from boxinst_commonality_tcd_04.native_raster import selvabox_seamfix_local as L

HERE = os.path.dirname(os.path.abspath(__file__))
NR = os.path.abspath(os.path.join(HERE, "..", ".."))
PKG = os.path.dirname(NR)
OUT = os.path.join(HERE, "gtbox_sam_persub.json")
CK = os.path.join(HERE, "gtbox_sam_persub_tiles")
DEV, SUB, ORIG = "mps", 1024, (0, 512, 1024)


def load_base():
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    return Sam3TrackerModel.from_pretrained(L.SAM_BASE).to(DEV).eval(), Sam3TrackerProcessor.from_pretrained(L.SAM_BASE)


def run(sam, proc, img, boxes, out_px, pre=None):
    """img PIL square; optionally resized to `pre` before the processor; masks returned at out_px."""
    import torch
    W = img.size[0]
    if pre and pre != W:
        img = img.resize((pre, pre), Image.BILINEAR); f = pre / W
    else:
        f = 1.0
    inp = proc(images=img, input_boxes=[(boxes * f).tolist()], return_tensors="pt")
    with torch.no_grad():
        emb = sam.get_image_embeddings(inp["pixel_values"].to(DEV))
    res = []
    for i in range(0, len(boxes), 64):
        with torch.no_grad():
            so = sam(input_boxes=inp["input_boxes"][:, i:i + 64].to(DEV), image_embeddings=emb, multimask_output=False)
        m = proc.post_process_masks(so.pred_masks.cpu(), inp["original_sizes"])[0].squeeze(1).numpy().astype(np.uint8)
        for mm in m:
            if mm.shape[0] != out_px:
                mm = np.asarray(Image.fromarray(mm * 255).resize((out_px, out_px), Image.BILINEAR)) >= 128
            res.append(mm.astype(bool))
    return res


def arm(sam, proc, rgb, B, persub, pre):
    whole = run(sam, proc, rgb, B, 2048, pre)
    if not persub:
        return whole
    out = list(whole); assign = {}
    for i, b in enumerate(B):
        c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2); best = None
        for oy in ORIG:
            for ox in ORIG:
                if b[0] >= ox and b[2] <= ox + SUB and b[1] >= oy and b[3] <= oy + SUB:
                    d = (c[0] - ox - SUB / 2) ** 2 + (c[1] - oy - SUB / 2) ** 2
                    if best is None or d < best[0]:
                        best = (d, oy, ox)
        if best:
            assign.setdefault(best[1:], []).append(i)
    for (oy, ox), idx in assign.items():
        ms = run(sam, proc, rgb.crop((ox, oy, ox + SUB, oy + SUB)), B[idx] - np.array([ox, oy, ox, oy]), SUB, pre)
        for i, m in zip(idx, ms):
            full = np.zeros((2048, 2048), bool); full[oy:oy + SUB, ox:ox + SUB] = m; out[i] = full
    return out


def boot(diff_by_tile, n=2000, seed=0):
    rs = np.random.RandomState(seed); tiles = list(diff_by_tile)
    s = [np.concatenate([diff_by_tile[tiles[j]] for j in rs.randint(0, len(tiles), len(tiles))]).mean() for _ in range(n)]
    return [round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4)]


def main():
    assert not os.path.exists(OUT), OUT
    os.makedirs(CK, exist_ok=True)
    gt = json.load(open(os.path.join(PKG, "test_gt.json")))
    tids = [t for t in sorted(gt) if gt[t]["trees"]]          # FULL set: every test tile with crowns
    ft, base = L.load_sam(DEV), load_base()
    t0 = time.time()
    for n, t in enumerate(tids):
        fp = os.path.join(CK, t + ".json")
        if os.path.exists(fp):
            continue
        polys = gt[t]["trees"]
        B = G.gt_boxes_2048(polys).astype(np.float64)
        g, valid = G.gt_rles_and_valid(polys, res=2048)
        rgb = L.load_rgb(t); rec = {"valid": valid.tolist()}
        for name, (sp, persub, pre) in {"ft_whole": (ft, False, 1777), "ft_sub": (ft, True, 1777),
                                        "base_whole": (base, False, None), "base_sub": (base, True, None)}.items():
            ms = arm(sp[0], sp[1], rgb, B, persub, pre)
            rec[name] = G.per_crown_iou_rle(G.to_rle(ms), g).round(5).tolist()
        json.dump(rec, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
        print(f"{n+1}/{len(tids)} {t} {len(polys)} crowns  {time.time()-t0:.0f}s", flush=True)
    # stored paper arms + LACE
    stored = {a: json.load(open(os.path.join(PKG, "results_439", f"gtbox_sam3_{a}_r2048.json")))["per_tile"] for a in ("ft", "base")}
    lace = {}
    for line in open(os.path.join(NR, "gtbox_guided_raw", "gtbox_lace_guided_r2048.jsonl")):
        r = json.loads(line); lace[r["tile"]] = r
    arms = ["lace", "ft_whole", "ft_sub", "base_whole", "base_sub", "paper_ft", "paper_base"]
    vals = {a: [] for a in arms}; by_tile = {a: {} for a in arms}
    for t in tids:
        rec = json.load(open(os.path.join(CK, t + ".json"))); v = np.array(rec["valid"], bool)
        assert v.tolist() == lace[t]["valid"] == stored["base"][t]["valid"], t
        cols = {"lace": lace[t]["iou"], "paper_ft": stored["ft"][t]["iou"], "paper_base": stored["base"][t]["iou"],
                **{a: rec[a] for a in ("ft_whole", "ft_sub", "base_whole", "base_sub")}}
        for a in arms:
            x = np.asarray(cols[a], float)[v]; vals[a].append(x); by_tile[a][t] = x
    summ = {"n_tiles": len(tids), "n_crowns": int(sum(len(x) for x in vals["lace"])), "device": DEV}
    for a in arms:
        x = np.concatenate(vals[a])
        summ[a] = {"mean_iou": round(float(x.mean()), 4), "ge50": round(float((x >= .5).mean()), 4),
                   "ge75": round(float((x >= .75).mean()), 4), "ge90": round(float((x >= .9).mean()), 4)}
    for a in ("ft_sub", "base_whole", "base_sub", "paper_base"):
        d = {t: by_tile["lace"][t] - by_tile[a][t] for t in tids}
        summ[f"lace_minus_{a}"] = {"mean": round(float(np.concatenate(list(d.values())).mean()), 4), "ci95_tile_boot": boot(d)}
    for mine, paper in (("ft_whole", "paper_ft"), ("base_whole", "paper_base")):
        a, b = np.concatenate(vals[mine]), np.concatenate(vals[paper])
        summ[f"parity_{mine}_vs_{paper}"] = {"mean_abs_diff": round(float(np.abs(a - b).mean()), 4),
                                             "mean_diff": round(float((a - b).mean()), 4)}
    json.dump({"summary": summ, "tiles": tids}, open(OUT, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
