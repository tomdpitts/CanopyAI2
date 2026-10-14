"""Audit 2026-10-14, item SelvaBox #1 (SAM 3 field of view). Oracle test, no detector involved:
prompt the FT SAM 3 with GROUND-TRUTH boxes two ways on the same tiles, local MPS, same weights.
  whole : 2048 tile -> 1777 (our selvabox_native_modal path; 20.3 cm/px at the 1008 px encoder)
  sub   : each box on the 1024 subtile (3x3 grid, stride 512) it sits most centrally in -> 1777
          (CanopyRS path: box prompted on its source tile; 10.2 cm/px at the encoder)
Metric: per-crown mask IoU vs the GT mask at 2048. Sample: every 15th test tile.

  PYTHONPATH=. .venv/bin/python boxinst_commonality_tcd_04/native_raster/audit_2026-10-14/selvabox_evidence/sam_fov_gtbox.py
"""
import json, os, time
import numpy as np
from PIL import Image
from pycocotools import mask as mu
from boxinst_commonality_tcd_04.native_raster import selvabox_seamfix_local as L

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "sam_fov_gtbox.json")
SAM_T, SUB, DEV = 1777, 1024, "mps"
ORIG = (0, 512, 1024)


def masks_on(sam, proc, img, boxes):
    """img PIL (any size, square), boxes (n,4) in img px -> list of bool masks at img size."""
    import torch
    W = img.size[0]
    f = SAM_T / W
    inputs = proc(images=img.resize((SAM_T, SAM_T), Image.BILINEAR), input_boxes=[(boxes * f).tolist()],
                  return_tensors="pt")
    with torch.no_grad():
        emb = sam.get_image_embeddings(inputs["pixel_values"].to(DEV))
    out = []
    for i in range(0, len(boxes), 64):
        with torch.no_grad():
            so = sam(input_boxes=inputs["input_boxes"][:, i:i + 64].to(DEV), image_embeddings=emb,
                     multimask_output=False)
        m = proc.post_process_masks(so.pred_masks.cpu(), inputs["original_sizes"])[0].squeeze(1).numpy()
        for mm in m.astype(np.uint8):
            if W != SAM_T:
                mm = np.asarray(Image.fromarray(mm * 255).resize((W, W), Image.BILINEAR)) >= 128
            out.append(mm.astype(bool))
    return out


def main():
    assert not os.path.exists(OUT), OUT
    gt = json.load(open(os.path.join(PKG, "test_gt.json")))
    tids = [t for t in sorted(gt) if gt[t]["trees"]][::15]
    sam, proc = L.load_sam(DEV)
    rows = []
    t0 = time.time()
    for n, t in enumerate(tids):
        polys = gt[t]["trees"]
        gm = [mu.decode(mu.merge(mu.frPyObjects([list(map(float, p))], 2048, 2048))).astype(bool) for p in polys]
        keep = [i for i, m in enumerate(gm) if m.any()]
        gm = [gm[i] for i in keep]
        B = []
        for m in gm:
            ys, xs = np.where(m); B.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        B = np.asarray(B, np.float64)
        rgb = L.load_rgb(t)
        whole = masks_on(sam, proc, rgb, B)
        # assign each box to the subtile that contains it with its centre closest to the subtile centre
        assign = {}
        for i, b in enumerate(B):
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            best = None
            for oy in ORIG:
                for ox in ORIG:
                    if b[0] >= ox and b[2] <= ox + SUB and b[1] >= oy and b[3] <= oy + SUB:
                        d = (c[0] - ox - SUB / 2) ** 2 + (c[1] - oy - SUB / 2) ** 2
                        if best is None or d < best[0]:
                            best = (d, oy, ox)
            assign.setdefault(best[1:] if best else None, []).append(i)
        sub = [None] * len(B)
        for key, idx in assign.items():
            if key is None:                      # crown wider than any subtile: whole-tile mask
                for i in idx:
                    sub[i] = whole[i]
                continue
            oy, ox = key
            crop = rgb.crop((ox, oy, ox + SUB, oy + SUB))
            ms = masks_on(sam, proc, crop, B[idx] - np.array([ox, oy, ox, oy]))
            for i, m in zip(idx, ms):
                full = np.zeros((2048, 2048), bool); full[oy:oy + SUB, ox:ox + SUB] = m; sub[i] = full
        for g, a, s in zip(gm, whole, sub):
            iou = lambda p: float((g & p).sum() / max((g | p).sum(), 1))
            rows.append({"tile": t, "area": int(g.sum()), "iou_whole": iou(a), "iou_sub": iou(s)})
        print(f"{n+1}/{len(tids)} {t} {len(gm)} crowns  {time.time()-t0:.0f}s", flush=True)
    json.dump({"rows": rows, "tiles": tids}, open(OUT + ".rows.tmp", "w"))
    a = np.array([r["iou_whole"] for r in rows]); s = np.array([r["iou_sub"] for r in rows])
    ar = np.array([r["area"] for r in rows])
    summ = {"n_tiles": len(tids), "n_crowns": len(rows), "device": DEV,
            "whole": {"mean_iou": a.mean(), "ge50": (a >= .5).mean(), "ge75": (a >= .75).mean()},
            "sub": {"mean_iou": s.mean(), "ge50": (s >= .5).mean(), "ge75": (s >= .75).mean()},
            "by_area_tercile": {}}
    edges = [0] + list(np.percentile(ar, [100 / 3, 200 / 3])) + [np.inf]
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (ar >= lo) & (ar < hi)
        summ["by_area_tercile"][f"t{k} {int(lo)}-{hi if np.isinf(hi) else int(hi)}px"] = {
            "n": int(sel.sum()), "whole": float(a[sel].mean()), "sub": float(s[sel].mean())}
    summ = json.loads(json.dumps(summ, default=float))
    json.dump({"summary": summ, "rows": rows, "tiles": tids}, open(OUT, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
