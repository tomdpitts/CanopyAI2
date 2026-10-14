"""Falsification control for sam_fov_gtbox.py: is the per-subtile gain RESOLUTION or FIELD OF VIEW?
Arm `sub_lowres`: identical to `sub` (same crowns, same subtile assignment, same per-subtile field of
view) except the 1024 subtile is first downsampled to 512 (2x, i.e. the information content the
whole-tile path delivers to the 1008 px encoder), then resized to 1777 as before.
  If sub_lowres ~ whole  -> the gain is resolution.   If sub_lowres ~ sub -> the gain is the crop/context.
Joins to sam_fov_gtbox.json rows by (tile, crown order).

  PYTHONPATH=. .venv/bin/python boxinst_commonality_tcd_04/native_raster/audit_2026-10-14/selvabox_evidence/sam_fov_control.py
"""
import importlib.util, json, os, time
import numpy as np
from PIL import Image
from pycocotools import mask as mu

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("fov", os.path.join(HERE, "sam_fov_gtbox.py"))
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
OUT = os.path.join(HERE, "sam_fov_control.json")


def main():
    assert not os.path.exists(OUT), OUT
    ref = json.load(open(os.path.join(HERE, "sam_fov_gtbox.json")))
    tids, rows_ref = ref["tiles"], ref["rows"]
    gt = json.load(open(os.path.join(F.PKG, "test_gt.json")))
    sam, proc = F.L.load_sam(F.DEV)
    out, t0, k = [], time.time(), 0
    for n, t in enumerate(tids):
        gm = [mu.decode(mu.merge(mu.frPyObjects([list(map(float, p))], 2048, 2048))).astype(bool)
              for p in gt[t]["trees"]]
        gm = [m for m in gm if m.any()]
        B = []
        for m in gm:
            ys, xs = np.where(m); B.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        B = np.asarray(B, np.float64)
        rgb = F.L.load_rgb(t)
        assign = {}
        for i, b in enumerate(B):
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2); best = None
            for oy in F.ORIG:
                for ox in F.ORIG:
                    if b[0] >= ox and b[2] <= ox + F.SUB and b[1] >= oy and b[3] <= oy + F.SUB:
                        d = (c[0] - ox - F.SUB / 2) ** 2 + (c[1] - oy - F.SUB / 2) ** 2
                        if best is None or d < best[0]:
                            best = (d, oy, ox)
            assign.setdefault(best[1:] if best else None, []).append(i)
        res = [None] * len(B)
        for key, idx in assign.items():
            if key is None:
                continue
            oy, ox = key
            crop = rgb.crop((ox, oy, ox + F.SUB, oy + F.SUB)).resize((F.SUB // 2, F.SUB // 2), Image.BILINEAR) \
                      .resize((F.SUB, F.SUB), Image.BILINEAR)     # 2x information loss, same geometry
            ms = F.masks_on(sam, proc, crop, B[idx] - np.array([ox, oy, ox, oy]))
            for i, m in zip(idx, ms):
                full = np.zeros((2048, 2048), bool); full[oy:oy + F.SUB, ox:ox + F.SUB] = m; res[i] = full
        for i, g in enumerate(gm):
            r = rows_ref[k]; assert r["tile"] == t and r["area"] == int(g.sum()), (t, i)
            v = None if res[i] is None else float((g & res[i]).sum() / max((g | res[i]).sum(), 1))
            out.append(dict(r, iou_sub_lowres=v)); k += 1
        print(f"{n+1}/{len(tids)} {t}  {time.time()-t0:.0f}s", flush=True)
    sel = [r for r in out if r["iou_sub_lowres"] is not None]
    summ = {a: {"mean_iou": float(np.mean([r[f"iou_{a}"] for r in sel])),
                "ge50": float(np.mean([r[f"iou_{a}"] >= .5 for r in sel]))}
            for a in ("whole", "sub", "sub_lowres")}
    summ["n_crowns"] = len(sel)
    json.dump({"summary": summ, "rows": out}, open(OUT, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
