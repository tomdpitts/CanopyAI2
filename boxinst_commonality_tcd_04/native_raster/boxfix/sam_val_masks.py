"""[split val|test, default val] SAM 3 zero-shot per-subtile masks (gtbox_sam_persub.arm, local MPS) for EVERY val GT crown, saved as RLE
(diag_val.py kept only small-crown masks). Lets the box-only shape prior be applied to SAM too (fairness check).
Gate: per-crown IoU == smallcrown/val_crowns 'sam_zs_sub' (val) / gtbox_sam_persub_tiles 'base_sub' (test) (tolerance reported, MPS nondeterminism).
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.boxfix.sam_val_masks"""
import importlib.util, json, os, time
import numpy as np
from PIL import Image
from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as G
HERE = os.path.dirname(os.path.abspath(__file__)); NR = os.path.dirname(HERE); PKG = os.path.dirname(NR); REPO = os.path.dirname(PKG)
import sys
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "val"
VAL_GT = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "val_gt.json") if SPLIT == "val" else os.path.join(PKG, "test_gt.json")
RGB = os.path.join(REPO, "data", "tcd", "train" if SPLIT == "val" else "test")
OUT = f"/Users/tompitts/dphil/feat_cache/boxfix_sam_{SPLIT}"
FEAT = "/Users/tompitts/dphil/feat_cache/" + ("feat_4p_val" if SPLIT == "val" else "feat_4p_test_dl/feat_4p_test")
spec = importlib.util.spec_from_file_location("gsp", os.path.join(NR, "audit_2026-10-14", "selvabox_evidence", "gtbox_sam_persub.py"))
GSP = importlib.util.module_from_spec(spec); spec.loader.exec_module(GSP)

def main():
    os.makedirs(OUT, exist_ok=True); base = GSP.load_base(); gt = json.load(open(VAL_GT))
    tids = [t for t in sorted(gt) if gt[t]["trees"] and os.path.exists(f"{FEAT}/{t}.npy")]
    t0 = time.time(); worst = 0.0
    for n, t in enumerate(tids):
        fp = os.path.join(OUT, t + ".json")
        if os.path.exists(fp): continue
        polys = gt[t]["trees"]; B = G.gt_boxes_2048(polys).astype(np.float64); g, valid = G.gt_rles_and_valid(polys, res=2048)
        img = Image.open(os.path.join(RGB, t + ".tif")).convert("RGB")
        sam = GSP.arm(base[0], base[1], img, B, True, None); rl = G.to_rle(sam)
        iou = G.per_crown_iou_rle(rl, g).round(5)
        ref = np.asarray(json.load(open(os.path.join(NR, "smallcrown", "val_crowns", t + ".json")))["sam_zs_sub"]) if SPLIT == "val" else \
            np.asarray(json.load(open(os.path.join(NR, "audit_2026-10-14", "selvabox_evidence", "gtbox_sam_persub_tiles", t + ".json")))["base_sub"])
        d = float(np.abs(iou - ref).max()) if len(ref) else 0.0; worst = max(worst, d)
        json.dump({"iou": iou.tolist(), "gate_max_abs_diff": d, "masks_rle": [{"size": r["size"], "counts": r["counts"].decode("ascii")} for r in rl]}, open(fp + ".tmp", "w"))
        os.replace(fp + ".tmp", fp)
        print(f"  {n+1}/{len(tids)} {t} {len(B)} crowns gate {d:.1e} (worst {worst:.1e}) {time.time()-t0:.0f}s", flush=True)
    print("[sam_val] done", flush=True)

if __name__ == "__main__":
    main()
