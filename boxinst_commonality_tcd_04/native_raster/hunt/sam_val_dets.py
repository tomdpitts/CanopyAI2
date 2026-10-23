"""SAM 3 zero-shot masks on LACE's VAL detections (108 tiles, det_phase4_L24_s0; boxes from
feat_cache/boxfix_ap_val_tiles, i.e. the boxes ap_val.py decoded, 40,058). Exactly A1's placement and prompting
(tierA/a1_lace_sam3_persub.py: most-central 1024 subtile, else whole tile; multimask off; local MPS).
Gives the val target (SAM masks @ LACE product ranking) and per-detection SAM masks for diagnosis.
Per-tile JSON -> feat_cache/hunt_sam_val_dets/ (resume).

  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.hunt.sam_val_dets
"""
import json, os, time
import numpy as np
from PIL import Image
from pycocotools import mask as mu
from boxinst_commonality_tcd_04.native_raster.tierA import a1_lace_sam3_persub as A1

HERE = os.path.dirname(os.path.abspath(__file__)); NR = os.path.dirname(HERE); REPO = os.path.dirname(os.path.dirname(NR))
SRC = "/Users/tompitts/dphil/feat_cache/boxfix_ap_val_tiles"; OUT = "/Users/tompitts/dphil/feat_cache/hunt_sam_val_dets"
RGB = os.path.join(REPO, "data", "tcd", "train")


def main():
    os.makedirs(OUT, exist_ok=True)
    tids = sorted(f[:-5] for f in os.listdir(SRC) if f.endswith(".json")); assert len(tids) == 108
    todo = [t for t in tids if not os.path.exists(os.path.join(OUT, t + ".json"))]
    print(f"[sam_val_dets] {len(todo)}/108 to do", flush=True)
    sam, proc = A1.load_base(); t0 = time.time()
    for n, t in enumerate(todo):
        src = json.load(open(os.path.join(SRC, t + ".json")))
        B = np.asarray(src["boxes_2048"], np.float64).reshape(-1, 4)
        rles, sams, where = [None] * len(B), [None] * len(B), [None] * len(B)
        if len(B):
            rgb = Image.open(os.path.join(RGB, t + ".tif")).convert("RGB"); assert rgb.size == (2048, 2048)
            for key, idx in A1.assign(B).items():
                if key is None:
                    ms, sc = A1.prompt(sam, proc, rgb, B[idx])
                else:
                    oy, ox = key
                    ms, sc = A1.prompt(sam, proc, rgb.crop((ox, oy, ox + A1.SUB, oy + A1.SUB)), B[idx] - np.array([ox, oy, ox, oy]))
                for i, m, s in zip(idx, ms, sc):
                    if key is None:
                        full = m.astype(np.uint8)
                    else:
                        full = np.zeros((2048, 2048), np.uint8); full[oy:oy + A1.SUB, ox:ox + A1.SUB] = m
                    rles[i] = {"size": [2048, 2048], "counts": mu.encode(np.asfortranarray(full))["counts"].decode("ascii")}
                    sams[i] = s; where[i] = list(key) if key else None
        rec = {"boxes_2048": src["boxes_2048"], "scores_s": src["scores_s"], "scores_product": src["scores_product"],
               "sam_scores": sams, "masks_rle": rles, "subtile": where, "device": A1.DEV}
        json.dump(rec, open(os.path.join(OUT, t + ".json.tmp"), "w")); os.replace(os.path.join(OUT, t + ".json.tmp"), os.path.join(OUT, t + ".json"))
        el = time.time() - t0
        print(f"  {n+1}/{len(todo)} {t} {len(B)} boxes {el:.0f}s eta {el/(n+1)*(len(todo)-n-1):.0f}s", flush=True)
    print("[sam_val_dets] done", flush=True)


if __name__ == "__main__":
    main()
