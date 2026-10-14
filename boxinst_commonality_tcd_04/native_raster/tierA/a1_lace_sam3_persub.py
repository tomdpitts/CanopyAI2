"""Tier A1 (box_supervised_baselines/PLAN.md §13, approved 2026-10-14): LACE detector + zero-shot SAM 3.

Boxes and detector scores s: `guided_ablate_inputs/detscore_s{seed}.json` (the published LACE detector
output, 159,687 / 162,017 / 177,598 boxes, s >= 0.05). Model: `facebook/sam3` via HF Sam3Tracker, no
fine-tuning, local MPS. Placement: each box on the most central 1024 subtile (3x3 grid, stride 512) that
contains it, fed to the processor at 1024 (-> 1008 encoder, ~1:1); boxes no subtile contains go on the
whole 2048 tile. `multimask_output=False`; masks post-processed to the input size and pasted at 2048.
Per-tile JSON checkpoints in tierA/a1_tiles_s{seed}/, resume by listing. Scoring: tierA/score_tierA.py.

  PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.tierA.a1_lace_sam3_persub --seed 0
"""
import argparse, json, os, time
import numpy as np
from pycocotools import mask as mu
from boxinst_commonality_tcd_04.native_raster import selvabox_seamfix_local as L

HERE = os.path.dirname(os.path.abspath(__file__))
NR = os.path.dirname(HERE)
DEV, SUB, ORIG, BATCH = "mps", 1024, (0, 512, 1024), 64


def load_base():
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    return (Sam3TrackerModel.from_pretrained(L.SAM_BASE).to(DEV).eval(),
            Sam3TrackerProcessor.from_pretrained(L.SAM_BASE))


def prompt(sam, proc, img, boxes):
    """img PIL (square, fed as is); boxes (n,4) in img px -> (bool masks at img size, sam scores)."""
    import torch
    inp = proc(images=img, input_boxes=[boxes.tolist()], return_tensors="pt")
    with torch.no_grad():
        emb = sam.get_image_embeddings(inp["pixel_values"].to(DEV))
    ms, sc = [], []
    for i in range(0, len(boxes), BATCH):
        with torch.no_grad():
            so = sam(input_boxes=inp["input_boxes"][:, i:i + BATCH].to(DEV), image_embeddings=emb,
                     multimask_output=False)
        m = proc.post_process_masks(so.pred_masks.cpu(), inp["original_sizes"])[0].squeeze(1).numpy()
        ms.extend(list(m.astype(bool)))
        sc.extend(np.asarray(so.iou_scores.cpu()).reshape(len(m), -1)[:, 0].round(4).tolist())
    return ms, sc


def assign(B):
    groups = {}
    for i, b in enumerate(B):
        c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2); best = None
        for oy in ORIG:
            for ox in ORIG:
                if b[0] >= ox and b[2] <= ox + SUB and b[1] >= oy and b[3] <= oy + SUB:
                    d = (c[0] - ox - SUB / 2) ** 2 + (c[1] - oy - SUB / 2) ** 2
                    if best is None or d < best[0]:
                        best = (d, oy, ox)
        groups.setdefault(best[1:] if best else None, []).append(i)
    return groups


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); a = ap.parse_args()
    det = json.load(open(os.path.join(NR, "guided_ablate_inputs", f"detscore_s{a.seed}.json")))
    tiles_dir = os.path.join(HERE, f"a1_tiles_s{a.seed}"); os.makedirs(tiles_dir, exist_ok=True)
    tids = sorted(det); assert len(tids) == 439
    done = {f[:-5] for f in os.listdir(tiles_dir) if f.endswith(".json")}
    todo = [t for t in tids if t not in done]
    print(f"[A1 s{a.seed}] {len(tids)} tiles, {len(done)} done, {len(todo)} to do", flush=True)
    sam, proc = load_base(); t0 = time.time()
    import platform, torch, transformers, PIL
    envp = os.path.join(tiles_dir, "_env.json")
    if not os.path.exists(envp):
        json.dump({"device": DEV, "machine": platform.machine(), "platform": platform.platform(),
                   "python": platform.python_version(), "torch": torch.__version__,
                   "transformers": transformers.__version__, "numpy": np.__version__,
                   "pillow": PIL.__version__, "started": time.strftime("%Y-%m-%d %H:%M")},
                  open(envp, "w"), indent=1)
    for n, t in enumerate(todo):
        B = np.asarray(det[t]["boxes_2048"], np.float64).reshape(-1, 4)
        rles, sams, where = [None] * len(B), [None] * len(B), [None] * len(B)
        if len(B):
            rgb = L.load_rgb(t)
            for key, idx in assign(B).items():
                if key is None:
                    ms, sc = prompt(sam, proc, rgb, B[idx]); off = None
                else:
                    oy, ox = key
                    ms, sc = prompt(sam, proc, rgb.crop((ox, oy, ox + SUB, oy + SUB)),
                                    B[idx] - np.array([ox, oy, ox, oy]))
                for i, m, s in zip(idx, ms, sc):
                    if key is None:
                        full = m.astype(np.uint8)
                    else:
                        full = np.zeros((2048, 2048), np.uint8); full[oy:oy + SUB, ox:ox + SUB] = m
                    e = mu.encode(np.asfortranarray(full))
                    rles[i] = {"size": [2048, 2048], "counts": e["counts"].decode("ascii")}
                    sams[i] = s; where[i] = list(key) if key else None
        rec = {"boxes_2048": det[t]["boxes_2048"], "scores": det[t]["scores"], "sam_scores": sams,
               "masks_rle": rles, "subtile": where, "device": DEV}
        tmp = os.path.join(tiles_dir, t + ".json.tmp"); json.dump(rec, open(tmp, "w"))
        os.replace(tmp, os.path.join(tiles_dir, t + ".json"))
        el = time.time() - t0
        print(f"  {len(done)+n+1}/{len(tids)} {t} {len(B)} boxes  {el:.0f}s  eta {el/(n+1)*(len(todo)-n-1):.0f}s", flush=True)
    print(f"[A1 s{a.seed}] done", flush=True)


if __name__ == "__main__":
    main()
