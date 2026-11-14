"""SAM 3 zero-shot box prompts on the identical PhenoBench GT boxes. Same path as the LIVECell probe's sam_run.py
(HF Sam3TrackerModel facebook/sam3, whole image, processor -> 1008, multimask_output=False, masks at native res).
Headline = uncropped (as sam3_eval_tcd); box-clipped stored as sensitivity arm. RGB image (PIL convert RGB).

  cd zeroshot_phenobench/code && PB_RUN=<sample60|val> PYTHONPATH=/Users/tompitts/dphil/CanopyAI2 ../../../.venv/bin/python -u sam_run.py
"""
import json, os, time

import numpy as np
from PIL import Image

from common import RES, DATA, stems, gt_image, enc, rle_json
import _verbatim as V
import lace_core as LC

DEV, MODEL, BATCH = "mps", "facebook/sam3", 64


def main():
    import torch, transformers
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    sam = Sam3TrackerModel.from_pretrained(MODEL).to(DEV).eval(); proc = Sam3TrackerProcessor.from_pretrained(MODEL)
    for a in ("sam3", "sam3_boxclip"):
        os.makedirs(os.path.join(RES, a), exist_ok=True)
    S = stems(); print(f"[pb sam] {len(S)} images, torch {torch.__version__}, transformers {transformers.__version__}", flush=True); T0 = time.time()
    for k, t in enumerate(S):
        if all(os.path.exists(os.path.join(RES, a, t + ".json")) for a in ("sam3", "sam3_boxclip")):
            continue
        boxes, gt_rles, valid, _, _ = gt_image(t); H = W = 1024
        img = Image.open(os.path.join(DATA, "images", t + ".png")).convert("RGB")
        t0 = time.time()
        inp = proc(images=img, input_boxes=[boxes.astype(np.float64).tolist()], return_tensors="pt")
        with torch.no_grad():
            emb = sam.get_image_embeddings(inp["pixel_values"].to(DEV))
        masks, scores = [], []
        for i in range(0, len(boxes), BATCH):
            with torch.no_grad():
                so = sam(input_boxes=inp["input_boxes"][:, i:i + BATCH].to(DEV), image_embeddings=emb, multimask_output=False)
            m = proc.post_process_masks(so.pred_masks.cpu(), inp["original_sizes"])[0].squeeze(1).numpy().astype(bool)
            masks.extend(list(m)); scores.extend(so.iou_scores.float().cpu().numpy().reshape(-1).tolist())
        torch.mps.synchronize(); sam_s = time.time() - t0
        assert len(masks) == len(boxes) and masks[0].shape == (H, W)
        for a, ms in (("sam3", masks), ("sam3_boxclip", [m & LC.box_clip(b, H, W) for m, b in zip(masks, boxes)])):
            pr = [enc(m) for m in ms]; iou = V.per_crown_iou_rle(pr, gt_rles)
            rec = {"arm": a, "image": t, "masks_rle": [rle_json(r) for r in pr], "iou": iou.tolist(), "valid": valid.tolist(),
                   "double": V.double_assignment_rle(pr), "sam_iou_score": scores, "sam_s": round(sam_s, 3), "model": MODEL}
            fp = os.path.join(RES, a, t + ".json"); json.dump(rec, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
        iu = json.load(open(os.path.join(RES, "sam3", t + ".json")))["iou"]
        print(f"  {k+1}/{len(S)} {t} n={len(boxes)} sam3 {np.mean(iu):.3f} ({sam_s:.1f}s)  [{time.time()-T0:.0f}s]", flush=True)
    print("[pb sam] done", flush=True)


if __name__ == "__main__":
    main()
