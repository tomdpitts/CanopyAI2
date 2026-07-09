"""Run a fine-tuned DINOv3 + decoder checkpoint on an arbitrary RGB image and
visualise the predicted tree-canopy mask (RGB | canopy overlay), saved to
claude_outputs/. Inference only (deterministic) — no seed needed.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
ROOT = os.path.join(HERE, "..", "..")
from finetune import Decoder, FTEncoder  # noqa: E402

OUTDIR = os.path.join(ROOT, "claude_outputs")


@torch.no_grad()
def predict(enc, dec, rgb01, dev, scales=(512, 768)):
    H, W = rgb01.shape[:2]
    acc = np.zeros((2, H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    for s in scales:
        ys = sorted(set(list(range(0, max(1, H - s) + 1, s)) + [max(0, H - s)]))
        xs = sorted(set(list(range(0, max(1, W - s) + 1, s)) + [max(0, W - s)]))
        for y in ys:
            for x in xs:
                cr = np.ascontiguousarray(rgb01[y:y + s, x:x + s].transpose(2, 0, 1))
                lg = dec(enc.features(torch.from_numpy(cr)[None].to(dev)), cr.shape[1:])[0].cpu().numpy()
                acc[:, y:y + s, x:x + s] += lg; cnt[y:y + s, x:x + s] += 1
    logits = acc / np.maximum(cnt, 1)
    return logits[1] > logits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt", default=os.path.join(HERE, "ckpt", "ft_v3_last4_best.pt"))
    ap.add_argument("--scales", default="512,768")
    ap.add_argument("--disp", type=int, default=1400)   # per-panel display width
    ap.add_argument("--tag", default="")                # output filename suffix
    ap.add_argument("--boxes", action="store_true")     # draw connected-component bounding boxes
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.time()

    ck = torch.load(a.ckpt, map_location=dev)
    args = ck.get("args", {})
    enc = FTEncoder(args.get("last_k", 4), args.get("feat_layers", 4), dev)
    dec = Decoder(enc.C * args.get("feat_layers", 4), hidden=args.get("hidden", 256)).to(dev)
    msd = enc.bb.state_dict()
    for n, v in ck["bb_trainable"].items():
        msd[n] = v.to(dev)
    enc.bb.load_state_dict(msd)
    dec.load_state_dict({k: v.to(dev) for k, v in ck["decoder"].items()})
    enc.eval(); dec.eval()
    print(f"[viz] ckpt step={ck['step']} val_f1={ck['val_f1']:.4f} dev={dev}", flush=True)

    rgb = np.asarray(Image.open(a.image).convert("RGB"), np.uint8)
    print(f"[viz] image {rgb.shape}", flush=True)
    pred = predict(enc, dec, rgb.astype(np.float32) / 255.0, dev,
                   tuple(int(s) for s in a.scales.split(",")))
    frac = float(pred.mean())
    print(f"[viz] predicted canopy fraction = {frac:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    from PIL import ImageDraw

    def small(arr):
        h, w = arr.shape[:2]
        return np.asarray(Image.fromarray(arr).resize((a.disp, int(a.disp * h / w)), Image.BILINEAR))

    left = small(rgb)
    suffix = ""
    if a.boxes:                       # connected-component boxes drawn on the downscaled image
        from scipy import ndimage
        lab, _ = ndimage.label(pred)
        sc = a.disp / pred.shape[1]
        base = Image.fromarray(left.copy())
        d = ImageDraw.Draw(base)
        nb = 0
        for sl in ndimage.find_objects(lab):
            if sl is None:
                continue
            y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            if (y1 - y0) < 4 or (x1 - x0) < 4:
                continue
            d.rectangle([x0 * sc, y0 * sc, x1 * sc, y1 * sc], outline=(255, 40, 40), width=2)
            nb += 1
        right = np.asarray(base)
        suffix = "_boxes"
        print(f"[viz] {nb} crown boxes (connected components)", flush=True)
    else:
        ov = rgb.astype(np.float32).copy()
        ov[pred] = 0.5 * ov[pred] + 0.5 * np.array([40, 220, 40], np.float32)
        right = small(ov.astype(np.uint8))

    pad = np.full((left.shape[0], 8, 3), 255, np.uint8)
    panel = np.concatenate([left, pad, right], 1)
    os.makedirs(OUTDIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.image))[0]
    out = os.path.join(OUTDIR, f"{stem}_canopy{a.tag}{suffix}.png")
    Image.fromarray(panel).save(out)
    print(f">>> saved {out}  (canopy_frac={frac:.3f})", flush=True)


if __name__ == "__main__":
    main()
