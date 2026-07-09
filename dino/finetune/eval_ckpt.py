"""Eval a saved fine-tune checkpoint on the 439-tile TCD holdout (matched protocol).

Reloads pretrained DINOv3 + overwrites the trainable (last-k blocks + final norm)
params and the decoder from the checkpoint, then runs the same multi-scale
sliding-window semantic eval used everywhere else. Runs standalone so it can go in
parallel with an ongoing training process (reads a snapshot copy of the ckpt).
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from finetune import Decoder, FTEncoder, evaluate  # noqa: E402
from tcd_data import load_split  # noqa: E402

ROOT = os.path.join(HERE, "..", "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "ckpt", "ft_best_snap.pt"))
    ap.add_argument("--scales", default="512,768")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--flips", action="store_true")   # h/v-flip TTA
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
    print(f"[eval] loaded step={ck['step']} val_f1={ck['val_f1']:.4f} dev={dev}", flush=True)

    test = load_split(os.path.join(ROOT, "data/tcd/test"))
    if a.limit:
        test = test[: a.limit]
    scales = tuple(int(s) for s in a.scales.split(","))
    res = evaluate(enc, dec, test, dev, scales=scales, flips=a.flips)
    out = {"ckpt_step": ck["step"], "val_f1": ck["val_f1"], "scales": list(scales),
           "flips": a.flips, "n_test": len(test), "test_f1": res["micro_f1"],
           "test_iou": res["micro_iou"], "min": round((time.time() - t0) / 60, 1)}
    tag = "tta" if a.flips else "notta"
    json.dump(out, open(os.path.join(HERE, f"eval_step{ck['step']}_{tag}.json"), "w"), indent=2)
    print(f"\n>>> ckpt step {ck['step']} on {len(test)} test tiles [flips={a.flips}]: "
          f"F1={res['micro_f1']:.4f} IoU={res['micro_iou']:.4f} "
          f"(val was {ck['val_f1']:.4f}) [{out['min']}min]", flush=True)


if __name__ == "__main__":
    main()
