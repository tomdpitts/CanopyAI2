"""Cache frozen DINOv3 features + rgb512 + occupancy targets for the cohort.

Features are the only non-trivial compute; cached once (float16) and reused across
every probe fit / rung / N / seed in the sweep. Two backbones (web + sat ViT-L) so
the exploration can compare them and compute the delta-of-deltas vs raw features.
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))   # dino/
sys.path.insert(0, HERE)
from dinov3_seg import Dinov3Seg  # noqa: E402
from data import CACHE, load_cohort, load_rgb512, occupancy_target  # noqa: E402

BACKBONES = {
    "web": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "sat": "facebook/dinov3-vitl16-pretrain-sat493m",
}


def main():
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    recs = load_cohort()
    os.makedirs(os.path.join(CACHE, "rgb512"), exist_ok=True)
    os.makedirs(os.path.join(CACHE, "occ"), exist_ok=True)
    print(f"[cache] {len(recs)} scenes", flush=True)

    # rgb512 + occupancy (cheap, always)
    for r in recs:
        rp = os.path.join(CACHE, "rgb512", r.scene + ".npy")
        op = os.path.join(CACHE, "occ", r.scene + ".npy")
        if not os.path.exists(rp):
            np.save(rp, load_rgb512(r))
        if not os.path.exists(op):
            np.save(op, occupancy_target(r.boxes))
    print("[cache] rgb512 + occ done", flush=True)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    for tag, mid in BACKBONES.items():
        outdir = os.path.join(CACHE, f"feat_{tag}")
        os.makedirs(outdir, exist_ok=True)
        todo = [r for r in recs if not os.path.exists(os.path.join(outdir, r.scene + ".npy"))]
        if not todo:
            print(f"[cache] feat_{tag} already complete", flush=True)
            continue
        t0 = time.time()
        model = Dinov3Seg(mid, device=dev)
        for i, r in enumerate(todo):
            rgb = np.load(os.path.join(CACHE, "rgb512", r.scene + ".npy")).astype(np.float32) / 255.0
            x = torch.from_numpy(rgb.transpose(2, 0, 1))[None].to(dev)
            f = model.features(x)[0].cpu().numpy().astype(np.float16)   # (C,32,32)
            np.save(os.path.join(outdir, r.scene + ".npy"), f)
            if i % 40 == 0:
                print(f"[cache] feat_{tag} {i}/{len(todo)}", flush=True)
        del model
        print(f"[cache] feat_{tag} done ({time.time()-t0:.0f}s, shape={f.shape})", flush=True)


if __name__ == "__main__":
    main()
